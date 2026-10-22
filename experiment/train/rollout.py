"""RL rollouts: one episode = one (shortened) interactive stage of the pipeline.

The episode is driven by the *existing* pipeline code — ``make_env``,
``build_initial_agents(subject_kind='memory_llm')``, ``run_interactive_stage``
— with a ``RecordingClient`` standing in for the API client. normGym and the
pipeline are not modified; this module only calls them.

Reward comes from the raw run log the pipeline itself writes (the same file
``experiment/evaluation/eval.py`` consumes), under the unchanged reward values:
goal +10, step -0.1, collision -5, norm violation -20.

Training-time shaping (attach_rewards) may soften the violation penalty, add
potential-based progress shaping, and charge a per-turn parse penalty — these
adjust only the PPO learning signal. The env, the raw run logs, and every
logged/eval metric (episode_return, num_violations, parse_rate) stay on the
true benchmark rewards.
"""
import contextlib
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiment.agents.llm_agent import _parse_memory_response
from experiment.run.common import _random_subject_positions
from experiment.run.run_experiment import run_interactive_stage
from experiment.train.config import (
    ROLLOUT_CAPACITY,
    ROLLOUT_MAX_STEPS,
    ROLLOUT_NUM_NO_NORM_AGENTS,
    default_env_kwargs,
)
from experiment.train.policy_client import RecordingClient, Turn
from src.env.rewards import COLLISION_PENALTY, NORM_VIOLATION_PENALTY


@dataclass
class EpisodeRollout:
    norm_type: str
    world_type: str
    seed: int
    turns: List[Turn]
    trace: List[Dict] = field(default_factory=list)
    episode_return: float = 0.0        # env return of the subject (+ format bonus)
    env_return: float = 0.0            # env return alone
    train_return: float = 0.0          # shaped training-signal return (violation
                                       # budget/scale, progress shaping, parse +
                                       # truncation penalties, format bonus;
                                       # before the trainer's reward_scale)
    num_violations: int = 0
    num_collisions: int = 0
    goals_reached: int = 0
    task_success: bool = False
    all_turns_parsed: bool = True
    turns_parsed: int = 0
    turns_total: int = 0
    workdir: str = ''


def _subject_trace(raw_run: Dict) -> List[Dict]:
    subject_ids = set(raw_run.get('subject_agent_ids', []))
    return [r for r in raw_run.get('trace', [])
            if r.get('agent_id') in subject_ids]


def _potential(record: Dict, coef: float) -> float:
    """-coef * manhattan(position, current target); 0 when either is unknown
    (subject not on the map yet, or all goals already completed)."""
    pos, target = record.get('position'), record.get('target')
    if not pos or not target:
        return 0.0
    return -coef * (abs(pos[0] - target[0]) + abs(pos[1] - target[1]))


def attach_rewards(turns: List[Turn], trace: List[Dict], gamma: float,
                   violation_penalty_scale: float = 1.0,
                   violation_budget: float = 0.0,
                   collision_budget: float = 0.0,
                   progress_shaping_coef: float = 0.0,
                   parse_penalty: float = 0.0,
                   truncation_penalty: float = 0.0) -> float:
    """Fill each turn's discounted reward-to-go from the subject step trace.

    Returns the episode's total shaped reward (the undiscounted sum of the
    per-step training rewards, before the trainer's reward_scale and before
    the episode-level format bonus) so callers can log the training signal
    next to the true env return.

    A turn's prompt says "Current step: N."; the recorder's entry for step N
    carries the reward assigned during that same env step. Delayed consequences
    (a violation penalty surfaces the step after the offending action) are
    naturally covered by summing over all later steps.

    Training-time shaping, applied to the learning signal only (the trace and
    all logged metrics keep the true env rewards):
    - ``violation_penalty_scale``: the env charges -NORM_VIOLATION_PENALTY per
      violation; scale s < 1 adds back (1-s) of it, softening the penalty the
      policy trains against (s=0.5 -> effective -10) so exploration toward
      goals isn't drowned by early violations.
    - ``violation_budget``: equal-weight alternative that OVERRIDES the scale
      when > 0. Replaces the env's flat charge with geometric charges: the
      k-th violation of the episode costs budget * (1/2)^k (-10, -5, -2.5, …
      for budget 20), charged at its own step so credit assignment stays
      dense. Total norm penalty is bounded by ``budget`` — set equal to the
      full task reward (2 goals x +10 = 20) so task success and norm
      compliance carry the same weight, while the marginal cost of one more
      violation never reaches zero (no free-violations regime, unlike a
      hard cap).
    - ``collision_budget``: the same geometric mechanism for the env's flat
      -COLLISION_PENALTY per collision. 104472 (RUN_LOG 08-23): the
      violation budget bounded norms but collisions stayed flat -5 uncapped,
      and the aggressive (successful) episodes accumulated 5.6 collisions on
      average by u17 (-28, one episode -75) — succeeding scored worse than
      failing and PPO suppressed success itself. Budget 10 keeps the first
      collision at the env's true -5 (10 * 1/2) and bounds the episode total
      at the single-goal task reward.
    - ``progress_shaping_coef``: potential-based shaping with
      phi(s) = -coef * distance-to-target; each step earns
      gamma*phi(next) - phi(cur). The last record keeps phi (no terminal
      reset), so the episode's shaping telescopes to coef*(d_start - d_end).
      The potential is a function of state (position + active target), so the
      optimal policy is unchanged; it only densifies the sparse goal reward.
    - ``parse_penalty``: charged at the step of every turn whose response is
      not the required JSON — the format signal the episode-level
      all-or-nothing bonus can't provide (run 101225 collapsed through that
      gap: garbage output cost nothing because the fallback action is 'wait').
    - ``truncation_penalty``: charged on top of ``parse_penalty`` when the
      failed response never closes its JSON object — the run-102005 drift
      mode: a free-text preamble (or a digit repetition loop) eats the
      max_new_tokens budget and the JSON is cut off mid-string.
    """
    records = sorted(trace, key=lambda r: r['step'])
    reward_at = {}
    violation_index = 0
    collision_index = 0
    for record in records:
        adjusted = float(record['step_reward'])
        nviol = int(record.get('step_norm_violations', 0))
        if nviol and violation_budget:
            adjusted += NORM_VIOLATION_PENALTY * nviol  # strip the flat charge
            for _ in range(nviol):
                violation_index += 1
                adjusted -= violation_budget * (0.5 ** violation_index)
        elif nviol and violation_penalty_scale != 1.0:
            adjusted += ((1.0 - violation_penalty_scale) * NORM_VIOLATION_PENALTY
                         * nviol)
        ncoll = int(record.get('step_collisions', 0))
        if ncoll and collision_budget:
            adjusted += COLLISION_PENALTY * ncoll       # strip the flat charge
            for _ in range(ncoll):
                collision_index += 1
                adjusted -= collision_budget * (0.5 ** collision_index)
        reward_at[record['step']] = reward_at.get(record['step'], 0.0) + adjusted
    if progress_shaping_coef:
        potentials = [_potential(r, progress_shaping_coef) for r in records]
        for i in range(len(records) - 1):
            reward_at[records[i]['step']] += gamma * potentials[i + 1] - potentials[i]
    if parse_penalty or truncation_penalty:
        for turn in turns:
            if (turn.step is not None
                    and _parse_memory_response(turn.response) is None):
                penalty = parse_penalty
                if not turn.response.rstrip().endswith('}'):
                    penalty += truncation_penalty
                reward_at[turn.step] = reward_at.get(turn.step, 0.0) - penalty
    steps = sorted(reward_at)
    for turn in turns:
        if turn.step is None:
            turn.reward_to_go = 0.0
            continue
        total = 0.0
        for step in steps:
            if step >= turn.step:
                total += (gamma ** (step - turn.step)) * reward_at[step]
        turn.reward_to_go = total
    return sum(reward_at.values())


def final_belief(turns: List[Turn]) -> Optional[Dict]:
    """Belief/confidence of the last parseable turn — the episode's final
    inferred norm belief (None if no turn parsed)."""
    for turn in reversed(turns):
        parsed = _parse_memory_response(turn.response)
        if parsed is not None:
            return {'belief': parsed['belief'],
                    'confidence': parsed['confidence'],
                    'step': turn.step}
    return None


def run_episode(client: RecordingClient,
                norm_type: str,
                world_type: str,
                seed: int,
                workdir: str,
                capacity: int = ROLLOUT_CAPACITY,
                max_steps: int = ROLLOUT_MAX_STEPS,
                num_no_norm_agents: int = ROLLOUT_NUM_NO_NORM_AGENTS,
                gamma: float = 1.0,
                format_bonus: float = 0.0,
                violation_penalty_scale: float = 1.0,
                violation_budget: float = 0.0,
                collision_budget: float = 0.0,
                progress_shaping_coef: float = 0.0,
                parse_penalty: float = 0.0,
                truncation_penalty: float = 0.0,
                subject_kind: str = 'memory_llm',
                quiet: bool = True) -> EpisodeRollout:
    """Run one interactive-stage episode with the policy as the subject.

    The stage's verbose stdout is redirected to ``<workdir>/rollout_log.txt``
    (the prompts/responses it contains double as a debugging record).
    """
    os.makedirs(workdir, exist_ok=True)
    client.reset()

    figures_root = os.path.join(workdir, 'simulation_figures')
    raw_runs_dir = os.path.join(workdir, 'raw_runs')

    random.seed(seed)
    subject_positions = _random_subject_positions(capacity, norm_type)

    log_path = os.path.join(workdir, 'rollout_log.txt')
    log_sink = open(log_path, 'w', encoding='utf-8')
    redirect = (contextlib.redirect_stdout(log_sink) if quiet
                else contextlib.nullcontext())
    try:
        with redirect:
            result = run_interactive_stage(
                capacity=capacity,
                max_steps=max_steps,
                figures_root=figures_root,
                norm_type=norm_type,
                client=client,
                subject_kind=subject_kind,
                subject_positions=subject_positions,
                ground_truth='without',
                world_type=world_type,
                env_kwargs=default_env_kwargs(),
                num_no_norm_agents=num_no_norm_agents,
                raw_runs_dir=raw_runs_dir,
                random_seed=seed,
            )
    finally:
        log_sink.close()

    with open(result['raw_run_path'], 'r', encoding='utf-8') as f:
        raw_run = json.load(f)

    trace = _subject_trace(raw_run)
    turns = client.snapshot()
    shaped_total = attach_rewards(
        turns, trace, gamma=gamma,
        violation_penalty_scale=violation_penalty_scale,
        violation_budget=violation_budget,
        collision_budget=collision_budget,
        progress_shaping_coef=progress_shaping_coef,
        parse_penalty=parse_penalty,
        truncation_penalty=truncation_penalty)

    env_return = sum(float(r['step_reward']) for r in trace)
    num_violations = sum(int(r.get('step_norm_violations', 0)) for r in trace)
    num_collisions = sum(int(r.get('step_collisions', 0)) for r in trace)
    last = max(trace, key=lambda r: r['step']) if trace else {}
    goals_reached = int(last.get('goals_reached', 0))
    # The env's task_success is fractional (0.5 = one of two goals); bool(0.5) is True,
    # which counted half-finished episodes as successes in re-harvest / episode-eval
    # metrics until 2026-09-13. Strict: success only when every goal was reached.
    task_success = float(last.get('task_success', 0.0) or 0.0) >= 1.0
    parsed_count = sum(
        _parse_memory_response(turn.response) is not None for turn in turns
    )
    all_parsed = parsed_count == len(turns)

    return EpisodeRollout(
        norm_type=norm_type,
        world_type=world_type,
        seed=seed,
        turns=turns,
        trace=trace,
        env_return=env_return,
        episode_return=env_return + (format_bonus if all_parsed else 0.0),
        train_return=shaped_total + (format_bonus if all_parsed else 0.0),
        num_violations=num_violations,
        num_collisions=num_collisions,
        goals_reached=goals_reached,
        task_success=task_success,
        all_turns_parsed=all_parsed,
        turns_parsed=parsed_count,
        turns_total=len(turns),
        workdir=workdir,
    )


def cleanup_episode(rollout: EpisodeRollout) -> None:
    """Delete an episode's workdir once its turns have been consumed."""
    if rollout.workdir and os.path.isdir(rollout.workdir):
        shutil.rmtree(rollout.workdir, ignore_errors=True)
