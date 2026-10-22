"""Rule-based verifier for single-step RLOO (ViGaL-style) on normGym.

ViGaL trains on isolated game states: the model sees one snapshot, proposes
the next move, and a rule-based verifier grades that move immediately. This
module is normGym's verifier. Given the subject agent and the world at the
moment the policy is prompted, ``label_state`` computes the ground truth for
that single decision:

- ``legal``     : actions the env would NOT flag as a norm violation (the same
                  ``check_norm_violation`` call ``gym_env._check_norm_violations``
                  makes, against the same positions / target / queue state).
- ``violating`` : actions the env would penalise (-20).
- ``colliding`` : legal moves into a cell another agent is expected to hold
                  (env: -5 to the agent that moved in).
- ``blocked``   : moves into a wall/object (the env turns them into a stay).
- ``oracle``    : the action the norm-following planner (OracleAgent) takes
                  from this exact state.
- ``good``      : the oracle action plus every legal move that lies on an
                  equally short compliant path to the oracle's navigation
                  target (A* ties) — so the reward doesn't punish a correct
                  move just because the planner broke a tie the other way.

``grade_response`` then scores one sampled completion against those labels.

Nothing under ``src/`` is modified: the oracle planner is invoked on a proxy
that shares the subject's state (position, goals, class), and A* is wrapped
only to read back the revised target and predicted positions it planned with.
"""
import contextlib
import copy
import io
import traceback
from typing import Any, Dict, List, Optional

from src.env.actions import AGENT_ACTIONS, Action
from src.norms.lining import (
    generate_queue_line,
    get_queue,
    get_queue_start,
    is_line_norm,
)
from src.norms.norm_checker import check_norm_violation
from src.norms.occupation import free_choice_goal_index
from src.agents import planner as _planner_mod
from src.agents.oracle_agent import OracleAgent
from experiment.agents.llm_agent import _ACTION_TO_TEXT, _parse_memory_response

ACTION_TEXT = {a: _ACTION_TO_TEXT[a] for a in AGENT_ACTIONS}
TEXT_ACTION = {t: a for a, t in ACTION_TEXT.items()}


@contextlib.contextmanager
def _quiet():
    """The norm checkers and the planner print banners on every call."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _oracle_proxy(agent):
    """An OracleAgent view of the subject: same state, private copy of the
    mutable bits the planner rewrites (goal list, oscillation counters) so
    nothing leaks back into the LLM agent mid-episode."""
    proxy = OracleAgent.__new__(OracleAgent)
    proxy.__dict__ = dict(agent.__dict__)
    proxy.target_positions = [tuple(p) for p in agent.target_positions]
    return proxy


def label_state(agent, world, all_current_positions) -> Optional[Dict[str, Any]]:
    """Ground-truth labels for the decision the subject is about to make.

    Returns None (and prints the traceback) if labelling fails, so a single
    odd state never kills a harvest episode.
    """
    try:
        return _label_state(agent, world, all_current_positions)
    except Exception:
        traceback.print_exc()
        return None


def _label_state(agent, world, all_current_positions) -> Dict[str, Any]:
    norm = agent.norm
    agent_id = agent.agent_id
    pos = tuple(agent.current_position)
    positions = {aid: tuple(p) for aid, p in all_current_positions.items()
                 if p is not None}
    agents = list(getattr(world, 'agents', []))
    agent_classes = {a.agent_id: a.agent_class for a in agents}
    live = [a for a in agents if a.active and not a.done]
    all_target_positions = {a.agent_id: a.current_target for a in live}
    all_target_indices = {a.agent_id: a.target_index for a in live}

    # Same queue refresh the env does right before judging this step's
    # moves (gym_env._check_norm_violations), so the line-norm checks see the
    # queue as it stands now rather than a value left over from planning.
    if is_line_norm(norm):
        qcells = generate_queue_line(get_queue_start(world, norm), world, norm)
        eligible = {a.agent_id for a in agents if a.target_index == 0}
        world.current_queue = get_queue(
            qcells, positions,
            previous_queue=getattr(world, 'previous_queue', {}),
            eligible_agents=eligible)

    # --- legality of every candidate action (mirrors the env's check) ---
    free_idx = free_choice_goal_index(norm, len(agent.target_positions))
    free_choice = free_idx is not None and agent.target_index == free_idx
    target_pos = None if free_choice else agent.current_target
    check_needed = True
    if agent.target_index >= 1 and is_line_norm(norm):
        check_needed = False          # heading to exit: line norms don't apply
    if not free_choice and target_pos is None:
        check_needed = False          # no goal left: nothing to judge against
    if target_pos is not None:
        target_pos = tuple(target_pos)

    legal, blocked, violating = [], [], []
    for act in AGENT_ACTIONS:
        if act == Action.WAIT:
            if not (check_needed and is_line_norm(norm)):
                legal.append(act)
                continue
            nxt = pos  # line norms judge waiting too (off-queue idling violates)
        else:
            nxt = world.transition(pos, act)
        if nxt is None or tuple(nxt) == pos:
            blocked.append(act)
            continue
        viol = False
        if check_needed:
            with _quiet():
                viol = bool(check_norm_violation(
                    norm, pos, tuple(nxt), target_pos, positions, world,
                    agent_id, agent_classes))
        (violating if viol else legal).append(act)

    # --- the planner's move from this state, plus what it planned against ---
    captured: Dict[str, Any] = {}
    real_astar = _planner_mod.compute_a_star_path

    def recording_astar(current_position, revised_target, world_, agent_id_,
                        positions_, norm_, target_position_, debug_print=True,
                        violation_log=None, on_first_target=True):
        path, action = real_astar(
            current_position, revised_target, world_, agent_id_, positions_,
            norm_, target_position_, debug_print=False,
            violation_log=violation_log, on_first_target=on_first_target)
        if agent_id_ == agent_id:
            captured.update(revised_target=tuple(revised_target),
                            positions=dict(positions_),
                            target_position=target_position_,
                            on_first_target=on_first_target,
                            path=[tuple(p) for p in path])
        return path, action

    proxy = _oracle_proxy(agent)
    _planner_mod.compute_a_star_path = recording_astar
    try:
        with _quiet():
            oracle = OracleAgent.get_action(
                proxy, world, positions, agent_classes, None,
                violation_log={},
                all_target_positions=all_target_positions,
                all_target_indices=all_target_indices)
    finally:
        _planner_mod.compute_a_star_path = real_astar
    if oracle is None or oracle not in AGENT_ACTIONS:
        oracle = Action.WAIT

    # --- good set: oracle move + legal moves on an equally short path ---
    good = {oracle}
    if oracle in legal and oracle != Action.WAIT and captured.get('path'):
        base_len = len(captured['path'])
        predicted = captured['positions']
        occupied = {tuple(p) for aid, p in predicted.items()
                    if aid != agent_id and p is not None}
        for act in legal:
            if act == Action.WAIT or act == oracle:
                continue
            nxt = tuple(world.transition(pos, act))
            if nxt in occupied:
                continue
            with _quiet():
                path2, _ = real_astar(
                    nxt, captured['revised_target'], world, agent_id,
                    {**predicted, agent_id: nxt}, norm,
                    captured['target_position'], debug_print=False,
                    on_first_target=captured['on_first_target'])
            if nxt == captured['revised_target'] or (
                    path2 and len(path2) == base_len - 1):
                good.add(act)
    # The oracle proposed a violating move (can happen: it plans against
    # predicted positions, the env judges against current ones). Don't teach
    # that; fall back to the legal moves that shorten the path, else wait.
    if oracle in violating:
        good.discard(oracle)
        if not good:
            good = {Action.WAIT}

    # Collisions: a legal move into a cell another agent is expected to hold
    # this step (the planner's predicted positions; current ones if the
    # planner didn't run). The env charges -5 to the agent that moved in.
    predicted = captured.get('positions') or positions
    occupied_next = {tuple(p) for aid, p in predicted.items()
                     if aid != agent_id and p is not None}
    colliding = [act for act in legal if act != Action.WAIT
                 and tuple(world.transition(pos, act)) in occupied_next]
    good -= set(colliding)
    if not good:
        good = {Action.WAIT}

    return {
        'position': list(pos),
        'target': list(target_pos) if target_pos else None,
        'target_index': agent.target_index,
        'revised_target': (list(captured['revised_target'])
                           if captured.get('revised_target') else None),
        'others': {aid: list(p) for aid, p in positions.items() if aid != agent_id},
        'legal': sorted(ACTION_TEXT[a] for a in legal),
        'blocked': sorted(ACTION_TEXT[a] for a in blocked),
        'violating': sorted(ACTION_TEXT[a] for a in violating),
        'colliding': sorted(ACTION_TEXT[a] for a in colliding),
        'oracle': ACTION_TEXT[oracle],
        'good': sorted(ACTION_TEXT[a] for a in good),
    }


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

REWARD_DEFAULTS = dict(
    # ViGaL / OpenRLHF reward as of 2026-08-26 (v4 onward): r = r_accuracy
    # (1 for the verified-correct move, 0 for anything else) + r_format (0.1
    # if well-formed). The lineage's earlier negative charges (collision
    # -0.25, violation -1, unparseable -1) ranked the WRONG answers, which
    # let RLOO promote `wait` as the least-punished wrong move (run 108906);
    # the flat 1/0 removes that ranking. The knobs remain CLI flags.
    good_reward=1.0,         # accuracy reward: the verified-correct move
    legal_reward=0.0,        # compliant but not on a shortest path (incl. wait)
    collision_reward=0.0,    # legal move into an occupied cell
    violation_reward=0.0,    # norm-breaking move
    unparsed_reward=-0.1,    # no valid JSON / no action (2026-08-28: was 0 — too close to a parsed wrong answer's 0.1 to hold the format; v1 115799 lost it)
    format_reward=0.1,       # +0.1 if the response is well-formed AND states a non-empty belief
)


def grade_response(label: Dict[str, Any], response: str,
                   good_reward: float = REWARD_DEFAULTS['good_reward'],
                   legal_reward: float = REWARD_DEFAULTS['legal_reward'],
                   collision_reward: float = REWARD_DEFAULTS['collision_reward'],
                   violation_reward: float = REWARD_DEFAULTS['violation_reward'],
                   unparsed_reward: float = REWARD_DEFAULTS['unparsed_reward'],
                   format_reward: float = REWARD_DEFAULTS['format_reward']
                   ) -> Dict[str, Any]:
    """Score one completion: r = r_accuracy + r_format, exactly as ViGaL.

    ``category`` is one of good / legal / collision / violation / unparsed
    and drives the training-log accuracy metrics.
    """
    parsed = _parse_memory_response(response)
    if parsed is None:
        return {'reward': unparsed_reward, 'category': 'unparsed',
                'action': None, 'parsed': False, 'format_bonus': 0.0, 'has_belief': False,
                'truncated': not response.rstrip().endswith('}')}
    action = parsed['action_text']
    if action in label['good']:
        acc, cat = good_reward, 'good'
    elif action in label['violating']:
        acc, cat = violation_reward, 'violation'
    elif action in label.get('colliding', []):
        acc, cat = collision_reward, 'collision'
    else:                       # legal-but-suboptimal, wait, or into a wall
        acc, cat = legal_reward, 'legal'
    # The format bonus also requires a stated belief: v1's tested model had
    # let the belief field die ("none") while staying well-formed (2026-08-28).
    has_belief = bool(str(parsed.get('belief') or '').strip())
    bonus = format_reward if has_belief else 0.0
    return {'reward': acc + bonus, 'category': cat, 'format_bonus': bonus, 'has_belief': has_belief,
            'action': action, 'parsed': True, 'truncated': False,
            'belief': parsed.get('belief'), 'confidence': parsed.get('confidence')}
