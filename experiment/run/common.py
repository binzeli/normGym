"""Shared building blocks for the experiment pipeline.

Everything here is agent-kind agnostic: the "subject" agent under study can be
an LLM agent, an RL agent, or an oracle planner (the baselines). Scripted
background agents are always OracleAgents. Subject agents are identified at
runtime via ``is_subject_agent`` so new agent kinds plug in without touching the
pipeline runners.
"""
import os
import sys
import random
import json
import re
import shutil
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXPERIMENT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'experiment')
sys.path.insert(0, PROJECT_ROOT)

from src.env.gym_env import NormEnv
from src.agents.base_agent import AGENT_COLORS
from src.agents.oracle_agent import OracleAgent
from src.agents.agent_init import (
    errand_targets,
    get_agent_entry_position,
    get_agent_targets,
)
from src.norms.occupation import (
    at_region_boundary,
    is_occupation_door_norm,
    is_occupation_seat_norm,
    region_hold_class,
    region_hold_count,
    region_staged,
    seat_norm_has_compliant_chair,
    subject_enters_last,
    subject_entry_door,
    subject_test_seat,
    subject_test_seat_placeable,
    subject_test_seat_pool,
    waiting_zone_cells,
)
from experiment.agents.human_agent import HumanAgent
from experiment.agents.llm_agent import LLMAgent
from experiment.agents.memory_llm_agent import MemoryLLMAgent
from experiment.agents.rl_agent import RLAgent


# The oracle-planner baselines: an OracleAgent run as the subject under study,
# one planning around the norm, one ignoring it — the same two-sided comparison
# runner/run_simulation.py draws, with the subject's runs logged and evaluated.
# See experiment/run/run_oracle.py.
ORACLE_SUBJECT_KINDS = ('oracle_with_norm', 'oracle_without_norm')

# The subject kinds driven by an LLM: they need a client, write a memory file,
# and carry their interactive-stage belief into the test stage. 'memory_llm' is
# the same LLM subject prompted with the reasoning behind its recent steps and
# its earlier penalised steps (see experiment/agents/memory_llm_agent).
# 'full_memory_llm' is RL-TRAINING ONLY (full episode history, no belief line;
# see experiment/train/full_memory_llm_agent.py) — the pipeline experiments
# never select it.
LLM_SUBJECT_KINDS = ('llm', 'memory_llm', 'full_memory_llm')

SUBJECT_KINDS = LLM_SUBJECT_KINDS + ('rl',) + ORACLE_SUBJECT_KINDS

WORLD_TYPE_CHOICES = ['cafe', 'movie_theater', 'bank', 'pharmacy',
                      'grocery_store', 'clothing_store', 'library',
                      'post_office', 'hospital', 'barbershop']


def is_llm_subject_kind(subject_kind) -> bool:
    return subject_kind in LLM_SUBJECT_KINDS


def is_subject_agent(agent) -> bool:
    """True for the agent under study, false for the scripted background agents.

    An LLM / RL subject is a newcomer agent, so ``not is_oracle`` identifies it.
    The oracle-planner baselines run an OracleAgent as the subject, so those
    carry an explicit flag (set in make_agent).
    """
    return getattr(agent, 'is_subject', False) or not agent.is_oracle


def norm_choices():
    norms_path = os.path.join(PROJECT_ROOT, 'src', 'norms', 'norm_descriptions.json')
    with open(norms_path, encoding='utf-8') as f:
        return [n['name'] for n in json.load(f)['norms']]


def force_load_repo_env():
    """Load repo .env values, overriding inherited shell values for this run."""
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if not os.path.exists(env_path):
        return

    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


class _TeeStdout:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_with_raw_stdout_log(log_path: str, fn, *args, **kwargs):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    original_stdout = sys.stdout
    with open(log_path, 'w', encoding='utf-8') as log_file:
        sys.stdout = _TeeStdout(original_stdout, log_file)
        try:
            return fn(*args, **kwargs)
        finally:
            sys.stdout = original_stdout


# ---------------------------------------------------------------------------
# Final norm-hypothesis extraction (LLM subjects only)
# ---------------------------------------------------------------------------

def parse_final_hypothesis_response(text: str) -> Optional[dict]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return None

    if not isinstance(parsed, dict):
        return None

    hypothesis = parsed.get("hypothesis")
    confidence = parsed.get("confidence")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        return None
    if not isinstance(confidence, (int, float)):
        return None

    return {
        "hypothesis": hypothesis.strip(),
        "confidence": max(0.0, min(1.0, float(confidence))),
    }


def append_final_hypothesis(client, memory_path: Optional[str],
                            stage_label: str = "interactive-stage") -> None:
    """Ask the LLM for a final norm hypothesis over a memory file and save it in place."""
    if not memory_path or not os.path.exists(memory_path):
        print(f"[WARN] {stage_label} memory not found; final hypothesis skipped: {memory_path}")
        return

    with open(memory_path, 'r', encoding='utf-8') as f:
        memory = json.load(f)
    if not isinstance(memory, dict):
        raise RuntimeError(f"{stage_label} memory is not a JSON object: {memory_path}")

    prompt = "\n".join([
        f"You are given the full {stage_label} memory from a grid-world social-norm inference run.",
        "Based only on this memory, provide a concise final hypothesis of the social norm and your confidence.",
        "Return exactly one JSON object and no other text:",
        '{ "hypothesis": "<concise natural-language norm>", "confidence": <0-1 number> }',
        f"{stage_label} memory:",
        json.dumps(memory, ensure_ascii=False, indent=2),
    ])

    parsed = None
    last_response = None
    for attempt in range(1, 4):
        response = client.response(
            image_paths=[],
            prompt=prompt,
            system_prompt=f"Summarize the inferred social norm from {stage_label} memory.",
            max_new_tokens=512,
            temperature=0.7,
        )
        print(f"[INFO] Final hypothesis response ({stage_label}): {response}")
        parsed = parse_final_hypothesis_response(response)
        if parsed is not None:
            if attempt > 1:
                print(f"[INFO] Parsed final hypothesis on retry {attempt}/3")
            break
        last_response = response
        if attempt < 3:
            print(f"[WARN] Invalid final hypothesis JSON; retrying ({attempt + 1}/3)")

    if parsed is None:
        raise RuntimeError(
            f"LLM returned invalid final {stage_label} hypothesis JSON after 3 attempts; "
            f"last raw response: {last_response!r}"
        )

    memory["final_hypothesis"] = parsed["hypothesis"]
    memory["final_hypothesis_confidence"] = parsed["confidence"]
    with open(memory_path, 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Final hypothesis saved to: {memory_path}")


# ---------------------------------------------------------------------------
# Agent / env construction
# ---------------------------------------------------------------------------

def make_agent(agent_id, world, norm_type, client, subject_kind=None,
               arrival_time=999999, extra_claimed=None, ground_truth='without',
               prompt_stage='interactive',
               reference_memory_path: Optional[str] = None,
               max_steps: Optional[int] = None,
               memory_filename: Optional[str] = None,
               knows_norm: bool = True,
               subject_class: Optional[str] = None,
               entry_position: Optional[Tuple[int, int]] = None,
               test_index: Optional[int] = None,
               placement_stage: Optional[str] = None):
    """Factory for one agent.

    subject_kind None -> scripted OracleAgent (knows_norm controls whether it
    follows the norm); 'llm' -> LLMAgent; 'memory_llm' -> MemoryLLMAgent (the
    same LLM subject, prompted with the reasoning behind its recent steps and its
    earlier penalised steps); 'rl' -> RLAgent (its action-selection
    logic lives entirely in experiment/agents/rl_agent.py — the pipeline only
    runs the gym env); 'human' -> HumanAgent (the web human experiment,
    experiment/human/app.py; the app attaches the browser action queue after
    construction so this factory stays session-agnostic); 'oracle_with_norm' /
    'oracle_without_norm' -> the planner baselines, an OracleAgent that plans
    around the norm or ignores it, marked as the subject under study.

    Goals are always drawn for the active norm, whatever the subject kind, so
    every subject kind draws the same number of values from the global RNG and
    therefore gets the identical setting.

    subject_class pins the subject agent's color instead of drawing it at
    random (used to sweep all three colors across the test runs). The random
    draw still happens either way, so the global RNG stream is unchanged.

    entry_position pins the agent's entrance instead of drawing it at random
    (used to fix the subject's door in the test stage). As with subject_class,
    the random draw still happens, so the global RNG stream is unchanged.

    In the test stage the subject's first goal — its seat — is likewise pinned
    for the norms that place it relative to their door (the door norms; see
    subject_test_seat), replacing the randomly drawn chair. The draw still
    happens and the replacement uses no RNG of its own.

    test_index is the 1-based index of the test run, forwarded to
    subject_test_seat for the norms whose pinned placement varies per run
    (SUBJECT_TEST_RUN_PLACEMENTS in src/norms/occupation.py).

    placement_stage defaults to prompt_stage for existing callers. Zero-shot
    tests explicitly use test placement with the existing zero-shot prompt.
    """
    placement_stage = prompt_stage if placement_stage is None else placement_stage
    is_subject = subject_kind is not None
    position = get_agent_entry_position(world)
    if entry_position is not None:
        position = tuple(entry_position)
    agent_norm = norm_type if (is_subject or knows_norm) else None
    # Targets are drawn from the run's norm, not the agent's: a no-norm agent
    # still heads for the same goals as everyone else (e.g. the edge-line
    # service cell), it just ignores the norm on the way. Mirrors
    # runner/run_simulation._build_agents.
    target_positions = get_agent_targets(
        position, world, norm_type, extra_claimed=extra_claimed
    )
    agent_class = random.choice(list(AGENT_COLORS))
    if is_subject and subject_class is not None:
        if subject_class not in AGENT_COLORS:
            raise ValueError(f"Unknown subject_class: {subject_class!r}")
        agent_class = subject_class

    seat_region_pool = None
    if is_subject and placement_stage == 'test' and target_positions:
        seat = subject_test_seat(norm_type, world, agent_class,
                                 claimed=extra_claimed, test_index=test_index)
        if seat is not None:
            target_positions = [seat] + list(target_positions[1:])
            # Kept on the agent so a planner can fall back within the region when
            # the pinned chair is taken (see OracleAgent.get_action).
            seat_region_pool = subject_test_seat_pool(norm_type, world, agent_class,
                                                      test_index=test_index)

    def built(agent):
        if seat_region_pool:
            agent.seat_region_pool = seat_region_pool
        # Which goal is the agent's to choose is a property of the run's norm
        # family, not of the agent: under a door norm the exit door is chosen and
        # the seat assigned, under a seat norm the reverse, and under any other
        # norm both are assigned. Agents that do not know the norm cannot work
        # this out for themselves, hence the flags. See OracleAgent.get_action.
        if is_occupation_door_norm(norm_type):
            agent.free_choice_door = True
        elif is_occupation_seat_norm(norm_type):
            agent.free_choice_seat = True
        return agent

    if is_llm_subject_kind(subject_kind):
        if subject_kind == 'llm':
            agent_cls = LLMAgent
        elif subject_kind == 'full_memory_llm':
            # RL-training only; imported lazily so the pipeline's normal paths
            # never load the training tree.
            from experiment.train.full_memory_llm_agent import FullMemoryLLMAgent
            agent_cls = FullMemoryLLMAgent
        else:
            agent_cls = MemoryLLMAgent
        return built(agent_cls(
            agent_id=agent_id,
            agent_class=agent_class,
            position=position,
            arrival_time=arrival_time,
            norm=norm_type,
            client=client,
            target_positions=target_positions,
            ground_truth=ground_truth,
            prompt_stage=prompt_stage,
            reference_memory_path=reference_memory_path,
            max_steps=max_steps,
            memory_filename=memory_filename,
        ))
    if subject_kind == 'human':
        return built(HumanAgent(
            agent_id=agent_id,
            agent_class=agent_class,
            position=position,
            arrival_time=arrival_time,
            norm=norm_type,
            target_positions=target_positions,
            name="You",
        ))
    if subject_kind == 'rl':
        return built(RLAgent(
            agent_id=agent_id,
            agent_class=agent_class,
            position=position,
            arrival_time=arrival_time,
            norm=norm_type,
            target_positions=target_positions,
        ))
    if subject_kind in ORACLE_SUBJECT_KINDS:
        # The same OracleAgent the background agents use, with the same
        # norm / no-norm split runner/run_simulation.py draws: with the norm it
        # plans around it, with norm=None it takes the shortest path to the same
        # goals. is_subject marks it as the agent under study (see
        # is_subject_agent) so the pipeline records and evaluates its runs.
        subject = OracleAgent(
            agent_id=agent_id,
            agent_class=agent_class,
            position=position,
            arrival_time=arrival_time,
            norm=norm_type if subject_kind == 'oracle_with_norm' else None,
            target_positions=target_positions,
        )
        subject.is_subject = True
        return built(subject)
    if subject_kind is not None:
        raise ValueError(f"Unknown subject_kind: {subject_kind!r}")

    return built(OracleAgent(
        agent_id=agent_id,
        agent_class=agent_class,
        position=position,
        arrival_time=arrival_time,
        norm=agent_norm,
        target_positions=target_positions,
    ))


def build_initial_agents(capacity, subject_positions, world, norm_type, client,
                         ground_truth, subject_kind='llm',
                         prompt_stage='interactive',
                         reference_memory_path=None,
                         max_steps=None,
                         memory_filename=None,
                         num_no_norm_agents=0,
                         subject_class=None,
                         test_index=None,
                         placement_stage=None):
    """Build the initial capacity batch with staggered arrival times.

    subject_class pins the color of the subject agent(s); background agents
    keep their random colors.

    In the test stage the subject's start and its seat goal are pinned so the
    norm always costs it a trip:
    - seat norms : the entrance is pinned to the far side of the seating area the
      norm permits it (see subject_entry_door).
    - door norms : the seat is pinned relative to the door the norm names — away
      from it when the norm makes that door mandatory, beside it when the norm
      forbids it (see subject_test_seat, applied in make_agent).
    - the X-cell norms : both are pinned per layout, entrance and seating region
      chosen so the direct walk between them misses the X cells
      (xcell_must_visit) or runs over them (xcell_forbidden); see
      SUBJECT_TEST_PLACEMENT_BY_WORLD.
    Background agents always enter at a random door and take a random seat, and
    so does the subject under every other norm.

    test_index (1-based) selects the run's placement for the norms whose pinned
    entrance/seat varies per test run (SUBJECT_TEST_RUN_PLACEMENTS).

    placement_stage can select test setup independently of the agent's prompt;
    omitting it preserves the existing prompt_stage-based setup.
    """
    placement_stage = prompt_stage if placement_stage is None else placement_stage
    subject_entrance = (subject_entry_door(norm_type, world, subject_class,
                                           test_index=test_index)
                        if placement_stage == 'test' else None)

    agents = []
    next_arrival = 1
    claimed_targets: set = set()
    no_norm_remaining = num_no_norm_agents
    for i in range(capacity):
        is_subject = i in subject_positions
        knows_norm = True
        if not is_subject and no_norm_remaining > 0:
            knows_norm = False
            no_norm_remaining -= 1

        agent = make_agent(
            f"C{i + 1}",
            world,
            norm_type,
            client,
            subject_kind=subject_kind if is_subject else None,
            arrival_time=next_arrival,
            extra_claimed=claimed_targets,
            ground_truth=ground_truth,
            prompt_stage=prompt_stage,
            reference_memory_path=reference_memory_path,
            max_steps=max_steps,
            memory_filename=memory_filename,
            knows_norm=knows_norm,
            subject_class=subject_class,
            entry_position=subject_entrance if is_subject else None,
            test_index=test_index if is_subject else None,
            placement_stage=placement_stage,
        )
        if agent.target_positions:
            claimed_targets.add(agent.target_positions[0])
        next_arrival += random.randint(1, 3)
        agents.append(agent)

    if placement_stage == 'test':
        _stage_region_zone(agents, world, norm_type)
    return agents


def _stage_region_zone(agents, world, norm_type):
    """Stage the waiting zone for a region-norm test run: shut it, and keep
    everyone else out of it (see region_staged).

    Two halves of one arrangement:

    - The earliest-arriving background agents become the zone holders, so the
      zone is shut well before the subject walks up to it. They keep their slots
      until release_region_holders frees them, which is what makes the zone shut
      at the one moment that decides the subject's violation. How many, and what
      shuts the zone, depends on the norm: the capacity norms fill every slot,
      while single_class_only puts 1-3 agents of one class in it and pins that
      class to something other than the subject's, so the subject is barred by
      the class rule rather than by a headcount.
    - Every other background agent is sent on an errand outside the zone (a chair
      or the counter) instead of the waiting-zone cell the region norms would
      normally give it. Otherwise the whole batch queues for the same slots, the
      queue jams the zone's edge, and the subject spends the episode colliding
      with it rather than reaching its goal.

    Goals are reassigned after the batch is drawn. The only draws of its own are
    single_class_only's holder count and class, taken once the whole batch exists
    and the RNG stream is in the same state for every subject kind, so all of
    them get the same staging.
    """
    if not region_staged(norm_type):
        return

    background = [a for a in agents if not is_subject_agent(a)]
    # Only an agent already bound for a zone cell can hold one; under these norms
    # that is every background agent, but the zone may have fewer cells than the
    # batch has agents.
    zone = waiting_zone_cells(world)
    eligible = [a for a in background
                if a.target_positions and tuple(a.target_positions[0]) in zone]
    hold_count = min(region_hold_count(norm_type), len(eligible))
    if hold_count <= 0:
        return

    holders = eligible[:hold_count]
    holder_ids = {a.agent_id for a in holders}
    errand_runners = [a for a in background if a.agent_id not in holder_ids]

    subject = next((a for a in agents if is_subject_agent(a)), None)
    holder_class = region_hold_class(
        norm_type, subject.agent_class if subject else None
    )

    for agent in holders:
        agent.hold_at_first_target = True
        if holder_class is not None:
            # One class, and not the subject's: a mixed zone is not a state
            # single_class_only allows, and matching the subject would let it in.
            agent.agent_class = holder_class
        # Start the holder on its zone cell rather than at a door. Walking there
        # takes a variable number of steps, and in the tighter worlds the subject
        # can reach the zone first and find it open — the norm then never binds
        # and the norm-blind subject records no violation. Seating the holder at
        # step 1 makes the zone full for the whole run up to the release.
        if agent.target_positions:
            agent.current_position = tuple(agent.target_positions[0])
            agent.previous_position = agent.current_position
            agent.arrival_time = 1

    claimed = {tuple(a.target_positions[0]) for a in agents if a.target_positions}
    spare = [pos for pos in errand_targets(world) if tuple(pos) not in claimed]
    for agent in errand_runners:
        if not agent.target_positions or not spare:
            continue
        agent.target_positions[0] = spare.pop(0)

    subject_class = subject.agent_class if subject else None
    print(f"[INFO] Waiting-zone holders (released when the subject reaches the "
          f"zone edge): {[a.agent_id for a in holders]}"
          + (f", all {holder_class} vs {subject_class} subject"
             if holder_class is not None else ""))
    print(f"[INFO] Scripted agents kept outside the zone: "
          f"{[(a.agent_id, a.target_positions[0]) for a in errand_runners]}")


def release_region_holders(env, subject_ids, norm_type):
    """Release the waiting-zone holders one step after the subject reaches the
    zone boundary, so both subject kinds meet a full zone and only the norm-blind
    one enters it.

    The release is deliberately one step late. On the step the subject arrives at
    the edge the holders stay put, so a norm-blind subject stepping in that step
    is over capacity and is recorded as violating, while a norm-aware subject
    finds entry barred. Only then do the holders start for the exit, freeing the
    slot the norm-aware subject needs to finish the run.

    `_region_release_pending` latches on the arrival step, so a norm-aware
    subject that steps back from the edge while it waits still gets the release.
    """
    if not region_staged(norm_type):
        return
    holders = [a for a in env.agents if getattr(a, 'hold_at_first_target', False)]
    if not holders:
        return

    zone = waiting_zone_cells(env.world)
    subject_arrived = any(
        agent.active and not agent.done
        and agent.agent_id in subject_ids
        and agent.target_index == 0
        and at_region_boundary(env.world, agent.current_position)
        for agent in env.agents
    )

    for holder in holders:
        if getattr(holder, '_region_release_pending', False):
            holder.hold_at_first_target = False
            print(f"[INFO] Waiting-zone holder {holder.agent_id} released; "
                  f"heading for the exit and freeing its slot")
        elif subject_arrived and holder.current_position in zone:
            holder._region_release_pending = True
            print(f"[INFO] Subject reached the waiting-zone edge; holder "
                  f"{holder.agent_id} keeps its slot for this step, then leaves")


def make_env(norm_type, figures_dir, num_agents, world_type, max_steps,
             use_sprites=True, use_isometric=False, entrance_side='bottom',
             map_type='default', interpolation_steps=0):
    """Construct a NormEnv with the given parameters."""
    env = NormEnv(
        num_agents=num_agents,
        max_steps=max_steps,
        norm_type=norm_type,
        output_dir=figures_dir,
        render_mode='rgb_array',
        world_type=world_type,
        use_isometric=use_isometric,
        map_type=map_type,
        interpolation_steps=interpolation_steps,
    )
    if not seat_norm_has_compliant_chair(norm_type, env.world):
        print(f"[WARN] No chair in {world_type} satisfies {norm_type}; "
              f"agents cannot complete their seat goal without violating the norm")
    if not subject_test_seat_placeable(norm_type, env.world):
        print(f"[WARN] {world_type} has no chair on the side {norm_type} seats the "
              f"subject on; the test-stage subject keeps its randomly drawn chair")
    return env


def get_agent_entry_position_not_last(world, last_entrance=None):
    entrances = world.entrances
    if last_entrance is not None and len(entrances) > 1:
        entrances = [e for e in entrances if e != last_entrance]
    return random.choice(entrances)


def respawn_agent(env, agent, next_replacement_step, last_spawn_entrance,
                  keep_out_of_region=False) -> Tuple[int, Any]:
    """Place a waiting agent onto the map at a fresh entrance.

    keep_out_of_region redirects the replacement's first goal from the waiting
    zone to an errand outside it. The region-capacity test runs set it so the
    only agent ever in the zone is the staged holder: without it a replacement
    walks in the moment the holder leaves and shuts the zone again, and the
    subject queueing at the edge never gets the slot the release freed.

    Returns the updated (next_replacement_step, last_spawn_entrance).
    """
    arrival = max(env.current_step + 1, next_replacement_step)
    agent.arrival_time = arrival
    next_replacement_step = arrival + 1

    chosen_entrance = get_agent_entry_position_not_last(env.world, last_spawn_entrance)
    agent.current_position = chosen_entrance
    agent.previous_position = chosen_entrance
    last_spawn_entrance = chosen_entrance
    # env.norm_type, not agent.norm: a no-norm replacement must still target
    # the run's goals (e.g. the edge-line service cell), like make_agent.
    agent.target_positions = get_agent_targets(chosen_entrance, env.world, env.norm_type)
    if keep_out_of_region and agent.target_positions:
        claimed = {tuple(a.target_positions[0]) for a in env.agents
                   if not getattr(a, 'done', True) and getattr(a, 'target_positions', None)}
        spare = next((pos for pos in errand_targets(env.world)
                      if tuple(pos) not in claimed), None)
        if spare is not None:
            agent.target_positions[0] = spare
    env.add_agent(agent)
    env.num_agents = len(env.agents)
    return next_replacement_step, last_spawn_entrance


def build_scripted_pool(pool_size, background_slots, num_no_norm_agents,
                        world, norm_type, client, ground_truth,
                        first_agent_number):
    """Build the waiting queues of scripted replacement customers.

    Returns (waiting_norm_customer, waiting_no_norm_customer, next_agent_number).
    """
    waiting_norm_customer: deque = deque()
    waiting_no_norm_customer: deque = deque()
    agent_number = first_agent_number
    for i in range(pool_size):
        knows_norm = not (
            background_slots > 0
            and (i % background_slots) < num_no_norm_agents
        )
        agent = make_agent(
            f"C{agent_number}",
            world,
            norm_type,
            client,
            subject_kind=None,
            ground_truth=ground_truth,
            knows_norm=knows_norm,
        )
        agent_number += 1
        if agent.norm is None:
            waiting_no_norm_customer.append(agent)
        else:
            waiting_norm_customer.append(agent)
    return waiting_norm_customer, waiting_no_norm_customer, agent_number


def pop_scripted_replacement(done_agent, waiting_norm_customer, waiting_no_norm_customer):
    """Pick the scripted replacement for a finished scripted agent, or None."""
    was_no_norm = done_agent.norm is None
    if was_no_norm and waiting_no_norm_customer:
        return waiting_no_norm_customer.popleft()
    if not was_no_norm and waiting_norm_customer:
        return waiting_norm_customer.popleft()
    if waiting_norm_customer:
        return waiting_norm_customer.popleft()
    if waiting_no_norm_customer:
        return waiting_no_norm_customer.popleft()
    return None


# ---------------------------------------------------------------------------
# Raw per-step recording of the subject agent
# ---------------------------------------------------------------------------

def _agent_id_of(agent) -> Optional[str]:
    agent_id = getattr(agent, 'agent_id', None)
    if agent_id is None and isinstance(agent, dict):
        agent_id = agent.get('agent_id') or agent.get('id')
    return agent_id


def _entered_map(agent) -> bool:
    """True once the agent has been on the map (still there, or already left)."""
    return bool(getattr(agent, 'active', False) or getattr(agent, 'done', False))


def _task_success(agent) -> float:
    """1.0 once the agent reached every goal, 0.5 when it reached one of two."""
    if agent.done:
        return 1.0
    num_goals = len(agent.target_positions or [])
    if not num_goals:
        return 0.0
    return min(agent.target_index, num_goals) / num_goals


def _action_name(action) -> Optional[str]:
    """Actions arrive as an Action enum; keep the raw log JSON-serializable."""
    if isinstance(action, dict):
        action = action.get('action')
    if action is None:
        return None
    return getattr(action, 'name', None) or str(action)


class RunRecorder:
    """Raw per-step log of the subject agent(s) of one run.

    One record per step the subject is on the map:

        {"step": 39, "agent_id": "C3", "position": [5, 14], "action": "WAIT",
         "target": [2, 6], "step_reward": -0.1, "cumulative_reward": -3.5,
         "other_agent_cumulative_reward": {"C1": 18.0, "C2": 17.7},
         "collided_with": [], "step_collisions": 0, "step_norm_violations": 0,
         "total_collisions": 1, "total_norm_violations": 0,
         "goals_reached": 0, "num_goals": 2, "task_success": 0.0, "done": false}

    ``experiment/evaluation/eval.py`` reads the emitted JSON and derives every
    metric from it; this class only observes the env.

    A run has one subject on the map at a time, but the interactive stage
    replaces the subject when it finishes, so a run can cover several subject
    instances; each step record carries its ``agent_id``.
    """

    def __init__(self, env):
        self.env = env
        self.trace: List[Dict[str, Any]] = []
        self._finished: set = set()
        self._subjects: Dict[str, Any] = {}  # agent_id -> agent, in appearance order
        # Running totals across every subject the run covers: the interactive
        # stage replaces the subject when it finishes, and the totals keep
        # climbing across the replacements rather than resetting to 0.
        self._total_collisions = 0
        self._total_violations = 0

    def record_step(self, info: Dict[str, Any]) -> None:
        """Record one env step. Call right after ``env.step()`` with its info."""
        step = info.get('step', self.env.current_step)
        actions = info.get('actions', {}) or {}
        collisions = info.get('collisions', {}) or {}
        violations = info.get('norm_violations', []) or []
        step_rewards = info.get('step_rewards', {}) or {}

        for agent in self.env.agents:
            if not is_subject_agent(agent):
                continue
            agent_id = agent.agent_id
            if agent_id in self._finished or not _entered_map(agent):
                continue

            self._subjects.setdefault(agent_id, agent)
            target = agent.current_target

            # Scripted agents the subject contested a cell with this step.
            collided_with = self._collided_with(agent_id, collisions)
            step_violations = sum(1 for v in violations if v and v[0] == agent_id)
            self._total_collisions += len(collided_with)
            self._total_violations += step_violations

            num_goals = len(agent.target_positions or [])
            goals_reached = min(agent.target_index, num_goals)

            self.trace.append({
                'step': step,
                'agent_id': agent_id,
                'position': list(agent.current_position) if agent.current_position else None,
                'action': _action_name(actions.get(agent_id)),
                'target': list(target) if target else None,
                'step_reward': round(float(step_rewards.get(agent_id, 0.0)), 3),
                'cumulative_reward': round(float(agent.total_reward), 3),
                'other_agent_cumulative_reward': {
                    a.agent_id: round(float(a.total_reward), 3)
                    for a in self.env.agents
                    if a.agent_id != agent_id and _entered_map(a)
                },
                'collided_with': collided_with,
                'step_collisions': len(collided_with),
                'step_norm_violations': step_violations,
                'total_collisions': self._total_collisions,
                'total_norm_violations': self._total_violations,
                'goals_reached': goals_reached,
                'num_goals': num_goals,
                'task_success': _task_success(agent),
                'done': agent.done,
            })

            if agent.done:
                self._finished.add(agent_id)

    def _collided_with(self, subject_id: str, collisions: Dict[Any, Any]) -> List[str]:
        """Scripted agents that contested a cell with the subject this step."""
        others: List[str] = []
        for agents_at_cell in collisions.values():
            ids = {_agent_id_of(a) for a in agents_at_cell}
            if subject_id not in ids:
                continue
            others.extend(sorted(
                _agent_id_of(a) for a in agents_at_cell
                if _agent_id_of(a) != subject_id and not is_subject_agent(a)
            ))
        return others

    def _flush_pending_violations(self) -> None:
        """Fold in violations the env detected but never surfaced.

        The env reports a non-spacing violation in the step *after* it was
        committed, so a violation on the subject's final step would otherwise be
        lost when the run ends.
        """
        touched = False
        for violation in getattr(self.env, '_pending_norm_violations', None) or []:
            agent_id = violation[0]
            if agent_id not in self._subjects:
                continue
            for record in reversed(self.trace):
                if record['agent_id'] == agent_id:
                    record['step_norm_violations'] += 1
                    touched = True
                    break
        if touched:
            # Re-derive the running totals so they stay monotonic across the
            # whole trace after a mid-run step count was bumped.
            self._recompute_totals()

    def _recompute_totals(self) -> None:
        """Rebuild the cumulative ``total_*`` columns from the per-step counts."""
        collisions = violations = 0
        for record in self.trace:
            collisions += record['step_collisions']
            violations += record['step_norm_violations']
            record['total_collisions'] = collisions
            record['total_norm_violations'] = violations
        self._total_collisions = collisions
        self._total_violations = violations

    def raw_run(self, **meta) -> Dict[str, Any]:
        """The raw run log: run metadata, final agent state, and the step trace."""
        self._flush_pending_violations()
        return {
            **meta,
            'subject_agent_ids': list(self._subjects),
            'steps': self.env.current_step,
            'agent_rewards': {
                a.agent_id: round(float(a.total_reward), 3)
                for a in self.env.agents if _entered_map(a)
            },
            'agent_is_subject': {
                a.agent_id: is_subject_agent(a)
                for a in self.env.agents if _entered_map(a)
            },
            'trace': self.trace,
        }


# ---------------------------------------------------------------------------
# Episode summaries
# ---------------------------------------------------------------------------

def summarize_norm_violations(episode_log) -> Dict[str, Any]:
    by_agent: Dict[str, int] = {}
    for step_data in episode_log:
        for violation in step_data.get('norm_violations', []):
            agent_id = violation[0]
            by_agent[agent_id] = by_agent.get(agent_id, 0) + 1
    return {
        'total': sum(by_agent.values()),
        'by_agent': dict(sorted(by_agent.items())),
    }


def summarize_collisions(episode_log) -> Dict[str, Any]:
    by_agent: Dict[str, int] = {}
    total = 0
    for step_data in episode_log:
        collisions = step_data.get('collisions', {})
        if not isinstance(collisions, dict):
            continue
        total += len(collisions)
        for agents in collisions.values():
            for agent in agents:
                agent_id = getattr(agent, 'agent_id', None)
                if agent_id is None and isinstance(agent, dict):
                    agent_id = agent.get('agent_id') or agent.get('id')
                if agent_id is None:
                    continue
                by_agent[agent_id] = by_agent.get(agent_id, 0) + 1
    return {
        'total': total,
        'by_agent': dict(sorted(by_agent.items())),
    }


def summarize_episode(env, done_agent_ids, step_count, figures_root) -> Dict[str, Any]:
    """Build the standard episode-result dict from a finished env run."""
    subject_agent_ids = {a.agent_id for a in env.agents if is_subject_agent(a)}

    print(f"\n[OK] All figures saved to '{figures_root}/' directory")
    print(f"  Total steps: {step_count}")
    print(f"  Total agents processed: {len(done_agent_ids)}")
    print(f"  Subject agents on map: {sorted(subject_agent_ids)}")

    return {
        'steps': env.current_step,
        'customers_count': len(env.agents),
        'num_collisions': summarize_collisions(env.episode_log),
        'subject_agent_ids': sorted(subject_agent_ids),
        'agent_rewards': {a.agent_id: a.total_reward for a in env.agents},
        'agent_classes': {a.agent_id: a.agent_class for a in env.agents},
        'agent_norms': {a.agent_id: a.norm for a in env.agents},
        'episode_log': env.episode_log,
        'final_states': {
            a.agent_id: ('done' if a.done else 'active')
            for a in env.agents
        },
        'agents': [{'id': a.agent_id, 'type': a.agent_type.value} for a in env.agents],
    }


# ---------------------------------------------------------------------------
# Frame filtering (keep only frames where a subject agent was on the map)
# ---------------------------------------------------------------------------

def _frame_step(filename: str):
    if not filename.startswith('step_') or not filename.endswith('.png'):
        return None
    step_part = filename[len('step_'):-len('.png')].split('_sub', 1)[0]
    try:
        return int(step_part)
    except ValueError:
        return None


def filter_subject_episode_frames(figures_dir: str, episode: Dict[str, Any]) -> None:
    subject_ids = set(episode.get('subject_agent_ids', []))
    if not subject_ids or not os.path.isdir(figures_dir):
        return

    keep_steps = set()
    for step_data in episode.get('episode_log', []):
        step = step_data.get('step')
        states = step_data.get('agent_states', {})
        actions = step_data.get('actions', {})
        has_active_subject = any(
            agent_id in subject_ids
            and state.get('active')
            and not state.get('done')
            for agent_id, state in states.items()
        )
        has_subject_action = any(agent_id in actions for agent_id in subject_ids)
        if has_active_subject or has_subject_action:
            keep_steps.add(step)

    for filename in os.listdir(figures_dir):
        step = _frame_step(filename)
        if step is not None and step not in keep_steps:
            os.remove(os.path.join(figures_dir, filename))

    video_path = os.path.join(figures_dir, "simulation.mp4")
    if os.path.exists(video_path):
        os.remove(video_path)


# ---------------------------------------------------------------------------
# Test-stage runner (same-world and new-world test loops)
#
# Every pipeline runner (run_experiment.py's same-world and new-world test
# stages, run_oracle.py's test stage) needs the identical protocol: N runs,
# each on a freshly reseeded RNG, with the subject agent's color rotating
# through the fixed orange/green/blue schedule so every block of three runs
# covers all three colors exactly once. run_test_series is that protocol.
# ---------------------------------------------------------------------------

def _random_subject_positions(capacity, norm_type=None):
    """One subject slot, entering as the 3rd..7th agent (0-indexed 2..6).

    Under the norms in SUBJECT_TEST_ENTERS_LAST the slot is pinned to the end of
    the batch instead, so the subject walks into a room the other `capacity - 1`
    agents already occupy. The random draw still happens and is discarded, so the
    global RNG stream — and therefore every other part of the setting — is the
    same as it would be without the pin, and identical across subject kinds.
    """
    if capacity < 3:
        raise ValueError("capacity must be at least 3 so the subject can enter in 3rd place or later")
    drawn = random.randint(2, min(6, capacity - 1))
    if subject_enters_last(norm_type):
        return {capacity - 1}
    return {drawn}


def subject_class_schedule(seed, count):
    """`count` subject colors, laid out so every consecutive block of three
    covers all of AGENT_COLORS exactly once.

    Used for both the interactive stage (the subject and each of its
    replacements take the next color, so the run always exercises all three)
    and the test stages (each block of three runs covers all three colors).
    The permutation of each block comes from a private RNG seeded with the
    norm's base seed, so the order is fixed per norm and identical across
    subject kinds while leaving the global RNG stream — and therefore every
    stage's setting — untouched.
    """
    rng = random.Random(seed)
    schedule = []
    while len(schedule) < count:
        block = list(AGENT_COLORS)
        rng.shuffle(block)
        if schedule and block[0] == schedule[-1]:
            # Without this the seam between two blocks can repeat a color; swap
            # the first two so consecutive entries always differ while the block
            # still covers all three.
            block[0], block[1] = block[1], block[0]
        schedule.extend(block)
    return schedule[:count]


def run_test_simulation(env, initial_agents, waiting_norm_customer,
                        waiting_no_norm_customer, figures_root, norm_type,
                        stage_label):
    """Test stage: scripted customers come and go; the run ends as soon as the
    subject agent finishes all its goals (or max_steps truncates it)."""
    print("=" * 70)
    print(f"{stage_label} Simulation - WITH {norm_type.upper()} Norm")
    print("=" * 70)
    print(f"Figures will be saved to: {figures_root}/\n")

    obs, info = env.reset(options={'agents': initial_agents})
    env.output_dir = figures_root
    env.render()

    recorder = RunRecorder(env)
    subject_ids = {a.agent_id for a in initial_agents if is_subject_agent(a)}
    done_agent_ids: set = set()
    terminated = False
    truncated = False
    step_count = 0
    next_replacement_step = 1
    last_spawn_entrance = None
    subject_finished = False

    while not (terminated or truncated or subject_finished):
        obs, reward, terminated, truncated, info = env.step()
        step_count = info['step']
        recorder.record_step(info)
        env.render()

        release_region_holders(env, subject_ids, norm_type)

        newly_done = [
            agent for agent in env.agents
            if agent.done and agent.agent_id not in done_agent_ids
        ]

        for agent in newly_done:
            done_agent_ids.add(agent.agent_id)
            if agent.agent_id in subject_ids:
                continue  # the subject is never replaced in test mode

            new_agent = pop_scripted_replacement(
                agent, waiting_norm_customer, waiting_no_norm_customer
            )
            if new_agent is None:
                continue
            next_replacement_step, last_spawn_entrance = respawn_agent(
                env, new_agent, next_replacement_step, last_spawn_entrance,
                keep_out_of_region=region_staged(norm_type),
            )

        if subject_ids and subject_ids.issubset(done_agent_ids):
            subject_finished = True
            print(f"[INFO] Subject agent finished all goals at step {step_count}; ending test run")

    episode = summarize_episode(env, done_agent_ids, step_count, figures_root)
    episode['recorder'] = recorder
    return episode


def write_raw_run(episode, raw_runs_dir, run_id, **meta):
    """Dump one run's raw log (metadata + per-step subject trace) to JSON.

    These files are the only input experiment/evaluation/eval.py needs: it reads
    them back and derives every metric, so metrics can be recomputed or
    redefined without re-running the experiment.
    """
    os.makedirs(raw_runs_dir, exist_ok=True)
    raw_run = episode.pop('recorder').raw_run(run_id=run_id, **meta)
    raw_path = os.path.join(raw_runs_dir, f"{run_id}.json")
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(raw_run, f, indent=2)
    print(f"[OK] Raw run log saved to: {raw_path}")
    return raw_path


def run_test_stage(capacity, max_steps, figures_dir, norm_type, client,
                   subject_kind, subject_positions, world_type, env_kwargs,
                   reference_memory_path, memory_filename, stage_label,
                   num_no_norm_agents, raw_runs_dir, run_id,
                   keep_all_frames=False, subject_class=None, test_index=None,
                   ground_truth='without', prompt_stage='test', mode=None):
    """Stage 2/3 single run: same-world or new-world test with scripted turnover.

    keep_all_frames keeps the rendered PNG of every step of the run; by default
    only the steps the subject agent was on the map are kept.

    subject_class pins the subject agent's color for this run; run_test_series
    uses it to give the subject each of the three colors exactly once across a
    block of test runs.

    ground_truth and prompt_stage affect only the subject's information, not
    the test setup or simulation loop. mode is optional output metadata.
    """
    background_slots = capacity - len(subject_positions)
    customer_pool_size = capacity * 20

    env = make_env(norm_type, figures_dir, capacity, world_type, max_steps, **env_kwargs)

    initial_agents = build_initial_agents(
        capacity, subject_positions, env.world, norm_type, client,
        ground_truth=ground_truth,
        subject_kind=subject_kind,
        prompt_stage=prompt_stage,
        placement_stage='test',
        reference_memory_path=reference_memory_path,
        memory_filename=memory_filename,
        num_no_norm_agents=num_no_norm_agents,
        subject_class=subject_class,
        test_index=test_index,
    )
    subject_ids = {a.agent_id for a in initial_agents if is_subject_agent(a)}
    subject_classes = sorted(
        {a.agent_class for a in initial_agents if is_subject_agent(a)}
    )
    print(f"[INFO] Initial agents: {[a.agent_id for a in initial_agents]}")
    print(f"[INFO] Subject-controlled ({subject_kind}): {sorted(subject_ids)}")
    print(f"[INFO] Subject color: {', '.join(subject_classes)}")
    print(f"[INFO] Subject entry position: {sorted(subject_positions)[0] + 1}")
    subject_doors = sorted(
        {a.current_position for a in initial_agents if is_subject_agent(a)}
    )
    print(f"[INFO] Subject entry door: {subject_doors}")
    print(f"[INFO] World: {world_type}")
    print(
        f"[INFO] Mode={stage_label}: scripted customers are replaced when done; "
        f"the run ends when the subject finishes all goals or max_steps={max_steps}\n"
    )

    waiting_norm_customer, waiting_no_norm_customer, _ = build_scripted_pool(
        customer_pool_size, background_slots, num_no_norm_agents,
        env.world, norm_type, client, ground_truth='without',
        first_agent_number=capacity + 1,
    )

    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
    os.makedirs(figures_dir)

    episode = run_test_simulation(
        env=env,
        initial_agents=initial_agents,
        waiting_norm_customer=waiting_norm_customer,
        waiting_no_norm_customer=waiting_no_norm_customer,
        figures_root=figures_dir,
        norm_type=norm_type,
        stage_label=stage_label,
    )
    if not keep_all_frames:
        filter_subject_episode_frames(figures_dir, episode)

    subject_done = all(
        episode['final_states'].get(agent_id) == 'done' for agent_id in subject_ids
    )

    mode_metadata = {'mode': mode} if mode is not None else {}
    raw_run_path = write_raw_run(
        episode, raw_runs_dir, run_id,
        stage=stage_label,
        agent_type=subject_kind,
        norm_type=norm_type,
        world_type=world_type,
        ground_truth=ground_truth,
        capacity=capacity,
        max_steps=max_steps,
        subject_position=sorted(subject_positions)[0] + 1,
        **mode_metadata,
    )

    return {
        'stage': stage_label,
        'agent_type': subject_kind,
        'ground_truth': ground_truth,
        'norm_type': norm_type,
        'world_type': world_type,
        'capacity': capacity,
        'subject_position': sorted(subject_positions)[0] + 1,
        'subject_class': subject_classes[0] if subject_classes else None,
        'figures_dir': figures_dir,
        'raw_run_path': raw_run_path,
        'memory_filename': memory_filename,
        'subject_completed': subject_done,
        'steps': episode['steps'],
        'num_collisions': episode['num_collisions'],
        'norm_violations': summarize_norm_violations(episode['episode_log']),
        'agent_rewards': episode['agent_rewards'],
        'agent_classes': episode['agent_classes'],
        'final_states': episode['final_states'],
        **mode_metadata,
    }


def run_test_series(*, count, capacity, test_max_steps, figures_root, norm_type,
                    client, subject_kind, world_type, env_kwargs,
                    reference_memory_path, num_no_norm_agents, raw_runs_dir,
                    random_seed, seed_offset=0, stage_label='test',
                    keep_all_frames=False, ground_truth='without',
                    prompt_stage='test', mode=None):
    """Run a block of `count` test-stage runs and return their stage-result dicts.

    world_type is either a single world (every run uses it — the same-world
    test stage) or a sequence of `count` worlds, one per run (the new-world
    test stage; each run's id and figures dir fold in its world so they don't
    collide). Every run reseeds the global RNG with
    ``random_seed + seed_offset + <1-based run index>`` before drawing its
    setup, and the subject's color comes from
    ``subject_class_schedule(random_seed, count)`` so a block of three runs
    always covers orange / green / blue exactly once — callers running several
    blocks back to back (e.g. same-world tests then new-world tests) should
    give each block a seed_offset that keeps their index ranges from
    overlapping.
    """
    world_varies = isinstance(world_type, (list, tuple))
    worlds = list(world_type) if world_varies else [world_type] * count
    subject_classes = subject_class_schedule(random_seed, count)

    stage_results = []
    for index in range(1, count + 1):
        world = worlds[index - 1]
        test_seed = random_seed + seed_offset + index
        random.seed(test_seed)

        suffix = f"_{world}" if world_varies else ""
        run_id = f"{stage_label}_{index:03d}{suffix}"

        test_result = run_test_stage(
            capacity=capacity,
            max_steps=test_max_steps,
            figures_dir=os.path.join(figures_root, run_id),
            norm_type=norm_type,
            client=client,
            subject_kind=subject_kind,
            subject_positions=_random_subject_positions(capacity, norm_type),
            world_type=world,
            env_kwargs=env_kwargs,
            reference_memory_path=reference_memory_path,
            memory_filename=f"{run_id}.json" if is_llm_subject_kind(subject_kind) else None,
            stage_label=stage_label,
            num_no_norm_agents=num_no_norm_agents,
            raw_runs_dir=raw_runs_dir,
            run_id=run_id,
            subject_class=subject_classes[index - 1],
            keep_all_frames=keep_all_frames,
            test_index=index,
            ground_truth=ground_truth,
            prompt_stage=prompt_stage,
            mode=mode,
        )
        test_result['test_index'] = index
        test_result['random_seed'] = test_seed
        stage_results.append(test_result)

    return stage_results
