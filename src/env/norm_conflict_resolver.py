"""
Standalone helpers for resolving and detecting norm conflicts during step execution.

These functions are extracted from NormEnv so they can be tested and reused
independently.  Each function takes explicit parameters instead of relying on
``self``.
"""

from typing import Any, Callable, Dict, List, Set, Tuple

from src.env.actions import Action
from src.norms.lining import (
    generate_queue_line,
    get_next_queue_cell,
    get_queue,
    get_queue_start,
    is_line_norm,
)
from src.norms.navigation import check_navigation_norm
from src.norms.occupation import (
    region_capacity,
    region_single_class_only,
    is_occupation_region_norm,
    waiting_zone_cells,
    check_occupation_norm,
)


# ---------------------------------------------------------------------------
# Conflict resolution (called per agent during _get_all_agent_next_positions)
# ---------------------------------------------------------------------------

def resolve_spacing_norm_conflict(
    agent,
    step_data: Dict[str, Any],
    agent_classes: Dict[str, int],
    agents: list,
    world,
    can_move_fn: Callable,
) -> None:
    """
    For a spacing-norm agent, check if its current next_position either lands
    on the same cell as another agent's committed next position or violates the
    spacing norm.  If so, reassign the agent to the first alternative action
    (UP/DOWN/LEFT/RIGHT/WAIT) that satisfies both constraints.

    Args:
        agent: The agent whose next_position may need to be changed.
        step_data: Step data containing already-committed agent actions.
        agent_classes: Mapping of agent_id to agent_class.
        agents: Full list of agents in the environment.
        world: The environment world object.
        can_move_fn: Callable(agent, pos) -> bool — position reachability check.
    """
    # Build two position maps using only ACTIVE, non-done agents:
    # - committed_positions: agents already finalized this step (definitive next positions)
    # - all_predicted: best guess for all active agents (for spacing-distance checks)
    # Inactive agents (not yet arrived) are excluded — their spawn position
    # must not trigger false spacing violations.
    committed_positions: Dict[str, Tuple[int, int]] = {}
    all_predicted: Dict[str, Tuple[int, int]] = {}
    for other in agents:
        if other.agent_id == agent.agent_id:
            continue
        if not other.active or other.done:
            continue
        if other.norm is None or not other.is_oracle:
            continue
        if other.agent_id in step_data['actions']:
            pos = step_data['actions'][other.agent_id]['next_position']
            committed_positions[other.agent_id] = pos
            all_predicted[other.agent_id] = pos
        elif other.next_position is not None:
            all_predicted[other.agent_id] = other.next_position
        # agents with next_position=None and not yet committed are skipped

    # Only check same-cell against COMMITTED agents — an uncommitted agent's
    # planned next_position may coincide with agent.next_position even when
    # can_move_fn already approved the move (e.g. they're both heading to the
    # same empty cell; collision resolution handles that separately).
    same_cell_conflict = agent.next_position in committed_positions.values()
    spacing_violated = check_navigation_norm(
        agent.norm,
        agent.current_position,
        agent.next_position,
        all_predicted,
        world,
        agent.agent_id,
        agent_classes,
    )

    if not same_cell_conflict and not spacing_violated:
        return

    print(
        f"Spacing norm agent {agent.agent_id}: next_pos {agent.next_position} "
        f"conflicts (same_cell={same_cell_conflict}, "
        f"spacing_violated={spacing_violated}); searching alternative action."
    )

    candidate_actions = [
        Action.MOVE_UP, Action.MOVE_DOWN,
        Action.MOVE_LEFT, Action.MOVE_RIGHT,
        Action.WAIT,
    ]
    new_action = None
    new_pos = agent.current_position  # fallback: stay put

    for alt_action in candidate_actions:
        alt_pos = world.transition(agent.current_position, alt_action)
        if not can_move_fn(agent, alt_pos):
            continue
        if alt_pos in committed_positions.values():
            continue
        if check_navigation_norm(
            agent.norm,
            agent.current_position,
            alt_pos,
            all_predicted,
            world,
            agent.agent_id,
            agent_classes,
        ):
            continue
        new_action = alt_action
        new_pos = alt_pos
        break

    if new_action is None:
        new_action = Action.WAIT
        new_pos = agent.current_position

    print(
        f"Spacing norm agent {agent.agent_id}: reassigned action "
        f"{new_action} -> next_pos {new_pos}."
    )
    agent.action = new_action
    agent.next_position = new_pos

def resolve_queue_norm_conflict(
    agents: list,
    step_data: Dict[str, Any],
    world,
    norm_type: str,
    can_move_fn: Callable,
) -> None:
    """
    For line norms: make sure the open cell at the end of the queue gets filled
    when a norm-following oracle agent is standing next to it.

    The queue norm obliges any agent that is not yet in the queue and is adjacent
    to the open end-of-queue cell to step into it — every other move is a
    violation.  Agents plan against *predicted* positions of the others, so an
    agent can wrongly believe a neighbour is about to take that cell (and route
    around it, since stepping onto a predicted occupant also carries the
    collision penalty).  The grader then scores against real positions, where the
    cell is still empty, and flags them all.

    This runs once per step after every next_position is committed: if the slot
    is open and nobody is stepping in, the earliest-arriving eligible oracle
    neighbour is reassigned into it.  Called only when the slot's neighbourhood
    is made up of oracle norm agents — a free-choice (LLM/human) agent next to
    the slot owns its decision, and is left to comply or violate on its own.

    Args:
        agents: Full list of agents in the environment.
        step_data: Step data containing this step's committed agent actions.
        world: The environment world object.
        norm_type: The active global norm string.
        can_move_fn: Callable(agent, pos) -> bool — position reachability check.
    """
    if not is_line_norm(norm_type):
        return

    active = [a for a in agents if a.active and not a.done]
    positions = {a.agent_id: a.current_position for a in active}

    queue_cells = generate_queue_line(get_queue_start(world, norm_type), world, norm_type)
    if len(queue_cells) < 2:
        return

    # Agents past their first target are only passing through — they never count
    # as queue members (same rule the grader uses).
    eligible_agents = {a.agent_id for a in active if a.target_index == 0}
    current_queue = get_queue(
        queue_cells,
        positions,
        previous_queue=world.previous_queue,
        eligible_agents=eligible_agents,
    )

    # Publish it: check_queue_line reads world.current_queue, and the value left by
    # the planning phase can be stale by now. Swap/push resolution runs right after
    # this and must judge norms against the same queue the grader will use.
    world.current_queue = current_queue

    slot = get_next_queue_cell(queue_cells, current_queue)
    if slot is None or slot in positions.values():
        return  # queue is full, or the end-of-queue cell is already taken

    def committed_pos(agent):
        return step_data['actions'].get(agent.agent_id, {}).get(
            'next_position', agent.next_position,
        )

    queue_members = {aid for aid in current_queue.values() if aid is not None}
    displaced = step_data.get('collision_displaced', set())
    neighbours = set(world.get_neighbors(slot))

    # Agents the norm obliges to step into the slot: standing next to it, not yet
    # in the queue, still heading to the counter, and moving by choice.
    obliged = [
        a for a in active
        if a.current_position in neighbours
        and a.target_index == 0
        and a.agent_id not in queue_members
        and a.agent_id not in displaced
    ]

    claimer = next((a for a in active if committed_pos(a) == slot), None)

    if claimer is None:
        # A free-choice (LLM/human) agent next to the slot owns its decision: don't
        # step an oracle in over its head, and don't excuse the others if it stays.
        if any(not (a.is_oracle and a.norm is not None and is_line_norm(a.norm))
               for a in obliged):
            return

        for agent in sorted(obliged, key=lambda a: (a.arrival_time, a.agent_id)):
            if not can_move_fn(agent, slot):
                continue
            action = next(
                (act for act in (Action.MOVE_UP, Action.MOVE_DOWN,
                                 Action.MOVE_LEFT, Action.MOVE_RIGHT)
                 if world.transition(agent.current_position, act) == slot),
                None,
            )
            if action is None:
                continue

            print(
                f"Queue norm agent {agent.agent_id}: end-of-queue cell {slot} is open "
                f"and unclaimed; reassigning {agent.action} -> {action} "
                f"(next_pos {agent.next_position} -> {slot})."
            )
            agent.action = action
            agent.next_position = slot
            step_data['actions'][agent.agent_id] = {
                'action': action,
                'current_position': agent.current_position,
                'next_position': slot,
                'intended_position': slot,
            }
            claimer = agent
            break

    if claimer is None:
        return  # nobody could take the slot — leave the step as planned

    # Only one agent fits in the slot. The other obliged neighbours physically
    # cannot comply once it is claimed, so record them: the grader judges their
    # move with the slot counted as occupied, which is the state they were in.
    yielded = {a.agent_id for a in obliged if a.agent_id != claimer.agent_id}
    if yielded:
        step_data['queue_slot_claim'] = {
            'slot': slot,
            'claimer': claimer.agent_id,
            'yielded': yielded,
        }
        print(
            f"Queue norm: {claimer.agent_id} takes end-of-queue cell {slot}; "
            f"{sorted(yielded)} could not and are judged against a filled slot."
        )


def check_spacing_violations(
    step_data: Dict[str, Any],
    agent_classes: Dict[str, int],
    norm_type: str,
    agents: list,
    exempt_agent_ids: Set[str] = frozenset(),
) -> List[Tuple]:
    """
    For adjacency-based spacing norms (019–023), check all pairs of active agents'
    next positions (i.e., where they will be after this step).  Returns a list of
    (agent_id, from_pos, to_pos) violation tuples — the caller is responsible for
    routing them into pending or current step data.

    adjacency rules:
      spacing_no_adjacent        : Manhattan distance == 1  (cardinal)
      spacing_no_diagonal        : |Δrow| == 1 and |Δcol| == 1  (diagonal)

    Args:
        step_data: Step data containing committed agent actions.
        agent_classes: Mapping of agent_id to agent_class.
        norm_type: The active spacing norm string.
        agents: Full list of agents in the environment.
        exempt_agent_ids: Agents whose move this step was forced by collision
            displacement — they are never flagged for the violation it causes.
    """
    active = [
        a for a in agents
        if a.active and not a.done and a.agent_id in step_data['actions']
    ]

    violations: List[Tuple] = []
    violated_ids: Set[str] = set()

    for i, agent_a in enumerate(active):
        pos_a = step_data['actions'][agent_a.agent_id].get('next_position')
        if pos_a is None:
            continue
        for agent_b in active[i + 1:]:
            pos_b = step_data['actions'][agent_b.agent_id].get('next_position')
            if pos_b is None:
                continue

            row_diff = abs(pos_a[0] - pos_b[0])
            col_diff = abs(pos_a[1] - pos_b[1])
            manhattan = row_diff + col_diff
            is_cardinal_adj = manhattan == 1
            is_diagonal_adj = row_diff == 1 and col_diff == 1
            str_a = agent_classes.get(agent_a.agent_id, 0)
            str_b = agent_classes.get(agent_b.agent_id, 0)

            pair_violated = False
            if norm_type == 'spacing_no_adjacent' and is_cardinal_adj:
                pair_violated = True
            elif norm_type == 'spacing_no_diagonal' and is_diagonal_adj:
                pair_violated = True

            if pair_violated:
                for (agent, to_pos), (partner, _) in [
                    ((agent_a, pos_a), (agent_b, pos_b)),
                    ((agent_b, pos_b), (agent_a, pos_a)),
                ]:
                    if agent.agent_id in exempt_agent_ids:
                        continue  # moved by collision displacement, not by choice
                    if agent.agent_id not in violated_ids:
                        action_data = step_data['actions'][agent.agent_id]
                        from_pos = action_data.get('current_position')
                        if from_pos == to_pos:
                            if action_data.get('action') != Action.PUSH:
                                continue  # agent stayed without pushing — not responsible for the violation
                            if partner.agent_id in exempt_agent_ids:
                                # The stayer held its cell while its collision
                                # partner was force-displaced next to it — the
                                # adjacency was created by the resolution, not
                                # by either agent's choice.
                                continue
                        violated_ids.add(agent.agent_id)
                        violations.append((agent.agent_id, from_pos, to_pos))
                        print(f"  [NORM VIOLATION] {agent.agent_id} violated {norm_type} norm: {from_pos} -> {to_pos} (will appear in next step)")

    return violations


def resolve_occupation_region_norm_conflict(
    agent,
    step_data: Dict[str, Any],
    agent_classes: Dict[str, int],
    agents: list,
    world,
    can_move_fn: Callable,
) -> None:
    """
    For a waiting-zone occupation region agent (is_occupation_region_norm), check
    whether the agent's committed next_position would break the region norm given
    where the other agents end up this step.  Handles every region norm:

      - capacity_1/2/3    : at most N agents may occupy the zone.
      - single_class_only : only one agent class may occupy the zone at a time.

    If entering would violate the norm, reassign the agent to the first
    alternative action (UP/DOWN/LEFT/RIGHT/WAIT) that does not (and that does not
    land on a cell another agent has already committed to).

    Called per agent (right after resolve_spacing_norm_conflict) as next_positions
    are set, so each agent counts the zone occupants implied by the other agents'
    next positions — committed ones for agents already finalized this step, and
    their best-guess next_position otherwise.  The planner already blocks entry to
    a disallowed zone, but agents plan independently against the same start-of-step
    snapshot, so two may decide to enter the same/last slot simultaneously; this
    resolver makes the later-processed agent yield once the slot is taken.

    Args:
        agent: The agent whose next_position may need to be changed.
        step_data: Step data containing already-committed agent actions.
        agent_classes: Mapping of agent_id to agent_class.
        agents: Full list of agents in the environment.
        world: The environment world object.
        can_move_fn: Callable(agent, pos) -> bool — position reachability check.
    """
    zone = waiting_zone_cells(world)

    # Only relevant when the agent is entering the zone from outside it. An agent
    # already inside the zone keeps its slot and may move within it freely.
    if agent.current_position in zone or agent.next_position not in zone:
        return

    # Other active norm agents' next positions: committed where finalized this
    # step, otherwise their current best-guess next_position. These are the
    # occupants check_occupation_norm counts when judging the zone.
    others_next: Dict[str, Tuple[int, int]] = {}
    for other in agents:
        if other.agent_id == agent.agent_id:
            continue
        if not other.active or other.done:
            continue
        if other.norm is None or not other.is_oracle:
            continue
        if other.next_position is not None:
            others_next[other.agent_id] = other.next_position

    # Does entering the zone at next_position break the region norm?
    if not check_occupation_norm(
        agent.norm, agent.current_position, agent.next_position,
        others_next, world, agent.agent_id, agent_classes,
    ):
        return  # entry is allowed

    print(
        f"Occupation region agent {agent.agent_id}: next_pos {agent.next_position} "
        f"would break {agent.norm}; searching alternative action."
    )

    candidate_actions = [
        Action.MOVE_UP, Action.MOVE_DOWN,
        Action.MOVE_LEFT, Action.MOVE_RIGHT,
        Action.WAIT,
    ]
    new_action = None
    new_pos = agent.current_position  # fallback: stay put (outside the zone)

    for alt_action in candidate_actions:
        alt_pos = world.transition(agent.current_position, alt_action)
        if not can_move_fn(agent, alt_pos):
            continue
        if alt_pos in others_next.values():
            continue
        if check_occupation_norm(
            agent.norm, agent.current_position, alt_pos,
            others_next, world, agent.agent_id, agent_classes,
        ):
            continue
        new_action = alt_action
        new_pos = alt_pos
        break

    if new_action is None:
        new_action = Action.WAIT
        new_pos = agent.current_position

    print(
        f"Occupation region agent {agent.agent_id}: reassigned action "
        f"{new_action} -> next_pos {new_pos}."
    )
    agent.action = new_action
    agent.next_position = new_pos


def check_occupation_region_violations(
    step_data: Dict[str, Any],
    norm_type: str,
    agent_classes: Dict[str, int],
    agents: list,
    world,
    exempt_agent_ids: Set[str] = frozenset(),
) -> List[Tuple]:
    """
    For occupation region norms (is_occupation_region_norm), check where agents
    actually end up this step and flag the agents that entered the waiting zone
    this step in violation of the norm:

      - capacity_1/2/3    : enterers beyond the available slots.
      - single_class_only : enterers whose class differs from the class that holds
                            the zone (stayers' class, or the first enterer's class
                            when the zone was empty).

    Returns a list of (agent_id, from_pos, to_pos) violation tuples — the caller
    routes them into pending/current step data. Agents already inside the zone
    keep their slots/class; only agents that moved in from outside are reported.
    This mirrors check_spacing_violations and catches violations the resolver
    cannot fix — notably LLM/human agents, which are never reassigned.

    Args:
        step_data: Step data containing committed agent actions.
        norm_type: The active occupation norm string.
        agent_classes: Mapping of agent_id to agent_class.
        agents: Full list of agents in the environment.
        world: The environment world object.
        exempt_agent_ids: Agents pushed into the zone by collision displacement
            — they are never flagged, and they don't consume capacity slots
            ahead of agents that entered by choice.
    """
    if not is_occupation_region_norm(norm_type):
        return []

    zone = waiting_zone_cells(world)
    active = [
        a for a in agents
        if a.active and not a.done and a.agent_id in step_data['actions']
    ]

    stayers = []   # agents already in the zone that remain inside
    enterers = []  # agents that moved into the zone from outside this step
    for a in active:
        action_data = step_data['actions'][a.agent_id]
        from_pos = action_data.get('current_position')
        to_pos = action_data.get('next_position')
        if to_pos not in zone:
            continue
        if from_pos in zone:
            stayers.append(a)
        else:
            enterers.append(a)

    # Displaced agents go last so they take the leftover (violating) slots —
    # which they are exempt from — instead of pushing a voluntary enterer
    # over capacity, and so allowed_class comes from a voluntary enterer.
    enterers.sort(key=lambda a: a.agent_id in exempt_agent_ids)

    cap = region_capacity(norm_type)
    if cap is not None:
        if len(stayers) + len(enterers) <= cap:
            return []
        allowed_to_enter = max(0, cap - len(stayers))
        losers = enterers[allowed_to_enter:]
    elif region_single_class_only(norm_type):
        if not enterers:
            return []
        stayer_classes = {agent_classes.get(a.agent_id) for a in stayers}
        allowed_class = (next(iter(stayer_classes)) if stayer_classes
                         else agent_classes.get(enterers[0].agent_id))
        losers = [a for a in enterers
                  if agent_classes.get(a.agent_id) != allowed_class]
    else:
        return []

    losers = [a for a in losers if a.agent_id not in exempt_agent_ids]

    violations: List[Tuple] = []
    for a in losers:
        action_data = step_data['actions'][a.agent_id]
        from_pos = action_data.get('current_position')
        to_pos = action_data.get('next_position')
        violations.append((a.agent_id, from_pos, to_pos))
        print(f"  [NORM VIOLATION] {a.agent_id} violated {norm_type} norm: {from_pos} -> {to_pos} (will appear in next step)")

    return violations


