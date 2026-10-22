"""
Standalone planning algorithms used by oracle_agent.py (and potentially other agents).
All functions are pure with respect to agent state — agent fields that were formerly
accessed via `self` are passed as explicit parameters instead.
"""

from typing import Tuple, Optional, List, Dict, Set
import heapq

from src.env.actions import Action
from src.env.rewards import COLLISION_PENALTY, GOAL_REWARD, STEP_COST
from src.norms.norm_checker import check_norm_violation
from src.norms.lining import (
    generate_queue_line,
    get_queue,
    get_queue_frame,
    is_line_norm,
    get_line_initial_direction,
)
from src.norms.navigation import (
    is_spacing_norm,
    is_navigation_norm,
    is_xcell_norm,
    xcell_blocked_cells,
    xcell_requires_visit,
)


def compute_shortest_path(
    current_position: Tuple[int, int],
    target: Tuple[int, int],
    world,
) -> List[Tuple[int, int]]:
    """
    Compute shortest path from current_position to target using BFS.

    Returns:
        List of positions representing the path (excluding current position,
        including target).  Empty list if no path found or already at target.
    """
    from collections import deque

    if current_position == target:
        return []

    queue = deque([(current_position, [current_position])])
    visited = {current_position}

    while queue:
        current_pos, path = queue.popleft()

        for neighbor in world.get_neighbors(current_pos):
            if neighbor in visited:
                continue

            visited.add(neighbor)
            new_path = path + [neighbor]

            if neighbor == target:
                return new_path[1:]  # Exclude starting position

            queue.append((neighbor, new_path))

    return []


def get_blocked_queue_cells(
    norm: Optional[str],
    target_position: Optional[Tuple[int, int]],
    world,
    agent_id: str,
    agent_current_position: Tuple[int, int],
    debug_print: bool = True,
    on_first_target: bool = True,
) -> Set[Tuple[int, int]]:
    """
    For line norms, determine which queue cells should be blocked when the agent
    is not yet in the queue.

    Returns:
        Set of blocked positions, or empty set if not applicable.
    """
    blocked_queue_cells: Set[Tuple[int, int]] = set()

    if is_line_norm(norm):
        if on_first_target:
            queue_cells = generate_queue_line(target_position, world, norm)
            queue_entry_cell = queue_cells[1] if len(queue_cells) > 1 else target_position

            if (
                agent_current_position != queue_entry_cell
                and agent_id not in world.current_queue.values()
            ):
                blocked_queue_cells.add(target_position)

                direction = get_line_initial_direction(norm, world)
                if direction is not None:
                    _, _, right_vec = get_queue_frame(world)
                    opposite_position = (
                        target_position[0] - direction * right_vec[0],
                        target_position[1] - direction * right_vec[1],
                    )
                    if (0 <= opposite_position[0] < world.height
                            and 0 <= opposite_position[1] < world.width):
                        blocked_queue_cells.add(opposite_position)
                    if debug_print:
                        print(
                            f"Agent {agent_id} not in queue, blocking target position: "
                            f"{target_position} and opposite side: {opposite_position}"
                        )
                elif debug_print:
                    print(
                        f"Agent {agent_id} not in queue, blocking target position: {target_position}"
                    )
        else:
            # Agent has finished the first target — block all queue cells so they
            # cannot step back into the queue at any point during departure.
            agent_obj = next((a for a in getattr(world, 'agents', []) if a.agent_id == agent_id), None)
            if agent_obj is not None and agent_obj.target_positions:
                counter_pos = agent_obj.target_positions[0]
                queue_cells = generate_queue_line(counter_pos, world, norm)
                blocked_queue_cells.update(queue_cells)
                if debug_print:
                    print(
                        f"Agent {agent_id} has finished first target {counter_pos},"
                        f" blocking all queue cells: {set(queue_cells)}"
                    )

    return blocked_queue_cells


def heuristic(pos: Tuple[int, int], target: Tuple[int, int]) -> int:
    """Manhattan distance heuristic for A*."""
    return abs(pos[0] - target[0]) + abs(pos[1] - target[1])


def get_movement_cost(
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int],
    all_current_positions: Dict[str, Tuple[int, int]],
    world,
    agent_id: str,
    norm: Optional[str],
    target_position: Optional[Tuple[int, int]],
    is_immediate_next: bool = True,
) -> float:
    """
    Calculate cost of moving from from_pos to to_pos.

    Cost is the step cost for normal movement and the collision penalty if
    collision with another agent is possible.
    Returns float('inf') if the move violates a norm or is otherwise invalid.
    """
    if is_immediate_next:
        for aid, pos in all_current_positions.items():
            if aid != agent_id and pos == to_pos:
                base_cost = COLLISION_PENALTY
                break
        else:
            base_cost = STEP_COST
    else:
        base_cost = STEP_COST

    # Reward for reaching target
    if to_pos != from_pos and target_position and to_pos == target_position:
        base_cost -= GOAL_REWARD

    return base_cost


def compute_a_star_path(
    current_position: Tuple[int, int],
    revised_target: Tuple[int, int],
    world,
    agent_id: str,
    all_current_positions: Dict[str, Tuple[int, int]],
    norm: Optional[str],
    target_position: Optional[Tuple[int, int]],
    debug_print: bool = True,
    violation_log: Optional[Dict[str, List[str]]] = None,
    on_first_target: bool = True,
) -> Tuple[List[Tuple[int, int]], Optional[Action]]:
    """
    Compute shortest path from current_position to target using A*.

    Uses get_movement_cost() for action costs and world.transition() to determine
    actual next positions.  Includes WAIT to allow the agent to pause.

    Args:
        debug_print: If True, print detailed action costs and norm violations.
                     Set to False when predicting other agents.

    Returns:
        Tuple of (path, first_action):
        - path: List of positions (excluding current, including target).
        - first_action: First action to take, or None if no path / already at target.
    """
    if current_position == revised_target:
        return ([], None)

    # Apply norm logic when heading to the first (primary) target,
    # OR always for navigation norms (check norms on all moves).
    active_norm = norm if (on_first_target or is_navigation_norm(norm)) else None

    movement_actions = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]

    blocked_queue_cells = get_blocked_queue_cells(
        norm, target_position, world, agent_id, current_position, debug_print,
        on_first_target=on_first_target,
    )

    # Cells the norm bars for the whole route, not just the immediate step (the
    # X cells under xcell_forbidden — see xcell_blocked_cells). The per-step norm
    # check below cannot see them coming, so they are cut from the search space.
    blocked_cells = xcell_blocked_cells(active_norm, world)

    open_set: list = []
    counter = 0

    start_h = heuristic(current_position, revised_target)
    heapq.heappush(open_set, (start_h, counter, current_position, [current_position], []))

    visited: Set[Tuple[int, int]] = set()
    g_scores: Dict[Tuple[int, int], float] = {current_position: 0}

    while open_set:
        _, _, current_pos, path, action_seq = heapq.heappop(open_set)

        if current_pos == revised_target:
            first_action = action_seq[0] if action_seq else None
            return (path[1:], first_action)

        for action in movement_actions:
            is_immediate_next = len(action_seq) == 0
            if action == Action.WAIT:
                next_pos = current_pos
            else:
                next_pos = world.transition(current_pos, action)

                if next_pos in visited or next_pos == current_pos or next_pos is None:
                    continue

                # Sittable cells (chairs, etc.) are only a valid destination,
                # not a pass-through — skip if not the final target.
                if world.is_sittable(next_pos) and next_pos != revised_target:
                    continue

                if next_pos in blocked_queue_cells:
                    if debug_print:
                        print(
                            f"Skipping action {action} from {current_pos} -> {next_pos}"
                            " - blocked queue cell"
                        )
                    continue

                if next_pos in blocked_cells:
                    if debug_print:
                        print(
                            f"Skipping action {action} from {current_pos} -> {next_pos}"
                            f" - cell forbidden by {active_norm} for the whole route"
                        )
                    continue

            if not is_immediate_next:
                visited.add(next_pos)

            if is_immediate_next and active_norm:
                if check_norm_violation(active_norm, current_pos, next_pos, target_position, all_current_positions, world, agent_id):
                    if violation_log is not None:
                        violation_log.setdefault(agent_id, []).append(str(action))
                    if debug_print:
                        print(f"Skipping action {action} from {current_pos} -> {next_pos} due to norm violation")
                    continue

            move_cost = get_movement_cost(
                current_pos, next_pos,
                all_current_positions, world,
                agent_id, active_norm, target_position,
                is_immediate_next=is_immediate_next,
            )

            if move_cost == float('inf'):
                if debug_print:
                    print(f"Skipping action {action} from {current_pos} -> {next_pos} due to infinite cost.")
                continue

            if debug_print:
                print(
                    f"Agent {agent_id} action {action} from {current_pos} -> {next_pos},"
                    f" cost: {move_cost}"
                )

            tentative_g = g_scores[current_pos] + move_cost

            if next_pos not in g_scores or tentative_g < g_scores[next_pos]:
                g_scores[next_pos] = tentative_g
                f_score = tentative_g + heuristic(next_pos, revised_target)

                counter += 1
                heapq.heappush(
                    open_set,
                    (f_score, counter, next_pos, path + [next_pos], action_seq + [action]),
                )

    return ([], None)


def predictive_planner(
    agent_id: str,
    current_position: Tuple[int, int],
    norm: Optional[str],
    revised_target: Tuple[int, int],
    world,
    all_current_positions: Dict[str, Tuple[int, int]],
    collision_actions: Optional[Dict[str, Action]] = None,
    violation_log: Optional[Dict[str, List[str]]] = None,
    all_target_positions: Optional[Dict[str, Tuple[int, int]]] = None,
    all_target_indices: Optional[Dict[str, int]] = None,
) -> Optional[Action]:
    """
    Level-2 planner with one-step lookahead over other agents.

    Predicts where other agents will move, then plans for agent_id considering
    those predicted positions.

    Algorithm:
    1. For colliding agents, use their next states from collision_actions.
    2. For non-colliding agents, predict their actions via compute_a_star_path (level-1).
    3. Predict other agents' next states using world.transition().
    4. Plan for agent_id using the mixed state (predicted states of others + current
       state of agent_id).

    Returns:
        The next action to take, or None if no valid action found.
    """
    target_position = (all_target_positions or {}).get(agent_id)
    on_first_target = (all_target_indices or {}).get(agent_id, 0) == 0

    other_agents = [
        (aid, pos) for aid, pos in all_current_positions.items() if aid != agent_id
    ]
    other_agents_sorted = sorted(
        other_agents, key=lambda x: x[0]
    )

    updated_positions = all_current_positions.copy()
    collision_actions = collision_actions or {}

    print("planning for agent:", agent_id)
    print(f"Predicting agents in order: {[aid for aid, _ in other_agents_sorted]}")
    print(f"Collision actions: {collision_actions}")

    # If the planning agent is already in the queue, skip predicting all other agents.
    if is_line_norm(norm) and agent_id in world.current_queue.values():
        print(f"Agent {agent_id} is in the queue; skipping all other-agent prediction.")
    else:
        for other_id, position in other_agents_sorted:
            if other_id in collision_actions:
                collision_action = collision_actions[other_id]
                print(f"Agent {other_id} has collision action: {collision_action}")

                next_pos = world.transition(position, collision_action)
                if next_pos:
                    updated_positions[other_id] = next_pos
                    print(f"Updated position for agent {other_id} from collision: {next_pos}")
            else:
                # Predict this agent's target using its own target position
                other_true_target = (all_target_positions or {}).get(other_id, target_position)
                other_on_first_target = (all_target_indices or {}).get(other_id, 0) == 0
                other_revised_target = other_true_target
                if other_on_first_target and is_line_norm(norm):
                    queue_cells = generate_queue_line(other_true_target, world, norm)
                    next_queue_index = (max(world.current_queue.keys()) + 1) if len(world.current_queue) > 1 else 1
                    if next_queue_index < len(queue_cells):
                        # other agent's target is a valid queue destination
                        queue_entry_cell = queue_cells[next_queue_index]
                        if position != queue_entry_cell and other_id not in world.current_queue.values():
                            other_revised_target = queue_entry_cell
                    # else: target is not a queue destination (e.g. agent heading to exit);
                    # leave other_revised_target = other_true_target

                elif other_on_first_target and is_xcell_norm(norm):
                    other_agent_obj = next((a for a in world.agents if a.agent_id == other_id), None)
                    if xcell_requires_visit(norm):
                        x_cells = world.zones.get('affordance_x', [])
                        if x_cells:
                            candidates = x_cells if not getattr(other_agent_obj, 'visited_x_cells', set()) else []
                            if candidates:
                                other_revised_target = min(
                                    candidates,
                                    key=lambda c: abs(c[0] - position[0]) + abs(c[1] - position[1])
                                )

                predicted_path, _ = compute_a_star_path(
                    position,
                    other_revised_target,
                    world,
                    other_id,
                    updated_positions,
                    norm,          # assume same norm for prediction
                    other_true_target,
                    debug_print=False,
                    on_first_target=other_on_first_target,
                )

                if predicted_path:
                    updated_positions[other_id] = predicted_path[0]

    print("mixed positions for planning:", updated_positions)

    path, action = compute_a_star_path(
        current_position,
        revised_target,
        world,
        agent_id,
        updated_positions,
        norm,
        target_position,
        debug_print=True,
        violation_log=violation_log,
        on_first_target=on_first_target,
    )
    print(f"=== Final path for agent {agent_id}: {path}, action: {action} ===")

    return action
