from typing import Tuple, List, Dict, Optional, Set


# Queue cells extend at most this many cells out from the counter, measured
# along the queue's forward (away-from-counter) axis. For the classic
# top-wall counter on row 1 this reproduces the old "never below row 5" cap.
QUEUE_MAX_DEPTH = 4

# Waiting off the queue path counts as a violation only within this many cells
# (Chebyshev distance) of the counter: for the cafe counter at (1, 6) that is
# the box rows 1..3 x cols 4..8.
COUNTER_WAIT_RADIUS = 2


ZIGZAG_LINE_VARIANTS = {
    "zigzag_right2_line": (1, 2),
    "zigzag_left1_line": (-1, 1),
    "zigzag_right1_line": (1, 1),
    "zigzag_left2_line": (-1, 2),
}

ONE_BEND_LINE_VARIANTS = {
    "bend_right1_line": (1, 1),
    "bend_left2_line": (-1, 2),
    "bend_left1_line": (-1, 1),
    "bend_right2_line": (1, 2),
}

# (lateral direction, number of diagonal stair steps): each stair step is one
# lateral cell then one away cell; after the stairs the line runs straight away.
STAIRCASE_LINE_VARIANTS = {
    "staircase_right_line": (1, 2),
    "staircase_left_line": (-1, 2),
}

# (lateral direction, away span, lateral span): first away_span cells straight
# away from the counter, then lateral_span cells sideways, then straight away.
LSHAPE_LINE_VARIANTS = {
    "lshape_right_line": (1, 2, 2),
    "lshape_left_line": (-1, 2, 2),
}

# Square offsets are in the queue's local frame: (steps away from the
# counter, lateral steps to the right of an agent facing the counter).
SQUARE_LINE_VARIANTS = {
    "square_right_line": (
        (0, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (2, 0),
        (2, -1),
        (1, -1),
    ),
    "square_left_line": (
        (0, 0),
        (0, -1),
        (1, -1),
        (2, -1),
        (2, 0),
        (2, 1),
        (1, 1),
    ),
}

EDGE_LINE_VARIANTS = {
    "edge_left_line": -1,
    "edge_right_line": 1,
}

LEFT_LINE_NORMS = {
    "zigzag_left1_line",
    "zigzag_left2_line",
    "bend_left1_line",
    "bend_left2_line",
    "edge_left_line",
    "square_left_line",
    "staircase_left_line",
    "lshape_left_line",
}

def is_line_norm(norm: Optional[str]) -> bool:
    return norm in {
        "straight_line",
        *ZIGZAG_LINE_VARIANTS.keys(),
        *ONE_BEND_LINE_VARIANTS.keys(),
        *STAIRCASE_LINE_VARIANTS.keys(),
        *LSHAPE_LINE_VARIANTS.keys(),
        *SQUARE_LINE_VARIANTS.keys(),
        *EDGE_LINE_VARIANTS.keys(),
    }


def get_counter_position(world) -> Optional[Tuple[int, int]]:
    counter_zone = getattr(world, 'zones', {}).get('counter')
    if counter_zone:
        return tuple(counter_zone[0])
    return None


def is_near_counter(pos: Tuple[int, int], world,
                    radius: int = COUNTER_WAIT_RADIUS) -> bool:
    """True if pos lies within `radius` cells (Chebyshev) of the counter."""
    counter_pos = get_counter_position(world)
    if counter_pos is None:
        return False
    return (abs(pos[0] - counter_pos[0]) <= radius
            and abs(pos[1] - counter_pos[1]) <= radius)


def counter_front_cell(counter_pos: Tuple[int, int],
                       world_width: int) -> Tuple[int, int]:
    """The first queue cell (service spot) for a counter.

    One row below a top-wall counter; the adjacent interior cell for a
    counter on the left wall (column 1) or right wall (column width-2).
    """
    r, c = counter_pos
    if c == 1:
        return (r, c + 1)
    if c == world_width - 2:
        return (r, c - 1)
    return (r + 1, c)


def get_queue_frame(world) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Return (origin, down_vec, right_vec) of the queue's local frame.

    origin is the first queue cell, directly in front of the counter;
    down_vec points away from the counter (the direction the queue grows);
    right_vec is 'right' from the perspective of an agent standing in the
    queue facing the counter.
    """
    counter_pos = get_counter_position(world) or (1, 6)
    r, c = counter_pos
    if c == 1:  # counter on the left wall — queue grows rightward
        return (r, c + 1), (0, 1), (-1, 0)
    if c == world.width - 2:  # counter on the right wall — queue grows leftward
        return (r, c - 1), (0, -1), (1, 0)
    return (r + 1, c), (1, 0), (0, 1)  # counter on the top wall — queue grows downward


def counter_service_cell(world, norm: Optional[str] = None) -> Tuple[int, int]:
    """The cell an agent must stand on to be served, for this norm.

    Edge lines are served from the counter's side: the neighbour of the counter
    on the queue's lateral side, so the head of the line already stands against
    the wall and the whole queue runs along that edge. Every other norm is
    served from the cell directly in front of the counter.
    """
    origin, _, right_vec = get_queue_frame(world)
    direction = EDGE_LINE_VARIANTS.get(norm)
    if direction is None:
        return origin
    counter_pos = get_counter_position(world)
    if counter_pos is None:
        return origin
    return (counter_pos[0] + direction * right_vec[0],
            counter_pos[1] + direction * right_vec[1])


def get_queue_start(world, norm: Optional[str] = None) -> Tuple[int, int]:
    """The first queue cell (the service spot for this norm's counter)."""
    return counter_service_cell(world, norm)


def get_line_initial_direction(norm: Optional[str], world) -> Optional[int]:
    """Return initial lateral direction (in the queue's local frame) for
    deterministic line variants: +1 = the agent's right when facing the
    counter, -1 = their left."""
    if norm in ZIGZAG_LINE_VARIANTS:
        return ZIGZAG_LINE_VARIANTS[norm][0]
    if norm in ONE_BEND_LINE_VARIANTS:
        return ONE_BEND_LINE_VARIANTS[norm][0]
    if norm in STAIRCASE_LINE_VARIANTS:
        return STAIRCASE_LINE_VARIANTS[norm][0]
    if norm in LSHAPE_LINE_VARIANTS:
        return LSHAPE_LINE_VARIANTS[norm][0]
    if norm in SQUARE_LINE_VARIANTS:
        # lateral sign of the square's first sideways step
        return 1 if SQUARE_LINE_VARIANTS[norm][1][1] >= 0 else -1
    if norm in EDGE_LINE_VARIANTS:
        return EDGE_LINE_VARIANTS[norm]
    return None


def _queue_depth(cell: Tuple[int, int],
                 counter_pos: Tuple[int, int],
                 down_vec: Tuple[int, int]) -> int:
    """Steps from the counter to cell along the queue's forward axis."""
    return ((cell[0] - counter_pos[0]) * down_vec[0]
            + (cell[1] - counter_pos[1]) * down_vec[1])


def _cap_queue_at_max_depth(
    cells: List[Tuple[int, int]],
    counter_pos: Tuple[int, int],
    down_vec: Tuple[int, int],
    max_depth: int = QUEUE_MAX_DEPTH,
) -> List[Tuple[int, int]]:
    """Return the queue prefix before it would extend past max_depth."""
    capped: List[Tuple[int, int]] = []
    for cell in cells:
        if _queue_depth(cell, counter_pos, down_vec) > max_depth:
            break
        capped.append(cell)
    return capped


def _in_interior(world, cell: Tuple[int, int]) -> bool:
    r, c = cell
    return 1 <= r <= world.height - 2 and 1 <= c <= world.width - 2


def _is_queue_cell_walkable(world, cell: Tuple[int, int]) -> bool:
    if hasattr(world, "is_walkable"):
        return world.is_walkable(cell)
    r, c = cell
    return world.grid[r, c] in [0, 2]


def _prune_deadends(
    cells: Set[Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    pruned = set(cells)
    changed = True
    while changed:
        changed = False
        to_remove = []
        for r, c in pruned:
            neighbors = sum(
                (r + dr, c + dc) in pruned
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )
            if neighbors < 2:
                to_remove.append((r, c))
        if to_remove:
            for cell in to_remove:
                pruned.discard(cell)
            changed = True
    return pruned


def _generate_queue_edge(
    target_position: Tuple[int, int],
    world,
    direction: int,
    counter_pos: Tuple[int, int],
    down_vec: Tuple[int, int],
    right_vec: Tuple[int, int],
    max_depth: int = QUEUE_MAX_DEPTH,
) -> List[Tuple[int, int]]:
    candidate_cells = {
        cell for cell in getattr(world, "walkable_cells", set())
        if 0 <= _queue_depth(cell, counter_pos, down_vec) <= max_depth
    }
    if not candidate_cells:
        for r in range(world.height):
            for c in range(world.width):
                cell = (r, c)
                if (0 <= _queue_depth(cell, counter_pos, down_vec) <= max_depth
                        and _is_queue_cell_walkable(world, cell)):
                    candidate_cells.add(cell)

    loop_cells = _prune_deadends(candidate_cells)

    lateral = (direction * right_vec[0], direction * right_vec[1])
    first_step = (target_position[0] + lateral[0], target_position[1] + lateral[1])
    cells = [target_position]
    if target_position not in loop_cells or first_step not in loop_cells:
        return _cap_queue_at_max_depth(cells, counter_pos, down_vec, max_depth=max_depth)
    cells.append(first_step)

    direction_priority = [
        (-down_vec[0], -down_vec[1]),  # toward the counter
        lateral,                       # continue laterally along the wall
        down_vec,                      # away from the counter
        (-lateral[0], -lateral[1]),    # lateral back
    ]

    def find_path(
        current: Tuple[int, int],
        visited: Set[Tuple[int, int]],
    ) -> Optional[List[Tuple[int, int]]]:
        if _queue_depth(current, counter_pos, down_vec) == max_depth:
            return [current]

        for dr, dc in direction_priority:
            nxt = (current[0] + dr, current[1] + dc)
            if nxt not in loop_cells or nxt in visited:
                continue
            suffix = find_path(nxt, visited | {nxt})
            if suffix is not None:
                return [current] + suffix
        return None

    path = find_path(first_step, {target_position, first_step})
    if path is None:
        return _cap_queue_at_max_depth(cells, counter_pos, down_vec, max_depth=max_depth)

    cells = _cap_queue_at_max_depth([target_position] + path, counter_pos, down_vec, max_depth=max_depth)
    entrances = set(getattr(world, "entrances", []))
    while len(cells) > 2 and any(
        abs(cells[-1][0] - entrance[0]) + abs(cells[-1][1] - entrance[1]) == 1
        for entrance in entrances
    ):
        cells.pop()
    return cells


def _generate_queue_zigzag_variant(
    target_position: Tuple[int, int],
    world,
    direction: int,
    span: int,
    down_vec: Tuple[int, int] = (1, 0),
    right_vec: Tuple[int, int] = (0, 1),
) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    r, c = target_position

    if world.grid[r, c] not in [0, 2]:
        return cells
    cells.append((r, c))

    # Primary axis: away from the counter; secondary axis: lateral zigzag
    while True:
        for _ in range(span):
            nr = r + direction * right_vec[0]
            nc = c + direction * right_vec[1]
            if not _in_interior(world, (nr, nc)):
                return cells
            if world.grid[nr, nc] not in [0, 2]:
                return cells
            r, c = nr, nc
            cells.append((r, c))

        nr, nc = r + down_vec[0], c + down_vec[1]
        if not _in_interior(world, (nr, nc)):
            return cells
        if world.grid[nr, nc] not in [0, 2]:
            return cells
        r, c = nr, nc
        cells.append((r, c))
        direction *= -1


def _generate_queue_one_bend(
    target_position: Tuple[int, int],
    world,
    direction: int,
    span: int,
    down_vec: Tuple[int, int] = (1, 0),
    right_vec: Tuple[int, int] = (0, 1),
) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    r, c = target_position

    if world.grid[r, c] not in [0, 2]:
        return cells
    cells.append((r, c))

    # First bend laterally, then straight away from the counter
    for _ in range(span):
        nr = r + direction * right_vec[0]
        nc = c + direction * right_vec[1]
        if not _in_interior(world, (nr, nc)):
            return cells
        if world.grid[nr, nc] not in [0, 2]:
            return cells
        r, c = nr, nc
        cells.append((r, c))

    while True:
        nr, nc = r + down_vec[0], c + down_vec[1]
        if not _in_interior(world, (nr, nc)):
            return cells
        if world.grid[nr, nc] not in [0, 2]:
            return cells
        r, c = nr, nc
        cells.append((r, c))


def _generate_queue_staircase(
    target_position: Tuple[int, int],
    world,
    direction: int,
    num_steps: int,
    down_vec: Tuple[int, int] = (1, 0),
    right_vec: Tuple[int, int] = (0, 1),
) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    r, c = target_position

    if world.grid[r, c] not in [0, 2]:
        return cells
    cells.append((r, c))

    # Diagonal stair steps: one lateral cell, then one away cell, num_steps times
    for _ in range(num_steps):
        for dr, dc in (
            (direction * right_vec[0], direction * right_vec[1]),
            down_vec,
        ):
            nr, nc = r + dr, c + dc
            if not _in_interior(world, (nr, nc)):
                return cells
            if world.grid[nr, nc] not in [0, 2]:
                return cells
            r, c = nr, nc
            cells.append((r, c))

    # Then straight away from the counter
    while True:
        nr, nc = r + down_vec[0], c + down_vec[1]
        if not _in_interior(world, (nr, nc)):
            return cells
        if world.grid[nr, nc] not in [0, 2]:
            return cells
        r, c = nr, nc
        cells.append((r, c))


def _generate_queue_lshape(
    target_position: Tuple[int, int],
    world,
    direction: int,
    away_span: int,
    lateral_span: int,
    down_vec: Tuple[int, int] = (1, 0),
    right_vec: Tuple[int, int] = (0, 1),
) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    r, c = target_position

    if world.grid[r, c] not in [0, 2]:
        return cells
    cells.append((r, c))

    # First straight away from the counter, then the lateral leg, then away again
    steps = (
        [down_vec] * away_span
        + [(direction * right_vec[0], direction * right_vec[1])] * lateral_span
    )
    for dr, dc in steps:
        nr, nc = r + dr, c + dc
        if not _in_interior(world, (nr, nc)):
            return cells
        if world.grid[nr, nc] not in [0, 2]:
            return cells
        r, c = nr, nc
        cells.append((r, c))

    while True:
        nr, nc = r + down_vec[0], c + down_vec[1]
        if not _in_interior(world, (nr, nc)):
            return cells
        if world.grid[nr, nc] not in [0, 2]:
            return cells
        r, c = nr, nc
        cells.append((r, c))


def _generate_queue_square_variant(
    target_position: Tuple[int, int],
    world,
    offsets: Tuple[Tuple[int, int], ...],
    down_vec: Tuple[int, int] = (1, 0),
    right_vec: Tuple[int, int] = (0, 1),
) -> List[Tuple[int, int]]:
    target_r, target_c = target_position
    cells: List[Tuple[int, int]] = []

    for dd, dl in offsets:
        r = target_r + dd * down_vec[0] + dl * right_vec[0]
        c = target_c + dd * down_vec[1] + dl * right_vec[1]
        if not (0 <= r < world.height and 0 <= c < world.width):
            return cells
        if world.grid[r, c] not in [0, 2]:
            return cells
        cells.append((r, c))

    return cells


def _counter_pos_for(target_position: Tuple[int, int],
                     world,
                     down_vec: Tuple[int, int]) -> Tuple[int, int]:
    """Counter cell used as the depth-cap reference; falls back to the cell
    one step behind the queue head if the world has no counter zone."""
    counter_pos = get_counter_position(world)
    if counter_pos is not None:
        return counter_pos
    return (target_position[0] - down_vec[0], target_position[1] - down_vec[1])


def generate_queue_line(target_position: Tuple[int, int], world, norm: str) -> List[Tuple[int, int]]:
    _, down_vec, right_vec = get_queue_frame(world)
    counter_pos = _counter_pos_for(target_position, world, down_vec)
    cells: List[Tuple[int, int]]
    if norm == "straight_line":
        cells = generate_queue_straight(target_position, world)
    elif norm in ZIGZAG_LINE_VARIANTS:
        direction, span = ZIGZAG_LINE_VARIANTS[norm]
        cells = _generate_queue_zigzag_variant(
            target_position, world, direction, span, down_vec, right_vec,
        )
    elif norm in ONE_BEND_LINE_VARIANTS:
        direction, span = ONE_BEND_LINE_VARIANTS[norm]
        cells = _generate_queue_one_bend(
            target_position, world, direction, span, down_vec, right_vec,
        )
    elif norm in STAIRCASE_LINE_VARIANTS:
        direction, num_steps = STAIRCASE_LINE_VARIANTS[norm]
        cells = _generate_queue_staircase(
            target_position, world, direction, num_steps, down_vec, right_vec,
        )
    elif norm in LSHAPE_LINE_VARIANTS:
        direction, away_span, lateral_span = LSHAPE_LINE_VARIANTS[norm]
        cells = _generate_queue_lshape(
            target_position, world, direction, away_span, lateral_span,
            down_vec, right_vec,
        )
    elif norm in SQUARE_LINE_VARIANTS:
        cells = _generate_queue_square_variant(
            target_position,
            world,
            SQUARE_LINE_VARIANTS[norm],
            down_vec,
            right_vec,
        )
    elif norm in EDGE_LINE_VARIANTS:
        cells = _generate_queue_edge(
            target_position,
            world,
            EDGE_LINE_VARIANTS[norm],
            counter_pos,
            down_vec,
            right_vec,
        )
    else:
        cells = generate_queue_straight(target_position, world)
    return _cap_queue_at_max_depth(cells, counter_pos, down_vec)


def generate_queue_straight(target_position: Tuple[int, int],
                               world) -> List[Tuple[int, int]]:
    """Produce straight-line queue from the counter front extending away from it."""
    _, down_vec, _ = get_queue_frame(world)
    counter_pos = _counter_pos_for(target_position, world, down_vec)
    cells = []
    r, c = target_position
    while _in_interior(world, (r, c)):
        if world.grid[r, c] in [0, 2]:
            cells.append((r, c))
        r, c = r + down_vec[0], c + down_vec[1]
    return _cap_queue_at_max_depth(cells, counter_pos, down_vec)

def get_queue(candidate_cells: List[Tuple[int, int]],
              all_current_positions: Dict[str, Tuple[int, int]],
              previous_queue: Optional[Dict[int, str]] = None,
              eligible_agents: Optional[Set[str]] = None,
              ) -> Dict[int, str]:
    """
    Return the queue. Queue is defined as a line of positions starting from the target position.
    The format is a dictionary mapping from position index to agent_id.
    Only agents that are in candidate_cells (in the line) and are in the front of the line are included.

    If previous_queue is provided, empty slots that fall between agents who were in the
    previous queue are treated as moving gaps (the queue is advancing) and those trailing
    agents are still counted as part of the queue.

    If eligible_agents is provided, only those agent_ids are considered when occupying a
    queue cell. Agents that have already finished the first target (and are merely passing
    through a queue cell on their way elsewhere) are excluded so they are not mistaken for
    queue members.
    """
    queue = {}
    position_to_agent = {
        pos: agent_id
        for agent_id, pos in all_current_positions.items()
        if eligible_agents is None or agent_id in eligible_agents
    }
    # The head cell (service spot) is occupied by whoever physically stands on
    # it, even an agent that has just been served and is no longer eligible:
    # the next agent in line may only step up once the head has fully left.
    if candidate_cells:
        head = candidate_cells[0]
        if head not in position_to_agent:
            for agent_id, pos in all_current_positions.items():
                if pos == head:
                    position_to_agent[head] = agent_id
                    break
    has_previous_queue = bool(previous_queue)

    # Set of agents that were in the previous queue (ignoring None placeholders)
    prev_queue_agents: Set[str] = (
        {aid for aid in previous_queue.values() if aid is not None}
        if has_previous_queue else set()
    )

    # Iterate through candidate cells in order and build the queue
    position_index = 0
    none_count = 0  # at most one empty gap is allowed (queue advancing one step)
    for k, cell in enumerate(candidate_cells):
        if cell in position_to_agent:
            # Agent is at this position in the line
            agent_id = position_to_agent[cell]
            queue[position_index] = agent_id
            position_index += 1
        else:
            if none_count >= 1:
                # A single-step advance leaves at most one empty gap.  A second empty
                # cell means the queue is no longer continuous, so it ends here.
                break
            elif position_index == 0 and has_previous_queue:
                # Front of queue is empty but a previous queue existed — front person left
                queue[position_index] = None
                position_index += 1
                none_count += 1
            elif prev_queue_agents:
                # Check whether the immediately following candidate cell holds an agent
                # that was in the previous queue.  If so, this is a single moving gap (the
                # queue is advancing one step), so we insert a placeholder and continue.
                next_cell = candidate_cells[k + 1] if k + 1 < len(candidate_cells) else None
                gap_is_moving = (
                    next_cell is not None
                    and position_to_agent.get(next_cell) in prev_queue_agents
                )
                if gap_is_moving:
                    queue[position_index] = None  # moving gap — queue is still continuous
                    position_index += 1
                    none_count += 1
                else:
                    break  # True end of queue
            else:
                break  # Stop at the first empty position
    print("Detected queue:", queue)
    return queue


def get_next_queue_cell(queue_cells: List[Tuple[int, int]],
                        current_queue: Dict[int, Optional[str]]) -> Optional[Tuple[int, int]]:
    """The cell an agent that is not yet in the queue must step into: the first
    cell past the current end of the queue (the entry cell when no queue exists).

    Returns None if the queue already fills every queue cell.
    """
    if not queue_cells:
        return None

    queue_entry_cell = queue_cells[1] if len(queue_cells) > 1 else queue_cells[0]

    no_queue_detected = (
        len(current_queue) == 0
        or (len(current_queue) == 1 and current_queue.get(0) is None)
    )
    if no_queue_detected:
        return queue_entry_cell

    queue_end_index = max(
        (idx for idx in current_queue.keys() if idx < len(queue_cells)),
        default=None,
    )
    if queue_end_index is None or queue_end_index + 1 >= len(queue_cells):
        return None
    return queue_cells[queue_end_index + 1]


def check_queue_line(to_pos: Tuple[int, int],
                     all_current_positions: Dict[str, Tuple[int, int]], 
                     world, 
                     agent_id: str, 
                     target_position: Tuple[int, int], 
                     agent_classes,
                     norm: str = 'straight_line') -> bool:
    """
    Check if moving to to_pos violates the queue norm.
    Returns True if it violates, False otherwise.
    
    Logic:
    1. Check if previous queue exists
       - If NO: queue_end is at first position (queue_cells[0])
       - If YES: Check if front person left
         - If FRONT LEFT: 
           * Agents IN queue move up one position
           * Agents NOT in queue move toward queue end
         - If FRONT NOT LEFT:
           * Agents IN queue cannot move
           * Agents NOT in queue move toward queue end
    """
    agent_position = all_current_positions.get(agent_id)
    
    # Generate queue positions
    queue_cells = generate_queue_line(target_position, world, norm)
    if len(queue_cells) == 0:
        return False
    queue_entry_cell = queue_cells[1] if len(queue_cells) > 1 else queue_cells[0]
    
    
    
    #current_queue = get_queue(queue_cells, all_current_positions, has_previous_queue= len(world.previous_queue) != 0)
    current_queue = world.current_queue
    current_index = next((idx for idx, aid in current_queue.items() if aid == agent_id), None)

    # An agent may only step onto the counter (queue_cells[0]) from queue_cells[1].
    # Any move directly to the counter from elsewhere is a violation.
    if (to_pos == queue_cells[0] and to_pos != agent_position
            and agent_position != queue_entry_cell):
        return True

    # A departing agent (past their first target) must not re-enter any queue cell.
    agent_obj = next((a for a in getattr(world, 'agents', []) if a.agent_id == agent_id), None)
    if (agent_obj is not None and agent_obj.target_index > 0
            and to_pos in queue_cells and to_pos != agent_position):
        return True

    # If every queue cell is occupied (queue is full), the agent has nowhere to join —
    # allow free movement without restriction.
    occupied_positions = set(all_current_positions.values())
    if (
        current_index is None
        and all(cell in occupied_positions for cell in queue_cells)
    ):
        return False

    # Waiting (staying put) off the queue path but near the counter is a
    # violation: an agent that has not been served yet must join the line rather
    # than idle beside it. Only cells within COUNTER_WAIT_RADIUS of the counter
    # count as "near"; waiting further away (e.g. at the entrance) is fine.
    # (A full queue is handled above — then waiting anywhere is allowed.)
    if (
        current_index is None
        and agent_position is not None
        and to_pos == agent_position
        and agent_position not in queue_cells
        and (agent_obj is None or agent_obj.target_index == 0)
        and is_near_counter(agent_position, world)
    ):
        return True

    if len(current_queue) == 0 or (current_queue[0] == None and len(current_queue) == 1): # no queue detected
        return False
        if agent_position != queue_entry_cell:
            queue_end = queue_entry_cell
            return is_moving_toward_queue_end(agent_position, to_pos, queue_end)
        else:
            return is_moving_toward_queue_end(agent_position, to_pos, target_position)  # if agent is at the first position, they should move toward target
    
    else:

        if current_index is not None:
            # Agent IS in the queue.
            if current_index == 0:
                # Head of the line (being served): staying is fine; any move
                # off the service cell is the agent leaving, not a queue move.
                return to_pos != agent_position
            predecessor_left = current_queue.get(current_index - 1) is None
            if predecessor_left:
                # Predecessor's slot is empty: agent must advance exactly one
                # step forward — staying put here is also a violation.
                expected_pos = queue_cells[current_index - 1]
                return to_pos != expected_pos
            else:
                # Predecessor still present — agent must stay; any move violates.
                return to_pos != agent_position

        else:
            # Agent NOT in queue — use the global front signal to decide behaviour.
            front_left = current_queue[0] is None

            if front_left:
                # CASE A: Front person LEFT — queue is advancing.
                # Check that to_pos is not on queue (except the last position which will
                # be vacant after the queue ripples forward one step).
                queue_positions = {queue_cells[idx] for idx, aid in current_queue.items() if idx < len(queue_cells)}
                queue_end_index = max((idx for idx in current_queue.keys() if idx < len(queue_cells)), default=0)
                restricted_positions = queue_positions - {queue_cells[queue_end_index]}
                if to_pos in restricted_positions:
                    return True  # Violation: trying to move to position occupied by queue member

                return False
                queue_end = queue_cells[queue_end_index+1] if queue_end_index + 1 < len(queue_cells) else queue_cells[queue_end_index]
                return is_moving_toward_queue_end(agent_position, to_pos, queue_end)

            else:
                # CASE B: Front person did NOT leave — queue is stationary.
                queue_positions = {queue_cells[idx] for idx, aid in current_queue.items() if idx < len(queue_cells)}
                if to_pos in queue_positions:
                    return True  # Violation: trying to move to position occupied by queue member
                
                return False

                queue_end_index = max(current_queue.keys())
                next_index = queue_end_index + 1
                queue_end = queue_cells[next_index] if next_index < len(queue_cells) else queue_cells[queue_end_index]
                return is_moving_toward_queue_end(agent_position, to_pos, queue_end)


def is_moving_toward_queue_end(agent_position: Tuple[int, int],
                               to_pos: Tuple[int, int],
                               queue_end: Tuple[int, int]) -> bool:
    """
    Check if agent is moving toward the end of queue.
    
    Uses Manhattan distance to determine if agent is getting closer to queue_end.
    
    Args:
        agent_position: Current position of agent
        to_pos: Proposed new position for agent
        queue_end: Position the agent should move toward
    
    Returns:
        False if moving toward queue_end or staying (valid)
        True if moving away from queue_end (violation)
    """

    current_dist = abs(agent_position[0] - queue_end[0]) + abs(agent_position[1] - queue_end[1])
    to_pos_dist = abs(to_pos[0] - queue_end[0]) + abs(to_pos[1] - queue_end[1])

    # Moving away from queue violates norm
    if to_pos_dist >= current_dist:
        return True   # Violation: moved away
    
    return False      # Valid: moved toward or stayed


def check_straight_line(to_pos: Tuple[int, int],
                        all_current_positions: Dict[str, Tuple[int, int]],
                        world,
                        agent_id: str,
                        target_position: Tuple[int, int],
                        agent_classes) -> bool:
    queue_cells = generate_queue_straight(target_position, world)
    print("straight queue cells:", queue_cells)
    return check_queue_line(to_pos, all_current_positions, world, agent_id, target_position, agent_classes, norm='straight_line')


def check_line_norm(to_pos: Tuple[int, int],
                    all_current_positions: Dict[str, Tuple[int, int]],
                    world,
                    agent_id: str,
                    target_position: Tuple[int, int],
                    agent_classes,
                    norm: str) -> bool:
    queue_cells = generate_queue_line(target_position, world, norm)
    print(f"{norm} queue cells:", queue_cells)
    return check_queue_line(
        to_pos,
        all_current_positions,
        world,
        agent_id,
        target_position,
        agent_classes,
        norm=norm,
    )
