from typing import Dict, Optional, Tuple, Set, List

from src.world.base_world import CellType

# ---------------------------------------------------------------------------
# 1. Spacing norms — removed from the norm catalog.
#    The set is kept (empty) so the spacing code paths below stay inert.
# ---------------------------------------------------------------------------
SPACING_NORMS: Set[str] = set()

# ---------------------------------------------------------------------------
# 2. Movement norms (norm_017 – norm_026) — fully implemented
# ---------------------------------------------------------------------------
# Global-clock rhythm norms: the agent may not move on env steps whose residue
# modulo `modulus` is in `wait_residues`. Every agent is in sync, whatever step
# it entered on. (no_odd_turn / no_even_turn are the modulus-2 members.)
GLOBAL_CLOCK_NORMS: Dict[str, Tuple[int, Tuple[int, ...]]] = {
    'movement_no_odd_turn': (2, (1,)),        # norm_019
    'movement_no_even_turn': (2, (0,)),       # norm_020
    'movement_no_turn_mod3_0': (3, (0,)),     # norm_021
    'movement_no_turn_mod3_1': (3, (1,)),     # norm_022
    'movement_no_turn_mod3_2': (3, (2,)),     # norm_023
}

# Per-agent rhythm norms, counted on the agent's own move history rather than
# the env clock: (max consecutive moves, required rest turns after them). The
# agent must not move if it moved on `max_run` consecutive preceding turns
# (move2_wait1: never three moves in a row), or, when rest > 1, if it moved on
# any of the preceding `rest` turns. Idling longer than required is always
# allowed, and a move the collision handler forced on the agent does not count
# as its move.
AGENT_RHYTHM_NORMS: Dict[str, Tuple[int, int]] = {
    'movement_move1_wait1': (1, 1),           # norm_024
    'movement_move2_wait1': (2, 1),           # norm_025
    'movement_move3_wait1': (3, 1),           # norm_026
}

MOVEMENT_NORMS = {
    'movement_clockwise_move',       # norm_017
    'movement_counterclockwise_move',# norm_018
    *GLOBAL_CLOCK_NORMS,
    *AGENT_RHYTHM_NORMS,
}

# ---------------------------------------------------------------------------
# 3. X-cell norms (norm_027 – norm_028) — fully implemented
# ---------------------------------------------------------------------------
XCELL_NORMS = {
    'xcell_must_visit',              # norm_027
    'xcell_forbidden',               # norm_028
}

NAVIGATION_NORMS = SPACING_NORMS | MOVEMENT_NORMS | XCELL_NORMS



# ---------------------------------------------------------------------------
# Predicate helpers
# ---------------------------------------------------------------------------

def is_navigation_norm(norm: Optional[str]) -> bool:
    """Return True if norm belongs to any navigation subcategory."""
    return norm in NAVIGATION_NORMS


def is_spacing_norm(norm: Optional[str]) -> bool:
    """Return True if norm is a spacing navigation norm (none remain in the catalog)."""
    return norm in SPACING_NORMS


def is_movement_norm(norm: Optional[str]) -> bool:
    """Return True if norm is a movement navigation norm (norm_017–026)."""
    return norm in MOVEMENT_NORMS


def is_global_clock_norm(norm: Optional[str]) -> bool:
    """Return True if norm restricts movement by the env step's residue."""
    return norm in GLOBAL_CLOCK_NORMS


def is_agent_rhythm_norm(norm: Optional[str]) -> bool:
    """Return True if norm restricts movement by the agent's own move history."""
    return norm in AGENT_RHYTHM_NORMS


def is_xcell_norm(norm: Optional[str]) -> bool:
    """Return True if norm is an X-cell navigation norm (norm_027–028)."""
    return norm in XCELL_NORMS


def xcell_requires_visit(norm: Optional[str]) -> bool:
    """Return True if the norm requires agents to visit X cells before their goal."""
    return norm == 'xcell_must_visit'


def xcell_is_forbidden(norm: Optional[str]) -> bool:
    """Return True if X cells are completely forbidden under this norm."""
    return norm == 'xcell_forbidden'


def xcell_blocked_cells(norm: Optional[str], world) -> set:
    """The cells a path planner must route around entirely under `norm`, empty
    for every norm but xcell_forbidden.

    The step-by-step norm check only rules out the move an agent is about to
    make, which is not enough here: an X cell forbidden on the immediate step is
    just as forbidden three steps out, so a planner that only checks the first
    move plans straight through the X cells, refuses the last step onto one, and
    replans the same route — the agent oscillates at the edge instead of
    detouring. Handing the whole set to the search makes it plan the detour up
    front. See compute_a_star_path in src/agents/planner.py.
    """
    if not xcell_is_forbidden(norm):
        return set()
    cells = {tuple(c) for c in world.zones.get('affordance_x', [])}
    return cells | {tuple(c) for c in getattr(world, 'x_overlay', set())}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Spacing variant checker (norms removed from the catalog; kept for reference)
# ---------------------------------------------------------------------------

def _is_diagonal(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Return True when a and b differ by exactly 1 in both row and column."""
    return abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1


def _are_agents_moving_towards_each_other(
    agent_from: Tuple[int, int],
    agent_to: Tuple[int, int],
    other_from: Tuple[int, int],
    other_to: Tuple[int, int],
) -> bool:
    """
    Check if two agents are moving towards each other.
    
    Returns True if the Manhattan distance between them is decreasing
    (i.e., distance after movement < distance before movement).
    This means the agents are not moving from different directions away
    from each other.
    """
    current_distance = _manhattan(agent_from, other_from)
    next_distance = _manhattan(agent_to, other_to)
    
    return next_distance < current_distance


def _are_agents_moving_in_same_direction(
    agent_from: Tuple[int, int],
    agent_to: Tuple[int, int],
    other_from: Tuple[int, int],
    other_to: Tuple[int, int],
) -> bool:
    """
    Check if two agents are moving in the same direction.
    
    Computes direction vectors for both agents and returns True if they are identical.
    A direction vector is computed as the normalized (sign-based) difference.
    
    Returns True if both agents have the same direction of movement.
    """
    def get_direction(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Get normalized direction vector."""
        dr = 1 if to_pos[0] > from_pos[0] else (-1 if to_pos[0] < from_pos[0] else 0)
        dc = 1 if to_pos[1] > from_pos[1] else (-1 if to_pos[1] < from_pos[1] else 0)
        return (dr, dc)
    
    agent_direction = get_direction(agent_from, agent_to)
    other_direction = get_direction(other_from, other_to)
    
    return agent_direction == other_direction


def _check_spacing_variant(
    norm: str,
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int],
    all_current_positions: Dict[str, Tuple[int, int]],
    agent_id: str,
    agent_classes: Dict[str, int],
    world,
) -> bool:
    """
    Check whether moving from_pos → to_pos violates a spacing norm.

    Both spacing norms were removed from the norm catalog, so SPACING_NORMS is
    empty and this checker is never reached; it is kept for reference.

    Other agents' positions are taken from all_current_positions (predicted
    next positions), falling back to their real current positions when not
    listed.

    spacing_no_adjacent:
        Violation if to_pos is cardinally adjacent (Manhattan == 1) to any
        other agent.

    spacing_no_diagonal:
        Violation if to_pos is diagonally adjacent (|Δrow| == 1 and
        |Δcol| == 1) to any other agent.

    Returns True (violation) when the rule is broken.
    """
    # Build real current positions from world.agents for fall-back lookups.
    real_positions: Dict[str, Tuple[int, int]] = {
        a.agent_id: a.current_position for a in getattr(world, 'agents', [])
    }

    for other_id, other_predicted_pos in all_current_positions.items():
        if other_id == agent_id:
            continue

        other_pos = other_predicted_pos if other_predicted_pos is not None else real_positions.get(other_id)
        if other_pos is None:
            continue

        # Get the other agent's real current position for direction check
        other_real_pos = real_positions.get(other_id)
        if other_real_pos is None:
            other_real_pos = other_pos

        dist = _manhattan(to_pos, other_pos)

        # spacing_no_adjacent: no cardinal adjacency
        if norm == 'spacing_no_adjacent':
            if dist == 1:
                print(
                    f"Spacing Norm Violated ({norm}): agent {agent_id} at {to_pos} "
                    f"would be adjacent to {other_id} at {other_pos}."
                )
                return True
            if to_pos == other_pos:
                print(
                    f"Spacing Norm Violated ({norm}): agent {agent_id} at {to_pos} "
                    f"would occupy the same cell as {other_id}."
                )
                return True

        # spacing_no_diagonal: no diagonal adjacency
        elif norm == 'spacing_no_diagonal':
            if _is_diagonal(to_pos, other_pos):
                print(
                    f"Spacing Norm Violated ({norm}): agent {agent_id} at {to_pos} "
                    f"would be diagonally adjacent to {other_id} at {other_pos}."
                )
                return True

    return False


# ---------------------------------------------------------------------------
# Movement variant checker (norms 014–017)
# ---------------------------------------------------------------------------

def _compute_outer_loop(world) -> List[Tuple[int, int]]:
    """
    Compute the outermost circular walkable path in the world, excluding dead-ends.

    Algorithm:
    1. Start from the set of all walkable cells.
    2. Iteratively prune dead-ends: remove any cell with fewer than 2 walkable
       neighbors in the remaining set (i.e. it cannot be part of a loop).
    3. From the remaining loop cells, find the one closest to the grid boundary
       (the true outermost edge cell) and do a clockwise BFS/walk to order
       the loop as an ordered list.

    The result is cached on the world object as `_outer_loop`.
    """
    if hasattr(world, '_outer_loop') and world._outer_loop is not None:
        return world._outer_loop

    walkable: Set[Tuple[int, int]] = set(world.walkable_cells)

    # Prune dead-ends iteratively until stable
    changed = True
    while changed:
        changed = False
        to_remove = []
        for cell in walkable:
            r, c = cell
            neighbors_in_loop = sum(
                1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                if (r+dr, c+dc) in walkable
            )
            if neighbors_in_loop < 2:
                to_remove.append(cell)
        if to_remove:
            for cell in to_remove:
                walkable.discard(cell)
            changed = True

    if not walkable:
        world._outer_loop = []
        return []

    # Pick a start cell: prefer outermost (smallest row, then smallest col)
    start = min(walkable, key=lambda p: (p[0], p[1]))

    # Walk the loop clockwise: at each step prefer right > down > left > up
    # (clockwise priority in screen coords)
    clockwise_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

    loop: List[Tuple[int, int]] = [start]
    prev = None
    current = start

    # Determine the first direction: from start go right if possible, else down, etc.
    first_dir = None
    for dr, dc in clockwise_dirs:
        nxt = (current[0] + dr, current[1] + dc)
        if nxt in walkable:
            first_dir = (dr, dc)
            break

    # Walk using right-hand rule (keep the wall on the right)
    # Encode directions as indices into clockwise_dirs
    dir_idx = clockwise_dirs.index(first_dir)
    prev = current
    current = (current[0] + first_dir[0], current[1] + first_dir[1])
    loop.append(current)

    max_steps = len(walkable) + 1
    steps = 0
    while current != start and steps < max_steps:
        steps += 1
        # Try: turn right first, then straight, then left, then back
        for turn in [-1, 0, 1, 2]:  # right, straight, left, back
            next_dir_idx = (dir_idx + turn) % 4
            dr, dc = clockwise_dirs[next_dir_idx]
            nxt = (current[0] + dr, current[1] + dc)
            if nxt in walkable and nxt != prev:
                prev = current
                current = nxt
                dir_idx = next_dir_idx
                if current != start:
                    loop.append(current)
                break
        else:
            break  # Stuck

    print(f"[Outer Loop] Circular path ({len(loop)} cells): {loop}")

    # Print entrance cells adjacent to the loop as door entry points
    loop_set_final = set(loop)
    if hasattr(world, 'zones'):
        entrance_cells = world.zones.get('entrance', [])
        entry_points = []
        for ec in entrance_cells:
            if ec not in loop_set_final:
                r, c = ec
                adj_loop = [
                    (r + dr, c + dc)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if (r + dr, c + dc) in loop_set_final
                ]
                if adj_loop:
                    entry_points.append((ec, adj_loop[0]))
        if entry_points:
            for ep, lp in entry_points:
                print(f"[Outer Loop] Door entry point: {ep} → enters loop at {lp}")

    world._outer_loop = loop
    return loop


def _get_outer_loop_set(world) -> Set[Tuple[int, int]]:
    """Return the outer loop as a set for O(1) membership checks."""
    if hasattr(world, '_outer_loop_set') and world._outer_loop_set is not None:
        return world._outer_loop_set
    loop = _compute_outer_loop(world)
    world._outer_loop_set = set(loop)
    return world._outer_loop_set


def _clockwise_next(pos: Tuple[int, int], world) -> Optional[Tuple[int, int]]:
    """Return the next cell in clockwise order on the outer loop, or None."""
    loop = _compute_outer_loop(world)
    if not loop or pos not in loop:
        return None
    idx = loop.index(pos)
    return loop[(idx + 1) % len(loop)]


def _is_clockwise_movement(
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int],
    world,
) -> bool:
    """
    Check if the movement from_pos → to_pos is allowed under a clockwise constraint.

    Rules:
    1. The agent must be on the outermost circular path (dead-end-free loop).
    2. If from_pos IS on the loop, to_pos must be the immediately next cell in
       the clockwise direction on that loop.
    3. Exception for door/entrance cells: if from_pos is an entrance cell
       adjacent to the loop, the agent may move onto the loop (step to the
       adjacent loop cell) to join the circular path.

    Note: the final step onto the agent's target cell is already permitted
    unconditionally in _check_movement_variant before this function is called.
    """
    if from_pos == to_pos:
        return True  # WAIT is always fine

    loop_set = _get_outer_loop_set(world)
    loop = _compute_outer_loop(world)

    if from_pos not in loop_set:
        # Allow entrance/door cells to step onto the loop
        # (agent is joining the circular path from outside)
        if to_pos in loop_set and _manhattan(from_pos, to_pos) == 1:
            return True
        return False

    # On the loop: to_pos must be the next clockwise cell
    if not loop:
        return True
    idx = loop.index(from_pos)
    next_cw = loop[(idx + 1) % len(loop)]
    return to_pos == next_cw


def _is_counterclockwise_movement(
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int],
    world,
) -> bool:
    """
    Check if the movement from_pos → to_pos is allowed under a counterclockwise
    constraint (mirror of _is_clockwise_movement).

    Rules:
    1. The agent must be on the outermost circular path (dead-end-free loop).
    2. If from_pos IS on the loop, to_pos must be the immediately next cell in
       the counterclockwise direction on that loop.
    3. Exception for door/entrance cells: if from_pos is an entrance cell
       adjacent to the loop, the agent may move onto the loop (step to the
       adjacent loop cell) to join the circular path.

    Note: the final step onto the agent's target cell is already permitted
    unconditionally in _check_movement_variant before this function is called.
    """
    if from_pos == to_pos:
        return True

    loop_set = _get_outer_loop_set(world)
    loop = _compute_outer_loop(world)

    if from_pos not in loop_set:
        # Allow entrance/door cells to step onto the loop
        if to_pos in loop_set and _manhattan(from_pos, to_pos) == 1:
            return True
        return False

    # On the loop: to_pos must be the next counterclockwise cell
    if not loop:
        return True
    idx = loop.index(from_pos)
    next_ccw = loop[(idx - 1) % len(loop)]
    return to_pos == next_ccw


def _is_blocked_in_direction(
    from_pos: Tuple[int, int],
    direction: Tuple[int, int],
    agent_id: str,
    world,
) -> bool:
    """
    Check if the agent at from_pos is blocked from continuing in the given direction.

    Blocked means the next cell in that direction is:
    - out of grid bounds, OR
    - not walkable (wall / obstacle), OR
    - currently occupied by another agent.
    """
    dr, dc = direction
    next_pos = (from_pos[0] + dr, from_pos[1] + dc)

    # Out of bounds
    if not (0 <= next_pos[0] < world.height and 0 <= next_pos[1] < world.width):
        return True

    # Not walkable
    if not world.is_walkable(next_pos):
        return True

    # Occupied by another agent (use their current positions)
    current_positions = {
        a.agent_id: a.current_position
        for a in getattr(world, 'agents', [])
        if a.agent_id != agent_id
    }
    if next_pos in current_positions.values():
        return True

    return False



def _rhythm_requires_rest(norm: str, agent_id: str, world) -> bool:
    """True if the per-agent rhythm norm forbids a move on the current turn.

    Reads agent.move_steps — the env steps on which the agent moved by its own
    choice (maintained by NormEnv after each step) — and agent.current_step.
    """
    max_run, rest = AGENT_RHYTHM_NORMS[norm]
    agent = next((a for a in getattr(world, 'agents', []) if a.agent_id == agent_id), None)
    if agent is None:
        return False
    current_turn = getattr(agent, 'current_step', 0)
    move_steps = getattr(agent, 'move_steps', set())
    if rest > 1:
        # Must rest `rest` turns after each move: any move in the last `rest` turns forbids moving.
        return any((current_turn - k) in move_steps for k in range(1, rest + 1))
    # Must rest after `max_run` consecutive moves.
    return all((current_turn - k) in move_steps for k in range(1, max_run + 1))


def _check_movement_variant(
    norm: str,
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int],
    all_current_positions: Dict[str, Tuple[int, int]],
    agent_id: str,
    agent_classes: Dict[str, int],
    world,
    target_position: Optional[Tuple[int, int]] = None,
) -> bool:
    """
    Check whether moving from_pos → to_pos violates one of the four
    movement norms (014–017).

    norm_017  movement_clockwise_move:
        Agent must move in a clockwise direction around the environment center.
        Violation if the move is not clockwise.

    norm_018  movement_counterclockwise_move:
        Agent must move in a counterclockwise direction around the environment center.
        Violation if the move is not counterclockwise.

    norm_019  movement_no_odd_turn:
        Agent may only move on even-numbered turns (2, 4, 6, ...).
        Violation if trying to move on an odd turn (1, 3, 5, ...).

    norm_020  movement_no_even_turn:
        Agent may only move on odd-numbered turns (1, 3, 5, ...).
        Violation if trying to move on an even turn (2, 4, 6, ...).

    norm_021–035  movement_no_turn_mod3_{0,1,2}:
        Agent may not move on turns whose number mod 3 equals 0 / 1 / 2.

    norm_024–038  movement_move1_wait1 / move2_wait1 / move3_wait1:
        Per-agent rhythm on the agent's own move history: never two / three /
        four moves in a row.

    Returns True (violation) when the rule is broken.
    """
    # Check for WAIT action (no movement) - not a violation for movement norms
    # since the agent is not trying to move
    if from_pos == to_pos:
        return False

    # Rhythm norms are checked before the target-position exception: the turn
    # restriction applies even when the destination is the agent's goal cell.
    if norm in GLOBAL_CLOCK_NORMS:
        # norm_019/020/021–023: may only move on env steps outside the wait residues.
        modulus, wait_residues = GLOBAL_CLOCK_NORMS[norm]
        agent = next((a for a in getattr(world, 'agents', []) if a.agent_id == agent_id), None)
        current_turn = agent.current_step if agent else 0
        if current_turn > 0 and current_turn % modulus in wait_residues:
            print(
                f"Movement Norm Violated ({norm}): agent {agent_id} "
                f"cannot move on turn {current_turn} (turn mod {modulus} = {current_turn % modulus})."
            )
            return True
        return False

    if norm in AGENT_RHYTHM_NORMS:
        # norm_024–038: may not move if it moved on the preceding turns the
        # rhythm says it must rest after (see AGENT_RHYTHM_NORMS).
        if _rhythm_requires_rest(norm, agent_id, world):
            agent = next((a for a in getattr(world, 'agents', []) if a.agent_id == agent_id), None)
            current_turn = agent.current_step if agent else 0
            print(
                f"Movement Norm Violated ({norm}): agent {agent_id} "
                f"must rest on turn {current_turn} after its recent moves."
            )
            return True
        return False

    # Never block the final step directly onto the goal cell for path/direction norms
    if target_position is not None and to_pos == target_position:
        return False

    # norm_017: must move clockwise
    if norm == 'movement_clockwise_move':
        if not _is_clockwise_movement(from_pos, to_pos, world):
            print(
                f"Movement Norm Violated ({norm}): agent {agent_id} at {from_pos} "
                f"moving to {to_pos} is not clockwise around the environment center."
            )
            return True

    # norm_018: must move counterclockwise
    elif norm == 'movement_counterclockwise_move':
        if not _is_counterclockwise_movement(from_pos, to_pos, world):
            print(
                f"Movement Norm Violated ({norm}): agent {agent_id} at {from_pos} "
                f"moving to {to_pos} is not counterclockwise around the environment center."
            )
            return True

    return False


# ---------------------------------------------------------------------------
# X-cell variant checker (norms 019–020)
# ---------------------------------------------------------------------------

def _check_xcell_variant(
    norm: str,
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int],
    world,
    agent_id: str,
    target_position: Optional[Tuple[int, int]] = None,
) -> bool:
    """
    Check whether moving from_pos → to_pos violates an X-cell norm (019–020).

    norm_027  xcell_must_visit:
        Violation if to_pos is the agent's goal and no X cell has been visited yet.

    norm_028  xcell_forbidden:
        Violation if to_pos is an X-marked cell.

    Returns True (violation) when the rule is broken.
    """
    r, c = to_pos
    if not (0 <= r < world.height and 0 <= c < world.width):
        return False

    x_overlay: set = getattr(world, 'x_overlay', set())
    is_x_cell = (world.grid[r, c] == CellType.AFFORDANCE_X.value) or (to_pos in x_overlay)

    if norm == 'xcell_forbidden':
        if is_x_cell:
            print(
                f"X-Cell Norm Violated ({norm}): agent {agent_id} "
                f"attempted to step on forbidden X cell at {to_pos}."
            )
            return True

    elif norm == 'xcell_must_visit':
        if target_position is not None and to_pos == target_position:
            agent = next((a for a in getattr(world, 'agents', []) if a.agent_id == agent_id), None)
            if agent is not None and getattr(agent, 'target_index', 0) == 0:
                x_cells = world.zones.get('affordance_x', [])
                if x_cells:
                    visited = getattr(agent, 'visited_x_cells', set())
                    if not visited:
                        print(
                            f"X-Cell Norm Violated ({norm}): agent {agent_id} "
                            f"reached target {to_pos} without visiting any X cell."
                        )
                        return True

    return False


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def check_navigation_norm(
    norm: str,
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int],
    all_current_positions: Dict[str, Tuple[int, int]],
    world,
    agent_id: str,
    agent_classes: Dict[str, int],
    target_position: Optional[Tuple[int, int]] = None,
) -> bool:
    """
    Check whether moving from_pos → to_pos violates a navigation norm.

    Returns True if the move is a violation, False otherwise.

    Currently implemented subcategories
    ------------------------------------
    Spacing       : removed from the catalog (SPACING_NORMS is empty).
    Movement      (norm_017–020): fully implemented.
    X-cell        (norm_027–028): fully implemented.
    """
    if is_spacing_norm(norm):
        return _check_spacing_variant(
            norm,
            from_pos,
            to_pos,
            all_current_positions,
            agent_id,
            agent_classes,
            world,
        )

    if is_movement_norm(norm):
        return _check_movement_variant(
            norm,
            from_pos,
            to_pos,
            all_current_positions,
            agent_id,
            agent_classes,
            world,
            target_position=target_position,
        )

    if is_xcell_norm(norm):
        return _check_xcell_variant(norm, from_pos, to_pos, world, agent_id, target_position=target_position)

    return False
