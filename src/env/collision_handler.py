"""
Collision detection and resolution for multi-agent simulation.
Handles position conflicts when multiple agents occupy the same cell.
"""
import contextlib
import io
import random
from typing import Any, Callable, List, Dict, Tuple, Set
from src.env.actions import Action
from src.agents.base_agent import Agent
from src.norms.lining import is_line_norm
from src.norms.navigation import is_navigation_norm, check_navigation_norm
from src.norms.norm_checker import check_norm_violation
from src.norms.occupation import is_occupation_region_norm, check_occupation_norm

def detect_collisions(agents) -> Dict[Tuple[int, int], List[Agent]]:
    """
    Detect positions where multiple agents are located.
    
    Returns:
        {pos: agents_list } 
        Only includes positions with multiple agents (collisions)
    """
    # Group agents by position (exclude done agents)
    position_to_agents: Dict[Tuple[int, int], List[Agent]] = {}
    
    for agent in agents:
        if not agent.active or agent.done:
            continue
        
        pos = agent.current_position
        if pos not in position_to_agents:
            position_to_agents[pos] = []
        position_to_agents[pos].append(agent)
    
    # Filter to only collisions (multiple agents at same position)
    collisions = {pos: agents_list for pos, agents_list in position_to_agents.items() 
                 if len(agents_list) > 1}
    
    return collisions


def resolve_collisions(agents, world, collisions) -> Dict[str, Action]:
    """
    Resolve collisions (neutral):

    Any agent class can move into a cell occupied by any other class.
    Two cases:
      1. One or more movers arrive at the spot of a waiting agent:
         movers get PUSH (stay), waiters are displaced to a neighbour.
      2. Two or more movers compete for the same empty spot:
         one is chosen randomly to stay (PUSH); the rest are displaced.

    When no neighbour of the collision cell is free, chain displacement kicks
    in: a non-colliding oracle occupant of a neighbouring cell is pushed to one
    of its own free neighbours (recursively, up to _MAX_CHAIN_DEPTH agents per
    chain) so the displaced agent can still leave the collision cell instead of
    staying overlapped and re-triggering the same collision every step.

    Returns:
        Dict mapping agent_id to Action for all agents involved in collisions
    """
    _MOVE_ACTIONS = {Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT}

    agent_actions: Dict[str, Action] = {}
    agent_classes = {a.agent_id: a.agent_class for a in agents}

    # Build occupied set (all active, non-done agents)
    occupied = {agent.current_position for agent in agents
                if agent.active and not agent.done}

    # Non-colliding active agents by cell — chain-displacement candidates.
    colliding_ids = {a.agent_id for agents_list in collisions.values()
                     for a in agents_list}
    pos_to_agent = {a.current_position: a for a in agents
                    if a.active and not a.done and a.agent_id not in colliding_ids}

    for pos, conflicting_agents in collisions.items():
        movers  = [a for a in conflicting_agents if a.action in _MOVE_ACTIONS]
        waiters = [a for a in conflicting_agents if a not in movers]

        if movers and waiters:
            # Case 1: one mover stays (random), all other agents are displaced
            winner    = random.choice(movers)
            stayers   = [winner]
            displaced = [a for a in conflicting_agents if a is not winner]
        elif len(movers) > 1:
            # Case 2: multiple movers — random winner
            winner    = random.choice(movers)
            stayers   = [winner]
            displaced = [a for a in movers if a is not winner]
        elif movers:
            # Single mover — it stays
            stayers   = movers
            displaced = [a for a in conflicting_agents if a not in stayers]
        else:
            # All waiters (e.g. an agent spawned onto an occupied entrance):
            # the earliest arrival keeps the cell, the rest are displaced —
            # otherwise everyone "stays" and the stack never separates.
            winner    = min(conflicting_agents,
                            key=lambda a: (a.arrival_time, a.agent_id))
            stayers   = [winner]
            displaced = [a for a in conflicting_agents if a is not winner]

        for agent in stayers:
            agent_actions[agent.agent_id] = Action.PUSH

        # Displace pushed agents to nieghbour positions
        for agent in displaced:
            old_pos = agent.current_position
            neighbors = world.get_neighbors(pos)

            # Prefer downward neighbours (higher row number)
            current_row = pos[0]
            upward_neighbors = [n for n in neighbors if n[0] < current_row]
            other_neighbors  = [n for n in neighbors if n[0] >= current_row]
            prioritized = other_neighbors + upward_neighbors

            # Prefer free cells that don't break the displaced agent's own norm
            # (mirrors resolve_swap_pairs); fall back to any free cell.
            free_neighbors = [n for n in prioritized if n not in occupied]
            preferred = [
                n for n in free_neighbors
                if not redirect_violates_norm(agent, n, agent_classes, agents, world)
            ]
            candidates = preferred if preferred else free_neighbors

            if not candidates:
                # No free neighbour: chain displacement — push a non-colliding
                # oracle occupant of a neighbouring cell out of the way so this
                # agent can still leave the collision cell.
                for n in prioritized:
                    occupant = pos_to_agent.get(n)
                    if (occupant is None or not occupant.is_oracle
                            or occupant.agent_id in agent_actions):
                        continue
                    if _push_occupant_chain(
                            occupant, occupied, pos_to_agent, agent_actions,
                            agent_classes, agents, world,
                            depth=_MAX_CHAIN_DEPTH,
                            chain_ids={agent.agent_id, occupant.agent_id}):
                        candidates = [n]
                        break

            if candidates:
                neighbor = candidates[0]
                occupied.add(neighbor)
                action_str  = _get_direction_action(old_pos, neighbor)
                action_enum = getattr(Action, action_str.upper(), Action.WAIT)
                agent_actions[agent.agent_id] = action_enum
            else:
                agent_actions[agent.agent_id] = Action.WAIT

    return agent_actions


# Max number of bystander agents that may be pushed in one displacement chain.
_MAX_CHAIN_DEPTH = 3


def _push_occupant_chain(occupant, occupied, pos_to_agent, agent_actions,
                         agent_classes, agents, world, depth, chain_ids) -> bool:
    """
    Force ``occupant`` (a non-colliding oracle agent) to vacate its cell so a
    collision-displaced agent can take it. Prefer a free neighbour (one that
    doesn't break the occupant's own norm, mirroring the main displacement
    loop); when none exists, recursively push the next non-colliding oracle
    occupant, up to ``depth`` agents per chain. Never pushes human/LLM agents.

    On success the forced move is written into ``agent_actions`` — so the
    pushed agent is treated exactly like a collision-displaced agent (exempt
    from norm checks and penalties for this move) — and True is returned.
    """
    old_pos = occupant.current_position
    neighbors = world.get_neighbors(old_pos)

    free_neighbors = [n for n in neighbors if n not in occupied]
    preferred = [
        n for n in free_neighbors
        if not redirect_violates_norm(occupant, n, agent_classes, agents, world)
    ]
    candidates = preferred if preferred else free_neighbors

    new_pos = None
    if candidates:
        new_pos = candidates[0]
    elif depth > 1:
        for n in neighbors:
            nxt = pos_to_agent.get(n)
            if (nxt is None or not nxt.is_oracle
                    or nxt.agent_id in chain_ids
                    or nxt.agent_id in agent_actions):
                continue
            if _push_occupant_chain(nxt, occupied, pos_to_agent, agent_actions,
                                    agent_classes, agents, world,
                                    depth - 1, chain_ids | {nxt.agent_id}):
                new_pos = n
                break

    if new_pos is None:
        return False

    occupied.add(new_pos)
    action_str = _get_direction_action(old_pos, new_pos)
    agent_actions[occupant.agent_id] = getattr(Action, action_str.upper(), Action.WAIT)
    print(f"Chain displaced: {occupant.agent_id} pushed {old_pos} -> {new_pos}")
    return True


def detect_swaps(agents) -> List[Tuple[Agent, Agent]]:
    """
    Detect pairs of active agents that are about to swap positions
    (agent A moves to B's current cell while B moves to A's current cell).

    Returns:
        List of (agent_a, agent_b) tuples (each pair appears once).
    """
    _MOVE_ACTIONS = {Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT}

    active = [
        a for a in agents
        if a.active and not a.done and a.action in _MOVE_ACTIONS and a.next_position is not None
    ]

    pos_to_agent = {a.current_position: a for a in active}

    swap_pairs: List[Tuple[Agent, Agent]] = []
    seen: set = set()

    for agent in active:
        other = pos_to_agent.get(agent.next_position)
        if other is None or other is agent:
            continue
        if other.next_position == agent.current_position:
            key = tuple(sorted((id(agent), id(other))))
            if key not in seen:
                seen.add(key)
                swap_pairs.append((agent, other))

    return swap_pairs


def resolve_swap_pairs(
    agents,
    world,
    step_data: Dict[str, Any],
    is_queried_fn: Callable[[str], bool],
) -> None:
    """
    Detect agents about to swap positions and redirect one to avoid the exchange.

    For each swap pair, a non-exempt agent (oracle, not queried) is chosen
    randomly and redirected to a free neighbour. Falls back to WAIT if none is
    available. No collision penalty is applied.

    Iterates until no swaps remain, because redirecting one agent may create a
    new swap with a third agent (e.g. the redirected position is the current
    position of another agent that is itself moving away).

    Args:
        agents: Full list of agents in the environment.
        world: The environment world object.
        step_data: Step data containing committed agent actions.
        is_queried_fn: Callable(agent_id) -> bool — whether the agent is the
            queried agent (exempt from redirection).
    """
    agent_classes = {a.agent_id: a.agent_class for a in agents}
    max_iterations = len(agents) + 1
    for _ in range(max_iterations):
        swap_pairs = detect_swaps(agents)
        if not swap_pairs:
            break

        for agent_a, agent_b in swap_pairs:
            exempt_a = not agent_a.is_oracle or is_queried_fn(agent_a.agent_id)
            exempt_b = not agent_b.is_oracle or is_queried_fn(agent_b.agent_id)
            candidates = [a for a, exempt in ((agent_a, exempt_a), (agent_b, exempt_b)) if not exempt]
            if not candidates:
                continue  # both exempt — let the swap happen

            # Prefer the candidate that is further from its target (lower priority to move).
            # Fall back to random if targets are unavailable.
            def _dist_to_target(a):
                if a.current_target is None:
                    return 0
                return abs(a.current_position[0] - a.current_target[0]) + abs(a.current_position[1] - a.current_target[1])

            redirected = max(candidates, key=_dist_to_target) if len(candidates) > 1 else candidates[0]
            old_next = redirected.next_position

            # Collect all committed next positions except the redirected agent's own.
            # Also include current positions of agents that are moving, to avoid
            # redirecting into a position that would create a new swap.
            committed = {
                data['next_position']
                for aid, data in step_data['actions'].items()
                if aid != redirected.agent_id
            }
            # A redirect to `nb` would create a new swap only if an agent
            # currently at `nb` is heading back to `redirected.current_position`.
            would_create_swap = {
                a.current_position
                for a in agents
                if a.agent_id != redirected.agent_id
                and a.next_position == redirected.current_position
            }

            # Collect valid neighbours (exclude swap target, positions that
            # would immediately create a new swap, and positions that would
            # break the redirected agent's own norm — spacing/occupation).
            valid_neighbors = [
                nb for nb in world.get_neighbors(redirected.current_position)
                if nb != old_next
                and nb not in committed
                and nb not in would_create_swap
                and not redirect_violates_norm(redirected, nb, agent_classes, agents, world)
            ]

            # Pick the neighbour closest to the agent's target position;
            # fall back to WAIT (current position) if none are available.
            if valid_neighbors and redirected.current_target is not None:
                target = redirected.current_target
                valid_neighbors.sort(
                    key=lambda nb: abs(nb[0] - target[0]) + abs(nb[1] - target[1])
                )
                new_pos = valid_neighbors[0]
            elif valid_neighbors:
                new_pos = valid_neighbors[0]
            else:
                new_pos = redirected.current_position  # fall back to WAIT

            new_action = getattr(
                Action,
                _get_direction_action(redirected.current_position, new_pos).upper(),
                Action.WAIT,
            )
            redirected.action = new_action
            redirected.next_position = new_pos
            step_data['actions'][redirected.agent_id]['action'] = new_action
            step_data['actions'][redirected.agent_id]['next_position'] = new_pos
            print(f"Swap redirected: {redirected.agent_id} from {old_next} -> {new_pos}")


@contextlib.contextmanager
def _quiet():
    """Swallow the norm checkers' violation banners.

    redirect_violates_norm asks "*would* this candidate cell violate the norm?"
    for every neighbour it considers, and the checkers print a violation banner
    whenever the answer is yes. Those moves are rejected, never taken, so
    letting the banners through would fill the logs with violations that were in
    fact avoided — and make a log grep overcount real ones.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def redirect_violates_norm(
    agent,
    pos: Tuple[int, int],
    agent_classes: Dict[str, int],
    agents,
    world,
) -> bool:
    """
    Return True if redirecting ``agent`` to ``pos`` would break the agent's
    own personal norm.

    Mirrors the checks done in resolve_spacing_norm_conflict and
    resolve_occupation_region_norm_conflict so the swap resolver does not
    push an agent into a norm-violating cell while avoiding the swap.
    Newcomer (non-oracle) agents and norm-less agents are never constrained here.

    Every navigation norm is covered, not just the spacing ones: an agent
    redirected onto a forbidden X cell, or into a turn its movement norm bars,
    is charged the violation penalty for a move the environment forced on it,
    not one it chose. Door and seat norms are deliberately absent — they
    constrain which cell an agent finishes on, which a transit redirect cannot
    change.

    Args:
        agent: The agent being redirected.
        pos: The candidate redirect position.
        agent_classes: Mapping of agent_id to agent_class.
        agents: Full list of agents in the environment.
        world: The environment world object.
    """
    if agent.norm is None or not agent.is_oracle:
        return False

    # Predicted next positions of the other active norm agents this step
    # (their committed/best-guess next_position), matching the per-agent
    # resolvers' occupancy/spacing snapshots. When called from collision
    # resolution (before any next_position is set this step), fall back to
    # current positions so the check still sees the other agents.
    predicted: Dict[str, Tuple[int, int]] = {}
    for other in agents:
        if other.agent_id == agent.agent_id:
            continue
        if not other.active or other.done:
            continue
        if other.norm is None or not other.is_oracle:
            continue
        if other.next_position is not None:
            predicted[other.agent_id] = other.next_position
        else:
            predicted[other.agent_id] = other.current_position

    if is_navigation_norm(agent.norm):
        with _quiet():
            return check_navigation_norm(
                agent.norm, agent.current_position, pos,
                predicted, world, agent.agent_id, agent_classes,
                target_position=agent.current_target,
            )
    if is_occupation_region_norm(agent.norm):
        with _quiet():
            return check_occupation_norm(
                agent.norm, agent.current_position, pos,
                predicted, world, agent.agent_id, agent_classes,
            )
    if is_line_norm(agent.norm):
        # Agents past their first target are not graded under a line norm, so a
        # redirect cannot make them violate it.
        if agent.target_index >= 1 or agent.current_target is None:
            return False
        # Line norms are graded against every agent's pre-move position (unlike the
        # spacing/occupation snapshots above, which use next positions). Judge the
        # redirect on that same state so the swap resolver never pushes an agent
        # into a cell the violation check will then flag it for.
        current_positions = {
            a.agent_id: a.current_position
            for a in agents if a.active and not a.done
        }
        with _quiet():
            return check_norm_violation(
                agent.norm, agent.current_position, pos, agent.current_target,
                current_positions, world, agent.agent_id, agent_classes,
            )
    return False


def _get_direction_action(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
    """
    Calculate the direction action from one position to another.
    
    Args:
        from_pos: Starting position
        to_pos: Ending position
    
    Returns:
        Action string (move_up, move_down, move_left, move_right)
    """
    from_r, from_c = from_pos
    to_r, to_c = to_pos

    if to_r < from_r and to_c == from_c:
        return 'move_up'
    elif to_r > from_r and to_c == from_c:
        return 'move_down'
    elif to_r == from_r and to_c < from_c:
        return 'move_left'
    elif to_r == from_r and to_c > from_c:
        return 'move_right'
    else:
        return 'wait'


