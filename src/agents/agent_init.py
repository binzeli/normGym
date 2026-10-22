import random
from src.norms.lining import counter_service_cell
from src.norms.navigation import is_navigation_norm, is_movement_norm
from src.norms.occupation import (
    is_occupation_norm,
    is_occupation_region_norm,
    is_occupation_door_norm,
    is_occupation_seat_norm,
)


def get_agent_entry_position(world, norm_type=None, last_entrance=None):
    """Return a randomly chosen entrance cell from the world.

    For movement norms, avoids last_entrance so consecutive agents
    do not enter through the same door.
    """
    entrances = world.entrances
    if is_movement_norm(norm_type) and last_entrance is not None and len(entrances) > 1:
        candidates = [e for e in entrances if e != last_entrance]
    else:
        candidates = list(entrances)
    return random.choice(candidates)


def _counter_front_cell(world, norm_type=None):
    """The counter cell an agent is sent to (the first queue position).

    One row below a top-wall counter, or the adjacent interior cell when the
    counter sits on the left/right wall — except under the edge-line norms,
    where service happens from the counter's side cell. Mirrors
    world.is_at_counter. Falls back to (2, 6) if no counter zone.
    """
    counter_zone = world.zones.get('counter')
    if counter_zone:
        return counter_service_cell(world, norm_type)
    return (2, 6)


def _sittable_positions(world):
    """All sittable chair cells across the world's zones."""
    return [
        pos
        for positions in world.zones.values()
        for pos in positions
        if world.is_sittable(pos)
    ]


def errand_targets(world):
    """First-goal candidates that lie outside the waiting zone: the counter-front
    cell and every chair.

    Under the region norms every agent is normally sent to a waiting-zone cell,
    so a full batch queues for the same few slots. The test stage gives all but
    the staged zone holders one of these errands instead, keeping them clear of
    the region and out of the subject's approach (see _stage_region_zone in
    experiment/run/common.py).
    """
    zone = set(world.waiting_zone_cells)
    candidates = [_counter_front_cell(world)] + _sittable_positions(world)
    return [pos for pos in candidates if pos not in zone]


def _pick_first_target(first_target_candidates, world, extra_claimed=None):
    """Choose a first target, preferring cells not already claimed by other
    agents. Excludes first targets claimed by agents that are not yet done,
    plus any positions in extra_claimed (used to prevent duplicates within a
    batch being built before world.agents is populated)."""
    claimed = {
        a.target_positions[0]
        for a in getattr(world, 'agents', [])
        if not getattr(a, 'done', True) and getattr(a, 'target_positions', None)
    }
    if extra_claimed:
        claimed = claimed | extra_claimed
    available_candidates = [p for p in first_target_candidates if p not in claimed]
    if not available_candidates:
        available_candidates = first_target_candidates
    return random.choice(available_candidates)


def get_agent_targets(entry_position, world, norm_type=None, extra_claimed=None):
    """Compute target_positions list for one agent.
    entry_position is excluded from possible target entrances.
    First-target candidates exclude positions already claimed by undone agents
    currently registered on the world, plus any positions in extra_claimed
    (used to prevent duplicates within a batch being built before world.agents
    is populated)."""
    if is_navigation_norm(norm_type):
        first_target_candidates = [_counter_front_cell(world)] + _sittable_positions(world)
        first_target = _pick_first_target(first_target_candidates, world, extra_claimed)
        exit_door = random.choice(world.entrances)
        return [first_target, exit_door]
    elif is_occupation_norm(norm_type):
        waiting_cells = list(world.waiting_zone_cells)
        sittable_positions = _sittable_positions(world)
        if is_occupation_region_norm(norm_type):
            # norm_029–031: first settle on a cell inside the waiting zone,
            # then exit through one of the doors.
            first_target = _pick_first_target(waiting_cells, world, extra_claimed)
            exit_door = random.choice(world.entrances)
            return [first_target, exit_door]
        elif is_occupation_door_norm(norm_type):
            # norm_032–038: first settle on a chair / counter-front cell /
            # waiting-zone cell, then exit through one of the doors.
            first_target_candidates = sittable_positions + [_counter_front_cell(world)] + waiting_cells
            first_target = _pick_first_target(first_target_candidates, world, extra_claimed)
            exit_door = random.choice(world.entrances)
            return [first_target, exit_door]
        else:
            # norm_039–045 (seat norms): first sit on a chair, then exit a door.
            first_target = _pick_first_target(sittable_positions, world, extra_claimed)
            exit_door = random.choice(world.entrances)
            return [first_target, exit_door]
    else:
        exit_door = random.choice(world.entrances)
        return [_counter_front_cell(world, norm_type), exit_door]
