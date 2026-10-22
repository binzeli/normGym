import random
from typing import Tuple, Optional, List, Dict

from src.agents.base_agent import AGENT_COLORS

# ---------------------------------------------------------------------------
# Occupation norms (norm_029 – norm_050)
#
# Three subcategories, distinguished by where the agent's *first* goal lies:
#   - region norms : the goal is a cell inside the waiting zone
#   - door norms   : first goal is a chair / (2,6) / waiting-zone cell, then a door
#   - seat norms   : first goal is a sittable chair, then a door
# ---------------------------------------------------------------------------

# norm_029 – norm_031: rules about occupying the waiting zone.
OCCUPATION_REGION_NORMS = {
    'occupation_region_capacity_1',
    'occupation_region_capacity_2',
    'occupation_region_single_class_only',
}

# norm_032 – norm_038, norm_046 – norm_047: rules about which door an agent may
# exit through. Every world has exactly one door per border side, labeled
# A = left, B = bottom, C = right (see DOOR_NORM_RULES). The class_specific
# suffix names the doors of (orange, green, blue) in that fixed class order,
# so the base mapping is ABC (see DOOR_BY_CLASS_BY_NORM).
OCCUPATION_DOOR_NORMS = {
    'occupation_door_all_exit_A',
    'occupation_door_all_exit_B',
    'occupation_door_all_exit_C',
    'occupation_door_A_forbidden',
    'occupation_door_B_forbidden',
    'occupation_door_C_forbidden',
    'occupation_door_class_specific',
    'occupation_door_class_specific_BCA',
    'occupation_door_class_specific_CAB',
}

# norm_039 – norm_045, norm_048 – norm_050: rules about which chair an agent
# may sit on. The positional members bind one quadrant of the room (see
# SEAT_QUADRANT_RULES); top-right is left out because only four layouts have
# chairs there. The class_specific rot variants rotate the class→quadrant
# assignment (SEAT_REGION_BY_CLASS_BY_NORM); in_order_right fills the columns
# from the right instead of the left (SEAT_ORDER_KEYS).
OCCUPATION_SEAT_NORMS = {
    'occupation_seat_class_specific',
    'occupation_seat_class_specific_rot1',
    'occupation_seat_class_specific_rot2',
    'occupation_seat_bottom_left_only',
    'occupation_seat_no_bottom_left',
    'occupation_seat_top_left_only',
    'occupation_seat_bottom_right_only',
    'occupation_seat_no_bottom_right',
    'occupation_seat_in_order',
    'occupation_seat_in_order_right',
}

OCCUPATION_NORMS = (
    OCCUPATION_REGION_NORMS | OCCUPATION_DOOR_NORMS | OCCUPATION_SEAT_NORMS
)


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# Region-capacity norms: maximum agents allowed in the waiting zone simultaneously.
REGION_CAPACITY = {
    'occupation_region_capacity_1': 1,
    'occupation_region_capacity_2': 2,
}

# norm_031: how many agents hold the waiting zone in a single_class_only test
# run, drawn per run from this inclusive range. The norm caps no number, only the
# mix of classes, so any size in this range shuts the zone to the subject just as
# firmly (see region_hold_count / region_hold_class).
SINGLE_CLASS_HOLD_RANGE = (1, 3)

# norm_029: the subject enters as the last agent of the initial batch instead of
# at a random slot, so it walks in on a room the scripted customers already hold.
# With capacity 1 the zone is a single slot the others are cycling through, and a
# subject dropped in at a random slot can find itself racing them from step 1;
# entering last puts the contention in front of it rather than around it. Read by
# the experiment pipeline (experiment/run/common.py) via
# subject_enters_last(); the batch size for this norm comes from norm_capacity in
# experiment_config.json.
SUBJECT_TEST_ENTERS_LAST = {
    'occupation_region_capacity_1',
}

# norm_032–033, norm_035–038: the side each single-door norm binds and how.
# (side, mandatory): mandatory=True means every agent must exit that side
# (all_exit), False means that side is forbidden and any other door is fine.
# Sides follow the door labels A = left, B = bottom, C = right.
DOOR_NORM_RULES = {
    'occupation_door_all_exit_A': ('left', True),
    'occupation_door_all_exit_B': ('bottom', True),
    'occupation_door_all_exit_C': ('right', True),
    'occupation_door_A_forbidden': ('left', False),
    'occupation_door_B_forbidden': ('bottom', False),
    'occupation_door_C_forbidden': ('right', False),
}

# norm_040–041, norm_043–045: the quadrant each positional seat norm binds and
# how. (quadrant, only): only=True permits chairs in that quadrant alone,
# False forbids that quadrant's chairs and permits every other chair.
# Quadrants are cut by region_divider / seat_region_of.
SEAT_QUADRANT_RULES = {
    'occupation_seat_bottom_left_only': ('bottom_left', True),
    'occupation_seat_no_bottom_left': ('bottom_left', False),
    'occupation_seat_top_left_only': ('top_left', True),
    'occupation_seat_bottom_right_only': ('bottom_right', True),
    'occupation_seat_no_bottom_right': ('bottom_right', False),
}

# norm_034: the door each agent class is allowed to exit through (left / bottom / right).
DOOR_BY_CLASS = {
    'orange': 'left',
    'green': 'bottom',
    'blue': 'right',
}

# norm_034, norm_046 – norm_047: the class→door assignment each class_specific
# door norm binds. The suffix spells the doors of (orange, green, blue) in that
# order with A = left, B = bottom, C = right, so the base mapping is ABC and
# the variants are its two cyclic rotations — every class changes doors and all
# three doors stay in use.
DOOR_BY_CLASS_BY_NORM = {
    'occupation_door_class_specific': DOOR_BY_CLASS,
    'occupation_door_class_specific_BCA': {
        'orange': 'bottom',
        'green': 'right',
        'blue': 'left',
    },
    'occupation_door_class_specific_CAB': {
        'orange': 'right',
        'green': 'left',
        'blue': 'bottom',
    },
}

# norm_039: the room quadrant each agent class must take its seat in. Chairs all
# render the same, so the rule is positional rather than chair-color based (see
# seat_region_of for how the quadrants are cut).
SEAT_REGION_BY_CLASS = {
    'orange': 'top_left',
    'green': 'bottom_left',
    'blue': 'bottom_right',
}

# norm_039, norm_048 – norm_049: the class→quadrant assignment each
# class_specific seat norm binds. The rot variants are the two cyclic rotations
# of the base mapping over the same three quadrants (top-right stays excluded),
# so a layout feasible for one mapping is feasible for all of them — only
# layouts with chairs in all three quadrants qualify either way
# (seat_norm_has_compliant_chair).
SEAT_REGION_BY_CLASS_BY_NORM = {
    'occupation_seat_class_specific': SEAT_REGION_BY_CLASS,
    'occupation_seat_class_specific_rot1': {
        'orange': 'bottom_left',
        'green': 'bottom_right',
        'blue': 'top_left',
    },
    'occupation_seat_class_specific_rot2': {
        'orange': 'bottom_right',
        'green': 'top_left',
        'blue': 'bottom_left',
    },
}

# norm_040/041 and norm_043–045: the door the subject agent must enter through during the
# test stage, so that the seating region the norm points at (bottom-left) is on
# the far side of the room from where the subject starts. Read by the experiment
# pipeline (experiment/run/common.py) via subject_entry_door(). norm_039
# pins its entrance the same way, but per class rather than per norm — its
# permitted region depends on the subject's color, so the door is computed from
# the layout instead of looked up here (see subject_entry_door).
SUBJECT_TEST_ENTRY_SIDE = {
    'occupation_seat_bottom_left_only': 'right',
    'occupation_seat_no_bottom_left': 'left',
    'occupation_seat_top_left_only': 'right',
    'occupation_seat_bottom_right_only': 'left',
    'occupation_seat_no_bottom_right': 'right',
}

# norm_032 – norm_038: where the subject agent's seat (its first goal) is placed
# in the test stage, relative to the door the norm names. 'opposite' seats it in
# the quadrants across the room from that door, so reaching the door it must use
# is a trip; 'near' seats it in the quadrants the door sits on, so the closest
# exit is the one the norm forbids. Either way the norm-blind shortest path and
# the compliant path come apart. Read by the experiment pipeline via
# subject_test_seat(); the norm_034 door is per class (DOOR_BY_CLASS).
SUBJECT_TEST_SEAT_PLACEMENT = {
    'occupation_door_all_exit_A': ('left', 'opposite'),
    'occupation_door_all_exit_B': ('bottom', 'opposite'),
    'occupation_door_all_exit_C': ('right', 'opposite'),
    'occupation_door_A_forbidden': ('left', 'near'),
    'occupation_door_B_forbidden': ('bottom', 'near'),
    'occupation_door_C_forbidden': ('right', 'near'),
    'occupation_door_class_specific': (None, 'opposite'),
    'occupation_door_class_specific_BCA': (None, 'opposite'),
    'occupation_door_class_specific_CAB': (None, 'opposite'),
}

# Per-test-run placements for the 2026-08-30 door/seat norm variants
# (norm_035–038, norm_043–045): keyed by (norm, world class), one (entrance side, seat)
# per test run, cycled by the 1-based test index. Only the runs of these
# norms vary their setting; every other norm keeps the single placement the
# tables above derive.
#
# A seat of None pins the entrance only — the seat norms leave the chair as
# the subject's free choice, and their variety comes from the entrance (plus
# the per-run seed). Where one side repeats (no_bottom_right in the cafe),
# it is because it is the only diverging entrance in that layout.
#
# Every row was audited with BFS path distances over the empty map so the
# divergence property holds run by run: the shortest-path choice violates the
# norm, and the compliant choice costs a detour of +1..+15 steps. Re-audit
# whenever a map's doors or chairs move.
SUBJECT_TEST_RUN_PLACEMENTS = {
    ('occupation_door_all_exit_B', 'CafeWorld'): (
        ('left', (1, 1)), ('left', (3, 3)), ('right', (1, 1))),
    ('occupation_door_all_exit_B', 'PostOfficeWorld'): (
        ('right', (1, 13)), ('right', (3, 12)), ('right', (1, 10))),
    ('occupation_door_all_exit_C', 'CafeWorld'): (
        ('left', (6, 1)), ('bottom', (8, 5)), ('left', (1, 1))),
    ('occupation_door_all_exit_C', 'BarbershopWorld'): (
        ('left', (2, 1)), ('left', (8, 1)), ('bottom', (8, 2))),
    ('occupation_door_B_forbidden', 'CafeWorld'): (
        ('bottom', (8, 5)), ('bottom', (8, 9)), ('right', (8, 9))),
    ('occupation_door_B_forbidden', 'PharmacyWorld'): (
        ('bottom', (9, 3)), ('bottom', (9, 2)), ('left', (9, 3))),
    ('occupation_door_C_forbidden', 'CafeWorld'): (
        ('right', (6, 13)), ('right', (6, 11)), ('bottom', (8, 13))),
    ('occupation_door_C_forbidden', 'ClothingStoreWorld'): (
        ('right', (4, 12)), ('right', (5, 12)), ('bottom', (6, 12))),
    ('occupation_seat_top_left_only', 'CafeWorld'): (
        ('right', None), ('bottom', None), ('left', None)),
    ('occupation_seat_top_left_only', 'MovieTheaterWorld'): (
        ('right', None), ('bottom', None), ('left', None)),
    ('occupation_seat_bottom_right_only', 'CafeWorld'): (
        ('left', None), ('bottom', None), ('left', None)),
    ('occupation_seat_bottom_right_only', 'HospitalWorld'): (
        ('left', None), ('left', None), ('left', None)),
    ('occupation_seat_no_bottom_right', 'CafeWorld'): (
        ('right', None), ('right', None), ('right', None)),
    ('occupation_seat_no_bottom_right', 'HospitalWorld'): (
        ('right', None), ('bottom', None), ('right', None)),
}


def _test_run_placement(norm: Optional[str], world, test_index: Optional[int]):
    """The (entrance side, seat) pinned for this test run, or None when the
    norm/world pair has no per-run placements or no test index was given."""
    if test_index is None:
        return None
    runs = SUBJECT_TEST_RUN_PLACEMENTS.get((norm, type(world).__name__))
    if not runs:
        return None
    return runs[(test_index - 1) % len(runs)]

# Norms whose test-stage entrance and first goal are pinned per layout rather
# than by a rule read off the doors. Keyed by (norm, world class); the entrance
# is a border side, the seat pool the quadrants (see seat_region_of) the first
# goal is drawn from, nearest chair to the pinned door first. Quadrants of None
# pin only the entrance and leave the first goal to the random chair draw.
#
# norm_027 (xcell_must_visit) is placed this way because what makes the norm
# bind is the geometry of the X cells, not of the doors: each layout's entrance
# and seating region are chosen so the direct walk from door to seat misses both
# X cells, and the compliant route has to detour to one.
#
# norm_028 (xcell_forbidden) is placed this way for the mirror-image reason: the
# entrance and seating region are chosen so the direct walk from door to seat
# runs straight over the X cells, and the compliant route has to detour around
# them. post_office needs no seat pin — its X cells sit in a row right above the
# bottom door, so every chair in the layout is behind them.
#
# Every named side has a door and every named quadrant a chair in the layout it
# is listed for (checked by subject_test_seat_placeable).
SUBJECT_TEST_PLACEMENT_BY_WORLD = {
    ('xcell_must_visit', 'BankWorld'): ('bottom', ('bottom_left',)),
    ('xcell_must_visit', 'PharmacyWorld'): ('bottom', ('top_right',)),
    ('xcell_must_visit', 'BarbershopWorld'): ('bottom', ('top_left', 'bottom_left')),
    ('xcell_must_visit', 'CafeWorld'): ('left', ('bottom_left',)),
    ('xcell_forbidden', 'LibraryWorld'): ('right', ('top_right',)),
    ('xcell_forbidden', 'PostOfficeWorld'): ('bottom', None),
    ('xcell_forbidden', 'GroceryStoreWorld'): ('bottom', ('bottom_right',)),
    ('xcell_forbidden', 'CafeWorld'): ('bottom', ('bottom_right',)),
}

# The two quadrants (see seat_region_of) lying along each border of the room,
# and the border facing it across the room.
SIDE_QUADRANTS = {
    'left': ('top_left', 'bottom_left'),
    'right': ('top_right', 'bottom_right'),
    'top': ('top_left', 'top_right'),
    'bottom': ('bottom_left', 'bottom_right'),
}
OPPOSITE_SIDE = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}

# Seats the general rule above cannot supply, chosen by hand per layout. Keyed
# by (norm, world class, agent class). hospital seats every chair in rows 7 and
# 9, so 'opposite the bottom door' — the top of the room — has nothing to offer
# a green subject; (7,12) is the chair farthest from the bottom door on the side
# no other class is sent to. The named seat's quadrant becomes the pool the
# subject may fall back within, exactly as for the rule-derived seats.
SUBJECT_TEST_SEAT_OVERRIDES = {
    ('occupation_door_class_specific', 'HospitalWorld', 'green'): (7, 12),
}


# ---------------------------------------------------------------------------
# Predicate helpers
# ---------------------------------------------------------------------------

def is_occupation_norm(norm: Optional[str]) -> bool:
    """Return True if norm belongs to any occupation subcategory."""
    return norm in OCCUPATION_NORMS


def is_occupation_region_norm(norm: Optional[str]) -> bool:
    """Return True if norm is a waiting-zone occupation norm (norm_029–031)."""
    return norm in OCCUPATION_REGION_NORMS


def is_occupation_door_norm(norm: Optional[str]) -> bool:
    """Return True if norm is a door occupation norm (norm_032–038)."""
    return norm in OCCUPATION_DOOR_NORMS


def is_occupation_seat_norm(norm: Optional[str]) -> bool:
    """Return True if norm is a seat occupation norm (norm_039–045)."""
    return norm in OCCUPATION_SEAT_NORMS


def region_capacity(norm: str) -> Optional[int]:
    """Return the max number of agents allowed in the waiting zone, or None."""
    return REGION_CAPACITY.get(norm)


def region_single_class_only(norm: str) -> bool:
    """Return True if the region may only hold agents of a single class at a time."""
    return norm == 'occupation_region_single_class_only'


def subject_enters_last(norm: Optional[str]) -> bool:
    """Return True if the test-stage subject takes the last slot of the initial
    batch rather than a random one (see SUBJECT_TEST_ENTERS_LAST)."""
    return norm in SUBJECT_TEST_ENTERS_LAST


# ---------------------------------------------------------------------------
# Region variant (norm_029 – norm_031)
# ---------------------------------------------------------------------------

def waiting_zone_cells(world) -> set:
    """The set of cells making up the world's waiting zone (occupation target region)."""
    return set(world.zones.get('waiting_zone', []))


def region_staged(norm: Optional[str]) -> bool:
    """True if the test stage stages this norm's waiting zone (holders + release).

    A stable predicate, unlike region_hold_count, which draws a number for
    single_class_only and so must be called exactly once per run.
    """
    return is_occupation_region_norm(norm)


def region_hold_count(norm: Optional[str], rng=None) -> int:
    """How many scripted agents hold the waiting zone for the test stage.

    The capacity norms fill the zone exactly: holding `capacity` agents there
    leaves the zone shut when the subject reaches its edge, so the norm binds at
    the moment the subject acts — the norm-blind subject walks in and violates,
    the norm-aware one waits. The holders are released the step after the subject
    arrives at the boundary (see release_region_holders in
    experiment/run/common.py), so both subjects then reach the goal and the
    only difference between them is the violation.

    single_class_only has no numeric capacity — its rule is about *which* class
    holds the zone, not how many agents are in it — so the count is drawn from
    SINGLE_CLASS_HOLD_RANGE per run. Any size bars the subject equally, since
    what shuts the zone is the holders' class (see region_hold_class); varying it
    keeps the runs from all presenting the same picture.

    Draws from the global RNG unless `rng` is given. The draw happens after every
    agent of the batch is built, at which point the stream is in the same state
    for every subject kind, so all of them get the same count.
    """
    if not is_occupation_region_norm(norm):
        return 0
    cap = region_capacity(norm)
    if cap is not None:
        return cap
    if region_single_class_only(norm):
        return (rng or random).randint(*SINGLE_CLASS_HOLD_RANGE)
    return 0


def region_hold_class(norm: Optional[str],
                      subject_class: Optional[str],
                      rng=None) -> Optional[str]:
    """The class the zone holders take, or None to leave their drawn classes.

    Under single_class_only the holders must share one class — a mixed zone is
    not a state the norm allows — and that class must differ from the subject's,
    which is what bars the subject: agent_may_enter_region lets a newcomer in
    only if it matches the class already inside. The capacity norms bar by count
    instead, so their holders keep whatever class they drew.
    """
    if not region_single_class_only(norm):
        return None
    others = [c for c in AGENT_COLORS if c != subject_class]
    if not others:
        return None
    return (rng or random).choice(others)


def at_region_boundary(world, pos: Optional[Tuple[int, int]]) -> bool:
    """True if pos is inside the waiting zone or one step from it.

    The trigger for releasing the zone holders: the subject has walked up to the
    edge of the zone and its next step would carry it in.
    """
    if pos is None:
        return False
    zone = waiting_zone_cells(world)
    if not zone:
        return False
    return any(abs(pos[0] - r) + abs(pos[1] - c) <= 1 for r, c in zone)


def agents_in_region(all_current_positions: Dict[str, Tuple[int, int]],
                     world,
                     exclude_id: Optional[str] = None) -> List[str]:
    """Return the ids of agents occupying the waiting zone.

    Agents are read from all_current_positions, which holds every active,
    non-done agent's position. A region-norm agent only becomes `done` after it
    exits through a door (its targets are [waiting_cell, exit_door]), so an agent
    that is ordering, waiting, or idling in the zone is still active and present
    here — it counts toward capacity and keeps others out while it sits there.
    An agent on its way out of the zone frees its slot. The moving agent
    (exclude_id) is omitted.
    """
    zone = waiting_zone_cells(world)
    occupants = set()
    for aid, pos in all_current_positions.items():
        if aid == exclude_id or pos is None:
            continue
        if pos in zone:
            occupants.add(aid)
    return list(occupants)


def agent_may_enter_region(norm: str,
                           agent_id: str,
                           agent_classes: Dict[str, str],
                           all_current_positions: Dict[str, Tuple[int, int]],
                           world) -> bool:
    """Return True if the agent is permitted to enter the waiting zone right now."""
    inside = agents_in_region(all_current_positions, world, exclude_id=agent_id)

    cap = region_capacity(norm)
    if cap is not None:
        return len(inside) < cap

    if region_single_class_only(norm):
        # Only one class may occupy the region at a time. If it already holds
        # agents, the newcomer must share their class; an empty region is open
        # to anyone (simultaneous-entry ties are broken by collision resolution).
        classes_inside = {agent_classes.get(a) for a in inside}
        if not classes_inside:
            return True
        return agent_classes.get(agent_id) in classes_inside

    return True


# ---------------------------------------------------------------------------
# Door variant (norm_032 – norm_038)
# ---------------------------------------------------------------------------

def door_side(world, pos: Tuple[int, int]) -> Optional[str]:
    """Classify an entrance/door cell by which border of the grid it sits on.

    Returns 'left', 'right', 'bottom', 'top', or None when the cell is not on a
    border. The default layout's doors map as: left (col 0), bottom (last row),
    right (last col).
    """
    r, c = pos
    if c <= 0:
        return 'left'
    if c >= world.width - 1:
        return 'right'
    if r >= world.height - 1:
        return 'bottom'
    if r <= 0:
        return 'top'
    return None


def door_exit_allowed(norm: str, agent_class: Optional[str], side: Optional[str]) -> bool:
    """Return True if an agent of `agent_class` may exit through the given side."""
    if side is None:
        return True

    rule = DOOR_NORM_RULES.get(norm)
    if rule is not None:
        # all_exit_X: everyone must leave through side X.
        # X_forbidden: side X is forbidden; any other door is fine.
        norm_side, mandatory = rule
        return side == norm_side if mandatory else side != norm_side

    class_map = DOOR_BY_CLASS_BY_NORM.get(norm)
    if class_map is not None:
        # Each class is bound to one door, per the norm's class→door assignment
        # (the base ABC mapping or one of its rotations).
        return class_map.get(agent_class) == side

    return True


# ---------------------------------------------------------------------------
# Seat variant (norm_039 – norm_045)
# ---------------------------------------------------------------------------

def region_divider(world) -> Tuple[int, int]:
    """(min_row, max_col): the row/column cut that splits the room into
    quadrants. Rows from min_row down to the bottom wall are the "bottom" half;
    columns from the left wall out to max_col are the "left" half.

    The cut is drawn through the doors: min_row is the left door's row and
    max_col the bottom door's column, so the bottom-left quadrant is exactly the
    corner square those two doors enclose. In the cafe (left door (5,0), bottom
    door (10,7)) that is rows 5-10 x columns 0-7.

    Falls back to halving the grid when the world has no door on one of those
    sides.
    """
    doors = list(getattr(world, 'entrances', []))
    left_rows = [r for (r, c) in doors if door_side(world, (r, c)) == 'left']
    bottom_cols = [c for (r, c) in doors if door_side(world, (r, c)) == 'bottom']
    min_row = min(left_rows) if left_rows else (world.height + 1) // 2
    max_col = max(bottom_cols) if bottom_cols else world.width // 2 - 1
    return min_row, max_col


def seat_region_of(world, pos: Tuple[int, int]) -> str:
    """The quadrant `pos` falls in: 'top_left', 'top_right', 'bottom_left' or
    'bottom_right', cut by region_divider(). Used by the positional seat norms
    (norm_040–041, norm_043–045), which replaced the chair-color rules now that all chairs
    render the same."""
    min_row, max_col = region_divider(world)
    r, c = pos
    vertical = 'bottom' if r >= min_row else 'top'
    horizontal = 'left' if c <= max_col else 'right'
    return f'{vertical}_{horizontal}'


def in_bottom_left_region(world, pos: Tuple[int, int]) -> bool:
    """Return True if `pos` lies in the bottom-left region — the corner square
    enclosed by the left door and the bottom door (see region_divider)."""
    return seat_region_of(world, pos) == 'bottom_left'


def seat_norm_has_compliant_chair(norm: Optional[str], world) -> bool:
    """True if `world` holds a chair for every agent the positional seat norms
    constrain.

    Not every layout has chairs in each quadrant (grocery_store and post_office
    have none in the bottom-left region at all), which would leave
    occupation_seat_bottom_left_only unsatisfiable there — and
    occupation_seat_class_specific needs a chair in all three class quadrants so
    no class is stranded. Callers use this to warn before running such a
    pairing. Norms other than the positional seat norms are always reported
    feasible.
    """
    seats = _ordered_seats(world)
    class_regions = SEAT_REGION_BY_CLASS_BY_NORM.get(norm)
    if class_regions is not None:
        regions = {seat_region_of(world, s) for s in seats}
        return set(class_regions.values()) <= regions
    rule = SEAT_QUADRANT_RULES.get(norm)
    if rule is not None:
        quadrant, only = rule
        return any((seat_region_of(world, s) == quadrant) == only for s in seats)
    return True


# norm_042, norm_050: the fixed fill order each in_order norm binds, as a sort
# key over chair cells. The base norm fills column by column from the left,
# in_order_right from the right; both go top→bottom within a column.
SEAT_ORDER_KEYS = {
    'occupation_seat_in_order': lambda p: (p[1], p[0]),
    'occupation_seat_in_order_right': lambda p: (-p[1], p[0]),
}


def is_seat_in_order_norm(norm: Optional[str]) -> bool:
    """True if norm is a fill-order seat norm (norm_042, norm_050)."""
    return norm in SEAT_ORDER_KEYS


def _ordered_seats(world, norm: Optional[str] = None) -> List[Tuple[int, int]]:
    """All chair cells in the fill order `norm` binds (SEAT_ORDER_KEYS), falling
    back to the base order — column by column left→right, top→bottom within a
    column — for every other norm, where the order is only a deterministic
    tie-break, not a rule."""
    key = SEAT_ORDER_KEYS.get(norm, SEAT_ORDER_KEYS['occupation_seat_in_order'])
    return sorted(world.zones.get('chairs', []), key=key)


def _occupied_seats(all_current_positions: Dict[str, Tuple[int, int]],
                    world,
                    exclude_id: Optional[str] = None) -> set:
    """Seat cells held by other agents, per their predicted positions in
    all_current_positions (the moving agent, exclude_id, is omitted)."""
    occupied = set()
    for aid, pos in all_current_positions.items():
        if aid == exclude_id or pos is None:
            continue
        if world.is_sittable(pos):
            occupied.add(pos)
    return occupied


def seat_in_order_index(world, norm: str = 'occupation_seat_in_order') -> int:
    """Read-only: the seat index all agents currently aim at under the in-order
    norm, i.e. the fill frontier stored on `world`. The frontier is advanced by
    advance_seat_in_order_frontier() during goal selection; here we only read
    it (used by the norm-compliance check)."""
    ordered = _ordered_seats(world, norm)
    if not ordered:
        return 0
    return getattr(world, 'seat_in_order_frontier', 0) % len(ordered)


def advance_seat_in_order_frontier(all_current_positions: Dict[str, Tuple[int, int]],
                                   world,
                                   norm: str = 'occupation_seat_in_order') -> int:
    """Advance and return the in-order fill frontier (in `norm`'s fill order,
    see SEAT_ORDER_KEYS) from the seats currently occupied (per
    all_current_positions).

    The frontier is a forward-only index remembered on `world`: it steps past
    the target seat as soon as that seat is actually occupied, so a seat vacated
    *behind* the frontier no longer looks like the next seat (once seat 1 has
    been taken, agents move on to seat 2 even after seat 1 empties). When the
    last seat is filled the frontier wraps back to the first seat, restarting
    the cycle. Driven from real positions during goal selection (pick_seat), so
    planner look-ahead never moves it."""
    ordered = _ordered_seats(world, norm)
    if not ordered:
        return 0
    occupied = _occupied_seats(all_current_positions, world)
    frontier = getattr(world, 'seat_in_order_frontier', 0) % len(ordered)
    # Step past the leading run of occupied seats starting at the frontier.
    while frontier < len(ordered) and ordered[frontier] in occupied:
        frontier += 1
    if frontier >= len(ordered):
        frontier = 0  # last seat filled -> restart the fill cycle
    world.seat_in_order_frontier = frontier
    return frontier


def seat_violates(norm: str,
                  to_pos: Tuple[int, int],
                  agent_id: str,
                  agent_class: Optional[str],
                  all_current_positions: Dict[str, Tuple[int, int]],
                  world) -> bool:
    """Return True if sitting on the chair at `to_pos` violates the seat norm."""
    class_regions = SEAT_REGION_BY_CLASS_BY_NORM.get(norm)
    if class_regions is not None:
        # The chair must lie in the quadrant bound to the agent's class, per
        # the norm's class→quadrant assignment (base mapping or a rotation).
        return seat_region_of(world, to_pos) != class_regions.get(agent_class)

    rule = SEAT_QUADRANT_RULES.get(norm)
    if rule is not None:
        # only=True : only chairs in the named quadrant may be occupied.
        # only=False: chairs in the named quadrant may not be occupied.
        quadrant, only = rule
        in_quadrant = seat_region_of(world, to_pos) == quadrant
        return not in_quadrant if only else in_quadrant

    if is_seat_in_order_norm(norm):
        # Only the current in-order seat (right after the highest occupied seat
        # in this norm's fill order) may be claimed.
        ordered = _ordered_seats(world, norm)
        if not ordered:
            return False
        return to_pos != ordered[seat_in_order_index(world, norm)]

    return False


# ---------------------------------------------------------------------------
# Free-choice goals (LLM / human agents)
#
# Interactive agents are not handed the specific compliant cell for the norm-
# constrained leg of their trip. Instead that goal is left to free choice and
# graded purely by norm compliance:
#   - seat norms : the chair (first target) is the free choice — the agent must
#                  decide which chair to sit on; the exit door stays preset.
#   - door norms : the exit door (last target) is the free choice — the agent
#                  must decide which door to leave through; the chair stays preset.
# For seat and door norms, task success and norm compliance are decoupled: ANY
# chair / door completes the goal, and stepping onto a non-compliant one is
# flagged as a violation (handled by check_occupation_norm with
# target_position=None) but still counts as finishing the goal. Region norms
# keep compliance-graded completion (the zone cell must be one the agent is
# permitted to occupy).
# ---------------------------------------------------------------------------

def free_choice_goal_index(norm: Optional[str], num_targets: int) -> Optional[int]:
    """Index of the target an interactive (LLM/human) agent must choose freely
    (hidden from the agent, graded by norm compliance). Returns None when the
    norm has no free-choice goal.

    Applies to every occupation norm:
      - seat norms (including occupation_seat_in_order): the agent picks a seat;
        any chair completes the goal, and a non-compliant one is additionally
        flagged as a violation. (A seated agent is latched to its seat by
        free_goal_settled so the advancing fill frontier cannot later re-judge
        it.)
      - region norms: the agent picks which waiting-zone cell to stand on, and
        the goal is met by any cell of the zone it is permitted to occupy — the
        capacity / single-class rule is the grading (see free_goal_satisfied).
      - door norms: the agent picks the exit door."""
    if num_targets < 1:
        return None
    if is_occupation_seat_norm(norm) or is_occupation_region_norm(norm):
        return 0
    if is_occupation_door_norm(norm):
        return num_targets - 1
    return None


def free_goal_settled(agent, world, all_current_positions: Dict[str, Tuple[int, int]]) -> bool:
    """Whether `agent` has settled on its free-choice goal cell.

    The first satisfying cell the agent reaches is latched on ``agent._free_goal_cell``
    so a later change in the grading — notably the occupation_seat_in_order fill
    frontier advancing past the seat the agent took — cannot strand a settled
    agent mid-sit. The latch is dropped if the agent leaves that cell (e.g. it
    is pushed off), letting it choose again.

    This is completion-only logic; stepping onto a non-compliant chair/door is
    still flagged as a violation by check_occupation_norm regardless of the latch.
    """
    latched = getattr(agent, '_free_goal_cell', None)
    if latched is not None:
        if agent.current_position == latched:
            return True
        agent._free_goal_cell = None  # left / pushed off — re-choose
    if free_goal_satisfied(agent.norm, agent.current_position, agent.agent_id,
                           agent.agent_class, all_current_positions, world):
        agent._free_goal_cell = agent.current_position
        return True
    return False


def _agent_classes_from_world(world) -> Dict[str, str]:
    """The class of every agent the env has registered on the world.

    The env publishes its agent list as ``world.agents`` each step precisely so
    norm checks can read state the caller does not carry (see gym_env). Returns
    an empty mapping when no list is registered.
    """
    return {
        agent.agent_id: agent.agent_class
        for agent in (getattr(world, 'agents', None) or [])
        if getattr(agent, 'agent_id', None) is not None
    }


def free_goal_satisfied(norm: str,
                        pos: Tuple[int, int],
                        agent_id: str,
                        agent_class: Optional[str],
                        all_current_positions: Dict[str, Tuple[int, int]],
                        world) -> bool:
    """True if `pos` is a settling cell for the agent's free-choice occupation
    goal: any chair (seat norms), a waiting-zone cell the agent is permitted to
    occupy (region norms), or any exit door (door norms). This is the completion
    test for an interactive agent's free goal — for seat and door norms it is
    deliberately norm-blind, so task success and norm violation are measured
    independently (a wrong chair / door still finishes the goal; the violation
    is flagged separately by check_occupation_norm)."""
    if is_occupation_seat_norm(norm):
        return world.is_sittable(pos)
    if is_occupation_region_norm(norm):
        # Any cell of the waiting zone will do, provided the capacity /
        # single-class rule leaves room for this agent. agent_may_enter_region
        # excludes the agent itself, so asking it about a cell the agent already
        # stands on reads as "is my presence here permitted".
        if pos not in waiting_zone_cells(world):
            return False
        return agent_may_enter_region(
            norm, agent_id, _agent_classes_from_world(world),
            all_current_positions, world)
    if is_occupation_door_norm(norm):
        return pos in set(getattr(world, 'entrances', []))
    return False


# ---------------------------------------------------------------------------
# Goal selection — pick a norm-compliant door / seat for an agent
#
# Used by the planning agent to overwrite the (randomly assigned) exit door or
# seat goal with one that actually satisfies the active occupation norm.
# ---------------------------------------------------------------------------

def _door_farthest_from(world, seats: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """The entrance whose nearest seat in `seats` is the farthest away (Manhattan),
    i.e. the door on the opposite side of the room from that group of seats.
    Ties keep the world's entrance order, so the choice is deterministic."""
    doors = list(getattr(world, 'entrances', []))
    if not doors or not seats:
        return None
    return max(doors, key=lambda d: min(abs(d[0] - s[0]) + abs(d[1] - s[1])
                                        for s in seats))


def subject_entry_door(norm: Optional[str], world,
                       agent_class: Optional[str] = None,
                       test_index: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """The entrance the subject agent must start at in the test stage under
    `norm`, or None when the norm pins no door (every other norm keeps the
    random entrance).

    The positional seat norms pin it so the subject always has to travel to
    reach its permitted seats:
    - occupation_seat_bottom_left_only : the right door.
    - occupation_seat_no_bottom_left : the left door.
    - occupation_seat_class_specific : the door opposite the quadrant the
      subject's class is bound to — the entrance farthest from that quadrant's
      chairs (needs agent_class; without it the entrance stays random).

    The X-cell norms pin it per layout instead (SUBJECT_TEST_PLACEMENT_BY_WORLD),
    together with the seating region its first goal is drawn from.

    When the world has no door on the named side, or holds no chair for the
    class, None is returned and the caller keeps the random entrance.

    The norms in SUBJECT_TEST_RUN_PLACEMENTS pin a different entrance per test
    run: pass the 1-based `test_index` to select that run's side (without it
    the norm's single-placement rule applies).
    """
    run = _test_run_placement(norm, world, test_index)
    if run is not None:
        side = run[0]
        doors = [d for d in getattr(world, 'entrances', [])
                 if door_side(world, d) == side]
        return doors[0] if doors else None

    by_world = SUBJECT_TEST_PLACEMENT_BY_WORLD.get((norm, type(world).__name__))
    side = by_world[0] if by_world else SUBJECT_TEST_ENTRY_SIDE.get(norm)
    if side is not None:
        doors = [d for d in getattr(world, 'entrances', []) if door_side(world, d) == side]
        return doors[0] if doors else None

    class_regions = SEAT_REGION_BY_CLASS_BY_NORM.get(norm)
    if class_regions is not None and agent_class is not None:
        want = class_regions.get(agent_class)
        seats = [s for s in _ordered_seats(world) if seat_region_of(world, s) == want]
        return _door_farthest_from(world, seats)

    return None


def subject_test_seat_pool(norm: Optional[str], world,
                           agent_class: Optional[str] = None,
                           test_index: Optional[int] = None) -> List[Tuple[int, int]]:
    """Every chair the subject agent may be seated on in the test stage under
    `norm`, best first — the region subject_test_seat draws from (see there).
    Empty when the norm pins no seat or the layout has no chair in the region.

    The caller keeps the whole list, not just its head, so an agent whose seat
    turns out to be occupied can divert to the next chair in the same region
    instead of waiting out the episode on a chair it cannot have.

    The norms in SUBJECT_TEST_RUN_PLACEMENTS pin a different seat per test run:
    with a 1-based `test_index` that run's seat leads the pool, followed by the
    other audited run seats of the same (norm, world) pairing nearest-first.
    The pool deliberately stops there — every seat in it was divergence-checked,
    whereas the norm's wider permitted quadrants hold chairs whose nearest exit
    is already the compliant one (cafe's (6,11) under B_forbidden), and a
    fallback onto such a chair silently voids the test. When every audited seat
    is claimed, the pinned one is kept and briefly contested rather than
    trading the test away. A run placement whose seat is None (the seat norms)
    pins no seat, exactly like the norms outside the placement tables.
    """
    run = _test_run_placement(norm, world, test_index)
    if run is not None:
        pinned = run[1]
        if pinned is None:
            return []
        audited = []
        for _, seat in SUBJECT_TEST_RUN_PLACEMENTS[(norm, type(world).__name__)]:
            if seat is not None and seat != pinned and seat not in audited:
                audited.append(seat)
        audited.sort(key=lambda s: abs(s[0] - pinned[0]) + abs(s[1] - pinned[1]))
        return [pinned] + audited

    by_world = SUBJECT_TEST_PLACEMENT_BY_WORLD.get((norm, type(world).__name__))
    if by_world is not None:
        # Layout-pinned: the quadrants are named outright and the pool runs from
        # the chair nearest the pinned entrance outwards. Quadrants of None pin
        # the entrance only, leaving the caller its randomly drawn chair.
        side, quadrants = by_world
        if quadrants is None:
            return []
        doors = [d for d in getattr(world, 'entrances', []) if door_side(world, d) == side]
        seats = [s for s in _ordered_seats(world)
                 if seat_region_of(world, s) in set(quadrants)]
        if not doors or not seats:
            return []
        return sorted(seats, key=lambda s: min(abs(s[0] - d[0]) + abs(s[1] - d[1])
                                               for d in doors))

    placement = SUBJECT_TEST_SEAT_PLACEMENT.get(norm)
    if placement is None:
        return []
    side, mode = placement
    if side is None:
        side = DOOR_BY_CLASS_BY_NORM.get(norm, {}).get(agent_class)
    if side is None:
        return []

    doors = [d for d in getattr(world, 'entrances', []) if door_side(world, d) == side]
    if not doors:
        return []

    override = SUBJECT_TEST_SEAT_OVERRIDES.get(
        (norm, type(world).__name__, agent_class))
    if override is not None:
        quadrants = {seat_region_of(world, override)}
    else:
        want_side = OPPOSITE_SIDE[side] if mode == 'opposite' else side
        quadrants = set(SIDE_QUADRANTS[want_side])
    seats = [s for s in _ordered_seats(world) if seat_region_of(world, s) in quadrants]

    def door_distance(seat):
        return min(abs(seat[0] - d[0]) + abs(seat[1] - d[1]) for d in doors)

    ordered = sorted(seats, key=door_distance, reverse=(mode == 'opposite'))
    if override is not None and override in ordered:
        # The named seat leads its own quadrant's pool, whatever the distance
        # ordering would have put first.
        ordered = [override] + [s for s in ordered if s != override]
    return ordered


def subject_test_seat(norm: Optional[str], world,
                      agent_class: Optional[str] = None,
                      claimed: Optional[set] = None,
                      test_index: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """The chair the subject agent is sent to as its first goal in the test stage
    under `norm`, or None when the norm pins no seat (the caller then keeps the
    randomly drawn one).

    The door norms pin a seat by rule — for them the seat is the fixed first goal
    and the exit door is the agent's free choice, so where the agent is seated
    decides how far the compliant door is:
    - occupation_door_all_exit_A : a chair in the two quadrants across the room
      from the left door, so the mandatory left exit is the far one.
    - occupation_door_A_forbidden : a chair in the two quadrants along the left
      door, so the nearest exit is the forbidden one.
    - occupation_door_class_specific : a chair in the two quadrants across the
      room from the door bound to the subject's class (needs agent_class).

    The quadrants come from seat_region_of, so the rule is the same cut the seat
    norms use. Within the permitted quadrants the seat is the chair farthest
    from ('opposite') or nearest to ('near') that door, skipping chairs already
    claimed by an active agent on the world or listed in `claimed` — no RNG, so
    pinning the seat leaves the caller's random stream untouched. Returns None
    when the world has no door on that side or no chair in the permitted
    quadrants.

    The X-cell norms pin their seat per layout instead, in the quadrants named by
    SUBJECT_TEST_PLACEMENT_BY_WORLD and nearest the pinned entrance first: under
    xcell_must_visit so the subject's direct route to the seat misses the X cells
    it must visit, under xcell_forbidden so that route runs over them.
    """
    pool = subject_test_seat_pool(norm, world, agent_class, test_index=test_index)
    if not pool:
        return None
    taken = set(claimed or set()) | _claimed_first_targets(world)
    free = [s for s in pool if s not in taken]
    return free[0] if free else pool[0]


def seat_available(world, seat: Tuple[int, int], occupied: set) -> bool:
    """True if `seat` is a chair an agent can still take: nobody sitting on it,
    and at least one free walkable cell beside it to step in from.

    The approach test matters for chairs boxed in by furniture — cafe's (7,1) has
    tables above and below and the wall to its left, so (7,2) is the only way in.
    An agent that parks there leaves the chair free but unreachable for the rest
    of the episode, and a subject holding out for it burns the whole run.
    """
    if seat in occupied:
        return False
    r, c = seat
    return any(world.is_walkable(n) and n not in occupied
               for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))


def _claimed_first_targets(world) -> set:
    """First-goal cells already spoken for by agents still on the map. Mirrors
    the claim set src.agents.agent_init._pick_first_target avoids, so a pinned
    seat does not collide with a chair another agent is already heading to."""
    return {
        tuple(a.target_positions[0])
        for a in getattr(world, 'agents', [])
        if not getattr(a, 'done', True) and getattr(a, 'target_positions', None)
    }


def subject_test_seat_placeable(norm: Optional[str], world) -> bool:
    """True if subject_test_seat can place the seat for every subject color in
    `world` (vacuously true for the norms that pin no seat).

    A layout with all its chairs on one side leaves some placement impossible —
    post_office has no chair in the left half, so the subject cannot be seated
    beside the forbidden left door. Callers warn on that: the run still works,
    the subject just keeps its randomly drawn chair and the trip to its door is
    whatever the draw makes it.
    """
    by_world = SUBJECT_TEST_PLACEMENT_BY_WORLD.get((norm, type(world).__name__))
    if by_world is not None:
        # Layout-pinned placements name the quadrants outright and so are the
        # same for every color; one check covers them all. Quadrants of None pin
        # no seat at all, so there is nothing to place.
        if by_world[1] is None:
            return True
        return subject_test_seat(norm, world) is not None
    if norm not in SUBJECT_TEST_SEAT_PLACEMENT:
        return True
    classes = (DOOR_BY_CLASS_BY_NORM[norm]
               if SUBJECT_TEST_SEAT_PLACEMENT[norm][0] is None else [None])
    return all(subject_test_seat(norm, world, c) is not None for c in classes)


def pick_exit_door(norm: Optional[str],
                   agent_class: Optional[str],
                   world,
                   from_pos: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
    """Return the door (entrance cell) this agent leaves through under `norm`.

    - occupation_door_all_exit_{A,B,C} : the door on the norm's side
      (A = left, B = bottom, C = right).
    - occupation_door_{A,B,C}_forbidden : any door off the norm's side.
    - the class_specific norms : the door bound to the agent's class under the
      norm's mapping (DOOR_BY_CLASS_BY_NORM; the base norm is orange→left,
      green→bottom, blue→right).
    - any other norm, including None : every door qualifies. A norm-blind agent
      lands here, so it walks out of whichever door is closest — the shortest
      path, norm or no norm.

    `from_pos` is the cell the agent leaves from (its seat), and the nearest
    qualifying door to it wins, ties going to the world's entrance order. Without
    it one of the qualifying doors is drawn at random instead. The caller latches
    the result (the goal is reused across steps), so neither choice makes the
    agent oscillate between doors.
    """
    doors = list(getattr(world, 'entrances', []))
    if not doors:
        return None

    rule = DOOR_NORM_RULES.get(norm)
    if rule is not None:
        norm_side, mandatory = rule
        allowed = [d for d in doors
                   if (door_side(world, d) == norm_side) == mandatory]
    elif norm in DOOR_BY_CLASS_BY_NORM:
        want = DOOR_BY_CLASS_BY_NORM[norm].get(agent_class)
        allowed = [d for d in doors if door_side(world, d) == want]
    else:
        allowed = doors

    if not allowed:
        allowed = doors
    if from_pos is None:
        return random.choice(allowed)
    return min(allowed, key=lambda d: abs(d[0] - from_pos[0]) + abs(d[1] - from_pos[1]))


def pick_seat(norm: str,
              agent_class: Optional[str],
              all_current_positions: Dict[str, Tuple[int, int]],
              world,
              agent_id: str,
              claimed_targets: Optional[set] = None,
              agent_pos: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
    """Return a norm-compliant seat goal for this agent.

    - occupation_seat_class_specific : a chair in the agent's class quadrant
      (orange→top-left, green→bottom-left, blue→bottom-right).
    - the positional quadrant norms (SEAT_QUADRANT_RULES) : a chair inside the
      named quadrant ('only' rules) or outside it ('no' rules).
    - occupation_seat_in_order : the current fill-frontier seat (all agents aim
      at the same seat; claimed_targets is ignored for this norm).

    For the class/region norms the agent takes the nearest allowed seat: pass
    agent_pos to rank the compliant seats by Manhattan distance from it (fill
    order breaks ties; without agent_pos, plain fill order is kept). Seats
    already occupied (per all_current_positions) or claimed by other agents
    (claimed_targets) are skipped when possible, so an agent whose nearest seat
    is taken moves on to the next nearest. The choice is deterministic.
    """
    ordered = _ordered_seats(world)
    if not ordered:
        return None

    if is_seat_in_order_norm(norm):
        # Every agent aims at the same single seat: the fill frontier (in this
        # norm's fill order). Advancing it here (from the real current
        # positions) moves the goal on as soon as the target seat is taken and
        # keeps it there once a seat empties behind.
        ordered = _ordered_seats(world, norm)
        return ordered[advance_seat_in_order_frontier(all_current_positions, world, norm)]

    if agent_pos is not None:
        # Nearest-first: Manhattan distance from the agent, fill order as the
        # deterministic tie-break (ordered is already in fill order, and
        # Python's sort is stable).
        ordered = sorted(
            ordered,
            key=lambda s: abs(s[0] - agent_pos[0]) + abs(s[1] - agent_pos[1]),
        )

    taken = _occupied_seats(all_current_positions, world, exclude_id=agent_id)
    if claimed_targets:
        taken = taken | set(claimed_targets)

    quadrant_rule = SEAT_QUADRANT_RULES.get(norm)
    if norm in SEAT_REGION_BY_CLASS_BY_NORM:
        want = SEAT_REGION_BY_CLASS_BY_NORM[norm].get(agent_class)
        pool = [s for s in ordered if seat_region_of(world, s) == want]
    elif quadrant_rule is not None:
        quadrant, only = quadrant_rule
        pool = [s for s in ordered
                if (seat_region_of(world, s) == quadrant) == only]
    else:
        pool = ordered

    if not pool:
        return None

    free = [s for s in pool if s not in taken]
    return free[0] if free else pool[0]


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def check_occupation_norm(norm: str,
                          _from_pos: Tuple[int, int],
                          to_pos: Tuple[int, int],
                          all_current_positions: Dict[str, Tuple[int, int]],
                          world,
                          agent_id: str,
                          agent_classes: Dict[str, str],
                          target_position: Optional[Tuple[int, int]] = None) -> bool:
    """
    Check whether moving to `to_pos` (the agent's next position) violates the
    occupation norm. Returns True when the next position is a violation.

    Other agents' positions are read from all_current_positions (the per-step
    map of agent_id -> position), mirroring the navigation norms, so occupancy
    checks reflect every agent's predicted next position.

    Returns True (violation) when:
    - region norms : to_pos is a waiting-zone cell the agent may not enter right now
                     (capacity / single-class rules).
    - door norms   : to_pos is the door the agent is exiting through and that
                     door is forbidden for the agent (all-exit-left / left-
                     forbidden / class-specific rules).
    - seat norms   : to_pos is the chair the agent is sitting on and the chair
                     is not allowed for the agent (class / region / fill-order
                     rules).

    `agent_classes`    – maps agent_id -> color class, used for class-based rules.
    `target_position`  – the agent's current target; used to limit door/seat
                         checks to the cell the agent is actually claiming so
                         pass-through moves are not flagged.
    """
    r, c = to_pos
    if not (0 <= r < world.height and 0 <= c < world.width):
        return False

    agent_class = agent_classes.get(agent_id)

    # Region-entry checks (norm_029–031)
    if is_occupation_region_norm(norm):
        if to_pos in waiting_zone_cells(world):
            if not agent_may_enter_region(norm, agent_id, agent_classes, all_current_positions, world):
                return True
        return False

    # Door-exit checks (norm_032–038): only flag the cell the agent is exiting through.
    if is_occupation_door_norm(norm):
        is_door = to_pos in set(getattr(world, 'entrances', []))
        is_target = target_position is None or to_pos == target_position
        if is_door and is_target:
            if not door_exit_allowed(norm, agent_class, door_side(world, to_pos)):
                return True
        return False

    # Seat checks (norm_039–045): only flag the chair the agent is settling on.
    if is_occupation_seat_norm(norm):
        is_target = target_position is None or to_pos == target_position
        if world.is_sittable(to_pos) and is_target:
            if seat_violates(norm, to_pos, agent_id, agent_class, all_current_positions, world):
                return True
        return False

    return False
