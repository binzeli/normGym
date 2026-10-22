import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from src.env.actions import Action


class CellType(Enum):
    """Types of cells in the grid world"""
    # Shared types
    EMPTY = 0
    WALL = 1
    ENTRANCE = 2
    COUNTER = 3
    QUEUE_ZONE = 4

    # Cafe
    PICKUP_SHELF = 5
    CHAIR = 6
    TABLE_2TOP = 7
    TABLE_4TOP = 8
    KITCHEN = 10
    LAPTOP_BAR_COUNTER = 11

    # Movie Theater
    SCREEN_ROOM = 12
    ROPE_BARRIER = 13
    THEATER_SEAT = 14
    THEATER_SEAT_D = 65
    CONCESSION_STAND = 15
    SNACK_STAND = 16

    # Bank
    VAULT = 17
    ATM_ROOM = 18
    ROPE_STANCHION = 19
    WRITING_DESK = 20
    DEPOSIT_SLIPS = 21
    WAITING_CHAIR = 22

    # Pharmacy
    STOCKROOM = 23
    MEDICINE_SHELF = 24
    DISPLAY_CASE = 25
    BP_STATION = 26
    RX_PICKUP = 27

    # Grocery Store
    STORAGE_ROOM = 28
    PRODUCT_SHELF = 29
    CART_CORRAL = 30
    BAGGING_AREA = 31

    # Cafe Barista Station
    PASTRY_DISPLAY = 32
    COFFEE_GRINDER = 33
    MENU_BOARD = 34
    ESPRESSO_MACHINE = 35

    # Shared decorations
    FLOWER_POT = 36

    # Clothing Store
    CLOTHING_STOCKROOM = 37
    CLOTHING_RACK = 38
    FITTING_ROOM = 39
    MANNEQUIN = 40
    SHOE_DISPLAY = 41

    # Library
    BOOKSHELF = 42
    READING_DESK = 43
    COMPUTER_STATION = 44
    RETURN_DROP = 45
    STUDY_ROOM = 46

    # Post Office
    MAILBOX = 47
    PACKAGE_SHELF = 48
    SORTING_ROOM = 49
    PO_BOX = 50
    SCALE_STATION = 51

    # Hospital
    NURSES_STATION = 52
    HOSPITAL_BED = 53
    WAITING_BENCH = 54
    EXAM_ROOM = 55
    TRIAGE_AREA = 56

    # Barbershop
    BARBER_CHAIR = 57
    WAITING_COUCH_R = 58
    WAITING_COUCH_L = 59 

    MIRROR_STATION = 60
    SUPPLY_CABINET = 61
    MAGAZINE_RACK = 62

    # Affordance cells (shared across all worlds)
    AFFORDANCE_X = 64

    # Color-tagged chairs (visual color + semantic tag for color-priority norms).
    # Use these so norm code can do `if grid[r,c] == CellType.RED_CHAIR.value: ...`.
    RED_CHAIR = 66
    ORANGE_CHAIR = 67
    BLUE_CHAIR = 68
    GREEN_CHAIR = 69


# Cell types that agents can walk on
WALKABLE_TYPES = {CellType.EMPTY.value, CellType.ENTRANCE.value, CellType.AFFORDANCE_X.value}

# Mapping from CellType to zone key for _identify_zones
CELL_TO_ZONE = {
    CellType.ENTRANCE.value: 'entrance',
    CellType.COUNTER.value: 'counter', 
    CellType.QUEUE_ZONE.value: 'queue',
    # Cafe
    CellType.PICKUP_SHELF.value: 'pickup_shelf',
    CellType.LAPTOP_BAR_COUNTER.value: 'laptop_bar_counter',
    CellType.CHAIR.value: 'chairs',
    CellType.TABLE_2TOP.value: 'tables_2top',
    CellType.TABLE_4TOP.value: 'tables_4top',
    CellType.KITCHEN.value: 'kitchen',
    # Movie Theater
    CellType.SCREEN_ROOM.value: 'screen_room',
    CellType.ROPE_BARRIER.value: 'rope_barriers',
    CellType.THEATER_SEAT.value: 'theater_seats',
    CellType.THEATER_SEAT_D.value: 'theater_seats_d',
    CellType.CONCESSION_STAND.value: 'concession_stand',
    CellType.SNACK_STAND.value: 'snack_stand',
    # Bank
    CellType.VAULT.value: 'vault',
    CellType.ATM_ROOM.value: 'atm_room',
    CellType.ROPE_STANCHION.value: 'rope_stanchions',
    CellType.WRITING_DESK.value: 'writing_desks',
    CellType.DEPOSIT_SLIPS.value: 'deposit_slips',
    CellType.WAITING_CHAIR.value: 'waiting_chairs',
    # Pharmacy
    CellType.STOCKROOM.value: 'stockroom',
    CellType.MEDICINE_SHELF.value: 'medicine_shelves',
    CellType.DISPLAY_CASE.value: 'display_case',
    CellType.BP_STATION.value: 'bp_stations',
    CellType.RX_PICKUP.value: 'rx_pickup',
    # Grocery
    CellType.STORAGE_ROOM.value: 'storage_room',
    CellType.PRODUCT_SHELF.value: 'product_shelves',
    CellType.CART_CORRAL.value: 'cart_corrals',
    CellType.BAGGING_AREA.value: 'bagging_area',
    # Cafe Barista Station
    CellType.PASTRY_DISPLAY.value: 'pastry_display',
    CellType.COFFEE_GRINDER.value: 'coffee_grinder',
    CellType.MENU_BOARD.value: 'menu_board',
    CellType.ESPRESSO_MACHINE.value: 'espresso_machine',
    # Shared decorations
    CellType.FLOWER_POT.value: 'flower_pots',
    # Clothing Store
    CellType.CLOTHING_STOCKROOM.value: 'clothing_stockroom',
    CellType.CLOTHING_RACK.value: 'clothing_racks',
    CellType.FITTING_ROOM.value: 'fitting_rooms',
    CellType.MANNEQUIN.value: 'mannequins',
    CellType.SHOE_DISPLAY.value: 'shoe_displays',
    # Library
    CellType.BOOKSHELF.value: 'bookshelves',
    CellType.READING_DESK.value: 'reading_desks',
    CellType.COMPUTER_STATION.value: 'computer_stations',
    CellType.RETURN_DROP.value: 'return_drop',
    CellType.STUDY_ROOM.value: 'study_room',
    # Post Office
    CellType.MAILBOX.value: 'mailboxes',
    CellType.PACKAGE_SHELF.value: 'package_shelves',
    CellType.SORTING_ROOM.value: 'sorting_room',
    CellType.PO_BOX.value: 'po_boxes',
    CellType.SCALE_STATION.value: 'scale_stations',
    # Hospital
    CellType.NURSES_STATION.value: 'nurses_station',
    CellType.HOSPITAL_BED.value: 'hospital_beds',
    CellType.WAITING_BENCH.value: 'waiting_benches',
    CellType.EXAM_ROOM.value: 'exam_room',
    CellType.TRIAGE_AREA.value: 'triage_area',
    # Barbershop
    CellType.BARBER_CHAIR.value: 'barber_chairs',
    CellType.WAITING_COUCH_R.value: 'waiting_couch_right',
    CellType.WAITING_COUCH_L.value: 'waiting_couch_left',
    CellType.MIRROR_STATION.value: 'mirror_stations',
    CellType.SUPPLY_CABINET.value: 'supply_cabinet',
    CellType.MAGAZINE_RACK.value: 'magazine_racks',
    # Affordance cells
    CellType.AFFORDANCE_X.value: 'affordance_x',
    # Color-tagged chairs (also added to 'chairs' generically in _identify_zones)
    CellType.RED_CHAIR.value: 'red_chairs',
    CellType.ORANGE_CHAIR.value: 'orange_chairs',
    CellType.BLUE_CHAIR.value: 'blue_chairs',
    CellType.GREEN_CHAIR.value: 'green_chairs',
}

# Set of cell values treated as a chair (sittable, colored, etc.)
CHAIR_LIKE = {
    CellType.CHAIR.value,
    CellType.RED_CHAIR.value,
    CellType.ORANGE_CHAIR.value,
    CellType.BLUE_CHAIR.value,
    CellType.GREEN_CHAIR.value,
}

# Color-tagged chair palette used for random assignment and rendering tints.
COLORED_CHAIRS = (
    CellType.RED_CHAIR.value,
    CellType.ORANGE_CHAIR.value,
    CellType.BLUE_CHAIR.value,
)

CHAIR_COLOR_RGB = {
    CellType.RED_CHAIR.value: (220, 60, 60),
    CellType.ORANGE_CHAIR.value: (245, 150, 50),
    CellType.BLUE_CHAIR.value: (60, 130, 220),
    # Same darkness/saturation profile as the others (low secondary channels).
    CellType.GREEN_CHAIR.value: (70, 185, 80),
}

# Print symbols for each cell type
PRINT_SYMBOLS = {
    CellType.EMPTY.value: '·',
    CellType.WALL.value: '█',
    CellType.ENTRANCE.value: 'E',
    CellType.COUNTER.value: 'C',
    CellType.QUEUE_ZONE.value: 'Q',
    # Cafe
    CellType.PICKUP_SHELF.value: 'P',
    CellType.LAPTOP_BAR_COUNTER.value: 'L',
    CellType.CHAIR.value: 'h',
    CellType.TABLE_2TOP.value: '▪',
    CellType.TABLE_4TOP.value: '▪',
    CellType.KITCHEN.value: 'K',
    # Movie Theater
    CellType.SCREEN_ROOM.value: 'S',
    CellType.ROPE_BARRIER.value: 'R',
    CellType.THEATER_SEAT.value: 's',
    CellType.THEATER_SEAT_D.value: 'sd',
    CellType.CONCESSION_STAND.value: 'F',
    CellType.SNACK_STAND.value: 'N',
    # Bank
    CellType.VAULT.value: 'V',
    CellType.ATM_ROOM.value: 'A',
    CellType.ROPE_STANCHION.value: 'R',
    CellType.WRITING_DESK.value: 'D',
    CellType.DEPOSIT_SLIPS.value: 'P',
    CellType.WAITING_CHAIR.value: 'h',
    # Pharmacy
    CellType.STOCKROOM.value: 'K',
    CellType.MEDICINE_SHELF.value: 'M',
    CellType.DISPLAY_CASE.value: 'G',
    CellType.BP_STATION.value: 'B',
    CellType.RX_PICKUP.value: 'P',
    # Grocery
    CellType.STORAGE_ROOM.value: 'K',
    CellType.PRODUCT_SHELF.value: 'M',
    CellType.CART_CORRAL.value: 'R',
    CellType.BAGGING_AREA.value: 'P',
    # Cafe Barista Station
    CellType.PASTRY_DISPLAY.value: 'Y',
    CellType.COFFEE_GRINDER.value: 'G',
    CellType.MENU_BOARD.value: 'M',
    CellType.ESPRESSO_MACHINE.value: 'X',
    # Shared decorations
    CellType.FLOWER_POT.value: '@',
    # Clothing Store
    CellType.CLOTHING_STOCKROOM.value: 'S',
    CellType.CLOTHING_RACK.value: 'R',
    CellType.FITTING_ROOM.value: 'F',
    CellType.MANNEQUIN.value: 'M',
    CellType.SHOE_DISPLAY.value: 'H',
    # Library
    CellType.BOOKSHELF.value: 'B',
    CellType.READING_DESK.value: 'D',
    CellType.COMPUTER_STATION.value: 'O',
    CellType.RETURN_DROP.value: 'R',
    CellType.STUDY_ROOM.value: 'S',
    # Post Office
    CellType.MAILBOX.value: 'M',
    CellType.PACKAGE_SHELF.value: 'P',
    CellType.SORTING_ROOM.value: 'S',
    CellType.PO_BOX.value: 'B',
    CellType.SCALE_STATION.value: 'W',
    # Hospital
    CellType.NURSES_STATION.value: 'N',
    CellType.HOSPITAL_BED.value: 'B',
    CellType.WAITING_BENCH.value: 'w',
    CellType.EXAM_ROOM.value: 'X',
    CellType.TRIAGE_AREA.value: 'T',
    # Barbershop
    CellType.BARBER_CHAIR.value: 'b',
    CellType.WAITING_COUCH_R.value: 'wr',
    CellType.WAITING_COUCH_L.value: 'wl',
    CellType.MIRROR_STATION.value: 'M',
    CellType.SUPPLY_CABINET.value: 'S',
    CellType.MAGAZINE_RACK.value: 'R',
    # Affordance cells
    CellType.AFFORDANCE_X.value: 'X',
    # Color-tagged chairs
    CellType.RED_CHAIR.value: 'hR',
    CellType.ORANGE_CHAIR.value: 'hO',
    CellType.BLUE_CHAIR.value: 'hB',
    CellType.GREEN_CHAIR.value: 'hG',
}


class BaseWorld:
    """
    Base class for grid-based world environments.
    Subclasses implement _create_layout() to define the specific world layout.
    """

    WORLD_NAME = "World"

    # Entrance cell positions for this world. Subclasses override this class
    # attribute to place entrances at different coordinates (e.g. a world with
    # a different-shaped layout). Defaults to the standard bottom/left/right
    # entrance set shared by most worlds.
    ENTRANCES: List[Tuple[int, int]] = [(5, 0), (10, 7), (5, 14)]

    # Subclasses override this to define which zone keys to include in object positions text
    OBJECT_POSITION_ZONES: List[Tuple[str, str]] = [
        ('entrance', 'entrances'),
        ('counter', 'counter'),
        ('laptop_bar_counter', 'laptop_bar_counter'),
        ('chairs', 'chairs'),
        ('tables_2top', 'tables_2top'),
        ('tables_4top', 'tables_4top'),
        ('kitchen', 'kitchen'),
    ]

    def __init__(self, width: int = 15, height: int = 11,
                 map_type: str = "default"):
        self.width = width
        self.height = height
        self.map_type = map_type
        self.grid = np.zeros((height, width), dtype=int)

        self.entrances = self._default_entrances()

        self.zones: Dict[str, List[Tuple[int, int]]] = {}
        self.walkable_cells: Set[Tuple[int, int]] = set()
        # Positions that carry X-cell affordance functionality overlaid on their
        # underlying cell type (e.g. a CHAIR can be an X cell while still rendering
        # as a chair).  Populated by subclasses via add_x_overlay().
        self.x_overlay: Set[Tuple[int, int]] = set()

        self._create_layout()
        self._identify_zones()

        self.previous_queue = {}
        self.current_queue = {}
        # Set by the env once the norm is known; is_at_counter needs it because
        # the edge-line norms are served from a different cell.
        self.norm_type: Optional[str] = None
        # Fill-frontier index for the seat-in-order norms (occupation_seat_in_order
        # and its fill-order variants): the single seat all agents currently aim
        # at. Forward-only; advanced during goal selection.
        self.seat_in_order_frontier = 0


    def _default_entrances(self) -> List[Tuple[int, int]]:
        return list(self.ENTRANCES)

    def _create_layout(self):
        raise NotImplementedError("Subclasses must implement _create_layout()")

    def _clear_entrance_adjacent(self):
        """Clear the cell immediately inward from each entrance so that a
        world's entrances (whether the default set or a per-world override of
        ENTRANCES) never end up walled in by neighboring furniture.

        Assumes entrances sit on the grid's boundary; the inward direction is
        derived from which wall the entrance is on.
        """
        for (r, c) in self.entrances:
            if r == 0:
                inward = (r + 1, c)
            elif r == self.height - 1:
                inward = (r - 1, c)
            elif c == 0:
                inward = (r, c + 1)
            elif c == self.width - 1:
                inward = (r, c - 1)
            else:
                continue
            if 0 <= inward[0] < self.height and 0 <= inward[1] < self.width:
                self.grid[inward] = CellType.EMPTY.value


    def transition(self, current_position: Tuple[int, int], action) -> Optional[Tuple[int, int]]:
        r, c = current_position

        if action == Action.MOVE_UP:
            next_pos = (r - 1, c)
        elif action == Action.MOVE_DOWN:
            next_pos = (r + 1, c)
        elif action == Action.MOVE_LEFT:
            next_pos = (r, c - 1)
        elif action == Action.MOVE_RIGHT:
            next_pos = (r, c + 1)
        else:
            return current_position

        if self.is_walkable(next_pos) or self.is_sittable(next_pos):
            return next_pos
        else:
            return current_position

    def is_at_counter(self, position: Tuple[int, int]) -> bool:
        counter_zone = self.zones.get('counter')
        if not counter_zone:
            return False
        # The service cell depends on the norm: edge lines are served from the
        # counter's side cell, every other norm from the cell in front of it.
        from src.norms.lining import counter_service_cell
        return position == counter_service_cell(self, getattr(self, 'norm_type', None))

    def _add_rectangle(self, start_row: int, start_col: int,
                       height: int, width: int, cell_type: CellType):
        for r in range(start_row, start_row + height):
            for c in range(start_col, start_col + width):
                if 0 <= r < self.height and 0 <= c < self.width:
                    self.grid[r, c] = cell_type.value

    def _add_2top_with_chairs(self, row: int, col: int):
        self.grid[row, col] = CellType.TABLE_2TOP.value
        self.grid[row - 1, col] = CellType.CHAIR.value
        self.grid[row + 1, col] = CellType.CHAIR.value

    def _add_4top_with_chairs(self, row: int, col: int):
        for r, c in [(row, col), (row, col + 1), (row + 1, col), (row + 1, col + 1)]:
            self.grid[r, c] = CellType.TABLE_4TOP.value
        for r, c in [(row - 1, col), (row - 1, col + 1), (row + 2, col), (row + 2, col + 1)]:
            if 0 <= r < self.height and 0 <= c < self.width:
                self.grid[r, c] = CellType.CHAIR.value

    def add_x_overlay(self, pos: Tuple[int, int]):
        """Mark a cell as an X-cell overlay without replacing its underlying CellType.

        The cell gains X-cell affordance functionality (norm checks treat it as
        AFFORDANCE_X) while its sprite/icon is drawn below the cell's normal sprite.
        The cell is also made walkable so agents can visit it.

        Call this from _apply_affordance_layout() (before _identify_zones runs) or
        after construction — both are handled correctly.
        """
        self.x_overlay.add(pos)
        # Eagerly update walkable/zones in case _identify_zones has already run
        # (e.g. called post-construction). _identify_zones also processes x_overlay
        # so calling this pre-construction is safe too.
        self.walkable_cells.add(pos)
        self.zones.setdefault('affordance_x', [])
        if pos not in self.zones['affordance_x']:
            self.zones['affordance_x'].append(pos)

    def _identify_zones(self):
        self.walkable_cells = set()
        self.zones = {}
        for r in range(self.height):
            for c in range(self.width):
                cell = self.grid[r, c]
                pos = (r, c)

                if cell in WALKABLE_TYPES:
                    self.walkable_cells.add(pos)

                zone_key = CELL_TO_ZONE.get(cell)
                if zone_key:
                    self.zones.setdefault(zone_key, []).append(pos)
                    # Color-tagged chairs also count as generic 'chairs' so any
                    # code that iterates the chairs zone keeps working.
                    if cell in CHAIR_LIKE and zone_key != 'chairs':
                        self.zones.setdefault('chairs', []).append(pos)
                elif cell == CellType.EMPTY.value:
                    self.zones.setdefault('open_space', []).append(pos)

        # x_overlay cells are always walkable (may overlay non-walkable types)
        for pos in self.x_overlay:
            self.walkable_cells.add(pos)
            self.zones.setdefault('affordance_x', [])
            if pos not in self.zones['affordance_x']:
                self.zones['affordance_x'].append(pos)

        # Waiting zone: cells declared in the subclass's WAITING_ZONE attribute.
        # They stay plain floor (no solid tile); the renderer draws the dashed
        # border + footprint overlay, and occupation norms target them.
        for pos in getattr(self, 'WAITING_ZONE', ()):
            self.zones.setdefault('waiting_zone', [])
            if pos not in self.zones['waiting_zone']:
                self.zones['waiting_zone'].append(pos)
            self.walkable_cells.add(pos)

    def randomize_chair_colors(self, rng=None):
        """Replace every CHAIR cell with a random RED/ORANGE/BLUE_CHAIR.

        Called by the env on reset using its seeded RNG so chair coloring is
        reproducible per simulation seed. Re-runs _identify_zones so the
        'chairs' / 'red_chairs' / etc. zones reflect the new assignment.

        `rng` may be a numpy Generator/RandomState or None (uses numpy global).
        """
        import numpy as np
        palette = list(COLORED_CHAIRS)
        for r in range(self.height):
            for c in range(self.width):
                if int(self.grid[r, c]) == CellType.CHAIR.value:
                    if rng is None:
                        choice = palette[np.random.randint(len(palette))]
                    else:
                        choice = palette[int(rng.integers(len(palette)))
                                         if hasattr(rng, 'integers')
                                         else int(rng.randint(len(palette)))]
                    self.grid[r, c] = choice
        self._identify_zones()

    def is_walkable(self, pos):
        r, c = pos
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        return pos in self.walkable_cells

    def is_sittable(self, pos):
        r, c = pos
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        return self.grid[r, c] in (
            CellType.CHAIR.value,
            CellType.RED_CHAIR.value,
            CellType.ORANGE_CHAIR.value,
            CellType.BLUE_CHAIR.value,
            CellType.GREEN_CHAIR.value,
            CellType.THEATER_SEAT.value,
            CellType.THEATER_SEAT_D.value,
            CellType.WAITING_CHAIR.value,
        )

    def get_neighbors(self, pos):
        r, c = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (r + dr, c + dc)
            if self.is_walkable(new_pos):
                neighbors.append(new_pos)
        return neighbors

    def get_cell_type(self, pos):
        r, c = pos
        if not (0 <= r < self.height and 0 <= c < self.width):
            return None
        return CellType(self.grid[r, c])

    def print_layout(self):
        print("\n" + "=" * (self.width * 2 + 2))
        print(" " * ((self.width * 2 - len(self.WORLD_NAME)) // 2) + self.WORLD_NAME)
        print("=" * (self.width * 2 + 2))

        for r in range(self.height):
            row_str = ""
            for c in range(self.width):
                row_str += PRINT_SYMBOLS.get(self.grid[r, c], '?') + " "
            print(row_str)
        print("=" * (self.width * 2 + 2) + "\n")

    def build_object_positions_text(self) -> str:
        lines: List[str] = []
        for zone_key, label in self.OBJECT_POSITION_ZONES:
            positions = sorted(self.zones.get(zone_key, []))
            if positions:
                lines.append(f"- {label}: {positions}")
        lines.extend(self._neutral_zone_position_lines())
        return "\n".join(lines)

    def _neutral_zone_position_lines(self) -> List[str]:
        """Object-positions lines for the two norm-bearing regions.

        The X cells are surfaced as 'zone Q' and the waiting zone as 'zone R'.
        Both labels are deliberately meaningless: a real-world name (a carpet, a
        waiting area) makes "do not step on it" or "only one agent at a time"
        guessable from common sense instead of from the agent's observations,
        which is what these episodes are meant to test.

        Every world defines both regions, so these lines are appended for all
        maps regardless of each world's OBJECT_POSITION_ZONES. They are also
        emitted whatever norm is active, so their presence leaks nothing about
        which norm the group is following.

        Both are marked '(walkable)': the cells are plain floor as far as the
        physics goes — an X cell stays enterable even when the active norm
        forbids it — so the agent should not read them as obstacles. That says
        nothing about whether entering them is permitted, which is still left
        for the agent to infer.
        """
        lines: List[str] = []
        for zone_key, label in (
            ('affordance_x', 'zone Q (walkable)'),
            ('waiting_zone', 'zone R (walkable)'),
        ):
            positions = sorted(self.zones.get(zone_key, []))
            if positions:
                lines.append(f"- {label}: {positions}")
        return lines

    @property
    def waiting_zone_cells(self) -> List[Tuple[int, int]]:
        """Return the cells of the waiting zone (the occupation-norm target region)."""
        return self.zones.get('waiting_zone', [])
