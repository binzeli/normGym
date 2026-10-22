from src.world.base_world import BaseWorld, CellType


class BankWorld(BaseWorld):
    """
    Bank branch: teller window, deposit slips counter, ATM room, vault,
    writing desks, waiting chairs, rope stanchions.
    """

    WORLD_NAME = "BANK LAYOUT"
    
    ENTRANCES = [(5, 0), (10, 7), (5, 14)]


    ISO_WINDOW_CELLS = {
        (0, 14), (1, 14), (2, 14), (3, 14), (4, 14), (6, 14), (7, 14),
        (8, 14), (9, 14), (10, 14),                          # right wall
        (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6),
        (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13),  # bottom wall
    }
    # Pendant lamps hang over the back row, so no agent can stand behind
    # one (a lamp is drawn on top and would cover them).
    ISO_LAMPS = [
        (1, 3),
        (1, 6),
        (1, 9),
        (1, 12),
    ]

    OBJECT_POSITION_ZONES = [
        ('entrance', 'entrances'),
        ('counter', 'teller_window'),
        ('deposit_slips', 'deposit_slips_counter'),
        ('atm_room', 'atm_room'),
        ('vault', 'vault'),
        ('writing_desks', 'writing_desks'),
        ('waiting_chairs', 'waiting_chairs'),
        ('rope_stanchions', 'rope_stanchions'),
    ]

    def _create_layout(self):
        self.grid.fill(CellType.WALL.value)
        self._add_rectangle(1, 1, 9, 13, CellType.EMPTY)

        for entrance in self.entrances:
            self.grid[entrance] = CellType.ENTRANCE.value

        # Teller window on the top wall (col 8 — varied per world).
        self.grid[1, 5] = CellType.COUNTER.value

        self._add_rectangle(1, 10, 2, 4, CellType.VAULT)

        self.grid[9, 9] = CellType.ROPE_STANCHION.value
        self.grid[9, 13] = CellType.ROPE_STANCHION.value

        self.grid[9, 10] = CellType.ATM_ROOM.value
        self.grid[9, 11] = CellType.ATM_ROOM.value
        self.grid[9, 12] = CellType.ATM_ROOM.value

        # Writing-desk waiting chairs (left)
        self.grid[7, 1] = CellType.CHAIR.value
        self.grid[8, 1] = CellType.CHAIR.value

        self.grid[6, 5] = CellType.WRITING_DESK.value
        self.grid[6, 6] = CellType.WRITING_DESK.value
        self.grid[6, 7] = CellType.WRITING_DESK.value

        self.grid[8, 4] = CellType.CHAIR.value
        self.grid[8, 5] = CellType.CHAIR.value

        self.grid[9, 1] = CellType.FLOWER_POT.value

        # Waiting area (right)
        self.grid[1, 1] = CellType.CHAIR.value
        self.grid[1, 2] = CellType.CHAIR.value


        self.grid[7, 13] = CellType.CHAIR.value
        self.grid[8, 13] = CellType.CHAIR.value



        # Clear inward-adjacent cells so entrances aren't blocked
        self._clear_entrance_adjacent()
        # Waiting zone: plain floor, rendered as a footprint overlay; the
        # occupation-norm target region (see BaseWorld._identify_zones).
        self.WAITING_ZONE = {(7, 10), (7, 11), (7, 12), (8, 10), (8, 11), (8, 12)}
        # X cells: one near each entrance

        self.grid[7, 8] = CellType.AFFORDANCE_X.value
        self.grid[4, 13] = CellType.AFFORDANCE_X.value

