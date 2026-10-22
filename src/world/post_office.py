from src.world.base_world import BaseWorld, CellType


class PostOfficeWorld(BaseWorld):
    """
    Post office: service window, sorting room, PO boxes, package shelves,
    scale stations, mailboxes.
    """

    WORLD_NAME = "POST OFFICE LAYOUT"

    ENTRANCES = [(4, 0), (10, 7), (2, 14)]

    
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
        ('counter', 'service_window'),
        ('sorting_room', 'sorting_room'),
        ('po_boxes', 'po_boxes'),
        ('package_shelves', 'package_shelves'),
        ('scale_stations', 'scale_stations'),
        ('mailboxes', 'mailboxes'),
    ]

    def _create_layout(self):
        self.grid.fill(CellType.WALL.value)
        self._add_rectangle(1, 1, 9, 13, CellType.EMPTY)

        for entrance in self.entrances:
            self.grid[entrance] = CellType.ENTRANCE.value

        # Service window on the top wall (col 8 — varied per world).
        self.grid[7, 1] = CellType.COUNTER.value
        self.grid[1, 1] = CellType.SCALE_STATION.value
        self.grid[1, 2] = CellType.SCALE_STATION.value
        self.grid[1, 3] = CellType.SCALE_STATION.value



        self.grid[1, 5] = CellType.MAILBOX.value
        self.grid[1, 6] = CellType.MAILBOX.value
        self.grid[1, 7] = CellType.MAILBOX.value
        self.grid[1, 8] = CellType.MAILBOX.value
        self.grid[4, 5] = CellType.MAILBOX.value
        self.grid[4, 6] = CellType.MAILBOX.value
        self.grid[4, 7] = CellType.MAILBOX.value
        self.grid[4, 8] = CellType.MAILBOX.value

        self.grid[3, 5] = CellType.FLOWER_POT.value
        self.grid[3, 6] = CellType.FLOWER_POT.value
        self.grid[3, 7] = CellType.FLOWER_POT.value
        self.grid[3, 8] = CellType.FLOWER_POT.value





        # self.grid[r, c] = CellType.PO_BOX.value

        self.grid[9, 9] = CellType.PACKAGE_SHELF.value
        self.grid[9, 10] = CellType.PACKAGE_SHELF.value
        self.grid[9, 11] = CellType.PACKAGE_SHELF.value
        self.grid[9, 12] = CellType.PACKAGE_SHELF.value
        self.grid[9, 13] = CellType.PACKAGE_SHELF.value

        # Waiting benches
        for c in (10, 11, 12, 13):
            self.grid[1, c] = CellType.CHAIR.value

        for c in (10, 11, 12):
            self.grid[3, c] = CellType.CHAIR.value
            self.grid[7, c] = CellType.CHAIR.value

        # Clear inward-adjacent cells so entrances aren't blocked
        self._clear_entrance_adjacent()
        # Waiting zone: plain floor, rendered as a footprint overlay; the
        # occupation-norm target region (see BaseWorld._identify_zones).
        self.WAITING_ZONE = {(2, 1), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)}
        # X cells: one near each entrance
        self.grid[8, 6] = CellType.AFFORDANCE_X.value
        self.grid[8, 7] = CellType.AFFORDANCE_X.value
        self.grid[8, 8] = CellType.AFFORDANCE_X.value
       
