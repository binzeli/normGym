from src.world.base_world import BaseWorld, CellType


class ClothingStoreWorld(BaseWorld):
    """
    Clothing Store: checkout counter, stockroom, clothing racks,
    fitting rooms, mannequins, shoe displays.
    """

    WORLD_NAME = "CLOTHING STORE LAYOUT"

    ENTRANCES = [(5, 0), (10, 6), (3, 14)]

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
        ('counter', 'checkout'),
        ('clothing_stockroom', 'stockroom'),
        ('clothing_racks', 'clothing_racks'),
        ('fitting_rooms', 'fitting_rooms'),
        ('mannequins', 'mannequins'),
        ('shoe_displays', 'shoe_displays'),
    ]

    def _create_layout(self):
        self.grid.fill(CellType.WALL.value)
        self._add_rectangle(1, 1, 9, 13, CellType.EMPTY)

        for entrance in self.entrances:
            self.grid[entrance] = CellType.ENTRANCE.value

        # Checkout counter on the top wall (col 5 — varied per world).
        self.grid[1, 6] = CellType.COUNTER.value
        self.grid[1, 1] = CellType.FITTING_ROOM.value
        self.grid[1, 2] = CellType.MANNEQUIN.value
        self.grid[1, 3] = CellType.FITTING_ROOM.value
        self.grid[1, 4] = CellType.MANNEQUIN.value

        self.grid[9, 1] = CellType.SHOE_DISPLAY.value
        self.grid[9, 2] = CellType.SHOE_DISPLAY.value
        self.grid[9, 3] = CellType.SHOE_DISPLAY.value
        self.grid[9, 4] = CellType.SHOE_DISPLAY.value


        self._add_rectangle(1, 10, 2, 4, CellType.CLOTHING_STOCKROOM)

        self.grid[9, 13] = CellType.CLOTHING_RACK.value
        self.grid[7, 1] = CellType.CLOTHING_RACK.value
        self.grid[7, 3] = CellType.CLOTHING_RACK.value
        self.grid[7, 4] = CellType.CLOTHING_RACK.value

        self.grid[9, 9] = CellType.CLOTHING_RACK.value
        self.grid[9, 10] = CellType.CLOTHING_RACK.value
        self.grid[9, 11] = CellType.CLOTHING_RACK.value
        self.grid[9, 12] = CellType.CLOTHING_RACK.value

        self.grid[7, 9] = CellType.CLOTHING_RACK.value
        self.grid[7, 10] = CellType.CLOTHING_RACK.value
        self.grid[7, 11] = CellType.CLOTHING_RACK.value
        self.grid[7, 12] = CellType.CLOTHING_RACK.value


        # Fitting-area seating: stool clusters


        self.grid[7, 2] = CellType.CHAIR.value
        self.grid[7, 3] = CellType.CHAIR.value
        self.grid[7, 4] = CellType.CHAIR.value

        self.grid[5, 12] = CellType.CHAIR.value
        self.grid[6, 12] = CellType.CHAIR.value
        self.grid[4, 11] = CellType.CHAIR.value
        self.grid[4, 12] = CellType.CHAIR.value

        # Clear inward-adjacent cells so entrances aren't blocked
        self._clear_entrance_adjacent()
        # Waiting zone: plain floor, rendered as a footprint overlay; the
        # occupation-norm target region (see BaseWorld._identify_zones).
        self.WAITING_ZONE = {(2, 1), (3, 1), (2, 3), (3, 3)}
        # X cells: one near each entrance
        self.grid[7, 1] = CellType.AFFORDANCE_X.value
        self.grid[7, 13] = CellType.AFFORDANCE_X.value
        self.grid[8, 8] = CellType.AFFORDANCE_X.value

