from src.world.base_world import BaseWorld, CellType


class PharmacyWorld(BaseWorld):
    """
    Pharmacy: Rx counter, ready-prescriptions pickup, stockroom,
    medicine shelf aisles, display case, blood pressure stations.
    """

    WORLD_NAME = "PHARMACY LAYOUT"

    ENTRANCES = [(5, 0), (10, 4), (4, 14)]

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
        ('counter', 'rx_counter'),
        ('rx_pickup', 'ready_prescriptions'),
        ('stockroom', 'stockroom'),
        ('medicine_shelves', 'medicine_shelves'),
        ('display_case', 'display_case'),
        ('bp_stations', 'bp_stations'),
    ]

    def _create_layout(self):
        self.grid.fill(CellType.WALL.value)
        self._add_rectangle(1, 1, 9, 13, CellType.EMPTY)

        for entrance in self.entrances:
            self.grid[entrance] = CellType.ENTRANCE.value

        # Rx counter on the top wall (col 7 — varied per world).
        self.grid[7, 13] = CellType.COUNTER.value

        self.grid[1, 1] = CellType.MEDICINE_SHELF.value
        self.grid[1, 2] = CellType.MEDICINE_SHELF.value
        self.grid[1, 3] = CellType.MEDICINE_SHELF.value

        self.grid[2, 1] = CellType.RX_PICKUP.value
        self.grid[2, 2] = CellType.DISPLAY_CASE.value
        self.grid[2, 3] = CellType.BP_STATION.value



        self._add_rectangle(1, 10, 2, 4, CellType.STOCKROOM)


        self.grid[7, 4] = CellType.MEDICINE_SHELF.value
        self.grid[7, 5] = CellType.MEDICINE_SHELF.value
        self.grid[7, 6] = CellType.MEDICINE_SHELF.value
        self.grid[6, 4] = CellType.MEDICINE_SHELF.value
        self.grid[6, 5] = CellType.MEDICINE_SHELF.value
        self.grid[6, 6] = CellType.MEDICINE_SHELF.value

        self.grid[2, 5] = CellType.CHAIR.value
        self.grid[2, 6] = CellType.CHAIR.value
        self.grid[3, 5] = CellType.CHAIR.value
        self.grid[2, 7] = CellType.CHAIR.value
        self.grid[2, 8] = CellType.CHAIR.value
        self.grid[3, 8] = CellType.CHAIR.value
        self.grid[9, 1] = CellType.CHAIR.value
        self.grid[9, 2] = CellType.CHAIR.value
        self.grid[9, 3] = CellType.CHAIR.value

        # Clear inward-adjacent cells so entrances aren't blocked
        self._clear_entrance_adjacent()
        # Waiting zone: plain floor, rendered as a footprint overlay; the
        # occupation-norm target region (see BaseWorld._identify_zones).
        self.WAITING_ZONE = {(3, 1), (4, 1), (3, 2), (4, 2), (3, 3), (4, 3)}
        # X cells: one near each entrance
        self.grid[3, 13] = CellType.AFFORDANCE_X.value
        self.grid[9, 8] = CellType.AFFORDANCE_X.value


