from src.world.base_world import BaseWorld, CellType


class BarbershopWorld(BaseWorld):
    """
    Barbershop: reception counter, supply cabinet, barber chairs,
    mirror stations, waiting couches, magazine racks.
    """

    WORLD_NAME = "BARBERSHOP LAYOUT"

    ENTRANCES = [(6, 0), (10, 4), (7, 14)]

    ISO_WINDOW_CELLS = {
        (0, 14), (1, 14), (2, 14), (3, 14), (4, 14), (5, 14), (6, 14),
        (8, 14), (9, 14), (10, 14),                          # right wall
        (10, 0), (10, 1), (10, 2), (10, 3), (10, 5), (10, 6), (10, 7),
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
        ('counter', 'reception'),
        ('supply_cabinet', 'supply_cabinet'),
        ('barber_chairs', 'barber_chairs'),
        ('mirror_stations', 'mirror_stations'),
        ('waiting_couches', 'waiting_couches'),
        ('magazine_racks', 'magazine_racks'),
    ]

    def _create_layout(self):
        self.grid.fill(CellType.WALL.value)
        self._add_rectangle(1, 1, 9, 13, CellType.EMPTY)

        for entrance in self.entrances:
            self.grid[entrance] = CellType.ENTRANCE.value


        # Reception counter on the top wall (col 4 — varied from the default
        # center so worlds differ); queue corridor down col 4 stays clear.
        self.grid[3, 13] = CellType.COUNTER.value
        self.grid[1, 2] = CellType.MAGAZINE_RACK.value
        self.grid[9, 10] = CellType.MAGAZINE_RACK.value
        self.grid[9, 12] = CellType.MAGAZINE_RACK.value

        # Barber chairs, drawn from the barber-chair sprite.
        self.grid[1, 1] = CellType.MIRROR_STATION.value
        self.grid[2, 1] = CellType.CHAIR.value
        self.grid[1, 3] = CellType.MIRROR_STATION.value
        self.grid[2, 3] = CellType.CHAIR.value

        self.grid[8, 1] = CellType.CHAIR.value
        self.grid[8, 2] = CellType.CHAIR.value
        self.grid[9, 1] = CellType.MIRROR_STATION.value
        self.grid[9, 2] = CellType.MIRROR_STATION.value

        self.grid[9, 13] = CellType.MIRROR_STATION.value
        self.grid[8, 13] = CellType.CHAIR.value
        self.grid[9, 11] = CellType.MIRROR_STATION.value
        self.grid[8, 11] = CellType.CHAIR.value
        self.grid[9, 9] = CellType.MIRROR_STATION.value
        self.grid[8, 9] = CellType.CHAIR.value

        self.grid[4, 6] = CellType.WAITING_COUCH_R.value
        self.grid[4, 8] = CellType.WAITING_COUCH_R.value
        self.grid[4, 5] = CellType.WAITING_COUCH_L.value
        self.grid[4, 7] = CellType.WAITING_COUCH_L.value

        self.grid[1, 5] = CellType.SUPPLY_CABINET.value
        self.grid[1, 6] = CellType.SUPPLY_CABINET.value
        self.grid[1, 7] = CellType.SUPPLY_CABINET.value

        # Clear inward-adjacent cells so entrances aren't blocked
        self._clear_entrance_adjacent()
        # Waiting zone: plain floor, rendered as a footprint overlay; the
        # occupation-norm target region (see BaseWorld._identify_zones).
        self.WAITING_ZONE = {(3, 1), (4, 1), (3, 2), (3, 3), (4, 2), (4, 3)}
        # X cells: one near each entrance
        self.grid[6, 13] = CellType.AFFORDANCE_X.value
        self.grid[8, 6] = CellType.AFFORDANCE_X.value
