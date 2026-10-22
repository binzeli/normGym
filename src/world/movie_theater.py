from src.world.base_world import BaseWorld, CellType


class MovieTheaterWorld(BaseWorld):
    """
    Movie theater lobby: ticket counter, concession stand, screen/projection room,
    theater seats, snack stand, rope barriers.
    """

    WORLD_NAME = "MOVIE THEATER LAYOUT"

    ENTRANCES = [(6, 0), (10, 7), (5, 14)]

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
        ('counter', 'ticket_counter'),
        ('concession_stand', 'concession_stand'),
        ('screen_room', 'screen_room'),
        ('theater_seats', 'theater_seats'),
        ('snack_stand', 'snack_stand'),
        ('rope_barriers', 'rope_barriers'),
    ]

    def _create_layout(self):
        self.grid.fill(CellType.WALL.value)
        self._add_rectangle(1, 1, 9, 13, CellType.EMPTY)

        for entrance in self.entrances:
            self.grid[entrance] = CellType.ENTRANCE.value

        # Ticket counter on the top wall (col 7 — varied per world).
        self.grid[1, 8] = CellType.COUNTER.value
        self.grid[1, 11] = CellType.CONCESSION_STAND.value
        self.grid[1, 12] = CellType.CONCESSION_STAND.value
        self.grid[1, 13] = CellType.CONCESSION_STAND.value

        self._add_rectangle(1, 1, 1, 4, CellType.SCREEN_ROOM)
        self._add_rectangle(9, 11, 1, 3, CellType.SCREEN_ROOM)

        self.grid[9, 1] = CellType.SCREEN_ROOM.value
        self.grid[9, 2] = CellType.SCREEN_ROOM.value
        self.grid[9, 3] = CellType.SCREEN_ROOM.value

        # Front seat bank
        self.grid[8, 1] = CellType.CHAIR.value
        self.grid[8, 2] = CellType.CHAIR.value
        self.grid[8, 3] = CellType.CHAIR.value





        # Right seating block
        self.grid[8, 11] = CellType.CHAIR.value
        self.grid[8, 12] = CellType.CHAIR.value
        self.grid[8, 13] = CellType.CHAIR.value
        self.grid[2, 1] = CellType.CHAIR.value
        self.grid[2, 2] = CellType.CHAIR.value
        self.grid[2, 3] = CellType.CHAIR.value
        self.grid[2, 4] = CellType.CHAIR.value


        self.grid[6, 6] = CellType.SNACK_STAND.value
        self.grid[6, 7] = CellType.SNACK_STAND.value
        self.grid[6, 8] = CellType.SNACK_STAND.value
        self.grid[7, 6] = CellType.SNACK_STAND.value
        self.grid[7, 7] = CellType.SNACK_STAND.value
        self.grid[7, 8] = CellType.SNACK_STAND.value


        self.grid[1, 5] = CellType.ROPE_BARRIER.value
        self.grid[2, 5] = CellType.ROPE_BARRIER.value

        # Clear inward-adjacent cells so entrances aren't blocked
        self._clear_entrance_adjacent()
        # Waiting zone: plain floor, rendered as a footprint overlay; the
        # occupation-norm target region (see BaseWorld._identify_zones).
        self.WAITING_ZONE = {(2, 11), (2, 12), (2, 13), (3, 11), (3, 12), (3, 13)}
        # X cells: one near each entrance
        self.grid[6, 13] = CellType.AFFORDANCE_X.value
        self.grid[9, 6] = CellType.AFFORDANCE_X.value
        self.grid[4, 1] = CellType.AFFORDANCE_X.value
        