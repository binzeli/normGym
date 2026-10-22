from src.world.base_world import BaseWorld, CellType


class LibraryWorld(BaseWorld):
    """
    Library: checkout desk, study room, bookshelves, reading desks,
    computer stations, book return drop.
    """

    WORLD_NAME = "LIBRARY LAYOUT"
    ENTRANCES = [(6, 0), (10, 7), (8, 14)]

    
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
        ('counter', 'checkout_desk'),
        ('study_room', 'study_room'),
        ('bookshelves', 'bookshelves'),
        ('reading_desks', 'reading_desks'),
        ('computer_stations', 'computer_stations'),
        ('return_drop', 'book_return'),
    ]

    def _create_layout(self):
        self.grid.fill(CellType.WALL.value)
        self._add_rectangle(1, 1, 9, 13, CellType.EMPTY)

        for entrance in self.entrances:
            self.grid[entrance] = CellType.ENTRANCE.value

        # Checkout desk on the top wall (col 4 — varied per world).
        self.grid[3, 1] = CellType.COUNTER.value
        self.grid[1, 7] = CellType.RETURN_DROP.value
        self.grid[1, 8] = CellType.RETURN_DROP.value
        self.grid[1, 9] = CellType.RETURN_DROP.value

        self.grid[7, 10] = CellType.READING_DESK.value

        # Reading-chair lounge (replaces the old study room)
        for c in (11, 12, 13):
            self.grid[1, c] = CellType.CHAIR.value
        self.grid[3, 12] = CellType.CHAIR.value
        self.grid[4, 12] = CellType.CHAIR.value
        
        self.grid[9, 2] = CellType.CHAIR.value
        self.grid[9, 4] = CellType.CHAIR.value




        self.grid[7, 9] = CellType.COMPUTER_STATION.value
        self.grid[7, 11] = CellType.COMPUTER_STATION.value

        for c in (1, 3, 5):
            self.grid[9, c] = CellType.BOOKSHELF.value
        self.grid[6, 4] = CellType.BOOKSHELF.value
        self.grid[6, 5] = CellType.BOOKSHELF.value
        self.grid[7, 4] = CellType.BOOKSHELF.value
        self.grid[7, 5] = CellType.BOOKSHELF.value
       
        self.grid[7, 8] = CellType.FLOWER_POT.value
        self.grid[7, 12] = CellType.FLOWER_POT.value

        # Clear inward-adjacent cells so entrances aren't blocked
        self._clear_entrance_adjacent()
        # Waiting zone: plain floor, rendered as a footprint overlay; the
        # occupation-norm target region (see BaseWorld._identify_zones).
        self.WAITING_ZONE = {(2, 8), (2, 9), (3, 8), (3, 9), (3, 7), (2, 7)}
        # X cells: one near each entrance
        self.grid[5, 1] = CellType.AFFORDANCE_X.value
        self.grid[7, 13] = CellType.AFFORDANCE_X.value
        
