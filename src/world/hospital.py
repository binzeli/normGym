from src.world.base_world import BaseWorld, CellType


class HospitalWorld(BaseWorld):
    """
    Hospital: reception counter, exam room, nurses station,
    hospital beds, waiting benches, triage area.
    """

    WORLD_NAME = "HOSPITAL LAYOUT"

    ENTRANCES = [(5, 0), (10, 7), (7, 14)]

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
        ('counter', 'reception'),
        ('exam_room', 'exam_room'),
        ('nurses_station', 'nurses_station'),
        ('hospital_beds', 'hospital_beds'),
        ('waiting_benches', 'waiting_benches'),
        ('triage_area', 'triage_area'),
    ]

    def _create_layout(self):
        self.grid.fill(CellType.WALL.value)
        self._add_rectangle(1, 1, 9, 13, CellType.EMPTY)

        for entrance in self.entrances:
            self.grid[entrance] = CellType.ENTRANCE.value

        # Reception on the top wall (col 5 — varied per world).
        self.grid[3, 13] = CellType.COUNTER.value

        self.grid[1, 1] = CellType.TRIAGE_AREA.value
        self.grid[1, 2] = CellType.TRIAGE_AREA.value
        self.grid[1, 3] = CellType.TRIAGE_AREA.value


        # Waiting benches — red
        self.grid[7, 2] = CellType.CHAIR.value
        self.grid[7, 3] = CellType.CHAIR.value
        self.grid[7, 4] = CellType.CHAIR.value

        # Waiting benches (middle)
        self.grid[7, 10] = CellType.CHAIR.value
        self.grid[7, 11] = CellType.CHAIR.value
        self.grid[7, 12] = CellType.CHAIR.value


        # Waiting benches (right)
        self.grid[9, 10] = CellType.CHAIR.value
        self.grid[9, 11] = CellType.CHAIR.value
        self.grid[9, 12] = CellType.CHAIR.value


        self.grid[1,5] = CellType.EXAM_ROOM.value
        self.grid[1,6] = CellType.EXAM_ROOM.value
        self.grid[1,7] = CellType.EXAM_ROOM.value
        self.grid[1,8] = CellType.EXAM_ROOM.value

        self.grid[3,5] = CellType.EXAM_ROOM.value
        self.grid[3,6] = CellType.EXAM_ROOM.value
        self.grid[3,7] = CellType.EXAM_ROOM.value
        self.grid[3,8] = CellType.EXAM_ROOM.value

        self.grid[9, 1] = CellType.FLOWER_POT.value
        self.grid[9, 2] = CellType.NURSES_STATION.value
        self.grid[9, 3] = CellType.NURSES_STATION.value
        self.grid[9, 4] = CellType.FLOWER_POT.value

        self.grid[7, 6] = CellType.HOSPITAL_BED.value
        self.grid[8, 6] = CellType.HOSPITAL_BED.value
        self.grid[7, 8] = CellType.HOSPITAL_BED.value
        self.grid[8, 8] = CellType.HOSPITAL_BED.value

        # Clear inward-adjacent cells so entrances aren't blocked
        self._clear_entrance_adjacent()
        # Waiting zone: plain floor, rendered as a footprint overlay; the
        # occupation-norm target region (see BaseWorld._identify_zones).
        self.WAITING_ZONE = {(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)}
        # X cells: one near each entrance

        self.grid[7, 1] = CellType.AFFORDANCE_X.value
        self.grid[7, 13] = CellType.AFFORDANCE_X.value
        self.grid[8, 7] = CellType.AFFORDANCE_X.value

