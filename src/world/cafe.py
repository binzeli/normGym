from src.world.base_world import BaseWorld, CellType


class CafeWorld(BaseWorld):
    """
    Grid-based cafe environment with multiple zones and navigation paths.
    Supports norm-based multi-agent interactions.
    """

    WORLD_NAME = "CAFE LAYOUT"

    ENTRANCES = [(5, 0), (10, 7), (5, 14)]

    ISO_WINDOW_CELLS = {
      (1, 0), (2, 0), (3, 0),
      (4, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0)
  }


    ISO_HIDDEN_WALL_CELLS = {(r, 14) for r in range(1, 11)}

    # The cafe's waiting zone: the cells agents settle in / treat as the waiting
    # area for occupation norms. Rendered as the dashed-border + footprints
    # overlay (see the renderers' _draw_waiting_region). Plain floor underneath,
    # so no solid tile is drawn.
    WAITING_ZONE = {
        (1, 10), (2, 10), (3, 10),
        (1, 11), (2, 11), (3, 11),
    }

    ISO_LAMPS = [
        (1, 6),
        (1, 9),
        (1, 3),
        (1, 0), 
        (1, 12), 
    ]

    def __init__(self, width: int = 15, height: int = 11,
                 map_type: str = "default"):
        super().__init__(width=width, height=height, map_type=map_type)

    def _create_layout(self):
        """Create the cafe layout with all zones, chairs, and counters"""
        self.grid.fill(CellType.WALL.value)

        # Main floor space
        self._add_rectangle(1, 1, 9, 13, CellType.EMPTY)

        # Entrances
        for entrance in self.entrances:
            self.grid[entrance] = CellType.ENTRANCE.value

        # Counter (always fixed)
        self.grid[1, 6] = CellType.COUNTER.value

        # Pickup shelf
        #self.grid[1, 4] = CellType.PICKUP_SHELF.value

        # Small tables along the left wall (rows 6-9, col 1). Only the (7,1)
        # 2-top remains, with its single chair above it at (6,1); (8,1)/(9,1)
        # hold flower pots instead of a second table and chair (set further
        # down, after the rest of the furniture).
        self._add_2top_with_one_top_chair(7, 1)


        self._add_barista_station(1, 12)
        # Top tables, keeping only their top chairs ((1,1) and (3,3)): the bottom
        # chairs (3,1)/(5,3) are removed so the strictly nearest chair from the
        # left door is (6,1) — inside the bottom-left region — which keeps the
        # bottom-left seat norms meaningful for agents entering there.
        self._add_2top_with_chairs(2, 1)
        self._add_2top_with_chairs(4, 3)
        self.grid[3, 1] = CellType.EMPTY.value
        self.grid[5, 3] = CellType.EMPTY.value

        for r, c in self.entrances:
            self.grid[r, c] = CellType.ENTRANCE.value
        # Clear inward-adjacent cells so entrances aren't blocked
        self._clear_entrance_adjacent()
        # Waiting zone (WAITING_ZONE) stays plain floor; it is registered as a
        # zone in _identify_zones and drawn as a footprint overlay by the renderer.


        # X cells:
        self.grid[8, 8] = CellType.AFFORDANCE_X.value
        self.grid[9, 8] = CellType.AFFORDANCE_X.value
        self.grid[7, 8] = CellType.AFFORDANCE_X.value
        self.grid[6, 8] = CellType.AFFORDANCE_X.value
       
        


        # Lower-left tables (rows 6-9, cols 1-5)
        self._add_2top_with_one_chair(7, 3)
        self._add_2top_with_one_chair(7, 5)
        # Lower-right tables (rows 6-8, cols 9-13)
        self._add_2top_with_chairs(7, 9)
        self._add_2top_with_chairs(7, 11)
        self._add_2top_with_chairs(7, 13)

        # Remove the top chair of the (7,9) table
        self.grid[6, 9] = CellType.EMPTY.value
        # Remove the bottom chair of the (7,11) table
        self.grid[8, 11] = CellType.EMPTY.value

        # Flower pots
        self.grid[9, 13] = CellType.FLOWER_POT.value
        self.grid[8, 1] = CellType.FLOWER_POT.value
        self.grid[9, 1] = CellType.FLOWER_POT.value



    def _add_barista_station(self, start_row: int, start_col: int):
        """Add a 3x3 barista station with cafe equipment.
        Layout:
            Row 0: PASTRY_DISPLAY   ESPRESSO_MACHINE  WALL
            Row 1: PASTRY_DISPLAY    WALL            WALL
            Row 2: MENU_BOARD       WALL            WALL
        """
        layout = [
            [CellType.PASTRY_DISPLAY, CellType.ESPRESSO_MACHINE, CellType.MENU_BOARD],
            [CellType.PASTRY_DISPLAY, CellType.EMPTY, CellType.MENU_BOARD],
            [CellType.PASTRY_DISPLAY, CellType.EMPTY, CellType.MENU_BOARD],
        ]
        for dr, row_types in enumerate(layout):
            for dc, cell_type in enumerate(row_types):
                r, c = start_row + dr, start_col + dc
                if 0 <= r < self.height and 0 <= c < self.width:
                    self.grid[r, c] = cell_type.value

    def _add_2top_with_chairs(self, row: int, col: int,
                              chair_type: int = CellType.CHAIR.value):
        """Add a 2-top table with 2 chairs (vertical arrangement)"""
        # Table (single cell)
        table_pos = (row, col)
        self.grid[row, col] = CellType.TABLE_2TOP.value

        # Chairs above and below
        chair_top = (row - 1, col)
        chair_bottom = (row + 1, col)

        self.grid[chair_top[0], chair_top[1]] = chair_type
        self.grid[chair_bottom[0], chair_bottom[1]] = chair_type

    def _add_2top_with_one_chair(self, row: int, col: int,
                                 chair_type: int = CellType.CHAIR.value):
        """Add a 2-top table with 2 chairs (vertical arrangement)"""
        # Table (single cell)
        table_pos = (row, col)
        self.grid[row, col] = CellType.TABLE_2TOP.value

        # Chairs above and below
        chair_bottom = (row + 1, col)

        self.grid[chair_bottom[0], chair_bottom[1]] = chair_type

    def _add_2top_with_one_top_chair(self, row: int, col: int,
                                     chair_type: int = CellType.CHAIR.value):
        """Add a 2-top table with a single chair above it"""
        # Table (single cell)
        table_pos = (row, col)
        self.grid[row, col] = CellType.TABLE_2TOP.value

        # Chair above
        chair_top = (row - 1, col)

        self.grid[chair_top[0], chair_top[1]] = chair_type


    def _add_rectangle(self, start_row: int, start_col: int, 
                       height: int, width: int, cell_type: CellType):
        """Helper to add rectangular zones"""
        for r in range(start_row, start_row + height):
            for c in range(start_col, start_col + width):
                if 0 <= r < self.height and 0 <= c < self.width:
                    self.grid[r, c] = cell_type.value
    
    def _identify_zones(self):
        """Identify and store all zone locations"""
        self.walkable_cells = set()
        self.zones = {
            'entrance': [],
            'counter': [],
            'queue': [],
            'pickup_shelf': [],
            'chairs': [],
            'tables_2top': [],
            'tables_4top': [],
            'kitchen': [],
            'flower_pots': [],
            'open_space': []
        }
        
        for r in range(self.height):
            for c in range(self.width):
                cell = self.grid[r, c]
                pos = (r, c)
                
                if cell == CellType.ENTRANCE.value:
                    self.zones['entrance'].append(pos)
                    self.walkable_cells.add(pos)
                elif cell == CellType.COUNTER.value:
                    self.zones['counter'].append(pos)
                elif cell == CellType.PICKUP_SHELF.value:
                    self.zones['pickup_shelf'].append(pos)
                elif cell == CellType.CHAIR.value:
                    self.zones['chairs'].append(pos)
                elif cell == CellType.RED_CHAIR.value:
                    self.zones.setdefault('red_chairs', []).append(pos)
                    self.zones['chairs'].append(pos)
                elif cell == CellType.ORANGE_CHAIR.value:
                    self.zones.setdefault('orange_chairs', []).append(pos)
                    self.zones['chairs'].append(pos)
                elif cell == CellType.BLUE_CHAIR.value:
                    self.zones.setdefault('blue_chairs', []).append(pos)
                    self.zones['chairs'].append(pos)
                elif cell == CellType.GREEN_CHAIR.value:
                    self.zones.setdefault('green_chairs', []).append(pos)
                    self.zones['chairs'].append(pos)
                elif cell == CellType.TABLE_2TOP.value:
                    self.zones['tables_2top'].append(pos)
                elif cell == CellType.TABLE_4TOP.value:
                    self.zones['tables_4top'].append(pos)
                elif cell == CellType.KITCHEN.value:
                    self.zones['kitchen'].append(pos)
                elif cell == CellType.FLOWER_POT.value:
                    self.zones['flower_pots'].append(pos)
                elif cell == CellType.AFFORDANCE_X.value:
                    self.zones.setdefault('affordance_x', []).append(pos)
                    self.walkable_cells.add(pos)
                elif cell == CellType.EMPTY.value:
                    self.zones['open_space'].append(pos)
                    self.walkable_cells.add(pos)

        # Waiting zone: registered from the WAITING_ZONE footprint cells (plain
        # floor underneath). These are the occupation-norm target cells; the
        # renderer draws the dashed-border + footprint overlay over them.
        for pos in getattr(self, 'WAITING_ZONE', ()):  # noqa: B009
            self.zones.setdefault('waiting_zone', [])
            if pos not in self.zones['waiting_zone']:
                self.zones['waiting_zone'].append(pos)
            self.walkable_cells.add(pos)

        # x_overlay cells are always walkable regardless of underlying cell type
        for pos in self.x_overlay:
            self.walkable_cells.add(pos)
            self.zones.setdefault('affordance_x', [])
            if pos not in self.zones['affordance_x']:
                self.zones['affordance_x'].append(pos)
    
    def is_walkable(self, pos):
        """Check if a position is walkable"""
        r, c = pos
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        return pos in self.walkable_cells
    
    def is_sittable(self, pos):
        """Check if a position is a chair that can be sat in"""
        r, c = pos
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        cell_type = self.grid[r, c]
        return cell_type in (
            CellType.CHAIR.value,
            CellType.RED_CHAIR.value,
            CellType.ORANGE_CHAIR.value,
            CellType.BLUE_CHAIR.value,
            CellType.GREEN_CHAIR.value,
        )
    
    def get_neighbors(self, pos):
        """
        Get walkable neighboring positions including diagonal positions.
        Returns neighbors in 8 directions: up, down, left, right, and 4 diagonals.
        """
        r, c = pos
        neighbors = []
        # 8 directions: orthogonal + diagonal
        # Format: (row_offset, col_offset)
        directions = [
            (-1, 0),   # up
            (1, 0),    # down
            (0, -1),   # left
            (0, 1)
        ]
        for dr, dc in directions:
            new_pos = (r + dr, c + dc)
            if self.is_walkable(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def get_cell_type(self, pos):
        """Get the type of cell at a position"""
        r, c = pos
        if not (0 <= r < self.height and 0 <= c < self.width):
            return None
        return CellType(self.grid[r, c])

    def build_object_positions_text(self) -> str:
        """Build a compact, structured text block describing positions of key objects on the grid."""
        zone_labels = {
            'entrance': 'entrances',
            'counter': 'counter',
            'chairs': 'chairs',
            'tables_2top': 'tables_2top',
        }

        lines: List[str] = []

        for zone_key, label in zone_labels.items():
            positions = sorted(self.zones.get(zone_key, []))
            if positions:
                lines.append(f"- {label}: {positions}")

        lines.extend(self._neutral_zone_position_lines())

        return "\n".join(lines)
