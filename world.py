import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum

class CellType(Enum):
    """Types of cells in the cafe grid world"""
    EMPTY = 0
    WALL = 1
    ENTRANCE = 2
    COUNTER = 3
    QUEUE_ZONE = 4
    PICKUP_SHELF = 5
    CHAIR = 6
    TABLE_2TOP = 7
    TABLE_4TOP = 8
    AISLE = 9
    KITCHEN = 10
    LAPTOP_BAR_COUNTER = 11

class Direction(Enum):
    """Movement directions"""
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    STAY = (0, 0)

class CafeWorld:
    """
    Grid-based cafe environment with multiple zones and navigation paths.
    Supports norm-based multi-agent interactions.
    """
    
    def __init__(self, width: int = 15, height: int = 10):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.agent_positions: Dict[str, Tuple[int, int]] = {}
        self.item_positions: Dict[Tuple[int, int], List[str]] = {}
        
        # Zone definitions (will be populated by _create_layout)
        self.zones: Dict[str, List[Tuple[int, int]]] = {}
        
        # Seating assignments (chair_pos -> table_pos mapping)
        self.chair_to_table: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # Navigation graph for pathfinding
        self.walkable_cells: Set[Tuple[int, int]] = set()
        
        # Create the cafe layout
        self._create_layout()
        self._identify_zones()
        
    def _create_layout(self):
        """Create the cafe layout with all zones, chairs, and counters"""
        # Fill with walls initially
        self.grid.fill(CellType.WALL.value)
        
        # Main floor space (large open area)
        self._add_rectangle(1, 1, 10, 13, CellType.EMPTY)
        
        # ENTRANCE (bottom center)
        entrance_row = 9
        entrance_cols = range(6,10)
        for col in entrance_cols:
            self.grid[entrance_row, col] = CellType.ENTRANCE.value

        # COUNTER (top middle area - staff service counter)
        # Counter itself (staff area behind counter)
        counter_row = 1
        counter_cols = range(6, 8)
        for c in counter_cols:
            self.grid[counter_row, c] = CellType.COUNTER.value

        
        # PICKUP SHELF (left of the counter)
        self._add_rectangle(1, 4, 1, 1, CellType.PICKUP_SHELF)

        # KITCHEN (to the right)
        self._add_rectangle(1, 12, 3, 3, CellType.KITCHEN)
        
        # LAPTOP BAR 
        laptop_bar_col = 1
        for row in range(6, 9):
            self.grid[row, laptop_bar_col] = CellType.LAPTOP_BAR_COUNTER.value
            self.grid[row, laptop_bar_col + 1] = CellType.CHAIR.value
      
        # 2-TOP TABLES WITH CHAIRS (small tables - left area)
        # Table 1 (top left dining area)
        self._add_2top_with_chairs(3, 1)
        # Table 2
        self._add_2top_with_chairs(3, 3)
        # Table 3 (middle left)
        self._add_2top_with_chairs(7, 4)
        
        # 4-TOP TABLES WITH CHAIRS (larger tables - right area)
        # Table 1 (top right)
        self._add_4top_with_chairs(3, 9)
        # Table 2 (middle right)
        self._add_4top_with_chairs(7, 9)
        # Table 3 (bottom right)
        self._add_4top_with_chairs(7, 12)
        
    
    def _add_2top_with_chairs(self, row: int, col: int):
        """Add a 2-top table with 2 chairs (vertical arrangement)"""
        # Table (single cell)
        table_pos = (row, col)
        self.grid[row, col] = CellType.TABLE_2TOP.value
        
        # Chairs above and below
        chair_top = (row - 1, col)
        chair_bottom = (row + 1, col)
        
        self.grid[chair_top[0], chair_top[1]] = CellType.CHAIR.value
        self.grid[chair_bottom[0], chair_bottom[1]] = CellType.CHAIR.value
        
        # Map chairs to table
        self.chair_to_table[chair_top] = table_pos
        self.chair_to_table[chair_bottom] = table_pos
    
    def _add_4top_with_chairs(self, row: int, col: int):
        """Add a 4-top table with 4 chairs"""
        # Table (2x2)
        table_positions = [
            (row, col), (row, col + 1),
            (row + 1, col), (row + 1, col + 1)
        ]
        for r, c in table_positions:
            self.grid[r, c] = CellType.TABLE_4TOP.value
        
        table_center = (row, col)  # Use top-left as reference
        
        # Chairs around the table
        chairs = [
            (row - 1, col),      # top-left
            (row - 1, col + 1),  # top-right
            (row + 2, col),      # bottom-left
            (row + 2, col + 1),  # bottom-right
        ]
        
        for chair_pos in chairs:
            r, c = chair_pos
            if 0 <= r < self.height and 0 <= c < self.width:
                self.grid[r, c] = CellType.CHAIR.value
                self.chair_to_table[chair_pos] = table_center
    
    def _add_rectangle(self, start_row: int, start_col: int, 
                       height: int, width: int, cell_type: CellType):
        """Helper to add rectangular zones"""
        for r in range(start_row, start_row + height):
            for c in range(start_col, start_col + width):
                if 0 <= r < self.height and 0 <= c < self.width:
                    self.grid[r, c] = cell_type.value
    
    def _identify_zones(self):
        """Identify and store all zone locations"""
        self.zones = {
            'entrance': [],
            'counter': [],
            'queue': [],
            'pickup_shelf': [],
            'laptop_bar_counter': [],
            'chairs': [],
            'tables_2top': [],
            'tables_4top': [],
            'aisles': [],
            'kitchen': [],
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
                # elif cell == CellType.QUEUE_ZONE.value:
                #     self.zones['queue'].append(pos)
                #     self.walkable_cells.add(pos)
                elif cell == CellType.PICKUP_SHELF.value:
                    self.zones['pickup_shelf'].append(pos)
                elif cell == CellType.LAPTOP_BAR_COUNTER.value:
                    self.zones['laptop_bar_counter'].append(pos)
                elif cell == CellType.CHAIR.value:
                    self.zones['chairs'].append(pos)
                elif cell == CellType.TABLE_2TOP.value:
                    self.zones['tables_2top'].append(pos)
                elif cell == CellType.TABLE_4TOP.value:
                    self.zones['tables_4top'].append(pos)
                elif cell == CellType.AISLE.value:
                    self.zones['aisles'].append(pos)
                    self.walkable_cells.add(pos)
                elif cell == CellType.KITCHEN.value:
                    self.zones['kitchen'].append(pos)
                elif cell == CellType.EMPTY.value:
                    self.zones['open_space'].append(pos)
                    self.walkable_cells.add(pos)
    
    def is_walkable(self, pos):
        """Check if a position is walkable (excluding chairs/tables)"""
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
        return cell_type == CellType.CHAIR.value
    
    def get_neighbors(self, pos) :
        """Get walkable neighboring positions"""
        r, c = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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
    
    def print_layout(self):
        """
        Print a visual representation of the cafe layout.
        
        Args:
            agents: Optional dictionary of agent_id -> (row, col) positions
        """
        # Symbol mapping for different cell types
        symbols = {
            CellType.EMPTY.value: '·',
            CellType.WALL.value: '█',
            CellType.ENTRANCE.value: 'E',
            CellType.COUNTER.value: 'C',
            CellType.PICKUP_SHELF.value: 'P',
            CellType.LAPTOP_BAR_COUNTER.value: 'L',
            CellType.CHAIR.value: 'h',
            CellType.TABLE_2TOP.value: '▪',
            CellType.TABLE_4TOP.value: '▪',
            CellType.AISLE.value: '-',
            CellType.KITCHEN.value: 'K'
        }
        
    
        print("\n" + "=" * (self.width * 2 + 2))
        print(" " * ((self.width * 2 - 15) // 2) + "CAFE LAYOUT")
        print("=" * (self.width * 2 + 2))
        
        for r in range(self.height):
            row_str = ""
            for c in range(self.width):
                pos = (r, c)
                row_str += symbols[self.grid[r, c]] + " "
            print(row_str)
        print("=" * (self.width * 2 + 2))
        print("\nLEGEND:")
        print("  █ = Wall         E = Entrance      · = Open Space")
        print("  C = Counter      ▪ = Table         P = Pickup Shelf")
        print("  K = Kitchen      h = Chair         ")
        print("=" * (self.width * 2 + 2) + "\n")
    




