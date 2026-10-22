from src.world.base_world import BaseWorld, CellType


class GroceryStoreWorld(BaseWorld):
    """
    Grocery store: checkout counter, bagging areas, storage rooms,
    product shelf aisles, cart corrals.
    """

    WORLD_NAME = "GROCERY STORE LAYOUT"

    ENTRANCES = [(3, 0), (10, 7), (5, 14)]

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
        ('counter', 'checkout_counter'),
        ('bagging_area', 'bagging_areas'),
        ('storage_room', 'storage_rooms'),
        ('product_shelves', 'product_shelves'),
        ('cart_corrals', 'cart_corrals'),
    ]

    def _create_layout(self):
        self.grid.fill(CellType.WALL.value)
        self._add_rectangle(1, 1, 9, 13, CellType.EMPTY)

        for entrance in self.entrances:
            self.grid[entrance] = CellType.ENTRANCE.value

        # Checkout counter on the top wall (col 5 — varied per world).
        self.grid[7, 1] = CellType.COUNTER.value
        self.grid[1, 11] = CellType.BAGGING_AREA.value
        self.grid[1, 12] = CellType.BAGGING_AREA.value
        self.grid[1, 13] = CellType.BAGGING_AREA.value


        for c in (8, 9, 10):
            self.grid[1, c] = CellType.CART_CORRAL.value
        self.grid[7, 6] = CellType.CART_CORRAL.value
        self.grid[8, 6] = CellType.CART_CORRAL.value

       
        self._add_rectangle(1, 1, 2, 3, CellType.STORAGE_ROOM)

        self.grid[7, 10] = CellType.PRODUCT_SHELF.value
        self.grid[7, 8] = CellType.PRODUCT_SHELF.value
        self.grid[7, 9] = CellType.PRODUCT_SHELF.value
        self.grid[6, 10] = CellType.PRODUCT_SHELF.value
        self.grid[6, 8] = CellType.PRODUCT_SHELF.value
        self.grid[6, 9] = CellType.PRODUCT_SHELF.value

        self.grid[9, 10] = CellType.PRODUCT_SHELF.value
        self.grid[9, 8] = CellType.PRODUCT_SHELF.value
        self.grid[9, 9] = CellType.PRODUCT_SHELF.value

        # Waiting benches
        for c in (11, 12):
            self.grid[7, c] = CellType.CHAIR.value
            self.grid[9, c] = CellType.CHAIR.value

        self.grid[1, 5] = CellType.CHAIR.value
        self.grid[1, 6] = CellType.CHAIR.value
        self.grid[1, 7] = CellType.CHAIR.value

        
        # Clear inward-adjacent cells so entrances aren't blocked
        self._clear_entrance_adjacent()
        # Waiting zone: plain floor, rendered as a footprint overlay; the
        # occupation-norm target region (see BaseWorld._identify_zones).
        self.WAITING_ZONE = {(2, 11), (2, 12), (2, 13), (3, 11), (3, 12), (3, 13)}
        # X cells: one near each entrance
        self.grid[5, 1] = CellType.AFFORDANCE_X.value
        self.grid[8, 7] = CellType.AFFORDANCE_X.value

