import os
import numpy as np
from typing import Dict, Optional
from PIL import Image

from src.world.base_world import CellType, CHAIR_COLOR_RGB


def _tint_pil(img: "Image.Image", color, strength: float = 0.85) -> "Image.Image":
    """Tint a PIL RGBA image toward `color` while preserving shading/alpha.

    Each pixel's RGB is blended toward (lum * color), with `strength`
    controlling how strongly the tint dominates the original color.
    """
    if img is None:
        return None
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    rgb = arr[..., :3]
    a = arr[..., 3:4]
    # Luminance (Rec. 601) gives us the shading detail.
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]
    target = lum * (np.array(color, dtype=np.float32) / 255.0)[None, None, :]
    out_rgb = rgb * (1.0 - strength) + target * strength
    out = np.concatenate([out_rgb, a], axis=-1)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def _tint_np(arr: "np.ndarray", color, strength: float = 0.85) -> "np.ndarray":
    """Tint a float32 RGBA numpy array (range 0..1) toward `color`."""
    if arr is None:
        return None
    rgb = arr[..., :3]
    a = arr[..., 3:4]
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]
    target = lum * (np.array(color, dtype=np.float32) / 255.0)[None, None, :]
    out_rgb = rgb * (1.0 - strength) + target * strength
    return np.concatenate([out_rgb, a], axis=-1).astype(np.float32)


# The flat chair base sprite is darker than the isometric one, so the same
# CHAIR_COLOR_RGB tint comes out darker in the flat render. Brightening the
# flat base luminance by this factor before tinting makes the resulting chair
# colors match the iso render exactly.
FLAT_CHAIR_BRIGHTEN = 1.25

# Plain chairs are tinted to one neutral walnut brown in every world so seating
# reads as wooden furniture. Deliberately kept away from the orange/green/blue
# agent-class hues so a chair is never mistaken for a color tag.
PLAIN_CHAIR_TINT = (140, 92, 55)
PLAIN_CHAIR_TINT_STRENGTH = 0.70


def _brighten_np(arr: "np.ndarray", factor: float) -> "np.ndarray":
    """Scale the RGB luminance of a float32 RGBA array (alpha preserved)."""
    if arr is None:
        return None
    out = arr.copy()
    out[..., :3] = np.clip(out[..., :3] * factor, 0.0, 1.0)
    return out


# Mapping from CellType int value -> sprite filename (relative to assets_dir)
SHARED_SPRITES = {
    CellType.EMPTY.value: "cells/shared/empty.png",
    CellType.WALL.value: "cells/shared/wall.png",
    CellType.ENTRANCE.value: "cells/shared/entrance.png",
    CellType.COUNTER.value: "cells/shared/counter.png",
    CellType.QUEUE_ZONE.value: "cells/shared/queue_zone.png",
    CellType.FLOWER_POT.value: "cells/shared/flower_pot.png",
    CellType.AFFORDANCE_X.value: "cells/shared/affordance_x.png",
}

WORLD_SPRITES = {
    'cafe': {
        CellType.COUNTER.value: "cells/cafe/counter.png",
        CellType.PICKUP_SHELF.value: "cells/cafe/pickup_shelf.png",
        CellType.CHAIR.value: "cells/cafe/chair.png",
        CellType.TABLE_2TOP.value: "cells/cafe/table_2top.png",
        CellType.TABLE_4TOP.value: "cells/cafe/table_4top.png",
        CellType.KITCHEN.value: "cells/cafe/kitchen.png",
        CellType.LAPTOP_BAR_COUNTER.value: "cells/cafe/laptop_bar_counter.png",
        CellType.PASTRY_DISPLAY.value: "cells/cafe/pastry_display.png",
        CellType.COFFEE_GRINDER.value: "cells/cafe/coffee_grinder.png",
        CellType.MENU_BOARD.value: "cells/cafe/menu_board.png",
        CellType.ESPRESSO_MACHINE.value: "cells/cafe/espresso_machine.png",
    },
    'movie_theater': {
        CellType.COUNTER.value: "cells/movie_theater/counter.png",
        CellType.SCREEN_ROOM.value: "cells/movie_theater/screen_room.png",
        CellType.ROPE_BARRIER.value: "cells/movie_theater/rope_barrier.png",
        CellType.THEATER_SEAT.value: "cells/movie_theater/theater_seat.png",
        CellType.THEATER_SEAT_D.value: "cells/movie_theater/theater_seat_d.png",
        CellType.CONCESSION_STAND.value: "cells/movie_theater/concession_stand.png",
        CellType.SNACK_STAND.value: "cells/movie_theater/snack_stand.png",
    },
    'bank': {
        CellType.COUNTER.value: "cells/bank/counter.png",
        CellType.VAULT.value: "cells/bank/vault.png",
        CellType.ATM_ROOM.value: "cells/bank/atm_room.png",
        CellType.ROPE_STANCHION.value: "cells/bank/rope_stanchion.png",
        CellType.WRITING_DESK.value: "cells/bank/writing_desk.png",
        CellType.DEPOSIT_SLIPS.value: "cells/bank/deposit_slips.png",
        CellType.WAITING_CHAIR.value: "cells/bank/waiting_chair.png",
    },
    'pharmacy': {
        CellType.COUNTER.value: "cells/pharmacy/counter.png",
        CellType.STOCKROOM.value: "cells/pharmacy/stockroom.png",
        CellType.MEDICINE_SHELF.value: "cells/pharmacy/medicine_shelf.png",
        CellType.DISPLAY_CASE.value: "cells/pharmacy/display_case.png",
        CellType.BP_STATION.value: "cells/pharmacy/bp_station.png",
        CellType.RX_PICKUP.value: "cells/pharmacy/rx_pickup.png",
        CellType.WAITING_BENCH.value: "cells/pharmacy/bench.png",
    },
    'grocery_store': {
        CellType.COUNTER.value: "cells/grocery_store/counter.png",
        CellType.STORAGE_ROOM.value: "cells/grocery_store/storage_room.png",
        CellType.PRODUCT_SHELF.value: "cells/grocery_store/product_shelf.png",
        CellType.CART_CORRAL.value: "cells/grocery_store/cart_corral.png",
        CellType.BAGGING_AREA.value: "cells/grocery_store/bagging_area.png",
        CellType.WAITING_BENCH.value: "cells/grocery_store/waiting_bench.png",
    },
    'clothing_store': {
        CellType.COUNTER.value: "cells/clothing_store/counter.png",
        CellType.CLOTHING_STOCKROOM.value: "cells/clothing_store/clothing_stockroom.png",
        CellType.CLOTHING_RACK.value: "cells/clothing_store/clothing_rack.png",
        CellType.FITTING_ROOM.value: "cells/clothing_store/fitting_room.png",
        CellType.MANNEQUIN.value: "cells/clothing_store/mannequin.png",
        CellType.SHOE_DISPLAY.value: "cells/clothing_store/shoe_display.png",
        CellType.WAITING_CHAIR.value: "cells/clothing_store/stool.png",
    },
    'library': {
        CellType.COUNTER.value: "cells/library/counter.png",
        CellType.BOOKSHELF.value: "cells/library/bookshelf.png",
        CellType.READING_DESK.value: "cells/library/reading_desk.png",
        CellType.COMPUTER_STATION.value: "cells/library/computer_station.png",
        CellType.RETURN_DROP.value: "cells/library/return_drop.png",
        CellType.STUDY_ROOM.value: "cells/library/study_room.png",
        CellType.WAITING_CHAIR.value: "cells/library/reading_chair.png",
    },
    'post_office': {
        CellType.COUNTER.value: "cells/post_office/counter.png",
        CellType.MAILBOX.value: "cells/post_office/mailbox.png",
        CellType.PACKAGE_SHELF.value: "cells/post_office/package_shelf.png",
        CellType.SORTING_ROOM.value: "cells/post_office/sorting_room.png",
        CellType.PO_BOX.value: "cells/post_office/po_box.png",
        CellType.SCALE_STATION.value: "cells/post_office/scale_station.png",
        CellType.WAITING_BENCH.value: "cells/post_office/waiting_bench.png",
    },
    'hospital': {
        CellType.COUNTER.value: "cells/hospital/counter.png",
        CellType.NURSES_STATION.value: "cells/hospital/nurses_station.png",
        CellType.HOSPITAL_BED.value: "cells/hospital/hospital_bed.png",
        CellType.WAITING_BENCH.value: "cells/hospital/waiting_bench.png",
        CellType.EXAM_ROOM.value: "cells/hospital/exam_room.png",
        CellType.TRIAGE_AREA.value: "cells/hospital/triage_area.png",
    },
    'barbershop': {
        CellType.COUNTER.value: "cells/barbershop/counter.png",
        CellType.BARBER_CHAIR.value: "cells/barbershop/barber_chair.png",
        CellType.WAITING_COUCH_R.value: "cells/barbershop/waiting_couch_right.png",
        CellType.WAITING_COUCH_L.value: "cells/barbershop/waiting_couch_left.png",
        CellType.MIRROR_STATION.value: "cells/barbershop/mirror_station.png",
        CellType.SUPPLY_CABINET.value: "cells/barbershop/supply_cabinet.png",
        CellType.MAGAZINE_RACK.value: "cells/barbershop/magazine_rack.png",
    },
}

# Per-world primary seat cell type used as the base sprite for tinting
# RED_CHAIR / ORANGE_CHAIR / BLUE_CHAIR colored clusters.
WORLD_SEAT_CELL = {
    'cafe': CellType.CHAIR.value,
    'movie_theater': CellType.THEATER_SEAT.value,
    'bank': CellType.WAITING_CHAIR.value,
    'hospital': CellType.WAITING_BENCH.value,
    'barbershop': CellType.BARBER_CHAIR.value,
    'clothing_store': CellType.WAITING_CHAIR.value,
    'library': CellType.WAITING_CHAIR.value,
    'grocery_store': CellType.WAITING_BENCH.value,
    'pharmacy': CellType.WAITING_BENCH.value,
    'post_office': CellType.WAITING_BENCH.value,
}

# Agent sprite filenames, keyed by (agent color class, state).
AGENT_SPRITES = {
    ('orange', 'normal'): "agents/orange.png",
    ('green', 'normal'): "agents/green.png",
    ('blue', 'normal'): "agents/blue.png",
    ('orange', 'push'): "agents/orange_push.png",
    ('green', 'push'): "agents/green_push.png",
    ('blue', 'push'): "agents/blue_push.png",
    ('orange', 'order'): "agents/orange_order.png",
    ('green', 'order'): "agents/green_order.png",
    ('blue', 'order'): "agents/blue_order.png",
    ('orange', 'done'): "agents/orange_done.png",
    ('green', 'done'): "agents/green_done.png",
    ('blue', 'done'): "agents/blue_done.png",
    # Walk-direction variants (flat renderer only; 'normal' is the
    # down/front-facing sprite). Missing files are simply skipped, so the
    # iso loader falls back gracefully.
    ('orange', 'up'): "agents/orange_up.png",
    ('green', 'up'): "agents/green_up.png",
    ('blue', 'up'): "agents/blue_up.png",
    ('orange', 'left'): "agents/orange_left.png",
    ('green', 'left'): "agents/green_left.png",
    ('blue', 'left'): "agents/blue_left.png",
    ('orange', 'right'): "agents/orange_right.png",
    ('green', 'right'): "agents/green_right.png",
    ('blue', 'right'): "agents/blue_right.png",
}


class SpriteCache:
    """Loads, caches, and composites PNG sprites for grid rendering."""

    def __init__(self, assets_dir: str, px_per_cell: int = 80):
        self.assets_dir = assets_dir
        self.px_per_cell = px_per_cell
        self._cell_sprites: Dict[int, np.ndarray] = {}
        self._agent_sprites: Dict[tuple, np.ndarray] = {}
        self._floor_tile: Optional[np.ndarray] = None
        self._floor_tiles: list = []  # tile variants for per-cell variation
        self._wall_face: Optional[np.ndarray] = None
        self._llm_star: Optional[np.ndarray] = None
        # Isometric sprite caches (PIL Images, not numpy arrays)
        self._iso_cell_sprites: Dict[int, Image.Image] = {}
        self._iso_agent_sprites: Dict[tuple, Image.Image] = {}
        self._iso_floor_tile: Optional[Image.Image] = None
        self._iso_llm_star: Optional[Image.Image] = None
        # Flat renderer chair_up variants (numpy arrays, keyed by cell_val)
        self._flat_chair_up: Optional[np.ndarray] = None
        self._flat_chair_up_by_color: Dict[int, np.ndarray] = {}
        # Static-background caches. The floor and furniture (cell) layers depend
        # only on the world size / grid contents, which don't change during a run,
        # yet save_frame() rebuilt them (Python pixel-loop compositing) on every
        # frame — wasteful, and a noticeable chunk of the per-frame render cost in
        # the live web viewer. Cache keyed on the inputs so correctness holds even
        # if the grid ever changes.
        self._floor_canvas_cache: Dict[tuple, np.ndarray] = {}
        self._cell_layer_cache: Dict[tuple, np.ndarray] = {}
        self._static_bg_cache: Dict[tuple, np.ndarray] = {}

    def _load_and_resize(self, rel_path: str, size: int) -> Optional[np.ndarray]:
        """Load a PNG and resize to (size, size). Returns RGBA float32 array or None."""
        full_path = os.path.join(self.assets_dir, rel_path)
        if not os.path.isfile(full_path):
            return None
        img = Image.open(full_path).convert("RGBA")
        img = img.resize((size, size), Image.NEAREST)
        return np.array(img, dtype=np.float32) / 255.0

    def load_world_sprites(self, world_type: str):
        """Load shared + world-specific cell sprites into the cache."""
        self._cell_sprites.clear()

        # Load shared sprites
        for cell_val, rel_path in SHARED_SPRITES.items():
            arr = self._load_and_resize(rel_path, self.px_per_cell)
            if arr is not None:
                self._cell_sprites[cell_val] = arr

        # Load world-specific sprites
        world_map = WORLD_SPRITES.get(world_type, {})
        for cell_val, rel_path in world_map.items():
            arr = self._load_and_resize(rel_path, self.px_per_cell)
            if arr is not None:
                self._cell_sprites[cell_val] = arr

        # Load floor tile variants. Each world can bring its own floor
        # material (floor/<world>_{0..2}.png — tile, carpet, checker...);
        # otherwise fall back to the shared wood planks. Variants add
        # per-cell texture variation.
        world_floor = [
            t for t in (
                self._load_and_resize(f"floor/{world_type}_{i}.png", self.px_per_cell)
                for i in range(3)
            ) if t is not None
        ]
        if world_floor:
            self._floor_tiles = world_floor
        else:
            self._floor_tiles = [
                t for t in (
                    self._load_and_resize("floor/wood_tile.png", self.px_per_cell),
                    self._load_and_resize("floor/wood_tile_1.png", self.px_per_cell),
                    self._load_and_resize("floor/wood_tile_2.png", self.px_per_cell),
                ) if t is not None
            ]
        self._floor_tile = self._floor_tiles[0] if self._floor_tiles else None

        # The static-background caches bake in the floor and cell sprites, so
        # they must not survive a world (re)load.
        self._floor_canvas_cache.clear()
        self._cell_layer_cache.clear()
        self._static_bg_cache.clear()

        # Wall face: drawn for wall cells whose south neighbor is interior,
        # giving the 3/4 top-down depth look. Optional — build_cell_layer
        # falls back to the flat wall sprite when absent.
        self._wall_face = self._load_and_resize("cells/shared/wall_face.png", self.px_per_cell)

        # Load agent sprites (no tint — use original sprite colors as-is).
        self._agent_sprites.clear()
        agent_size = int(self.px_per_cell * 0.75)  # 48px at 64px cell, 60px at 80px cell
        for key, rel_path in AGENT_SPRITES.items():
            arr = self._load_and_resize(rel_path, agent_size)
            if arr is not None:
                self._agent_sprites[key] = arr

        # Synthesize directional 'done' variants ('done_up' etc.). The done
        # sprite is the front-facing sprite plus a check badge, and there is
        # no done art for the walking directions, so done agents rendered
        # front-facing regardless of movement. Extract the badge (pixels
        # where done differs from normal) and stamp it onto each directional
        # sprite so done agents keep facing their movement direction.
        for color in ('orange', 'green', 'blue'):
            normal = self._agent_sprites.get((color, 'normal'))
            done = self._agent_sprites.get((color, 'done'))
            if normal is None or done is None or normal.shape != done.shape:
                continue
            badge_mask = np.any(np.abs(done - normal) > 0.02, axis=-1)
            if not badge_mask.any():
                continue
            for direction in ('up', 'left', 'right'):
                base = self._agent_sprites.get((color, direction))
                if base is None:
                    continue
                variant = base.copy()
                variant[badge_mask] = done[badge_mask]
                self._agent_sprites[(color, f'done_{direction}')] = variant

        # Colored chair sprites: tint the world's primary seat sprite for each
        # color. Falls back to loading the cafe chair directly if the world's
        # own seat sprite wasn't loaded (e.g. non-cafe world with no CHAIR key).
        seat_val = WORLD_SEAT_CELL.get(world_type, CellType.CHAIR.value)
        base_chair = self._cell_sprites.get(seat_val)
        if base_chair is None:
            base_chair = self._cell_sprites.get(CellType.CHAIR.value)
        if base_chair is None:
            base_chair = self._load_and_resize("cells/cafe/chair.png", self.px_per_cell)
        if base_chair is not None:
            bright_chair = _brighten_np(base_chair, FLAT_CHAIR_BRIGHTEN)
            # Plain chairs: every world draws them from its own seat sprite,
            # tinted to the shared walnut brown.
            self._cell_sprites[CellType.CHAIR.value] = _tint_np(
                bright_chair, PLAIN_CHAIR_TINT, PLAIN_CHAIR_TINT_STRENGTH)
            for chair_val, color in CHAIR_COLOR_RGB.items():
                self._cell_sprites[chair_val] = _tint_np(bright_chair, color)

        # Flat chair_up: the base chair sprite faces down (backrest at top), so
        # the up-facing variant (backrest at bottom) is just a vertical flip.
        # This lets a chair face up or down depending on where its table sits.
        self._flat_chair_up = None
        self._flat_chair_up_by_color = {}
        if base_chair is not None:
            bright_chair_up = _brighten_np(
                np.flipud(base_chair).copy(), FLAT_CHAIR_BRIGHTEN)
            self._flat_chair_up = _tint_np(
                bright_chair_up, PLAIN_CHAIR_TINT, PLAIN_CHAIR_TINT_STRENGTH)
            for chair_val, color in CHAIR_COLOR_RGB.items():
                self._flat_chair_up_by_color[chair_val] = _tint_np(
                    bright_chair_up, color)

        # Load LLM star
        self._llm_star = self._load_and_resize("agents/llm_star.png", agent_size // 2)

    def get_cell_sprite(self, cell_type: int) -> Optional[np.ndarray]:
        """Return cached RGBA array for a cell type, or None if not loaded."""
        return self._cell_sprites.get(cell_type)

    def get_agent_sprite(self, agent_class: str, state: str = 'normal') -> Optional[np.ndarray]:
        """Return cached RGBA array for an agent color class, or None if not loaded."""
        return self._agent_sprites.get((agent_class, state))

    def get_flat_chair_up(self, cell_val: int) -> Optional[np.ndarray]:
        """Return the up-facing chair sprite for the flat renderer.

        Returns the tinted variant for colored chairs, the base chair_up for
        plain chairs, or None if the asset wasn't loaded.
        """
        tinted = self._flat_chair_up_by_color.get(cell_val)
        if tinted is not None:
            return tinted
        return self._flat_chair_up

    def get_llm_star(self) -> Optional[np.ndarray]:
        return self._llm_star

    def build_floor_canvas(self, width: int, height: int) -> np.ndarray:
        """Tile the floor texture across a (height*px, width*px) RGB canvas."""
        cache_key = (width, height)
        cached = self._floor_canvas_cache.get(cache_key)
        if cached is not None:
            return cached
        h_px = height * self.px_per_cell
        w_px = width * self.px_per_cell
        canvas = np.ones((h_px, w_px, 3), dtype=np.float32)

        if self._floor_tiles:
            n = len(self._floor_tiles)
            for r in range(height):
                for c in range(width):
                    # Deterministic per-cell variant pick breaks up the
                    # repeating-texture look without any RNG state.
                    tile = self._floor_tiles[(r * 13 + c * 7) % n]
                    tile_rgb = tile[:, :, :3]
                    y0 = r * self.px_per_cell
                    x0 = c * self.px_per_cell
                    canvas[y0:y0 + self.px_per_cell, x0:x0 + self.px_per_cell] = tile_rgb

        self._floor_canvas_cache[cache_key] = canvas
        return canvas

    def build_static_background(self, grid: np.ndarray, x_overlay: set) -> np.ndarray:
        """Floor + X-overlay + furniture composited into one cached RGBA image.

        These layers are static during a run, but save_frame() used to draw them
        as three separate imshow() calls every frame — and matplotlib re-ran the
        expensive RGB->RGBA conversion + resample on each. Compositing them once
        into a single RGBA array (cached) collapses that to one cheap imshow per
        frame, which is the dominant per-frame render cost for the live viewer.
        """
        cache_key = (grid.shape, grid.tobytes(), frozenset(x_overlay))
        cached = self._static_bg_cache.get(cache_key)
        if cached is not None:
            return cached

        h, w = grid.shape
        # Start from the floor as an opaque RGBA base.
        floor_rgb = self.build_floor_canvas(w, h)
        canvas = np.ones((floor_rgb.shape[0], floor_rgb.shape[1], 4), dtype=np.float32)
        canvas[:, :, :3] = floor_rgb

        # Ambient shadows where the floor meets walls (light from the top),
        # grounding the room in 3/4 perspective.
        self._composite_wall_shadows(canvas, grid)

        # X-overlay markers go below furniture.
        if x_overlay:
            x_sprite = self._cell_sprites.get(CellType.AFFORDANCE_X.value)
            if x_sprite is not None:
                for r, c in x_overlay:
                    if 0 <= r < h and 0 <= c < w:
                        self.alpha_composite(canvas, x_sprite,
                                             r * self.px_per_cell, c * self.px_per_cell)

        # Furniture / cell sprites on top (reuse cached cell layer).
        cell_layer = self.build_cell_layer(grid)
        self.alpha_composite(canvas, cell_layer, 0, 0)

        self._static_bg_cache[cache_key] = canvas
        return canvas

    def _edge_shadow_tile(self, side: str, strength: float) -> np.ndarray:
        """Cell-sized RGBA gradient fading inward from one edge (soft AO)."""
        p = self.px_per_cell
        depth = max(2, int(p * 0.30))
        ramp = np.linspace(strength, 0.0, depth, dtype=np.float32)
        alpha = np.zeros((p, p), dtype=np.float32)
        if side == 'top':
            alpha[:depth, :] = ramp[:, None]
        elif side == 'bottom':
            alpha[-depth:, :] = ramp[::-1][:, None]
        elif side == 'left':
            alpha[:, :depth] = ramp[None, :]
        elif side == 'right':
            alpha[:, -depth:] = ramp[::-1][None, :]
        tile = np.zeros((p, p, 4), dtype=np.float32)
        tile[:, :, 0] = 25 / 255.0
        tile[:, :, 1] = 18 / 255.0
        tile[:, :, 2] = 12 / 255.0
        tile[:, :, 3] = alpha
        return tile

    def _composite_wall_shadows(self, canvas: np.ndarray, grid: np.ndarray):
        """Darken floor edges adjacent to walls (light source at the top)."""
        h, w = grid.shape
        wall = CellType.WALL.value
        shadows = {
            'top': self._edge_shadow_tile('top', 0.38),      # wall above casts down
            'left': self._edge_shadow_tile('left', 0.22),
            'right': self._edge_shadow_tile('right', 0.22),
            'bottom': self._edge_shadow_tile('bottom', 0.12),
        }
        for r in range(h):
            for c in range(w):
                if int(grid[r, c]) == wall:
                    continue
                y0, x0 = r * self.px_per_cell, c * self.px_per_cell
                if r > 0 and int(grid[r - 1, c]) == wall:
                    self.alpha_composite(canvas, shadows['top'], y0, x0)
                if r < h - 1 and int(grid[r + 1, c]) == wall:
                    self.alpha_composite(canvas, shadows['bottom'], y0, x0)
                if c > 0 and int(grid[r, c - 1]) == wall:
                    self.alpha_composite(canvas, shadows['left'], y0, x0)
                if c < w - 1 and int(grid[r, c + 1]) == wall:
                    self.alpha_composite(canvas, shadows['right'], y0, x0)

    def build_x_overlay_layer(self, x_overlay: set, grid_shape: tuple) -> np.ndarray:
        """Composite the AFFORDANCE_X sprite at every x_overlay position.

        The resulting layer is intended to be drawn *below* the normal cell layer
        so the X marker appears underneath furniture sprites.
        """
        h, w = grid_shape
        h_px = h * self.px_per_cell
        w_px = w * self.px_per_cell
        canvas = np.zeros((h_px, w_px, 4), dtype=np.float32)
        x_sprite = self._cell_sprites.get(CellType.AFFORDANCE_X.value)
        if x_sprite is None:
            return canvas
        for r, c in x_overlay:
            if 0 <= r < h and 0 <= c < w:
                y0 = r * self.px_per_cell
                x0 = c * self.px_per_cell
                self.alpha_composite(canvas, x_sprite, y0, x0)
        return canvas

    # Wall-like cells for wall-face autotiling: walls above these show their
    # flat top cap, not the vertical face. MENU_BOARD is wall-mounted and
    # bakes its own wall backdrop into the sprite, so it continues the wall.
    _WALL_LIKE = frozenset({
        CellType.WALL.value,
        CellType.MENU_BOARD.value,
    })

    # Cell types that a chair "faces toward" (table-like cells).
    _CHAIR_FACES = frozenset({
        CellType.TABLE_2TOP.value,
        CellType.TABLE_4TOP.value,
        CellType.COUNTER.value,
        CellType.LAPTOP_BAR_COUNTER.value,
    })
    # Chair cell values that support the up-facing variant.
    _CHAIR_VALS = frozenset({
        CellType.CHAIR.value,
        CellType.RED_CHAIR.value,
        CellType.ORANGE_CHAIR.value,
        CellType.BLUE_CHAIR.value,
        CellType.GREEN_CHAIR.value,
    })

    def build_cell_layer(self, grid: np.ndarray) -> np.ndarray:
        """Composite all non-empty cell sprites onto one RGBA canvas."""
        cache_key = (grid.shape, grid.tobytes())
        cached = self._cell_layer_cache.get(cache_key)
        if cached is not None:
            return cached
        h, w = grid.shape
        h_px = h * self.px_per_cell
        w_px = w * self.px_per_cell
        canvas = np.zeros((h_px, w_px, 4), dtype=np.float32)

        for r in range(h):
            for c in range(w):
                cell_val = int(grid[r, c])
                if cell_val == 0:
                    continue  # empty / floor
                sprite = self._cell_sprites.get(cell_val)
                # Wall autotiling: walls whose south neighbor is interior show
                # their vertical face (3/4 perspective); walls buried in the
                # wall mass (or on the bottom edge) show the flat top cap.
                if (cell_val == CellType.WALL.value
                        and self._wall_face is not None
                        and r + 1 < h
                        and int(grid[r + 1, c]) not in self._WALL_LIKE):
                    sprite = self._wall_face
                # Use up-facing chair when a table-like cell is directly above.
                if (cell_val in self._CHAIR_VALS
                        and r > 0
                        and int(grid[r - 1, c]) in self._CHAIR_FACES):
                    up = self.get_flat_chair_up(cell_val)
                    if up is not None:
                        sprite = up
                if sprite is None:
                    continue
                y0 = r * self.px_per_cell
                x0 = c * self.px_per_cell
                self.alpha_composite(canvas, sprite, y0, x0)

        self._cell_layer_cache[cache_key] = canvas
        return canvas

    @staticmethod
    def alpha_composite(canvas: np.ndarray, sprite: np.ndarray, y_px: int, x_px: int):
        """In-place alpha blending of sprite onto canvas at (y_px, x_px)."""
        sh, sw = sprite.shape[:2]
        ch, cw = canvas.shape[:2]

        # Clip to canvas bounds
        y_end = min(y_px + sh, ch)
        x_end = min(x_px + sw, cw)
        y_start = max(y_px, 0)
        x_start = max(x_px, 0)

        sy_start = y_start - y_px
        sx_start = x_start - x_px
        sy_end = sy_start + (y_end - y_start)
        sx_end = sx_start + (x_end - x_start)

        region = canvas[y_start:y_end, x_start:x_end]
        src = sprite[sy_start:sy_end, sx_start:sx_end]

        alpha = src[:, :, 3:4]
        region[:, :, :3] = region[:, :, :3] * (1 - alpha) + src[:, :, :3] * alpha
        region[:, :, 3:4] = np.maximum(region[:, :, 3:4], alpha)

    # ======================= Isometric sprite methods ==========================

    def _load_iso_image(self, rel_path: str) -> Optional[Image.Image]:
        """Load a PNG as a PIL RGBA Image (no resize — iso sprites have varied dimensions)."""
        full_path = os.path.join(self.assets_dir, rel_path)
        if not os.path.isfile(full_path):
            return None
        return Image.open(full_path).convert("RGBA")

    def load_iso_sprites(self, world_type: str):
        """Load isometric sprites for the given world type."""
        self._iso_cell_sprites.clear()
        self._iso_agent_sprites.clear()

        # Shared iso cell sprites
        for cell_val, rel_path in SHARED_SPRITES.items():
            iso_path = "iso/" + rel_path
            img = self._load_iso_image(iso_path)
            if img is not None:
                self._iso_cell_sprites[cell_val] = img

        # World-specific iso cell sprites
        world_map = WORLD_SPRITES.get(world_type, {})
        for cell_val, rel_path in world_map.items():
            iso_path = "iso/" + rel_path
            img = self._load_iso_image(iso_path)
            if img is not None:
                self._iso_cell_sprites[cell_val] = img

        # Iso floor tile
        self._iso_floor_tile = self._load_iso_image("iso/floor/wood_tile.png")

        # Iso agent sprites already carry orange/green/blue body colors baked
        # in by runner/sprites/generate_iso_agents.py — no tinting needed here.
        for key, rel_path in AGENT_SPRITES.items():
            iso_path = "iso/" + rel_path
            img = self._load_iso_image(iso_path)
            if img is not None:
                self._iso_agent_sprites[key] = img

        # Colored chair sprites (iso): tint the base chair sprite per color.
        base_iso_chair = self._iso_cell_sprites.get(CellType.CHAIR.value)
        if base_iso_chair is None:
            # Same fallback as the flat loader: use the world's primary seat
            # sprite for plain chairs when the world has no CHAIR sprite.
            seat_val = WORLD_SEAT_CELL.get(world_type, CellType.CHAIR.value)
            base_iso_chair = self._iso_cell_sprites.get(seat_val)
        if base_iso_chair is not None:
            self._iso_cell_sprites[CellType.CHAIR.value] = _tint_pil(
                base_iso_chair, PLAIN_CHAIR_TINT, PLAIN_CHAIR_TINT_STRENGTH)
            for chair_val, color in CHAIR_COLOR_RGB.items():
                self._iso_cell_sprites[chair_val] = _tint_pil(base_iso_chair,
                                                              color)

        # Iso LLM star
        self._iso_llm_star = self._load_iso_image("iso/agents/llm_star.png")

        # Glazed storefront (renderer swaps this in for camera-facing walls).
        # `storefront` = run ends / standalone (has side posts);
        # `storefront_mid` = run interior (no posts -> seamless facade).
        self._iso_storefront = self._load_iso_image(
            "iso/cells/shared/storefront.png")
        self._iso_storefront_mid = self._load_iso_image(
            "iso/cells/shared/storefront_mid.png")

    def get_iso_cell_sprite(self, cell_type: int) -> Optional[Image.Image]:
        """Return cached isometric PIL Image for a cell type."""
        return self._iso_cell_sprites.get(cell_type)

    def get_iso_agent_sprite(self, agent_class: str, state: str = 'normal') -> Optional[Image.Image]:
        """Return cached isometric PIL Image for an agent color class."""
        return self._iso_agent_sprites.get((agent_class, state))

    def get_iso_floor_tile(self) -> Optional[Image.Image]:
        """Return cached isometric floor tile PIL Image."""
        return self._iso_floor_tile

    def get_iso_llm_star(self) -> Optional[Image.Image]:
        """Return cached isometric LLM star PIL Image."""
        return self._iso_llm_star

    def get_iso_storefront(self, mid: bool = False) -> Optional[Image.Image]:
        """Return the glazed-storefront PIL Image. mid=True returns the
        post-less run-interior variant (falls back to the posted one)."""
        if mid:
            m = getattr(self, "_iso_storefront_mid", None)
            if m is not None:
                return m
        return getattr(self, "_iso_storefront", None)
