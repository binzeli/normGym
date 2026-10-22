"""Isometric (2.5D) renderer for normGym grids.

Produces static PNG frames using only PIL -- no matplotlib dependency.
Activated via the --isometric CLI flag.

The scene is supersampled: everything is laid out at `scale` x the 80x40 base
tile and the pixel-art sprites are upscaled nearest-neighbour, so they stay
crisp. Agents are drawn at `agent_px` canvas pixels per art pixel (1.5x the
furniture's pixel size at the default scale of 2), which keeps the characters
in proportion with the furniture instead of towering over it.

A frame is painted in three layers:
  1. floor   -- floor tiles, doormats, shadows, light pools, waiting region,
                and violation highlights;
  2. objects -- furniture, walls and agents in one back-to-front depth order,
                so an agent behind a counter is hidden by it;
  3. overlay -- lamps, violation marks, name tags and the step
                counter, always on top and never occluded.
"""

import math
import os
import re
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from src.env.actions import Action
from src.world.base_world import CellType

_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../assets/fonts/DejaVuSans-Bold.ttf')


def _font(size):
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except (OSError, IOError):
        return ImageFont.load_default(size)


class IsometricRenderer:
    BASE_TILE_W = 80
    BASE_TILE_H = 40

    # Cell values rendered as flat floor-level tiles (no cast contact shadow).
    # empty(0), entrance(2), queue_zone(4), affordance X(64).
    _FLAT_CELLS = frozenset({0, 2, 4, 64})
    _WALL = 1

    SLAB_T = 18          # diorama floor-slab thickness (base px)
    # Default back-wall decor per world; a world can set
    # ISO_WALL_DECOR = {(row, col): kind} instead.
    _DEFAULT_WALL_DECOR = {
        'CafeWorld': {(0, 2): 'frame', (0, 4): 'shelf', (0, 7): 'board',
                      (0, 8): 'board', (0, 10): 'frame_tall', (0, 11): 'shelf'},
        'LibraryWorld': {(0, 2): 'frame', (0, 3): 'frame_tall', (0, 5): 'shelf',
                         (0, 11): 'board', (0, 12): 'frame'},
        'HospitalWorld': {(0, 2): 'cross:red', (0, 6): 'clock', (0, 9): 'frame',
                          (0, 11): 'board'},
        'BarbershopWorld': {(0, 2): 'pole', (0, 5): 'mirror', (0, 8): 'frame', (0, 10): 'pole',
                            (0, 12): 'frame_tall'},
        'PostOfficeWorld': {(0, 3): 'clock', (0, 6): 'board', (0, 9): 'poster', (0, 12): 'frame'},
        'ClothingStoreWorld': {(0, 4): 'mirror', (0, 7): 'poster', (0, 8): 'poster',
                               (0, 12): 'frame_tall'},
        'PharmacyWorld': {(0, 5): 'cross:green', (0, 7): 'board', (0, 9): 'clock'},
        'GroceryStoreWorld': {(0, 5): 'board', (0, 7): 'poster', (0, 9): 'frame'},
        'BankWorld': {(0, 3): 'clock', (0, 5): 'frame', (0, 7): 'frame_tall', (0, 8): 'frame'},
        'MovieTheaterWorld': {(0, 6): 'poster', (0, 8): 'poster', (0, 10): 'poster'},
    }
    _DECOR_ART = [
        ((150, 196, 222), (112, 160, 96), (250, 214, 120)),   # sky / hills / sun
        ((236, 186, 150), (196, 110, 90), (255, 240, 200)),   # dusk
        ((186, 210, 196), (80, 120, 140), (240, 240, 230)),   # lake
    ]

    # Agent class -> accent color (name-tag rim), matching the shirt colors.
    CLASS_RGB = {
        'orange': (240, 146, 52),
        'green': (84, 182, 92),
        'blue': (74, 138, 222),
    }

    def __init__(self, sprite_cache, world, agents_module=None, scale=2,
                 show_hud=True):
        self.sprite_cache = sprite_cache
        # The room is drawn as a floating model on a studio backdrop: each
        # world's own floor, walls and wall decor, the camera-facing walls cut
        # away, a visible floor slab and a soft drop shadow.
        self.show_hud = show_hud
        # Display options: status badges (done / ordering / pushing) beside
        # names; the newcomer's star inside its name tag (its star at its
        # bottom-right is always drawn); and the violation mark -- 'x' (a red
        # X over the violating agent) or 'disc' (a red disc at the cell's
        # front tip).
        self.status_badges = True
        self.tag_star = False
        self.violation_mark = 'x'
        self.world = world
        self.grid_w = world.width   # 15
        self.grid_h = world.height  # 10

        # 'LibraryWorld' -> 'library': selects the world's own diorama art.
        self.world_type = re.sub(r'(?<!^)(?=[A-Z])', '_',
                                 type(world).__name__.replace('World', '')).lower()
        self.S = S = int(scale)
        self.TILE_W = self.BASE_TILE_W * S
        self.TILE_H = self.BASE_TILE_H * S
        # Canvas pixels per agent art pixel: 1.5x the furniture's pixel size.
        self.agent_px = max(1, round(1.5 * S))
        self._scaled = {}

        self.font_name = _font(10 * S)
        self.font_hud = _font(15 * S)
        try:
            self.font_time = ImageFont.truetype(os.path.join(
                os.path.dirname(_FONT_PATH), 'Cantarell-Bold.otf'), 20 * S)
        except (OSError, IOError):
            self.font_time = self.font_hud

        # Canvas dimensions
        self.padding_top = 60 * S
        self.padding_bottom = 40 * S
        self.padding_side = 40 * S

        # x_offset so that (row=grid_h-1, col=0) is at left padding
        self.x_offset = (self.grid_h - 1) * (self.TILE_W // 2) + self.padding_side
        self.y_offset = self.padding_top

        # Compute bounds across all tiles, accounting for actual sprite sizes
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                sx, sy = self.grid_to_screen(r, c)
                # Floor tile bounds
                min_x = min(min_x, sx - self.TILE_W // 2)
                max_x = max(max_x, sx + self.TILE_W // 2)
                min_y = min(min_y, sy)
                max_y = max(max_y, sy + self.TILE_H)

                # Cell sprite bounds (may be taller due to extrusion/backrest)
                cell_val = int(self.world.grid[r, c])
                if cell_val != 0:
                    cell_sprite = self._cell_sprite(cell_val)
                    if cell_sprite is not None:
                        cw, ch = cell_sprite.size
                        paste_x = sx - cw // 2
                        paste_y = sy + self.TILE_H - ch
                        min_x = min(min_x, paste_x)
                        max_x = max(max_x, paste_x + cw)
                        min_y = min(min_y, paste_y)
                        max_y = max(max_y, paste_y + ch)

        self.canvas_w = int(max_x - min_x) + 2 * self.padding_side
        self.canvas_h = int(max_y - min_y) + self.padding_top + self.padding_bottom

        # Adjust offset so min_x lands at padding_side
        self.x_offset = self.x_offset - int(min_x) + self.padding_side
        self.y_offset = self.padding_top - int(min_y)

    # ------------------------------------------------------------------
    # Sprites (upscaled nearest-neighbour, cached)
    # ------------------------------------------------------------------

    def _up(self, img, factor=None):
        """Return `img` upscaled by an integer factor (default: scene scale)."""
        if img is None:
            return None
        f = self.S if factor is None else factor
        if f == 1:
            return img
        key = (id(img), f)
        hit = self._scaled.get(key)
        if hit is None:
            # Keep a reference to the source so its id() stays unique.
            hit = (img, img.resize((img.width * f, img.height * f), Image.NEAREST))
            self._scaled[key] = hit
        return hit[1]

    def _diorama_art(self, rel_path):
        """An optional diorama asset (cached; None when the file is absent)."""
        cache = self.__dict__.setdefault('_diorama_cache', {})
        if rel_path not in cache:
            cache[rel_path] = self.sprite_cache._load_iso_image(rel_path)
        return cache[rel_path]

    def _edges(self, r, c):
        """Which outer walls the border cell (r, c) lies on."""
        sides = []
        if c == 0:
            sides.append('left')
        if r == 0:
            sides.append('top')
        if r == self.grid_h - 1:
            sides.append('bottom')
        if c == self.grid_w - 1:
            sides.append('right')
        return sides

    def _door_cells(self):
        cache = getattr(self, '_door_cache', None)
        if cache is None:
            cache = self._door_cache = frozenset(
                (int(r), int(c)) for r, c in getattr(self.world, 'entrances', ())
                if self._edges(int(r), int(c)))
        return cache

    def _wall_art(self, prefix, side):
        """A door / wall-stub sprite for this world (the cafe's as fallback)."""
        art = (self._diorama_art(f"iso/cells/shared/{prefix}_{self.world_type}_{side}.png")
               or self._diorama_art(f"iso/cells/shared/{prefix}_cafe_{side}.png"))
        return self._up(art) if art is not None else None

    def _door_art(self, r, c):
        """A doorway in a back wall (left / top), or a framed gap in the
        knee-high stub of a cut-away wall (bottom / right)."""
        side = self._edges(r, c)[0]
        prefix = 'door' if side in ('left', 'top') else 'doorstub'
        art = self._wall_art(prefix, side)
        return [art] if art is not None else []

    def _stub_art(self, r, c):
        arts = [self._wall_art('stub', side) for side in self._edges(r, c)
                if side in ('bottom', 'right')]
        return [a for a in arts if a is not None]

    def _cell_sprite(self, cell_val):
        if cell_val == self._WALL:
            wall = (self._diorama_art(f"iso/cells/shared/wall_{self.world_type}.png")
                    or self._diorama_art("iso/cells/shared/wall_plaster.png"))
            if wall is not None:
                return self._up(wall)
        if cell_val == CellType.CHAIR.value:
            # Plain chairs are tinted brown for the flat view; the diorama
            # shows the world's own seat (e.g. the library's armchair).
            from src.env.sprite_loader import WORLD_SEAT_CELL
            seat = WORLD_SEAT_CELL.get(self.world_type)
            if seat is not None and seat != CellType.CHAIR.value:
                art = self.sprite_cache.get_iso_cell_sprite(seat)
                if art is not None:
                    return self._up(art)
        return self._up(self.sprite_cache.get_iso_cell_sprite(cell_val))

    def _agent_sprite(self, agent_class, key):
        return self._up(self.sprite_cache.get_iso_agent_sprite(agent_class, key),
                        self.agent_px)

    def grid_to_screen(self, row, col):
        """Convert grid (row, col) to screen pixel coords (top-left of diamond tile)."""
        x = (col - row) * (self.TILE_W // 2) + self.x_offset
        y = (col + row) * (self.TILE_H // 2) + self.y_offset
        return x, y

    def _grid_vec(self, dr, dc):
        """Screen-space vector of a grid displacement (dr, dc)."""
        return ((dc - dr) * (self.TILE_W / 2), (dc + dr) * (self.TILE_H / 2))

    def _diamond_points(self, cx, cy, s=1.0):
        """Return the 4 vertices of a tile diamond centered at (cx, cy),
        shrunk by factor `s`."""
        hw = self.TILE_W / 2 * s
        hh = self.TILE_H / 2 * s
        return [
            (cx, cy - hh),      # top
            (cx + hw, cy),      # right
            (cx, cy + hh),      # bottom
            (cx - hw, cy),      # left
        ]

    # ------------------------------------------------------------------
    # Lighting / ambiance
    # ------------------------------------------------------------------

    def _soft_shadow(self, canvas, cx, cy, rx, ry, alpha=95, blur=4):
        """Composite a blurred dark ellipse (contact / ambient-occlusion shadow)
        centered at canvas pixel (cx, cy)."""
        pad = blur * 3
        w = int(2 * rx + 2 * pad)
        h = int(2 * ry + 2 * pad)
        if w <= 0 or h <= 0:
            return
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.ellipse([pad, pad, w - pad, h - pad], fill=(18, 15, 12, alpha))
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
        px, py = int(cx - w / 2), int(cy - h / 2)
        px = max(0, min(px, self.canvas_w - w))
        py = max(0, min(py, self.canvas_h - h))
        if w > self.canvas_w or h > self.canvas_h:
            return
        canvas.alpha_composite(layer, (px, py))

    def _light_pool(self, canvas, cx, cy, rx, ry, color=(255, 209, 150),
                    max_alpha=46, blur=10):
        """Composite a soft warm glow (a pendant-light pool) on the floor."""
        pad = blur * 3
        w = int(2 * rx + 2 * pad)
        h = int(2 * ry + 2 * pad)
        if w <= 0 or h <= 0 or w > self.canvas_w or h > self.canvas_h:
            return
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.ellipse([pad, pad, w - pad, h - pad],
                   fill=(color[0], color[1], color[2], max_alpha))
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
        px, py = int(cx - w / 2), int(cy - h / 2)
        px = max(0, min(px, self.canvas_w - w))
        py = max(0, min(py, self.canvas_h - h))
        canvas.alpha_composite(layer, (px, py))

    def _floor_variant(self, r, c):
        """Return the floor tile with a small deterministic brightness shift
        per (r, c) so the wood doesn't read as one repeated stamp."""
        base = (self._diorama_art(f"iso/floor/{self.world_type}.png")
                or self.sprite_cache.get_iso_floor_tile())
        if base is None:
            return None
        cache = getattr(self, "_floor_var_cache", None)
        if cache is None:
            # Tight range: boards differ subtly, never patchy.
            factors = (0.96, 0.985, 1.0, 1.015, 1.04)
            base = self._up(base)
            cache = [ImageEnhance.Brightness(base).enhance(f) for f in factors]
            self._floor_var_cache = cache
        # Key by row only: a board runs along constant r (the +col axis),
        # so the whole board keeps one tone -> plank continuity preserved.
        idx = (r * 3 + 1) % len(cache)
        return cache[idx]

    def _room_corners(self):
        """Screen corners (top, right, bottom, left) of the whole floor."""
        hw, hh = self.TILE_W // 2, self.TILE_H // 2
        x, y = self.grid_to_screen(0, 0)
        top = (x, y)
        x, y = self.grid_to_screen(0, self.grid_w - 1)
        right = (x + hw, y + hh)
        x, y = self.grid_to_screen(self.grid_h - 1, self.grid_w - 1)
        bottom = (x, y + 2 * hh)
        x, y = self.grid_to_screen(self.grid_h - 1, 0)
        left = (x - hw, y + hh)
        return top, right, bottom, left

    def _studio_backdrop(self):
        """Diorama backdrop: a soft warm-grey studio gradient with the room's
        blurred drop shadow under where the floor slab sits."""
        S = self.S
        w, h = self.canvas_w, self.canvas_h
        col = Image.new("RGBA", (1, h))
        cp = col.load()
        top_c, bot_c = (241, 237, 231), (218, 212, 203)
        for y in range(h):
            t = y / max(1, h - 1)
            c = tuple(int(top_c[i] + (bot_c[i] - top_c[i]) * t) for i in range(3))
            cp[0, y] = c + (255,)
        bg = col.resize((w, h))
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        drop = self.SLAB_T * S + 18 * S
        pts = [(x, y + drop) for x, y in self._room_corners()]
        ImageDraw.Draw(shadow).polygon(pts, fill=(60, 48, 36, 90))
        shadow = shadow.filter(ImageFilter.GaussianBlur(26 * S))
        bg.alpha_composite(shadow)
        return bg

    def _draw_slab(self, draw):
        """The floor's visible thickness along the two camera-facing edges:
        a wooden plank edge over a darker base."""
        S = self.S
        T, band = self.SLAB_T * S, 5 * S
        _, right, bottom, left = self._room_corners()
        for a, b, wood, base in ((left, bottom, (178, 130, 88), (104, 80, 62)),
                                 (bottom, right, (150, 108, 72), (84, 64, 50))):
            draw.polygon([a, b, (b[0], b[1] + band), (a[0], a[1] + band)], fill=wood + (255,))
            draw.polygon([(a[0], a[1] + band), (b[0], b[1] + band),
                          (b[0], b[1] + T), (a[0], a[1] + T)], fill=base + (255,))
            draw.line([(a[0], a[1] + T), (b[0], b[1] + T)], fill=(60, 46, 36, 255), width=S)

    def _storefront(self, mid):
        """Storefront window sprite, its grey roof cap
        is recolored to the plaster walls' walnut cap."""
        src = self.sprite_cache.get_iso_storefront(mid=mid)
        if src is None:
            return None
        cache = self.__dict__.setdefault('_storefront_cache', {})
        if mid not in cache:
            img = src.copy()
            px = img.load()
            for y in range(min(img.height, 52)):
                for x in range(img.width):
                    r, g, b, a = px[x, y]
                    in_cap = abs(x - 40) / 40 + abs(y - 27) / 21 <= 1.02
                    if a and in_cap and abs(r - g) < 10 and abs(g - b) < 10:
                        f = r / 150
                        px[x, y] = (min(255, int(116 * f)), min(255, int(86 * f)),
                                    min(255, int(62 * f)), a)
            cache[mid] = img
        return self._up(cache[mid])

    def _wall_face_point(self, r, c, u, v):
        """Canvas point on the room-facing (+row) face of back-wall cell
        (r, c): u runs 0..1 left to right along the face, v 0..1 top to
        bottom (geometry of the 80x132 wall sprite)."""
        S = self.S
        sx, sy = self.grid_to_screen(r, c)
        ox, oy = sx - 40 * S, sy - 92 * S
        return (ox + 40 * S * u, oy + (24 + 20 * u) * S + 88 * S * v)

    def _face_quad(self, r, c, u0, v0, u1, v1):
        p = self._wall_face_point
        return [p(r, c, u0, v0), p(r, c, u1, v0), p(r, c, u1, v1), p(r, c, u0, v1)]

    def _draw_wall_decor(self, draw, r, c, kind):
        """Pixel-art decor painted on a back-wall face."""
        q = lambda *a: self._face_quad(r, c, *a)
        if kind in ('frame', 'frame_tall'):
            u0, u1, v0, v1 = (0.24, 0.76, 0.12, 0.44) if kind == 'frame' else (0.32, 0.68, 0.08, 0.50)
            draw.polygon(q(u0, v0, u1, v1), fill=(86, 60, 42, 255))
            du, dv = 0.05, 0.035
            draw.polygon(q(u0 + du, v0 + dv, u1 - du, v1 - dv), fill=(240, 232, 214, 255))
            a0, a1, b0, b1 = u0 + 2 * du, u1 - 2 * du, v0 + 2 * dv, v1 - 2 * dv
            mid = b0 + (b1 - b0) * 0.55
            palette = self._DECOR_ART[(r * 7 + c) % len(self._DECOR_ART)]
            draw.polygon(q(a0, b0, a1, mid), fill=palette[0] + (255,))
            draw.polygon(q(a0, mid, a1, b1), fill=palette[1] + (255,))
            # a sun / accent dot
            cx, cy = self._wall_face_point(r, c, a0 + (a1 - a0) * 0.68, b0 + (mid - b0) * 0.45)
            rr = 2.5 * self.S
            draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=palette[2] + (255,))
        elif kind == 'shelf':
            draw.polygon(q(0.14, 0.38, 0.86, 0.42), fill=(120, 84, 56, 255))
            for u, pot, leaf in ((0.24, (196, 112, 76), (76, 150, 82)),
                                 (0.50, (232, 226, 214), (60, 128, 70)),
                                 (0.72, (110, 140, 170), (92, 168, 96))):
                draw.polygon(q(u - 0.07, 0.29, u + 0.07, 0.38), fill=pot + (255,))
                cx, cy = self._wall_face_point(r, c, u, 0.25)
                rx, ry = 5 * self.S, 4.5 * self.S
                draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=leaf + (255,))
        elif kind == 'board':
            draw.polygon(q(0.18, 0.10, 0.82, 0.48), fill=(92, 64, 44, 255))
            draw.polygon(q(0.23, 0.14, 0.77, 0.44), fill=(46, 62, 56, 255))
            for v in (0.21, 0.28, 0.35):
                draw.polygon(q(0.30, v, 0.68, v + 0.018), fill=(226, 232, 222, 255))
        elif kind == 'clock':
            pt = lambda u, v: self._wall_face_point(r, c, u, v)
            ring = [pt(0.5 + 0.2 * math.cos(a), 0.27 + 0.13 * math.sin(a))
                    for a in [i * math.pi / 12 for i in range(24)]]
            face = [pt(0.5 + 0.16 * math.cos(a), 0.27 + 0.1 * math.sin(a))
                    for a in [i * math.pi / 12 for i in range(24)]]
            draw.polygon(ring, fill=(70, 52, 40, 255))
            draw.polygon(face, fill=(246, 242, 232, 255))
            draw.line([pt(0.5, 0.27), pt(0.5, 0.2)], fill=(40, 34, 30, 255), width=self.S)
            draw.line([pt(0.5, 0.27), pt(0.6, 0.27)], fill=(40, 34, 30, 255), width=self.S)
        elif kind.startswith('cross'):
            color = {'green': (52, 150, 96), 'red': (206, 56, 52)}.get(kind.split(':')[-1], (206, 56, 52))
            draw.polygon(q(0.3, 0.12, 0.7, 0.42), fill=color + (255,))
            draw.polygon(q(0.45, 0.16, 0.55, 0.38), fill=(250, 250, 250, 255))
            draw.polygon(q(0.36, 0.23, 0.64, 0.31), fill=(250, 250, 250, 255))
        elif kind == 'poster':
            palette = [((40, 50, 90), (230, 160, 60)), ((30, 70, 96), (240, 210, 120)),
                       ((30, 90, 90), (236, 200, 110)), ((60, 40, 90), (120, 200, 230))]
            bg, accent = palette[(r * 5 + c) % len(palette)]
            draw.polygon(q(0.26, 0.06, 0.74, 0.52), fill=(214, 172, 80, 255))
            draw.polygon(q(0.3, 0.09, 0.7, 0.49), fill=bg + (255,))
            draw.polygon(q(0.36, 0.16, 0.64, 0.34), fill=accent + (255,))
            draw.polygon(q(0.34, 0.39, 0.66, 0.43), fill=(244, 240, 228, 255))
        elif kind == 'pole':
            draw.polygon(q(0.42, 0.06, 0.58, 0.1), fill=(200, 204, 212, 255))
            draw.polygon(q(0.42, 0.48, 0.58, 0.52), fill=(200, 204, 212, 255))
            draw.polygon(q(0.44, 0.1, 0.56, 0.48), fill=(246, 246, 246, 255))
            for i in range(6):
                v0 = 0.1 + i * 0.065
                color = (206, 48, 48) if i % 2 == 0 else (52, 86, 170)
                draw.polygon([self._wall_face_point(r, c, 0.44, v0 + 0.03),
                              self._wall_face_point(r, c, 0.56, v0),
                              self._wall_face_point(r, c, 0.56, v0 + 0.025),
                              self._wall_face_point(r, c, 0.44, v0 + 0.055)], fill=color + (255,))
        elif kind == 'mirror':
            draw.polygon(q(0.22, 0.08, 0.78, 0.5), fill=(150, 112, 70, 255))
            draw.polygon(q(0.26, 0.11, 0.74, 0.47), fill=(196, 222, 232, 255))
            draw.polygon([self._wall_face_point(r, c, 0.34, 0.4), self._wall_face_point(r, c, 0.46, 0.16),
                          self._wall_face_point(r, c, 0.52, 0.16), self._wall_face_point(r, c, 0.4, 0.4)],
                         fill=(232, 244, 250, 255))

    def _make_background(self):
        """The studio backdrop (cached per canvas size)."""
        cache = getattr(self, "_bg_cache", None)
        if cache is None or cache.size != (self.canvas_w, self.canvas_h):
            cache = self._bg_cache = self._studio_backdrop()
        return cache.copy()

    # ------------------------------------------------------------------
    # World-defined cell sets
    # ------------------------------------------------------------------

    def _window_cells(self):
        """Frozenset of (row, col) WALL cells drawn as glazed storefront.

        Uses the world's `ISO_WINDOW_CELLS` if defined (cafe — editable in
        the world file); otherwise falls back to the original rule (any
        WALL cell on a camera-facing run) so other worlds are unchanged.
        """
        cache = getattr(self, "_winset_cache", None)
        if cache is not None:
            return cache
        explicit = getattr(self.world, "ISO_WINDOW_CELLS", None)
        if explicit is not None:
            cache = frozenset(
                (int(r), int(c)) for (r, c) in explicit
                if 0 <= r < self.grid_h and 0 <= c < self.grid_w
                and int(self.world.grid[r, c]) == self._WALL)
        else:
            cache = frozenset(
                (r, c)
                for r in range(self.grid_h) for c in range(self.grid_w)
                if int(self.world.grid[r, c]) == self._WALL
                and (r == self.grid_h - 1 or c == self.grid_w - 1))
        # Windows on a hidden (cut-away) wall would vanish: put them on the
        # left wall instead, as in the cafe.
        hidden = self._hidden_wall_cells()
        visible = frozenset(c for c in cache if c not in hidden)
        if not visible:
            visible = frozenset(
                (r, 0) for r in range(1, self.grid_h - 1)
                if int(self.world.grid[r, 0]) == self._WALL)
        cache = visible
        self._winset_cache = cache
        return cache

    def _hidden_wall_cells(self):
        """Frozenset of (row, col) WALL cells the renderer should NOT draw.

        Uses the world's `ISO_HIDDEN_WALL_CELLS` if defined (editable in the
        world file) — lets a world open up a wall run (e.g. the camera-facing
        right edge) so the view inside isn't blocked. In addition, the front
        (bottom-row) wall is always hidden so it never occludes the interior in
        the iso view. The wall still exists logically (boundary / collision);
        only its sprite + AO are skipped, so the open floor shows through."""
        cache = getattr(self, "_hidden_wall_cache", None)
        if cache is not None:
            return cache
        cells = set()
        explicit = getattr(self.world, "ISO_HIDDEN_WALL_CELLS", None)
        if explicit:
            cells.update(
                (int(r), int(c)) for (r, c) in explicit
                if 0 <= r < self.grid_h and 0 <= c < self.grid_w
                and int(self.world.grid[r, c]) == self._WALL)
        # Always hide the front (bottom) wall row so it doesn't block the view.
        bottom = self.grid_h - 1
        cells.update(
            (bottom, c) for c in range(self.grid_w)
            if int(self.world.grid[bottom, c]) == self._WALL)
        if not explicit:
            # The diorama is a cut-away: the right wall faces the camera too
            # (worlds that set ISO_HIDDEN_WALL_CELLS, like the cafe, choose).
            right = self.grid_w - 1
            cells.update(
                (r, right) for r in range(self.grid_h)
                if int(self.world.grid[r, right]) == self._WALL)
        cache = frozenset(cells)
        self._hidden_wall_cache = cache
        return cache

    def _lamp_cells(self):
        """List of (row, col) pendant-lamp anchors. Uses the world's
        `ISO_LAMPS` if defined, else the original cafe defaults."""
        lamps = getattr(self.world, "ISO_LAMPS", None)
        if lamps is None:
            lamps = [(2, 6), (1, 12), (4, 3)]
        return [(int(r), int(c)) for (r, c) in lamps]

    def _waiting_region_cells(self):
        """Frozen set of (row, col) cells in the waiting region (visual only).
        Uses the world's `WAITING_ZONE` if defined, else empty."""
        cache = getattr(self, "_waiting_region_cache", None)
        if cache is not None:
            return cache
        cells = getattr(self.world, "WAITING_ZONE", None) or ()
        cache = frozenset((int(r), int(c)) for (r, c) in cells
                          if 0 <= r < self.grid_h and 0 <= c < self.grid_w)
        self._waiting_region_cache = cache
        return cache

    # ------------------------------------------------------------------
    # Floor-level overlays
    # ------------------------------------------------------------------

    def _draw_waiting_region(self, draw):
        """Draw a dashed yellow border around the waiting-region cells and a
        small footprint pair inside each cell. Pure visual overlay."""
        region = self._waiting_region_cells()
        if not region:
            return
        S = self.S
        hw = self.TILE_W // 2
        hh = self.TILE_H // 2
        yellow = (255, 210, 40, 235)
        yellow_dark = (190, 150, 20, 200)
        # Neighbor (dr, dc) -> edge endpoints on the diamond:
        #   (-1, 0) top-right edge   : top -> right
        #   ( 0,+1) bottom-right edge: right -> bottom
        #   (+1, 0) bottom-left edge : bottom -> left
        #   ( 0,-1) top-left edge    : left -> top
        edges = {
            (-1, 0): ('top', 'right'),
            (0, 1): ('right', 'bottom'),
            (1, 0): ('bottom', 'left'),
            (0, -1): ('left', 'top'),
        }
        for (r, c) in region:
            sx, sy = self.grid_to_screen(r, c)
            cx, cy = sx, sy + hh
            verts = {
                'top': (cx, cy - hh),
                'right': (cx + hw, cy),
                'bottom': (cx, cy + hh),
                'left': (cx - hw, cy),
            }
            # Dashed perimeter — only on edges facing outside the region.
            for (dr, dc), (a, b) in edges.items():
                if (r + dr, c + dc) in region:
                    continue
                self._draw_dashed_line(draw, verts[a], verts[b], yellow,
                                       width=2 * S, dash=6 * S, gap=4 * S)

            # Footprints (two small ovals) centered on the cell.
            self._draw_footprints(draw, cx, cy, yellow_dark)

    def _draw_dashed_line(self, draw, p1, p2, color, width=2, dash=6, gap=4):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length <= 0:
            return
        ux, uy = dx / length, dy / length
        step = dash + gap
        n = int(length // step) + 1
        for i in range(n):
            s = i * step
            e = min(s + dash, length)
            if s >= length:
                break
            xs, ys = x1 + ux * s, y1 + uy * s
            xe, ye = x1 + ux * e, y1 + uy * e
            draw.line([(xs, ys), (xe, ye)], fill=color, width=width)

    def _draw_footprints(self, draw, cx, cy, color):
        # Two small ovals angled along the iso "forward" axis, sitting on the
        # tile floor. Sized to read as a footprint pair without crowding.
        S = self.S
        foot_w, foot_h = 5 * S, 3 * S
        # Offset feet a few pixels apart along the iso "depth" axis.
        offs = [(-4 * S, S), (4 * S, -S)]
        for ox, oy in offs:
            x0 = cx + ox - foot_w // 2
            y0 = cy + oy - foot_h // 2
            draw.ellipse([x0, y0, x0 + foot_w, y0 + foot_h], fill=color)

    def _draw_status_badge(self, draw, cx, cy, state):
        """Draw a small status badge centered at (cx, cy) reflecting agent state.

        Mirrors the top-down status cues: a green disc + white check when the
        agent is done, a white clock face with a blue rim while it is being
        served (ordering), and a red disc + white "!" when pushing.
        'normal'/'waiting' draw nothing.
        """
        if state == 'done':
            fill = (52, 199, 89, 255)       # green
        elif state == 'order':
            fill = (244, 247, 252, 255)     # clock face
        elif state == 'push':
            fill = (255, 69, 58, 255)       # red
        else:
            return

        S = self.S
        r = 8 * S
        # Disc with a ring so it reads against any background (blue for the
        # clock, as in the top-down view; white otherwise).
        ring = (58, 112, 204, 255) if state == 'order' else (255, 255, 255, 255)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=ring, width=2 * S)
        white = (255, 255, 255, 255)
        w = 2 * S

        if state == 'done':
            # Check mark.
            draw.line([(cx - 4 * S, cy), (cx - S, cy + 4 * S), (cx + 5 * S, cy - 4 * S)],
                      fill=white, width=w, joint='curve')
        elif state == 'order':
            # Clock hands (pointing to 12 and 3).
            blue = (58, 112, 204, 255)
            draw.line([(cx, cy), (cx, cy - 5 * S)], fill=blue, width=w)
            draw.line([(cx, cy), (cx + 4 * S, cy)], fill=blue, width=w)
        elif state == 'push':
            # Exclamation mark.
            draw.line([(cx, cy - 5 * S), (cx, cy + S)], fill=white, width=w)
            draw.ellipse([cx - S, cy + 3 * S, cx + S, cy + 5 * S], fill=white)

    def _draw_star(self, draw, cx, cy, r, fill, outline):
        """Five-pointed star centered at (cx, cy) with outer radius r."""
        pts = []
        for i in range(10):
            rad = r if i % 2 == 0 else r * 0.45
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
        draw.polygon(pts, fill=fill, outline=outline)

    def _tag_size(self, text, newcomer):
        """(width, height) of an agent's name pill."""
        S = self.S
        l, t, rr, b = self.font_name.getbbox(text)
        star_w = 11 * S if newcomer else 0
        return rr - l + 12 * S + star_w, b - t + 6 * S

    def _draw_name_tag(self, draw, cx, bottom, text, accent, newcomer, state):
        """Rounded name pill centered above an agent: dark glass with a rim in
        the agent's class color, a gold star for the newcomer (non-oracle)
        agent and the status badge on its right end."""
        S = self.S
        font = self.font_name
        l, t, _, _ = font.getbbox(text)
        pad_x, pad_y = 6 * S, 3 * S
        star_w = 11 * S if newcomer else 0
        w, h = self._tag_size(text, newcomer)
        x0 = int(cx - w / 2)
        y0 = int(bottom - h)
        draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=h // 2,
                               fill=(30, 28, 36, 215), outline=accent + (255,),
                               width=max(1, S))
        tx = x0 + pad_x + star_w
        if newcomer:
            self._draw_star(draw, x0 + pad_x + 4 * S, y0 + h / 2, 5 * S,
                            fill=(255, 205, 40, 255), outline=(150, 100, 0, 255))
        draw.text((tx - l, y0 + pad_y - t), text, fill=(255, 255, 255, 255), font=font)
        self._draw_status_badge(draw, x0 + w + 7 * S, y0 + h / 2, state)

    def _layout_tags(self, tags):
        """Lift name tags that would overlap so every name stays readable.

        Tags are placed greedily by priority -- the newcomer first, then
        agents nearer the camera -- each one moved up past any tag already
        placed that it would cover. Returns (priority-ordered) tuples
        (cx, bottom, natural_bottom, agent, state, newcomer).
        """
        S = self.S
        entries = []
        for fx, bottom, agent, state in tags:
            newcomer = getattr(agent, 'show_star', not agent.is_oracle)
            if not self.status_badges:
                state = 'normal'
            w, h = self._tag_size(agent.name, newcomer and self.tag_star)
            badge = 16 * S if state in ('done', 'order', 'push') else 0
            entries.append((not newcomer, -bottom, fx, bottom, w, h, badge,
                            agent, state, newcomer))
        entries.sort(key=lambda e: (e[0], e[1]))
        gap = 2 * S
        placed, out = [], []
        for _, _, fx, natural, w, h, badge, agent, state, newcomer in entries:
            x0, x1 = fx - w / 2 - gap, fx + w / 2 + badge + gap
            bottom = natural
            for _ in range(len(placed) + 1):
                hit = [b for b in placed
                       if x0 < b[2] and b[0] < x1 and bottom - h - gap < b[3] and b[1] < bottom + gap]
                if not hit:
                    break
                bottom = min(b[1] for b in hit) - gap
            placed.append((x0, bottom - h, x1, bottom))
            out.append((fx, bottom, natural, agent, state, newcomer))
        return out

    def _draw_hud(self, draw, label_step):
        """The time step, "t = 22", at the top-left."""
        S = self.S
        halo = (241, 237, 231, 255)          # the studio backdrop
        draw.text((26 * S, 20 * S), f"t = {label_step}", font=self.font_time,
                  fill=(44, 38, 33, 255), stroke_width=2 * S, stroke_fill=halo)

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    VIOLATION_RGB = (222, 30, 30)

    def _agent_state(self, agent, is_subframe):
        """Status of an agent for its badge (mirror the flat renderer's logic).

        On sub-frames the agent is animating the PREVIOUS step's move, so use
        its previous_* state — otherwise the order/done badge would appear a
        step early (e.g. orange before reaching the counter). Push badge ("!")
        is deferred by one step: it reflects previous_action, not the freshly
        resolved action. So on the step two agents first land on the same cell
        (action just became PUSH) no "!" shows; it appears the next step, when
        the pusher actually displaces the other agent.
        """
        eff_push = getattr(agent, 'previous_action', None) == Action.PUSH
        if is_subframe:
            eff_done = getattr(agent, 'previous_done', agent.done)
            eff_target_index = getattr(agent, 'previous_target_index', getattr(agent, 'target_index', 0))
            eff_target_wait_time = getattr(agent, 'previous_target_wait_time', getattr(agent, 'target_wait_time', 0))
        else:
            eff_done = agent.done
            eff_target_index = getattr(agent, 'target_index', 0)
            eff_target_wait_time = getattr(agent, 'target_wait_time', 0)

        if eff_push:
            return 'push'
        if eff_done or eff_target_index >= 1:
            return 'done'
        if eff_target_wait_time != 0:
            return 'order'
        if (self.world.map_type == 'affordance'
                and agent.active and not agent.done
                and hasattr(agent, 'target_position') and agent.target_position
                and agent.current_position == agent.target_position):
            return 'waiting'
        return 'normal'

    def _flat_star(self):
        """The top-down view's star sprite, sized to half the iso agent height.

        Upscaled nearest-neighbour by a whole factor so the pixel art stays
        crisp (24 px art -> ~14 agent art pixels tall).
        """
        if '_flat_star_img' not in self.__dict__:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../assets/sprites/agents/llm_star.png')
            try:
                art = Image.open(path).convert('RGBA')
                f = max(1, round(14 * self.agent_px / art.height))
                self._flat_star_img = self._up(art, f)
            except (OSError, IOError):
                self._flat_star_img = None
        return self._flat_star_img

    def _agent_image(self, agent, state, is_subframe):
        """The agent's sprite for this frame (facing its movement direction)."""
        # waiting falls back to normal sprite with orange tint applied
        key = 'normal' if state == 'waiting' else state
        # Walking agents face their movement direction (same convention as
        # the top-down view: sub-frames show the animating action). Moves
        # away from the camera (up-right / up-left on screen) show the back;
        # moves toward it show the matching side profile.
        if key in ('normal', 'done'):
            facing_action = agent.previous_action if is_subframe else agent.action
            direction = {
                Action.MOVE_UP: 'up',
                Action.MOVE_LEFT: 'up',
                Action.MOVE_DOWN: 'left',
                Action.MOVE_RIGHT: 'right',
            }.get(facing_action)
            if direction is not None:
                key = direction
        img = self._agent_sprite(agent.agent_class, key)
        if img is None and key in ('up', 'left', 'right'):
            img = self._agent_sprite(agent.agent_class, 'normal')
        if img is not None and state == 'waiting':
            # Tint only the character, not its transparent padding.
            base = img.convert('RGBA')
            img = Image.blend(base, Image.new('RGBA', base.size, (255, 165, 0, 255)), 0.45)
            img.putalpha(base.getchannel('A'))
        return img

    def _layout_agents(self, agents, step, interp_positions):
        """Screen placement of every visible agent.

        Returns a list of dicts with the agent, its feet point (fx, fy), a
        walking bob offset and a painter's-order depth.
        """
        is_subframe = interp_positions is not None
        position_to_agents: Dict[Tuple[float, float], List[Any]] = {}
        for agent in agents:
            if not (agent.active or getattr(agent, 'arrival_time', None) == step):
                continue
            # During interpolation, draw the agent at its float sub-frame
            # position so movement is smooth; otherwise its integer cell.
            if is_subframe and agent.agent_id in interp_positions:
                pos = interp_positions[agent.agent_id]
            else:
                pos = agent.current_position
            position_to_agents.setdefault(pos, []).append(agent)

        S = self.S
        placed = []
        for (r, c), group in position_to_agents.items():
            num = len(group)
            if num == 1:
                offsets = [(0, 0)]
            elif num == 2:
                offsets = [(-11 * S, 0), (11 * S, 0)]
            elif num == 3:
                offsets = [(-15 * S, 0), (0, 0), (15 * S, 0)]
            else:
                offsets = [(int(20 * S * math.cos(2 * math.pi * i / num)),
                            int(10 * S * math.sin(2 * math.pi * i / num)))
                           for i in range(num)]
            sx, sy = self.grid_to_screen(r, c)
            # Mid-move: a small two-step walking bob.
            fr, fc = r - math.floor(r), c - math.floor(c)
            t = fr if fr > 1e-6 else fc
            bob = int(round(1.5 * self.agent_px * abs(math.sin(2 * math.pi * t))))
            # Draw after the object on the cell being entered/left, whichever
            # is nearer the camera.
            depth = math.ceil(r + c - 1e-6)
            for (ox, oy), agent in zip(offsets, group):
                placed.append({
                    'agent': agent,
                    'fx': int(sx + ox),
                    'fy': int(sy + self.TILE_H // 2 + 5 * S + oy),
                    'bob': bob,
                    'depth': depth + oy * 1e-4,
                })
        return placed

    # ------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------

    def render_frame(self, agents, step, step_data=None, queried_agent_label=None,
                     interp_positions=None,
                     display_step=None, push_pairs_override=None):
        """Render one simulation frame as a PIL Image.

        Args:
            agents: List of agent objects
            step: Current step number
            step_data: Dict with 'actions', 'norm_violations', 'push_pairs', etc.
            queried_agent_label: Agent label to hide arrows for
            interp_positions: Optional {agent_id: (row, col)} float positions for
                smooth sub-frame animation. When given, agents are drawn at these
                interpolated positions instead of their integer current_position.
            display_step: Step number to show in the label (defaults to `step`).
            push_pairs_override: Optional push pairs to draw instead of step_data's.

        Returns:
            PIL.Image.Image (RGBA)
        """
        S = self.S
        is_subframe = interp_positions is not None
        label_step = display_step if display_step is not None else step
        canvas = self._make_background()
        draw = ImageDraw.Draw(canvas)
        hidden_walls = self._hidden_wall_cells()
        x_overlay = getattr(self.world, 'x_overlay', set())

        placed = self._layout_agents(agents, step, interp_positions)
        feet = {p['agent'].agent_id: (p['fx'], p['fy']) for p in placed}
        # Violation marks appear on keyframes only, once the agent has arrived
        # on the cell -- never on the sub-frames animating its move there
        # (the flat renderer and web viewer follow the same rule).
        norm_violations = (step_data.get('norm_violations', [])
                           if step_data and not is_subframe else [])

        # ============ 1. Floor layer ============
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                sx, sy = self.grid_to_screen(r, c)
                cx = sx
                cy = sy + self.TILE_H // 2  # center of diamond

                # Floor tile (deterministic per-tile shade variation)
                floor_sprite = self._floor_variant(r, c)
                if floor_sprite is not None:
                    # Wood-base diamond underneath: sub-pixel seams between
                    # adjacent tiles then show wood, not the bright background.
                    draw.polygon(self._diamond_points(cx, cy),
                                 fill=(197, 162, 119, 255))
                    canvas.alpha_composite(floor_sprite,
                                           (sx - floor_sprite.width // 2, sy))
                else:
                    draw.polygon(self._diamond_points(cx, cy),
                                 fill=(230, 210, 180, 255), outline=(200, 180, 150, 255))

                # X-overlay sits underneath any furniture on the cell.
                if (r, c) in x_overlay:
                    x_sprite = self._cell_sprite(CellType.AFFORDANCE_X.value)
                    if x_sprite is not None:
                        canvas.alpha_composite(
                            x_sprite, (sx - x_sprite.width // 2,
                                       sy + self.TILE_H - x_sprite.height))

                cell_val = int(self.world.grid[r, c])
                if cell_val == 2:  # ENTRANCE -> doormat
                    mat = self._diamond_points(cx, cy, 0.60)
                    draw.polygon(mat, fill=(122, 84, 56, 255),
                                 outline=(78, 52, 33, 255), width=S)
                    draw.polygon(self._diamond_points(cx, cy, 0.45),
                                 outline=(150, 110, 78, 255), width=S)
                if cell_val in self._FLAT_CELLS and cell_val != 0:
                    flat = self._cell_sprite(cell_val)
                    if flat is not None:
                        canvas.alpha_composite(
                            flat, (sx - flat.width // 2, sy + self.TILE_H - flat.height))

                # Contact shadow / wall ambient occlusion, on the floor so
                # objects feel grounded.
                if cell_val == self._WALL:
                    if (r, c) in hidden_walls:
                        continue

                    # Only "back" walls (interior floor toward the viewer at
                    # r+1 or c+1) cast AO — keeps shadows off the exterior.
                    def _interior_floor(rr, cc):
                        return (0 <= rr < self.grid_h and 0 <= cc < self.grid_w
                                and int(self.world.grid[rr, cc]) != self._WALL)
                    if _interior_floor(r + 1, c) or _interior_floor(r, c + 1):
                        self._soft_shadow(canvas, cx,
                                          cy + int(self.TILE_H * 0.55),
                                          int(self.TILE_W * 0.40),
                                          int(self.TILE_H * 0.32),
                                          alpha=55, blur=7 * S)
                elif cell_val not in self._FLAT_CELLS:
                    self._soft_shadow(canvas, cx, cy + 4 * S,
                                      int(self.TILE_W * 0.32),
                                      int(self.TILE_H * 0.30),
                                      alpha=95, blur=4 * S)

        self._draw_slab(draw)

        floor = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        fdraw = ImageDraw.Draw(floor)
        self._draw_waiting_region(fdraw)
        # Norm violations: the offending destination cell glows red.
        for violation in norm_violations:
            vr, vc = violation[2]   # (agent_id, from_pos, to_pos)
            vx, vy = self.grid_to_screen(vr, vc)
            rgb = self.VIOLATION_RGB
            fdraw.polygon(self._diamond_points(vx, vy + self.TILE_H // 2, 0.92),
                          fill=rgb + (90,), outline=rgb + (230,),
                          width=2 * S)
        canvas.alpha_composite(floor)

        # Pendant-light pools: warm glow on the floor beneath each lamp.
        lamp_cells = self._lamp_cells()
        for lr, lc in lamp_cells:
            lsx, lsy = self.grid_to_screen(lr, lc)
            self._light_pool(canvas, lsx, lsy + self.TILE_H // 2, 72 * S, 42 * S,
                             max_alpha=48, blur=10 * S)

        # Agent contact shadows.
        for p in placed:
            self._soft_shadow(canvas, p['fx'], p['fy'], 12 * S, 5 * S,
                              alpha=120, blur=2 * S)

        # ============ 2. Objects + agents, back to front ============
        items = []
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                cell_val = int(self.world.grid[r, c])
                if (r, c) in self._door_cells():
                    for art in self._door_art(r, c):
                        items.append((r + c, 0, (r, c, None, art)))
                    continue
                if cell_val in self._FLAT_CELLS:
                    continue
                if cell_val == self._WALL and (r, c) in hidden_walls:
                    # A cut-away wall leaves a knee-high stub on its outer edge.
                    for art in self._stub_art(r, c):
                        items.append((r + c, 0, (r, c, None, art)))
                    continue
                items.append((r + c, 0, (r, c, cell_val)))
        for p in placed:
            items.append((p['depth'], 1, p))
        items.sort(key=lambda it: (it[0], it[1]))

        tags = []   # name tags, drawn in the overlay layer
        stars = []  # newcomer stars (sprite, x, y), also in the overlay layer
        for _, kind, item in items:
            if kind == 0 and len(item) == 4:
                r, c, _, art = item
                sx, sy = self.grid_to_screen(r, c)
                canvas.alpha_composite(art, (sx - art.width // 2, sy + self.TILE_H - art.height))
                continue
            if kind == 0:
                r, c, cell_val = item
                sx, sy = self.grid_to_screen(r, c)
                cell_sprite = self._cell_sprite(cell_val)
                # Glazed-storefront windows (set defined on the world —
                # editable in the world file).
                if cell_val == self._WALL:
                    winset = self._window_cells()
                    if (r, c) in winset:
                        up, dn = (r - 1, c) in winset, (r + 1, c) in winset
                        lf, rt = (r, c - 1) in winset, (r, c + 1) in winset
                        # Post-less mid only in the interior of a straight
                        # run; corners / ends keep their framing posts.
                        mid = (up and dn) or (lf and rt)
                        storefront = self._storefront(mid)
                        if storefront is not None:
                            cell_sprite = storefront
                if cell_sprite is not None:
                    # Center horizontally on tile, bottom-align with diamond bottom.
                    canvas.alpha_composite(
                        cell_sprite, (sx - cell_sprite.width // 2,
                                      sy + self.TILE_H - cell_sprite.height))
                    if cell_val == self._WALL:
                        decor = getattr(self.world, 'ISO_WALL_DECOR', None)
                        if decor is None:
                            decor = self._DEFAULT_WALL_DECOR.get(type(self.world).__name__, {})
                        if (r, c) in decor and (r, c) not in self._window_cells():
                            self._draw_wall_decor(draw, r, c, decor[(r, c)])
                else:
                    draw.polygon(self._diamond_points(sx, sy + self.TILE_H // 2),
                                 fill=self._cell_fallback_color(cell_val),
                                 outline=(0, 0, 0, 80))
                continue

            agent = item['agent']
            fx, fy = item['fx'], item['fy'] - item['bob']
            state = self._agent_state(agent, is_subframe)
            img = self._agent_image(agent, state, is_subframe)
            if img is not None:
                top = fy - img.height
                canvas.alpha_composite(img, (fx - img.width // 2, top))
                if getattr(agent, 'show_star', not agent.is_oracle):
                    # The newcomer's star: the top-down view's sprite, in the
                    # same proportions (half the agent's height, centred
                    # 0.35 agent-heights right of and below its centre).
                    star = self._flat_star()
                    if star is not None:
                        off = round(0.35 * img.height)
                        scx, scy = fx + off, top + img.height // 2 + off
                        stars.append((star, scx - star.width // 2, scy - star.height // 2))
            else:
                # Fallback: colored circle
                r_size = 12 * S
                top = fy - 2 * r_size
                draw.ellipse([fx - r_size, top, fx + r_size, fy],
                             fill=(30, 144, 255, 255), outline=(0, 0, 0, 255), width=2 * S)
            tags.append((fx, top - 4 * S, agent, state))

        # ============ 3. Overlay ============
        # Pendant lamp fixtures hang above everything.
        for lr, lc in lamp_cells:
            lsx, lsy = self.grid_to_screen(lr, lc)
            px = lsx
            cyl = lsy + self.TILE_H // 2
            # No ceiling in the diorama: the cords stop just above the walls.
            y_ceiling = cyl - 112 * S
            y_shade_top = cyl - 80 * S
            y_shade_bot = cyl - 62 * S
            draw.line([(px, y_ceiling), (px, y_shade_top)],
                      fill=(46, 38, 30, 255), width=2 * S)
            draw.polygon([(px - 13 * S, y_shade_bot), (px + 13 * S, y_shade_bot),
                          (px + 6 * S, y_shade_top), (px - 6 * S, y_shade_top)],
                         fill=(58, 44, 32, 255), outline=(36, 27, 19, 255))
            draw.line([(px - 13 * S, y_shade_bot), (px + 13 * S, y_shade_bot)],
                      fill=(96, 74, 52, 255), width=S)

        # Warm bulbs + glow.
        for lr, lc in lamp_cells:
            lsx, lsy = self.grid_to_screen(lr, lc)
            px = lsx
            y_shade_bot = lsy + self.TILE_H // 2 - 62 * S
            self._light_pool(canvas, px, y_shade_bot + 3 * S,
                             16 * S, 16 * S, color=(255, 226, 168), max_alpha=120,
                             blur=6 * S)
            draw.ellipse([px - 3 * S, y_shade_bot - S, px + 3 * S, y_shade_bot + 5 * S],
                         fill=(255, 236, 196, 255))


        ui = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        udraw = ImageDraw.Draw(ui)
        # Newcomer stars sit in the overlay so walls and doorframes never hide them.
        for star, x, y in stars:
            ui.alpha_composite(star, (x, y))

        # Norm violations: a red "x" disc at the front tip of the cell, or a
        # bare red X over the violating agent's body.
        for violation in norm_violations:
            vr, vc = violation[2]
            vx, vy = self.grid_to_screen(vr, vc)
            if self.violation_mark == 'x':
                fx, fy = feet.get(violation[0], (vx, vy + self.TILE_H // 2 + 5 * S))
                bx, by = fx, fy - 14 * self.agent_px      # middle of the body
                k = 12 * S
                rgb = self.VIOLATION_RGB + (255,)
                udraw.line([(bx - k, by - k), (bx + k, by + k)], fill=rgb, width=5 * S)
                udraw.line([(bx - k, by + k), (bx + k, by - k)], fill=rgb, width=5 * S)
                continue
            bx, by = vx, vy + self.TILE_H - 4 * S
            rad = 15 * S
            udraw.ellipse([bx - rad, by - rad, bx + rad, by + rad],
                          fill=(225, 35, 35, 255), outline=(255, 255, 255, 255), width=3 * S)
            k = 6.5 * S
            udraw.line([(bx - k, by - k), (bx + k, by + k)], fill=(255, 255, 255, 255), width=3 * S)
            udraw.line([(bx - k, by + k), (bx + k, by - k)], fill=(255, 255, 255, 255), width=3 * S)

        # Name tags (with status badge and newcomer star), de-overlapped and
        # drawn lowest priority first so the newcomer's tag ends up on top.
        # `show_star` lets a caller mark an agent without changing its type
        # (is_oracle also drives env logic), e.g. a demo's norm-unaware agent.
        for fx, bottom, natural, agent, state, newcomer in reversed(self._layout_tags(tags)):
            accent = self.CLASS_RGB.get(agent.agent_class, (200, 200, 200))
            if natural - bottom > 2 * S:
                # Lifted tag: a thin leader line back down to the agent's head.
                udraw.line([(fx, bottom), (fx, natural)], fill=(30, 28, 36, 170), width=S)
            self._draw_name_tag(udraw, fx, bottom, agent.name, accent,
                                newcomer=newcomer and self.tag_star, state=state)

        if self.show_hud:
            self._draw_hud(udraw, label_step)
        canvas.alpha_composite(ui)
        return canvas

    def _cell_fallback_color(self, cell_val):
        """Return an RGBA fallback color for a cell type when no sprite is available."""
        colors = {
            1: (95, 95, 95, 255),       # wall
            2: (99, 185, 106, 255),     # entrance
            3: (213, 125, 60, 255),     # counter
            4: (232, 203, 104, 255),    # queue
            6: (201, 79, 79, 255),      # chair
            7: (138, 90, 59, 255),      # table
            8: (138, 90, 59, 255),      # table
            10: (207, 213, 219, 255),   # kitchen
            11: (181, 106, 58, 255),    # laptop bar
            12: (59, 59, 92, 255),      # screen room
            13: (201, 168, 76, 255),    # rope barrier
            14: (122, 79, 138, 255),    # theater seat
            15: (230, 168, 50, 255),    # concession
            16: (232, 201, 74, 255),    # snack stand
            17: (74, 74, 74, 255),      # vault
            18: (90, 143, 175, 255),    # ATM
            19: (201, 168, 76, 255),    # rope stanchion
            20: (107, 142, 107, 255),   # writing desk
            21: (230, 168, 50, 255),    # deposit slips
            22: (138, 106, 90, 255),    # waiting chair
            23: (207, 213, 219, 255),   # stockroom
            24: (74, 168, 201, 255),    # medicine shelf
            25: (126, 200, 168, 255),   # display case
            26: (201, 90, 106, 255),    # BP station
            27: (230, 168, 50, 255),    # Rx pickup
            28: (207, 213, 219, 255),   # storage room
            29: (90, 154, 90, 255),     # product shelf
            30: (138, 138, 138, 255),   # cart corral
            31: (230, 168, 50, 255),    # bagging area
            32: (219, 170, 130, 255),  # pastry display
            33: (139, 90, 60, 255),    # coffee grinder
            34: (60, 60, 60, 255),     # menu board
            35: (100, 100, 110, 255),  # espresso machine
        }
        return colors.get(cell_val, (200, 200, 200, 255))

    def save_frame(self, path, agents, step, step_data=None, queried_agent_label=None,
                   interp_positions=None,
                   display_step=None, push_pairs_override=None):
        """Render and save a frame to disk."""
        img = self.render_frame(agents, step, step_data, queried_agent_label,
                                interp_positions=interp_positions,
                                display_step=display_step,
                                push_pairs_override=push_pairs_override)
        img.save(path, "PNG")
