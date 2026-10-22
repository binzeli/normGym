"""Generate the isometric interior wall sprite (cream plaster over a wooden
wainscot, walnut cap) that replaces the grey corrugated block in the iso view.

Same 80x132 footprint as iso/cells/shared/wall.png: a diamond cap with top
vertex (40, 4) / side vertices (0|80, 24) / bottom vertex (40, 44), and two
88 px faces below it. Drawn pixel by pixel so it matches the pixel-art props.

Writes assets/sprites/iso/cells/shared/wall_plaster.png.
Usage (from the repo root):  python runner/sprites/generate_iso_walls.py
"""
import os

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'assets/sprites/iso/cells/shared/wall_plaster.png')

W, H, FACE = 80, 132, 88
PLASTER = (236, 224, 204)
PLASTER_TOP = (226, 212, 190)      # a touch darker right under the cap
RAIL = (146, 104, 70)
WOOD = (172, 122, 80)
WOOD_LINE = (142, 98, 62)
BASE = (98, 70, 48)
CAP = (116, 86, 62)
CAP_EDGE = (80, 58, 42)
RIGHT_SHADE = 0.84                 # the right face turns away from the light


def shade(c, f):
    return tuple(max(0, min(255, round(v * f))) for v in c)


img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
px = img.load()
for x in range(W):
    left = x < W // 2
    y_top = 24 + x // 2 if left else 44 - (x - W // 2) // 2
    f = 1.0 if left else RIGHT_SHADE
    for y in range(y_top, min(H, y_top + FACE)):
        v = (y - y_top) / FACE
        if v < 0.04:
            c = PLASTER_TOP
        elif v < 0.58:
            c = PLASTER
        elif v < 0.62:
            c = RAIL
        elif v < 0.95:
            c = WOOD_LINE if x % 6 == 0 else WOOD
        else:
            c = BASE
        px[x, y] = shade(c, f) + (255,)

d = ImageDraw.Draw(img)
d.polygon([(40, 4), (80, 24), (40, 44), (0, 24)], fill=CAP + (255,), outline=CAP_EDGE + (255,))
img.save(OUT)
print('wrote', OUT, img.size)
