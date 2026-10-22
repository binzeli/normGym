#!/usr/bin/env python3
"""Generate a carpet/rug-look sprite for the AFFORDANCE_X cell.

The old sprite was a solid grey diamond covering the whole tile, which read
as an impassable block. This draws an inset rug instead: the wood floor shows
around the edges, with a darker border band and fabric speckles, so the cell
reads as walkable.

Outputs:
    assets/sprites/iso/cells/shared/affordance_x.png  (80x40 isometric)
    assets/sprites/cells/shared/affordance_x.png      (64x64 flat)
"""

import os
import random

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(ROOT, "assets", "sprites")

BODY = (198, 198, 204, 255)       # carpet pile
BORDER = (152, 152, 162, 255)     # binding around the edge
STITCH = (174, 174, 183, 255)     # inner accent line
SPECKLE_LIGHT = (216, 216, 222, 255)
SPECKLE_DARK = (178, 178, 187, 255)
UNDER_SHADOW = (0, 0, 0, 45)      # thin shadow under the front edges


def _diamond(cx, cy, hw, hh, scale):
    return [(cx, cy - hh * scale), (cx + hw * scale, cy),
            (cx, cy + hh * scale), (cx - hw * scale, cy)]


def generate_iso(out_size=(80, 40), ss=4):
    """Isometric rug: inset diamond with border, stitch line and speckles."""
    w, h = out_size[0] * ss, out_size[1] * ss
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy, hw, hh = w / 2, h / 2, w / 2, h / 2

    body_s, border_s, stitch_s = 0.86, 0.86, 0.72
    # Thin shadow peeking out under the two front (lower) edges.
    pts = _diamond(cx, cy + 1.5 * ss, hw, hh, body_s)
    draw.polygon(pts, fill=UNDER_SHADOW)
    # Border band, then body inset inside it.
    draw.polygon(_diamond(cx, cy, hw, hh, border_s), fill=BORDER)
    draw.polygon(_diamond(cx, cy, hw, hh, body_s - 0.07), fill=BODY)
    # Inner stitch line.
    draw.line(_diamond(cx, cy, hw, hh, stitch_s) + [
        _diamond(cx, cy, hw, hh, stitch_s)[0]], fill=STITCH, width=ss)

    # Fabric speckles (deterministic), foreshortened 2:1 like the floor.
    rng = random.Random(7)
    for _ in range(46):
        u = rng.uniform(-0.62, 0.62)
        v = rng.uniform(-0.62, 0.62)
        if abs(u) + abs(v) > 0.62:
            continue
        x, y = cx + u * hw, cy + v * hh
        color = SPECKLE_LIGHT if rng.random() < 0.6 else SPECKLE_DARK
        r = rng.choice([1.0, 1.5]) * ss
        draw.ellipse([x - r, y - r / 2, x + r, y + r / 2], fill=color)

    return img.resize(out_size, Image.LANCZOS)


def generate_flat(out_size=64, ss=4):
    """Flat (top-down) rug: inset rounded square with border and speckles."""
    s = out_size * ss
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    inset, rad = int(s * 0.08), int(s * 0.06)
    draw.rounded_rectangle([inset, inset, s - inset, s - inset],
                           radius=rad, fill=BORDER)
    inner = inset + int(s * 0.045)
    draw.rounded_rectangle([inner, inner, s - inner, s - inner],
                           radius=rad, fill=BODY)
    stitch = inset + int(s * 0.11)
    draw.rounded_rectangle([stitch, stitch, s - stitch, s - stitch],
                           radius=max(1, rad // 2), outline=STITCH, width=ss)

    rng = random.Random(7)
    lo, hi = stitch + 2 * ss, s - stitch - 2 * ss
    for _ in range(40):
        x, y = rng.uniform(lo, hi), rng.uniform(lo, hi)
        color = SPECKLE_LIGHT if rng.random() < 0.6 else SPECKLE_DARK
        r = rng.choice([1.0, 1.5]) * ss
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    return img.resize((out_size, out_size), Image.LANCZOS)


def main():
    iso_path = os.path.join(ASSETS_DIR, "iso", "cells", "shared",
                            "affordance_x.png")
    flat_path = os.path.join(ASSETS_DIR, "cells", "shared", "affordance_x.png")
    generate_iso().save(iso_path)
    print(f"Created {os.path.relpath(iso_path, ROOT)}")
    generate_flat().save(flat_path)
    print(f"Created {os.path.relpath(flat_path, ROOT)}")


if __name__ == "__main__":
    main()
