#!/usr/bin/env python3
"""Generate flat sprites for affordance cell types (yellow + X)."""

import os
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(ROOT, "assets", "sprites")


def pixels_to_image(pixel_grid, palette, transparent_index=-1, out_size=64):
    h = len(pixel_grid)
    w = len(pixel_grid[0])
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for y, row in enumerate(pixel_grid):
        for x, idx in enumerate(row):
            if idx == transparent_index:
                continue
            r, g, b = palette[idx]
            img.putpixel((x, y), (r, g, b, 255))
    return img.resize((out_size, out_size), Image.NEAREST)


# --- affordance yellow (bright yellow tile) ---
YELLOW_PAL = [
    (255, 220, 50),   # 0 bright yellow
    (230, 195, 30),   # 1 darker yellow
    (255, 240, 100),  # 2 highlight
    (200, 170, 20),   # 3 shadow
]
YELLOW_PX = [
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,2,0,0,0,0,0,0,2,0,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,0,0,0,2,0,0,0,0,0,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,2,0,0,3],
    [3,0,2,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,2,0,0,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,0,0,2,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,2,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,2,0,0,0,0,0,0,0,0,0,2,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
]

# --- affordance X (plain grey tile) ---
X_PAL = [
    (200, 200, 205),  # 0 tile background
    (200, 200, 205),  # 1 (unused, kept for palette compat)
    (220, 220, 225),  # 2 highlight
    (170, 170, 175),  # 3 shadow
]
X_PX = [
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,2,0,0,0,0,0,0,2,0,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,0,0,0,2,0,0,0,0,0,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,2,0,0,3],
    [3,0,2,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,2,0,0,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,0,0,2,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,2,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,2,0,0,0,0,0,0,0,0,0,2,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
]

CELL_DEFS = [
    ("cells/shared", "affordance_yellow.png", YELLOW_PAL, YELLOW_PX),
    ("cells/shared", "affordance_x.png", X_PAL, X_PX),
]


def main():
    print(f"Generating affordance sprites in {ASSETS_DIR}/")
    count = 0
    for subdir, filename, palette, px_grid in CELL_DEFS:
        dirpath = os.path.join(ASSETS_DIR, subdir)
        os.makedirs(dirpath, exist_ok=True)
        filepath = os.path.join(dirpath, filename)
        img = pixels_to_image(px_grid, palette, out_size=64)
        img.save(filepath)
        count += 1
        print(f"  Created {os.path.relpath(filepath, ROOT)}")
    print(f"\nDone! Generated {count} affordance sprites.")


if __name__ == "__main__":
    main()
