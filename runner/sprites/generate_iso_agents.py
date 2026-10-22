"""Build the isometric agent sprites from the top-down human sprites.

The iso view used to draw agents as coloured shapes (square / circle /
triangle). This derives human-looking iso sprites from the same pixel-art
characters the top-down renderer uses (assets/sprites/agents/), so an agent
looks the same in both views:

  * the baked-in floor shadow is dropped (the iso renderer casts its own),
  * the status badges are dropped (the iso renderer draws its own badge
    above the name), but the push pose (arms out) is kept,
  * the art is cropped to the character and saved at one pixel per art
    pixel (the source is drawn in 2x2 blocks); the renderer upscales it with
    nearest-neighbour so it stays crisp at any integer scale.

Writes assets/sprites/iso/agents/<color>{,_up,_left,_right,_order,_done,_push}.png.

Usage (from the repo root):  python runner/sprites/generate_iso_agents.py
"""
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "assets/sprites/agents")
DST = os.path.join(ROOT, "assets/sprites/iso/agents")
COLORS = ("orange", "green", "blue")
BADGE_ROWS = 28   # source rows above this hold the status badge, below it the arms


def load(name):
    arr = np.array(Image.open(os.path.join(SRC, name)).convert("RGBA"))
    arr[arr[..., 3] < 255] = 0          # the floor shadow is the only translucent art
    return arr


def save(arr, name):
    img = Image.fromarray(arr[::2, ::2])
    img = img.crop(img.getbbox())
    img.save(os.path.join(DST, name))
    return img.size


for color in COLORS:
    normal = load(f"{color}.png")
    for suffix in ("", "_up", "_left", "_right"):
        print(color + suffix, save(load(f"{color}{suffix}.png"), f"{color}{suffix}.png"))

    # Push: keep the outstretched arms, drop the "!" badge.
    push = load(f"{color}_push.png")
    pose = normal.copy()
    arms = np.any(push != normal, axis=-1)
    arms[:BADGE_ROWS] = False
    pose[arms] = push[arms]
    print(color + "_push", save(pose, f"{color}_push.png"))

    # Order / done are shown by the renderer's badge; the body is the plain pose.
    for state in ("order", "done"):
        save(normal, f"{color}_{state}.png")
