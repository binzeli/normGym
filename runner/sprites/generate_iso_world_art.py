"""Generate the isometric (diorama) art for the non-cafe worlds.

Every prop is drawn in code with runner/sprites/iso_art.py, in the style of the
hand-drawn cafe props (outlines in a darker shade of each surface, three-tone
faces, inset panels, small props on top). Per world this writes:

    assets/sprites/iso/cells/<world>/<prop>.png   one per cell type
    assets/sprites/iso/floor/<world>.png          the floor tile
    assets/sprites/iso/cells/shared/wall_<world>.png  the wall block

The file names follow the top-down sprites in sprite_loader.WORLD_SPRITES,
so the iso loader picks them up without any mapping changes.

Usage (from the repo root):  python runner/sprites/generate_iso_world_art.py [world ...]
"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iso_art import (Sprite, book_row, mix, rng_for, shade)  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISO = os.path.join(ROOT, 'assets/sprites/iso')

# Height rule (base pixels above the floor). An agent is ~42 px tall, and one
# standing behind a prop must stay visible: solid props (shelves, cabinets,
# bins, chair backs) stop at SOLID_MAX, below the shoulders; thin parts
# (monitors, lamps) may reach THIN_MAX, still below the head.
SOLID_MAX = 34
THIN_MAX = 40
DESK_Z = 22


def out(world, name):
    d = os.path.join(ISO, 'cells', world)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

def floor_tile(path, base, line, pattern='planks', accent=None):
    """An 80x40 floor diamond: 'planks' | 'carpet' | 'tiles' | 'checker' | 'marble'."""
    img = Image.new('RGBA', (80, 40), (0, 0, 0, 0))
    px = img.load()
    rng = rng_for(path)
    for y in range(40):
        for x in range(80):
            # inside the diamond?
            if abs(x - 39.5) / 40 + abs(y - 19.5) / 20 > 1.0:
                continue
            u = (x / 40 + y / 20) / 2          # tile coords
            v = (y / 20 - x / 40) / 2 + 0.5
            c = base
            if pattern == 'planks':
                band = int(v * 4)
                c = shade(base, (0.96, 1.0, 0.98, 1.03)[band % 4])
                if abs(v * 4 - round(v * 4)) < 0.035:
                    c = line
            elif pattern == 'carpet':
                n = rng.random()
                c = shade(base, 0.95 + 0.1 * n)
                if accent and (abs(u - 0.5) < 0.04 or abs(v - 0.5) < 0.04):
                    c = mix(c, accent, 0.16)
            elif pattern == 'tiles':
                if abs(u * 2 - round(u * 2)) < 0.03 or abs(v * 2 - round(v * 2)) < 0.03:
                    c = line
                else:
                    c = shade(base, 0.98 + 0.04 * ((int(u * 2) + int(v * 2)) % 2))
            elif pattern == 'marble':
                n = rng.random()
                c = shade(base, 0.97 + 0.05 * n)
                vein = abs((u * 1.7 + v * 0.6 + 0.15 * ((u * 7) % 1)) % 0.5 - 0.25) < 0.012
                if vein:
                    c = line
                if abs(u * 2 - round(u * 2)) < 0.012 or abs(v * 2 - round(v * 2)) < 0.012:
                    c = shade(line, 1.04)
            elif pattern == 'checker':
                c = base if (int(u * 2) + int(v * 2)) % 2 == 0 else (accent or line)
            px[x, y] = tuple(c) + (255,)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


WALL_Z = 88            # height of a wall block's faces (base px)
STUB_Z = 14            # knee-wall stub along the cut-away (camera-facing) edges
DOOR_Z = 64            # doorway height in the back walls
TRIM = (96, 66, 44)    # door-frame wood


def door_set(world, plaster, wainscot, cap):
    """The door art for one world, all in iso/cells/shared/:

    door_<world>_left.png / _top.png   a doorway in a back wall (left wall:
        the wall runs along rows; top wall: along columns): full-height
        jambs and lintel with wood trim, open through to the outside;
    stub_<world>_bottom.png / _right.png   the knee-high wall left where a
        cut-away wall stood (filling the wall cell, like the 2D view);
    doorstub_<world>_bottom.png / _right.png   a door in that wall row: a gap
        the size of the cell, framed by two short wood posts (the door mat
        shows through it).
    """
    d = os.path.join(ISO, 'cells/shared')
    wall_c = mix(plaster, wainscot, 0.25)

    def doorway(along_v):
        sp = Sprite(40 + WALL_Z + 6)
        # jambs (a, b) along the wall run, lintel across the top
        if along_v:   # left wall: runs along v, room side is u = 1
            parts = [(0, 1, 0.0, 0.18), (0, 1, 0.82, 1.0)]
            lintel = (0, 1, 0.18, 0.82)
        else:         # top wall: runs along u, room side is v = 1
            parts = [(0.0, 0.18, 0, 1), (0.82, 1.0, 0, 1)]
            lintel = (0.18, 0.82, 0, 1)
        sp.slab(*(lintel if along_v else lintel), 0.5, shade(TRIM, 0.9))   # threshold
        for u0, u1, v0, v1 in parts:
            sp.box(u0, u1, v0, v1, 0, WALL_Z, wall_c, top_color=cap)
        sp.box(*lintel, DOOR_Z, WALL_Z, wall_c, top_color=cap)
        # wood trim round the opening on the room side
        if along_v:
            sp.box(0.86, 1.0, 0.14, 0.2, 0, DOOR_Z + 2, TRIM)
            sp.box(0.86, 1.0, 0.8, 0.86, 0, DOOR_Z + 2, TRIM)
            sp.box(0.86, 1.0, 0.14, 0.86, DOOR_Z, DOOR_Z + 4, TRIM)
        else:
            sp.box(0.14, 0.2, 0.86, 1.0, 0, DOOR_Z + 2, TRIM)
            sp.box(0.8, 0.86, 0.86, 1.0, 0, DOOR_Z + 2, TRIM)
            sp.box(0.14, 0.86, 0.86, 1.0, DOOR_Z, DOOR_Z + 4, TRIM)
        return sp

    doorway(True).save(os.path.join(d, f'door_{world}_left.png'))
    doorway(False).save(os.path.join(d, f'door_{world}_top.png'))

    for side in ('bottom', 'right'):
        sp = Sprite(40 + STUB_Z + 4)
        sp.box(0.0, 1.0, 0.0, 1.0, 0, STUB_Z, wainscot, top_color=cap)
        sp.save(os.path.join(d, f'stub_{world}_{side}.png'))

        sp = Sprite(40 + STUB_Z + 16)
        if side == 'bottom':      # the wall row runs along u
            posts = [(0.0, 0.14, 0.0, 1.0), (0.86, 1.0, 0.0, 1.0)]
        else:                     # the wall column runs along v
            posts = [(0.0, 1.0, 0.0, 0.14), (0.0, 1.0, 0.86, 1.0)]
        for q in posts:
            sp.box(*q, 0, STUB_Z, wainscot, top_color=cap)
            sp.box(q[0] + 0.01, q[1] - 0.01, q[2] + 0.01, q[3] - 0.01, STUB_Z, STUB_Z + 12, TRIM)
        sp.save(os.path.join(d, f'doorstub_{world}_{side}.png'))


def wall_block(path, plaster, wainscot, cap, rail=None, plank=6):
    """Same footprint as the cafe's wall_plaster.png (80x132); also writes
    the world's door and wall-stub art (door_set)."""
    world = os.path.basename(path)[len('wall_'):-len('.png')]
    door_set(world, plaster, wainscot, cap)
    W, H, FACE = 80, 132, 88
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    px = img.load()
    rail = rail or shade(wainscot, 0.8)
    for x in range(W):
        left = x < W // 2
        y_top = 24 + x // 2 if left else 44 - (x - W // 2) // 2
        f = 1.0 if left else 0.84
        for y in range(y_top, min(H, y_top + FACE)):
            v = (y - y_top) / FACE
            if v < 0.04:
                c = shade(plaster, 0.95)
            elif v < 0.58:
                c = plaster
            elif v < 0.62:
                c = rail
            elif v < 0.95:
                c = shade(wainscot, 0.85) if x % plank == 0 else wainscot
            else:
                c = shade(wainscot, 0.6)
            px[x, y] = shade(c, f) + (255,)
    from PIL import ImageDraw
    ImageDraw.Draw(img).polygon([(40, 4), (80, 24), (40, 44), (0, 24)], fill=tuple(cap) + (255,),
                                outline=shade(cap, 0.7) + (255,))
    img.save(path)


def desk(sp, color, z=26, u0=0.1, u1=0.9, v0=0.22, v1=0.78, legs=True, panel=False):
    """A desk/table top at height z with legs (or a closed front panel)."""
    if panel:
        b = sp.box(u0, u1, v0, v1, 0, z - 4, shade(color, 0.9))
        b.panel('left', 0.08, 0.92, 0.12, 0.85, shade(color, 0.9))
        b.panel('right', 0.1, 0.9, 0.12, 0.85, shade(color, 0.9))
    elif legs:
        for (u, v) in ((u0 + 0.05, v1 - 0.05), (u1 - 0.05, v1 - 0.05), (u1 - 0.05, v0 + 0.05)):
            sp.leg(u, v, 0, z - 4, shade(color, 0.75))
    return sp.box(u0 - 0.02, u1 + 0.02, v0 - 0.02, v1 + 0.02, z - 4, z, color)


def armchair(sp, fabric, wood=(110, 72, 46)):
    """An upholstered armchair, back on the far (top-right) side."""
    for (u, v) in ((0.2, 0.8), (0.8, 0.8), (0.8, 0.22)):
        sp.leg(u, v, 0, 6, wood)
    sp.box(0.14, 0.86, 0.16, 0.86, 6, 16, shade(fabric, 0.9))          # base
    seat = sp.box(0.2, 0.8, 0.3, 0.84, 16, 20, shade(fabric, 1.05))   # cushion
    seat.top_rect(0.1, 0.9, 0.15, 0.85, shade(fabric, 1.22))
    sp.box(0.14, 0.86, 0.14, 0.3, 16, SOLID_MAX, fabric)               # back
    sp.box(0.12, 0.24, 0.3, 0.86, 16, 26, shade(fabric, 0.95))          # left arm
    sp.box(0.76, 0.88, 0.3, 0.86, 16, 26, shade(fabric, 0.95))          # right arm


# -- more shared builders ---------------------------------------------------

STOCK = {
    # kind: (palette, widths, height range)
    'boxes': ([(214, 86, 64), (64, 132, 196), (236, 196, 72), (96, 172, 104), (236, 236, 228),
               (180, 110, 196), (240, 150, 64)], (4, 5, 6), (0.55, 0.92)),
    'bottles': ([(196, 132, 60), (236, 236, 228), (84, 160, 104), (80, 130, 200), (214, 96, 84)],
                (2, 3, 3), (0.45, 0.85)),
    'cans': ([(206, 60, 52), (70, 150, 84), (236, 190, 60), (64, 110, 190), (230, 120, 50)],
             (3, 3, 4), (0.42, 0.62)),
    'produce': ([(214, 60, 48), (240, 160, 40), (120, 180, 60), (236, 214, 70), (150, 70, 140)],
                (4, 5), (0.35, 0.55)),
    'candy': ([(64, 132, 196), (236, 196, 72), (96, 172, 104), (236, 236, 228), (180, 110, 196),
               (120, 200, 230)], (4, 5, 6), (0.55, 0.92)),
    'parcels': ([(190, 146, 98), (176, 132, 86), (204, 162, 112)], (6, 7, 9), (0.45, 0.9)),
    'folded': ([(232, 188, 196), (176, 206, 226), (236, 232, 220), (120, 140, 176), (204, 120, 110),
                (150, 186, 150)], (6, 7, 8), (0.22, 0.42)),
    'shoes': ([(60, 48, 44), (176, 68, 60), (236, 232, 224), (120, 84, 56), (70, 100, 160)],
              (5, 6), (0.22, 0.32)),
    'magazines': ([(214, 70, 70), (70, 130, 200), (240, 196, 70), (120, 180, 110), (200, 110, 190)],
                  (3, 4), (0.7, 0.95)),
}


def stock_row(box, face, s0, s1, t0, t1, rng, kind):
    """Goods standing on a shelf (inside face rect s0..s1 x t0..t1)."""
    palette, widths, (h0, h1) = STOCK[kind]
    face_px = 40 * abs(box.u1 - box.u0) if face == 'left' else 40 * abs(box.v1 - box.v0)
    span = (s1 - s0) * face_px
    x = 0.0
    while x < span - 1.5:
        w = rng.choice(widths)
        if x + w > span:
            break
        if rng.random() < 0.06:
            x += w
            continue
        c = rng.choice(palette)
        h = rng.uniform(h0, h1)
        a, b = s0 + (s1 - s0) * x / span, s0 + (s1 - s0) * (x + w) / span
        tt = t0 + (t1 - t0) * h
        box.rect(face, a, b, t0, tt, c)
        box.vline(face, b, shade(c, 0.6), t0, tt)
        box.hline(face, tt, shade(c, 1.25), a, b)
        mid = t0 + (tt - t0) * 0.5
        if kind in ('boxes', 'cans', 'candy'):
            box.hline(face, mid, (250, 248, 240) if kind == 'boxes' else (210, 214, 220), a, b)
        elif kind == 'bottles':
            box.hline(face, tt, shade(c, 0.5), a, b)                     # cap
            box.hline(face, mid, (250, 246, 236), a, b)                  # label
        elif kind == 'parcels':
            box.vline(face, (a + b) / 2, (226, 204, 160), t0, tt)       # tape
        elif kind == 'folded':
            box.hline(face, t0 + (tt - t0) * 0.5, shade(c, 0.78), a, b)
        elif kind == 'magazines':
            box.hline(face, t0 + (tt - t0) * 0.75, (250, 248, 240), a, b)
        x += w + (1 if kind in ('produce', 'shoes', 'parcels') else 0)


def shelf_unit(sp, frame, rng, kind, z=SOLID_MAX, tiers=2, u0=0.06, u1=0.94, v0=0.06, v1=0.94,
               back=None, faces=('left', 'right')):
    """Open shelving stocked with `kind` goods on the camera-facing faces."""
    unit = sp.box(u0, u1, v0, v1, 0, z, frame)
    bounds = [(0.1 + i * 0.86 / tiers, 0.1 + (i + 1) * 0.86 / tiers - 0.06) for i in range(tiers)]
    for face in faces:
        unit.rect(face, 0.06, 0.94, 0.06, 0.94, back or shade(frame, 0.55))
        for t0, t1 in bounds:
            stock_row(unit, face, 0.08, 0.92, t0, t1, rng, kind)
        for t0, _ in bounds + [(0.96, None)]:
            unit.rect(face, 0.05, 0.95, t0 - 0.045, t0 + 0.0, shade(frame, 1.15))
        unit.vline(face, 0.05, shade(frame, 0.85))
        unit.vline(face, 0.95, shade(frame, 0.65))
    unit.top_rect(0.08, 0.92, 0.08, 0.92, shade(frame, 1.18))
    return unit


def cabinet(sp, color, z=SOLID_MAX - 4, u0=0.1, u1=0.9, v0=0.18, v1=0.82, style='doors',
            knob=(214, 180, 96)):
    """A closed cabinet with door or drawer panels on both visible faces."""
    b = sp.box(u0, u1, v0, v1, 0, z, color)
    for face in ('left', 'right'):
        base = shade(color, 0.94 if face == 'left' else 0.74)
        if style == 'doors':
            b.panel(face, 0.07, 0.48, 0.1, 0.9, base)
            b.panel(face, 0.52, 0.93, 0.1, 0.9, base)
            for s in (0.42, 0.58):
                b.rect(face, s - 0.02, s + 0.02, 0.48, 0.58, knob)
        else:
            for t0, t1 in ((0.08, 0.36), (0.38, 0.64), (0.66, 0.92)):
                b.panel(face, 0.08, 0.92, t0, t1, base)
                b.rect(face, 0.44, 0.56, (t0 + t1) / 2 - 0.03, (t0 + t1) / 2 + 0.03, knob)
    return b


def bench(sp, seat, frame=(96, 96, 104), back=True, u0=0.08, u1=0.92, slats=3):
    """A bench along +col, its back on the far (top-right) side."""
    for u in (u0 + 0.06, u1 - 0.06):
        sp.leg(u, 0.74, 0, 12, frame)
        sp.leg(u, 0.36, 0, 12, frame)
    s = sp.box(u0, u1, 0.3, 0.8, 12, 15, seat)
    for i in range(1, slats):
        sp.line(s.tp(0, i / slats), s.tp(1, i / slats), shade(seat, 0.78))
    if back:
        sp.box(u0, u1, 0.26, 0.34, 15, 30, shade(seat, 0.96))
        sp.box(u0, u1, 0.24, 0.36, 30, 32, shade(seat, 1.1))


def seat_pair(sp, shell, frame=(150, 156, 166)):
    """Two moulded plastic chairs on a shared beam (waiting-room seating)."""
    sp.box(0.08, 0.92, 0.5, 0.58, 6, 9, frame)
    for u in (0.16, 0.84):
        sp.leg(u, 0.54, 0, 6, frame)
    for a, b in ((0.1, 0.48), (0.52, 0.9)):
        seat = sp.box(a, b, 0.32, 0.8, 9, 14, shell)
        seat.top_rect(0.1, 0.9, 0.15, 0.85, shade(shell, 1.18))
        sp.box(a, b, 0.26, 0.36, 14, 30, shade(shell, 0.95))


def room_block(sp, wall, roof, z=SOLID_MAX, door=None, sign=None, window=None):
    """One cell of a back room: walls up to z, a flat roof, optional door,
    window and sign on the room-facing (left) side."""
    b = sp.box(0.02, 0.98, 0.02, 0.98, 0, z, wall, top_color=roof)
    b.top_rect(0.1, 0.9, 0.1, 0.9, shade(roof, 0.94))
    b.hline('left', 0.92, shade(wall, 0.75))
    b.hline('right', 0.92, shade(wall, 0.6))
    if door:
        b.rect('left', 0.32, 0.68, 0.0, 0.8, door, shade(door, 0.6))
        b.rect('left', 0.6, 0.64, 0.38, 0.46, (214, 180, 96))
    if window:
        b.rect('left' if not door else 'right', 0.2, 0.8, 0.42, 0.78, window, shade(wall, 0.6))
    if sign:
        color, icon = sign
        b.rect('right', 0.3, 0.7, 0.42, 0.78, color, shade(color, 0.6))
        if icon == 'cross':
            b.rect('right', 0.46, 0.54, 0.47, 0.73, (250, 250, 250))
            b.rect('right', 0.36, 0.64, 0.56, 0.64, (250, 250, 250))
    return b


def register(sp, u=0.6, v=0.45, z=DESK_Z + 4, color=(60, 62, 70)):
    sp.box(u - 0.1, u + 0.1, v - 0.1, v + 0.1, z, z + 4, color)
    scr = sp.box(u - 0.08, u + 0.08, v - 0.08, v - 0.02, z + 4, z + 11, shade(color, 1.1))
    scr.rect('left', 0.1, 0.9, 0.15, 0.85, (120, 200, 170))


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

LIB_WOOD = (128, 82, 52)
LIB_WOOD_LIGHT = (172, 120, 78)
LIB_GREEN = (52, 104, 78)


def library():
    w = 'library'
    rng = rng_for(w)

    # Bookshelf: a low two-shelf walnut stack (below an agent's shoulders),
    # books on both visible faces.
    sp = Sprite(40 + SOLID_MAX + 4)
    case = sp.box(0.06, 0.94, 0.06, 0.94, 0, SOLID_MAX, LIB_WOOD)
    for face in ('left', 'right'):
        case.rect(face, 0.06, 0.94, 0.06, 0.94, shade(LIB_WOOD, 0.48))      # back
        for t0, t1 in ((0.1, 0.46), (0.56, 0.92)):
            book_row(case, face, 0.08, 0.92, t0, t1, rng)
        for t in (0.07, 0.51, 0.95):                                          # shelves
            case.rect(face, 0.05, 0.95, t - 0.035, t + 0.025, LIB_WOOD_LIGHT)
        case.vline(face, 0.05, shade(LIB_WOOD, 0.8), 0.0, 1.0)
        case.vline(face, 0.95, shade(LIB_WOOD, 0.62), 0.0, 1.0)
    case.top_rect(0.08, 0.92, 0.08, 0.92, shade(LIB_WOOD, 1.2))
    sp.save(out(w, 'bookshelf.png'))

    # Reading desk: green banker's lamp, an open book, a book stack.
    sp = Sprite(40 + THIN_MAX + 2)
    top = desk(sp, LIB_WOOD_LIGHT, z=DESK_Z)
    top.top_rect(0.12, 0.62, 0.38, 0.92, (238, 232, 214), shade((238, 232, 214), 0.7), dz=0.5)  # book
    sp.line(top.tp(0.37, 0.38), top.tp(0.37, 0.92), (170, 160, 140))
    for i, c in enumerate(((176, 62, 52), (58, 92, 150), (196, 150, 58))):          # stack
        sp.box(0.66, 0.86, 0.5, 0.78, DESK_Z + i * 3, DESK_Z + 3 + i * 3, c)
    sp.leg(0.5, 0.2, DESK_Z, THIN_MAX - 6, (176, 146, 70), w=0.02)                  # lamp stem
    sp.box(0.38, 0.62, 0.12, 0.3, THIN_MAX - 6, THIN_MAX, LIB_GREEN)                # lamp shade
    sp.box(0.44, 0.56, 0.14, 0.26, DESK_Z, DESK_Z + 2, (176, 146, 70))              # lamp base
    sp.save(out(w, 'reading_desk.png'))

    # Computer station: a low monitor facing the room, keyboard, tower.
    sp = Sprite(40 + THIN_MAX + 2)
    sp.box(0.7, 0.86, 0.3, 0.7, 0, DESK_Z - 6, (70, 72, 82))                        # tower
    desk(sp, LIB_WOOD_LIGHT, z=DESK_Z)
    sp.box(0.44, 0.56, 0.4, 0.52, DESK_Z, DESK_Z + 3, (60, 62, 70))                 # stand
    mon = sp.box(0.2, 0.8, 0.36, 0.46, DESK_Z + 3, THIN_MAX, (54, 56, 64))
    mon.rect('left', 0.06, 0.94, 0.1, 0.88, (92, 150, 196))                          # screen
    mon.rect('left', 0.12, 0.6, 0.6, 0.78, (150, 196, 228))
    mon.hline('left', 0.38, (120, 172, 210), 0.12, 0.85)
    sp.slab(0.25, 0.75, 0.6, 0.76, DESK_Z + 0.5, (210, 210, 214))                    # keyboard
    sp.save(out(w, 'computer_station.png'))

    # Book return: a lacquered drop bin with a brass slot and a book icon.
    sp = Sprite(40 + SOLID_MAX + 2)
    MAROON = (150, 48, 52)
    bin_ = sp.box(0.18, 0.82, 0.22, 0.78, 0, SOLID_MAX - 4, MAROON)
    sp.box(0.14, 0.86, 0.18, 0.82, SOLID_MAX - 4, SOLID_MAX, shade(MAROON, 1.1))      # lid
    bin_.rect('left', 0.18, 0.82, 0.68, 0.82, (40, 22, 22))                          # slot
    bin_.hline('left', 0.66, (214, 176, 90), 0.16, 0.84)
    bin_.rect('left', 0.38, 0.62, 0.2, 0.52, (236, 228, 208), (190, 180, 160))       # icon
    bin_.vline('left', 0.5, (150, 140, 120), 0.2, 0.52)
    bin_.panel('right', 0.15, 0.85, 0.12, 0.6, MAROON)
    sp.save(out(w, 'return_drop.png'))

    # Circulation desk: closed wooden desk, books, a brass bell, a lamp.
    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, LIB_WOOD, z=DESK_Z + 4, u0=0.06, u1=0.94, v0=0.2, v1=0.8, panel=True)
    zt = DESK_Z + 4
    for i, c in enumerate(((58, 92, 150), (64, 128, 84))):
        sp.box(0.14, 0.4, 0.3, 0.58, zt + i * 3, zt + 3 + i * 3, c)
    sp.box(0.58, 0.7, 0.55, 0.67, zt, zt + 4, (214, 176, 90))                        # bell
    sp.px(0.64, 0.61, zt + 6, (250, 226, 150))
    sp.leg(0.8, 0.3, zt, THIN_MAX - 5, (176, 146, 70), w=0.02)
    sp.box(0.7, 0.9, 0.22, 0.38, THIN_MAX - 5, THIN_MAX, LIB_GREEN)
    sp.save(out(w, 'counter.png'))

    # Reading chair: a deep-green leather armchair (low back).
    sp = Sprite(40 + SOLID_MAX + 2)
    armchair(sp, (64, 116, 88))
    sp.save(out(w, 'reading_chair.png'))

    floor_tile(os.path.join(ISO, 'floor', f'{w}.png'), (186, 140, 96), (150, 108, 72), 'planks')
    wall_block(os.path.join(ISO, 'cells/shared', f'wall_{w}.png'),
               plaster=(214, 222, 204), wainscot=(118, 78, 50), cap=(88, 60, 40))


# ---------------------------------------------------------------------------
# Hospital: mint walls, pale vinyl, white and teal furniture
# ---------------------------------------------------------------------------

def hospital():
    w = 'hospital'
    WHITE, TEAL, MINT = (236, 240, 240), (64, 150, 150), (186, 222, 210)

    # Reception counter: white desk with a teal front band, monitor, clipboard.
    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, WHITE, z=DESK_Z + 4, u0=0.06, u1=0.94, v0=0.2, v1=0.8, panel=True)
    zt = DESK_Z + 4
    sp.box(0.06, 0.94, 0.79, 0.81, 6, 12, TEAL, top=False)
    register(sp, 0.62, 0.42, zt, (70, 76, 86))
    sp.slab(0.18, 0.4, 0.5, 0.72, zt + 0.5, (240, 236, 220))                       # clipboard
    sp.slab(0.2, 0.38, 0.52, 0.56, zt + 0.8, (90, 140, 160), outline=False)
    sp.save(out(w, 'counter.png'))

    # Nurses' station: L-shaped white desk, two monitors, files, a phone.
    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, WHITE, z=DESK_Z + 4, u0=0.06, u1=0.94, v0=0.18, v1=0.82, panel=True)
    zt = DESK_Z + 4
    for u in (0.3, 0.68):
        m = sp.box(u - 0.14, u + 0.14, 0.26, 0.32, zt + 2, THIN_MAX, (60, 64, 72))
        m.rect('left', 0.1, 0.9, 0.12, 0.85, (96, 190, 170))
        m.hline('left', 0.5, (60, 220, 140), 0.15, 0.85)                           # vitals line
    for i, c in enumerate(((236, 196, 72), (64, 132, 196), (214, 86, 64))):
        sp.box(0.14, 0.34, 0.55, 0.75, zt + i * 2, zt + 2 + i * 2, c)
    sp.box(0.6, 0.76, 0.55, 0.72, zt, zt + 3, (70, 70, 76))                          # phone
    sp.save(out(w, 'nurses_station.png'))

    # Hospital bed: white frame, pillow, pale-blue blanket, low side rail.
    sp = Sprite(40 + SOLID_MAX + 2)
    for (u, v) in ((0.14, 0.78), (0.86, 0.78), (0.86, 0.24)):
        sp.leg(u, v, 0, 8, (150, 156, 166))
    sp.box(0.08, 0.92, 0.2, 0.82, 8, 12, (196, 202, 210))                            # frame
    mat = sp.box(0.1, 0.9, 0.22, 0.8, 12, 17, WHITE)
    sp.box(0.62, 0.9, 0.26, 0.76, 17, 21, (250, 250, 250))                           # pillow
    blanket = sp.box(0.1, 0.6, 0.22, 0.8, 17, 19, (150, 196, 226))
    blanket.hline('left', 0.5, (190, 220, 240))
    sp.box(0.88, 0.94, 0.18, 0.84, 8, 30, (196, 202, 210))                            # headboard
    sp.line((0.2, 0.84, 22), (0.6, 0.84, 22), (170, 176, 186))                        # rail
    sp.leg(0.2, 0.84, 17, 22, (170, 176, 186), w=0.012)
    sp.leg(0.6, 0.84, 17, 22, (170, 176, 186), w=0.012)
    sp.save(out(w, 'hospital_bed.png'))

    # Waiting bench (the hospital's seat): teal moulded chairs on a beam.
    sp = Sprite(40 + SOLID_MAX + 2)
    seat_pair(sp, TEAL)
    sp.save(out(w, 'waiting_bench.png'))

    # Exam room: pale partition with a door and a red-cross sign.
    sp = Sprite(40 + SOLID_MAX + 2)
    room_block(sp, MINT, (226, 236, 232), door=(236, 240, 240), sign=((214, 64, 60), 'cross'))
    sp.save(out(w, 'exam_room.png'))

    # Triage: a curtain screen beside a wheeled cart with a vitals monitor.
    sp = Sprite(40 + THIN_MAX + 2)
    for u in (0.08, 0.92):
        sp.leg(u, 0.18, 0, SOLID_MAX + 2, (170, 176, 186), w=0.015)
    cur = sp.box(0.08, 0.92, 0.16, 0.2, 4, SOLID_MAX, (196, 228, 214))
    for s in (0.2, 0.4, 0.6, 0.8):
        cur.vline('left', s, (164, 204, 188), 0.0, 1.0)
    for (u, v) in ((0.36, 0.72), (0.74, 0.72), (0.74, 0.42)):
        sp.leg(u, v, 0, 16, (150, 156, 166), w=0.02)
    cart = sp.box(0.3, 0.8, 0.38, 0.78, 16, 20, WHITE)
    sp.box(0.36, 0.56, 0.48, 0.7, 20, 24, (214, 64, 60))                              # kit
    m = sp.box(0.6, 0.78, 0.44, 0.5, 20, 34, (60, 64, 72))
    m.rect('left', 0.1, 0.9, 0.15, 0.85, (40, 60, 56))
    m.hline('left', 0.5, (90, 230, 140), 0.12, 0.88)
    sp.save(out(w, 'triage_area.png'))

    floor_tile(os.path.join(ISO, 'floor', f'{w}.png'), (222, 226, 224), (196, 202, 200), 'tiles')
    wall_block(os.path.join(ISO, 'cells/shared', f'wall_{w}.png'),
               plaster=(206, 230, 222), wainscot=(150, 172, 180), cap=(110, 126, 134), plank=10)


# ---------------------------------------------------------------------------
# Barbershop: checkerboard floor, red leather, chrome, dark teal wainscot
# ---------------------------------------------------------------------------

def barbershop():
    w = 'barbershop'
    rng = rng_for(w)
    RED, CHROME, WOOD = (176, 48, 48), (196, 202, 210), (104, 66, 44)

    # Reception counter: dark wood, register, pomade jars.
    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, WOOD, z=DESK_Z + 4, u0=0.06, u1=0.94, v0=0.2, v1=0.8, panel=True)
    zt = DESK_Z + 4
    register(sp, 0.66, 0.42, zt)
    for i, c in enumerate(((60, 120, 180), (214, 160, 60), (176, 48, 48))):
        sp.box(0.14 + i * 0.12, 0.22 + i * 0.12, 0.58, 0.68, zt, zt + 5, c)
    sp.save(out(w, 'counter.png'))

    # Barber chair (the seat): chrome pedestal, red leather, headrest.
    sp = Sprite(40 + SOLID_MAX + 2)
    sp.box(0.26, 0.74, 0.3, 0.76, 0, 2, CHROME)                                         # base
    sp.box(0.44, 0.56, 0.46, 0.6, 2, 11, shade(CHROME, 0.85))                           # pedestal
    sp.box(0.2, 0.8, 0.62, 0.8, 4, 7, CHROME)                                           # footrest
    seat = sp.box(0.22, 0.78, 0.32, 0.74, 11, 16, RED)
    seat.top_rect(0.1, 0.9, 0.12, 0.88, shade(RED, 1.2))
    sp.box(0.22, 0.78, 0.24, 0.36, 16, 30, shade(RED, 0.95))                            # back
    sp.box(0.36, 0.64, 0.24, 0.32, 30, SOLID_MAX, shade(RED, 0.9))                      # headrest
    for u0_, u1_ in ((0.16, 0.24), (0.76, 0.84)):
        sp.box(u0_, u1_, 0.34, 0.74, 16, 22, CHROME)                                    # arms
    sp.save(out(w, 'barber_chair.png'))

    # Waiting couch, two halves (arm on the left / right end).
    for name, arm_lo in (('waiting_couch_left.png', True), ('waiting_couch_right.png', False)):
        sp = Sprite(40 + SOLID_MAX + 2)
        LEATHER = (150, 84, 60)
        for u in (0.12, 0.88):
            sp.leg(u, 0.78, 0, 5, (60, 40, 30))
        sp.box(0.04, 0.96, 0.2, 0.84, 5, 13, shade(LEATHER, 0.88))
        c = sp.box(0.04, 0.96, 0.32, 0.84, 13, 17, LEATHER)
        sp.line(c.tp(0.5, 0), c.tp(0.5, 1), shade(LEATHER, 0.7))
        sp.box(0.04, 0.96, 0.18, 0.32, 13, 30, shade(LEATHER, 0.94))
        a0, a1 = (0.02, 0.14) if arm_lo else (0.86, 0.98)
        sp.box(a0, a1, 0.3, 0.84, 13, 23, shade(LEATHER, 0.92))
        sp.save(out(w, name))

    # Mirror station: vanity with tools and bottles, a low mirror behind.
    sp = Sprite(40 + THIN_MAX + 2)
    mir = sp.box(0.1, 0.9, 0.08, 0.14, DESK_Z, THIN_MAX, (150, 112, 70))
    mir.rect('left', 0.06, 0.94, 0.08, 0.92, (190, 220, 232))
    mir.rect('left', 0.2, 0.32, 0.25, 0.85, (230, 244, 250))
    cabinet(sp, (232, 226, 214), z=DESK_Z, u0=0.1, u1=0.9, v0=0.16, v1=0.8, style='drawers')
    sp.box(0.08, 0.92, 0.14, 0.82, DESK_Z, DESK_Z + 2, (70, 60, 56))                     # top
    for i, c in enumerate(((60, 120, 180), (214, 60, 60), (236, 196, 72))):
        sp.box(0.18 + i * 0.12, 0.25 + i * 0.12, 0.3, 0.38, DESK_Z + 2, DESK_Z + 9, c)
    sp.slab(0.6, 0.82, 0.5, 0.6, DESK_Z + 2.5, (200, 204, 212))                          # scissors
    sp.save(out(w, 'mirror_station.png'))

    # Supply cabinet: teal drawers with folded towels on top.
    sp = Sprite(40 + SOLID_MAX + 2)
    cabinet(sp, (60, 118, 120), z=SOLID_MAX - 8, style='drawers')
    for i in range(3):
        sp.box(0.2, 0.62, 0.3, 0.7, SOLID_MAX - 8 + i * 2.6, SOLID_MAX - 5.4 + i * 2.6, (244, 244, 238))
    sp.save(out(w, 'supply_cabinet.png'))

    # Magazine rack: slanted wooden rack of colorful magazines.
    sp = Sprite(40 + SOLID_MAX + 2)
    shelf_unit(sp, (150, 104, 66), rng, 'magazines', z=SOLID_MAX - 6, u0=0.14, u1=0.86, v0=0.3,
               v1=0.7)
    sp.save(out(w, 'magazine_rack.png'))

    floor_tile(os.path.join(ISO, 'floor', f'{w}.png'), (216, 206, 190), (194, 184, 168), 'tiles')
    wall_block(os.path.join(ISO, 'cells/shared', f'wall_{w}.png'),
               plaster=(238, 226, 206), wainscot=(46, 98, 100), cap=(40, 56, 58),
               rail=(176, 48, 48), plank=8)


# ---------------------------------------------------------------------------
# Post office: blue and red, cardboard and brass
# ---------------------------------------------------------------------------

def post_office():
    w = 'post_office'
    rng = rng_for(w)
    BLUE, NAVY, RED, CARD = (52, 92, 168), (40, 56, 96), (196, 52, 52), (190, 146, 98)

    # Service counter: blue counter, stamp sheets, a parcel, a bell.
    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, BLUE, z=DESK_Z + 4, u0=0.06, u1=0.94, v0=0.2, v1=0.8, panel=True)
    zt = DESK_Z + 4
    sp.box(0.06, 0.94, 0.79, 0.81, 4, 8, RED, top=False)
    p = sp.box(0.16, 0.42, 0.3, 0.56, zt, zt + 8, CARD)
    p.vline('left', 0.5, (226, 204, 160))
    register(sp, 0.66, 0.42, zt)
    sp.slab(0.5, 0.7, 0.6, 0.74, zt + 0.5, (236, 230, 214))
    sp.save(out(w, 'counter.png'))

    # Mailbox: the blue street box, domed top and a pull-down slot.
    sp = Sprite(40 + SOLID_MAX + 2)
    for (u, v) in ((0.3, 0.72), (0.7, 0.72), (0.7, 0.32)):
        sp.leg(u, v, 0, 5, NAVY)
    box = sp.box(0.24, 0.76, 0.26, 0.74, 5, 27, BLUE)
    sp.box(0.26, 0.74, 0.3, 0.7, 27, 31, shade(BLUE, 1.05))
    sp.box(0.3, 0.7, 0.36, 0.64, 31, SOLID_MAX, shade(BLUE, 1.1))
    box.rect('left', 0.2, 0.8, 0.7, 0.82, NAVY)                                         # slot
    box.rect('left', 0.3, 0.7, 0.3, 0.52, (240, 240, 236), (180, 180, 176))             # label
    box.rect('left', 0.38, 0.62, 0.38, 0.44, RED)
    sp.save(out(w, 'mailbox.png'))

    # Package shelf: steel shelving full of parcels.
    sp = Sprite(40 + SOLID_MAX + 2)
    shelf_unit(sp, (150, 156, 166), rng, 'parcels', back=(110, 114, 122))
    sp.save(out(w, 'package_shelf.png'))

    # PO boxes: a block of small brass-fronted doors.
    sp = Sprite(40 + SOLID_MAX + 2)
    b = sp.box(0.04, 0.96, 0.04, 0.96, 0, SOLID_MAX, (176, 140, 84))
    for face in ('left', 'right'):
        for i in range(4):
            for j in range(3):
                s0, t0 = 0.06 + i * 0.225, 0.06 + j * 0.31
                b.panel(face, s0, s0 + 0.2, t0, t0 + 0.27, (196, 160, 98))
                b.rect(face, s0 + 0.14, s0 + 0.17, t0 + 0.11, t0 + 0.16, (90, 70, 40))
    sp.save(out(w, 'po_box.png'))

    # Sorting room: back room with pigeonhole sorting frames on its roof line.
    sp = Sprite(40 + SOLID_MAX + 2)
    room_block(sp, (190, 170, 140), (214, 198, 172), door=(150, 112, 70),
               window=(170, 200, 220))
    sp.save(out(w, 'sorting_room.png'))

    # Scale station: small table with a digital scale, a parcel and tape.
    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, (176, 132, 90), z=DESK_Z)
    sc = sp.box(0.22, 0.56, 0.36, 0.66, DESK_Z, DESK_Z + 3, (200, 204, 212))
    sc.top_rect(0.15, 0.85, 0.15, 0.85, (226, 230, 236))
    d = sp.box(0.26, 0.5, 0.66, 0.72, DESK_Z, DESK_Z + 6, (60, 64, 72))
    d.rect('left', 0.1, 0.9, 0.2, 0.8, (120, 220, 150))
    sp.box(0.3, 0.5, 0.42, 0.6, DESK_Z + 3, DESK_Z + 9, CARD)
    sp.box(0.66, 0.8, 0.5, 0.64, DESK_Z, DESK_Z + 4, (180, 140, 90))                    # tape
    sp.save(out(w, 'scale_station.png'))

    # Waiting bench (the seat): wooden slat bench.
    sp = Sprite(40 + SOLID_MAX + 2)
    bench(sp, (176, 128, 84), frame=(70, 74, 82))
    sp.save(out(w, 'waiting_bench.png'))

    floor_tile(os.path.join(ISO, 'floor', f'{w}.png'), (214, 204, 186), (186, 176, 158), 'tiles')
    wall_block(os.path.join(ISO, 'cells/shared', f'wall_{w}.png'),
               plaster=(206, 220, 236), wainscot=(52, 72, 120), cap=(36, 46, 76),
               rail=(196, 52, 52), plank=8)


# ---------------------------------------------------------------------------
# Clothing store: blush walls, pale oak, garments in soft colors
# ---------------------------------------------------------------------------

def clothing_store():
    w = 'clothing_store'
    rng = rng_for(w)
    OAK, WHITE, BLUSH = (206, 170, 126), (240, 236, 230), (226, 180, 176)
    GARMENTS = [(214, 96, 96), (96, 140, 200), (240, 200, 90), (110, 170, 130), (236, 236, 228),
                (170, 120, 190), (60, 64, 80)]

    # Counter: white counter, register, a shopping bag with handles.
    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, WHITE, z=DESK_Z + 4, u0=0.06, u1=0.94, v0=0.2, v1=0.8, panel=True)
    zt = DESK_Z + 4
    sp.box(0.06, 0.94, 0.79, 0.81, 2, 6, OAK, top=False)
    register(sp, 0.66, 0.42, zt)
    bag = sp.box(0.16, 0.38, 0.4, 0.56, zt, zt + 9, BLUSH)
    sp.line(bag.tp(0.3, 0.5), (0.24, 0.48, zt + 12), (120, 80, 80))
    sp.line((0.24, 0.48, zt + 12), bag.tp(0.7, 0.5), (120, 80, 80))
    sp.save(out(w, 'counter.png'))

    # Clothing rack: chrome rail with hanging garments.
    sp = Sprite(40 + THIN_MAX + 2)
    for u in (0.1, 0.9):
        sp.leg(u, 0.5, 0, 36, (196, 202, 210), w=0.02)
        sp.box(u - 0.06, u + 0.06, 0.42, 0.58, 0, 2, (170, 176, 186))
    sp.line((0.1, 0.5, 36), (0.9, 0.5, 36), (220, 224, 230), width=2)
    u = 0.15
    while u < 0.86:
        c = rng.choice(GARMENTS)
        g = sp.box(u, u + 0.075, 0.38, 0.62, 12 + rng.randint(0, 6), 34, c)
        u += 0.085
    sp.save(out(w, 'clothing_rack.png'))

    # Fitting room: a booth with a velvet curtain.
    sp = Sprite(40 + SOLID_MAX + 2)
    b = sp.box(0.04, 0.96, 0.04, 0.96, 0, SOLID_MAX, WHITE, top_color=(226, 222, 216))
    b.rect('left', 0.12, 0.88, 0.0, 0.88, (150, 60, 86))
    for s in (0.25, 0.4, 0.55, 0.7):
        b.vline('left', s, (120, 44, 66), 0.0, 0.88)
    b.hline('left', 0.9, (214, 180, 96), 0.08, 0.92)
    b.rect('right', 0.3, 0.7, 0.5, 0.75, (214, 180, 96))                                # sign
    sp.save(out(w, 'fitting_room.png'))

    # Mannequin: a dress form in an outfit, on a stand (shorter than an agent).
    sp = Sprite(40 + THIN_MAX + 2)
    sp.box(0.36, 0.64, 0.36, 0.64, 0, 2, (90, 90, 96))
    sp.leg(0.5, 0.5, 2, 12, (90, 90, 96), w=0.015)
    sp.box(0.38, 0.62, 0.42, 0.58, 12, 26, (196, 80, 96))                                # dress
    sp.box(0.4, 0.6, 0.44, 0.56, 26, 33, (196, 80, 96))
    sp.box(0.46, 0.54, 0.47, 0.53, 33, 36, (236, 220, 200))                              # neck
    sp.box(0.43, 0.57, 0.45, 0.55, 36, 40, (236, 220, 200))                              # head
    sp.save(out(w, 'mannequin.png'))

    # Shoe display: a low two-tier oak table of shoes.
    sp = Sprite(40 + SOLID_MAX + 2)
    shelf_unit(sp, OAK, rng, 'shoes', z=20, u0=0.08, u1=0.92, v0=0.2, v1=0.8,
               back=shade(OAK, 0.7))
    sp.save(out(w, 'shoe_display.png'))

    # Stockroom: back room door with a "staff" plate.
    sp = Sprite(40 + SOLID_MAX + 2)
    room_block(sp, BLUSH, (236, 214, 208), door=(236, 236, 232), sign=((60, 64, 80), None))
    sp.save(out(w, 'clothing_stockroom.png'))

    # Stool (the seat): round upholstered pouf.
    sp = Sprite(40 + SOLID_MAX + 2)
    for (u, v) in ((0.3, 0.7), (0.7, 0.7), (0.7, 0.3)):
        sp.leg(u, v, 0, 8, (176, 140, 96))
    s = sp.box(0.22, 0.78, 0.22, 0.78, 8, 16, (110, 150, 160))
    s.top_rect(0.1, 0.9, 0.1, 0.9, shade((110, 150, 160), 1.2))
    sp.save(out(w, 'stool.png'))

    floor_tile(os.path.join(ISO, 'floor', f'{w}.png'), (226, 206, 176), (200, 178, 146), 'planks')
    wall_block(os.path.join(ISO, 'cells/shared', f'wall_{w}.png'),
               plaster=(240, 214, 210), wainscot=(244, 240, 234), cap=(186, 150, 140), plank=10)


# ---------------------------------------------------------------------------
# Pharmacy: white and green, colorful packaging
# ---------------------------------------------------------------------------

def pharmacy():
    w = 'pharmacy'
    rng = rng_for(w)
    WHITE, GREEN = (240, 242, 238), (52, 150, 96)

    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, WHITE, z=DESK_Z + 4, u0=0.06, u1=0.94, v0=0.2, v1=0.8, panel=True)
    zt = DESK_Z + 4
    sp.box(0.06, 0.94, 0.79, 0.81, 8, 13, GREEN, top=False)
    register(sp, 0.66, 0.42, zt)
    sp.box(0.18, 0.36, 0.42, 0.58, zt, zt + 7, (236, 236, 228))                          # paper bag
    sp.save(out(w, 'counter.png'))

    # Medicine shelf: white shelving with boxes and bottles.
    sp = Sprite(40 + SOLID_MAX + 2)
    unit = sp.box(0.06, 0.94, 0.06, 0.94, 0, SOLID_MAX, WHITE, top_color=(214, 222, 216))
    unit.top_rect(0.12, 0.88, 0.12, 0.88, (200, 210, 204))
    for face in ('left', 'right'):
        unit.rect(face, 0.06, 0.94, 0.06, 0.94, (214, 220, 216))
        stock_row(unit, face, 0.08, 0.92, 0.1, 0.46, rng, 'boxes')
        stock_row(unit, face, 0.08, 0.92, 0.56, 0.9, rng, 'bottles')
        for t in (0.1, 0.53, 0.96):
            unit.rect(face, 0.05, 0.95, t - 0.045, t, (226, 230, 226))
        unit.hline(face, 0.01, GREEN)
    sp.save(out(w, 'medicine_shelf.png'))

    # Display case: glass case of skincare, like the cafe pastry case.
    sp = Sprite(40 + SOLID_MAX + 2)
    base = sp.box(0.08, 0.92, 0.2, 0.8, 0, 12, GREEN)
    base.panel('left', 0.08, 0.92, 0.2, 0.8, GREEN)
    glass = sp.box(0.08, 0.92, 0.2, 0.8, 12, 30, (176, 214, 226), top_color=(200, 230, 238))
    for i, c in enumerate(((236, 150, 170), (150, 200, 236), (240, 220, 140), (180, 230, 190))):
        sp.box(0.16 + i * 0.18, 0.28 + i * 0.18, 0.4, 0.56, 12, 18 + (i % 2) * 3, c)
    glass.vline('left', 0.02, (240, 250, 255))
    glass.vline('left', 0.35, (214, 238, 246), 0.3, 0.9)
    sp.save(out(w, 'display_case.png'))

    # BP station: a kiosk seat with a cuff and a screen post.
    sp = Sprite(40 + THIN_MAX + 2)
    sp.box(0.18, 0.62, 0.3, 0.78, 0, 12, (60, 130, 140))
    sp.box(0.18, 0.62, 0.26, 0.36, 12, 26, (60, 130, 140))
    sp.box(0.62, 0.8, 0.42, 0.62, 10, 16, (70, 74, 86))                                  # cuff
    sp.leg(0.76, 0.3, 0, 32, (150, 156, 166), w=0.02)
    scr = sp.box(0.62, 0.9, 0.26, 0.32, 30, THIN_MAX, (236, 240, 240))
    scr.rect('left', 0.1, 0.9, 0.15, 0.85, (110, 200, 170))
    sp.save(out(w, 'bp_station.png'))

    # Rx pickup: a counter window with a green Rx plate and order bags.
    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, WHITE, z=DESK_Z + 4, u0=0.06, u1=0.94, v0=0.2, v1=0.8, panel=True)
    zt = DESK_Z + 4
    plate = sp.box(0.2, 0.8, 0.22, 0.28, zt, THIN_MAX, GREEN)
    plate.rect('left', 0.25, 0.45, 0.3, 0.75, (250, 250, 250))
    plate.rect('left', 0.55, 0.75, 0.3, 0.75, (250, 250, 250))
    for u in (0.3, 0.55):
        sp.box(u - 0.08, u + 0.08, 0.5, 0.66, zt, zt + 7, (236, 236, 228))
    sp.save(out(w, 'rx_pickup.png'))

    # Stockroom: white back room, green door, a cross sign.
    sp = Sprite(40 + SOLID_MAX + 2)
    room_block(sp, (226, 232, 228), (236, 240, 238), door=GREEN, sign=(GREEN, 'cross'))
    sp.save(out(w, 'stockroom.png'))

    # Bench (the seat): green moulded chairs.
    sp = Sprite(40 + SOLID_MAX + 2)
    seat_pair(sp, GREEN)
    sp.save(out(w, 'bench.png'))

    floor_tile(os.path.join(ISO, 'floor', f'{w}.png'), (236, 238, 234), (210, 214, 210), 'tiles')
    wall_block(os.path.join(ISO, 'cells/shared', f'wall_{w}.png'),
               plaster=(236, 242, 236), wainscot=(52, 150, 96), cap=(40, 100, 70), plank=10)


# ---------------------------------------------------------------------------
# Grocery store: bright gondolas, fresh colors
# ---------------------------------------------------------------------------

def grocery_store():
    w = 'grocery_store'
    rng = rng_for(w)
    GREEN, WHITE, STEEL = (72, 140, 84), (238, 238, 232), (170, 176, 186)

    # Checkout: conveyor belt, register, a divider bar.
    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, WHITE, z=DESK_Z + 4, u0=0.06, u1=0.94, v0=0.2, v1=0.8, panel=True)
    zt = DESK_Z + 4
    sp.box(0.06, 0.94, 0.79, 0.81, 4, 10, GREEN, top=False)
    belt = sp.slab(0.1, 0.62, 0.5, 0.76, zt + 0.5, (52, 54, 60))
    for a in (0.2, 0.32, 0.44):
        sp.box(a, a + 0.08, 0.56, 0.68, zt, zt + 4, rng.choice(STOCK['produce'][0]))
    register(sp, 0.74, 0.42, zt)
    sp.save(out(w, 'counter.png'))

    # Product shelf: a gondola of cans, boxes and produce.
    sp = Sprite(40 + SOLID_MAX + 2)
    unit = sp.box(0.06, 0.94, 0.06, 0.94, 0, SOLID_MAX, WHITE, top_color=(212, 216, 210))
    unit.top_rect(0.12, 0.88, 0.12, 0.88, (196, 202, 196))
    for face in ('left', 'right'):
        unit.rect(face, 0.06, 0.94, 0.06, 0.94, (200, 204, 200))
        stock_row(unit, face, 0.08, 0.92, 0.1, 0.42, rng, rng.choice(('cans', 'produce')))
        stock_row(unit, face, 0.08, 0.92, 0.52, 0.9, rng, 'boxes')
        for t in (0.1, 0.5, 0.96):
            unit.rect(face, 0.05, 0.95, t - 0.05, t, (226, 228, 224))
        unit.rect(face, 0.0, 1.0, 0.0, 0.05, GREEN)                                     # kick plate
    sp.save(out(w, 'product_shelf.png'))

    # Cart corral: a steel rail with nested carts.
    sp = Sprite(40 + SOLID_MAX + 2)
    for u in (0.06, 0.94):                                                            # rail posts
        sp.leg(u, 0.22, 0, 26, STEEL, w=0.02)
    sp.line((0.06, 0.22, 26), (0.94, 0.22, 26), shade(STEEL, 1.1), width=2)
    for k in range(3):                                                              # nested carts
        o = k * 0.16
        for (u, v) in ((0.18 + o, 0.74), (0.18 + o, 0.36), (0.48 + o, 0.74)):
            sp.leg(u, v, 0, 5, (60, 60, 64), w=0.02)
        basket = sp.box(0.12 + o, 0.5 + o, 0.3, 0.78, 6, 20, STEEL)
        basket.top_rect(0.08, 0.92, 0.1, 0.9, shade(STEEL, 0.6))                    # open top
        for face in ('left', 'right'):
            for t in (0.33, 0.66):
                basket.hline(face, t, shade(STEEL, 0.72))
            for s in (0.25, 0.5, 0.75):
                basket.vline(face, s, shade(STEEL, 0.72))
        sp.box(0.5 + o, 0.55 + o, 0.3, 0.78, 18, 22, (214, 60, 60))                   # handle
    sp.save(out(w, 'cart_corral.png'))

    # Bagging area: a counter with paper bags.
    sp = Sprite(40 + SOLID_MAX + 2)
    desk(sp, (176, 132, 90), z=DESK_Z, panel=True)
    for u in (0.2, 0.46, 0.72):
        bag = sp.box(u, u + 0.18, 0.36, 0.62, DESK_Z, DESK_Z + 10, (206, 170, 120))
        bag.hline('left', 0.85, (180, 144, 98))
    sp.save(out(w, 'bagging_area.png'))

    # Storage room: back room with crates stacked by the door.
    sp = Sprite(40 + SOLID_MAX + 2)
    room_block(sp, (214, 206, 186), (226, 220, 204), door=(150, 156, 166))
    sp.save(out(w, 'storage_room.png'))

    # Waiting bench (the seat).
    sp = Sprite(40 + SOLID_MAX + 2)
    bench(sp, (196, 150, 98), frame=(60, 120, 76))
    sp.save(out(w, 'waiting_bench.png'))

    floor_tile(os.path.join(ISO, 'floor', f'{w}.png'), (232, 230, 220), (206, 204, 194), 'checker',
               accent=(214, 222, 212))
    wall_block(os.path.join(ISO, 'cells/shared', f'wall_{w}.png'),
               plaster=(244, 236, 206), wainscot=(72, 140, 84), cap=(52, 92, 60), plank=10)


# ---------------------------------------------------------------------------
# Bank: marble, navy and brass
# ---------------------------------------------------------------------------

def bank():
    w = 'bank'
    NAVY, BRASS, MARBLE = (44, 60, 104), (204, 170, 90), (232, 228, 220)

    # Teller counter: navy body, marble top, low brass rail.
    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, NAVY, z=DESK_Z + 4, u0=0.06, u1=0.94, v0=0.2, v1=0.8, panel=True)
    zt = DESK_Z + 4
    sp.box(0.04, 0.96, 0.18, 0.82, zt, zt + 2, MARBLE)
    sp.line((0.06, 0.3, zt + 12), (0.94, 0.3, zt + 12), BRASS, width=2)
    for u in (0.08, 0.5, 0.92):
        sp.leg(u, 0.3, zt + 2, zt + 12, BRASS, w=0.012)
    register(sp, 0.62, 0.5, zt + 2)
    sp.slab(0.16, 0.36, 0.55, 0.72, zt + 2.5, (236, 236, 228))
    sp.save(out(w, 'counter.png'))

    # Vault: brushed steel with a round brass-rimmed door.
    sp = Sprite(40 + SOLID_MAX + 2)
    STEEL = (124, 132, 146)
    v = sp.box(0.02, 0.98, 0.02, 0.98, 0, SOLID_MAX, STEEL, top_color=(150, 158, 170))
    for face in ('left', 'right'):
        v.rect(face, 0.2, 0.8, 0.12, 0.88, shade(STEEL, 0.9 if face == 'left' else 0.7),
               shade(STEEL, 0.5))
        v.rect(face, 0.28, 0.72, 0.2, 0.8, shade(STEEL, 1.05 if face == 'left' else 0.8))
        v.rect(face, 0.46, 0.54, 0.42, 0.58, BRASS)
        v.hline(face, 0.5, BRASS, 0.32, 0.68)
    sp.save(out(w, 'vault.png'))

    # ATM: a kiosk with screen, keypad and card slot.
    sp = Sprite(40 + SOLID_MAX + 2)
    a = sp.box(0.14, 0.86, 0.2, 0.8, 0, SOLID_MAX, (60, 84, 130))
    a.rect('left', 0.18, 0.82, 0.58, 0.88, (30, 40, 56))
    a.rect('left', 0.24, 0.76, 0.62, 0.84, (110, 210, 190))
    for i in range(3):
        for j in range(3):
            a.rect('left', 0.24 + i * 0.12, 0.32 + i * 0.12, 0.3 + j * 0.07, 0.35 + j * 0.07,
                   (220, 224, 230))
    a.rect('left', 0.66, 0.8, 0.36, 0.42, (20, 24, 30))
    a.panel('right', 0.15, 0.85, 0.1, 0.9, (60, 84, 130))
    sp.save(out(w, 'atm_room.png'))

    # Rope stanchion: brass posts with a red velvet rope.
    sp = Sprite(40 + THIN_MAX + 2)
    for u in (0.12, 0.88):
        sp.box(u - 0.07, u + 0.07, 0.43, 0.57, 0, 2, BRASS)
        sp.leg(u, 0.5, 2, 26, BRASS, w=0.02)
        sp.box(u - 0.03, u + 0.03, 0.47, 0.53, 26, 28, shade(BRASS, 1.15))
    for i in range(9):
        t = i / 8
        z = 24 - 6 * (1 - (2 * t - 1) ** 2)
        sp.box(0.12 + 0.76 * t - 0.03, 0.12 + 0.76 * t + 0.03, 0.48, 0.52, z - 1.5, z + 1.5,
               (176, 40, 52), outline=False)
    sp.save(out(w, 'rope_stanchion.png'))

    # Writing desk: a standing desk with slips and a chained pen.
    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, (124, 84, 56), z=DESK_Z + 6, u0=0.14, u1=0.86, v0=0.24, v1=0.76, panel=True)
    zt = DESK_Z + 6
    sp.box(0.12, 0.88, 0.22, 0.78, zt, zt + 2, MARBLE)
    for a0 in (0.2, 0.5):
        sp.slab(a0, a0 + 0.22, 0.4, 0.62, zt + 2.5, (240, 236, 222))
    sp.box(0.7, 0.8, 0.5, 0.56, zt + 2, zt + 3, (40, 40, 44))
    sp.save(out(w, 'writing_desk.png'))

    # Deposit slips: a slim rack of colored slip pockets.
    sp = Sprite(40 + SOLID_MAX + 2)
    r = sp.box(0.2, 0.8, 0.36, 0.64, 0, SOLID_MAX - 6, (124, 84, 56))
    for i, c in enumerate(((236, 236, 228), (240, 200, 120), (200, 220, 240), (240, 190, 190))):
        r.rect('left', 0.1, 0.9, 0.12 + i * 0.21, 0.28 + i * 0.21, c, shade(c, 0.7))
    sp.save(out(w, 'deposit_slips.png'))

    # Waiting chair (the seat): navy armchair.
    sp = Sprite(40 + SOLID_MAX + 2)
    armchair(sp, (60, 78, 128), wood=(90, 60, 40))
    sp.save(out(w, 'waiting_chair.png'))

    floor_tile(os.path.join(ISO, 'floor', f'{w}.png'), (234, 230, 222), (206, 200, 190), 'marble')
    wall_block(os.path.join(ISO, 'cells/shared', f'wall_{w}.png'),
               plaster=(238, 232, 218), wainscot=(44, 60, 104), cap=(30, 40, 70),
               rail=(204, 170, 90), plank=10)


# ---------------------------------------------------------------------------
# Movie theater: navy carpet, blue velvet, gold
# ---------------------------------------------------------------------------

def movie_theater():
    w = 'movie_theater'
    rng = rng_for(w)
    NAVY, VELVET, GOLD = (46, 56, 98), (60, 100, 164), (214, 172, 80)

    # Ticket counter: navy and gold, a popcorn tub and soda cups.
    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, NAVY, z=DESK_Z + 4, u0=0.06, u1=0.94, v0=0.2, v1=0.8, panel=True)
    zt = DESK_Z + 4
    sp.box(0.06, 0.94, 0.79, 0.81, 4, 7, GOLD, top=False)
    register(sp, 0.66, 0.42, zt)
    tub = sp.box(0.18, 0.34, 0.5, 0.66, zt, zt + 8, (236, 236, 228))
    tub.vline('left', 0.33, VELVET)
    tub.vline('left', 0.66, VELVET)
    sp.box(0.19, 0.33, 0.51, 0.65, zt + 8, zt + 10, (250, 226, 150))
    sp.save(out(w, 'counter.png'))

    # Theater seats: blue (and grey) velvet with armrests and a cup holder.
    for name, color in (('theater_seat.png', VELVET), ('theater_seat_d.png', (130, 132, 140))):
        sp = Sprite(40 + SOLID_MAX + 2)
        sp.box(0.18, 0.82, 0.3, 0.8, 0, 8, (50, 44, 50))
        seat = sp.box(0.2, 0.8, 0.34, 0.8, 8, 14, color)
        seat.top_rect(0.1, 0.9, 0.12, 0.88, shade(color, 1.18))
        back = sp.box(0.2, 0.8, 0.22, 0.36, 8, 32, shade(color, 0.95))
        back.hline('left', 0.85, shade(color, 1.2))
        for a0, a1 in ((0.12, 0.2), (0.8, 0.88)):
            sp.box(a0, a1, 0.3, 0.8, 8, 20, (60, 52, 58))
        sp.box(0.81, 0.87, 0.62, 0.72, 20, 21, (30, 26, 30))                            # cup holder
        sp.save(out(w, name))

    # Screen room: the auditorium wall, a lit doorway and a poster case.
    sp = Sprite(40 + SOLID_MAX + 2)
    b = room_block(sp, (50, 56, 80), (66, 74, 102), door=(28, 30, 44))
    b.rect('left', 0.3, 0.7, 0.82, 0.96, (250, 214, 120))                               # exit glow
    b.rect('right', 0.25, 0.75, 0.15, 0.85, GOLD, shade(GOLD, 0.6))
    b.rect('right', 0.3, 0.7, 0.2, 0.8, rng.choice(((60, 110, 190), (230, 170, 70), (90, 160, 110))))
    sp.save(out(w, 'screen_room.png'))

    # Rope barrier: gold posts with a blue velvet rope.
    sp = Sprite(40 + THIN_MAX + 2)
    for u in (0.12, 0.88):
        sp.box(u - 0.07, u + 0.07, 0.43, 0.57, 0, 2, GOLD)
        sp.leg(u, 0.5, 2, 26, GOLD, w=0.02)
        sp.box(u - 0.03, u + 0.03, 0.47, 0.53, 26, 28, shade(GOLD, 1.15))
    for i in range(9):
        t = i / 8
        z = 24 - 6 * (1 - (2 * t - 1) ** 2)
        sp.box(0.12 + 0.76 * t - 0.03, 0.12 + 0.76 * t + 0.03, 0.48, 0.52, z - 1.5, z + 1.5,
               VELVET, outline=False)
    sp.save(out(w, 'rope_barrier.png'))

    # Concession stand: counter with a popcorn machine (glass box) and cups.
    sp = Sprite(40 + THIN_MAX + 2)
    desk(sp, NAVY, z=DESK_Z + 4, u0=0.06, u1=0.94, v0=0.2, v1=0.8, panel=True)
    zt = DESK_Z + 4
    m = sp.box(0.14, 0.5, 0.3, 0.62, zt, THIN_MAX, (196, 230, 240), top_color=NAVY)
    sp.box(0.16, 0.48, 0.32, 0.6, zt, zt + 6, (250, 222, 140), outline=False)
    m.vline('left', 0.04, (240, 250, 255))
    for i, c in enumerate(((236, 236, 228), (60, 110, 200), (240, 200, 70))):
        sp.box(0.6 + i * 0.1, 0.66 + i * 0.1, 0.5, 0.58, zt, zt + 6, c)
    sp.save(out(w, 'concession_stand.png'))

    # Snack stand: a glass candy counter of colorful boxes.
    sp = Sprite(40 + SOLID_MAX + 2)
    shelf_unit(sp, (50, 56, 80), rng, 'candy', z=SOLID_MAX - 4, u0=0.06, u1=0.94, v0=0.18, v1=0.82,
               back=(32, 36, 52))
    sp.save(out(w, 'snack_stand.png'))

    floor_tile(os.path.join(ISO, 'floor', f'{w}.png'), (58, 66, 104), (44, 50, 82), 'carpet',
               accent=(214, 172, 80))
    wall_block(os.path.join(ISO, 'cells/shared', f'wall_{w}.png'),
               plaster=(78, 86, 120), wainscot=(40, 44, 68), cap=(30, 32, 50),
               rail=(214, 172, 80), plank=10)


WORLDS = {
    'library': library, 'hospital': hospital, 'barbershop': barbershop,
    'post_office': post_office, 'clothing_store': clothing_store, 'pharmacy': pharmacy,
    'grocery_store': grocery_store, 'bank': bank, 'movie_theater': movie_theater,
}

def cafe():
    # The cafe's props are hand-drawn; only its doors come from here, in the
    # colors of runner/sprites/generate_iso_walls.py.
    door_set('cafe', (236, 224, 204), (172, 122, 80), (116, 86, 62))


WORLDS['cafe'] = cafe

if __name__ == '__main__':
    for name in (sys.argv[1:] or list(WORLDS)):
        WORLDS[name]()
        print('wrote', name)
