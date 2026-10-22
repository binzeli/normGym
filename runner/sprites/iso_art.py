"""A small toolkit for drawing isometric pixel-art props in code.

Props are drawn in the tile's own isometric coordinates and come out crisp
(no anti-aliasing), styled like the hand-drawn cafe props: every surface is
outlined in a darker shade of its own color, faces have three tones (light
top, mid front-left, dark front-right), and panels are inset with a shadow
and a highlight line.

Coordinates: a sprite is TILE_W (80) wide; the cell's floor diamond sits at
its bottom. A point is (u, v, z): u runs 0..1 along +col (screen down-right),
v 0..1 along +row (screen down-left), z is height in pixels above the floor.
    (0,0) top vertex   (1,0) right vertex   (1,1) bottom   (0,1) left
The two faces of a box that face the camera are 'left' (the v = v1 side)
and 'right' (the u = u1 side). Face-local coordinates (s, t) run 0..1 left
to right on screen and bottom to top.
"""
import random

from PIL import Image, ImageDraw

TILE_W, TILE_H = 80, 40


def shade(c, f):
    return tuple(max(0, min(255, round(v * f))) for v in c[:3])


def mix(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


# Tone factors for the three visible faces of a box (light from top-left).
TOP, LEFT, RIGHT, EDGE = 1.12, 0.94, 0.74, 0.55


class Sprite:
    def __init__(self, height, width=TILE_W):
        self.w, self.h = width, height
        self.img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    # -- projection ---------------------------------------------------------
    def p(self, u, v, z=0.0):
        return (self.w / 2 + (TILE_W / 2) * (u - v),
                self.h - TILE_H + (TILE_H / 2) * (u + v) - z)

    def poly(self, pts3, fill, outline=None):
        pts = [self.p(*q) for q in pts3]
        self.d.polygon(pts, fill=tuple(fill) + (255,),
                       outline=(tuple(outline) + (255,)) if outline else None)

    def line(self, a, b, color, width=1):
        self.d.line([self.p(*a), self.p(*b)], fill=tuple(color) + (255,), width=width)

    def px(self, u, v, z, color):
        x, y = self.p(u, v, z)
        self.d.point((round(x), round(y)), fill=tuple(color) + (255,))

    # -- boxes --------------------------------------------------------------
    def box(self, u0, u1, v0, v1, z0, z1, color, outline=True, top=True,
            top_color=None, left_color=None, right_color=None):
        """A solid box; returns a Box handle for drawing on its faces."""
        e = shade(color, EDGE) if outline else None
        lc = left_color or shade(color, LEFT)
        rc = right_color or shade(color, RIGHT)
        self.poly([(u0, v1, z0), (u1, v1, z0), (u1, v1, z1), (u0, v1, z1)], lc, e)
        self.poly([(u1, v1, z0), (u1, v0, z0), (u1, v0, z1), (u1, v1, z1)], rc, e)
        if top:
            tc = top_color or shade(color, TOP)
            self.poly([(u0, v0, z1), (u1, v0, z1), (u1, v1, z1), (u0, v1, z1)], tc, e)
            # A highlight along the two front edges of the top.
            hl = shade(tc, 1.12)
            self.line((u0 + 0.02, v1, z1), (u1 - 0.01, v1, z1), hl)
            self.line((u1, v1 - 0.01, z1), (u1, v0 + 0.02, z1), hl)
        return Box(self, u0, u1, v0, v1, z0, z1)

    def slab(self, u0, u1, v0, v1, z, color, outline=True):
        """A flat quad at height z (a rug, a paper, a screen lying flat)."""
        self.poly([(u0, v0, z), (u1, v0, z), (u1, v1, z), (u0, v1, z)], color,
                  shade(color, EDGE) if outline else None)

    def leg(self, u, v, z0, z1, color, w=0.035):
        self.box(u - w, u + w, v - w, v + w, z0, z1, color, top=False)

    def save(self, path):
        self.img.save(path)


class Box:
    """Drawing on the two camera-facing faces and the top of a box."""

    def __init__(self, sp, u0, u1, v0, v1, z0, z1):
        self.sp, self.u0, self.u1, self.v0, self.v1, self.z0, self.z1 = sp, u0, u1, v0, v1, z0, z1

    def fp(self, face, s, t):
        """Face-local (s, t) -> (u, v, z)."""
        z = self.z0 + t * (self.z1 - self.z0)
        if face == 'left':
            return (self.u0 + s * (self.u1 - self.u0), self.v1, z)
        return (self.u1, self.v1 - s * (self.v1 - self.v0), z)

    def tp(self, a, b):
        """Top-local (a along u, b along v) -> (u, v, z1)."""
        return (self.u0 + a * (self.u1 - self.u0), self.v0 + b * (self.v1 - self.v0), self.z1)

    def rect(self, face, s0, s1, t0, t1, color, outline=None):
        pts = [self.fp(face, s0, t0), self.fp(face, s1, t0),
               self.fp(face, s1, t1), self.fp(face, s0, t1)]
        self.sp.poly(pts, color, outline)

    def panel(self, face, s0, s1, t0, t1, color):
        """An inset panel: darker field, shadow on top/left, light bottom/right."""
        self.rect(face, s0, s1, t0, t1, shade(color, 0.92))
        dark, light = shade(color, 0.62), shade(color, 1.18)
        self.sp.line(self.fp(face, s0, t1), self.fp(face, s1, t1), dark)
        self.sp.line(self.fp(face, s0, t0), self.fp(face, s0, t1), dark)
        self.sp.line(self.fp(face, s0, t0), self.fp(face, s1, t0), light)
        self.sp.line(self.fp(face, s1, t0), self.fp(face, s1, t1), light)

    def hline(self, face, t, color, s0=0.0, s1=1.0):
        self.sp.line(self.fp(face, s0, t), self.fp(face, s1, t), color)

    def vline(self, face, s, color, t0=0.0, t1=1.0):
        self.sp.line(self.fp(face, s, t0), self.fp(face, s, t1), color)

    def top_rect(self, a0, a1, b0, b1, color, outline=None, dz=0.0):
        pts = [self.tp(a0, b0), self.tp(a1, b0), self.tp(a1, b1), self.tp(a0, b1)]
        self.sp.poly([(u, v, z + dz) for u, v, z in pts], color, outline)


# Book-spine palette shared by bookshelves, desks and returns.
BOOKS = [(176, 62, 52), (58, 92, 150), (64, 128, 84), (196, 150, 58), (116, 70, 128),
         (186, 104, 60), (52, 118, 128), (150, 46, 66), (214, 196, 150), (88, 66, 52)]


def book_row(box, face, s0, s1, t0, t1, rng, lean=True):
    """A row of book spines standing on a shelf, inside face rect (s0..s1, t0..t1)."""
    face_px = 40 * abs(box.u1 - box.u0) if face == 'left' else 40 * abs(box.v1 - box.v0)
    span = (s1 - s0) * face_px                 # on-screen width in px
    x = 0.0
    while x < span - 1.5:
        w = rng.choice((2, 2, 3, 3, 4))
        if x + w > span:
            break
        if rng.random() < 0.08:                # a gap
            x += w
            continue
        c = rng.choice(BOOKS)
        h = rng.uniform(0.62, 0.96)
        a, b = s0 + (s1 - s0) * x / span, s0 + (s1 - s0) * (x + w) / span
        tt = t0 + (t1 - t0) * h
        box.rect(face, a, b, t0, tt, c)
        box.vline(face, b, shade(c, 0.6), t0, tt)            # spine edge
        box.hline(face, tt, shade(c, 1.25), a, b)            # top light
        if w >= 3 and rng.random() < 0.5:                    # a band on the spine
            box.hline(face, t0 + (tt - t0) * 0.7, shade(c, 1.35), a, b)
        x += w


def rng_for(name):
    return random.Random(name)
