"""Raster graphics: draw images and write real PNG files.

Closes the "no graphics" gap without a dependency. A canvas is an ordinary
value; drawing operations mutate it and `canvas_save` writes a valid PNG
using only the host's zlib.

    let c := canvas(400, 300).
    canvas_fill(c, "#101820").
    circle(c, 200, 150, 80, "#ffcc00").
    canvas_save(c, "out.png").

Colours are "#rgb", "#rrggbb", or a Map {r, g, b}. Coordinates are pixels
with the origin at the top left; anything off-canvas is clipped rather than
raising, so generated drawings never fail on a rounding error.
"""
from __future__ import annotations

import struct
import zlib

from .errors import VMError

_FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    ".": ("000", "000", "000", "000", "010"),
    "-": ("000", "000", "111", "000", "000"),
    ":": ("000", "010", "000", "010", "000"),
    "%": ("101", "001", "010", "100", "101"),
    " ": ("000", "000", "000", "000", "000"),
    "A": ("111", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("111", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "111", "100", "111"),
    "F": ("111", "100", "111", "100", "100"),
    "G": ("111", "100", "101", "101", "111"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "111"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "111"),
    "P": ("111", "101", "111", "100", "100"),
    "Q": ("111", "101", "101", "111", "011"),
    "R": ("111", "101", "110", "101", "101"),
    "S": ("111", "100", "111", "001", "111"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
}


class _Canvas:
    __slots__ = ("w", "h", "px")

    def __init__(self, w, h):
        self.w = w
        self.h = h
        # one bytearray, RGB triples, row-major
        self.px = bytearray(w * h * 3)


def _need(c, where):
    if not isinstance(c, _Canvas):
        raise VMError(f"{where}: first argument must be a canvas from canvas()")
    return c


def _colour(value, where):
    """Parse '#rgb', '#rrggbb' or {r, g, b} into a byte triple."""
    if isinstance(value, dict):
        try:
            return tuple(max(0, min(255, int(value[k]))) for k in ("r", "g", "b"))
        except (KeyError, TypeError, ValueError):
            raise VMError(f"{where}: colour Map needs numeric r, g and b") from None
    if not isinstance(value, str):
        raise VMError(f"{where}: colour must be Text like \"#ff8800\" or a Map")
    s = value.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise VMError(f"{where}: bad colour '{value}'; use \"#rgb\" or \"#rrggbb\"")
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise VMError(f"{where}: bad colour '{value}'") from None


def canvas(width, height):
    """Create a canvas. Pixels start black."""
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        raise VMError("canvas: width and height must be positive")
    if w * h > 40_000_000:
        raise VMError("canvas: image too large (over 40 megapixels)")
    return _Canvas(w, h)


def canvas_size(c):
    _need(c, "canvas_size")
    return {"width": c.w, "height": c.h}


def _set(c, x, y, rgb):
    if 0 <= x < c.w and 0 <= y < c.h:
        i = (y * c.w + x) * 3
        c.px[i:i + 3] = bytes(rgb)


def pixel(c, x, y, colour):
    """Set one pixel."""
    _need(c, "pixel")
    _set(c, int(x), int(y), _colour(colour, "pixel"))
    return None


def canvas_fill(c, colour):
    """Fill the whole canvas."""
    _need(c, "canvas_fill")
    c.px[:] = bytes(_colour(colour, "canvas_fill")) * (c.w * c.h)
    return None


def rect(c, x, y, w, h, colour, filled=True):
    """Draw a rectangle."""
    _need(c, "rect")
    rgb = _colour(colour, "rect")
    x, y, w, h = int(x), int(y), int(w), int(h)
    if filled:
        row = bytes(rgb) * max(0, min(w, c.w - x) if x >= 0 else min(w + x, c.w))
        for yy in range(max(0, y), min(y + h, c.h)):
            xs = max(0, x)
            xe = min(x + w, c.w)
            if xe > xs:
                i = (yy * c.w + xs) * 3
                c.px[i:i + (xe - xs) * 3] = bytes(rgb) * (xe - xs)
    else:
        for xx in range(x, x + w):
            _set(c, xx, y, rgb)
            _set(c, xx, y + h - 1, rgb)
        for yy in range(y, y + h):
            _set(c, x, yy, rgb)
            _set(c, x + w - 1, yy, rgb)
    return None


def line(c, x0, y0, x1, y1, colour):
    """Draw a straight line (Bresenham)."""
    _need(c, "line")
    rgb = _colour(colour, "line")
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _set(c, x0, y0, rgb)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return None


def circle(c, cx, cy, r, colour, filled=True):
    """Draw a circle."""
    _need(c, "circle")
    rgb = _colour(colour, "circle")
    cx, cy, r = int(cx), int(cy), int(r)
    if r <= 0:
        return None
    if filled:
        rr = r * r
        for yy in range(max(0, cy - r), min(cy + r + 1, c.h)):
            dy = yy - cy
            span = int((rr - dy * dy) ** 0.5) if rr >= dy * dy else -1
            if span < 0:
                continue
            xs, xe = max(0, cx - span), min(cx + span + 1, c.w)
            if xe > xs:
                i = (yy * c.w + xs) * 3
                c.px[i:i + (xe - xs) * 3] = bytes(rgb) * (xe - xs)
    else:
        x, y, err = r, 0, 0
        while x >= y:
            for px, py in ((x, y), (y, x), (-x, y), (-y, x),
                           (-x, -y), (-y, -x), (x, -y), (y, -x)):
                _set(c, cx + px, cy + py, rgb)
            y += 1
            err += 1 + 2 * y
            if 2 * (err - x) + 1 > 0:
                x -= 1
                err += 1 - 2 * x
    return None


def text(c, x, y, message, colour, scale=2):
    """Draw text with the built-in pixel font (A-Z, 0-9 and . - : %)."""
    _need(c, "text")
    rgb = _colour(colour, "text")
    x, y, scale = int(x), int(y), max(1, int(scale))
    cursor = x
    for ch in str(message).upper():
        glyph = _FONT.get(ch)
        if glyph is None:
            cursor += 4 * scale
            continue
        for ry, row in enumerate(glyph):
            for rx, bit in enumerate(row):
                if bit == "1":
                    for sy in range(scale):
                        for sx in range(scale):
                            _set(c, cursor + rx * scale + sx, y + ry * scale + sy, rgb)
        cursor += 4 * scale
    return None


def canvas_save(c, path):
    """Write the canvas as a PNG file."""
    _need(c, "canvas_save")
    raw = bytearray()
    stride = c.w * 3
    for y in range(c.h):
        raw.append(0)  # filter: none
        raw += c.px[y * stride:(y + 1) * stride]

    def chunk(tag, data):
        out = struct.pack(">I", len(data)) + tag + data
        return out + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", c.w, c.h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )
    try:
        with open(str(path), "wb") as fh:
            fh.write(png)
    except OSError as e:
        raise VMError(f"canvas_save: cannot write '{path}': {e}") from e
    return len(png)


def plot(path, series, width=640, height=400, colour="#3fa7ff", background="#0d1117"):
    """Write a line chart of a list of numbers. One call, one PNG."""
    if not isinstance(series, list) or not series:
        raise VMError("plot: series must be a non-empty List of numbers")
    try:
        values = [float(v) for v in series]
    except (TypeError, ValueError):
        raise VMError("plot: every item in the series must be a number") from None
    c = canvas(width, height)
    canvas_fill(c, background)
    pad = 30
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    grid = _colour("#243040", "plot")
    for i in range(5):
        gy = pad + int((height - 2 * pad) * i / 4)
        for gx in range(pad, width - pad):
            _set(c, gx, gy, grid)
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        px = pad + int((width - 2 * pad) * (i / max(1, n - 1)))
        py = height - pad - int((height - 2 * pad) * ((v - lo) / span))
        pts.append((px, py))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        line(c, x0, y0, x1, y1, colour)
    text(c, pad, 8, f"{hi:.2f}", "#8899aa", 1)
    text(c, pad, height - 18, f"{lo:.2f}", "#8899aa", 1)
    return canvas_save(c, path)
