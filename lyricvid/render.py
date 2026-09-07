"""Beat-reactive kinetic lyric-video renderer (Pillow + numpy -> ffmpeg pipe).

Full-frame maths runs on uint8 PIL images; the ambient light layer is composed
at 1/4 resolution and upscaled; only the lyric region and spectrum strip are
touched in float32.
"""
from __future__ import annotations

import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from .audio import Analysis
from .lyrics import Line, Song

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "ttf")
DISPLAY = "Anton.ttf"
BODY = "SpaceGrotesk-Bold.ttf"


@dataclass
class Palette:
    name: str
    bg_top: tuple
    bg_bottom: tuple
    accents: list
    done: tuple = (236, 238, 242)
    todo: tuple = (122, 128, 140)
    glow: float = 0.85
    grain: float = 0.045
    scan: float = 0.055
    vignette: float = 0.62
    pulse: float = 0.055
    chroma: float = 3.2
    bars: bool = True
    halftone: bool = False
    light_res: int = 4


PALETTES = {
    "kinetic": Palette("kinetic", (9, 10, 15), (21, 23, 33),
                       [(255, 45, 111), (0, 229, 255), (198, 255, 61), (255, 145, 0),
                        (157, 102, 255), (0, 255, 170)]),
    "minimal": Palette("minimal", (246, 244, 239), (226, 224, 217),
                       [(24, 24, 27), (60, 60, 70), (120, 90, 60), (40, 70, 90)],
                       done=(30, 30, 34), todo=(170, 167, 160), glow=0.0, grain=0.016, scan=0.0,
                       vignette=0.16, pulse=0.018, chroma=0.0, bars=False, light_res=6),
    "zine": Palette("zine", (233, 229, 219), (203, 197, 183),
                    [(214, 40, 40), (20, 20, 22), (240, 120, 20), (30, 60, 160)],
                    done=(32, 32, 35), todo=(124, 120, 112), glow=0.0, grain=0.10, scan=0.025,
                    vignette=0.34, pulse=0.07, chroma=1.5, bars=False, halftone=True, light_res=6),
    "reactive": Palette("reactive", (4, 5, 9), (13, 15, 26),
                        [(0, 229, 255), (157, 102, 255), (0, 255, 170), (255, 45, 111)],
                        glow=1.2, grain=0.04, scan=0.07, vignette=0.7, pulse=0.04, chroma=4.5,
                        bars=True),
}


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = os.path.join(FONT_DIR, name)
    if not os.path.exists(path):
        path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(path, max(6, int(size)))


def make_gradient(w, h, top, bottom, angle_deg=90.0) -> np.ndarray:
    ang = math.radians(angle_deg)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = (xs / max(w, 1)) * math.cos(ang) + (ys / max(h, 1)) * math.sin(ang)
    proj = (proj - proj.min()) / max(float(np.ptp(proj)), 1e-6)
    t = np.asarray(top, np.float32) / 255.0
    b = np.asarray(bottom, np.float32) / 255.0
    return t * (1 - proj[..., None]) + b * proj[..., None]


def make_radial(w, h, cx=0.5, cy=0.5, soft=0.62) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xs / max(w, 1) - cx) * (w / max(h, 1))
    dy = (ys / max(h, 1) - cy)
    return (np.clip(1.0 - np.sqrt(dx * dx + dy * dy) / soft, 0, 1) ** 1.7).astype(np.float32)


def make_vignette(w, h, strength) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xs / w - 0.5) * 2.0
    ny = (ys / h - 0.5) * 2.0
    d = np.sqrt(nx * nx * 0.85 + ny * ny)
    return np.clip(1.0 - strength * np.clip(d - 0.35, 0, 1) ** 1.5, 0.25, 1.0).astype(np.float32)[..., None]


def make_scanlines(h, w, strength, period=3) -> np.ndarray:
    if strength <= 0:
        return np.ones((h, 1, 1), np.float32)
    row = np.ones(h, np.float32)
    row[::period] = 1.0 - strength
    return row[:, None, None]


def make_halftone(w, h, period=7) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    px = (xs % period) - period / 2
    py = (ys % period) - period / 2
    d = np.sqrt(px * px + py * py) / (period / 2)
    return np.clip(0.55 + 0.45 * (1 - d), 0.45, 1.0).astype(np.float32)[..., None]


def make_grain_tiles(w, h, amplitude, n=5, seed=7):
    if amplitude <= 0:
        return []
    rng = np.random.default_rng(seed)
    lvl = int(round(amplitude * 255))
    tiles = []
    for _ in range(n):
        small = rng.integers(0, 256, (h // 4 + 1, w // 4 + 1), dtype=np.uint8)
        big = np.repeat(np.repeat(small, 4, 0)[:h], 4, 1)[:, :w].astype(np.int16) - 128
        pos = np.clip(big, 0, None) * lvl // 128
        neg = np.clip(-big, 0, None) * lvl // 128
        tiles.append((Image.fromarray(np.repeat(pos[..., None], 3, 2).astype(np.uint8), "RGB"),
                      Image.fromarray(np.repeat(neg[..., None], 3, 2).astype(np.uint8), "RGB")))
    return tiles


@dataclass
class LineLayout:
    line: Line
    subs: list
    rows: list
    row_w: list
    size: int
    block_w: int
    block_h: int
    row_h: int
    accent_idx: int
    kicker: str = ""
    flat: list = field(default_factory=list)
    pad: int = 0


def _wrap(words, widths, space, max_w, max_lines):
    subs = [[]]
    cur = 0
    for w_txt, w_w in zip(words, widths):
        add = w_w + (space if subs[-1] else 0)
        if cur + add > max_w and subs[-1]:
            if len(subs) >= max_lines:
                return None
            subs.append([(w_txt, w_w)])
            cur = w_w
        else:
            subs[-1].append((w_txt, w_w))
            cur += add
    return subs


def layout_line(line: Line, W: int, H: int, max_size: int, min_size: int, accent_idx: int,
                max_lines: int = 2) -> LineLayout:
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    max_w = int(W * 0.86)
    words = line.words or line.text.split()
    subs, size, space = None, min_size, 0
    cand = max_size
    while cand >= min_size:
        f = font(DISPLAY, cand)
        widths = [int(math.ceil(probe.textlength(w, font=f))) for w in words]
        space = int(math.ceil(probe.textlength(" ", font=f)))
        got = _wrap(words, widths, space, max_w, max_lines)
        if got is not None:
            subs, size = got, cand
            break
        cand = max(min_size, cand - 6)
    if subs is None:
        f = font(DISPLAY, min_size)
        widths = [int(math.ceil(probe.textlength(w, font=f))) for w in words]
        space = int(math.ceil(probe.textlength(" ", font=f)))
        subs = _wrap(words, widths, space, max_w, max_lines + 2) or \
            [[(w, wv) for w, wv in zip(words, widths)]]
        size = min_size
    row_w, widths_row = [], []
    for sub in subs:
        x, ws = 0, []
        for i, (_t, ww) in enumerate(sub):
            ws.append(x)
            x += ww + (space if i < len(sub) - 1 else 0)
        row_w.append(max(1, x - (space if len(sub) > 1 else 0)))
        widths_row.append(ws)
    block_w = max(row_w) if row_w else 10
    rows = [[x + (block_w - rw) // 2 for x in ws] for ws, rw in zip(widths_row, row_w)]
    row_h = int(size * 1.16)
    pad = int(size * 0.40)
    flat = [(w, x + pad, pad + ri * row_h) for ri, (sub, xs) in enumerate(zip(subs, rows))
            for (w, _ww), x in zip(sub, xs)]
    return LineLayout(line=line, subs=subs, rows=rows, row_w=row_w, size=size, block_w=block_w,
                      block_h=row_h * len(subs), row_h=row_h, accent_idx=accent_idx, flat=flat, pad=pad)


@dataclass
class RenderConfig:
    W: int = 1920
    H: int = 1080
    fps: int = 30
    style: str = "kinetic"
    title: str = ""
    artist: str = ""
    bars_visible: bool = True
    grain_scale: float = 1.0
    lead_in: float = 2.2
    outro: float = 2.0
    max_font: int = 132
    min_font: int = 46
    show_progress: bool = True
    show_kicker: bool = True
    bg_paths: list = field(default_factory=list)


class Renderer:
    def __init__(self, an: Analysis, song: Song, cfg: RenderConfig):
        self.an, self.song, self.cfg = an, song, cfg
        self.pal = PALETTES.get(cfg.style, PALETTES["kinetic"])
        self.W, self.H = cfg.W, cfg.H
        pal, W, H = self.pal, self.W, self.H
        self.base_cache: dict = {}
        lr = max(2, pal.light_res)
        self.lw, self.lh = max(8, W // lr), max(8, H // lr)
        self.g0s = make_radial(self.lw, self.lh, 0.5, 0.44, 0.75)
        self.g1s = make_radial(self.lw, self.lh, 0.26, 0.28, 0.44)
        self.g2s = make_radial(self.lw, self.lh, 0.76, 0.68, 0.48)
        self.grain = make_grain_tiles(W, H, pal.grain * cfg.grain_scale, n=5)
        self.lut_cache: dict = {}
        self.positions = [(0.5, 0.5), (0.5, 0.665), (0.5, 0.355), (0.5, 0.575)]
        self.layouts: list[LineLayout] = []
        prev_sec = None
        for ln in song.lines:
            lay = layout_line(ln, W, H, cfg.max_font, cfg.min_font,
                              self.phrase_index(ln.start) % max(len(pal.accents), 1))
            if ln.section and ln.section != prev_sec:
                lay.kicker = ln.section
            prev_sec = ln.section or prev_sec
            self.layouts.append(lay)
        self.line_positions = [self.positions[self.phrase_index(ln.start) % len(self.positions)]
                               for ln in song.lines]
        lum = 0.2126 * pal.bg_top[0] + 0.7152 * pal.bg_top[1] + 0.0722 * pal.bg_top[2]
        self.ink = (255, 255, 255) if lum < 128 else (22, 22, 24)
        self.duration = max(an.duration, song.duration or 0.0)
        self._spec: dict = {}
        self._grid_t0 = 0.0
        self._line_starts = np.array([ln.start for ln in song.lines]) if song.lines else np.zeros(0)
        self._line_ends = np.array([ln.end for ln in song.lines]) if song.lines else np.zeros(0)
        self.bg_plates: list = []
        self.bg_secs: list = []
        self._scrim = None
        if cfg.bg_paths:
            pw, phh = int(W * 1.30), int(H * 1.30)
            for p in cfg.bg_paths:
                im = Image.open(p).convert("RGB").resize((pw, phh), Image.LANCZOS)
                arr = (np.asarray(im).astype(np.float32) * 0.74).astype(np.uint8)
                self.bg_plates.append(Image.fromarray(arr, "RGB"))
            secs: list = []
            prev = object()
            for ln in song.lines:
                nm = (ln.section or "").strip()
                if nm != prev:
                    secs.append([ln.start, ln.end, nm])
                    prev = nm
                else:
                    secs[-1][1] = ln.end
            if secs:
                secs[0][0] = 0.0
                secs[-1][1] = self.duration + 1.0
                np_ = max(len(self.bg_plates), 1)
                tagged = [[s, e,
                           (1 if ("chorus" in nm.lower() or "hook" in nm.lower())
                            else 2 if ("bridge" in nm.lower() or "outro" in nm.lower())
                            else 0) % np_]
                          for s, e, nm in secs]
                co: list = []
                for s, e, pi in tagged:          # absorb mis-tagged islands (<4s)
                    if co and (e - s) < 4.0:
                        co[-1][1] = e
                    else:
                        co.append([s, e, pi])
                self.bg_secs = []
                for s, e, pi in co:              # merge same-plate neighbours
                    if self.bg_secs and self.bg_secs[-1][2] == pi:
                        self.bg_secs[-1][1] = e
                    else:
                        self.bg_secs.append([s, e, pi])
            self._scrim = (1.0 - 0.42 * make_radial(W, H, 0.5, 0.52, 0.85))[..., None].astype(np.float32)

    # ------------------------------------------------------------------ music
    def phrase_index(self, t: float) -> int:
        bars = self.an.bar_times
        if bars.size == 0:
            return int(max(t, 0) // 8)
        return int(np.searchsorted(bars, t)) // 8

    def beat_phase(self, t: float):
        b = self.an.beat_times
        default_iv = 60.0 / max(self.an.tempo or 90.0, 40.0)
        if b.size == 0:
            return t % default_iv, default_iv
        i = int(np.searchsorted(b, t)) - 1
        if i < 0:
            return 0.0, default_iv
        iv = float(b[min(i + 1, b.size - 1)] - b[i]) or default_iv
        return float(t - b[i]), max(iv, 0.12)

    def is_bar_hit(self, t: float, tol: float = 0.07) -> bool:
        bars = self.an.bar_times
        if bars.size == 0:
            return False
        i = int(np.argmin(np.abs(bars - t)))
        return abs(float(bars[i]) - t) <= tol

    def feat(self, name: str, t: float) -> float:
        arr = self._spec.get(name)
        if arr is None:
            return 0.0
        i = int((t - self._grid_t0) * self.cfg.fps)
        return float(arr[min(max(i, 0), arr.shape[0] - 1)])

    def precompute_grid(self, t_start: float, t_end: float):
        fps = self.cfg.fps
        n = int(math.ceil((t_end - t_start) * fps)) + 1
        grid = t_start + np.arange(n) / fps
        s = self.an.sample(grid)
        k = max(1, int(fps * 0.07))
        ker = np.ones(k, np.float32) / k
        for key in ("rms", "vocal"):
            s[key] = np.convolve(s[key], ker, mode="same")
        self._spec["rms"] = np.clip(s["rms"] / max(float(s["rms"].max()), 1e-6), 0, 1).astype(np.float32)
        self._spec["vocal"] = np.clip(s["vocal"] / max(float(s["vocal"].max()), 1e-6), 0, 1).astype(np.float32)
        c = s["centroid"]
        lo, hi = np.percentile(c, 5), np.percentile(c, 95)
        self._spec["cent"] = np.clip((c - lo) / max(hi - lo, 1e-3), 0, 1).astype(np.float32)
        self._spec["onset"] = np.clip(s["onset"] / max(float(s["onset"].max()), 1e-6), 0, 1).astype(np.float32)
        b = self.an.bands
        if b.size:
            idx = np.clip((grid / max(self.an.hop / self.an.sr, 1e-6)).astype(int), 0, b.shape[1] - 1)
            pf = b[:, idx].T
            self._spec["bands"] = np.clip(pf / (float(np.percentile(pf, 99.0)) or 1.0), 0, 1).astype(np.float32)
        self._grid_t0 = t_start

    def active_line(self, t: float) -> int:
        if self._line_starts.size == 0:
            return -1
        i = int(np.searchsorted(self._line_starts, t + 0.02)) - 1
        if i < 0:
            return -1
        return i if t < self._line_ends[i] + 0.12 else -1

    # ------------------------------------------------------------- background
    def base_for_phrase(self, ph: int) -> Image.Image:
        img = self.base_cache.get(ph)
        if img is not None:
            return img
        pal, W, H = self.pal, self.W, self.H
        mix = 0.5 + 0.5 * math.sin(ph * 1.7)
        base = (make_gradient(W, H, pal.bg_top, pal.bg_bottom, 90.0) * (1 - 0.4 * mix)
                + make_gradient(W, H, pal.bg_bottom, pal.bg_top, 62.0) * (0.4 * mix))
        base = base * make_vignette(W, H, pal.vignette) * make_scanlines(H, W, pal.scan)
        if pal.halftone:
            base = base * make_halftone(W, H)
        img = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8), "RGB")
        self.base_cache[ph] = img
        if len(self.base_cache) > 6:
            self.base_cache.pop(next(iter(self.base_cache)))
        return img

    def _bg_frame(self, t: float, beat_env: float) -> Image.Image:
        """Animated cinematic backdrop: Ken Burns drift per section (hard cut on
        section change) plus a subtle beat pump and a centre scrim for legibility."""
        secs = self.bg_secs
        i = len(secs) - 1
        for k, sec in enumerate(secs):
            if t < sec[1]:
                i = k
                break
        s0, s1, pi = secs[i]
        plate = self.bg_plates[min(pi, len(self.bg_plates) - 1)]
        PW, PH = plate.size
        p = float(np.clip((t - s0) / max(s1 - s0, 1.0), 0, 1))
        zoom = 1.04 + 0.14 * p + 0.012 * beat_env
        par = 1 if i % 2 == 0 else -1
        cx = 0.5 + par * 0.05 * (p - 0.5)
        cy = 0.5 - par * 0.04 * (p - 0.5)
        ww, wh = PW / zoom, PH / zoom
        x0 = min(max(cx * PW - ww / 2, 0), PW - ww)
        y0 = min(max(cy * PH - wh / 2, 0), PH - wh)
        crop = plate.crop((int(x0), int(y0), int(x0 + ww), int(y0 + wh)))
        arr = np.asarray(crop.resize((self.W, self.H), Image.BILINEAR)).astype(np.float32)
        arr = arr * (1.0 / 255.0) * self._scrim
        return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")

    def lut(self, key: int, bright: float) -> list:
        hit = self.lut_cache.get(key)
        if hit is None:
            hit = [min(255, int(i * bright + 0.5)) for i in range(256)] * 3
            self.lut_cache[key] = hit
        return hit

    def light_layer(self, ph, rms, voc, cen, beat_env, flash) -> Image.Image:
        pal = self.pal
        c1 = np.asarray(pal.accents[ph % len(pal.accents)], np.float32) / 255.0
        c2 = np.asarray(pal.accents[(ph + 2) % len(pal.accents)], np.float32) / 255.0
        g = pal.glow
        light = (self.g0s * ((0.10 + 0.32 * rms + 0.18 * beat_env) * g))[..., None] * (c1 * (0.55 + 0.45 * cen))
        light += (self.g1s * ((0.05 + 0.17 * voc) * g))[..., None] * c2
        light += (self.g2s * ((0.04 + 0.13 * beat_env) * g))[..., None] * (c1 * 0.7)
        if flash > 0:
            light += c1 * (0.13 * flash * g)
        return Image.fromarray((np.clip(light, 0, 1) * 255).astype(np.uint8), "RGB").resize(
            (self.W, self.H), Image.BILINEAR)

    # ------------------------------------------------------------------- text
    @lru_cache(maxsize=40)
    def _line_state_canvas(self, li: int, cur: int, pop: int):
        lay = self.layouts[li]
        pal = self.pal
        f = font(DISPLAY, lay.size)
        cw = lay.block_w + lay.pad * 2
        ch = lay.block_h + lay.pad * 2 + (int(lay.size * 0.34) if lay.kicker else 0)
        base = Image.new("RGBA", (max(8, cw), max(8, ch)), (0, 0, 0, 0))
        lit = Image.new("RGBA", (max(8, cw), max(8, ch)), (0, 0, 0, 0))
        db, dl = ImageDraw.Draw(base), ImageDraw.Draw(lit)
        accent = pal.accents[lay.accent_idx % len(pal.accents)]
        yoff = int(lay.size * 0.34) if lay.kicker else 0
        if lay.kicker and self.cfg.show_kicker:
            kf = font(BODY, max(18, int(lay.size * 0.185)))
            db.text((lay.pad + 2, max(0, yoff - int(lay.size * 0.30))), lay.kicker.upper(),
                    font=kf, fill=accent + (225,))
        lit_box = None
        for wi, (word, x, y) in enumerate(lay.flat):
            if wi == cur:
                k = pop / 100.0
                col = tuple(int(a + (255 - a) * (0.55 * k)) for a in accent)
                dl.text((x, y + yoff + (-4 if pop > 45 else 0)), word, font=f, fill=col + (255,))
                wb = dl.textbbox((x, y + yoff + (-4 if pop > 45 else 0)), word, font=f)
                lit_box = (max(0, wb[0] - 6), max(0, wb[1] - 6), min(cw, wb[2] + 6), min(ch, wb[3] + 6))
            else:
                db.text((x, y + yoff), word, font=f, fill=(pal.done if wi < cur else pal.todo) + (255,))
        if lit_box is None:
            lit_box = (0, 0, 4, 4)
        return base, lit.crop(lit_box), lit_box

    def _glow_alpha(self, canvas: Image.Image) -> np.ndarray:
        a_img = canvas.split()[3]
        small = a_img.resize((max(4, a_img.width // 4), max(4, a_img.height // 4)), Image.BILINEAR)
        small = small.filter(ImageFilter.GaussianBlur(radius=3.4))
        return np.asarray(small.resize(a_img.size, Image.BILINEAR)).astype(np.float32) * (1.0 / 255.0)

    def _chroma(self, canvas: Image.Image, amount: float, accent):
        a = np.asarray(canvas)
        alpha = a[..., 3].astype(np.float32) * (1.0 / 255.0)
        rgb = a[..., :3].astype(np.float32) * (1.0 / 255.0)
        tint = np.asarray(accent, np.float32) / 255.0
        if amount <= 0.45:
            return rgb, np.repeat(alpha[..., None], 3, 2)
        fringe = np.clip(tint * 0.45 + 0.62, 0, 1)
        rgb = np.where(alpha[..., None] > 0.02, rgb, fringe)
        off = int(round(amount))
        a_r = np.clip(np.maximum(alpha, np.roll(alpha, -off, axis=1) * 0.85), 0, 1)
        a_b = np.clip(np.maximum(alpha, np.roll(alpha, off, axis=1) * 0.85), 0, 1)
        return rgb, np.stack([a_r, alpha, a_b], 2)

    def _blit(self, img: Image.Image, x: int, y: int, rgb: np.ndarray, alpha: np.ndarray, mul: float = 1.0):
        W, H = self.W, self.H
        h, w = alpha.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        sx, sy = x0 - x, y0 - y
        hh, ww = y1 - y0, x1 - x0
        region = np.asarray(img.crop((x0, y0, x1, y1))).astype(np.float32) * (1.0 / 255.0)
        a = alpha[sy:sy + hh, sx:sx + ww] * mul
        col = rgb[sy:sy + hh, sx:sx + ww]
        out = region * (1 - a) + col * a
        img.paste(Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGB"), (x0, y0))

    def _additive(self, img: Image.Image, x: int, y: int, layer: np.ndarray):
        W, H = self.W, self.H
        h, w = layer.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        sx, sy = x0 - x, y0 - y
        hh, ww = y1 - y0, x1 - x0
        base = np.asarray(img.crop((x0, y0, x1, y1))).astype(np.float32) * (1.0 / 255.0)
        out = np.clip(base + layer[sy:sy + hh, sx:sx + ww], 0, 1)
        img.paste(Image.fromarray((out * 255).astype(np.uint8), "RGB"), (x0, y0))

    # ------------------------------------------------------------------ frame
    def frame(self, t: float, idx: int) -> np.ndarray:
        cfg, pal, W, H = self.cfg, self.pal, self.W, self.H
        rms, voc = self.feat("rms", t), self.feat("vocal", t)
        cen = self.feat("cent", t)
        since, iv = self.beat_phase(t)
        beat_env = math.exp(-since / max(iv * 0.42, 0.06))
        ph = self.phrase_index(t)
        accent = pal.accents[ph % len(pal.accents)]

        img = self.base_for_phrase(ph).copy() if not self.bg_plates else self._bg_frame(t, beat_env)
        bright = (0.86 + 0.18 * rms + 0.12 * beat_env) if self.bg_plates else \
                 (0.88 + 0.26 * rms + 0.20 * beat_env)
        img = img.point(self.lut(int(round(bright * 200)), bright))
        if pal.glow > 0:
            flash = 1.0 if self.is_bar_hit(t, 1.3 / cfg.fps) else 0.0
            img = ImageChops.screen(img, self.light_layer(ph, rms, voc, cen, beat_env, flash))

        li = self.active_line(t)
        if t < cfg.lead_in and (cfg.title or cfg.artist):
            self._draw_card(img, t, accent)
        elif li >= 0:
            self._draw_lyric(img, t, li, accent, rms, beat_env, iv)
        else:
            self._draw_idle(img, t, accent, beat_env)

        if cfg.bars_visible and pal.bars and "bands" in self._spec:
            self._draw_bars(img, t, accent, beat_env)

        if self.grain:
            pos, neg = self.grain[idx % len(self.grain)]
            img = ImageChops.subtract(ImageChops.add(img, pos), neg)

        if cfg.show_progress:
            arr = np.asarray(img).copy()
            prog = float(np.clip(t / max(self.duration, 1e-6), 0, 1))
            x = int(prog * W)
            c = (np.asarray(accent, np.float32) / 255.0 * 255).astype(np.uint8)
            rows = arr[H - 4:H]
            rows[:, :x] = (rows[:, :x].astype(np.float32) * 0.3 + c * 0.8).astype(np.uint8)
            rows[:, x:] = (rows[:, x:] * 0.72).astype(np.uint8)
            return arr
        return np.asarray(img)

    # ------------------------------------------------------------ typography
    def _draw_lyric(self, img, t, li, accent, rms, beat_env, iv):
        cfg, pal, W, H = self.cfg, self.pal, self.W, self.H
        lay = self.layouts[li]
        ln = lay.line
        age, ttl = t - ln.start, ln.end - t
        enter = float(np.clip(age / 0.20, 0, 1))
        enter_e = 1 - (1 - enter) ** 3

        wts = ln.word_times or [ln.start]
        cur = max(0, min(int(np.searchsorted(np.asarray(wts), t, side="right") - 1), len(wts) - 1))
        nxt = wts[cur + 1] if cur + 1 < len(wts) else ln.end
        wp = float(np.clip((t - wts[cur]) / max(nxt - wts[cur], 1e-3), 0, 1))
        pop = int((1 - wp) * 100) // 10

        base_c, lit_c, lit_box = self._line_state_canvas(li, cur, pop)

        px, py = self.line_positions[li]
        side = 1 if (li % 2 == 0) else -1
        slide = int((1 - enter_e) * side * W * 0.16)
        scale = (1.0 + (1 - enter_e) * 0.20) * (1.0 + pal.pulse * beat_env * (0.6 + 0.8 * rms))
        on = self.feat("onset", t)
        jx = int(round(math.sin(t * 91.0) * 2.4 * on))
        jy = int(round(math.cos(t * 77.0) * 1.8 * on))
        if abs(scale - 1.0) > 0.006:
            base_c = base_c.resize((max(8, int(base_c.width * scale)), max(8, int(base_c.height * scale))), Image.BILINEAR)
            lit_c = lit_c.resize((max(4, int(lit_c.width * scale)), max(4, int(lit_c.height * scale))), Image.BILINEAR)
            lit_box = [int(v * scale) for v in lit_box]

        cx = int(px * W - base_c.width / 2 + slide + jx)
        cy = int(py * H - base_c.height / 2 + jy)

        chroma = pal.chroma * (0.30 + 1.1 * beat_env)
        if age < 0.26:
            chroma *= 1.8
        if self.is_bar_hit(t, iv * 0.5):
            chroma *= 1.5

        if ttl < 0.12:  # slice-out hard cut
            k = max(0.0, ttl) / 0.12
            arr = np.asarray(base_c)
            nsl = 6
            sh = max(1, arr.shape[0] // nsl)
            for s_ in range(nsl):
                y0 = s_ * sh
                y1 = arr.shape[0] if s_ == nsl - 1 else y0 + sh
                piece = Image.fromarray(arr[y0:y1])
                dirn = 1 if s_ % 2 == 0 else -1
                img.paste(piece, (cx + dirn * int((1 - k) * (90 + 40 * s_)),
                                  cy + y0 + dirn * int((1 - k) * 10)), piece)
            return

        if pal.glow > 0:
            glow_a = self._glow_alpha(lit_c)
            strength = 0.55 * pal.glow * (0.5 + 0.8 * beat_env)
            c = np.asarray(accent, np.float32) / 255.0
            self._additive(img, cx + lit_box[0], cy + lit_box[1], glow_a[..., None] * c * strength)

        rgb, alpha = self._chroma(lit_c, chroma, accent)
        self._blit(img, cx + lit_box[0], cy + lit_box[1], rgb, alpha, mul=1.0 if age > 0.02 else 0.0)
        if age > 0.02:
            img.paste(base_c, (cx, cy), base_c)

    def _draw_card(self, img, t, accent):
        cfg, W, H = self.cfg, self.W, self.H
        k = 1 - (1 - float(np.clip(t / max(cfg.lead_in * 0.45, 0.2), 0, 1))) ** 3
        a = k * float(np.clip((cfg.lead_in - t) / 0.35, 0, 1))
        if a <= 0.02:
            return
        title = cfg.title or self.song.title or ""
        artist = cfg.artist or self.song.artist or ""
        cw, chh = int(W * 0.94), int(H * 0.42)
        canvas = Image.new("RGBA", (cw, chh), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)
        tsize = int(min(H * 0.16, max(40, cw * 1.62 / max(len(title), 1))))
        tf = font(DISPLAY, tsize)
        if title:
            tw = d.textlength(title.upper(), font=tf)
            d.text(((cw - tw) / 2, chh * 0.20), title.upper(), font=tf, fill=tuple(list(self.ink) + [255]))
        if artist:
            af = font(BODY, int(tsize * 0.24))
            aw = d.textlength(artist.upper(), font=af)
            d.text(((cw - aw) / 2, chh * 0.20 + tsize * 1.18), artist.upper(), font=af,
                   fill=tuple(list(accent) + [235]))
        rgb, alpha = self._chroma(canvas, 0.0, accent)
        self._blit(img, int((W - cw) / 2), int(H * 0.30 + (1 - k) * 80), rgb, alpha, mul=a)

    def _draw_idle(self, img, t, accent, beat_env):
        cfg, W, H = self.cfg, self.W, self.H
        if not (t > self.duration - cfg.outro and (cfg.title or self.song.title)):
            return
        label = (cfg.title or self.song.title).upper()
        cw = int(W * 0.7)
        canvas = Image.new("RGBA", (cw, 110), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)
        f = font(BODY, max(24, int(H * 0.038)))
        wpx = d.textlength(label, font=f)
        d.text(((cw - wpx) / 2, 24), label, font=f, fill=tuple(list(self.ink) + [int(140 + 90 * beat_env)]))
        rgb, alpha = self._chroma(canvas, 0.0, accent)
        self._blit(img, int(W * 0.15), int(H * 0.80), rgb, alpha)

    def _draw_bars(self, img, t, accent, beat_env):
        cfg, W, H = self.cfg, self.W, self.H
        bands = self._spec["bands"]
        i = int((t - self._grid_t0) * cfg.fps)
        vals = bands[min(max(i, 0), bands.shape[0] - 1)]
        nb = vals.size
        maxh = int(H * 0.072)
        base_y = H - 10
        y0 = max(0, base_y - maxh - 14)
        region = np.asarray(img.crop((0, y0, W, base_y))).astype(np.float32) * (1.0 / 255.0)
        c = np.asarray(accent, np.float32) / 255.0
        bw = W / nb
        rh = region.shape[0]
        heights = (np.clip(vals * (0.72 + 0.5 * beat_env), 0, 1.35) * maxh).astype(int)
        for b in range(nb):
            h = min(int(heights[b]), rh)
            if h <= 1:
                continue
            x0, x1 = int(b * bw), max(int(b * bw) + 2, int((b + 0.7) * bw))
            top = max(0, rh - h)
            grad = np.linspace(0.4, 1.0, rh - top, dtype=np.float32)[:, None, None]
            region[top:rh, x0:x1] = np.clip(region[top:rh, x0:x1] * 0.32
                                            + c * grad * (0.55 + 0.45 * float(vals[b])), 0, 1)
        img.paste(Image.fromarray((np.clip(region, 0, 1) * 255).astype(np.uint8), "RGB"), (0, y0))


def render_video(an, song, out_path, cfg, audio_src, t_start=0.0, t_end=None,
                 crf=20, preset="fast", progress=True) -> str:
    from .audio import ffmpeg_exe

    t_end = min(t_end or an.duration, an.duration)
    r = Renderer(an, song, cfg)
    r.precompute_grid(t_start, t_end)
    n_frames = max(1, int(round((t_end - t_start) * cfg.fps)))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{cfg.W}x{cfg.H}",
           "-pix_fmt", "rgb24", "-r", str(cfg.fps), "-i", "-"]
    if audio_src:
        cmd += ["-ss", f"{t_start:.3f}", "-i", audio_src, "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100"]
    else:
        cmd += ["-map", "0:v"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-t", f"{t_end - t_start:.3f}", "-movflags", "+faststart", out_path]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    try:
        for i in range(n_frames):
            proc.stdin.write(r.frame(t_start + i / cfg.fps, i).tobytes())
            if progress and (i % (cfg.fps * 5) == 0 or i == n_frames - 1):
                el = time.time() - t0
                rate = (i + 1) / max(el, 1e-6)
                print(f"[render] {i + 1}/{n_frames} frames  {rate:.1f} fps  "
                      f"eta {(n_frames - i - 1) / max(rate, 1e-6) / 60:.1f} min", flush=True)
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}")
    print(f"[render] wrote {out_path}  {os.path.getsize(out_path) / 1e6:.1f} MB "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out_path
