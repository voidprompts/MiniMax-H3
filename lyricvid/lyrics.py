"""Lyric parsing (LRC / plain text / JSON) and timing (authored or auto-derived)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from .audio import Analysis, snap

TS = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
META = re.compile(r"^\[(ti|ar|al|by|au|offset|re|ve|length|total):(.*)\]$", re.I)
BRACKET = re.compile(r"^[\[\(\{<](.*?)[\]\)\}>]$")
SECTION_WORDS = {
    "intro", "verse", "pre-chorus", "prechorus", "chorus", "hook", "bridge", "outro",
    "refrain", "drop", "interlude", "solo", "instrumental", "break", "ad-lib", "post-chorus", "tag",
}


@dataclass
class Line:
    text: str
    start: float
    end: float
    words: list[str]
    word_times: list[float]
    section: str = ""


@dataclass
class Song:
    lines: list[Line]
    duration: float
    title: str = ""
    artist: str = ""
    timed_by_user: bool = False


def parse(text: str) -> tuple[dict, list[tuple[float | None, str]]]:
    meta: dict[str, str] = {}
    timed: list[tuple[float | None, str]] = []
    for raw in text.splitlines():
        raw = raw.rstrip()
        if not raw.strip():
            timed.append((None, ""))
            continue
        m = META.match(raw.strip())
        if m:
            meta[m.group(1).lower()] = m.group(2).strip()
            continue
        stamps = [(int(a) * 60 + int(b) + float("0." + (c or "0"))) for a, b, c in TS.findall(raw)]
        body = TS.sub("", raw).strip()
        if stamps:
            timed.extend((s, body) for s in stamps)
        else:
            timed.append((None, body))
    return meta, timed


def load_lyrics(path_or_text: str) -> tuple[dict, list[tuple[float | None, str]]]:
    if path_or_text.strip().startswith("{"):
        data = json.loads(path_or_text)
        meta = {k: data[k] for k in ("title", "artist") if k in data}
        lines = [(i.get("start"), i.get("text", "")) if isinstance(i, dict) else (None, i)
                 for i in data.get("lines", [])]
        return meta, lines
    try:
        with open(path_or_text, "r", encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        content = path_or_text
    return parse(content)


def split_sections(lines: list[tuple[float | None, str]]) -> list[tuple[float | None, str, str, bool]]:
    out = []
    current = ""
    for t, txt in lines:
        m = BRACKET.match(txt.strip())
        first = m.group(1).strip().lower().split()[0].rstrip(":0123456789") if (m and m.group(1).strip()) else ""
        if m and first in SECTION_WORDS:
            current = m.group(1).strip()
            out.append((t, m.group(1).strip(), current, True))
        elif not txt.strip():
            out.append((t, "", current, True))
        else:
            out.append((t, txt, current, False))
    return out


def _words(txt: str) -> list[str]:
    return txt.split()


def word_times(start: float, end: float, words: list[str], an: Analysis) -> list[float]:
    if not words:
        return []
    dur = max(end - start, 0.25)
    weights = np.array([max(len(w), 2) ** 0.85 for w in words], dtype=float)
    weights /= weights.sum()
    raw = start + np.concatenate([[0.0], np.cumsum(weights)[:-1]]) * dur
    if an.onset_times.size:
        win = an.onset_times[(an.onset_times >= start - 0.05) & (an.onset_times <= end)]
        if win.size >= len(words):
            snapped = [snap(float(t), win, min(0.14, dur / (len(words) * 2.2))) for t in raw]
            mono: list[float] = []
            for t in snapped:
                if mono and t <= mono[-1] + 0.06:
                    t = mono[-1] + max(0.07, dur / (len(words) * 3))
                mono.append(float(t))
            raw = np.array(mono)
    return [float(min(t, end - 0.05)) for t in raw]


def auto_time(lines, an: Analysis, window=None, lead_in: float = 2.0, tail: float = 2.5) -> list[Line]:
    real = [(t, txt, sec) for (t, txt, sec, meta) in lines if txt.strip() and not meta]
    if not real:
        return []
    t0 = (window[0] if window else 0.0) + lead_in
    t1 = max((window[1] if window else an.duration) - tail, t0 + len(real) * 0.9)
    bars = an.bar_times[(an.bar_times >= t0 - 0.01) & (an.bar_times <= t1)]
    beats = an.beat_times[(an.beat_times >= t0 - 0.01) & (an.beat_times <= t1)]
    n = len(real)
    if bars.size >= n:
        step = max(1, int(np.floor(bars.size / n)))
        anchors = [float(bars[min(i * step, bars.size - 1)]) for i in range(n)]
    elif bars.size >= 2:
        anchors = [float(bars[i]) for i in np.linspace(0, bars.size - 1, n).round().astype(int)]
    elif beats.size >= n:
        step = max(1, int(np.floor(beats.size / n)))
        anchors = [float(beats[min(i * step, beats.size - 1)]) for i in range(n)]
    elif beats.size >= 2:
        anchors = [float(beats[i]) for i in np.linspace(0, beats.size - 1, n).round().astype(int)]
    else:
        anchors = list(np.linspace(t0, t1, n + 1)[:-1])
    min_hold = 1.1 if an.tempo >= 110 else 1.4
    fixed: list[float] = []
    for a in anchors:
        if fixed and a < fixed[-1] + min_hold:
            a = fixed[-1] + min_hold
        fixed.append(float(min(a, t1)))
    out: list[Line] = []
    for i, ((_, txt, sec), start) in enumerate(zip(real, fixed)):
        end = fixed[i + 1] if i + 1 < len(fixed) else min(t1 + 0.8, start + max(min_hold, 3.2))
        end = max(end, start + min_hold)
        words = _words(txt)
        out.append(Line(txt, start, end, words, word_times(start, end, words, an), section=sec))
    return out


def from_analysis(meta: dict, lines, an: Analysis, window=None,
                  lead_in: float = 2.0, tail: float = 2.5) -> Song:
    tagged = split_sections(lines)
    title = meta.get("ti", meta.get("title", ""))
    artist = meta.get("ar", meta.get("artist", ""))
    if any(t is not None for t, _ in lines):
        entries = [(t, txt, sec) for (t, txt, sec, mf) in tagged if txt.strip() and not mf]
        entries.sort(key=lambda e: e[0] if e[0] is not None else 1e9)
        off = float(meta.get("offset", "0") or 0) / 1000.0
        timed: list[Line] = []
        for i, (t, txt, sec) in enumerate(entries):
            st = max(0.0, float(t) + off)
            nxt = entries[i + 1][0] if i + 1 < len(entries) else None
            en = max(st + 0.6, float(nxt) + off - 0.06) if nxt is not None else \
                st + min(4.0, max(1.6, an.duration - st - 0.5))
            words = _words(txt)
            timed.append(Line(txt, st, en, words, word_times(st, en, words, an), section=sec))
        if an.onset_times.size:
            for ln in timed:
                ln.start = snap(ln.start, an.onset_times, 0.06)
                ln.word_times = word_times(ln.start, ln.end, ln.words, an)
        return Song(lines=timed, duration=an.duration, title=title, artist=artist, timed_by_user=True)
    timed = auto_time(tagged, an, window=window, lead_in=lead_in, tail=tail)
    return Song(lines=timed, duration=an.duration, title=title, artist=artist, timed_by_user=False)
