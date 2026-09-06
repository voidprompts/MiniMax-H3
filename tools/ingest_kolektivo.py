"""One-off ingest: speaker-tagged SRT for 'I Can't Live Without U' -> timed LRC.

Sound-effect cues are dropped, split fragments are re-joined at sentence
boundaries (spill words carry the next sentence's start), and each sentence
becomes one on-screen line with the cue timing of its first/last word.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyricvid.audio import analyze  # noqa: E402

AUDIO = "/home/user/work/input/kolektivo.m4a"
OUT = "/home/user/work/input/lyrics_kolektivo.lrc"

SRT = """
00:00:11,350 --> 00:00:14,010 [Speaker 1]
[singing] Remember those times. Girl, are

00:00:14,080 --> 00:00:14,940 [Speaker 1]
you ready?

00:00:16,560 --> 00:00:19,420 [Speaker 1]
We were sippin' on wine. Now we're going

00:00:19,440 --> 00:00:20,280 [Speaker 1]
steady.

00:00:22,020 --> 00:00:25,820 [Speaker 1]
Our love's so divine. We love so heavy.

00:00:27,360 --> 00:00:30,800 [Speaker 1]
Remember those times. Girl, are you ready?

00:00:32,700 --> 00:00:35,440 [Speaker 1]
You're on my, on my, on my, on my mind. I

00:00:35,520 --> 00:00:38,500 [Speaker 1]
think about you all the damn time. Love

00:00:38,580 --> 00:00:40,660 [Speaker 1]
you little sweet lime mind.

00:00:41,640 --> 00:00:45,990 [Speaker 1]
I can't live without you. Ooh, ooh, ooh,

00:00:46,120 --> 00:00:48,480 [Speaker 1]
ooh. I can't live without

00:00:48,580 --> 00:00:49,120 [Speaker 1]
you.

00:00:50,160 --> 00:00:53,800 [Speaker 1]
Ooh, ooh, ooh, ooh. I can't live without

00:00:53,940 --> 00:00:55,430 [Speaker 1]
you. Ooh,

00:00:56,740 --> 00:00:59,460 [Speaker 1]
ooh, ooh, ooh. I can't live without you.

00:01:00,160 --> 00:01:01,960 [Speaker 1]
Ooh, ooh, ooh, ooh.

00:01:26,030 --> 00:01:28,780 [Speaker 1]
You're on my, on my, on my, on my mind. I

00:01:28,860 --> 00:01:31,820 [Speaker 1]
think about you all the damn time. Love

00:01:31,880 --> 00:01:34,100 [Speaker 1]
you little sweet lime mind.

00:01:34,940 --> 00:01:36,900 [Speaker 1]
I can't live without you.

00:01:37,600 --> 00:01:41,840 [Speaker 1]
Ooh, ooh, ooh, ooh. I can't live without

00:01:41,880 --> 00:01:42,320 [Speaker 1]
you.

00:01:43,040 --> 00:01:47,140 [Speaker 1]
Ooh, ooh, ooh, ooh. I can't live without

00:01:47,260 --> 00:01:47,500 [Speaker 1]
you.

00:01:48,260 --> 00:01:52,600 [Speaker 1]
Ooh, ooh, ooh, ooh. I can't live without

00:01:52,620 --> 00:01:52,780 [Speaker 1]
you.

00:01:53,600 --> 00:01:54,760 [Speaker 1]
Ooh, ooh, ooh,

00:01:55,380 --> 00:01:56,120 [Speaker 1]
ooh.

00:01:56,180 --> 00:01:59,740 [Speaker 2]
[singing] We sow the seeds of love to

00:01:59,800 --> 00:02:03,040 [Speaker 2]
grow. The love received, the love we owe.

00:02:03,400 --> 00:02:05,800 [Speaker 2]
A love so fine, this love's aglow.

00:02:08,640 --> 00:02:11,800 [Speaker 2]
We sow the seeds of love to grow. The love

00:02:11,860 --> 00:02:14,760 [Speaker 2]
received, the love we owe. A love so

00:02:14,860 --> 00:02:16,780 [Speaker 2]
fine, this love's aglow.

00:02:17,140 --> 00:02:20,540 [Speaker 1]
[singing] I can't live without you. Ooh,

00:02:20,900 --> 00:02:21,920 [Speaker 1]
ooh, ooh, ooh.

00:02:22,840 --> 00:02:24,760 [Speaker 1]
I can't live without you.

00:02:26,240 --> 00:02:29,780 [Speaker 1]
Ooh, ooh, ooh, ooh. I can't live without

00:02:29,900 --> 00:02:32,650 [Speaker 1]
you. Ooh, ooh, ooh,

00:02:32,740 --> 00:02:35,520 [Speaker 1]
ooh. I can't live without you.

00:02:36,920 --> 00:02:40,480 [Speaker 1]
Ooh, ooh, ooh, ooh. I can't live without

00:02:40,580 --> 00:02:40,780 [Speaker 1]
you.

00:02:41,580 --> 00:02:43,800 [Speaker 1]
Ooh, ooh, ooh, ooh.

00:02:44,100 --> 00:02:47,269 [Speaker 2]
[singing] We sow the seeds of love to

00:02:47,269 --> 00:02:47,269 [Speaker 2]
grow. The love received, the love we owe.

00:02:47,269 --> 00:02:50,150 [Speaker 2]
A love so fine, this love's aglow.

00:02:50,180 --> 00:02:52,640 [Speaker 1]
[singing] I can't live without you. Ooh,

00:02:52,920 --> 00:02:55,180 [Speaker 1]
ooh, ooh, ooh. Live without you.

00:02:57,480 --> 00:02:59,350 [Speaker 1]
Ooh, ooh, ooh, ooh.
"""

SECTIONS = [
    (11.0, 32.6, "Verse"),
    (32.6, 41.6, "Pre-Chorus"),
    (41.6, 62.0, "Chorus"),
    (85.9, 94.9, "Pre-Chorus"),
    (94.9, 116.1, "Chorus"),
    (116.1, 137.1, "Bridge"),
    (137.1, 164.0, "Chorus"),
    (164.0, 170.1, "Bridge"),
    (170.1, 188.0, "Outro"),
]

CUE_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})[^\n]*\n([^\n]+)")


def ts(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def section_of(t: float) -> str:
    for lo, hi, name in SECTIONS:
        if lo <= t < hi:
            return name
    return "Outro"


def main() -> int:
    cues = []
    for m in CUE_RE.finditer(SRT):
        g = m.groups()
        start, end = ts(*g[0:4]), ts(*g[4:8])
        text = re.sub(r"\[[^\]]*\]", " ", g[8])          # drop [singing] etc.
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            cues.append((start, end, text))

    # word stream: sentences closed on . ? ! ; track per-sentence cue word counts
    sentences: list[list] = []   # [words, [(cue_idx, n_words)...]]
    cur_words: list[str] = []
    cur_cues: list[list] = []
    for ci, (_s, _e, text) in enumerate(cues):
        for w in text.split():
            if not cur_words:
                cur_cues = [[ci, 0]]
            if cur_cues[-1][0] != ci:
                cur_cues.append([ci, 0])
            cur_cues[-1][1] += 1
            cur_words.append(w)
            if w[-1] in ".!?":
                sentences.append([cur_words, cur_cues])
                cur_words, cur_cues = [], []
    if cur_words:
        sentences.append([cur_words, cur_cues])

    # split every cue span across the sentences touching it, by word weight
    bounds: dict[int, list[float]] = {}
    for ci, (cs, ce, _t) in enumerate(cues):
        touching = [(si, n) for si, (_w, cc) in enumerate(sentences) for c, n in cc if c == ci]
        tot = sum(n for _si, n in touching) or 1
        edges, t = [], cs
        for _si, n in touching:
            edges.append(t)
            t += (ce - cs) * n / tot
        edges.append(ce)
        bounds[ci] = edges

    lines: list[tuple[float, float, str]] = []
    for si, (words, cc) in enumerate(sentences):
        pos_f = sum(1 for s2, (_w2, c2) in enumerate(sentences)
                    if s2 < si and any(c == cc[0][0] for c, _n in c2))
        pos_l = sum(1 for s2, (_w2, c2) in enumerate(sentences)
                    if s2 < si and any(c == cc[-1][0] for c, _n in c2))
        start = bounds[cc[0][0]][pos_f]
        end = bounds[cc[-1][0]][pos_l + 1]
        lines.append((start, max(end, start + 0.5), " ".join(words)))

    # merge 1-word strays, enforce monotonic non-overlapping timings
    merged: list[tuple[float, float, str]] = []
    for s, e, t in lines:
        if len(t.split()) == 1 and merged:
            ps, pe, pt = merged[-1]
            merged[-1] = (ps, max(pe, e), pt + " " + t)
            continue
        if merged and s < merged[-1][1]:
            s = merged[-1][1] + 0.04
        if e <= s + 1.1:
            e = s + 1.3
        merged.append((s, e, t))

    an = analyze(AUDIO)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("[ti:I Can't Live Without U]\n[ar:Kolektivo ft. Gary Ales Brook]\n")
        prev = None
        for s, e, t in merged:
            sec = section_of(s)
            if sec != prev:
                f.write(f"[{sec}]\n")
                prev = sec
            f.write(f"[{int(s // 60):02d}:{s % 60:05.2f}]{t}\n")
    print(f"[ingest] {len(merged)} lines -> {OUT}  (bed {an.duration:.1f}s, {an.tempo:.0f} BPM)")
    for s, e, t in merged:
        print(f"  {s:7.2f}-{e:7.2f}  [{section_of(s):<10}] {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
