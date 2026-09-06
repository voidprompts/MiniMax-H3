"""Command line: audio + lyrics -> beat-synced lyric video."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .audio import analyze
from .lyrics import from_analysis, load_lyrics
from .render import PALETTES, RenderConfig, render_video

SIZES = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080),
         "21:9": (2560, 1080), "4:5": (1080, 1350)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render a beat-reactive lyric video.")
    p.add_argument("--audio", required=True)
    p.add_argument("--lyrics", required=True)
    p.add_argument("--out", default="output/lyric_video.mp4")
    p.add_argument("--style", default="kinetic", choices=sorted(PALETTES))
    p.add_argument("--ar", default="16:9", choices=sorted(SIZES))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=0.0)
    p.add_argument("--title", default="")
    p.add_argument("--artist", default="")
    p.add_argument("--lead-in", type=float, default=2.2)
    p.add_argument("--no-bars", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-kicker", action="store_true")
    p.add_argument("--grain", type=float, default=1.0)
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--preset", default="fast", choices=["veryfast", "fast", "medium", "slow"])
    p.add_argument("--dump-timing", default="")
    p.add_argument("--analyze-only", action="store_true")
    a = p.parse_args(argv)

    an = analyze(a.audio)
    print(f"[analyze] {os.path.basename(a.audio)}: {an.duration:.2f}s  tempo≈{an.tempo:.1f} BPM  "
          f"{an.beat_times.size} beats  {an.bar_times.size} bars  {an.onset_times.size} onsets",
          flush=True)
    if a.analyze_only:
        return 0

    meta, lines = load_lyrics(a.lyrics)
    end = min(a.end if a.end > 0 else an.duration, an.duration)
    song = from_analysis(meta, lines, an, window=(min(a.start, an.duration - 1.0), end),
                         lead_in=a.lead_in)
    if not song.lines:
        print("[error] no lyric lines found in --lyrics", file=sys.stderr)
        return 2
    song.lines = [ln for ln in song.lines if ln.start < end and ln.end > a.start]
    print(f"[lyrics] {len(song.lines)} lines | timing: "
          f"{'your timestamps' if song.timed_by_user else 'auto beat-grid'} | window "
          f"{song.lines[0].start:.2f}s -> {song.lines[-1].end:.2f}s", flush=True)

    if a.dump_timing:
        os.makedirs(os.path.dirname(os.path.abspath(a.dump_timing)) or ".", exist_ok=True)
        with open(a.dump_timing, "w", encoding="utf-8") as f:
            json.dump({"title": song.title or a.title, "artist": song.artist or a.artist,
                       "tempo": an.tempo, "duration": an.duration,
                       "lines": [{"start": round(l.start, 3), "end": round(l.end, 3),
                                  "section": l.section, "text": l.text} for l in song.lines]},
                      f, indent=2, ensure_ascii=False)

    W, H = SIZES[a.ar]
    W, H = (W // 2) * 2, (H // 2) * 2
    base_font = int(132 * (H / 1080 if H >= W else W / 1080))
    cfg = RenderConfig(W=W, H=H, fps=a.fps, style=a.style, title=a.title or song.title,
                       artist=a.artist or song.artist, bars_visible=not a.no_bars,
                       grain_scale=a.grain, lead_in=max(0.0, a.lead_in),
                       show_progress=not a.no_progress, show_kicker=not a.no_kicker,
                       max_font=base_font, min_font=int(base_font * 0.35))
    render_video(an, song, a.out, cfg, audio_src=a.audio, t_start=a.start, t_end=end,
                 crf=a.crf, preset=a.preset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
