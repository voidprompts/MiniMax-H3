"""One-off ingest: numbered SRT for Tickwanya 'Turning' -> timed LRC.

Sections are inferred from the hook: 'turning me on' / 'on on on' = Chorus,
'like that' block = Bridge, opener = Intro, everything else Verse.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyricvid.audio import analyze  # noqa: E402

AUDIO = "/home/user/work/input/tickwanya_turning.mp3"
OUT = "/home/user/work/input/lyrics_tickwanya.lrc"

SRT = """
1
00:00:02,160 --> 00:00:07,940
oh I hey yeah

2
00:00:10,196 --> 00:00:12,383
baby put your phone down

3
00:00:12,836 --> 00:00:17,023
come over here and talk to me

4
00:00:17,516 --> 00:00:21,703
hey yeah tell me what you want now

5
00:00:22,156 --> 00:00:26,223
just let me be your fantasy

6
00:00:27,868 --> 00:00:30,586
sitting here looking at you

7
00:00:30,586 --> 00:00:32,740
things that I wanna do

8
00:00:32,740 --> 00:00:36,212
you got me all in the mood oh

9
00:00:36,888 --> 00:00:38,852
cause baby you know

10
00:00:39,398 --> 00:00:41,730
when you start looking at me like that

11
00:00:41,730 --> 00:00:43,421
you're turning me on

12
00:00:44,078 --> 00:00:45,300
when you do that you

13
00:00:45,300 --> 00:00:46,438
drive me crazy

14
00:00:46,438 --> 00:00:48,061
baby you know

15
00:00:48,718 --> 00:00:51,110
when you start looking at me like that

16
00:00:51,110 --> 00:01:00,601
turning me on yeah you're turning me on oh on on on

17
00:01:02,544 --> 00:01:04,401
you're turning me on

18
00:01:05,944 --> 00:01:08,281
Tonight I Wanna let go

19
00:01:08,778 --> 00:01:12,921
and only you can set me free

20
00:01:13,498 --> 00:01:14,601
oh

21
00:01:15,498 --> 00:01:17,255
can we keep it simple

22
00:01:18,400 --> 00:01:23,173
just tell me everything you need yeah

23
00:01:23,746 --> 00:01:26,475
sitting here looking at you

24
00:01:26,475 --> 00:01:28,660
things that I wanna do

25
00:01:28,660 --> 00:01:32,173
you got me all in the mood yeah

26
00:01:32,840 --> 00:01:34,180
'cause baby you know

27
00:01:34,180 --> 00:01:37,660
baby you know when you start looking at me like that

28
00:01:37,660 --> 00:01:41,220
you turn me on turn me on when you do that you

29
00:01:41,220 --> 00:01:42,358
drive me crazy

30
00:01:42,358 --> 00:01:43,981
baby you know

31
00:01:44,660 --> 00:01:47,030
when you start looking at me like that

32
00:01:47,030 --> 00:01:50,960
turning me on yeah you're turning me on

33
00:01:54,020 --> 00:01:57,400
on on on

34
00:01:58,495 --> 00:02:00,280
you're turning me on

35
00:02:01,561 --> 00:02:03,921
boy I know you like that like that

36
00:02:03,921 --> 00:02:06,241
tell me that you like that yeah yeah

37
00:02:06,241 --> 00:02:08,561
boy I know you like that like that

38
00:02:08,561 --> 00:02:11,978
tell me that you like it baby cause baby you know

39
00:02:12,601 --> 00:02:14,940
when you start looking at me like that

40
00:02:14,940 --> 00:02:16,618
you turn me on

41
00:02:17,281 --> 00:02:18,500
when you do that you

42
00:02:18,500 --> 00:02:19,641
drive me crazy

43
00:02:19,641 --> 00:02:22,040
baby you know baby you know

44
00:02:22,040 --> 00:02:24,200
when you start looking at me like that

45
00:02:24,200 --> 00:02:25,820
you're turning me on

46
00:02:26,360 --> 00:02:28,180
yeah you're turning me on

47
00:02:29,080 --> 00:02:34,620
on on on

48
00:02:35,820 --> 00:02:37,500
you're turning me on

49
00:02:39,250 --> 00:02:41,050
are you turning me on

50
00:02:41,530 --> 00:02:43,330
are you turning me on
"""

CUE_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*\n([^\n]+)")


def ts(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def section_of(text: str, first: bool) -> str:
    t = text.lower()
    if first:
        return "Intro"
    if "like that" in t and "looking at me" not in t:
        return "Bridge"
    if "turning me on" in t or t.strip() in ("on on on",):
        return "Chorus"
    return "Verse"


def main() -> int:
    lines = []
    for i, m in enumerate(CUE_RE.finditer(SRT)):
        g = m.groups()
        lines.append((ts(*g[0:4]), ts(*g[4:8]), re.sub(r"\s+", " ", g[8]).strip(), i == 0))
    an = analyze(AUDIO)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("[ti:Turning]\n[ar:Tickwanya]\n")
        prev = None
        for s, e, t, first in lines:
            sec = section_of(t, first)
            if sec != prev:
                f.write(f"[{sec}]\n")
                prev = sec
            f.write(f"[{int(s // 60):02d}:{s % 60:05.2f}]{t}\n")
    print(f"[ingest] {len(lines)} lines -> {OUT}  (bed {an.duration:.1f}s, {an.tempo:.0f} BPM, "
          f"{an.onset_times.size} onsets)")
    for s, e, t, first in lines[:8]:
        print(f"  {s:7.2f}-{e:7.2f}  [{section_of(t, first):<6}] {t}")
    print("  ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
