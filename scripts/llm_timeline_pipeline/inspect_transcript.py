#!/usr/bin/env python3
"""Readable view of a stage1 workdir for drafting timeline.json.

  inspect_transcript.py <workdir>                   pause-grouped lines + warnings
  inspect_transcript.py <workdir> --words 440-450   exact word times in ranges
  inspect_transcript.py <workdir> --check           validate timeline.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def load_words(workdir: Path) -> list[dict]:
    data = json.loads((workdir / "stage1_transcript.json").read_text(encoding="utf-8"))
    words = data.get("word_segments") or [w for s in data.get("segments", []) for w in s.get("words", [])]
    return [w for w in words if "start" in w and "end" in w]


def group_lines(words: list[dict], pause_s: float = 0.5, max_words: int = 22) -> list[tuple[float, float, str]]:
    lines, cur = [], []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > pause_s or len(cur) >= max_words):
            lines.append((cur[0]["start"], cur[-1]["end"], " ".join(x["word"] for x in cur)))
            cur = []
        cur.append(w)
    if cur:
        lines.append((cur[0]["start"], cur[-1]["end"], " ".join(x["word"] for x in cur)))
    return lines


def find_loops(words: list[dict], n: int = 4, min_repeats: int = 3) -> list[tuple[float, float]]:
    """Whisper hallucination loops: the same n-gram repeated many times. Word timings inside are fake."""
    hits = defaultdict(list)
    for i in range(len(words) - n + 1):
        hits[tuple(w["word"] for w in words[i:i + n])].append(i)
    spans = []
    for idx in hits.values():
        if len(idx) >= min_repeats and idx[-1] - idx[0] <= len(idx) * n * 5:
            spans.append((words[idx[0]]["start"], words[idx[-1] + n - 1]["end"]))
    spans.sort()
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def find_gaps(words: list[dict], duration: float, min_gap_s: float = 3.0) -> list[tuple[float, float]]:
    edges = [0.0] + [x for w in words for x in (w["start"], w["end"])] + [duration]
    return [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2) if edges[i + 1] - edges[i] >= min_gap_s]


def mean_volume_db(audio: Path, start: float, end: float) -> float | None:
    try:
        out = subprocess.run(
            ["ffmpeg", "-nostats", "-hide_banner", "-ss", f"{start}", "-t", f"{end - start}", "-i", str(audio),
             "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True, timeout=60).stderr
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"mean_volume: (-?[\d.]+) dB", out)
    return float(m.group(1)) if m else None


def check_timeline(workdir: Path) -> int:
    t = json.loads((workdir / "timeline.json").read_text(encoding="utf-8"))
    dur = float(t.get("source_duration_s") or 1e12)
    segs = t.get("segments") or []
    errors = [] if t.get("strategy") == "keep_ranges" else ["strategy must be keep_ranges"]
    if not segs:
        errors.append("segments is empty")
    prev = 0.0
    for i, s in enumerate(segs):
        if not (prev <= s["start"] < s["end"] <= dur):
            errors.append(f"segment {i} {s['start']}-{s['end']} out of order/bounds (prev end {prev})")
        prev = s["end"]
    for e in errors:
        print("ERROR:", e)
    if not errors:
        kept = sum(s["end"] - s["start"] for s in segs)
        print(f"OK: {len(segs)} segments, kept {kept:.1f}s of {dur:.1f}s ({kept / 60:.1f} min)")
    return 1 if errors else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workdir", type=Path)
    p.add_argument("--words", nargs="+", metavar="A-B", help="print word timings for these second ranges")
    p.add_argument("--check", action="store_true", help="validate timeline.json")
    a = p.parse_args()

    if a.check:
        return check_timeline(a.workdir)
    words = load_words(a.workdir)
    if a.words:
        for r in a.words:
            lo, hi = map(float, r.split("-"))
            print(f"--{r}: " + " ".join(f"{w['word']}[{w['start']:.3f}-{w['end']:.3f}]" for w in words if lo <= w["start"] < hi))
        return 0

    manifest = a.workdir / "stage1_manifest.json"
    duration = words[-1]["end"]
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        duration = float(m.get("stage1_duration_s") or duration)
    for s, e, text in group_lines(words):
        print(f"{s:8.3f}-{e:8.3f} {text}")

    audio = next(iter(sorted(a.workdir.glob("stage1_audio*"))), None)
    print("\n# WARNINGS (word timings unreliable / speech missing from transcript)")
    for s, e in find_loops(words):
        print(f"LOOP  {s:8.3f}-{e:8.3f}  Whisper repetition loop; real speech content unknown")
    for s, e in find_gaps(words, duration):
        vol = mean_volume_db(audio, s, e) if audio else None
        speech = "speech likely" if vol is not None and vol > -35 else "probably silence"
        print(f"GAP   {s:8.3f}-{e:8.3f}  no words; mean {vol} dB -> {speech}")
    return 0


def _selftest() -> None:
    w = lambda t, s: {"word": t, "start": s, "end": s + 0.2}
    words = [w("a", 0), w("b", 0.3), w("c", 2)] + [w(x, 3 + i * 0.25) for i, x in enumerate(("x y z q " * 5).split())]
    assert [l[2] for l in group_lines(words)][:2] == ["a b", "c"]
    assert find_loops(words) and find_loops(words)[0][0] == 3
    assert find_loops(words[:3]) == []
    assert find_gaps([w("a", 1), w("b", 10)], 12) == [(1.2, 10)]


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        _selftest()
        print("selftest ok")
        sys.exit(0)
    sys.exit(main())
