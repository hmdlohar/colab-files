#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    ensure_word_segments,
    parse_timeline_file,
    transcript_to_text,
    transcript_words_to_text,
    validate_timeline,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a rebased transcript for the final stage2 cut using stage1_transcript.json and timeline.json."
    )
    parser.add_argument("transcript", type=Path, help="stage1_transcript.json")
    parser.add_argument("timeline", type=Path, help="timeline.json keep ranges")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <timeline_dir>/final_transcript.json",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Output text path. Defaults to <timeline_dir>/final_transcript.txt",
    )
    parser.add_argument(
        "--output-words-txt",
        type=Path,
        default=None,
        help="Output word timestamp text path. Defaults to <timeline_dir>/final_word_timestamps.txt",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def build_final_transcript(stage1_transcript: dict[str, Any], timeline: dict[str, Any]) -> dict[str, Any]:
    source_duration_s = float(timeline.get("source_duration_s") or 0.0)
    keep_segments = validate_timeline(timeline, duration_s=source_duration_s)
    stage1_transcript = ensure_word_segments(stage1_transcript)
    words = stage1_transcript.get("word_segments", [])

    final_segments: list[dict[str, Any]] = []
    final_words: list[dict[str, Any]] = []
    output_cursor = 0.0

    for keep_index, keep in enumerate(keep_segments, start=1):
        kept_words: list[dict[str, Any]] = []
        for word in words:
            start = word.get("start")
            end = word.get("end")
            token = " ".join(str(word.get("word", "")).split())
            if start is None or end is None or not token:
                continue
            start_s = float(start)
            end_s = float(end)
            if start_s < keep.start - 1e-6 or end_s > keep.end + 1e-6:
                continue

            rebased_word = {
                "word": token,
                "start": round(output_cursor + (start_s - keep.start), 3),
                "end": round(output_cursor + (end_s - keep.start), 3),
                "source_start": round(start_s, 3),
                "source_end": round(end_s, 3),
            }
            kept_words.append(rebased_word)
            final_words.append(rebased_word)

        if not kept_words:
            output_cursor += keep.duration
            continue

        final_segments.append(
            {
                "index": keep_index,
                "start": round(output_cursor, 3),
                "end": round(output_cursor + keep.duration, 3),
                "text": " ".join(word["word"] for word in kept_words),
                "source_start": round(keep.start, 3),
                "source_end": round(keep.end, 3),
                "word_count": len(kept_words),
            }
        )
        output_cursor += keep.duration

    return {
        "version": 1,
        "timebase": "final_video_seconds",
        "source_timebase": "stage1_video_seconds",
        "source_transcript": timeline.get("source_transcript"),
        "source_video": timeline.get("source_video"),
        "source_duration_s": round(source_duration_s, 3),
        "duration_s": round(output_cursor, 3),
        "strategy": "keep_ranges_rebased",
        "segment_count": len(final_segments),
        "word_count": len(final_words),
        "segments": final_segments,
        "word_segments": final_words,
    }


def main() -> int:
    args = parse_args()
    if not args.transcript.exists():
        raise FileNotFoundError(f"Transcript not found: {args.transcript}")
    if not args.timeline.exists():
        raise FileNotFoundError(f"Timeline not found: {args.timeline}")

    output_dir = args.timeline.resolve().parent
    output_json = args.output_json.resolve() if args.output_json else output_dir / "final_transcript.json"
    output_txt = args.output_txt.resolve() if args.output_txt else output_dir / "final_transcript.txt"
    output_words_txt = (
        args.output_words_txt.resolve() if args.output_words_txt else output_dir / "final_word_timestamps.txt"
    )

    stage1_transcript = load_json(args.transcript.resolve())
    timeline = parse_timeline_file(args.timeline.resolve())
    final_transcript = build_final_transcript(stage1_transcript, timeline)

    write_json(output_json, final_transcript)
    output_txt.write_text(transcript_to_text(final_transcript), encoding="utf-8")
    output_words_txt.write_text(transcript_words_to_text(final_transcript), encoding="utf-8")

    print(f"Transcript JSON: {output_json}")
    print(f"Transcript text: {output_txt}")
    print(f"Word timestamps: {output_words_txt}")
    print(f"Duration: {final_transcript['duration_s']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
