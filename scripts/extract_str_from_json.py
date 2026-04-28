#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, remainder_ms = divmod(total_ms, 3600 * 1000)
    minutes, remainder_ms = divmod(remainder_ms, 60 * 1000)
    secs, millis = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def build_srt(segments: list[dict]) -> str:
    blocks: list[str] = []
    index = 1

    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        text = normalize_text(str(segment.get("text", "")))

        if start is None or end is None or not text:
            continue

        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(float(start))} --> {format_srt_timestamp(float(end))}",
                    text,
                ]
            )
        )
        index += 1

    return "\n\n".join(blocks) + ("\n" if blocks else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert transcript JSON segments to SRT.")
    parser.add_argument("input", type=Path, help="Transcript JSON input file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Transcript JSON does not contain a segments list")

    output_path = args.input.with_suffix(".txt")
    srt_text = build_srt(segments)
    output_path.write_text(srt_text, encoding="utf-8")

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
