#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from common import (
    DEFAULT_AUDIO_BITRATE,
    DEFAULT_AUDIO_FADE_MS,
    DEFAULT_VIDEO_CRF,
    DEFAULT_VIDEO_PRESET,
    ROOT_DIR,
    add_backend_args,
    build_llm_prompt,
    build_timeline_template,
    ensure_word_segments,
    extract_audio_mp3,
    match_audio_duration,
    media_duration,
    print_progress,
    submit_transcript_job,
    transcript_to_text,
    transcript_words_to_text,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: tight VAD pass, transcript generation, and LLM timeline prompt generation."
    )
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for stage1 outputs. Defaults to <input_stem>_llm_timeline next to the input.",
    )
    parser.add_argument("--stage1-name", default="stage1_vad", help="Basename for the VAD-only video output.")
    parser.add_argument("--pad-before-ms", type=int, default=60, help="VAD padding before speech for stage1.")
    parser.add_argument("--pad-after-ms", type=int, default=90, help="VAD padding after speech for stage1.")
    parser.add_argument("--merge-gap-ms", type=int, default=120, help="Merge adjacent VAD speech regions closer than this.")
    parser.add_argument("--audio-fade-ms", type=int, default=DEFAULT_AUDIO_FADE_MS, help="Fade applied per VAD chunk.")
    parser.add_argument("--video-preset", default=DEFAULT_VIDEO_PRESET)
    parser.add_argument("--video-crf", type=int, default=DEFAULT_VIDEO_CRF)
    parser.add_argument("--audio-bitrate", default=DEFAULT_AUDIO_BITRATE)
    add_backend_args(parser)
    return parser.parse_args()


def resolve_work_dir(args: argparse.Namespace) -> Path:
    if args.work_dir is not None:
        return args.work_dir.resolve()
    return args.input.resolve().parent / f"{args.input.stem}_llm_timeline"


def build_stage1_vad_command(args: argparse.Namespace, stage1_video: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT_DIR / "vad_pause_editor.py"),
        "--input",
        str(args.input.resolve()),
        "--output",
        str(stage1_video),
        "--pad-before-ms",
        str(args.pad_before_ms),
        "--pad-after-ms",
        str(args.pad_after_ms),
        "--merge-gap-ms",
        str(args.merge_gap_ms),
        "--leading-keep-ms",
        "0",
        "--trailing-keep-ms",
        "0",
        "--preserve-short-pause-ms",
        "0",
        "--long-pause-step-ms",
        "0",
        "--long-pause-step-every-ms",
        "1000",
        "--max-keep-silence-ms",
        "0",
        "--audio-fade-ms",
        str(args.audio_fade_ms),
        "--video-preset",
        args.video_preset,
        "--video-crf",
        str(args.video_crf),
        "--audio-bitrate",
        args.audio_bitrate,
    ]


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    work_dir = resolve_work_dir(args)
    work_dir.mkdir(parents=True, exist_ok=True)

    stage1_video = work_dir / f"{args.stage1_name}{args.input.suffix}"
    transcript_json = work_dir / "stage1_transcript.json"
    transcript_txt = work_dir / "stage1_transcript.txt"
    transcript_words_txt = work_dir / "stage1_word_timestamps.txt"
    audio_mp3 = work_dir / "stage1_audio_16k.mp3"
    llm_prompt = work_dir / "llm_prompt.md"
    timeline_json = work_dir / "timeline.json"
    timeline_spec = Path(__file__).resolve().with_name("TIMELINE_SPEC.md")
    manifest_json = work_dir / "stage1_manifest.json"

    if stage1_video.exists():
        print_progress(f"Stage 1/3: Reusing existing stage1 VAD output: {stage1_video}")
    else:
        print_progress("Stage 1/3: Running tight VAD-only pass...")
        subprocess.run(build_stage1_vad_command(args, stage1_video), check=True)

    duration_s = media_duration(str(stage1_video))

    print_progress("Stage 2/3: Extracting stage1 audio for transcript...")
    extract_audio_mp3(stage1_video, audio_mp3)
    match_audio_duration(audio_mp3, audio_mp3, target_duration_s=duration_s)

    print_progress("Stage 3/3: Requesting transcript on the VAD output...")
    transcript = submit_transcript_job(
        backend=args.backend,
        api_base_url=args.api_base_url,
        runpod_endpoint_url=args.runpod_endpoint_url,
        runpod_status_base_url=args.runpod_status_base_url,
        audio_path=audio_mp3,
        model_name=args.model_name,
        language_code=args.language_code,
        chunk_length_s=args.chunk_length_s,
        poll_seconds=args.poll_seconds,
    )
    transcript = ensure_word_segments(transcript)
    write_json(transcript_json, transcript)
    transcript_text = transcript_to_text(transcript)
    transcript_words_text = transcript_words_to_text(transcript)
    transcript_txt.write_text(transcript_text, encoding="utf-8")
    transcript_words_txt.write_text(transcript_words_text, encoding="utf-8")

    if timeline_json.exists():
        backup = timeline_json.with_suffix(".json.bak")
        shutil.copy2(timeline_json, backup)
    write_json(timeline_json, build_timeline_template(stage1_video, transcript_json, duration_s))

    prompt_text = build_llm_prompt(
        source_video=stage1_video,
        transcript_json=transcript_json,
        transcript_txt=transcript_txt,
        transcript_words_txt=transcript_words_txt,
        transcript_words_text=transcript_words_text,
        timeline_json=timeline_json,
        timeline_spec=timeline_spec,
        source_duration_s=duration_s,
    )
    llm_prompt.write_text(prompt_text, encoding="utf-8")

    write_json(
        manifest_json,
        {
            "input_video": str(args.input.resolve()),
            "stage1_video": str(stage1_video),
            "stage1_duration_s": round(duration_s, 3),
            "transcript_json": str(transcript_json),
            "transcript_txt": str(transcript_txt),
            "transcript_words_txt": str(transcript_words_txt),
            "llm_prompt": str(llm_prompt),
            "timeline_json": str(timeline_json),
            "timeline_spec": str(timeline_spec),
            "stage1_vad_settings": {
                "pad_before_ms": args.pad_before_ms,
                "pad_after_ms": args.pad_after_ms,
                "merge_gap_ms": args.merge_gap_ms,
                "leading_keep_ms": 0,
                "trailing_keep_ms": 0,
                "preserve_short_pause_ms": 0,
                "long_pause_step_ms": 0,
                "max_keep_silence_ms": 0,
                "audio_fade_ms": args.audio_fade_ms,
            },
        },
    )

    print()
    print_progress("LLM prompt template:")
    print(prompt_text, end="")
    print_progress(f"Stage1 video:      {stage1_video}")
    print_progress(f"Transcript JSON:   {transcript_json}")
    print_progress(f"Transcript text:   {transcript_txt}")
    print_progress(f"Word timestamps:   {transcript_words_txt}")
    print_progress(f"Timeline JSON:     {timeline_json}")
    print_progress(f"Prompt template:   {llm_prompt}")
    print_progress(f"Manifest:          {manifest_json}")
    print_progress("Next: edit the prompt, send transcript/prompt to your LLM, then run stage2_apply_timeline.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
