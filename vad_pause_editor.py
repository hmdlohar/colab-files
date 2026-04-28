#!/usr/bin/env python3

"""
VAD-first pause editor for audio or video.

Pipeline:
1. Extract raw mono audio for VAD
2. Detect speech with Silero VAD
3. Add padding and merge close speech regions
4. Shorten only the non-speech gaps
5. Render edited audio/video
6. Optionally denoise the edited result

Examples:
  python vad_pause_editor.py \
      --input input.mp4 \
      --output output.mp4

  python vad_pause_editor.py \
      --input input.wav \
      --output output.wav \
      --denoise after

python vad_pause_editor.py     --input shrink.mkv     --output shrink_tight.mkv     --pad-before-ms 150     --pad-after-ms 200     --merge-gap-ms 140     --preserve-short-pause-ms 160     --long-pause-step-ms 90     --long-pause-step-every-ms 800     --max-keep-silence-ms 220     --transcript transcript.json     --filler-words "अः,हूं,मतलब,तो,ठीक,अब,ना,वो,अरे"     --filler-pad-before-ms 25     --filler-pad-after-ms 40     --audio-fade-ms 50     --video-preset fast     --video-crf 23     --audio-bitrate 128k

Dependencies:
  pip install soundfile numpy
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu



Optional:
  pip install denoiser

External:
  ffmpeg and ffprobe must be installed and in PATH.

Notes:
  - The first Silero VAD run downloads the model via torch.hub.
  - By default VAD runs on raw audio. Denoising happens after editing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

VAD_SAMPLE_RATE = 16000


@dataclass
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def run(cmd: Sequence[str]) -> None:
    subprocess.run(cmd, check=True)


def run_capture(cmd: Sequence[str]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def run_quiet(cmd: Sequence[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return

    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    raise subprocess.CalledProcessError(
        result.returncode,
        cmd,
        output=result.stdout,
        stderr=result.stderr,
    )


def ffmpeg_cmd(*args: str) -> List[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", *args]


def print_progress(message: str) -> None:
    print(message, flush=True)


def ffprobe_streams(path: str) -> Tuple[bool, bool]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        path,
    ]
    data = json.loads(run_capture(cmd))
    stream_types = {stream["codec_type"] for stream in data.get("streams", [])}
    return "video" in stream_types, "audio" in stream_types


def media_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    return float(run_capture(cmd).strip())


def extract_audio_for_vad(input_path: str, wav_path: str) -> None:
    cmd = ffmpeg_cmd(
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(VAD_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        wav_path,
    )
    run_quiet(cmd)


def load_silero_vad():
    import torch

    with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stderr(devnull):
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def detect_speech_segments(vad_wav_path: str, threshold: float) -> List[Segment]:
    import numpy as np
    import soundfile as sf
    import torch

    audio, sr = sf.read(vad_wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != VAD_SAMPLE_RATE:
        raise ValueError(f"Expected {VAD_SAMPLE_RATE} Hz audio, got {sr}")

    model, get_speech_timestamps = load_silero_vad()
    with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stderr(devnull):
        timestamps = get_speech_timestamps(
            torch.from_numpy(audio),
            model,
            sampling_rate=VAD_SAMPLE_RATE,
            threshold=threshold,
        )

    return [
        Segment(start=item["start"] / VAD_SAMPLE_RATE, end=item["end"] / VAD_SAMPLE_RATE)
        for item in timestamps
    ]


def expand_and_merge_segments(
    segments: Sequence[Segment],
    total_duration: float,
    pad_before_ms: int,
    pad_after_ms: int,
    merge_gap_ms: int,
) -> List[Segment]:
    if not segments:
        return []

    pad_before = pad_before_ms / 1000.0
    pad_after = pad_after_ms / 1000.0
    merge_gap = merge_gap_ms / 1000.0

    padded = [
        Segment(
            start=max(0.0, segment.start - pad_before),
            end=min(total_duration, segment.end + pad_after),
        )
        for segment in segments
    ]

    merged: List[Segment] = [padded[0]]
    for segment in padded[1:]:
        last = merged[-1]
        if segment.start - last.end <= merge_gap:
            last.end = max(last.end, segment.end)
        else:
            merged.append(segment)
    return merged


def shorten_gap_duration(
    gap_seconds: float,
    preserve_short_pause_ms: int,
    long_pause_step_ms: int,
    long_pause_step_every_ms: int,
    max_keep_silence_ms: int,
) -> float:
    gap_ms = int(round(gap_seconds * 1000))
    if gap_ms <= preserve_short_pause_ms:
        return gap_seconds

    extra_ms = max(0, gap_ms - preserve_short_pause_ms)
    step_count = extra_ms // long_pause_step_every_ms
    keep_ms = preserve_short_pause_ms + (step_count + 1) * long_pause_step_ms
    keep_ms = min(gap_ms, max_keep_silence_ms, keep_ms)
    return keep_ms / 1000.0


def build_output_segments(
    speech_segments: Sequence[Segment],
    total_duration: float,
    leading_keep_ms: int,
    trailing_keep_ms: int,
    preserve_short_pause_ms: int,
    long_pause_step_ms: int,
    long_pause_step_every_ms: int,
    max_keep_silence_ms: int,
) -> List[Segment]:
    if not speech_segments:
        return [Segment(0.0, total_duration)]

    kept: List[Segment] = []

    first = speech_segments[0]
    lead_keep = min(first.start, leading_keep_ms / 1000.0)
    if lead_keep > 0:
        kept.append(Segment(first.start - lead_keep, first.start))

    for index, current in enumerate(speech_segments):
        kept.append(Segment(current.start, current.end))

        if index == len(speech_segments) - 1:
            continue

        nxt = speech_segments[index + 1]
        gap = max(0.0, nxt.start - current.end)
        keep_gap = shorten_gap_duration(
            gap,
            preserve_short_pause_ms=preserve_short_pause_ms,
            long_pause_step_ms=long_pause_step_ms,
            long_pause_step_every_ms=long_pause_step_every_ms,
            max_keep_silence_ms=max_keep_silence_ms,
        )
        if keep_gap <= 0:
            continue

        gap_start = current.end + max(0.0, (gap - keep_gap) / 2.0)
        kept.append(Segment(gap_start, gap_start + keep_gap))

    last = speech_segments[-1]
    trail_keep = min(total_duration - last.end, trailing_keep_ms / 1000.0)
    if trail_keep > 0:
        kept.append(Segment(last.end, last.end + trail_keep))

    return merge_touching_segments(kept)


def merge_touching_segments(segments: Iterable[Segment], epsilon: float = 1e-4) -> List[Segment]:
    items = sorted((Segment(s.start, s.end) for s in segments), key=lambda item: item.start)
    if not items:
        return []

    merged = [items[0]]
    for segment in items[1:]:
        last = merged[-1]
        if segment.start <= last.end + epsilon:
            last.end = max(last.end, segment.end)
        else:
            merged.append(segment)
    return merged


def normalize_token(token: str) -> str:
    token = token.strip().lower()
    token = re.sub(r"^[^\w\u0900-\u097F]+|[^\w\u0900-\u097F]+$", "", token)
    return token


def load_filler_segments(
    transcript_path: str,
    filler_words: Sequence[str],
    filler_pad_before_ms: int,
    filler_pad_after_ms: int,
) -> List[Segment]:
    with open(transcript_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    wanted = {normalize_token(word) for word in filler_words if normalize_token(word)}
    if not wanted:
        return []

    filler_segments: List[Segment] = []
    for item in data.get("word_segments", []):
        raw_word = item.get("word", "")
        normalized = normalize_token(raw_word)
        if normalized not in wanted:
            continue

        start = item.get("start")
        end = item.get("end")
        if start is None or end is None or end <= start:
            continue

        filler_segments.append(
            Segment(
                start=max(0.0, float(start) - filler_pad_before_ms / 1000.0),
                end=float(end) + filler_pad_after_ms / 1000.0,
            )
        )

    return merge_touching_segments(filler_segments)


def subtract_cut_segments(keep_segments: Sequence[Segment], cut_segments: Sequence[Segment]) -> List[Segment]:
    if not cut_segments:
        return list(keep_segments)

    remaining: List[Segment] = []
    cuts = merge_touching_segments(cut_segments)

    for keep in keep_segments:
        current_start = keep.start
        for cut in cuts:
            if cut.end <= current_start:
                continue
            if cut.start >= keep.end:
                break

            if cut.start > current_start:
                remaining.append(Segment(current_start, min(cut.start, keep.end)))
            current_start = max(current_start, cut.end)
            if current_start >= keep.end:
                break

        if current_start < keep.end:
            remaining.append(Segment(current_start, keep.end))

    return merge_touching_segments(
        [segment for segment in remaining if segment.duration > 0.02]
    )


def concat_media_files(input_files: Sequence[str], output_path: str) -> None:
    if not input_files:
        raise ValueError("No rendered chunk files to concatenate")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".concat.txt",
        prefix="vad_pause_editor_",
        delete=False,
        encoding="utf-8",
    ) as concat_file:
        for path in input_files:
            concat_file.write(f"file '{path}'\n")
        concat_path = concat_file.name

    try:
        cmd = ffmpeg_cmd(
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path,
            "-c",
            "copy",
            output_path,
        )
        run_quiet(cmd)
    finally:
        try:
            os.unlink(concat_path)
        except FileNotFoundError:
            pass


def finalize_rendered_output(
    intermediate_path: str,
    output_path: str,
    has_video: bool,
    audio_bitrate: str,
) -> None:
    output_suffix = Path(output_path).suffix.lower()

    if not has_video and output_suffix == ".wav":
        cmd = ffmpeg_cmd("-y", "-i", intermediate_path, "-c:a", "pcm_s16le", output_path)
        run_quiet(cmd)
        return

    cmd = ffmpeg_cmd("-y", "-i", intermediate_path)
    if has_video:
        cmd += ["-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy"]
        if output_suffix in {".mp4", ".m4v", ".mov", ".mkv", ".mka"}:
            cmd += ["-c:a", "aac", "-b:a", audio_bitrate]
        else:
            cmd += ["-c:a", "aac", "-b:a", audio_bitrate]
    else:
        if output_suffix == ".flac":
            cmd += ["-c:a", "flac"]
        elif output_suffix == ".mp3":
            cmd += ["-c:a", "libmp3lame", "-q:a", "2"]
        elif output_suffix in {".ogg", ".opus"}:
            cmd += ["-c:a", "libopus", "-b:a", audio_bitrate]
        elif output_suffix == ".wav":
            shutil.copy2(intermediate_path, output_path)
            return
        else:
            cmd += ["-c:a", "aac", "-b:a", audio_bitrate]

    cmd.append(output_path)
    run_quiet(cmd)


def build_audio_fade_filter(duration: float, audio_fade_ms: int) -> str | None:
    if audio_fade_ms <= 0 or duration <= 0:
        return None

    fade_duration = min(audio_fade_ms / 1000.0, duration / 2.0)
    if fade_duration <= 0:
        return None

    fade_out_start = max(0.0, duration - fade_duration)
    return f"afade=t=in:st=0:d={fade_duration:.6f},afade=t=out:st={fade_out_start:.6f}:d={fade_duration:.6f}"


def render_segments(
    input_path: str,
    output_path: str,
    segments: Sequence[Segment],
    render_chunk_segments: int,
    render_chunk_duration_s: float,
    audio_fade_ms: int,
    video_preset: str,
    video_crf: int,
    audio_bitrate: str,
) -> None:
    if not segments:
        raise ValueError("No output segments to render")

    has_video, has_audio = ffprobe_streams(input_path)
    if not has_video and not has_audio:
        raise ValueError("Input has neither audio nor video streams")

    with tempfile.TemporaryDirectory(prefix="vad_pause_editor_render_") as tmpdir:
        segment_files: List[str] = []
        total = len(segments)
        renderable_segments = [segment for segment in segments if segment.duration > 0.02]
        renderable_total = len(renderable_segments)
        renderable_duration = sum(segment.duration for segment in renderable_segments)
        segment_suffix = ".mkv" if has_video else ".mka"
        intermediate_path = os.path.join(tmpdir, f"concat{segment_suffix}")
        rendered_count = 0
        rendered_duration = 0.0

        print_progress(
            f"Rendering {renderable_total} kept segments "
            f"({renderable_duration:.2f}s total kept duration)..."
        )

        for index, segment in enumerate(segments):
            segment_path = os.path.join(tmpdir, f"segment_{index:05d}{segment_suffix}")
            duration = segment.duration
            if duration <= 0.02:
                continue

            cmd = ffmpeg_cmd(
                "-y",
                "-ss",
                f"{segment.start:.6f}",
                "-i",
                input_path,
                "-t",
                f"{duration:.6f}",
                "-avoid_negative_ts",
                "make_zero",
            )
            if has_video:
                cmd += ["-c:v", "libx264", "-preset", video_preset, "-crf", str(video_crf)]
            else:
                cmd += ["-vn"]

            if has_audio:
                audio_filter = build_audio_fade_filter(duration, audio_fade_ms)
                if audio_filter:
                    cmd += ["-af", audio_filter]
                cmd += ["-c:a", "pcm_s16le"]
            else:
                cmd += ["-an"]

            cmd += [segment_path]
            run_quiet(cmd)
            segment_files.append(segment_path)
            rendered_count += 1
            rendered_duration += duration

            if (
                rendered_count == 1
                or rendered_count == renderable_total
                or rendered_count % 10 == 0
            ):
                percent = (rendered_count / renderable_total * 100.0) if renderable_total else 100.0
                print_progress(
                    f"Rendered segments: {rendered_count}/{renderable_total} "
                    f"({percent:.1f}%) | kept {rendered_duration:.2f}/{renderable_duration:.2f}s"
                )

        print_progress("Concatenating rendered segments...")
        concat_media_files(segment_files, intermediate_path)
        print_progress("Finalizing output container...")
        finalize_rendered_output(
            intermediate_path=intermediate_path,
            output_path=output_path,
            has_video=has_video,
            audio_bitrate=audio_bitrate,
        )


def denoise_wav(input_wav_path: str, output_wav_path: str, device: str) -> None:
    import numpy as np
    import soundfile as sf
    import torch

    try:
        from denoiser import pretrained
        from denoiser.dsp import convert_audio
    except ImportError as exc:
        raise RuntimeError(
            "Denoising requested, but `denoiser` is not installed. "
            "Run `pip install denoiser`."
        ) from exc

    model_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = pretrained.dns64().to(model_device)
    wav, sr = sf.read(input_wav_path, dtype="float32")

    if wav.ndim == 1:
        wav = wav[:, None]
    wav = wav.T

    chunk_size = sr * 10
    denoised_chunks = []

    for start in range(0, wav.shape[1], chunk_size):
        chunk = wav[:, start:start + chunk_size]
        wav_tensor = torch.from_numpy(chunk).to(model_device)
        wav_tensor = convert_audio(wav_tensor, sr, model.sample_rate, model.chin)
        with torch.no_grad():
            denoised = model(wav_tensor[None])[0]
        denoised_chunks.append(denoised.squeeze(0).cpu().numpy())

    output = np.concatenate(denoised_chunks, axis=1).T
    sf.write(output_wav_path, output, model.sample_rate)


def replace_audio_track(video_path: str, wav_path: str, output_path: str) -> None:
    cmd = ffmpeg_cmd(
        "-y",
        "-i",
        video_path,
        "-i",
        wav_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        output_path,
    )
    run_quiet(cmd)


def extract_audio_full(input_path: str, wav_path: str) -> None:
    cmd = ffmpeg_cmd(
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        wav_path,
    )
    run_quiet(cmd)


def print_summary(raw_speech_segments: Sequence[Segment], final_segments: Sequence[Segment], total_duration: float) -> None:
    speech_total = sum(segment.duration for segment in raw_speech_segments)
    kept_total = sum(segment.duration for segment in final_segments)
    removed = max(0.0, total_duration - kept_total)

    print(f"Detected speech segments: {len(raw_speech_segments)}")
    print(f"Original duration:      {total_duration:.2f}s")
    print(f"Speech duration:        {speech_total:.2f}s")
    print(f"Output kept duration:   {kept_total:.2f}s")
    print(f"Removed duration:       {removed:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shorten pauses in audio/video with Silero VAD.")
    parser.add_argument("--input", required=True, help="Input audio or video path")
    parser.add_argument("--output", required=True, help="Output audio or video path")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero VAD threshold")
    parser.add_argument("--pad-before-ms", type=int, default=150, help="Padding before speech")
    parser.add_argument("--pad-after-ms", type=int, default=200, help="Padding after speech")
    parser.add_argument("--merge-gap-ms", type=int, default=200, help="Merge speech segments if closer than this")
    parser.add_argument("--leading-keep-ms", type=int, default=150, help="Keep this much leading silence")
    parser.add_argument("--trailing-keep-ms", type=int, default=200, help="Keep this much trailing silence")
    parser.add_argument("--preserve-short-pause-ms", type=int, default=250, help="Keep pauses shorter than this untouched")
    parser.add_argument("--long-pause-step-ms", type=int, default=150, help="Extra kept pause added for longer gaps")
    parser.add_argument("--long-pause-step-every-ms", type=int, default=1000, help="Add one pause step per this many ms")
    parser.add_argument("--max-keep-silence-ms", type=int, default=450, help="Cap kept silence for long gaps")
    parser.add_argument(
        "--transcript",
        help="Optional WhisperX transcript JSON with word_segments for filler removal",
    )
    parser.add_argument(
        "--filler-words",
        default="अः,um,uh,umm,uhh,hmm,mmm",
        help="Comma-separated filler words to remove when --transcript is provided",
    )
    parser.add_argument("--filler-pad-before-ms", type=int, default=40, help="Padding before filler cut")
    parser.add_argument("--filler-pad-after-ms", type=int, default=60, help="Padding after filler cut")
    parser.add_argument("--audio-fade-ms", type=int, default=50, help="Fade in/out duration for each kept audio chunk")
    parser.add_argument(
        "--render-chunk-segments",
        type=int,
        default=120,
        help="Legacy option retained for compatibility; not used by the current renderer",
    )
    parser.add_argument(
        "--render-chunk-duration-s",
        type=float,
        default=300.0,
        help="Legacy option retained for compatibility; not used by the current renderer",
    )
    parser.add_argument(
        "--video-preset",
        default="fast",
        help="FFmpeg x264 preset for per-segment renders",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=23,
        help="FFmpeg x264 CRF for per-segment renders",
    )
    parser.add_argument(
        "--audio-bitrate",
        default="128k",
        help="Audio bitrate for per-segment renders",
    )
    parser.add_argument(
        "--denoise",
        choices=["none", "after"],
        default="none",
        help="Apply DNS64 denoising after editing",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for denoiser: auto, cpu, or cuda",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    total_duration = media_duration(input_path)
    has_video, has_audio = ffprobe_streams(input_path)
    if not has_audio:
        raise ValueError("Input must contain an audio stream")

    total_steps = 6 if args.denoise != "none" else 4

    with tempfile.TemporaryDirectory(prefix="vad_pause_editor_") as tmpdir:
        vad_wav = os.path.join(tmpdir, "vad_input.wav")
        print_progress(f"Step 1/{total_steps}: Extracting mono audio for VAD...")
        extract_audio_for_vad(input_path, vad_wav)

        print_progress(f"Step 2/{total_steps}: Detecting speech segments...")
        raw_speech = detect_speech_segments(vad_wav, threshold=args.vad_threshold)
        if not raw_speech:
            print("No speech detected. Copying input to output.", file=sys.stderr)
            shutil.copy2(input_path, output_path)
            return 0

        print_progress(f"Step 3/{total_steps}: Building keep/cut timeline...")
        speech_segments = expand_and_merge_segments(
            raw_speech,
            total_duration=total_duration,
            pad_before_ms=args.pad_before_ms,
            pad_after_ms=args.pad_after_ms,
            merge_gap_ms=args.merge_gap_ms,
        )

        output_segments = build_output_segments(
            speech_segments=speech_segments,
            total_duration=total_duration,
            leading_keep_ms=args.leading_keep_ms,
            trailing_keep_ms=args.trailing_keep_ms,
            preserve_short_pause_ms=args.preserve_short_pause_ms,
            long_pause_step_ms=args.long_pause_step_ms,
            long_pause_step_every_ms=args.long_pause_step_every_ms,
            max_keep_silence_ms=args.max_keep_silence_ms,
        )

        filler_cut_segments: List[Segment] = []
        if args.transcript:
            filler_words = [word.strip() for word in args.filler_words.split(",")]
            filler_cut_segments = load_filler_segments(
                transcript_path=args.transcript,
                filler_words=filler_words,
                filler_pad_before_ms=args.filler_pad_before_ms,
                filler_pad_after_ms=args.filler_pad_after_ms,
            )
            output_segments = subtract_cut_segments(output_segments, filler_cut_segments)

        if args.denoise == "none":
            print_progress(f"Step 4/{total_steps}: Rendering trimmed output...")
            render_segments(
                input_path,
                output_path,
                output_segments,
                render_chunk_segments=args.render_chunk_segments,
                render_chunk_duration_s=args.render_chunk_duration_s,
                audio_fade_ms=args.audio_fade_ms,
                video_preset=args.video_preset,
                video_crf=args.video_crf,
                audio_bitrate=args.audio_bitrate,
            )
        else:
            intermediate = os.path.join(tmpdir, f"edited{Path(output_path).suffix or '.mp4'}")
            print_progress(f"Step 4/{total_steps}: Rendering trimmed output...")
            render_segments(
                input_path,
                intermediate,
                output_segments,
                render_chunk_segments=args.render_chunk_segments,
                render_chunk_duration_s=args.render_chunk_duration_s,
                audio_fade_ms=args.audio_fade_ms,
                video_preset=args.video_preset,
                video_crf=args.video_crf,
                audio_bitrate=args.audio_bitrate,
            )

            denoised_wav = os.path.join(tmpdir, "denoised.wav")
            print_progress(f"Step 5/{total_steps}: Extracting edited audio for denoise...")
            extract_audio_full(intermediate, os.path.join(tmpdir, "edited.wav"))
            print_progress(f"Step 6/{total_steps}: Denoising and replacing audio track...")
            denoise_wav(os.path.join(tmpdir, "edited.wav"), denoised_wav, device=args.device)

            if has_video:
                replace_audio_track(intermediate, denoised_wav, output_path)
            else:
                shutil.copy2(denoised_wav, output_path)

        print_progress("Completed VAD edit.")
        print_summary(raw_speech, output_segments, total_duration)
        if filler_cut_segments:
            print(f"Removed filler regions:   {len(filler_cut_segments)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
