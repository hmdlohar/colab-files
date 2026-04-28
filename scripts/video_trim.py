#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_API_BASE_URL = os.getenv("COLAB_FILES_API_BASE_URL", "https://wcuug-34-125-224-111.run.pinggy-free.link")
DEFAULT_MODEL_NAME = os.getenv("COLAB_FILES_MODEL_NAME", "collabora/whisper-base-hindi")
DEFAULT_LANGUAGE_CODE = os.getenv("COLAB_FILES_LANGUAGE_CODE", "hi")
DEFAULT_CHUNK_LENGTH_S = int(os.getenv("COLAB_FILES_CHUNK_LENGTH_S", "30"))
DEFAULT_PAD_BEFORE_MS = 95
DEFAULT_PAD_AFTER_MS = 135
DEFAULT_MERGE_GAP_MS = 140
DEFAULT_LEADING_KEEP_MS = 150
DEFAULT_TRAILING_KEEP_MS = 200
DEFAULT_PRESERVE_SHORT_PAUSE_MS = 160
DEFAULT_LONG_PAUSE_STEP_MS = 90
DEFAULT_LONG_PAUSE_STEP_EVERY_MS = 800
DEFAULT_MAX_KEEP_SILENCE_MS = 220
DEFAULT_FILLER_WORDS = "अः,हूं,मतलब,तो,ठीक,अब,ना,वो,अरे"
DEFAULT_FILLER_PAD_BEFORE_MS = 25
DEFAULT_FILLER_PAD_AFTER_MS = 40
DEFAULT_VIDEO_PRESET = "fast"
DEFAULT_VIDEO_CRF = 18
DEFAULT_AUDIO_BITRATE = "128k"
DEFAULT_HTTP_TIMEOUT_S = int(os.getenv("COLAB_FILES_HTTP_TIMEOUT_S", "60"))
DEFAULT_RETRY_ATTEMPTS = int(os.getenv("COLAB_FILES_RETRY_ATTEMPTS", "8"))
DEFAULT_RETRY_INITIAL_DELAY_S = float(os.getenv("COLAB_FILES_RETRY_INITIAL_DELAY_S", "2"))
DEFAULT_RETRY_MAX_DELAY_S = float(os.getenv("COLAB_FILES_RETRY_MAX_DELAY_S", "30"))


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return 500 <= exc.code < 600
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, OSError)
    if isinstance(exc, TimeoutError):
        return True
    return False


def request_json_with_retry(request: urllib.request.Request, *, timeout: int = DEFAULT_HTTP_TIMEOUT_S) -> dict:
    delay = DEFAULT_RETRY_INITIAL_DELAY_S
    last_error: BaseException | None = None

    for attempt in range(1, DEFAULT_RETRY_ATTEMPTS + 1):
        try:
            context = ssl.create_default_context()
            with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= DEFAULT_RETRY_ATTEMPTS or not is_retryable_error(exc):
                raise
            print(
                f"network issue on attempt {attempt}/{DEFAULT_RETRY_ATTEMPTS}: {exc}. "
                f"retrying in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2, DEFAULT_RETRY_MAX_DELAY_S)

    if last_error is not None:
        raise last_error
    raise RuntimeError("request_json_with_retry failed unexpectedly")


def ffprobe_has_audio(path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return any(stream.get("codec_type") == "audio" for stream in data.get("streams", []))


def extract_audio(input_path: Path, mp3_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "32k",
        str(mp3_path),
    ]
    run(cmd)


def build_multipart_body(fields: dict[str, str], file_field: str, file_path: Path, boundary: str) -> tuple[bytes, str]:
    lines: list[bytes] = []

    for name, value in fields.items():
        lines.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                f"{value}\r\n".encode(),
            ]
        )

    filename = file_path.name
    content_type = "application/octet-stream"
    if file_path.suffix.lower() in {".mp3", ".m4a", ".aac"}:
        content_type = "audio/mpeg"
    elif file_path.suffix.lower() in {".wav"}:
        content_type = "audio/wav"
    elif file_path.suffix.lower() in {".flac"}:
        content_type = "audio/flac"
    elif file_path.suffix.lower() in {".mp3"}:
        content_type = "audio/mpeg"
    elif file_path.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"}:
        content_type = "video/mp4"

    file_bytes = file_path.read_bytes()
    lines.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )

    body = b"".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


def post_audio_for_transcript(
    api_base_url: str,
    audio_path: Path,
    model_name: str,
    language_code: str,
    chunk_length_s: int,
) -> dict:
    boundary = f"----colabfiles{int(time.time() * 1000)}"
    fields = {
        "model_name": model_name,
        "language_code": language_code,
        "chunk_length_s": str(chunk_length_s),
    }
    body, content_type = build_multipart_body(fields, "audio_file", audio_path, boundary)

    request = urllib.request.Request(
        f"{api_base_url.rstrip('/')}/api/tools/audio-to-transcript",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    return request_json_with_retry(request)


def fetch_job(api_base_url: str, job_id: str) -> dict:
    url = f"{api_base_url.rstrip('/')}/api/jobs/{job_id}"
    request = urllib.request.Request(url, method="GET")
    return request_json_with_retry(request)


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, method="GET")
    data = request_with_retry_bytes(request)
    destination.write_bytes(data)


def request_with_retry_bytes(request: urllib.request.Request, *, timeout: int = DEFAULT_HTTP_TIMEOUT_S) -> bytes:
    delay = DEFAULT_RETRY_INITIAL_DELAY_S
    last_error: BaseException | None = None

    for attempt in range(1, DEFAULT_RETRY_ATTEMPTS + 1):
        try:
            context = ssl.create_default_context()
            with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= DEFAULT_RETRY_ATTEMPTS or not is_retryable_error(exc):
                raise
            print(
                f"network issue on attempt {attempt}/{DEFAULT_RETRY_ATTEMPTS}: {exc}. "
                f"retrying in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2, DEFAULT_RETRY_MAX_DELAY_S)

    if last_error is not None:
        raise last_error
    raise RuntimeError("request_with_retry_bytes failed unexpectedly")


def wait_for_job(api_base_url: str, job_id: str, poll_seconds: int) -> dict:
    while True:
        job = fetch_job(api_base_url, job_id)
        status = job.get("status")
        if status in {"done", "error"}:
            return job
        print(f"job {job_id}: {status} - {job.get('message', '')}", flush=True)
        time.sleep(poll_seconds)


def build_vad_command(args: argparse.Namespace, transcript_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "vad_pause_editor.py"),
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--transcript",
        str(transcript_path),
        "--pad-before-ms",
        str(args.pad_before_ms),
        "--pad-after-ms",
        str(args.pad_after_ms),
        "--merge-gap-ms",
        str(args.merge_gap_ms),
        "--leading-keep-ms",
        str(args.leading_keep_ms),
        "--trailing-keep-ms",
        str(args.trailing_keep_ms),
        "--preserve-short-pause-ms",
        str(args.preserve_short_pause_ms),
        "--long-pause-step-ms",
        str(args.long_pause_step_ms),
        "--long-pause-step-every-ms",
        str(args.long_pause_step_every_ms),
        "--max-keep-silence-ms",
        str(args.max_keep_silence_ms),
        "--filler-words",
        args.filler_words,
        "--filler-pad-before-ms",
        str(args.filler_pad_before_ms),
        "--filler-pad-after-ms",
        str(args.filler_pad_after_ms),
        "--video-preset",
        args.video_preset,
        "--video-crf",
        str(args.video_crf),
        "--audio-bitrate",
        args.audio_bitrate,
    ]

    if args.denoise != "none":
        cmd.extend(["--denoise", args.denoise])
    if args.device:
        cmd.extend(["--device", args.device])

    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid local trim pipeline: local video -> remote transcript API -> local VAD editor."
    )
    parser.add_argument("input", type=Path, help="Input video file, such as .mkv or .mp4")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Final trimmed video path. Defaults to <input_stem>_trimmed<suffix>.",
    )
    parser.add_argument(
        "--transcript-output",
        type=Path,
        default=None,
        help="Where to save the transcript JSON. Defaults to <input_stem>_transcript.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs. Defaults to the input file's directory.",
    )
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        help="Base URL of the running web server API.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Whisper model name for transcript API.")
    parser.add_argument("--language-code", default=DEFAULT_LANGUAGE_CODE, help="Language code for transcript API.")
    parser.add_argument("--chunk-length-s", type=int, default=DEFAULT_CHUNK_LENGTH_S, help="Transcript chunk length.")
    parser.add_argument("--poll-seconds", type=int, default=5, help="Polling interval for the API job.")
    parser.add_argument("--pad-before-ms", type=int, default=DEFAULT_PAD_BEFORE_MS)
    parser.add_argument("--pad-after-ms", type=int, default=DEFAULT_PAD_AFTER_MS)
    parser.add_argument("--merge-gap-ms", type=int, default=DEFAULT_MERGE_GAP_MS)
    parser.add_argument("--leading-keep-ms", type=int, default=DEFAULT_LEADING_KEEP_MS)
    parser.add_argument("--trailing-keep-ms", type=int, default=DEFAULT_TRAILING_KEEP_MS)
    parser.add_argument("--preserve-short-pause-ms", type=int, default=DEFAULT_PRESERVE_SHORT_PAUSE_MS)
    parser.add_argument("--long-pause-step-ms", type=int, default=DEFAULT_LONG_PAUSE_STEP_MS)
    parser.add_argument("--long-pause-step-every-ms", type=int, default=DEFAULT_LONG_PAUSE_STEP_EVERY_MS)
    parser.add_argument("--max-keep-silence-ms", type=int, default=DEFAULT_MAX_KEEP_SILENCE_MS)
    parser.add_argument(
        "--filler-words",
        default=DEFAULT_FILLER_WORDS,
        help="Comma-separated filler words to remove.",
    )
    parser.add_argument("--filler-pad-before-ms", type=int, default=DEFAULT_FILLER_PAD_BEFORE_MS)
    parser.add_argument("--filler-pad-after-ms", type=int, default=DEFAULT_FILLER_PAD_AFTER_MS)
    parser.add_argument("--video-preset", default=DEFAULT_VIDEO_PRESET)
    parser.add_argument("--video-crf", type=int, default=DEFAULT_VIDEO_CRF)
    parser.add_argument("--audio-bitrate", default=DEFAULT_AUDIO_BITRATE)
    parser.add_argument("--denoise", choices=["none", "after"], default="none")
    parser.add_argument("--device", default="auto", help="Denoiser device, if enabled.")
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or args.input.parent
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        args.output = output_dir / f"{args.input.stem}_trimmed{args.input.suffix}"
    elif not args.output.is_absolute():
        args.output = (output_dir / args.output).resolve()

    if args.transcript_output is None:
        args.transcript_output = output_dir / f"{args.input.stem}_transcript.json"
    elif not args.transcript_output.is_absolute():
        args.transcript_output = (output_dir / args.transcript_output).resolve()


def default_transcript_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_transcript.json"


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if args.input.suffix.lower() not in {".mkv", ".mp4", ".mov", ".webm", ".m4v"}:
        print("Warning: input does not look like a video file, but the script will continue.", file=sys.stderr)

    resolve_output_paths(args)
    transcript_path = args.transcript_output or default_transcript_path(args.input, args.output.parent)

    if not ffprobe_has_audio(args.input):
        raise ValueError("Input video must contain an audio stream")

    with tempfile.TemporaryDirectory(prefix="video_trim_") as tmpdir:
        if transcript_path.exists():
            print(f"Using existing transcript: {transcript_path}")
            args.transcript_output = transcript_path
        else:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "input_audio.mp3"

            print("Extracting local audio...")
            extract_audio(args.input, audio_path)

            print("Uploading audio to transcript API...")
            job = post_audio_for_transcript(
                api_base_url=args.api_base_url,
                audio_path=audio_path,
                model_name=args.model_name,
                language_code=args.language_code,
                chunk_length_s=args.chunk_length_s,
            )

            job_id = job["id"]
            print(f"Queued job {job_id}, waiting for transcript...")
            job = wait_for_job(args.api_base_url, job_id, args.poll_seconds)
            if job.get("status") == "error":
                raise RuntimeError(f"Transcript job failed: {job.get('error') or job.get('message')}")

            artifacts = job.get("artifacts", [])
            transcript_artifact = next((item for item in artifacts if item.get("kind") == "json"), None)
            if transcript_artifact is None:
                raise RuntimeError("Transcript API did not return a JSON artifact")

            transcript_url = f"{args.api_base_url.rstrip('/')}{transcript_artifact['download_url']}"
            print(f"Downloading transcript to {transcript_path}...")
            download_file(transcript_url, transcript_path)
            args.transcript_output = transcript_path

        print("Running local VAD trim...")
        vad_cmd = build_vad_command(args, transcript_path)
        run(vad_cmd)

    print(f"Done. Final video: {args.output}")
    print(f"Transcript JSON: {transcript_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
