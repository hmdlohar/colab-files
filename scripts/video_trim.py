#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import contextlib
import importlib
import json
import os
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_API_BASE_URL = os.getenv("COLAB_FILES_API_BASE_URL", "https://helped-orca-wrongly.ngrok-free.app/")
DEFAULT_RUNPOD_ENDPOINT_URL = os.getenv("COLAB_FILES_RUNPOD_ENDPOINT_URL", "https://api.runpod.ai/v2/ndw2yl11bszv8c/run")
DEFAULT_MODEL_NAME = os.getenv("COLAB_FILES_MODEL_NAME", "collabora/whisper-base-hindi")
DEFAULT_LANGUAGE_CODE = os.getenv("COLAB_FILES_LANGUAGE_CODE", "hi")
DEFAULT_CHUNK_LENGTH_S = int(os.getenv("COLAB_FILES_CHUNK_LENGTH_S", "30"))
DEFAULT_PAD_BEFORE_MS = 150
DEFAULT_PAD_AFTER_MS = 200
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
DEFAULT_AUDIO_FADE_MS = 50
DEFAULT_VIDEO_PRESET = "fast"
DEFAULT_VIDEO_CRF = 18
DEFAULT_AUDIO_BITRATE = "128k"
DEFAULT_HTTP_TIMEOUT_S = int(os.getenv("COLAB_FILES_HTTP_TIMEOUT_S", "60"))
DEFAULT_RETRY_ATTEMPTS = int(os.getenv("COLAB_FILES_RETRY_ATTEMPTS", "8"))
DEFAULT_RETRY_INITIAL_DELAY_S = float(os.getenv("COLAB_FILES_RETRY_INITIAL_DELAY_S", "2"))
DEFAULT_RETRY_MAX_DELAY_S = float(os.getenv("COLAB_FILES_RETRY_MAX_DELAY_S", "30"))
DEFAULT_RUNPOD_STATUS_BASE_URL = os.getenv("COLAB_FILES_RUNPOD_STATUS_BASE_URL", "https://api.runpod.ai/v2/ndw2yl11bszv8c/status")
DEFAULT_RUNPOD_INLINE_MAX_BYTES = int(os.getenv("COLAB_FILES_RUNPOD_INLINE_MAX_BYTES", str(7 * 1024 * 1024)))
DEFAULT_PINGGY_START_TIMEOUT_S = int(os.getenv("COLAB_FILES_PINGGY_START_TIMEOUT_S", "30"))


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


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


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


def post_runpod_transcript_job(
    endpoint_url: str,
    model_name: str,
    language_code: str,
    chunk_length_s: int,
    api_key: str,
    *,
    audio_path: Path | None = None,
    audio_url: str | None = None,
) -> dict:
    if audio_path is None and audio_url is None:
        raise ValueError("Provide either audio_path or audio_url for Runpod submission")

    job_input = {
        "model_name": model_name,
        "language_code": language_code,
        "chunk_length_s": chunk_length_s,
    }
    if audio_url is not None:
        job_input["audio_url"] = audio_url
        job_input["filename"] = Path(urllib.parse.urlparse(audio_url).path).name or "input_audio.mp3"
    else:
        assert audio_path is not None
        job_input["audio_base64"] = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        job_input["filename"] = audio_path.name

    payload = {"input": job_input}
    request = urllib.request.Request(
        endpoint_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    return request_json_with_retry(request)


def fetch_job(api_base_url: str, job_id: str) -> dict:
    url = f"{api_base_url.rstrip('/')}/api/jobs/{job_id}"
    request = urllib.request.Request(url, method="GET")
    return request_json_with_retry(request)


def fetch_runpod_job(status_base_url: str, job_id: str, api_key: str) -> dict:
    request = urllib.request.Request(
        f"{status_base_url.rstrip('/')}/{job_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
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


def wait_for_runpod_job(status_base_url: str, job_id: str, api_key: str, poll_seconds: int) -> dict:
    while True:
        job = fetch_runpod_job(status_base_url, job_id, api_key)
        status = job.get("status")
        if status in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            return job
        print(f"job {job_id}: {status}", flush=True)
        time.sleep(poll_seconds)


def wait_for_http_server(base_url: str, filename: str, timeout_s: int = 10) -> None:
    deadline = time.time() + timeout_s
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/{urllib.parse.quote(filename)}",
        method="HEAD",
    )
    while time.time() < deadline:
        try:
            context = ssl.create_default_context()
            with urllib.request.urlopen(request, context=context, timeout=5):
                return
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    raise RuntimeError("Local HTTP server did not start in time")


@contextlib.contextmanager
def serve_directory(directory: Path):
    port = find_free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(directory),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_http_server(base_url, next(iter(p.name for p in directory.iterdir() if p.is_file()), ""))
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextlib.contextmanager
def open_pinggy_sdk_tunnel(local_port: int):
    pinggy = importlib.import_module("pinggy")
    tunnel = pinggy.start_tunnel(forwardto=f"localhost:{local_port}")
    deadline = time.time() + DEFAULT_PINGGY_START_TIMEOUT_S
    urls = None

    while time.time() < deadline:
        candidate = getattr(tunnel, "urls", None)
        if callable(candidate):
            candidate = candidate()
        if candidate:
            urls = candidate
            break
        is_active = getattr(tunnel, "is_active", None)
        if callable(is_active):
            try:
                print(f"Waiting for Pinggy SDK public URL... active={is_active()}", flush=True)
            except Exception:  # noqa: BLE001
                print("Waiting for Pinggy SDK public URL...", flush=True)
        else:
            print("Waiting for Pinggy SDK public URL...", flush=True)
        time.sleep(0.5)

    if not urls:
        is_active = getattr(tunnel, "is_active", None)
        active_state = None
        if callable(is_active):
            try:
                active_state = is_active()
            except Exception:  # noqa: BLE001
                active_state = None
        raise RuntimeError(
            f"Pinggy SDK tunnel did not return any public URLs within {DEFAULT_PINGGY_START_TIMEOUT_S}s. "
            f"active={active_state!r}"
        )

    public_url = next((url for url in urls if str(url).startswith("https://")), None) or str(urls[0])
    try:
        yield public_url.rstrip("/")
    finally:
        for method_name in ("stop", "close"):
            method = getattr(tunnel, method_name, None)
            if callable(method):
                method()
                break


@contextlib.contextmanager
def open_pinggy_http_tunnel(local_port: int):
    try:
        importlib.import_module("pinggy")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pinggy Python SDK is not installed. Run: pip install pinggy") from exc

    print("Starting Pinggy tunnel via Python SDK...", flush=True)
    with open_pinggy_sdk_tunnel(local_port) as public_url:
        yield public_url


def build_pinggy_audio_url(public_base_url: str, audio_path: Path) -> str:
    return f"{public_base_url.rstrip('/')}/{urllib.parse.quote(audio_path.name)}"


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
        "--audio-fade-ms",
        str(args.audio_fade_ms),
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
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument(
        "--colab",
        dest="backend",
        action="store_const",
        const="colab",
        help="Use the existing multipart web API backend.",
    )
    backend_group.add_argument(
        "--runpod",
        dest="backend",
        action="store_const",
        const="runpod",
        help="Use the deployed Runpod endpoint backend.",
    )
    parser.set_defaults(backend="colab")
    parser.add_argument(
        "--runpod-endpoint-url",
        default=DEFAULT_RUNPOD_ENDPOINT_URL,
        help="Runpod /run or /runsync endpoint URL.",
    )
    parser.add_argument(
        "--runpod-status-base-url",
        default=DEFAULT_RUNPOD_STATUS_BASE_URL,
        help="Runpod status URL base, without the job id suffix.",
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
    parser.add_argument("--audio-fade-ms", type=int, default=DEFAULT_AUDIO_FADE_MS)
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


def default_extracted_audio_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_16k_mono.mp3")


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
            audio_path = default_extracted_audio_path(args.input)

            if audio_path.exists():
                print(f"Using existing extracted audio: {audio_path}")
            else:
                print("Extracting local audio...")
                extract_audio(args.input, audio_path)

            if args.backend == "runpod":
                runpod_api_key = os.getenv("RUNPOD_API_KEY")
                if not runpod_api_key:
                    raise RuntimeError("RUNPOD_API_KEY must be set when using --runpod")

                if audio_path.stat().st_size <= DEFAULT_RUNPOD_INLINE_MAX_BYTES:
                    print("Submitting audio to Runpod transcript endpoint as base64...")
                    job = post_runpod_transcript_job(
                        endpoint_url=args.runpod_endpoint_url,
                        audio_path=audio_path,
                        model_name=args.model_name,
                        language_code=args.language_code,
                        chunk_length_s=args.chunk_length_s,
                        api_key=runpod_api_key,
                    )
                else:
                    print("Audio is too large for inline Runpod payload. Starting local file server and Pinggy tunnel...")
                    with serve_directory(audio_path.parent) as local_base_url:
                        local_port = int(urllib.parse.urlparse(local_base_url).port or 80)
                        with open_pinggy_http_tunnel(local_port) as public_base_url:
                            audio_url = build_pinggy_audio_url(public_base_url, audio_path)
                            print(f"Submitting audio URL to Runpod: {audio_url}")
                            job = post_runpod_transcript_job(
                                endpoint_url=args.runpod_endpoint_url,
                                audio_url=audio_url,
                                model_name=args.model_name,
                                language_code=args.language_code,
                                chunk_length_s=args.chunk_length_s,
                                api_key=runpod_api_key,
                            )

                job_id = job["id"]
                print(f"Queued Runpod job {job_id}, waiting for transcript...")
                job = wait_for_runpod_job(
                    args.runpod_status_base_url,
                    job_id,
                    runpod_api_key,
                    args.poll_seconds,
                )
                if job.get("status") != "COMPLETED":
                    raise RuntimeError(f"Runpod transcript job failed: {job.get('error') or job}")

                transcript = job.get("output")
                if not isinstance(transcript, dict):
                    raise RuntimeError("Runpod response did not include transcript JSON in output")
                print(f"Writing transcript to {transcript_path}...")
                transcript_path.write_text(
                    json.dumps(transcript, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
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
