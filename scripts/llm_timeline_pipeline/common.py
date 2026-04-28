#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vad_pause_editor import Segment, media_duration, render_segments  # noqa: E402


DEFAULT_API_BASE_URL = os.getenv("COLAB_FILES_API_BASE_URL", "https://helped-orca-wrongly.ngrok-free.app/")
DEFAULT_RUNPOD_ENDPOINT_URL = os.getenv("COLAB_FILES_RUNPOD_ENDPOINT_URL", "https://api.runpod.ai/v2/ndw2yl11bszv8c/run")
DEFAULT_RUNPOD_STATUS_BASE_URL = os.getenv("COLAB_FILES_RUNPOD_STATUS_BASE_URL", "https://api.runpod.ai/v2/ndw2yl11bszv8c/status")
DEFAULT_MODEL_NAME = os.getenv("COLAB_FILES_MODEL_NAME", "collabora/whisper-base-hindi")
DEFAULT_LANGUAGE_CODE = os.getenv("COLAB_FILES_LANGUAGE_CODE", "hi")
DEFAULT_CHUNK_LENGTH_S = int(os.getenv("COLAB_FILES_CHUNK_LENGTH_S", "30"))
DEFAULT_HTTP_TIMEOUT_S = int(os.getenv("COLAB_FILES_HTTP_TIMEOUT_S", "60"))
DEFAULT_RETRY_ATTEMPTS = int(os.getenv("COLAB_FILES_RETRY_ATTEMPTS", "8"))
DEFAULT_RETRY_INITIAL_DELAY_S = float(os.getenv("COLAB_FILES_RETRY_INITIAL_DELAY_S", "2"))
DEFAULT_RETRY_MAX_DELAY_S = float(os.getenv("COLAB_FILES_RETRY_MAX_DELAY_S", "30"))
DEFAULT_RUNPOD_INLINE_MAX_BYTES = int(os.getenv("COLAB_FILES_RUNPOD_INLINE_MAX_BYTES", str(7 * 1024 * 1024)))
DEFAULT_PINGGY_START_TIMEOUT_S = int(os.getenv("COLAB_FILES_PINGGY_START_TIMEOUT_S", "30"))

DEFAULT_AUDIO_FADE_MS = 50
DEFAULT_VIDEO_PRESET = "fast"
DEFAULT_VIDEO_CRF = 18
DEFAULT_AUDIO_BITRATE = "128k"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def print_progress(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
            print_progress(
                f"network issue on attempt {attempt}/{DEFAULT_RETRY_ATTEMPTS}: {exc}. "
                f"retrying in {delay:.1f}s"
            )
            time.sleep(delay)
            delay = min(delay * 2, DEFAULT_RETRY_MAX_DELAY_S)

    if last_error is not None:
        raise last_error
    raise RuntimeError("request_json_with_retry failed unexpectedly")


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
            print_progress(
                f"network issue on attempt {attempt}/{DEFAULT_RETRY_ATTEMPTS}: {exc}. "
                f"retrying in {delay:.1f}s"
            )
            time.sleep(delay)
            delay = min(delay * 2, DEFAULT_RETRY_MAX_DELAY_S)

    if last_error is not None:
        raise last_error
    raise RuntimeError("request_with_retry_bytes failed unexpectedly")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def extract_audio_mp3(input_path: Path, mp3_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
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
    elif file_path.suffix.lower() == ".wav":
        content_type = "audio/wav"
    elif file_path.suffix.lower() == ".flac":
        content_type = "audio/flac"
    elif file_path.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"}:
        content_type = "video/mp4"

    lines.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )

    body = b"".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


def post_audio_for_transcript(api_base_url: str, audio_path: Path, model_name: str, language_code: str, chunk_length_s: int) -> dict:
    boundary = f"----colabfiles{int(time.time() * 1000)}"
    body, content_type = build_multipart_body(
        {
            "model_name": model_name,
            "language_code": language_code,
            "chunk_length_s": str(chunk_length_s),
        },
        "audio_file",
        audio_path,
        boundary,
    )
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

    job_input: dict[str, Any] = {
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

    request = urllib.request.Request(
        endpoint_url,
        data=json.dumps({"input": job_input}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    return request_json_with_retry(request)


def fetch_job(api_base_url: str, job_id: str) -> dict:
    request = urllib.request.Request(f"{api_base_url.rstrip('/')}/api/jobs/{job_id}", method="GET")
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
    destination.write_bytes(request_with_retry_bytes(request))


def wait_for_job(api_base_url: str, job_id: str, poll_seconds: int) -> dict:
    while True:
        job = fetch_job(api_base_url, job_id)
        status = job.get("status")
        if status in {"done", "error"}:
            return job
        print_progress(f"job {job_id}: {status} - {job.get('message', '')}")
        time.sleep(poll_seconds)


def wait_for_runpod_job(status_base_url: str, job_id: str, api_key: str, poll_seconds: int) -> dict:
    while True:
        job = fetch_runpod_job(status_base_url, job_id, api_key)
        status = job.get("status")
        if status in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            return job
        print_progress(f"job {job_id}: {status}")
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
        filename = next((path.name for path in directory.iterdir() if path.is_file()), "")
        wait_for_http_server(base_url, filename)
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
    import importlib

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
        time.sleep(0.5)

    if not urls:
        raise RuntimeError(f"Pinggy SDK tunnel did not return any public URLs within {DEFAULT_PINGGY_START_TIMEOUT_S}s")

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
    import importlib

    try:
        importlib.import_module("pinggy")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pinggy Python SDK is not installed. Run: pip install pinggy") from exc

    print_progress("Starting Pinggy tunnel via Python SDK...")
    with open_pinggy_sdk_tunnel(local_port) as public_url:
        yield public_url


def build_pinggy_audio_url(public_base_url: str, audio_path: Path) -> str:
    return f"{public_base_url.rstrip('/')}/{urllib.parse.quote(audio_path.name)}"


def ensure_word_segments(transcript: dict) -> dict:
    if transcript.get("word_segments"):
        return transcript

    word_segments: list[dict[str, Any]] = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            start = word.get("start")
            end = word.get("end")
            if start is None or end is None:
                continue
            word_segments.append(
                {
                    "word": word.get("word", ""),
                    "start": float(start),
                    "end": float(end),
                }
            )
    transcript["word_segments"] = word_segments
    return transcript


def transcript_to_text(transcript: dict) -> str:
    lines: list[str] = []
    for index, segment in enumerate(transcript.get("segments", []), start=1):
        start = segment.get("start")
        end = segment.get("end")
        text = " ".join(str(segment.get("text", "")).split())
        if start is None or end is None or not text:
            continue
        lines.append(f"[{index:03d}] {float(start):8.3f} -> {float(end):8.3f}  {text}")
    return "\n".join(lines) + ("\n" if lines else "")


def transcript_words_to_text(transcript: dict) -> str:
    lines: list[str] = []
    for index, word in enumerate(transcript.get("word_segments", []), start=1):
        start = word.get("start")
        end = word.get("end")
        token = " ".join(str(word.get("word", "")).split())
        if start is None or end is None or not token:
            continue
        lines.append(f"[{index:04d}] {float(start):8.3f} -> {float(end):8.3f}  {token}")
    return "\n".join(lines) + ("\n" if lines else "")


def build_timeline_template(source_video: Path, source_transcript: Path, source_duration_s: float) -> dict:
    return {
        "version": 1,
        "timebase": "stage1_video_seconds",
        "source_video": str(source_video),
        "source_transcript": str(source_transcript),
        "source_duration_s": round(source_duration_s, 3),
        "strategy": "keep_ranges",
        "segments": [],
        "notes": [
            "Fill segments with the ordered keep ranges for the final cut.",
            "All times must be in seconds relative to the stage1 VAD output video.",
        ],
    }


def build_llm_prompt(
    *,
    source_video: Path,
    transcript_json: Path,
    transcript_txt: Path,
    transcript_words_txt: Path,
    transcript_words_text: str,
    timeline_json: Path,
    timeline_spec: Path,
    source_duration_s: float,
) -> str:
    header = "\n".join(
        [
            "You are preparing timeline.json for stage2 video editing.",
            "",
            "Files:",
            f"- Source video: {source_video}",
            f"- Transcript JSON: {transcript_json}",
            f"- Segment transcript text: {transcript_txt}",
            f"- Word timestamp text: {transcript_words_txt}",
            f"- Timeline spec: {timeline_spec}",
            f"- Write output here: {timeline_json}",
            "",
            f"Stage1 video duration: {source_duration_s:.3f} seconds",
            "",
            "Task:",
            "- Read the timeline spec first.",
            "- Decide which ranges of the stage1 video should be kept in the final output.",
            "- Use word-level timestamps from transcript_json word_segments and the word timestamp text file as the primary timing source.",
            "- Remove unnecessary words, repetitions, filler, and weak segments by excluding them from the keep ranges.",
            "- Do not cut inside a spoken word unless absolutely necessary.",
            "- Keep the final keep ranges ordered, non-overlapping, and in stage1 timebase.",
            "",
            "Editing goal:",
            "- Remove uninteresting segments from the video.",
            "- Remove filler words that make the speech boring.",
            "- Remove stretched segments that make the video boring for a YouTube audience.",
            "- Remove non-essential parts of the video, such as repeated explanations and stretched debug sections. Remove the stretched part, keep the useful debug part with the solution, and tighten it.",
            "- Make sure these removals still produce a continuous-feeling video timeline and preserve the information delivered in the video.",
            "",
            "Required output:",
            "- Produce valid JSON only.",
            "- Write the final keep ranges into timeline.json using the schema from the spec.",
        ]
    ) + "\n"
    transcript_block = [
        "",
        "Primary timing source for editing: word-level timestamps",
        "```text",
        transcript_words_text.rstrip(),
        "```",
        "",
    ]
    return header + "\n".join(transcript_block)


def submit_transcript_job(
    *,
    backend: str,
    api_base_url: str,
    runpod_endpoint_url: str,
    runpod_status_base_url: str,
    audio_path: Path,
    model_name: str,
    language_code: str,
    chunk_length_s: int,
    poll_seconds: int,
) -> dict:
    if backend == "runpod":
        runpod_api_key = os.getenv("RUNPOD_API_KEY")
        if not runpod_api_key:
            raise RuntimeError("RUNPOD_API_KEY must be set when using --runpod")

        if audio_path.stat().st_size <= DEFAULT_RUNPOD_INLINE_MAX_BYTES:
            print_progress("Submitting audio to Runpod transcript endpoint as base64...")
            job = post_runpod_transcript_job(
                endpoint_url=runpod_endpoint_url,
                audio_path=audio_path,
                model_name=model_name,
                language_code=language_code,
                chunk_length_s=chunk_length_s,
                api_key=runpod_api_key,
            )
        else:
            print_progress("Audio is too large for inline Runpod payload. Starting local file server and Pinggy tunnel...")
            with serve_directory(audio_path.parent) as local_base_url:
                local_port = int(urllib.parse.urlparse(local_base_url).port or 80)
                with open_pinggy_http_tunnel(local_port) as public_base_url:
                    audio_url = build_pinggy_audio_url(public_base_url, audio_path)
                    print_progress(f"Submitting audio URL to Runpod: {audio_url}")
                    job = post_runpod_transcript_job(
                        endpoint_url=runpod_endpoint_url,
                        audio_url=audio_url,
                        model_name=model_name,
                        language_code=language_code,
                        chunk_length_s=chunk_length_s,
                        api_key=runpod_api_key,
                    )

        job_id = job["id"]
        print_progress(f"Queued Runpod job {job_id}, waiting for transcript...")
        job = wait_for_runpod_job(runpod_status_base_url, job_id, runpod_api_key, poll_seconds)
        if job.get("status") != "COMPLETED":
            raise RuntimeError(f"Runpod transcript job failed: {job.get('error') or job}")
        transcript = job.get("output")
        if not isinstance(transcript, dict):
            raise RuntimeError("Runpod response did not include transcript JSON in output")
        return transcript

    print_progress("Uploading audio to transcript API...")
    job = post_audio_for_transcript(
        api_base_url=api_base_url,
        audio_path=audio_path,
        model_name=model_name,
        language_code=language_code,
        chunk_length_s=chunk_length_s,
    )
    job_id = job["id"]
    print_progress(f"Queued job {job_id}, waiting for transcript...")
    job = wait_for_job(api_base_url, job_id, poll_seconds)
    if job.get("status") == "error":
        raise RuntimeError(f"Transcript job failed: {job.get('error') or job.get('message')}")

    artifacts = job.get("artifacts", [])
    transcript_artifact = next((item for item in artifacts if item.get("kind") == "json"), None)
    if transcript_artifact is None:
        raise RuntimeError("Transcript API did not return a JSON artifact")

    tmp_path = audio_path.with_suffix(".transcript.download.json")
    transcript_url = f"{api_base_url.rstrip('/')}{transcript_artifact['download_url']}"
    download_file(transcript_url, tmp_path)
    transcript = json.loads(tmp_path.read_text(encoding="utf-8"))
    tmp_path.unlink(missing_ok=True)
    return transcript


def parse_timeline_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("timeline.json must contain a top-level object")
    return data


def validate_timeline(data: dict, *, duration_s: float) -> list[Segment]:
    if data.get("strategy") != "keep_ranges":
        raise ValueError("timeline.json strategy must be 'keep_ranges'")

    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("timeline.json must contain a non-empty segments array")

    validated: list[Segment] = []
    previous_end = 0.0
    for index, item in enumerate(raw_segments, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"segments[{index - 1}] must be an object")
        start = item.get("start")
        end = item.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"segments[{index - 1}] start/end must be numbers")
        start_s = float(start)
        end_s = float(end)
        if start_s < 0 or end_s <= start_s:
            raise ValueError(f"segments[{index - 1}] has invalid range {start_s} -> {end_s}")
        if end_s > duration_s + 1e-6:
            raise ValueError(f"segments[{index - 1}] ends after source duration {duration_s:.3f}s")
        if start_s < previous_end - 1e-6:
            raise ValueError(f"segments[{index - 1}] overlaps or is out of order")
        previous_end = end_s
        validated.append(Segment(start_s, end_s))

    return validated


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        help="Base URL of the running web server API.",
    )
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--colab", dest="backend", action="store_const", const="colab", help="Use the multipart web API backend.")
    backend_group.add_argument("--runpod", dest="backend", action="store_const", const="runpod", help="Use the deployed Runpod endpoint backend.")
    parser.set_defaults(backend="colab")
    parser.add_argument("--runpod-endpoint-url", default=DEFAULT_RUNPOD_ENDPOINT_URL, help="Runpod /run or /runsync endpoint URL.")
    parser.add_argument("--runpod-status-base-url", default=DEFAULT_RUNPOD_STATUS_BASE_URL, help="Runpod status URL base, without the job id suffix.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Whisper model name for transcript API.")
    parser.add_argument("--language-code", default=DEFAULT_LANGUAGE_CODE, help="Language code for transcript API.")
    parser.add_argument("--chunk-length-s", type=int, default=DEFAULT_CHUNK_LENGTH_S, help="Transcript chunk length.")
    parser.add_argument("--poll-seconds", type=int, default=5, help="Polling interval for transcript jobs.")
