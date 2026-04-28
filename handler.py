from __future__ import annotations

import base64
import os
import shutil
import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import runpod
import torch

from app.pipelines import process_audio_to_transcript
from app.store import JobStore


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = Path(os.getenv("COLAB_FILES_RUNS_DIR", "/tmp/colab-files-runs"))
UPLOADS_DIR = RUNS_DIR / "uploads"
JOB_STORE = JobStore(RUNS_DIR / "jobs")
DEFAULT_MODEL_NAME = os.getenv("COLAB_FILES_MODEL_NAME", "collabora/whisper-base-hindi")
DEFAULT_LANGUAGE_CODE = os.getenv("COLAB_FILES_LANGUAGE_CODE", "hi")
DEFAULT_CHUNK_LENGTH_S = int(os.getenv("COLAB_FILES_CHUNK_LENGTH_S", "30"))
ALIGN_MODEL_NAME = os.getenv("COLAB_FILES_ALIGN_MODEL_NAME") or None
ALIGN_MODEL_DIR = os.getenv("COLAB_FILES_ALIGN_MODEL_DIR") or None
ALIGN_MODEL_CACHE_ONLY = bool(int(os.getenv("COLAB_FILES_ALIGN_MODEL_CACHE_ONLY", "0")))


def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def safe_filename(name: str | None, fallback: str = "input.bin") -> str:
    if not name:
        return fallback
    filename = Path(name).name
    return filename or fallback


def download_to_path(audio_url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(audio_url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def decode_base64_to_path(audio_base64: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = audio_base64
    if "," in audio_base64 and audio_base64.split(",", 1)[0].startswith("data:"):
        payload = audio_base64.split(",", 1)[1]

    with destination.open("wb") as handle:
        handle.write(base64.b64decode(payload))


def resolve_input_file(job_input: dict, job_id: str) -> Path:
    audio_url = job_input.get("audio_url")
    audio_base64 = job_input.get("audio_base64")
    filename = safe_filename(job_input.get("filename"), fallback=f"{job_id}.bin")

    if audio_url:
        parsed = urlparse(audio_url)
        resolved_filename = safe_filename(Path(parsed.path).name or filename, fallback=filename)
        destination = UPLOADS_DIR / job_id / resolved_filename
        download_to_path(audio_url, destination)
        return destination

    if audio_base64:
        destination = UPLOADS_DIR / job_id / filename
        decode_base64_to_path(audio_base64, destination)
        return destination

    raise ValueError("Missing required input field: provide either audio_url or audio_base64")


def load_transcript(job_id: str) -> dict:
    transcript_path = JOB_STORE.job_dir(job_id) / "transcript.json"
    with transcript_path.open("r", encoding="utf-8") as handle:
        import json

        return json.load(handle)


def handler(job: dict) -> dict:
    ensure_dirs()
    job_input = job.get("input", {})
    job_id = job.get("id") or uuid.uuid4().hex[:12]
    input_path = resolve_input_file(job_input, job_id)

    model_name = job_input.get("model_name", DEFAULT_MODEL_NAME)
    language_code = job_input.get("language_code", DEFAULT_LANGUAGE_CODE)
    chunk_length_s = int(job_input.get("chunk_length_s", DEFAULT_CHUNK_LENGTH_S))

    JOB_STORE.create(
        job_id,
        tool="runpod-audio-to-transcript",
        input_name=input_path.name,
        message="Queued Runpod audio transcription",
    )
    JOB_STORE.update(job_id, status="running", message="Transcribing audio")

    artifacts = process_audio_to_transcript(
        JOB_STORE,
        job_id,
        input_path,
        model_name=model_name,
        language_code=language_code,
        chunk_length_s=chunk_length_s,
        device=device_name(),
        align_model_name=ALIGN_MODEL_NAME,
        align_model_dir=ALIGN_MODEL_DIR,
        align_model_cache_only=ALIGN_MODEL_CACHE_ONLY,
    )
    JOB_STORE.update(job_id, status="done", message="Transcript ready", artifacts=artifacts)

    return load_transcript(job_id)


runpod.serverless.start({"handler": handler})
