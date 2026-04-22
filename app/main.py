from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.pipelines import process_audio_to_transcript, process_video_to_shrinked_video
from app.store import Artifact, JobStore


BASE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
UPLOADS_DIR = RUNS_DIR / "uploads"
JOB_STORE = JobStore(RUNS_DIR / "jobs")
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("COLAB_FILES_WORKERS", "1")))

app = FastAPI(title="Colab Files", version="0.1.0")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def device_name() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    return name or "upload.bin"


async def save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def create_job(tool: str, input_name: str, message: str = ""):
    job_id = uuid.uuid4().hex[:12]
    return JOB_STORE.create(job_id, tool=tool, input_name=input_name, message=message)


def job_response(job_id: str) -> dict:
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dir = JOB_STORE.job_dir(job_id)
    artifacts = []
    for artifact in job.artifacts:
        artifacts.append(
            {
                **artifact.__dict__,
                "download_url": f"/api/jobs/{job_id}/download/{artifact.filename}",
            }
        )

    return {
        **job.to_dict(),
        "job_url": f"/jobs/{job_id}",
        "artifacts": artifacts,
        "job_dir": str(job_dir),
    }


async def enqueue_audio_to_transcript(
    audio_file: UploadFile,
    model_name: str,
    language_code: str,
    chunk_length_s: int,
) -> dict:
    ensure_dirs()
    job = create_job("audio-to-transcript", safe_filename(audio_file.filename), "Queued audio transcription")
    upload_path = UPLOADS_DIR / job.id / safe_filename(audio_file.filename)
    await save_upload(audio_file, upload_path)
    EXECUTOR.submit(run_audio_job, job.id, upload_path, model_name, language_code, chunk_length_s)
    return job_response(job.id)


async def enqueue_video_to_shrinked_video(
    video_file: UploadFile,
    model_name: str,
    language_code: str,
    chunk_length_s: int,
    vad_threshold: float,
    pad_before_ms: int,
    pad_after_ms: int,
    merge_gap_ms: int,
    leading_keep_ms: int,
    trailing_keep_ms: int,
    preserve_short_pause_ms: int,
    long_pause_step_ms: int,
    long_pause_step_every_ms: int,
    max_keep_silence_ms: int,
    filler_words: str,
    filler_pad_before_ms: int,
    filler_pad_after_ms: int,
    video_preset: str,
    video_crf: int,
    audio_bitrate: str,
) -> dict:
    ensure_dirs()
    job = create_job("video-to-shrinked-video", safe_filename(video_file.filename), "Queued video shrink job")
    upload_path = UPLOADS_DIR / job.id / safe_filename(video_file.filename)
    await save_upload(video_file, upload_path)
    filler_list = [item.strip() for item in filler_words.split(",") if item.strip()]
    EXECUTOR.submit(
        run_video_job,
        job.id,
        upload_path,
        model_name,
        language_code,
        chunk_length_s,
        vad_threshold,
        pad_before_ms,
        pad_after_ms,
        merge_gap_ms,
        leading_keep_ms,
        trailing_keep_ms,
        preserve_short_pause_ms,
        long_pause_step_ms,
        long_pause_step_every_ms,
        max_keep_silence_ms,
        filler_list,
        filler_pad_before_ms,
        filler_pad_after_ms,
        video_preset,
        video_crf,
        audio_bitrate,
    )
    return job_response(job.id)


def run_audio_job(
    job_id: str,
    input_path: Path,
    model_name: str,
    language_code: str,
    chunk_length_s: int,
) -> None:
    try:
        JOB_STORE.update(job_id, status="running", message="Transcribing audio")
        artifacts = process_audio_to_transcript(
            JOB_STORE,
            job_id,
            input_path,
            model_name=model_name,
            language_code=language_code,
            chunk_length_s=chunk_length_s,
            device=device_name(),
        )
        JOB_STORE.update(job_id, status="done", message="Transcript ready", artifacts=artifacts)
    except Exception as exc:
        JOB_STORE.update(job_id, status="error", message="Audio transcription failed", error=str(exc))


def run_video_job(
    job_id: str,
    input_path: Path,
    model_name: str,
    language_code: str,
    chunk_length_s: int,
    vad_threshold: float,
    pad_before_ms: int,
    pad_after_ms: int,
    merge_gap_ms: int,
    leading_keep_ms: int,
    trailing_keep_ms: int,
    preserve_short_pause_ms: int,
    long_pause_step_ms: int,
    long_pause_step_every_ms: int,
    max_keep_silence_ms: int,
    filler_words: Sequence[str],
    filler_pad_before_ms: int,
    filler_pad_after_ms: int,
    video_preset: str,
    video_crf: int,
    audio_bitrate: str,
) -> None:
    try:
        JOB_STORE.update(job_id, status="running", message="Transcribing video and building shrinked edit")
        artifacts = process_video_to_shrinked_video(
            JOB_STORE,
            job_id,
            input_path,
            model_name=model_name,
            language_code=language_code,
            chunk_length_s=chunk_length_s,
            device=device_name(),
            vad_threshold=vad_threshold,
            pad_before_ms=pad_before_ms,
            pad_after_ms=pad_after_ms,
            merge_gap_ms=merge_gap_ms,
            leading_keep_ms=leading_keep_ms,
            trailing_keep_ms=trailing_keep_ms,
            preserve_short_pause_ms=preserve_short_pause_ms,
            long_pause_step_ms=long_pause_step_ms,
            long_pause_step_every_ms=long_pause_step_every_ms,
            max_keep_silence_ms=max_keep_silence_ms,
            filler_words=filler_words,
            filler_pad_before_ms=filler_pad_before_ms,
            filler_pad_after_ms=filler_pad_after_ms,
            video_preset=video_preset,
            video_crf=video_crf,
            audio_bitrate=audio_bitrate,
        )
        JOB_STORE.update(job_id, status="done", message="Shrinked video ready", artifacts=artifacts)
    except Exception as exc:
        JOB_STORE.update(job_id, status="error", message="Video pipeline failed", error=str(exc))


@app.on_event("startup")
def on_startup() -> None:
    ensure_dirs()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    tools = [
        {
            "slug": "audio-to-transcript",
            "title": "Audio to Transcript",
            "description": "Upload audio and generate a timestamped transcript JSON.",
            "status": "active",
        },
        {
            "slug": "video-to-shrinked-video",
            "title": "Video to Shrinked Video",
            "description": "Upload video, transcribe it, and produce a shorter cut plus transcript.",
            "status": "active",
        },
        {
            "slug": "denoiser",
            "title": "Denoiser",
            "description": "Reserved for a future cleanup pipeline.",
            "status": "planned",
        },
        {
            "slug": "whisper-batch",
            "title": "Whisper Batch",
            "description": "Reserved for more transcription variants and batch jobs.",
            "status": "planned",
        },
    ]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Colab Files",
            "tools": tools,
        },
    )


@app.get("/tools/audio-to-transcript", response_class=HTMLResponse)
def audio_tool(request: Request):
    return templates.TemplateResponse(
        "tool_audio.html",
        {
            "request": request,
            "title": "Audio to Transcript",
        },
    )


@app.get("/tools/video-to-shrinked-video", response_class=HTMLResponse)
def video_tool(request: Request):
    return templates.TemplateResponse(
        "tool_video.html",
        {
            "request": request,
            "title": "Video to Shrinked Video",
        },
    )


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return templates.TemplateResponse(
        "job.html",
        {
            "request": request,
            "title": f"Job {job_id}",
            "job": job_response(job_id),
        },
    )


@app.get("/api/tools")
def api_tools():
    return {
        "tools": [
            {"slug": "audio-to-transcript", "title": "Audio to Transcript"},
            {"slug": "video-to-shrinked-video", "title": "Video to Shrinked Video"},
        ]
    }


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    return job_response(job_id)


@app.get("/api/jobs/{job_id}/download/{filename}")
def api_download(job_id: str, filename: str):
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dir = JOB_STORE.job_dir(job_id)
    safe_name = Path(filename).name
    allowed = {artifact.filename for artifact in job.artifacts}
    if safe_name not in allowed:
        raise HTTPException(status_code=404, detail="Artifact not found")
    file_path = job_dir / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(file_path, filename=file_path.name)


@app.post("/api/tools/audio-to-transcript")
async def api_audio_to_transcript(
    audio_file: UploadFile = File(...),
    model_name: str = Form("collabora/whisper-base-hindi"),
    language_code: str = Form("hi"),
    chunk_length_s: int = Form(30),
):
    return JSONResponse(
        await enqueue_audio_to_transcript(
            audio_file=audio_file,
            model_name=model_name,
            language_code=language_code,
            chunk_length_s=chunk_length_s,
        )
    )


@app.post("/api/tools/video-to-shrinked-video")
async def api_video_to_shrinked_video(
    video_file: UploadFile = File(...),
    model_name: str = Form("collabora/whisper-base-hindi"),
    language_code: str = Form("hi"),
    chunk_length_s: int = Form(30),
    vad_threshold: float = Form(0.5),
    pad_before_ms: int = Form(120),
    pad_after_ms: int = Form(180),
    merge_gap_ms: int = Form(200),
    leading_keep_ms: int = Form(150),
    trailing_keep_ms: int = Form(200),
    preserve_short_pause_ms: int = Form(250),
    long_pause_step_ms: int = Form(150),
    long_pause_step_every_ms: int = Form(1000),
    max_keep_silence_ms: int = Form(450),
    filler_words: str = Form("अः,um,uh,umm,uhh,hmm,mmm"),
    filler_pad_before_ms: int = Form(40),
    filler_pad_after_ms: int = Form(60),
    video_preset: str = Form("fast"),
    video_crf: int = Form(23),
    audio_bitrate: str = Form("128k"),
):
    return JSONResponse(
        await enqueue_video_to_shrinked_video(
            video_file=video_file,
            model_name=model_name,
            language_code=language_code,
            chunk_length_s=chunk_length_s,
            vad_threshold=vad_threshold,
            pad_before_ms=pad_before_ms,
            pad_after_ms=pad_after_ms,
            merge_gap_ms=merge_gap_ms,
            leading_keep_ms=leading_keep_ms,
            trailing_keep_ms=trailing_keep_ms,
            preserve_short_pause_ms=preserve_short_pause_ms,
            long_pause_step_ms=long_pause_step_ms,
            long_pause_step_every_ms=long_pause_step_every_ms,
            max_keep_silence_ms=max_keep_silence_ms,
            filler_words=filler_words,
            filler_pad_before_ms=filler_pad_before_ms,
            filler_pad_after_ms=filler_pad_after_ms,
            video_preset=video_preset,
            video_crf=video_crf,
            audio_bitrate=audio_bitrate,
        )
    )


@app.post("/tools/audio-to-transcript", response_class=HTMLResponse)
async def submit_audio_tool(
    request: Request,
    audio_file: UploadFile = File(...),
    model_name: str = Form("collabora/whisper-base-hindi"),
    language_code: str = Form("hi"),
    chunk_length_s: int = Form(30),
):
    job = await enqueue_audio_to_transcript(
        audio_file=audio_file,
        model_name=model_name,
        language_code=language_code,
        chunk_length_s=chunk_length_s,
    )
    return RedirectResponse(url=f"/jobs/{job['id']}", status_code=303)


@app.post("/tools/video-to-shrinked-video", response_class=HTMLResponse)
async def submit_video_tool(
    request: Request,
    video_file: UploadFile = File(...),
    model_name: str = Form("collabora/whisper-base-hindi"),
    language_code: str = Form("hi"),
    chunk_length_s: int = Form(30),
    vad_threshold: float = Form(0.5),
    pad_before_ms: int = Form(120),
    pad_after_ms: int = Form(180),
    merge_gap_ms: int = Form(200),
    leading_keep_ms: int = Form(150),
    trailing_keep_ms: int = Form(200),
    preserve_short_pause_ms: int = Form(250),
    long_pause_step_ms: int = Form(150),
    long_pause_step_every_ms: int = Form(1000),
    max_keep_silence_ms: int = Form(450),
    filler_words: str = Form("अः,um,uh,umm,uhh,hmm,mmm"),
    filler_pad_before_ms: int = Form(40),
    filler_pad_after_ms: int = Form(60),
    video_preset: str = Form("fast"),
    video_crf: int = Form(23),
    audio_bitrate: str = Form("128k"),
):
    job = await enqueue_video_to_shrinked_video(
        video_file=video_file,
        model_name=model_name,
        language_code=language_code,
        chunk_length_s=chunk_length_s,
        vad_threshold=vad_threshold,
        pad_before_ms=pad_before_ms,
        pad_after_ms=pad_after_ms,
        merge_gap_ms=merge_gap_ms,
        leading_keep_ms=leading_keep_ms,
        trailing_keep_ms=trailing_keep_ms,
        preserve_short_pause_ms=preserve_short_pause_ms,
        long_pause_step_ms=long_pause_step_ms,
        long_pause_step_every_ms=long_pause_step_every_ms,
        max_keep_silence_ms=max_keep_silence_ms,
        filler_words=filler_words,
        filler_pad_before_ms=filler_pad_before_ms,
        filler_pad_after_ms=filler_pad_after_ms,
        video_preset=video_preset,
        video_crf=video_crf,
        audio_bitrate=audio_bitrate,
    )
    return RedirectResponse(url=f"/jobs/{job['id']}", status_code=303)
