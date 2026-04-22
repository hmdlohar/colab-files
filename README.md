# colab-files

Media processing workspace for upload-driven transcription and pause-shortening workflows.

## What this project does

The web app currently exposes two tools:

1. `Audio to Transcript`
   - Upload audio
   - Generate `transcript.json`
   - Download the normalized mono 16 kHz WAV and transcript

2. `Video to Shrinked Video`
   - Upload video
   - Extract audio
   - Generate `transcript.json`
   - Render a shorter output video using the VAD editor
   - Download the final video and transcript

The layout is intentionally set up as a multi-page tool hub with a persistent side menu so future scripts can be added without redesigning the app.

## Run locally

Install the Python packages you need:

```bash
pip install -r requirements-web.txt
```

Install `ffmpeg` and `ffprobe`, then start the app:

```bash
uvicorn app.main:app --reload
```

Or use the short command:

```bash
make dev
```

To enable the transcription and video tools in the web app, also install the media stacks:

```bash
pip install -r requirements-audio.txt
pip install -r requirements-video.txt
```

For everything in one go, install:

```bash
pip install -r requirements.txt
```

Open:

```text
http://127.0.0.1:8000
```

## API

Available endpoints:

```text
GET  /api/tools
POST /api/tools/audio-to-transcript
POST /api/tools/video-to-shrinked-video
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/download/{filename}
```

The upload endpoints accept `multipart/form-data`.

## Project structure

- `wishperx.py` contains the WhisperX transcription helpers.
- `vad_pause_editor.py` contains the pause-shortening and media rendering helpers.
- `app/` contains the web server, templates, static assets, and job storage.
- `runs/` is created at runtime and is ignored by Git.

### Dependency split

- `requirements-web.txt` for the FastAPI shell and templates.
- `requirements-audio.txt` for `wishperx.py` and transcript generation.
- `requirements-video.txt` for `vad_pause_editor.py` and the shrink-video pipeline.
- `requirements.txt` as a full-stack umbrella install.
- `Makefile` provides short commands like `make dev` and `make install-all`.

## Notes

- The transcription pipeline defaults to `collabora/whisper-base-hindi`.
- The video workflow also writes a transcript JSON so the result can be reused by other scripts later.
- The app stores each job in its own folder under `runs/`.
