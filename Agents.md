# Agents

This repository is a small media-processing workspace with a web front end and two reusable CLI scripts.

## Current Architecture

- `app/main.py` is the FastAPI entry point.
- `app/pipelines.py` coordinates transcription and media editing.
- `app/store.py` persists job state and artifact metadata in `runs/jobs/<job_id>/job.json`.
- `wishperx.py` provides the WhisperX helpers used for audio/video transcription.
- `vad_pause_editor.py` provides the VAD-based pause-shortening renderer.

## Conventions

- Keep new tools as separate pages in the sidebar and on the home dashboard.
- Add APIs alongside the UI for every new tool.
- Put generated files under `runs/` and keep that directory ignored.
- Reuse the existing pipeline helpers instead of re-implementing media logic in the web layer.
- Preserve CLI usability when extending a script so it can still be run outside the server.
- For the staged video-tightening flow, use [skills/llm-timeline-agent/SKILL.md](/media/hyper2/HYPER/projects/node/colab-files/skills/llm-timeline-agent/SKILL.md) when the task is: run stage1, prepare `timeline.json` from the generated prompt/transcript artifacts, then run stage2.

## Adding a New Tool

1. Add the reusable processing function in `app/pipelines.py` or a new module under `app/`.
2. Add a job runner in `app/main.py`.
3. Add a page under `app/templates/`.
4. Add a matching API endpoint.
5. Add a link in the sidebar and a tile on the home page.

## Operational Notes

- The app uses a thread pool for background jobs.
- The first run of WhisperX or Silero VAD can download model weights.
- `ffmpeg` and `ffprobe` must be available on `PATH`.

## Extension Ideas

- Denoiser tool
- Batch Whisper jobs
- Transcript-only mode for video uploads
- More language presets and model presets
