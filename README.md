# colab-files

Media processing workspace for upload-driven transcription and pause-shortening workflows.

## What this project does

The web app currently exposes three tools:

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

3. `VoxCPM Ultimate Clone`
   - Upload reference audio
   - Optionally paste the exact prompt transcript
   - Auto-transcribe with WhisperX when the transcript is omitted
   - Generate a cloned WAV with VoxCPM2 using the ultimate cloning flow
   - Download the generated WAV, prompt transcript, and normalized reference WAV

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
POST /api/tools/voxcpm-ultimate-clone
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/download/{filename}
```

The upload endpoints accept `multipart/form-data`.

## Project structure

- `wishperx.py` contains the WhisperX transcription helpers.
- `vad_pause_editor.py` contains the pause-shortening and media rendering helpers.
- `voxcpm` powers the high-fidelity voice cloning tool.
- `app/` contains the web server, templates, static assets, and job storage.
- `runs/` is created at runtime and is ignored by Git.

### Dependency split

- `requirements-web.txt` for the FastAPI shell and templates.
- `requirements-audio.txt` for `wishperx.py` and transcript generation.
- `requirements-audio.txt` also includes VoxCPM for TTS and ultimate cloning.
- `requirements-video.txt` for `vad_pause_editor.py` and the shrink-video pipeline.
- `requirements-serverless.txt` for the Runpod queue worker image.
- `requirements.txt` as a full-stack umbrella install.
- `Makefile` provides short commands like `make dev` and `make install-all`.

## Notes

- The transcription pipeline defaults to `collabora/whisper-base-hindi`.
- The video workflow also writes a transcript JSON so the result can be reused by other scripts later.
- The default VoxCPM reference audio is stored in-repo at `app/assets/voxcpm-default-reference.wav`.
- The VoxCPM workflow uses the same reference clip for both `prompt_wav_path` and `reference_wav_path`.
- If you do not provide a reference transcript, the app derives one with WhisperX before calling VoxCPM.
- The app stores each job in its own folder under `runs/`.
- `scripts/video_trim.py` is the local hybrid path: it extracts audio locally, calls the remote audio API, and then runs the local VAD editor.

## Runpod Serverless

This repo now includes a queue-based Runpod worker:

- `handler.py` is the Runpod entrypoint.
- `Dockerfile` is the image Runpod builds from GitHub.
- The primary Whisper model should be configured as the endpoint cached model: `collabora/whisper-base-hindi`.

Current serverless behavior:

- The worker accepts either `input.audio_url` or `input.audio_base64` in the Runpod job payload.
- `wishperx.py` resolves the Runpod cached Hugging Face snapshot path automatically for the Whisper model.
- WhisperX alignment remains enabled.
- The Hindi alignment model is not covered by the single cached-model endpoint setting, so it may still download on a cold worker unless you later add your own persistent cache strategy.
- The worker returns the exact JSON content that is written to `transcript.json`.

Example job payload:

```json
{
  "input": {
    "audio_url": "https://example.com/audio.mp3",
    "filename": "audio.mp3",
    "model_name": "collabora/whisper-base-hindi",
    "language_code": "hi",
    "chunk_length_s": 30
  }
}
```

Example base64 payload:

```json
{
  "input": {
    "audio_base64": "BASE64_AUDIO_HERE",
    "filename": "audio.mp3",
    "model_name": "collabora/whisper-base-hindi",
    "language_code": "hi",
    "chunk_length_s": 30
  }
}
```

## Local hybrid trim script

Run the local hybrid script when the source video is too large to upload, but you still want the remote transcript API:

```bash
python scripts/video_trim.py input.mp4 \
  --api-base-url https://uykce-34-125-5-178.run.pinggy-free.link
```

It will:

1. Extract local audio from the video.
2. Upload only the audio to the transcript API.
3. Download the resulting `transcript.json`.
4. Run `vad_pause_editor.py` locally with that transcript.

By default it writes the final trimmed video and transcript into the same directory as the input video.

The script expects `ffmpeg` and `ffprobe` on `PATH` and the local VAD/editor dependencies installed.

To use the deployed Runpod endpoint instead of the Colab-style multipart API:

```bash
export RUNPOD_API_KEY=your_key_here
python scripts/video_trim.py input.mp4 --runpod
```

Backend selection:

- `--colab` uses the existing upload API and remains the default.
- `--runpod` sends extracted audio to the Runpod endpoint and writes the returned transcript JSON locally.
- For small extracted audio files, `--runpod` sends base64 inline.
- For larger extracted audio files, `--runpod` automatically starts a local Python file server plus a Pinggy tunnel and submits a temporary `audio_url` instead.

The large-file Runpod path requires the Pinggy Python SDK:

```bash
pip install pinggy
```

Its local VAD pass defaults to the same edit profile you use by hand:

```text
--pad-before-ms 95
--pad-after-ms 135
--merge-gap-ms 140
--preserve-short-pause-ms 160
--long-pause-step-ms 90
--long-pause-step-every-ms 800
--max-keep-silence-ms 220
--filler-words "अः,हूं,मतलब,तो,ठीक,अब,ना,वो,अरे"
--filler-pad-before-ms 25
--filler-pad-after-ms 40
--video-preset fast
--video-crf 18
--audio-bitrate 128k
```
