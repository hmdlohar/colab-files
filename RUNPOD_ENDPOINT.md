# Runpod Endpoint

This repo is deployed as a Runpod Serverless endpoint at:

```text
https://api.runpod.ai/v2/ndw2yl11bszv8c/run
```

Use this endpoint for audio transcription with the deployed `wishperx.py` worker.

## Auth

All requests need your Runpod API key:

```text
Authorization: Bearer YOUR_RUNPOD_API_KEY
```

## Endpoints

Async run:

```text
POST https://api.runpod.ai/v2/ndw2yl11bszv8c/run
```

Sync run:

```text
POST https://api.runpod.ai/v2/ndw2yl11bszv8c/runsync
```

Job status:

```text
GET https://api.runpod.ai/v2/ndw2yl11bszv8c/status/{job_id}
```

## Input shape

The worker accepts this JSON envelope:

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

or:

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

## Required fields

You must provide one of:

- `input.audio_url`
- `input.audio_base64`

If you use `audio_base64`, also send `filename` with the correct extension such as `.mp3`, `.wav`, or `.m4a`.

## Optional fields

- `filename`: defaults to a generated fallback name if omitted
- `model_name`: defaults to `collabora/whisper-base-hindi`
- `language_code`: defaults to `hi`
- `chunk_length_s`: defaults to `30`

## Async example

```bash
curl --request POST \
  --url https://api.runpod.ai/v2/ndw2yl11bszv8c/run \
  -H "authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "content-type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://example.com/audio.mp3",
      "filename": "audio.mp3",
      "model_name": "collabora/whisper-base-hindi",
      "language_code": "hi",
      "chunk_length_s": 30
    }
  }'
```

Typical async response:

```json
{
  "id": "runpod-job-id",
  "status": "IN_QUEUE"
}
```

Then poll:

```bash
curl --request GET \
  --url https://api.runpod.ai/v2/ndw2yl11bszv8c/status/runpod-job-id \
  -H "authorization: Bearer YOUR_RUNPOD_API_KEY"
```

When the job completes, Runpod returns the transcript under `output`.

## Sync example

```bash
curl --request POST \
  --url https://api.runpod.ai/v2/ndw2yl11bszv8c/runsync \
  -H "authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "content-type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://example.com/audio.mp3",
      "filename": "audio.mp3"
    }
  }'
```

## Base64 example

Create a base64 string:

```bash
base64 -w 0 /path/to/audio.mp3
```

If `-w` is unsupported on your system:

```bash
base64 /path/to/audio.mp3 | tr -d '\n'
```

Then call the endpoint:

```bash
curl --request POST \
  --url https://api.runpod.ai/v2/ndw2yl11bszv8c/run \
  -H "authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "content-type: application/json" \
  -d '{
    "input": {
      "audio_base64": "BASE64_AUDIO_HERE",
      "filename": "audio.mp3",
      "model_name": "collabora/whisper-base-hindi",
      "language_code": "hi",
      "chunk_length_s": 30
    }
  }'
```

The worker also accepts data-URL style payloads:

```json
{
  "input": {
    "audio_base64": "data:audio/mpeg;base64,BASE64_AUDIO_HERE",
    "filename": "audio.mp3"
  }
}
```

## Output shape

The worker returns the exact JSON content written to `transcript.json`.

That means the Runpod job result looks like:

```json
{
  "id": "runpod-job-id",
  "status": "COMPLETED",
  "output": {
    "segments": [
      {
        "start": 0.0,
        "end": 2.1,
        "text": "..."
      }
    ],
    "word_segments": [
      {
        "word": "...",
        "start": 0.12,
        "end": 0.44
      }
    ]
  }
}
```

Use `output` as the transcript object.

## Notes

- `audio_url` must be publicly reachable from Runpod. Temporary tunnel URLs may fail.
- `audio_base64` is safer for testing, but it increases request size.
- The endpoint is configured for the Whisper model `collabora/whisper-base-hindi`.
- WhisperX alignment remains enabled, so the response includes aligned segment and word timing data when successful.
