# `timeline.json` Spec

This file is consumed by `stage2_apply_timeline.py`.

Goal:
- `timeline.json` tells stage2 which ranges of the stage1 VAD video should be kept.
- Stage2 concatenates those keep ranges in order to produce the final video.

Important:
- All timestamps are in seconds.
- All timestamps are relative to the **stage1 VAD output video**, not the original source video.
- `segments` must be sorted in ascending time order.
- `segments` must not overlap.
- Each segment must satisfy `0 <= start < end <= source_duration_s`.
- When choosing cut points, prefer `word_segments` timestamps from the transcript JSON over coarse segment timestamps.

## Required shape

```json
{
  "version": 1,
  "timebase": "stage1_video_seconds",
  "source_video": "/absolute/path/to/stage1_vad.mkv",
  "source_transcript": "/absolute/path/to/stage1_transcript.json",
  "source_duration_s": 152.347,
  "strategy": "keep_ranges",
  "segments": [
    {
      "start": 0.420,
      "end": 8.910,
      "label": "Strong opening",
      "reason": "Clear statement of the main point"
    },
    {
      "start": 10.050,
      "end": 24.800,
      "label": "Useful explanation",
      "reason": "Keeps the important example, removes repetition"
    }
  ]
}
```

## Minimum required keys

Only these fields are strictly required by stage2:

```json
{
  "version": 1,
  "strategy": "keep_ranges",
  "segments": [
    { "start": 0.420, "end": 8.910 },
    { "start": 10.050, "end": 24.800 }
  ]
}
```

## Editing guidance for the LLM

- Use the transcript to choose **keep ranges**, not remove ranges.
- Use word-level timestamps from `word_segments` as the main source for precise trimming.
- Prefer cutting at transcript segment boundaries or quiet word boundaries.
- Use segment-level transcript entries mainly for reading/context, not for precise cut placement.
- Avoid cutting in the middle of a spoken word.
- Avoid micro-segments unless they are intentional.
- Keep chronological order.
- If a sentence is weak, repeated, or unnecessary, exclude that time range from `segments`.
- If only a few words are unnecessary, split the surrounding content into two keep ranges around that portion.

## Recommended style

- Keep `label` short.
- Keep `reason` factual.
- Use 3 decimal places when possible.

## What stage2 does

- Reads `segments`
- Validates ordering and bounds
- Renders only those ranges from the stage1 VAD output
- Concatenates them into the final edited video
