# LLM Timeline Pipeline

This pipeline splits editing into two passes.

## Stage 1

`stage1_vad_transcript.py`

What it does:
- runs a tight VAD-only silence removal pass on the input video
- transcribes the stage1 VAD output
- writes:
  - `stage1_vad.*`
  - `stage1_transcript.json`
  - `stage1_transcript.txt`
  - `timeline.json` template
  - `llm_prompt.md`
  - `stage1_manifest.json`

Example:

```bash
python scripts/llm_timeline_pipeline/stage1_vad_transcript.py input.mkv
```

## Stage 2

`stage2_apply_timeline.py`

What it does:
- reads the stage1 VAD output video
- reads `timeline.json`
- keeps only the requested ranges
- renders the final edited video

Example:

```bash
python scripts/llm_timeline_pipeline/stage2_apply_timeline.py \
  /path/to/workdir/stage1_vad.mkv \
  /path/to/workdir/timeline.json
```

## Timeline format

See [`TIMELINE_SPEC.md`](/media/hyper2/HYPER/projects/node/colab-files/scripts/llm_timeline_pipeline/TIMELINE_SPEC.md).
