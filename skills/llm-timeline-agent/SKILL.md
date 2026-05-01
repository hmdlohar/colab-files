---
name: llm-timeline-agent
description: Run the three-step LLM timeline editing workflow for a source video in this repository. Use when the user provides a source video and wants the agent to run stage1_vad_transcript.py, prepare timeline.json from the generated llm_prompt.md and transcript artifacts, then run stage2_apply_timeline.py.
---

# LLM Timeline Agent

Use this skill for the repo's staged video-tightening workflow.

## Inputs

- A source video path.
- Optional stage1 backend flags such as `--colab`, `--runpod`, `--api-base-url`, `--model-name`, or `--language-code`.
- Optional instruction about editing style if the user wants a stricter or lighter cut.

## Workflow

1. Confirm the source video exists.
2. Run stage1:

```bash
~/.venv/bin/python scripts/llm_timeline_pipeline/stage1_vad_transcript.py <source_video> [stage1 backend flags...]
```

3. Resolve the work directory:
   - default: `<source_video_parent>/<source_video_stem>_llm_timeline`
   - generated artifacts include:
     - `stage1_vad.*`
     - `stage1_transcript.json`
     - `stage1_transcript.txt`
     - `stage1_word_timestamps.txt`
     - `timeline.json`
     - `llm_prompt.md`
     - `stage1_manifest.json`
4. Read these files before editing `timeline.json`:
   - `<workdir>/llm_prompt.md`
   - `scripts/llm_timeline_pipeline/TIMELINE_SPEC.md`
   - `<workdir>/stage1_transcript.json`
   - `<workdir>/stage1_word_timestamps.txt`
5. Prepare `<workdir>/timeline.json` directly in the repo/workdir.
6. Validate the timeline mentally against the spec before stage2:
   - `strategy` must be `keep_ranges`
   - `segments` must be non-empty
   - ranges must be ordered and non-overlapping
   - all timestamps must be in stage1 video seconds
   - prefer word-level timestamps and avoid cutting inside words
7. Run stage2:

```bash
~/.venv/bin/python scripts/llm_timeline_pipeline/stage2_apply_timeline.py <workdir>/stage1_vad.<ext> <workdir>/timeline.json
```

8. Report:
   - work directory
   - final output path
   - whether stage1 was reused or regenerated
   - any assumptions made while drafting `timeline.json`

## Editing Standard For `timeline.json`

- Use `llm_prompt.md` as the task framing.
- Keep the final video information-dense and continuous-feeling.
- Remove filler, repetitions, weak detours, and stretched debugging unless they add useful context.
- Keep the useful problem statement, key debugging insight, and the resolution.
- Prefer fewer, stronger keep ranges over many tiny fragments unless fine-grained cuts are clearly justified.
- Use 3 decimal places when practical.

## Reuse Rules

- If stage1 artifacts already exist for the same source video, prefer reusing them unless the user asks for a fresh stage1 pass.
- If `timeline.json` already contains curated edits, do not overwrite it blindly. Read it first and either refine it or back it up before replacing it.

## Output Naming

- Stage1 output defaults to `<workdir>/stage1_vad<source_suffix>`.
- Stage2 output defaults to `<workdir>/stage1_vad_stage2<source_suffix>`.

## Example Invocation

When the user says:

```text
Use skills/llm-timeline-agent/SKILL.md for /path/to/video.mkv
```

the agent should execute the full workflow end-to-end inside this repository.
