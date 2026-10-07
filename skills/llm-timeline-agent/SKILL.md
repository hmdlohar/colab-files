---
name: llm-timeline-agent
description: One-step YouTube edit of a recorded video. Given a source video path, run stage1 (VAD + transcript), draft timeline.json (cut filler, repetition, stretched sections), and render with stage2. Use when the user gives a source video and wants it edited/tightened/made ready for YouTube.
---

# LLM Timeline Agent

The user gives a source video path. You deliver the edited video and a short review list. Run all three steps without stopping, unless the user says "review first". In that case, stop after writing `timeline.json` and show the cut list.

All commands run from the repo root: `/media/hyper2/HYPER/projects/node/colab-files`.

## Inputs

- Source video path. If the user gives a folder, use the single video in it, and ask only if there are several.
- Optional stage1 backend flags: `--colab`, `--runpod`, `--api-base-url`, `--model-name`, `--language-code`.
- Optional editing style, such as "light cut" or "aggressive cut". The default is a medium cut aiming for roughly 65–80% of the stage1 duration.

## Workflow

1. Confirm the source video exists. Workdir = `<source_parent>/<source_stem>_llm_timeline`.
2. **Stage1.** Skip this step if `<workdir>/stage1_manifest.json` and `stage1_transcript.json` already exist, unless the user asks for a fresh pass. Stage1 can take a while, so run it in the background and wait for it to finish:
   ```bash
   ~/.venv/bin/python scripts/llm_timeline_pipeline/stage1_vad_transcript.py <source_video> [backend flags...]
   ```
3. **Inspect.** Don't read the full `llm_prompt.md`, which is large. Its header holds the editing goal. Instead run:
   ```bash
   python3 scripts/llm_timeline_pipeline/inspect_transcript.py <workdir>
   ```
   This prints pause-grouped lines with start and end times, then a WARNINGS block:
   - `LOOP`: Whisper hallucinated a repeating phrase. Word timings inside it are fake, and the real content is unknown.
   - `GAP ... speech likely`: someone is talking but nothing was transcribed (often at 30s chunk boundaries).
4. **Draft keep ranges.** Read the lines and decide what to keep:
   - Cut: personal asides, sign-in or UI fumbling, "which one was it" clicking, repeated sentences (keep the last, cleanest take), filler like "umm / kya bolte hain", long browsing with no conclusion, and meta-chatter about the video itself.
   - Keep: the problem statement, the key comparisons and results, the decision and its reasoning, the resolution, the takeaway, and a short outro.
   - For `LOOP` and `GAP (speech likely)` regions, keep them when they sit inside a kept topic and cut them when they sit inside a cut topic. Never place a cut point inside one. Always list them in the final report.
   - Get exact cut points with:
     ```bash
     python3 scripts/llm_timeline_pipeline/inspect_transcript.py <workdir> --words 440-450 803-807
     ```
     Start a range about 0.05s before the first kept word, and end it about 0.08s after the last kept word. Never cross into the neighbouring word.
   - Prefer about 8–20 strong ranges over many tiny ones.
5. **Write `<workdir>/timeline.json`.** Follow the `scripts/llm_timeline_pipeline/TIMELINE_SPEC.md` schema, with `strategy: keep_ranges` and a short `label` and `reason` on each range.
   - If the file already has segments, back it up to `timeline.backup-<n>.json` first.
   - Validate it with:
     ```bash
     python3 scripts/llm_timeline_pipeline/inspect_transcript.py <workdir> --check
     ```
6. **Stage2.** This writes `<workdir>/stage1_vad_stage2<suffix>`:
   ```bash
   ~/.venv/bin/python scripts/llm_timeline_pipeline/stage2_apply_timeline.py <workdir>/stage1_vad.<ext> <workdir>/timeline.json
   ```
7. **Report** briefly:
   - final output path, and duration before → after
   - whether stage1 was reused
   - what was cut, one line per topic
   - **Review these timestamps** in the final video: LOOP/GAP regions that were kept, and any cut you were unsure about

## Re-edits

When the user says something like "keep the part about X" or "cut more at the end":
- Edit the existing `timeline.json` and don't redraft it from scratch.
- Re-run `--check`, then stage2.
