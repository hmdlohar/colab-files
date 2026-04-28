from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

from app.store import Artifact, JobStore


SAMPLE_RATE = 16000


def _wishperx():
    from wishperx import align_words, convert_to_mono_16k_wav, transcribe_chunks

    return align_words, convert_to_mono_16k_wav, transcribe_chunks


def _vad_editor():
    from vad_pause_editor import (
        build_output_segments,
        detect_speech_segments,
        expand_and_merge_segments,
        extract_audio_for_vad,
        ffprobe_streams,
        load_filler_segments,
        media_duration,
        render_segments,
        subtract_cut_segments,
    )

    return {
        "build_output_segments": build_output_segments,
        "detect_speech_segments": detect_speech_segments,
        "expand_and_merge_segments": expand_and_merge_segments,
        "extract_audio_for_vad": extract_audio_for_vad,
        "ffprobe_streams": ffprobe_streams,
        "load_filler_segments": load_filler_segments,
        "media_duration": media_duration,
        "render_segments": render_segments,
        "subtract_cut_segments": subtract_cut_segments,
    }


def ensure_word_segments(aligned_result: Dict) -> Dict:
    if aligned_result.get("word_segments"):
        return aligned_result

    word_segments: List[Dict] = []
    for segment in aligned_result.get("segments", []):
        for word in segment.get("words", []):
            start = word.get("start")
            end = word.get("end")
            if start is None or end is None:
                continue
            word_segments.append(
                {
                    "word": word.get("word", ""),
                    "start": float(start),
                    "end": float(end),
                }
            )

    aligned_result["word_segments"] = word_segments
    return aligned_result


def write_json(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def process_audio_to_transcript(
    store: JobStore,
    job_id: str,
    input_path: Path,
    *,
    model_name: str,
    language_code: str,
    chunk_length_s: int,
    device: str,
    align_model_name: str | None = None,
    align_model_dir: str | None = None,
    align_model_cache_only: bool | None = None,
) -> List[Artifact]:
    align_words, convert_to_mono_16k_wav, transcribe_chunks = _wishperx()
    job_dir = store.job_dir(job_id)
    wav_path = job_dir / f"{input_path.stem}_16k_mono.wav"
    transcript_path = job_dir / "transcript.json"

    convert_to_mono_16k_wav(str(input_path), str(wav_path), SAMPLE_RATE)
    segments = transcribe_chunks(
        wav_file=str(wav_path),
        model_name=model_name,
        device=device,
        sampling_rate=SAMPLE_RATE,
        chunk_length_s=chunk_length_s,
    )
    aligned = align_words(
        segments=segments,
        wav_file=str(wav_path),
        language_code=language_code,
        device=device,
        align_model_name=align_model_name,
        align_model_dir=align_model_dir,
        align_model_cache_only=(
            bool(int(os.getenv("COLAB_FILES_ALIGN_MODEL_CACHE_ONLY", "0")))
            if align_model_cache_only is None
            else align_model_cache_only
        ),
    )
    aligned = ensure_word_segments(aligned)
    write_json(transcript_path, aligned)

    return [
        Artifact(name="transcript", filename=transcript_path.name, kind="json"),
        Artifact(name="wav", filename=wav_path.name, kind="audio"),
    ]


def process_video_to_shrinked_video(
    store: JobStore,
    job_id: str,
    input_path: Path,
    *,
    model_name: str,
    language_code: str,
    chunk_length_s: int,
    device: str,
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
    align_model_name: str | None = None,
    align_model_dir: str | None = None,
    align_model_cache_only: bool | None = None,
) -> List[Artifact]:
    vad = _vad_editor()
    align_words, _, transcribe_chunks = _wishperx()
    job_dir = store.job_dir(job_id)
    output_suffix = input_path.suffix or ".mp4"
    transcript_path = job_dir / "transcript.json"
    final_video_path = job_dir / f"final{output_suffix}"

    has_video, has_audio = vad["ffprobe_streams"](str(input_path))
    if not has_audio:
        raise ValueError("Input must contain an audio stream")
    if not has_video:
        raise ValueError("Input must contain a video stream")

    extract_audio_path = job_dir / f"{input_path.stem}_16k_mono.wav"
    vad["extract_audio_for_vad"](str(input_path), str(extract_audio_path))

    segments = transcribe_chunks(
        wav_file=str(extract_audio_path),
        model_name=model_name,
        device=device,
        sampling_rate=SAMPLE_RATE,
        chunk_length_s=chunk_length_s,
    )
    aligned = align_words(
        segments=segments,
        wav_file=str(extract_audio_path),
        language_code=language_code,
        device=device,
        align_model_name=align_model_name,
        align_model_dir=align_model_dir,
        align_model_cache_only=(
            bool(int(os.getenv("COLAB_FILES_ALIGN_MODEL_CACHE_ONLY", "0")))
            if align_model_cache_only is None
            else align_model_cache_only
        ),
    )
    aligned = ensure_word_segments(aligned)
    write_json(transcript_path, aligned)

    total_duration = vad["media_duration"](str(input_path))
    raw_speech = vad["detect_speech_segments"](str(extract_audio_path), threshold=vad_threshold)
    if not raw_speech:
        shutil.copy2(input_path, final_video_path)
        return [
            Artifact(name="transcript", filename=transcript_path.name, kind="json"),
            Artifact(name="final_video", filename=final_video_path.name, kind="video"),
        ]

    speech_segments = vad["expand_and_merge_segments"](
        raw_speech,
        total_duration=total_duration,
        pad_before_ms=pad_before_ms,
        pad_after_ms=pad_after_ms,
        merge_gap_ms=merge_gap_ms,
    )

    output_segments = vad["build_output_segments"](
        speech_segments=speech_segments,
        total_duration=total_duration,
        leading_keep_ms=leading_keep_ms,
        trailing_keep_ms=trailing_keep_ms,
        preserve_short_pause_ms=preserve_short_pause_ms,
        long_pause_step_ms=long_pause_step_ms,
        long_pause_step_every_ms=long_pause_step_every_ms,
        max_keep_silence_ms=max_keep_silence_ms,
    )

    if filler_words:
        filler_cut_segments = vad["load_filler_segments"](
            transcript_path=str(transcript_path),
            filler_words=filler_words,
            filler_pad_before_ms=filler_pad_before_ms,
            filler_pad_after_ms=filler_pad_after_ms,
        )
        output_segments = vad["subtract_cut_segments"](output_segments, filler_cut_segments)

    vad["render_segments"](
        str(input_path),
        str(final_video_path),
        output_segments,
        render_chunk_segments=120,
        render_chunk_duration_s=300.0,
        video_preset=video_preset,
        video_crf=video_crf,
        audio_bitrate=audio_bitrate,
    )

    return [
        Artifact(name="transcript", filename=transcript_path.name, kind="json"),
        Artifact(name="final_video", filename=final_video_path.name, kind="video"),
    ]
