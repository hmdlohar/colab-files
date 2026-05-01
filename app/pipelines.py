from __future__ import annotations

import json
import os
import shutil
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import soundfile as sf

from app.store import Artifact, JobStore


SAMPLE_RATE = 16000
_VOXCPM_MODEL = None
_VOXCPM_SAMPLE_RATE = None
_VOXCPM_MODEL_KEY = None
_VOXCPM_LOCK = threading.Lock()


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


def _voxcpm():
    from voxcpm import VoxCPM

    return VoxCPM


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


def transcript_text(aligned_result: Dict) -> str:
    texts: List[str] = []
    for segment in aligned_result.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if text:
            texts.append(text)
    return " ".join(texts).strip()


def _ensure_voxcpm_model(model_name: str, load_denoiser: bool):
    global _VOXCPM_MODEL, _VOXCPM_SAMPLE_RATE, _VOXCPM_MODEL_KEY

    with _VOXCPM_LOCK:
        model_key = (model_name, bool(load_denoiser))
        if _VOXCPM_MODEL is not None and _VOXCPM_MODEL_KEY == model_key:
            return _VOXCPM_MODEL, _VOXCPM_SAMPLE_RATE

        import torch
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.disable = True

        VoxCPM = _voxcpm()
        model = VoxCPM.from_pretrained(
            model_name,
            load_denoiser=load_denoiser,
            optimize=False,
        )
        _VOXCPM_MODEL = model
        _VOXCPM_SAMPLE_RATE = int(model.tts_model.sample_rate)
        _VOXCPM_MODEL_KEY = model_key
        return _VOXCPM_MODEL, _VOXCPM_SAMPLE_RATE


def _coerce_audio_output(result) -> np.ndarray:
    if isinstance(result, np.ndarray):
        return result

    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, dict)):
        chunks = []
        for chunk in result:
            arr = np.asarray(chunk)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            chunks.append(arr)
        if not chunks:
            raise ValueError("VoxCPM returned no audio")
        return np.concatenate(chunks)

    return np.asarray(result)


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


def process_voxcpm_ultimate_clone(
    store: JobStore,
    job_id: str,
    reference_audio_path: Path,
    *,
    text: str,
    transcript: str,
    model_name: str,
    whisper_model_name: str,
    language_code: str,
    chunk_length_s: int,
    cfg_value: float,
    inference_timesteps: int,
    normalize_text: bool,
    denoise_reference: bool,
    seed: int | None,
    device: str,
) -> List[Artifact]:
    if not text.strip():
        raise ValueError("Target text is required")

    align_words, convert_to_mono_16k_wav, transcribe_chunks = _wishperx()
    job_dir = store.job_dir(job_id)
    normalized_reference_path = job_dir / "reference_16k_mono.wav"
    output_audio_path = job_dir / "voxcpm_ultimate_clone.wav"
    prompt_transcript_path = job_dir / "prompt_transcript.txt"
    prompt_transcript_json_path = job_dir / "prompt_transcript.json"

    convert_to_mono_16k_wav(str(reference_audio_path), str(normalized_reference_path), SAMPLE_RATE)

    prompt_text = transcript.strip()
    artifacts = [
        Artifact(name="generated_audio", filename=output_audio_path.name, kind="audio"),
        Artifact(name="prompt_transcript", filename=prompt_transcript_path.name, kind="text"),
        Artifact(name="normalized_reference", filename=normalized_reference_path.name, kind="audio"),
    ]

    if not prompt_text:
        segments = transcribe_chunks(
            wav_file=str(normalized_reference_path),
            model_name=whisper_model_name,
            device=device,
            sampling_rate=SAMPLE_RATE,
            chunk_length_s=chunk_length_s,
        )
        aligned = align_words(
            segments=segments,
            wav_file=str(normalized_reference_path),
            language_code=language_code,
            device=device,
            align_model_cache_only=bool(int(os.getenv("COLAB_FILES_ALIGN_MODEL_CACHE_ONLY", "0"))),
        )
        aligned = ensure_word_segments(aligned)
        prompt_text = transcript_text(aligned)
        if not prompt_text:
            raise ValueError("WhisperX could not extract transcript text from the reference audio")
        write_json(prompt_transcript_json_path, aligned)
        artifacts.append(Artifact(name="prompt_transcript_json", filename=prompt_transcript_json_path.name, kind="json"))

    prompt_transcript_path.write_text(prompt_text, encoding="utf-8")

    model, output_sample_rate = _ensure_voxcpm_model(model_name=model_name, load_denoiser=True)

    if seed is not None and seed >= 0:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    result = model.generate(
        text=text.strip(),
        prompt_wav_path=str(normalized_reference_path),
        prompt_text=prompt_text,
        reference_wav_path=str(normalized_reference_path),
        cfg_value=cfg_value,
        inference_timesteps=int(inference_timesteps),
        normalize=normalize_text,
        denoise=denoise_reference,
    )
    wav = _coerce_audio_output(result)
    sf.write(str(output_audio_path), wav, output_sample_rate)

    return artifacts
