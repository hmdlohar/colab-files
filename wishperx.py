import argparse
import gc
import json
import math
import os
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import whisperx
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def convert_to_mono_16k_wav(audio_file, wav_file, sampling_rate):
    if os.path.exists(wav_file):
        return

    waveform, sr = torchaudio.load(audio_file)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=sampling_rate,
        )
        waveform = resampler(waveform)

    torchaudio.save(wav_file, waveform, sampling_rate)

    del waveform
    clear_memory()


def transcribe_chunks(wav_file, model_name, device, sampling_rate, chunk_length_s):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

    if device == "cuda":
        model = model.half()

    model.eval()

    info = sf.info(wav_file)
    total_frames = info.frames
    chunk_frames = chunk_length_s * sampling_rate
    num_chunks = math.ceil(total_frames / chunk_frames)
    segments = []

    print(f"Processing {num_chunks} chunks (~{chunk_length_s}s each)...\n")

    with sf.SoundFile(wav_file) as f:
        for i in tqdm(range(num_chunks)):
            start_frame = i * chunk_frames
            f.seek(start_frame)

            chunk = f.read(
                frames=chunk_frames,
                dtype="float32",
                always_2d=False,
            )

            if chunk.size == 0:
                continue

            inputs = processor(
                chunk,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )

            input_features = inputs.input_features.to(device)
            if device == "cuda":
                input_features = input_features.half()

            with torch.no_grad():
                predicted_ids = model.generate(input_features)

            text = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0].strip()

            start_time = start_frame / sampling_rate
            end_time = min(start_frame + len(chunk), total_frames) / sampling_rate

            segments.append(
                {
                    "start": float(start_time),
                    "end": float(end_time),
                    "text": text,
                }
            )

            del chunk, inputs, input_features, predicted_ids
            clear_memory()

    del model, processor
    clear_memory()

    return segments


def align_words(segments, wav_file, language_code, device):
    align_model, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=device,
    )

    print("\nRunning alignment...")
    aligned_result = whisperx.align(
        segments,
        align_model,
        metadata,
        wav_file,
        device,
    )

    del align_model
    clear_memory()

    return aligned_result


def print_word_timestamps(aligned_result):
    for seg in aligned_result["segments"]:
        for word in seg.get("words", []):
            if "start" in word and "end" in word:
                print(
                    f"Word: {word['word']} | "
                    f"Start: {word['start']:.2f}s | "
                    f"End: {word['end']:.2f}s"
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe Hindi audio with Whisper and WhisperX word alignment."
    )
    parser.add_argument("audio_file", help="Input audio file, for example audio.mp3")
    parser.add_argument(
        "--output",
        default="transcript.json",
        help="Output JSON file path. Defaults to transcript.json",
    )
    parser.add_argument(
        "--wav-file",
        default=None,
        help="Intermediate mono 16 kHz WAV path. Defaults to <audio_stem>_16k_mono.wav",
    )
    parser.add_argument(
        "--model",
        default="collabora/whisper-base-hindi",
        help="Whisper model name or path.",
    )
    parser.add_argument(
        "--language-code",
        default="hi",
        help="Alignment language code. Defaults to hi.",
    )
    parser.add_argument(
        "--chunk-length-s",
        type=int,
        default=30,
        help="Chunk length in seconds. Defaults to 30.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    audio_file = args.audio_file
    sampling_rate = 16000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    wav_file = args.wav_file
    if wav_file is None:
        audio_path = Path(audio_file)
        wav_file = str(audio_path.with_name(f"{audio_path.stem}_16k_mono.wav"))

    convert_to_mono_16k_wav(audio_file, wav_file, sampling_rate)

    segments = transcribe_chunks(
        wav_file=wav_file,
        model_name=args.model,
        device=device,
        sampling_rate=sampling_rate,
        chunk_length_s=args.chunk_length_s,
    )

    aligned_result = align_words(
        segments=segments,
        wav_file=wav_file,
        language_code=args.language_code,
        device=device,
    )

    print_word_timestamps(aligned_result)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(aligned_result, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Full transcription with word-level timestamps saved to {args.output}")


if __name__ == "__main__":
    main()
