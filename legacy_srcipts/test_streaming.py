"""Streaming transcription test for Nemotron 3.5 ASR using AutoModelForRNNT and TextIteratorStreamer."""

import sys
import time
from pathlib import Path
from threading import Thread

import torch
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer
from transformers.audio_utils import load_audio

MODEL_PATH = "F:/KI_Model/Huggingface/nvidia--nemotron-3.5-asr-streaming-0.6b"


def resolve_audio(default_name: str) -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(default_name),
        repo_root / default_name,
        repo_root / "data" / "input" / "audio" / default_name,
        Path(__file__).resolve().parent / default_name,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return default_name


audio_path = resolve_audio("obama.mp3")

print(f"Loading processor and model from: {MODEL_PATH}")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForRNNT.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.float16,
)

# Lookahead tokens setting: 6 gives balanced streaming accuracy/latency
processor.set_num_lookahead_tokens(6)
print(f"Streaming latency: {processor.streaming_latency_ms:.1f} ms")

language = "en-US"

print(f"Loading audio file: {audio_path}")
sampling_rate = processor.feature_extractor.sampling_rate
audio = load_audio(
    audio_path,
    sampling_rate=sampling_rate,
)

first_chunk_inputs = processor(
    audio[: processor.num_samples_first_audio_chunk],
    sampling_rate=sampling_rate,
    is_streaming=True,
    is_first_audio_chunk=True,
    language=language,
    return_tensors="pt",
)
first_chunk_inputs = first_chunk_inputs.to(model.device, dtype=model.dtype)


def input_features_generator():
    yield first_chunk_inputs.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]

    mel_frame_idx = processor.num_mel_frames_first_audio_chunk
    hop_length = processor.feature_extractor.hop_length
    n_fft = processor.feature_extractor.n_fft

    start_idx = mel_frame_idx * hop_length - n_fft // 2
    while (end_idx := start_idx + processor.num_samples_per_audio_chunk) < audio.shape[0]:
        inputs = processor(
            audio[start_idx:end_idx],
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=False,
            language=language,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device, dtype=model.dtype)
        yield inputs.input_features

        mel_frame_idx += processor.num_mel_frames_per_audio_chunk
        start_idx = mel_frame_idx * hop_length - n_fft // 2


streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
generate_kwargs = {
    **first_chunk_inputs,
    "input_features": input_features_generator(),
    "streamer": streamer,
}

print("\n--- Live Streaming Transcription ---")
print("Model output: ", end="", flush=True)

start_time = time.perf_counter()
thread = Thread(target=model.generate, kwargs=generate_kwargs)
thread.start()

for text_chunk in streamer:
    print(text_chunk, end="", flush=True)

thread.join()
total_time = time.perf_counter() - start_time
print(f"\n\nTotal duration: {total_time:.2f}s on RTX 5070 Ti")
