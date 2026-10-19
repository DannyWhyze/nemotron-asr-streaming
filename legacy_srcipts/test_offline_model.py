"""Direct offline transcription test using AutoModelForRNNT and AutoProcessor."""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForRNNT, AutoProcessor
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


audio_path = resolve_audio("bcn_weather.mp3")

print(f"Loading processor and model from: {MODEL_PATH}")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForRNNT.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.float16,
)

print(f"Loading audio file: {audio_path}")
audio = load_audio(
    audio_path,
    sampling_rate=processor.feature_extractor.sampling_rate,
)

# Mode 1: Explicit language conditioning (en-US, clean transcript)
print("\n--- Mode 1: Known Language ('en-US', skip_special_tokens=True) ---")
start = time.perf_counter()
inputs = processor(audio, sampling_rate=processor.feature_extractor.sampling_rate, language="en-US")
inputs.to(model.device, dtype=model.dtype)
output = model.generate(**inputs, return_dict_in_generate=True)
text_en = processor.decode(output.sequences[0], skip_special_tokens=True)
dur_en = time.perf_counter() - start
print(f"Text:     {text_en}")
print(f"Duration: {dur_en:.2f}s")

# Mode 2: Auto language detection (keeps <xx-XX> language tag)
print("\n--- Mode 2: Auto Language Detection ('auto', skip_special_tokens=False) ---")
start = time.perf_counter()
inputs_auto = processor(audio, sampling_rate=processor.feature_extractor.sampling_rate, language="auto")
inputs_auto.to(model.device, dtype=model.dtype)
output_auto = model.generate(**inputs_auto, return_dict_in_generate=True)
text_auto = processor.decode(output_auto.sequences[0], skip_special_tokens=False)
dur_auto = time.perf_counter() - start
print(f"Text:     {text_auto}")
print(f"Duration: {dur_auto:.2f}s")
