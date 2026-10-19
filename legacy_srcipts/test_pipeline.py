import os
import sys
import time
from pathlib import Path

import torch
from transformers import pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
_candidates = [
    Path(os.getenv("MODELS_DIR", "models")) / "nvidia--nemotron-3.5-asr-streaming-0.6b",
    REPO_ROOT / "models" / "nvidia--nemotron-3.5-asr-streaming-0.6b",
]
MODEL_PATH = next((str(c) for c in _candidates if Path(c).is_dir()), "nvidia/nemotron-3.5-asr-streaming-0.6b")


def resolve_audio(default_name: str) -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    # Check CWD, repo root, and audio folder
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
    return "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"


audio_source = resolve_audio("bcn_weather.mp3")

print(f"Loading model from: {MODEL_PATH}")
pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH,
    device="cuda",
    dtype=torch.float16,
)

print(f"Transcribing audio: {audio_source}")

start = time.perf_counter()
result = pipe(audio_source)
duration = time.perf_counter() - start

print("\n--- Transcription Result ---")
print(f"Text:     {result['text']}")
print(f"Duration: {duration:.2f}s on RTX 5070 Ti")
