"""Standalone evaluation script for Nemotron 3.5 ASR GGUF via transcribe.cpp.

Demonstrates C++/GGML standalone execution of quantized GGUF weights (Q8_0)
with automatic 16 kHz WAV preparation, path discovery, and execution
of both batch and chunked streaming inference.
"""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = os.getenv("MODELS_DIR", "F:/KI_Model/Huggingface")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone evaluation script for Nemotron 3.5 ASR GGUF via transcribe.cpp."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Path to input audio file (default: bcn_weather.mp3 in data/input/audio/)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute backend to use: 'auto', 'cuda', or 'cpu' (default: auto)",
    )
    return parser.parse_args()


def find_binary() -> Path:
    """Locate the compiled transcribe-cli binary."""
    candidates = [
        REPO_ROOT / "transcribe_cpp" / "build" / "bin" / "Release" / "transcribe-cli.exe",
        REPO_ROOT / "transcribe_cpp" / "build" / "bin" / "transcribe-cli.exe",
        REPO_ROOT / "transcribe_cpp" / "build" / "bin" / "Release" / "transcribe-cli",
        REPO_ROOT / "transcribe_cpp" / "build" / "bin" / "transcribe-cli",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Fallback to PATH
    which_path = shutil.which("transcribe-cli")
    if which_path:
        return Path(which_path)

    raise FileNotFoundError(
        "Could not find compiled 'transcribe-cli' binary. "
        "Please build transcribe_cpp first or check transcribe_cpp/build/bin/."
    )


def find_gguf_model() -> Path:
    """Locate the Q8_0 GGUF model file."""
    candidates = [
        Path(DEFAULT_MODELS_DIR)
        / "handy-computer--nemotron-3.5-asr-streaming-0.6b-gguf"
        / "nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf",
        REPO_ROOT / "models" / "nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf",
        Path("nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"),
    ]
    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        f"GGUF model 'nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf' not found in {DEFAULT_MODELS_DIR}. "
        "Run utils/download_model_files.py to acquire it."
    )


def resolve_audio(input_arg: str | None = None) -> Path:
    """Resolve input audio path from CLI argument or default test audio."""
    default_name = "bcn_weather.mp3"
    target = input_arg or default_name

    candidates = [
        Path(target),
        REPO_ROOT / target,
        REPO_ROOT / "data" / "input" / "audio" / target,
        Path(__file__).resolve().parent / target,
        REPO_ROOT / "data" / "input" / "audio" / default_name,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()

    raise FileNotFoundError(f"Audio file '{target}' could not be found.")


def ensure_16k_mono_wav(audio_path: Path) -> Path:
    """Ensure the audio file is in 16 kHz Mono PCM WAV format required by transcribe.cpp."""
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    # Check if pre-converted 16k wav exists
    converted_path = audio_path.with_name(f"{audio_path.stem}_16k.wav")
    if converted_path.is_file():
        return converted_path

    print(f"[Prep] Converting '{audio_path.name}' to 16 kHz mono WAV...")
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(audio_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(converted_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            return converted_path

    # Fallback to soundfile / librosa in Python
    try:
        import librosa
        import soundfile as sf

        y, _ = librosa.load(str(audio_path), sr=16000, mono=True)
        sf.write(str(converted_path), y, 16000, subtype="PCM_16")
        return converted_path
    except Exception as exc:
        raise RuntimeError(f"Failed to convert audio to 16 kHz WAV: {exc}") from exc


def main():
    args = parse_args()
    cli_bin = find_binary()
    model_path = find_gguf_model()

    raw_audio = resolve_audio(args.audio)
    wav_audio = ensure_16k_mono_wav(raw_audio)

    print("=" * 70)
    print(" Nemotron 3.5 ASR Streaming 0.6B - GGUF (transcribe.cpp) Test")
    print("=" * 70)
    print(f"Binary:   {cli_bin}")
    print(f"Backend:  {args.backend}")
    print(f"Model:    {model_path.name} ({model_path.stat().st_size / (1024 * 1024):.1f} MB)")
    print(f"Audio:    {wav_audio.name}")
    print("-" * 70)

    # 1. Mode: Offline Batch Inference
    print(f"\n[Mode 1] Running Offline Batch Transcription (backend: {args.backend}, language: en-US)...")
    cmd_batch = [
        str(cli_bin),
        "-m",
        str(model_path),
        "-l",
        "en-US",
        "--backend",
        args.backend,
        str(wav_audio),
    ]

    start_t = time.perf_counter()
    res_batch = subprocess.run(cmd_batch, capture_output=True, text=True, check=False)
    batch_dur = time.perf_counter() - start_t

    if res_batch.returncode == 0:
        # Extract text line and realtime line from output
        lines = res_batch.stdout.splitlines()
        for line in lines:
            if line.startswith("text:") or line.strip().startswith("realtime:"):
                print(f"  {line}")
        print(f"  -> Total process wall time: {batch_dur:.2f}s")
    else:
        print(f"Batch failed (code {res_batch.returncode}):\n{res_batch.stderr}\n{res_batch.stdout}")

    # 2. Mode: Streaming Simulation (Chunked)
    print("\n" + "-" * 70)
    print(f"[Mode 2] Running Cache-Aware Chunked Streaming (backend: {args.backend}, chunk=320ms, lookahead=240ms)...")
    cmd_stream = [
        str(cli_bin),
        "-m",
        str(model_path),
        "--backend",
        args.backend,
        "--stream-chunk-ms",
        "320",
        "--stream-att-right",
        "3",
        str(wav_audio),
    ]

    start_t = time.perf_counter()
    res_stream = subprocess.run(cmd_stream, capture_output=True, text=True, check=False)
    stream_dur = time.perf_counter() - start_t

    if res_stream.returncode == 0:
        lines = res_stream.stdout.splitlines()
        for line in lines:
            if line.startswith("text:") or line.strip().startswith("realtime:"):
                print(f"  {line}")
        print(f"  -> Total stream simulation time: {stream_dur:.2f}s")
    else:
        print(f"Streaming failed (code {res_stream.returncode}):\n{res_stream.stderr}\n{res_stream.stdout}")

    print("=" * 70)
    print(" GGUF Evaluation Completed Successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
