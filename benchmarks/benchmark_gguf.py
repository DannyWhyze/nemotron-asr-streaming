"""Enterprise GGUF Benchmark Suite for Nemotron 3.5 ASR Streaming 0.6B via transcribe.cpp.

Measures RTF (Real-Time Factor), TTFT (Time-to-First-Token), and Throughput (Speedup xRT)
across Batch and Chunked Streaming inference paradigms on CUDA (GPU) and AVX2 (CPU).
Exports results to data/output/benchmarks/.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = os.getenv("MODELS_DIR", str(REPO_ROOT / "models"))
OUTPUT_DIR = REPO_ROOT / "data" / "output" / "benchmarks"
AUDIO_FILES = [
    REPO_ROOT / "data" / "input" / "audio" / "bcn_weather.mp3",
    REPO_ROOT / "data" / "input" / "audio" / "obama.mp3",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enterprise GGUF Benchmark Suite for Nemotron 3.5 ASR via transcribe.cpp."
    )
    parser.add_argument(
        "--backend",
        choices=["all", "cuda", "cpu", "auto"],
        default="all",
        help="Compute backend to benchmark: 'all' (runs both CUDA & CPU), 'cuda', 'cpu', or 'auto' (default: all)",
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

    which_path = shutil.which("transcribe-cli")
    if which_path:
        return Path(which_path)

    raise FileNotFoundError("Could not find compiled 'transcribe-cli' binary.")


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

    raise FileNotFoundError(f"GGUF model not found in {DEFAULT_MODELS_DIR}.")


def ensure_16k_mono_wav(audio_path: Path) -> Path:
    """Ensure audio is 16 kHz Mono WAV format."""
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    converted_path = audio_path.with_name(f"{audio_path.stem}_16k.wav")
    if converted_path.is_file():
        return converted_path

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [ffmpeg_bin, "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(converted_path)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return converted_path

    import librosa

    y, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    sf.write(str(converted_path), y, 16000, subtype="PCM_16")
    return converted_path


def get_audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.duration)


def get_registered_devices(cli_bin: Path) -> list[dict]:
    """Query and parse device list from transcribe-cli --list-devices."""
    try:
        res = subprocess.run([str(cli_bin), "--list-devices"], capture_output=True, text=True, check=True)
        devices = []
        current = {}
        for line in res.stdout.splitlines():
            m_dev = re.match(r"^\s*\[(\d+)\]\s+(.*?)$", line)
            if m_dev:
                if current:
                    devices.append(current)
                current = {"index": int(m_dev.group(1)), "device": m_dev.group(2).strip()}
                continue
            if current:
                m_props = re.search(r"name=(\S+)\s+kind=(\S+)\s+type=(\S+)", line)
                if m_props:
                    current["name"] = m_props.group(1)
                    current["kind"] = m_props.group(2)
                    current["type"] = m_props.group(3)
                m_mem = re.search(r"memory:\s+([0-9.]+)\s+GiB total", line)
                if m_mem:
                    current["memory"] = f"{m_mem.group(1)} GiB"
        if current:
            devices.append(current)
        return devices
    except Exception:
        return []


def get_hardware_info_for_backend(devices: list[dict], backend: str) -> dict:
    """Return hardware descriptor dictionary for the chosen backend."""
    target_kind = "cuda" if backend in ("cuda", "auto") else "cpu"
    matched = next((d for d in devices if d.get("kind") == target_kind), None)

    if matched:
        return {
            "device": matched["device"],
            "backend": f"{target_kind.upper()} ({matched['name']})",
            "memory": matched.get("memory", "Unknown"),
            "engine": f"transcribe.cpp (GGML {target_kind.upper()})",
        }

    return {
        "device": "Unknown Device",
        "backend": backend.upper(),
        "memory": "Unknown",
        "engine": "transcribe.cpp (GGML)",
    }


def benchmark_batch(cli_bin: Path, model_path: Path, wav_path: Path, duration: float, backend: str = "auto") -> dict:
    """Benchmark offline batch execution on the specified backend."""
    cmd = [
        str(cli_bin),
        "-m",
        str(model_path),
        "-l",
        "en-US",
        "--backend",
        backend,
        str(wav_path),
    ]

    start_wall = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    wall_dur = time.perf_counter() - start_wall

    if proc.returncode != 0:
        raise RuntimeError(f"Batch inference failed:\n{proc.stderr}\n{proc.stdout}")

    # Extract recognized text
    text_match = re.search(r"^text:\s+(.*)$", proc.stdout, re.MULTILINE)
    transcript = text_match.group(1).strip() if text_match else ""

    # Extract internal compute time
    rt_match = re.search(r"realtime:\s+\d+x\s+\(([0-9.]+)\s+ms", proc.stdout)
    compute_time_s = float(rt_match.group(1)) / 1000.0 if rt_match else wall_dur

    rtf = compute_time_s / duration
    speedup = duration / compute_time_s if compute_time_s > 0 else 0

    return {
        "engine": f"GGUF Batch ({backend.upper()})",
        "backend": backend.upper(),
        "audio_file": wav_path.name,
        "audio_duration_s": round(duration, 2),
        "compute_time_s": round(compute_time_s, 3),
        "wall_time_s": round(wall_dur, 3),
        "rtf": round(rtf, 4),
        "speedup_xrt": round(speedup, 1),
        "ttft_ms": None,
        "model_format": "GGUF Q8_0 (751 MB)",
        "transcript": transcript,
    }


def benchmark_streaming(
    cli_bin: Path, model_path: Path, wav_path: Path, duration: float, backend: str = "auto"
) -> dict:
    """Benchmark chunked streaming execution with live TTFT tracking."""
    cmd = [
        str(cli_bin),
        "-m",
        str(model_path),
        "--backend",
        backend,
        "--stream-chunk-ms",
        "320",
        "--stream-att-right",
        "3",
        str(wav_path),
    ]

    start_wall = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    ttft_ms = None
    output_lines = []

    for line in proc.stdout:
        output_lines.append(line)
        if "partial=" in line and ttft_ms is None:
            ttft_ms = (time.perf_counter() - start_wall) * 1000.0

    proc.wait()
    wall_dur = time.perf_counter() - start_wall
    full_output = "".join(output_lines)

    if proc.returncode != 0:
        raise RuntimeError(f"Streaming inference failed:\n{full_output}")

    # Extract final text
    text_match = re.search(r"^text:\s+(.*)$", full_output, re.MULTILINE)
    transcript = text_match.group(1).strip() if text_match else ""

    # Extract internal compute time
    rt_match = re.search(r"realtime:\s+\d+x\s+\(([0-9.]+)\s+ms", full_output)
    compute_time_s = float(rt_match.group(1)) / 1000.0 if rt_match else wall_dur

    rtf = compute_time_s / duration
    speedup = duration / compute_time_s if compute_time_s > 0 else 0

    return {
        "engine": f"GGUF Streaming ({backend.upper()}, 320ms)",
        "backend": backend.upper(),
        "audio_file": wav_path.name,
        "audio_duration_s": round(duration, 2),
        "compute_time_s": round(compute_time_s, 3),
        "wall_time_s": round(wall_dur, 3),
        "rtf": round(rtf, 4),
        "speedup_xrt": round(speedup, 1),
        "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "model_format": "GGUF Q8_0 (751 MB)",
        "transcript": transcript,
    }


def export_markdown_report(results: list[dict], devices: list[dict], model_path: Path, target_file: Path) -> None:
    """Generate comprehensive Markdown benchmark report."""
    dev_descs = [f"{d['device']} ({d.get('memory', 'N/A')})" for d in devices]
    dev_str = " | ".join(dev_descs) if dev_descs else "Detected Host Hardware"

    lines = [
        "# Nemotron 3.5 ASR Streaming 0.6B - GGUF (transcribe.cpp) Benchmark Report",
        "",
        f"**Date:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Hardware Environment:** {dev_str}",
        f"**Model File:** `{model_path.name}` ({model_path.stat().st_size / (1024 * 1024):.1f} MB)",
        "**Runtime:** `transcribe.cpp` (GGML CUDA + AVX2 Multi-Backend Binary)",
        "",
        "## Performance Metrics Summary",
        "",
        "| Engine | Backend | Audio File | Audio Duration | Compute Time | RTF (Real-Time Factor) | Speedup | TTFT | Wall Clock |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        ttft_str = f"{r['ttft_ms']} ms" if r["ttft_ms"] is not None else "-"
        lines.append(
            f"| **{r['engine']}** | `{r['backend']}` | `{r['audio_file']}` | {r['audio_duration_s']} s | "
            f"{r['compute_time_s']} s | **{r['rtf']}** | **{r['speedup_xrt']}x RT** | "
            f"{ttft_str} | {r['wall_time_s']} s |"
        )

    lines.extend(
        [
            "",
            "### Metric Definitions",
            "- **Compute Time:** Pure tensor processing time reported by the GGML runtime.",
            "- **RTF (Real-Time Factor):** `Compute Time / Audio Duration`. Values < 1.0 indicate faster than real-time processing.",
            "- **Speedup (xRT):** How many seconds of audio the model can transcribe within 1 second of compute.",
            "- **TTFT (Time-to-First-Token):** Latency from stream initiation until the first decoded partial token appears.",
            "- **Wall Clock:** Total process duration from CLI invocation to termination.",
            "",
            "## Sample Transcripts",
            "",
        ]
    )

    # Show unique transcripts per engine
    for r in results:
        lines.append(f"#### {r['engine']} on `{r['audio_file']}`")
        preview = r["transcript"][:300] + ("..." if len(r["transcript"]) > 300 else "")
        lines.append(f"> {preview}")
        lines.append("")

    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    print("=" * 75)
    print("  Nemotron 3.5 ASR Streaming 0.6B - GGUF (transcribe.cpp) Benchmark")
    print("=" * 75)

    cli_bin = find_binary()
    model_path = find_gguf_model()
    devices = get_registered_devices(cli_bin)

    print(f"Engine:   transcribe.cpp -> {cli_bin}")
    for d in devices:
        print(f"Device:   [{d['index']}] {d['device']} ({d.get('memory', 'N/A')}) - kind={d.get('kind')}")
    print(f"Model:    {model_path.name} ({model_path.stat().st_size / (1024 * 1024):.1f} MB)")
    print(f"Selected: {args.backend.upper()}")
    print("=" * 75)

    backends_to_test = ["cuda", "cpu"] if args.backend == "all" else [args.backend]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for backend in backends_to_test:
        hw_desc = get_hardware_info_for_backend(devices, backend)
        print(f"\n>>> Running Benchmark Suite for Backend: {backend.upper()} ({hw_desc['device']}) <<<\n")

        for idx, raw_audio in enumerate(AUDIO_FILES, start=1):
            if not raw_audio.is_file():
                print(f"Warning: Audio file not found: {raw_audio}")
                continue

            wav_audio = ensure_16k_mono_wav(raw_audio)
            duration = get_audio_duration(wav_audio)
            print(f"[{idx}/{len(AUDIO_FILES)}] Audio: {wav_audio.name} ({duration:.2f}s) | Backend: {backend.upper()}")

            # 1. Batch
            print(f"      -> Testing GGUF Offline Batch ({backend.upper()})...")
            res_batch = benchmark_batch(cli_bin, model_path, wav_audio, duration, backend=backend)
            results.append(res_batch)
            print(
                f"         Compute: {res_batch['compute_time_s']}s | RTF: {res_batch['rtf']} "
                f"({res_batch['speedup_xrt']}x RT) | Wall: {res_batch['wall_time_s']}s"
            )

            # 2. Streaming
            print(f"      -> Testing GGUF Chunked Streaming ({backend.upper()}, 320ms chunks)...")
            res_stream = benchmark_streaming(cli_bin, model_path, wav_audio, duration, backend=backend)
            results.append(res_stream)
            ttft_str = f"{res_stream['ttft_ms']} ms" if res_stream["ttft_ms"] else "N/A"
            print(
                f"         Compute: {res_stream['compute_time_s']}s | RTF: {res_stream['rtf']} "
                f"({res_stream['speedup_xrt']}x RT) | TTFT: {ttft_str} | Wall: {res_stream['wall_time_s']}s"
            )

    # Export
    json_path = OUTPUT_DIR / "benchmark_gguf_results.json"
    md_path = OUTPUT_DIR / "benchmark_gguf_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "devices": devices,
                "model_path": str(model_path),
                "benchmarks": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    export_markdown_report(results, devices, model_path, md_path)

    print("\n" + "=" * 75)
    print("  GGUF Benchmark Complete! Reports Exported Successfully:")
    print(f"  -> JSON Data: {json_path}")
    print(f"  -> Markdown:  {md_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
