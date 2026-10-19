"""Enterprise ASR Benchmark Suite for Nemotron 3.5 ASR Streaming 0.6B.

Measures RTF (Real-Time Factor), TTFT (Time-to-First-Token), Peak VRAM, and Throughput
across Pipeline, Direct Offline RNN-T, and Chunked Streaming paradigms.
Exports results to data/output/benchmarks/.
"""

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread

import soundfile as sf
import torch
from transformers import (
    AutoModelForRNNT,
    AutoProcessor,
    TextIteratorStreamer,
    pipeline,
)
from transformers.audio_utils import load_audio

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_model_path() -> str:
    candidates = [
        Path(os.getenv("MODELS_DIR", "models")) / "nvidia--nemotron-3.5-asr-streaming-0.6b",
        REPO_ROOT / "models" / "nvidia--nemotron-3.5-asr-streaming-0.6b",
    ]
    for c in candidates:
        if c.is_dir():
            return str(c.resolve())
    return "nvidia/nemotron-3.5-asr-streaming-0.6b"


MODEL_PATH = resolve_model_path()
OUTPUT_DIR = REPO_ROOT / "data" / "output" / "benchmarks"
AUDIO_FILES = [
    REPO_ROOT / "data" / "input" / "audio" / "bcn_weather.mp3",
    REPO_ROOT / "data" / "input" / "audio" / "obama.mp3",
]


def get_gpu_info() -> dict:
    if torch.cuda.is_available():
        return {
            "device": torch.cuda.get_device_name(0),
            "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
        }
    return {"device": "CPU", "total_vram_gb": 0, "cuda_version": "None", "torch_version": torch.__version__}


def get_audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.duration)


def benchmark_pipeline(model_path: str, audio_path: Path, audio_duration: float) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device="cuda",
        dtype=torch.float16,
    )

    start = time.perf_counter()
    result = pipe(str(audio_path))
    elapsed = time.perf_counter() - start

    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    rtf = elapsed / audio_duration
    speedup = audio_duration / elapsed if elapsed > 0 else 0

    return {
        "engine": "Pipeline (HuggingFace)",
        "audio_file": audio_path.name,
        "audio_duration_s": round(audio_duration, 2),
        "inference_time_s": round(elapsed, 3),
        "rtf": round(rtf, 4),
        "speedup_xrt": round(speedup, 1),
        "ttft_ms": None,
        "peak_vram_mb": round(peak_vram, 1),
        "transcript": result["text"].strip(),
    }


def benchmark_offline_rnnt(processor, model, audio_path: Path, audio_duration: float) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    sampling_rate = processor.feature_extractor.sampling_rate
    audio = load_audio(str(audio_path), sampling_rate=sampling_rate)

    start = time.perf_counter()
    inputs = processor(audio, sampling_rate=sampling_rate, language="auto")
    inputs.to(model.device, dtype=model.dtype)
    output = model.generate(**inputs, return_dict_in_generate=True)
    text = processor.decode(output.sequences[0], skip_special_tokens=True).strip()
    elapsed = time.perf_counter() - start

    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    rtf = elapsed / audio_duration
    speedup = audio_duration / elapsed if elapsed > 0 else 0

    return {
        "engine": "Direct RNN-T (AutoModelForRNNT)",
        "audio_file": audio_path.name,
        "audio_duration_s": round(audio_duration, 2),
        "inference_time_s": round(elapsed, 3),
        "rtf": round(rtf, 4),
        "speedup_xrt": round(speedup, 1),
        "ttft_ms": None,
        "peak_vram_mb": round(peak_vram, 1),
        "transcript": text,
    }


def benchmark_streaming(processor, model, audio_path: Path, audio_duration: float, lookahead: int = 6) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    processor.set_num_lookahead_tokens(lookahead)
    sampling_rate = processor.feature_extractor.sampling_rate
    audio = load_audio(str(audio_path), sampling_rate=sampling_rate)

    first_chunk_inputs = processor(
        audio[: processor.num_samples_first_audio_chunk],
        sampling_rate=sampling_rate,
        is_streaming=True,
        is_first_audio_chunk=True,
        language="en-US",
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    def input_generator():
        yield first_chunk_inputs.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]
        mel_idx = processor.num_mel_frames_first_audio_chunk
        hop = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft

        start_idx = mel_idx * hop - n_fft // 2
        while (end_idx := start_idx + processor.num_samples_per_audio_chunk) < audio.shape[0]:
            chunk_inputs = processor(
                audio[start_idx:end_idx],
                sampling_rate=sampling_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                language="en-US",
                return_tensors="pt",
            ).to(model.device, dtype=model.dtype)
            yield chunk_inputs.input_features
            mel_idx += processor.num_mel_frames_per_audio_chunk
            start_idx = mel_idx * hop - n_fft // 2

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
    generate_kwargs = {
        **first_chunk_inputs,
        "input_features": input_generator(),
        "streamer": streamer,
    }

    start = time.perf_counter()
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    full_text_chunks = []
    ttft_ms = None

    for chunk in streamer:
        if ttft_ms is None and chunk.strip():
            ttft_ms = (time.perf_counter() - start) * 1000.0
        full_text_chunks.append(chunk)

    thread.join()
    elapsed = time.perf_counter() - start

    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    rtf = elapsed / audio_duration
    speedup = audio_duration / elapsed if elapsed > 0 else 0

    return {
        "engine": f"Chunked Streaming (lookahead={lookahead})",
        "audio_file": audio_path.name,
        "audio_duration_s": round(audio_duration, 2),
        "inference_time_s": round(elapsed, 3),
        "rtf": round(rtf, 4),
        "speedup_xrt": round(speedup, 1),
        "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "peak_vram_mb": round(peak_vram, 1),
        "transcript": "".join(full_text_chunks).strip(),
    }


def export_markdown_report(results: list[dict], gpu_info: dict, target_file: Path) -> None:
    lines = [
        "# Nemotron 3.5 ASR Streaming 0.6B - Benchmark Report",
        "",
        f"**Date:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**GPU:** {gpu_info.get('device')} ({gpu_info.get('total_vram_gb')} GB VRAM)",
        f"**PyTorch:** {gpu_info.get('torch_version')} | **CUDA:** {gpu_info.get('cuda_version')}",
        f"**Model Path:** `{MODEL_PATH}`",
        "",
        "## Performance Metrics Summary",
        "",
        "| Engine | Audio File | Audio Duration | Inference Time | RTF (Real-Time Factor) | Speedup | TTFT | Peak VRAM |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        ttft_str = f"{r['ttft_ms']} ms" if r["ttft_ms"] is not None else "-"
        lines.append(
            f"| **{r['engine']}** | `{r['audio_file']}` | {r['audio_duration_s']} s | "
            f"{r['inference_time_s']} s | **{r['rtf']}** | **{r['speedup_xrt']}x RT** | "
            f"{ttft_str} | {r['peak_vram_mb']} MB |"
        )

    lines.extend(
        [
            "",
            "### Metric Definitions",
            "- **RTF (Real-Time Factor):** `Inference Time / Audio Duration`. Values < 1.0 indicate faster than real-time processing.",
            "- **Speedup (xRT):** How many seconds of audio the model can transcribe within 1 second of compute.",
            "- **TTFT (Time-to-First-Token):** Latency from stream initiation until the first decoded word appears.",
            "- **Peak VRAM:** Maximum GPU memory allocated during the benchmark run.",
            "",
            "## Sample Transcripts",
            "",
        ]
    )

    for r in results:
        lines.append(f"#### {r['engine']} on `{r['audio_file']}`")
        preview = r["transcript"][:300] + ("..." if len(r["transcript"]) > 300 else "")
        lines.append(f"> {preview}")
        lines.append("")

    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 75)
    print("  Nemotron 3.5 ASR Streaming 0.6B - Comprehensive Benchmark")
    print("=" * 75)

    gpu_info = get_gpu_info()
    print(f"Hardware: {gpu_info['device']} ({gpu_info['total_vram_gb']} GB VRAM)")
    print(f"Software: PyTorch {gpu_info['torch_version']} (CUDA {gpu_info['cuda_version']})")
    print(f"Model:    {MODEL_PATH}")
    print("=" * 75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Warmup
    print("\n[1/4] Performing GPU Warmup on bcn_weather.mp3...")
    warmup_audio = AUDIO_FILES[0]
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForRNNT.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.float16,
    )
    _ = benchmark_offline_rnnt(processor, model, warmup_audio, get_audio_duration(warmup_audio))
    print("      Warmup complete.")

    results = []

    # 2. Benchmark each audio file
    for idx, audio_path in enumerate(AUDIO_FILES, start=1):
        if not audio_path.is_file():
            print(f"Warning: Audio file not found: {audio_path}")
            continue

        duration = get_audio_duration(audio_path)
        print(f"\n[{idx + 1}/4] Benchmarking: {audio_path.name} (Duration: {duration:.2f}s)")

        # Engine A: Pipeline
        print("      -> Testing HuggingFace Pipeline...")
        res_pipe = benchmark_pipeline(MODEL_PATH, audio_path, duration)
        results.append(res_pipe)
        print(
            f"         Time: {res_pipe['inference_time_s']}s | RTF: {res_pipe['rtf']} ({res_pipe['speedup_xrt']}x RT) | VRAM: {res_pipe['peak_vram_mb']} MB"
        )

        # Engine B: Direct Offline RNN-T
        print("      -> Testing Direct Offline RNN-T...")
        res_offline = benchmark_offline_rnnt(processor, model, audio_path, duration)
        results.append(res_offline)
        print(
            f"         Time: {res_offline['inference_time_s']}s | RTF: {res_offline['rtf']} ({res_offline['speedup_xrt']}x RT) | VRAM: {res_offline['peak_vram_mb']} MB"
        )

        # Engine C: Chunked Streaming
        print("      -> Testing Chunked Streaming (lookahead=6)...")
        res_stream = benchmark_streaming(processor, model, audio_path, duration, lookahead=6)
        results.append(res_stream)
        ttft_str = f"{res_stream['ttft_ms']} ms" if res_stream["ttft_ms"] else "N/A"
        print(
            f"         Time: {res_stream['inference_time_s']}s | RTF: {res_stream['rtf']} ({res_stream['speedup_xrt']}x RT) | TTFT: {ttft_str} | VRAM: {res_stream['peak_vram_mb']} MB"
        )

    # 3. Export Reports
    json_path = OUTPUT_DIR / "benchmark_results.json"
    md_path = OUTPUT_DIR / "benchmark_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "hardware": gpu_info,
                "model_path": MODEL_PATH,
                "benchmarks": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    export_markdown_report(results, gpu_info, md_path)

    print("\n" + "=" * 75)
    print("  Benchmark Complete! Reports Exported Successfully:")
    print(f"  -> JSON Data: {json_path}")
    print(f"  -> Markdown:  {md_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
