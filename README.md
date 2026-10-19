# Nemotron 3.5 ASR streaming 0.6B evaluation and benchmark suite

[![Architecture: FastConformer + RNNT](https://img.shields.io/badge/Architecture-FastConformer%20%2B%20RNNT-blue.svg)](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
[![Model Size: 0.6B](https://img.shields.io/badge/Parameters-600M-orange.svg)](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
[![Hardware: NVIDIA RTX 5070 Ti](https://img.shields.io/badge/Validated%20Hardware-RTX%205070%20Ti%20(Blackwell)-green.svg)](https://www.nvidia.com)
[![CUDA: 12.8](https://img.shields.io/badge/CUDA-12.8-purple.svg)](https://developer.nvidia.com/cuda-downloads)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Model License: OpenMDW-1.1](https://img.shields.io/badge/Model%20License-OpenMDW--1.1-lightgrey.svg)](https://openmdw.ai/license/1-1/)

Benchmark suite and evaluation code for NVIDIA's Nemotron 3.5 ASR Streaming 0.6B model on workstation hardware. Tests cover offline batch decoding, low-latency streaming with rolling caches, language prompt conditioning, and GGUF deployment through transcribe.cpp.

---

## 1. Architecture overview

Nemotron 3.5 ASR Streaming 0.6B is a multilingual speech-to-text model built for real-time streaming and offline batch decoding.

```
                          ┌────────────────────────────────────────────────────────┐
                          │               Incoming Audio (16 kHz PCM)              │
                          └──────────────────────────┬─────────────────────────────┘
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   Stateful Cache-Aware FastConformer Encoder                     │
│  - 80 ms Frame Stride (Downsampling factor 8)                                    │
│  - Constant-Memory Rolling Attention & Convolution State Caches                   │
│  - Tunable Right Lookahead Window (R = 0 to 13 frames: 0 ms to 1040 ms)          │
└────────────────────────────────────────────┬─────────────────────────────────────┘
                                             ▼ Acoustic Embeddings (D=512)
                          ┌──────────────────┴──────────────────┐
                          │ Language Prompt Conditioning Vector │ (K=128 One-Hot Locale)
                          └──────────────────┬──────────────────┘
                                             ▼ Concatenated Features (D+K=640)
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          Joint RNN-Transducer (RNN-T)                            │
│  - Prediction Network (Prev Tokens) + Joint Network                              │
│  - Native Punctuation, Capitalization & Timestamps (PnC)                         │
│  - In-flight Language ID Detection (`auto` mode -> appends `<lang-XX>`)          │
└────────────────────────────────────────────┬─────────────────────────────────────┘
                                             ▼
                          ┌────────────────────────────────────────────────────────┐
                          │   Streaming Text Output (Live via TextIteratorStreamer)│
                          └────────────────────────────────────────────────────────┘
```

### Architecture highlights

* **Rolling state caches.** Traditional streaming ASR recomputes overlapping audio chunks, wasting cycles on redundant forward passes. Nemotron 3.5 processes each 80 ms frame once and passes fixed-size attention and convolution caches forward. On an RTX 5070 Ti, this avoids redundant compute and supports more concurrent streams.
* **Fixed memory footprint.** Sequence memory scaling is constant. An audio stream lasting 10 seconds or 10 hours uses the same VRAM footprint.
* **Native punctuation and casing.** The model produces capitalized words, periods, and commas directly. You do not need a separate punctuation model downstream.
* **Language prompt conditioning.** Supports 40 language locales. The target language is injected as a one-hot vector into the joint network. Setting language to auto infers language during decoding and appends a trailing tag.
* **C++ portability.** Run weights through PyTorch or directly in C++ using transcribe.cpp and GGUF quantizations.

---

## 2. Hardware benchmarks

### 2.1 Performance comparison on long-form audio (obama.mp3, ~204s)
Measured on workstation hardware (NVIDIA RTX 5070 Ti Blackwell 16 GB VRAM and AMD Ryzen 9 3900X CPU):

| Rank | Inference Engine & Backend | Compute Time | RTF | Speedup | Latency (TTFT) | Memory Footprint |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **GGUF Batch (`transcribe.cpp` Q8_0 - CUDA0)** | **10.42 s** | **0.0512** | **19.5x RT** | - | **716 MB** |
| 2 | **Direct RNN-T (`AutoModelForRNNT` - PyTorch CUDA)** | 11.68 s | 0.0573 | **17.4x RT** | - | 2,241 MB |
| 3 | **Pipeline (`transformers.pipeline` - PyTorch CUDA)** | 14.18 s | 0.0696 | **14.4x RT** | - | 3,457 MB |
| 4 | **GGUF Batch (`transcribe.cpp` Q8_0 - CPU AVX2)** | 28.42 s | 0.1396 | **7.2x RT** | - | 716 MB |
| 5 | **GGUF Streaming (`transcribe.cpp` Q8_0 - CUDA0, 320ms)** | **39.01 s** | **0.1917** | **5.2x RT** | 4,092 ms | **716 MB** |
| 6 | **Chunked Streaming (`lookahead=6` - PyTorch CUDA)** | 55.33 s | 0.2715 | **3.7x RT** | **1,170 ms** | 1,275 MB |
| 7 | **GGUF Streaming (`transcribe.cpp` Q8_0 - CPU AVX2, 320ms)** | 69.67 s | 0.3423 | **2.9x RT** | 5,555 ms | 716 MB |

### 2.2 PyTorch benchmarks on NVIDIA RTX 5070 Ti (FP16)
Measured with CUDA 12.8 and PyTorch 2.11.0+cu128 using official FP16 safetensors weights (2.43 GB):

| Inference Engine | Audio Input | Audio Length | Inference Time | Real-Time Factor (RTF) | Throughput (Speedup) | Latency (TTFT) | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct RNN-T (`AutoModelForRNNT`)** | `bcn_weather.mp3` | 11.04 s | 0.629 s | **0.0570** | **17.6x RT** | - | 1,281 MB |
| **Pipeline (`transformers.pipeline`)** | `bcn_weather.mp3` | 11.04 s | 1.468 s | **0.1329** | **7.5x RT** | - | 2,499 MB |
| **Chunked Streaming (`lookahead=6`)** | `bcn_weather.mp3` | 11.04 s | 2.924 s | **0.2648** | **3.8x RT** | 1,337 ms | 1,270 MB |
| **Direct RNN-T (`AutoModelForRNNT`)** | `obama.mp3` | 203.78 s | 11.680 s | **0.0573** | **17.4x RT** | - | 2,241 MB |
| **Pipeline (`transformers.pipeline`)** | `obama.mp3` | 203.78 s | 14.179 s | **0.0696** | **14.4x RT** | - | 3,457 MB |
| **Chunked Streaming (`lookahead=6`)** | `obama.mp3` | 203.78 s | 55.327 s | **0.2715** | **3.7x RT** | 1,170 ms | **1,275 MB** |

### 2.3 GGUF benchmarks via transcribe.cpp (CUDA and CPU)
Measured with compiled transcribe.cpp executing the quantized Q8_0 GGUF model (716.3 MB, no Python dependencies). The binary supports runtime backend selection via `--backend cuda` or `--backend cpu`:

| Inference Engine | Backend / Device | Audio File | Audio Length | Compute Time | RTF | Speedup | TTFT | Wall Clock |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GGUF Batch** | `CUDA (RTX 5070 Ti)` | `bcn_weather_16k.wav` | 11.04 s | 0.603 s | **0.0546** | **18.3x RT** | - | 2.432 s |
| **GGUF Streaming (320ms)** | `CUDA (RTX 5070 Ti)` | `bcn_weather_16k.wav` | 11.04 s | 2.299 s | **0.2083** | **4.8x RT** | 4,314 ms | 4.492 s |
| **GGUF Batch** | `CUDA (RTX 5070 Ti)` | `obama_16k.wav` | 203.52 s | 10.417 s | **0.0512** | **19.5x RT** | - | 12.408 s |
| **GGUF Streaming (320ms)** | `CUDA (RTX 5070 Ti)` | `obama_16k.wav` | 203.52 s | 39.014 s | **0.1917** | **5.2x RT** | 4,092 ms | 41.176 s |
| **GGUF Batch** | `CPU (Ryzen 9 3900X)` | `bcn_weather_16k.wav` | 11.04 s | 1.444 s | **0.1308** | **7.6x RT** | - | 3.437 s |
| **GGUF Streaming (320ms)** | `CPU (Ryzen 9 3900X)` | `bcn_weather_16k.wav` | 11.04 s | 4.165 s | **0.3773** | **2.7x RT** | 6,122 ms | 6.288 s |
| **GGUF Batch** | `CPU (Ryzen 9 3900X)` | `obama_16k.wav` | 203.52 s | 28.416 s | **0.1396** | **7.2x RT** | - | 30.680 s |
| **GGUF Streaming (320ms)** | `CPU (Ryzen 9 3900X)` | `obama_16k.wav` | 203.52 s | 69.665 s | **0.3423** | **2.9x RT** | 5,555 ms | 71.897 s |

### Key findings

1. **GGUF on CUDA is fastest.** Quantized Q8_0 GGUF on the RTX 5070 Ti ran at 19.5x real-time speed on the 203.5s audio (10.42s elapsed), beating PyTorch RNN-T at 17.4x.
2. **Streaming speedup.** Streaming GGUF on CUDA hit 5.2x real-time (39.01s), about 41% faster than PyTorch streaming (55.33s, 3.7x RT).
3. **CPU performance.** On an AMD Ryzen 9 3900X, the CPU backend reached 7.2x to 7.6x real-time speed in batch mode and 2.9x in streaming, without a GPU or Python runtime.
4. **Single binary backends.** Building transcribe.cpp with `-DTRANSCRIBE_CUDA=ON` includes both CUDA and CPU backends in one binary. Select the backend at launch with `--backend cuda` or `--backend cpu`.


---

## 3. Repository layout

```
stt_nemotron_3_5_asr_streaming_0_6/
├── benchmarks/                    # Benchmark scripts
│   ├── benchmark_asr.py           # GPU PyTorch benchmark suite (RTF, TTFT, VRAM)
│   └── benchmark_gguf.py          # Standalone C++ GGUF benchmark suite (compute RTF, speedup, TTFT)
├── data/
│   ├── input/audio/               # Reference evaluation audio (.mp3, .wav)
│   └── output/benchmarks/         # Generated benchmark reports
├── docs/                          # Architecture and model documentation
│   ├── README.md                  # Model details, quantization, and context sizing
│   └── backend_specification_1.md # STT server API and architecture specification
├── legacy_srcipts/                # Prototyping scripts
│   ├── test_pipeline.py           # Hugging Face pipeline test
│   ├── test_offline_model.py      # Direct AutoModelForRNNT batch decoding and auto language detection
│   ├── test_streaming.py          # Chunked generator with TextIteratorStreamer
│   ├── test_gguf.py               # Standalone C++ GGUF inference via transcribe-cli
│   └── README.md                  # Legacy script guide
├── server/                        # FastAPI STT server
│   ├── __init__.py                # Package metadata
│   ├── config.py                  # Pydantic BaseSettings configuration
│   ├── engine.py                  # In-memory ModelPoolEngine managing C++ model instances
│   ├── sanitizer.py               # Audio validation and float32 alignment check
│   ├── schemas.py                 # REST and WebSocket data contracts
│   ├── main.py                    # FastAPI application factory and lifespan manager
│   └── routes/                    # API route controllers
│       ├── health.py              # Health check and model catalog
│       ├── chunked_ws.py          # Chunked audio WebSocket endpoint (/ws/chunked, /ws/stt)
│       ├── rest_api.py            # OpenAI Whisper compatible POST /v1/audio/transcriptions
│       └── streaming_ws.py        # Streaming WebSocket endpoint (/ws/stream)
├── tests/                         # Pytest test suite
│   ├── test_config_schemas.py     # Configuration unit tests
│   ├── test_chunked_ws.py         # Chunked WebSocket integration tests
│   ├── test_engine.py             # STTEngine integration tests
│   ├── test_model_pool.py         # Async pool leasing and exhaustion tests
│   ├── test_native_bindings.py    # Direct C++ FFI tests
│   ├── test_rest_and_health.py    # REST and health tests
│   ├── test_sanitizer.py          # PCM chunk validation tests
│   ├── test_server_app.py         # Lifespan startup and shutdown tests
│   ├── test_server_client.py      # End-to-end integration tests
│   └── test_streaming_ws.py       # Streaming WebSocket tests
├── utils/
│   ├── download_model.py          # Full repository downloader with license tracking
│   └── download_model_files.py    # Selective file downloader with audit log
├── .env.example                   # Environment configuration template
├── pyproject.toml                 # uv project definition with CUDA 12.8 wheel index
├── run_server.py                  # CLI entrypoint for STT server
└── README.md                      # Primary project documentation (this file)
```

---

## 4. Setup and installation

### Prerequisites
* **Python.** 3.12 or newer.
* **Package manager.** `uv` for environment management.
* **GPU.** NVIDIA GPU with CUDA. Blackwell architectures (`sm_120`, e.g. RTX 5070 Ti) require CUDA 12.8 or newer. CPU-only and Apple Silicon are supported for GGUF runtimes.

### 1. Clone repo and install dependencies

```bash
git clone https://github.com/your-username/stt_nemotron_3_5_asr_streaming_0_6.git
cd stt_nemotron_3_5_asr_streaming_0_6

# Install dependencies with explicit CUDA 12.8 PyTorch wheels
uv sync
```

### 2. Configure environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set storage paths in `.env`:

```ini
# Path to local directory where downloaded models are stored
MODELS_DIR=/path/to/models

# Optional Hugging Face Token (required only for restricted models)
HF_TOKEN=your_huggingface_token_here
```

### 3. Download weights

Download PyTorch weights or GGUF quantized models with the download script:

```bash
# Download PyTorch model files (safetensors and configs)
uv run python utils/download_model_files.py nvidia/nemotron-3.5-asr-streaming-0.6b "model.safetensors" "config.json" "tokenizer.json" "generation_config.json" "preprocessor_config.json"

# Download GGUF Q8_0 quant for transcribe.cpp inference
uv run python utils/download_model_files.py handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf "nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"
```

The script verifies the commit SHA and writes a `download_info.json` audit file in the destination folder.

### 4. Build transcribe.cpp (required for GGUF and server)

Build the C++ engine with CMake:

```bash
# Clone transcribe.cpp
git clone https://github.com/handy-computer/transcribe.cpp.git transcribe_cpp

# Option A: Build with CUDA acceleration (NVIDIA GPU, Blackwell sm_120)
cmake -B transcribe_cpp/build -S transcribe_cpp -DTRANSCRIBE_CUDA=ON
cmake --build transcribe_cpp/build --config Release --target transcribe-cli

# Option B: Build CPU-only (AVX2 and FMA)
cmake -B transcribe_cpp/build -S transcribe_cpp -DTRANSCRIBE_CUDA=OFF
cmake --build transcribe_cpp/build --config Release --target transcribe-cli
```

The binary builds to `transcribe_cpp/build/bin/Release/transcribe-cli.exe` (Windows) or `transcribe_cpp/build/bin/transcribe-cli` (Linux/macOS).

---

## 5. Running the scripts

### 1. Run PyTorch benchmarks
Run the evaluation across PyTorch inference engines on NVIDIA RTX 5070 Ti:

```bash
uv run python benchmarks/benchmark_asr.py
```
Benchmark outputs are saved to `data/output/benchmarks/benchmark_summary.md` and `data/output/benchmarks/benchmark_results.json`.

### 2. Run GGUF benchmarks
Benchmark the quantized Q8_0 model via transcribe.cpp across batch and streaming modes:

```bash
# Full comparative benchmark across both CUDA (GPU) and AVX2 (CPU) backends
uv run python benchmarks/benchmark_gguf.py --backend all

# Restrict to CUDA GPU backend
uv run python benchmarks/benchmark_gguf.py --backend cuda

# Restrict to CPU AVX2 backend
uv run python benchmarks/benchmark_gguf.py --backend cpu
```
Outputs are saved to `data/output/benchmarks/benchmark_gguf_summary.md` and `data/output/benchmarks/benchmark_gguf_results.json`.

### 3. Offline batch transcription
Transcribe an audio file using direct `AutoModelForRNNT` decoding with language conditioning or automatic language detection:

```bash
# Uses default test audio or pass a custom path
uv run python legacy_srcipts/test_offline_model.py path/to/sample.mp3
```

### 4. Streaming transcription in terminal
Simulate real-time streaming audio input with lookahead tokens and asynchronous terminal streaming:

```bash
uv run python legacy_srcipts/test_streaming.py path/to/sample.mp3
```

### 5. Hugging Face pipeline test
Run a quick transcription test using Hugging Face's high-level `pipeline`:

```bash
uv run python legacy_srcipts/test_pipeline.py path/to/sample.mp3
```

### 6. GGUF inference with transcribe.cpp
Run quantized Q8_0 GGUF inference on CUDA (GPU) or AVX2 (CPU) with automated 16 kHz WAV conversion:

```bash
# Automatic backend selection (GPU prioritized if compiled)
uv run python legacy_srcipts/test_gguf.py path/to/sample.mp3

# Force CUDA GPU backend
uv run python legacy_srcipts/test_gguf.py path/to/sample.mp3 --backend cuda

# Force CPU AVX2 backend
uv run python legacy_srcipts/test_gguf.py path/to/sample.mp3 --backend cpu
```

### 7. FastAPI server
Run the FastAPI server with resident model pooling, streaming WebSocket, and OpenAI-compatible REST routes:

```bash
# Start server on default 127.0.0.1:8001 using auto-detected backend (CUDA)
uv run python run_server.py

# Custom host, port, or backend (cuda / cpu)
uv run python run_server.py --host 0.0.0.0 --port 8001 --backend cuda
```

#### Available endpoints

* **`GET /health`**. Health diagnostics, active sessions, and GPU VRAM telemetry.
* **`GET /v1/models`**. OpenAI-compatible model catalog.
* **`POST /v1/audio/transcriptions`**. OpenAI Whisper API compatible endpoint supporting `json`, `text`, and `verbose_json` (timestamps for words and segments):
  ```bash
  curl -X POST "http://127.0.0.1:8001/v1/audio/transcriptions" \
    -F "file=@data/input/audio/bcn_weather_16k.wav" \
    -F "model=nemotron-3.5-asr-streaming-0.6b" \
    -F "language=en-US" \
    -F "response_format=verbose_json"
  ```
* **`WS /ws/chunked`** (and alias `/ws/stt`). Chunked WebSocket endpoint for discrete audio segments (e.g. from VAD). Accepts raw 16 kHz mono float32 PCM frames, returns transcription text strings.
* **`WS /ws/stream`**. Bidirectional streaming WebSocket:
  - Client sends handshake: `{"action": "start", "language": "en-US", "chunk_ms": 320}`
  - Server confirms: `{"event": "ready", "session_id": "..."}`
  - Client streams PCM chunks; server replies with partial transcripts: `{"event": "partial", "committed": "...", "tentative": "...", "latency_ms": 15.2}`
  - Client finalizes: `{"action": "stop"}`; server sends `{"event": "final", "text": "...", "duration_s": 11.04, "rtf": 0.05}`

---

## 6. Supported languages

Nemotron 3.5 ASR Streaming 0.6B supports 40 language locales in three tiers:

* **Tier 1 (production ready, 19 locales).**  
  English (`en-US`, `en-GB`), Spanish (`es-US`, `es-ES`), French (`fr-FR`, `fr-CA`), German (`de-DE`), Italian (`it-IT`), Portuguese (`pt-BR`, `pt-PT`), Dutch (`nl-NL`), Russian (`ru-RU`), Turkish (`tr-TR`), Arabic (`ar-AR`), Hindi (`hi-IN`), Japanese (`ja-JP`), Korean (`ko-KR`), Vietnamese (`vi-VN`), Ukrainian (`uk-UA`).
* **Tier 2 (broad coverage, 13 locales).**  
  Polish (`pl-PL`), Swedish (`sv-SE`), Czech (`cs-CZ`), Norwegian Bokmål (`nb-NO`), Danish (`da-DK`), Bulgarian (`bg-BG`), Finnish (`fi-FI`), Croatian (`hr-HR`), Slovak (`sk-SK`), Mandarin (`zh-CN`), Hungarian (`hu-HU`), Romanian (`ro-RO`), Estonian (`et-EE`).
* **Tier 3 (adaptation ready, 8 locales).**  
  Greek (`el-GR`), Hebrew (`he-IL`), Lithuanian (`lt-LT`), Slovenian (`sl-SI`), Latvian (`lv-LV`), Maltese (`mt-MT`), Thai (`th-TH`), Norwegian Nynorsk (`nn-NO`).

---

## 7. Quantization and memory footprint

For embedded, CPU, or lightweight edge deployment, the model is available in GGUF format:

| Quantization | File Name | Memory Footprint | Target Environment |
| :--- | :--- | :---: | :--- |
| **Q8_0** | `nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf` | 751 MB | Default choice (lossless accuracy) |
| **Q6_K** | `nemotron-3.5-asr-streaming-0.6b-Q6_K.gguf` | 621 MB | Balanced edge workstations |
| **Q5_K_M** | `nemotron-3.5-asr-streaming-0.6b-Q5_K_M.gguf` | 560 MB | Memory-constrained systems |
| **Q4_K_M** | `nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf` | 496 MB | Low-memory embedded devices |

Run with `transcribe.cpp`:
```bash
# GPU acceleration via CUDA (NVIDIA RTX 5070 Ti)
./transcribe-cli \
  -m models/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf \
  --backend cuda \
  --stream-chunk-ms 320 \
  --stream-att-right 3 \
  -l en-US \
  input.wav

# CPU execution via AVX2 and FMA (AMD Ryzen and Intel)
./transcribe-cli \
  -m models/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf \
  --backend cpu \
  --stream-chunk-ms 320 \
  --stream-att-right 3 \
  -l en-US \
  input.wav
```

---

## 8. Licenses

Code licenses and model weight licenses are tracked independently:

### Code
The source code, benchmark utilities, and helper scripts in this repository are licensed under the [MIT License](https://opensource.org/licenses/MIT).

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

### Model weights
The pre-trained weights (`nvidia/nemotron-3.5-asr-streaming-0.6b`) and quantized checkpoints (`handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf`) are released under the [OpenMDW-1.1 License](https://openmdw.ai/license/1-1/).

* **Commercial use.** Permitted under OpenMDW-1.1 terms.
* **Attribution.** Distributions must preserve original copyright notices and license grant from NVIDIA.
* **Provenance.** Model downloads via `utils/download_model.py` and `utils/download_model_files.py` record the repository ID, Git commit SHA, timestamp, and license in a local `download_info.json` manifest.

---

## 9. References

* **NVIDIA NeMo Framework:** [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
* **GGUF Quantization Release:** [handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf)
* **Inference Runtime:** [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
* **OpenMDW License Specification:** [https://openmdw.ai/license/1-1/](https://openmdw.ai/license/1-1/)
