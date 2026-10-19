# Enterprise STT backend specification (transcribe.cpp and FastAPI)

> **Document version:** 1.1.0  
> **Status:** Active, production ready  
> **Scope:** Architecture, configuration, native FFI bridge, in-memory model pooling, sanitizing guards, schemas, and endpoint specifications for `server/`.

---

## 1. Executive summary

The transcribe.cpp STT server runs NVIDIA's Nemotron 3.5 ASR Streaming 0.6B model in quantized Q8_0 GGUF format via native C++ bindings (Blackwell CUDA 12.8 `sm_120` with AVX2 fallback).

### Key capabilities

1. **Resident model pool.** Hosts multiple resident `transcribe_cpp.Model` instances in GPU VRAM (default: 3 models, ~2.15 GB on an RTX 5070 Ti). Concurrent clients lease instances via an asynchronous queue (`acquire_model`), resolving the exclusive stream-lease bottleneck without subprocess overhead or model reloading.
2. **Asynchronous producer-consumer streaming.** Decouples WebSocket packet ingestion from GPU inference using an internal `asyncio.Queue(maxsize=10)` with automatic batch-draining under load, lookahead Conformer attention tuning (`att_context_right=3`), and instant cancellation handling.
3. **Hardened audio sanitizing.** Intercepts incoming audio chunks before C++ FFI invocation, enforcing size limits (max 1 MB), float32 byte-alignment (`len % 4 == 0`), and `np.isfinite()` validation to prevent state poisoning from NaN/Inf frames in Conformer recurrent caches.
4. **OpenAI Whisper compatibility.** Serves `POST /v1/audio/transcriptions` accepting standard multipart uploads and returning `json`, `text`, or `verbose_json` (with word and segment timestamps) computed in-memory.
5. **Backward compatibility.** Serves `/ws/chunked` and `/ws/stt` for discrete audio chunks.
6. **Concurrency gating.** Returns HTTP 503 (`Retry-After: 1`) on REST endpoints and WebSocket `StreamBusyEvent` (code 4029) when all pool instances are busy.

---

## 2. Component architecture and data flow

```
                           ┌────────────────────────────────────────┐
                           │              HTTP / WS Clients         │
                           │  - Legacy VAD Client (/ws/stt)         │
                           │  - Live Streaming Client (/ws/stream)  │
                           │  - OpenAI REST Client (/v1/audio/...)  │
                           └──────────────────┬─────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             FastAPI Application (server/)                        │
│                                                                                  │
│  ┌─────────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐  │
│  │      server/config.py   │  │  server/sanitizer.py │  │  server/schemas.py  │  │
│  │  - ServerConfig         │  │  - NaN/Inf filtering │  │  - Pydantic V2 DTOs │  │
│  │  - Native DLL & CUDA    │  │  - 4-Byte Alignment  │  │  - StreamBusyEvent  │  │
│  │    Bootstrap            │  │  - Buffer Size Caps  │  │  - Stream Events    │  │
│  └─────────────────────────┘  └──────────────────────┘  └─────────────────────┘  │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                            server/routes/                                   │ │
│  │   chunked_ws.py  │  streaming_ws.py (Producer/Consumer) │  rest_api.py       │ │
│  └──────────────────────────────────────┬──────────────────────────────────────┘ │
│                                         │                                        │
│                                         ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                       server/engine.py (ModelPoolEngine)                    │ │
│  │  - Resident Model Pool (asyncio.Queue[transcribe_cpp.Model])                │ │
│  │  - In-Memory Non-blocking Inference via ThreadPoolExecutor                 │ │
│  │  - acquire_model() Context Manager with Concurrency Gating                  │ │
│  └──────────────────────────────────────┬──────────────────────────────────────┘ │
│                                         │                                        │
│                                         ▼ In-Memory Native C ABI (ctypes FFI)    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                transcribe_cpp/build/bin/Release/transcribe.dll              │ │
│  │  - GGML CUDA 12.8 (sm_120, NVIDIA GeForce RTX 5070 Ti, 16 GB VRAM)          │ │
│  │  - Model: nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf (716.3 MB resident)     │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Configuration (server/config.py)

Configuration is managed through `pydantic_settings.BaseSettings` and reads from environment variables and `.env`.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `host` | `str` | `"127.0.0.1"` | Bind host IP address |
| `port` | `int` | `8001` | Bind TCP port |
| `models_dir` | `str` | `os.getenv("MODELS_DIR", "models")` | Root storage directory for models |
| `model_folder` | `str` | `"handy-computer--nemotron-3.5-asr-streaming-0.6b-gguf"` | Model directory name |
| `model_file` | `str` | `"nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"` | Target GGUF quant filename |
| `backend` | `Literal["auto", "cuda", "cpu"]` | `"auto"` | Compute backend selection |
| `max_workers` | `int` | `2` | Worker pool concurrency base |
| `default_language` | `str` | `"en-US"` | BCP-47 fallback language locale |
| `min_segment_duration_s` | `float` | `0.3` | Minimum audio length in seconds to trigger inference |
| `log_level` | `str` | `"INFO"` | Standard logging verbosity |
| `model_pool_size` | `int` | `3` | Number of resident model instances in VRAM |
| `max_chunk_size_bytes` | `int` | `1048576` (1 MB) | Maximum permitted incoming audio frame size |
| `max_queue_size` | `int` | `10` | Maximum queue depth for async audio buffering |

### Native DLL and CUDA setup (`setup_transcribe_environment`)
Executed automatically upon loading `server.config`:
1. Locates `transcribe.dll` in `transcribe_cpp/build/bin/Release/` and registers `TRANSCRIBE_LIBRARY`.
2. Adds `transcribe.dll` directory and CUDA 12.8 binaries (`D:\Programs\cuda128\bin`, `CUDA_PATH`) via `os.add_dll_directory()`.
3. Inserts `transcribe_cpp/bindings/python/src` into `sys.path`.

---

## 4. In-memory engine architecture (server/engine.py)

The `ModelPoolEngine` manages multi-instance model lifecycles, VRAM allocation, and non-blocking in-memory execution.

### 4.1 Pool initialization and warmup
1. Probes GPU capabilities via `torch.cuda` (`NVIDIA GeForce RTX 5070 Ti`, 16,302 MB total VRAM).
2. Instantiates $N$ (`model_pool_size=3`) independent `transcribe_cpp.Model` objects.
3. Performs a silent 320ms warmup inference pass on every instance to prime CUDA graphs and allocate tensor scratchpads.
4. Populates an `asyncio.Queue` acting as a token lease pool.

### 4.2 Pool leasing context manager
```python
async with engine.acquire_model(timeout=2.0) as model:
    # Exclusive access to this resident model instance
    session = model.session()
    ...
```
If all models are leased for longer than `timeout`, it raises `ModelPoolExhaustedError`.

### 4.3 In-memory inference methods
* **`transcribe_pcm(pcm_bytes, language, model)`** Converts raw float32 bytes into a NumPy buffer and passes directly to `session.run()`, returning `(text, duration_s)` without disk I/O.
* **`transcribe_audio_file(audio_path, language, model)`** Reads audio via `soundfile`, downmixes stereo to mono, resamples to 16 kHz using `soxr` if needed, and executes `session.run(..., timestamps="word")`, returning word- and segment-level alignments.

### 4.4 Clean teardown
During server shutdown, `engine.shutdown()` invokes `model.close()` on every pooled instance and flushes the thread pool executor to prevent memory leaks and Windows access violations.

---

## 5. Security and audio sanitizing (server/sanitizer.py)

Guards Conformer streaming caches against poisoning and DoS:
* **Size limit.** Rejects chunks exceeding 1 MB (`max_chunk_size_bytes`).
* **Alignment.** Enforces `len(chunk_bytes) % 4 == 0` for 32-bit floating-point audio.
* **Finite check.** Converts bytes to `np.float32` and asserts `np.isfinite(arr).all()`. Drops any chunk with `NaN`, `+Inf`, `-Inf`, or denormals before passing to C++.

---

## 6. API protocols and schemas

### 6.1 Telemetry (`GET /health`)
```json
{
  "status": "healthy",
  "uptime_seconds": 12.4,
  "backend": "CUDA",
  "device": "NVIDIA GeForce RTX 5070 Ti",
  "vram_used_mb": 2148.9,
  "vram_total_mb": 16302.4,
  "active_sessions": 0,
  "pool_size": 3,
  "available_models": 3
}
```

### 6.2 OpenAI-compatible REST API (`POST /v1/audio/transcriptions`)
* Multipart upload (`file`, `model`, `language`, `response_format`).
* Uses pooled model lease. If pool is exhausted, returns HTTP 503 Service Unavailable with header `Retry-After: 1`.

### 6.3 Streaming WebSocket (`/ws/stream`)
1. **Handshake.** Client sends `{"action": "start", "language": "en-US", "chunk_ms": 320, "lookahead": 3}`.
2. **Pool lease.** Server leases a free model. If full, returns `StreamBusyEvent` and closes with WebSocket code 4029:
   ```json
   {"event": "busy", "code": 4029, "message": "Model pool exhausted; server is busy. Please retry shortly."}
   ```
3. **Producer loop.** Receives binary chunks, sanitizes via `sanitize_pcm_chunk`, pushes to `audio_queue`. Drops oldest chunk if queue is full.
4. **Consumer loop.** Drains batches from queue, feeds `stream.feed()`, sends `StreamPartialEvent`:
   ```json
   {"event": "partial", "committed": "Yesterday it was", "tentative": "thirty", "latency_ms": 38.5}
   ```
5. **Stop action.** Client sends `{"action": "stop"}`. Server flushes remaining audio, finalizes Conformer state, and returns `StreamFinalEvent`:
   ```json
   {"event": "final", "text": "Yesterday it was thirty five degrees in Barcelona...", "duration_s": 11.04, "rtf": 0.126}
   ```
6. **Cancellation.** On disconnect, calls `session.cancel()`, drains queue, and returns model to pool within < 5ms.

---

## 7. Verification and test matrix

The test suite is executed via `pytest -v`:

| Test Module | Coverage Scope | Count | Result |
| :--- | :--- | :---: | :---: |
| `tests/test_native_bindings.py` | Direct FFI loading, CUDA device priming, Parakeet stream options | 1 | **PASS** |
| `tests/test_sanitizer.py` | Valid chunks, empty chunks, oversized, unaligned, NaN/Inf poison filters | 6 | **PASS** |
| `tests/test_model_pool.py` | Async pool concurrency, multi-lease, exhaustion timeout, recovery | 1 | **PASS** |
| `tests/test_streaming_ws.py` | Live lifecycle, 3-client concurrency, 4th client 4029 rejection, 15-chunk benchmark (<100ms, RTF <0.35), poison filtering | 4 | **PASS** |
| `tests/test_chunked_ws.py` | `/ws/chunked` and `/ws/stt` binary PCM frame processing with pool leasing | 2 | **PASS** |
| `tests/test_rest_and_health.py` | `GET /health`, `GET /v1/models`, `POST /v1/audio/transcriptions` (json and verbose) | 4 | **PASS** |
| `tests/test_config_schemas.py` | Default configuration, model path resolution, schema serialization | 2 | **PASS** |
| `tests/test_engine.py` | In-memory telemetry, PCM inference, WAV parsing | 3 | **PASS** |
| `tests/test_server_app.py` | FastAPI lifespan startup and shutdown | 1 | **PASS** |
| `tests/test_server_client.py` | End-to-end multi-endpoint pipeline verification | 1 | **PASS** |
| **Total Suite** | **Full end-to-end integration** | **25** | **25 / 25 PASS (100%)** |
