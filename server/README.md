# Nemotron 3.5 ASR Enterprise STT Server (`server/`)

An enterprise-grade, asynchronous FastAPI server providing low-latency speech-to-text inference powered by NVIDIA's **Nemotron 3.5 ASR Streaming 0.6B** in quantized Q8_0 GGUF format via `transcribe.cpp` (C++ / GGML with Blackwell CUDA 12.0 `sm_120` and AVX2 CPU fallback).

---

## 1. Package Architecture & Components

The `server/` package is structured into decoupled layers separating configuration, native engine management, data contracts, and protocol routers:

```
server/
├── __init__.py                # Package version metadata
├── config.py                  # Pydantic BaseSettings configuration with environment defaults
├── engine.py                  # In-memory ModelPoolEngine & VRAM manager (resident C++ models)
├── sanitizer.py               # Audio validation pipeline (size caps, 4-byte float32 alignment, NaN/Inf filter)
├── schemas.py                 # Pydantic V2 request, response, and WebSocket event models
├── main.py                    # FastAPI application factory & lifespan context manager
├── README.md                  # Package documentation (this file)
└── routes/                    # Modular API route controllers
    ├── health.py              # GET /health & GET /v1/models diagnostics
    ├── chunked_ws.py          # WS /ws/chunked & /ws/stt discrete chunk transcription endpoint
    ├── rest_api.py            # POST /v1/audio/transcriptions (OpenAI Whisper compatible)
    └── streaming_ws.py        # WS /ws/stream bidirectional streaming endpoint (producer-consumer)
```

### Module Responsibilities

* **`config.py` (`ServerConfig`)**: Loads settings from environment variables and `.env`. Dynamically locates the Q8_0 GGUF weights across candidate paths (`F:/KI_Model/Huggingface/...`), configures backend preference (`auto`, `cuda`, `cpu`), and sets concurrency / model pool limits (`MODEL_POOL_SIZE`, `MAX_QUEUE_SIZE`, `MAX_CHUNK_SIZE_BYTES`).
* **`engine.py` (`ModelPoolEngine`)**: Hosts resident `transcribe_cpp.Model` instances in GPU VRAM (configurable via `MODEL_POOL_SIZE`, default 3 $\approx$ 2.15 GB VRAM). Implements an asynchronous queue leasing pattern (`acquire_model`), eliminating subprocess spawning overhead and providing thread-safe, non-blocking in-memory inference.
* **`sanitizer.py` (`sanitize_pcm_chunk`)**: Hardened audio validation pipeline. Enforces 4-byte float32 memory alignment, maximum chunk size limits (DoS prevention), and filters non-finite values (`NaN`, `+Inf`, `-Inf`) to prevent state-poisoning in Conformer recurrent attention caches.
* **`schemas.py`**: Defines strict Pydantic models for REST endpoints (`TranscriptionResponse`, `VerboseTranscriptionResponse`), diagnostics (`HealthResponse`), and WebSocket streaming events (`StreamInitAction`, `StreamReadyEvent`, `StreamPartialEvent`, `StreamFinalEvent`, `StreamBusyEvent`).
* **`main.py` (`create_app`)**: Factory creating the FastAPI instance. Configures CORS, mounts route controllers, and registers the `lifespan` handler to ensure graceful engine warmup on startup and model/thread-pool drainage on shutdown.

---

## 2. API Endpoints & Protocols

### 2.1 Diagnostics & Telemetry (`server/routes/health.py`)

#### `GET /health`
Returns runtime operational health, compute backend details, and GPU memory metrics. Useful for Docker/Kubernetes readiness probes and client telemetry.

* **Response Schema (`HealthResponse`):**
  ```json
  {
    "status": "healthy",
    "uptime_seconds": 124.5,
    "backend": "CUDA (NVIDIA GeForce RTX 5070 Ti)",
    "vram_used_mb": 2148.9,
    "vram_total_mb": 16302.0,
    "active_sessions": 1,
    "pool_size": 3,
    "available_models": 2
  }
  ```
  * `status`: `"healthy"` (warmup completed and engine ready) or `"initializing"`.
  * `uptime_seconds`: Engine running time in seconds.
  * `backend`: Active hardware device (CUDA GPU or CPU fallback).
  * `vram_used_mb`: Memory footprint of resident Q8_0 GGUF models in VRAM (~716 MB per instance).
  * `vram_total_mb`: Total VRAM available on the GPU.
  * `active_sessions`: Number of concurrent active WebSocket streams.
  * `pool_size`: Total number of resident model instances in the pool.
  * `available_models`: Currently idle model instances available for immediate leasing.

#### `GET /v1/models`
Returns an OpenAI-compliant model catalog allowing generic OpenAI clients, SDKs, and UIs to discover the loaded ASR model.

* **Response:**
  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf",
        "object": "model",
        "owned_by": "nvidia",
        "permission": []
      }
    ]
  }
  ```

---

### 2.2 OpenAI-Compatible REST Transcription (`server/routes/rest_api.py`)

#### `POST /v1/audio/transcriptions`
Accepts multipart audio uploads conforming to the OpenAI Whisper API specification.

* **Parameters:**
  * `file`: Audio file (`.wav`, `.mp3`, `.ogg`, `.flac`).
  * `model`: Model identifier (default: `"nemotron-3.5-asr-streaming-0.6b"`).
  * `language`: Language code (e.g. `"en"`, `"en-US"`, `"de"`, `"de-DE"`).
  * `response_format`: `"json"` | `"text"` | `"verbose_json"` (default: `"json"`).
  * `temperature`: Sampling temperature (default: `0.0`).

* **Example Call:**
  ```bash
  curl -X POST "http://127.0.0.1:8001/v1/audio/transcriptions" \
    -F "file=@data/input/audio/bcn_weather_16k.wav" \
    -F "model=nemotron-3.5-asr-streaming-0.6b" \
    -F "language=en-US" \
    -F "response_format=verbose_json"
  ```

* **Verbose JSON Response:**
  Includes transcription text, duration, segment breakdown, and word-level timestamps:
  ```json
  {
    "task": "transcribe",
    "language": "en-US",
    "duration": 11.04,
    "text": "Yesterday it was thirty five degrees in Barcelona...",
    "segments": [
      {"id": 0, "start": 0.0, "end": 11.04, "text": "Yesterday it was..."}
    ],
    "words": [
      {"word": "Yesterday", "start": 1.20, "end": 1.52},
      {"word": "it", "start": 1.84, "end": 1.92}
    ]
  }
  ```

---

### 2.3 Chunked Audio WebSocket Endpoint (`server/routes/chunked_ws.py`)

#### `WS /ws/chunked` (and alias `WS /ws/stt`)
A stateless, chunk-based WebSocket endpoint designed for discrete audio segment transcription (e.g., from client-side Voice Activity Detection / VAD):

* **Transport:** WebSocket Binary Frames.
* **Payload:** Raw 16 kHz Mono `float32` Little-Endian PCM audio frames.
* **Processing:** Offloaded asynchronously to the native engine's worker thread pool without blocking the FastAPI event loop.
* **Response:** Text Frame containing the raw transcribed string (or empty string `""` for silence or short chunks $< 0.3$s).
* **Endpoints:** Available at both `/ws/chunked` (standard technical route) and `/ws/stt` (drop-in compatibility route).

---

### 2.4 Enterprise Live Streaming WebSocket (`server/routes/streaming_ws.py`)

#### `WS /ws/stream`
A bidirectional protocol designed for interactive live streaming clients.

* **Architectural Guarantees:**
  * **Asynchronous Producer-Consumer Decoupling:** Ingestion of binary WebSocket audio frames is decoupled from GPU inference via an internal `asyncio.Queue(maxsize=10)`. Under backpressure/congestion, the queue automatically batch-drains pending chunks to preserve sub-60ms responsiveness.
  * **Audio Sanitizing Pipeline (`server/sanitizer.py`):** Every incoming chunk is intercepted and checked before passing to C++ bindings. Rejects chunks exceeding `max_chunk_size_bytes` (1 MB), non-aligned buffers (`len % 4 != 0`), or chunks containing `NaN`/`Inf` to protect the Conformer attention cache from state poisoning.
  * **Concurrency Gating:** Each active stream borrows a `transcribe_cpp.Model` instance from `ModelPoolEngine`. If all pool instances are occupied, the server immediately returns a `StreamBusyEvent` (`code: 4029, event: "busy"`) instead of stalling or dropping the connection silently.

* **Protocol Flow:**
  1. **Handshake:** Client sends JSON text frame:
     ```json
     {"action": "start", "language": "en-US", "chunk_ms": 320, "lookahead": 3}
     ```
  2. **Server Ready:** Server confirms initialization:
     ```json
     {"event": "ready", "session_id": "c1f7b9e0-...", "sample_rate": 16000}
     ```
     *(Or returns `{"event": "busy", "code": 4029, "message": "Model pool exhausted..."}` if all models are busy).*
  3. **Continuous Streaming:** Client sends successive binary PCM frames (`bytes`). Server validates audio and returns partial hypotheses with live latency:
     ```json
     {"event": "partial", "committed": "Yesterday it was", "tentative": "", "latency_ms": 14.8}
     ```
  4. **Finalization:** Client sends `{"action": "stop"}`. Server replies with the full transcript and performance metrics:
     ```json
     {"event": "final", "text": "Yesterday it was thirty five degrees in Barcelona...", "duration_s": 11.04, "rtf": 0.0512}
     ```

---

## 3. Running the Server

Launch the server using the project CLI runner:

```bash
# Default launch (127.0.0.1:8001 with auto-detected CUDA backend)
uv run python run_server.py

# Custom host, port, and backend
uv run python run_server.py --host 0.0.0.0 --port 8001 --backend cuda
```

Interactive OpenAPI documentation is automatically available at:
* **Swagger UI:** `http://127.0.0.1:8001/docs`
* **ReDoc:** `http://127.0.0.1:8001/redoc`

---

## 4. Testing & Verification

Run the automated test suite covering all modules, engine lifecycles, and routes:

```bash
# Run full server test suite
uv run pytest tests/ -v
```
