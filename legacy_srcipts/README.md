# Legacy test and prototyping scripts

Standalone test scripts for evaluating different inference methods for Nemotron 3.5 ASR Streaming 0.6B on an NVIDIA RTX 5070 Ti.

---

## Script inventory

| Script | Engine / Class | Mode | Primary Use |
| :--- | :--- | :--- | :--- |
| [`test_pipeline.py`](test_pipeline.py) | `transformers.pipeline` | High-level pipeline | Quick sanity checks and simple file transcription |
| [`test_offline_model.py`](test_offline_model.py) | `AutoModelForRNNT` + `AutoProcessor` | Low-level batch inference | Direct tensor control, fixed prompt vs. auto language detection |
| [`test_streaming.py`](test_streaming.py) | `AutoModelForRNNT` + `TextIteratorStreamer` | Chunked streaming | Real-time terminal output with rolling caches |
| [`test_gguf.py`](test_gguf.py) | `transcribe-cli` (GGUF, C++) | Quantized Q8_0 | Standalone C++ execution without Python dependencies |

---

## Scripts

### 1. test_pipeline.py (high-level pipeline)
* **Architecture.** Wraps model loading, audio resampling via `soundfile`, and RNN-T decoding inside Hugging Face `pipeline("automatic-speech-recognition")`.
* **Execution.** Runs on CUDA in FP16 (`dtype=torch.float16`).
* **Command:**
  ```powershell
  uv run python legacy_srcipts/test_pipeline.py [path_to_audio]
  ```
  Default file: `data/input/audio/bcn_weather.mp3`

---

### 2. test_offline_model.py (direct model and language detection)
* **Architecture.** Instantiates `Nemotron3_5AsrForRNNT` and `Nemotron3_5AsrProcessor` directly.
* **Modes:**
  * **Fixed language conditioning.** Passes `language="en-US"` into the processor, guiding the RNN-T decoder with the language prompt vector. Decodes with `skip_special_tokens=True`.
  * **In-flight language detection.** Passes `language="auto"`. The model classifies the spoken language and appends the detected locale tag (e.g. `<en-US>`) to the end of the text.
* **Command:**
  ```powershell
  uv run python legacy_srcipts/test_offline_model.py [path_to_audio]
  ```
  Default file: `data/input/audio/bcn_weather.mp3`

---

### 3. test_streaming.py (chunked streaming)
* **Architecture.** Runs FastConformer in streaming mode (`is_streaming=True`) with rolling attention and convolution caches.
* **Lookahead.** Uses `processor.set_num_lookahead_tokens(6)` for an accuracy-latency tradeoff (~480 ms lookahead window).
* **Chunking.** Slices incoming audio into non-overlapping mel-spectrogram chunks.
* **Output.** Decodes in a background thread and streams tokens live to stdout via `TextIteratorStreamer`.
* **Command:**
  ```powershell
  uv run python legacy_srcipts/test_streaming.py [path_to_audio]
  ```
  Default file: `data/input/audio/obama.mp3`

---

### 4. test_gguf.py (standalone GGUF inference)
* **Architecture.** Calls the compiled `transcribe-cli` C++ binary (`transcribe_cpp/build/bin/Release/transcribe-cli.exe`) with the quantized Q8_0 GGUF model (751 MB).
* **Backends.** Switches between GPU (`--backend cuda`) and CPU (`--backend cpu`, default `--backend auto`).
* **Preprocessing.** Automatically resamples input files (e.g. `.mp3`) to 16 kHz mono WAV using `soundfile` before execution.
* **Modes.** Runs both offline batch decoding (up to 19.5x real-time on GPU, 7.2x on CPU) and streaming simulation (5.2x real-time on GPU, 2.9x on CPU).
* **Commands:**
  ```powershell
  # Auto backend selection (GPU prioritized if compiled)
  uv run python legacy_srcipts/test_gguf.py [path_to_audio]

  # Force CUDA backend
  uv run python legacy_srcipts/test_gguf.py path/to/audio.mp3 --backend cuda

  # Force CPU AVX2 backend
  uv run python legacy_srcipts/test_gguf.py path/to/audio.mp3 --backend cpu
  ```
  Default file: `data/input/audio/bcn_weather.mp3`

---

## Path lookup order

Scripts locate audio files in this sequence:
1. Path passed via CLI argument.
2. Current working directory.
3. Standard audio directory (`data/input/audio/`).
4. Project root (`stt_nemotron_3_5_asr_streaming_0_6/`).
5. Script directory (`legacy_srcipts/`).

---

## Requirements

* **GPU.** NVIDIA RTX 5070 Ti (16 GB VRAM, Blackwell architecture).
* **CUDA and PyTorch.** CUDA 12.8 wheel (`torch==2.11.0+cu128`, `sm_120` compatibility).
* **Weights.** Stored locally in the folder configured via `MODELS_DIR` in `.env`.

