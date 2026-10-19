# Nemotron 3.5 ASR streaming 0.6B model specification (GGUF)

## 1. Model overview

| Attribute | Details |
| :--- | :--- |
| **Model name** | `nemotron-3.5-asr-streaming-0.6b-gguf` |
| **Base model** | [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) (Commit `24b151a`) |
| **GGUF port** | [`handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf`](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf) |
| **Target runtime** | [`transcribe.cpp`](https://github.com/handy-computer/transcribe.cpp) (Reference: NeMo 26.06) |
| **Architecture** | Cache-aware FastConformer encoder with prompt-conditioned RNN-T decoder |
| **Parameters** | ~600 Million (0.6B) |
| **Primary task** | Multilingual speech-to-text and streaming transcription |
| **License** | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) (commercial use permitted under license terms) |
| **Hardware targets** | NVIDIA GPUs (Blackwell, Ada Lovelace, Hopper, Ampere, Turing), Apple Silicon (Metal), Vulkan, and x86_64/ARM CPU |

Nemotron 3.5 ASR is a streaming multilingual speech-to-text model from NVIDIA, quantized to GGUF format for C++ inference in `transcribe.cpp`. Rather than buffering overlapping audio windows and re-processing them, Nemotron 3.5 uses a cache-aware FastConformer that processes incoming frames once and maintains constant-size internal state caches.

---

## 2. Architecture highlights

### 2.1 Cache-aware streaming vs. buffered audio
* **Zero redundant compute.** Standard streaming systems buffer overlapping chunks (e.g. 10s window with 1s stride), recomputing old audio repeatedly. The cache-aware FastConformer keeps self-attention and convolution state caches across consecutive 80ms frames.
* **Higher concurrency.** Handles more concurrent streams at 80ms latency compared to buffered baselines on the same hardware.
* **Constant memory.** Audio processing is non-recurrent across sequence length. Memory overhead stays constant whether the audio is 10 seconds or 10 hours (`transcribe_capabilities.max_audio_ms == 0`).

### 2.2 Native punctuation, casing, and timestamps
* **Direct punctuation.** Generates true-cased text with periods, commas, apostrophes, and question marks directly, removing downstream punctuation models.
* **Alignment.** Supports token-level and word-level timestamp generation.

### 2.3 Language conditioning and detection
* **Language prompt conditioning.** Target language is injected as a 128-dimensional one-hot embedding concatenated across time frames with acoustic features before projection to the RNN-T decoder.
* **Automatic language detection.** Setting `target_lang=auto` infers the spoken language during decoding and appends a `<lang-XX>` tag after final punctuation.

---

## 3. Supported languages

The tokenizer includes vocabularies for 40 language locales in three tiers:

### Tier 1: Production ready (19 locales)
* **English:** `en-US`, `en-GB`
* **Spanish:** `es-US`, `es-ES`
* **French:** `fr-FR`, `fr-CA`
* **German:** `de-DE`
* **Italian:** `it-IT`
* **Portuguese:** `pt-BR`, `pt-PT`
* **Dutch:** `nl-NL`
* **Russian:** `ru-RU`
* **Turkish:** `tr-TR`
* **Arabic:** `ar-AR`
* **Hindi:** `hi-IN`
* **Japanese:** `ja-JP` (evaluated via CER)
* **Korean:** `ko-KR` (evaluated via CER)
* **Vietnamese:** `vi-VN`
* **Ukrainian:** `uk-UA`

### Tier 2: Broad coverage (13 locales)
* **Polish (`pl-PL`)**, **Swedish (`sv-SE`)**, **Czech (`cs-CZ`)**, **Norwegian Bokmål (`nb-NO`)**
* **Danish (`da-DK`)**, **Bulgarian (`bg-BG`)**, **Finnish (`fi-FI`)**, **Croatian (`hr-HR`)**
* **Slovak (`sk-SK`)**, **Mandarin (`zh-CN`)**, **Hungarian (`hu-HU`)**, **Romanian (`ro-RO`)**, **Estonian (`et-EE`)**

### Tier 3: Adaptation ready (8 locales)
* **Greek (`el-GR`)**, **Hebrew (`he-IL`)**, **Lithuanian (`lt-LT`)**, **Slovenian (`sl-SI`)**
* **Latvian (`lv-LV`)**, **Maltese (`mt-MT`)**, **Thai (`th-TH`)**, **Norwegian Nynorsk (`nn-NO`)**

---

## 4. Quantization and sizes

| Format | File Name | Size | FLEURS en WER (Offline) | LibriSpeech Clean WER | Recommended Scenario |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **F32** | `nemotron-3.5-asr-streaming-0.6b-F32.gguf` | 2.55 GB | 7.97% | 3.04% | Numerical reference |
| **F16** | `nemotron-3.5-asr-streaming-0.6b-F16.gguf` | 1.28 GB | 7.97% | 3.04% | High-end GPU deployments |
| **Q8_0** | `nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf` | 751 MB | **7.88%** | **3.05%** | Default choice (lossless accuracy) |
| **Q6_K** | `nemotron-3.5-asr-streaming-0.6b-Q6_K.gguf` | 621 MB | 8.02% | 3.08% | Balanced edge workstations |
| **Q5_K_M** | `nemotron-3.5-asr-streaming-0.6b-Q5_K_M.gguf` | 560 MB | 8.15% | 3.10% | Resource-constrained systems |
| **Q4_K_M** | `nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf` | 496 MB | 8.49% | 3.28% | Low-memory embedded devices |

---

## 5. Latency and lookahead

Latency and lookahead are controlled by `att_context_size = [56, R]`, where 56 is left context frames and $R$ is right lookahead frames (each frame is 80 ms):

| Mode / Lookahead ($R$) | Chunk Size | Lookahead Window | Latency Profile | Average WER (FLEURS Tier 1) |
| :---: | :---: | :---: | :--- | :---: |
| **$R = 0$** | 1 frame (80 ms) | 0 ms | Ultra-low latency voice agents / real-time dialog | 10.38% |
| **$R = 1$** | 2 frames (160 ms) | 80 ms | Low-latency telephony / live captions | 10.00% |
| **$R = 3$** | 4 frames (320 ms) | 240 ms | Interactive streaming | 9.49% |
| **$R = 6$** | 7 frames (560 ms) | 480 ms | Near-offline quality streaming | 9.12% |
| **$R = 13$** | 14 frames (1120 ms)| 1040 ms | Matches offline baseline | **8.84%** |

Tune streaming chunks in `transcribe.cpp` using `--stream-chunk-ms 1120 --stream-att-right {0,3,6,13}`.

---

## 6. Audio requirements and output format

### 6.1 Input audio requirements
* **Sample rate.** 16,000 Hz (16 kHz).
* **Channels.** Single channel (mono).
* **Format.** Uncompressed 16-bit PCM WAV or float32 PCM samples.
* **Duration.** No arbitrary limit; processed in a continuous single pass.

### 6.2 Resampling
Convert audio files that are not 16 kHz mono WAV using ffmpeg:
```bash
ffmpeg -i input_source.ext -ar 16000 -ac 1 -c:a pcm_s16le output_16k_mono.wav
```

### 6.3 Output format
* **Format.** UTF-8 string with punctuation.
* **Tag handling (`strip_lang_tags`):**
  * `false`: Output retains `<lang-XX>` marker (e.g. `Hello world.<en-US>`).
  * `true`: Language tag is removed, returning clean text.

---

## 7. Deployment instructions

### 7.1 C++ inference via transcribe.cpp
1. **Build binary:**
   ```bash
   git clone https://github.com/handy-computer/transcribe.cpp.git
   cd transcribe.cpp
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ```
2. **Offline batch execution:**
   ```bash
   ./build/bin/transcribe-cli \
     -m ./models/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf \
     --language de-DE \
     input.wav
   ```
3. **Streaming execution:**
   ```bash
   ./build/bin/transcribe-cli \
     -m ./models/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf \
     --stream-chunk-ms 320 \
     --stream-att-right 3 \
     --language auto \
     input.wav
   ```

### 7.2 Python inference via NeMo
```python
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-3.5-asr-streaming-0.6b"
)

transcriptions = asr_model.transcribe(
    paths2audio_files=["input_16k_mono.wav"],
    batch_size=1,
)
```

### 7.3 Security and licenses
* **Offline execution.** Run weights and code locally without outbound network calls during inference.
* **Checksum validation.** Verify SHA256 checksums and `download_info.json` upon downloading weights.
* **License compliance.** Verify intended usage complies with [OpenMDW-1.1](https://openmdw.ai/license/1-1/).

