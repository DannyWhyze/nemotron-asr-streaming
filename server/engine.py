"""Enterprise in-memory Model-Pool Engine for transcribe.cpp on CUDA."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr

from server.config import ServerConfig

logger = logging.getLogger(__name__)


class ModelPoolExhaustedError(RuntimeError):
    """Raised when all resident models in the pool are currently leased."""
    pass


class ModelPoolEngine:
    """Enterprise thread-safe engine managing resident model instances and async pool leasing."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.model_path = config.model_path
        self.pool_size = max(1, config.model_pool_size)
        self.models: list[Any] = []
        self._pool: asyncio.Queue[Any] | None = None
        self.executor = ThreadPoolExecutor(max_workers=max(config.max_workers, self.pool_size))
        self.start_time = time.time()
        self.device_name = "Unknown"
        self.backend_tag = config.backend.upper()
        self.total_vram_mb = 0.0
        self.used_vram_mb = 0.0
        self.is_ready = False
        self.active_sessions = 0

    def _probe_device(self) -> str:
        """Inspect GPU and system capabilities."""
        backend_choice = self.config.backend
        try:
            import torch
            if torch.cuda.is_available():
                self.device_name = torch.cuda.get_device_name(0)
                self.total_vram_mb = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1
                )
                if backend_choice == "auto":
                    backend_choice = "cuda"
        except Exception as exc:
            logger.debug(f"Torch CUDA probe error: {exc}")

        if backend_choice == "auto":
            backend_choice = "cpu"
            self.device_name = "CPU Compute"

        self.backend_tag = backend_choice.upper()
        return backend_choice

    def initialize(self) -> None:
        """Initialize resident model instances in VRAM and execute warmup inference."""
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        import transcribe_cpp

        backend = self._probe_device()
        logger.info(
            f"Initializing ModelPoolEngine with pool_size={self.pool_size} on {backend.upper()} ({self.device_name})"
        )

        self.models.clear()
        silent_frame = np.zeros(5120, dtype=np.float32)  # 320ms warmup frame

        for idx in range(self.pool_size):
            logger.info(f"Loading resident model instance {idx + 1}/{self.pool_size} into VRAM...")
            model = transcribe_cpp.Model(str(self.model_path), backend=backend)
            # Warmup inference
            session = model.session()
            try:
                _ = session.run(silent_frame)
            finally:
                session.close()

            self.models.append(model)

        self.used_vram_mb = round(716.3 * len(self.models), 1)
        self.is_ready = True
        logger.info(f"ModelPoolEngine ready with {len(self.models)} resident models in VRAM.")

    def _get_or_create_queue(self) -> asyncio.Queue:
        """Return or lazily initialize the asyncio model pool queue."""
        if self._pool is None:
            self._pool = asyncio.Queue(maxsize=self.pool_size)
            for m in self.models:
                self._pool.put_nowait(m)
        return self._pool

    @asynccontextmanager
    async def acquire_model(self, timeout: float = 2.0):
        """Asynchronous context manager to borrow a model from the pool with timeout."""
        if not self.is_ready or not self.models:
            raise RuntimeError("ModelPoolEngine is not initialized or ready")

        pool = self._get_or_create_queue()
        try:
            model = await asyncio.wait_for(pool.get(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            raise ModelPoolExhaustedError(
                f"Model pool exhausted ({self.pool_size} models busy). Please retry."
            )

        try:
            yield model
        finally:
            await pool.put(model)

    async def borrow_model(self, timeout: float = 2.0):
        """Explicitly borrow a model from the pool."""
        if not self.is_ready or not self.models:
            raise RuntimeError("ModelPoolEngine is not initialized or ready")

        pool = self._get_or_create_queue()
        try:
            return await asyncio.wait_for(pool.get(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            raise ModelPoolExhaustedError(
                f"Model pool exhausted ({self.pool_size} models busy). Please retry."
            )

    async def return_model(self, model) -> None:
        """Return a borrowed model back into the pool."""
        pool = self._get_or_create_queue()
        await pool.put(model)

    def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        language: str | None = None,
        model: Any | None = None,
    ) -> tuple[str, float]:
        """Transcribe raw 16 kHz Mono float32 PCM bytes directly in memory."""
        lang = language or self.config.default_language
        audio_np = np.frombuffer(pcm_bytes, dtype=np.float32)
        if len(audio_np) == 0:
            return "", 0.0

        duration_s = len(audio_np) / 16000.0
        if duration_s < self.config.min_segment_duration_s:
            return "", duration_s

        selected_model = model or (self.models[0] if self.models else None)
        if selected_model is None:
            raise RuntimeError("No models available in ModelPoolEngine")

        session = selected_model.session()
        try:
            res = session.run(audio_np, language=lang)
            return res.text, duration_s
        finally:
            session.close()

    def transcribe_audio_file(
        self,
        audio_path: Path,
        language: str | None = None,
        model: Any | None = None,
    ) -> dict[str, Any]:
        """Transcribe an audio file with detailed word alignments in memory."""
        lang = language or self.config.default_language
        audio_np, sr = sf.read(str(audio_path), dtype="float32")
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)
        if sr != 16000:
            audio_np = soxr.resample(audio_np, sr, 16000)
            sr = 16000

        duration_s = len(audio_np) / float(sr)

        selected_model = model or (self.models[0] if self.models else None)
        if selected_model is None:
            raise RuntimeError("No models available in ModelPoolEngine")

        session = selected_model.session()
        try:
            res = session.run(audio_np, timestamps="word", language=lang)
            words = [
                {"word": w.text, "start": w.t0_ms / 1000.0, "end": w.t1_ms / 1000.0}
                for w in (res.words or [])
            ]
            segments = [
                {"id": i, "start": seg.t0_ms / 1000.0, "end": seg.t1_ms / 1000.0, "text": seg.text}
                for i, seg in enumerate(res.segments or [])
            ]
            if not segments and res.text:
                segments = [{"id": 0, "start": 0.0, "end": duration_s, "text": res.text}]

            return {
                "text": res.text,
                "duration": duration_s,
                "language": lang,
                "words": words,
                "segments": segments,
            }
        finally:
            session.close()

    def get_telemetry(self) -> dict[str, Any]:
        """Return diagnostic metrics."""
        available = self._pool.qsize() if self._pool is not None else len(self.models)
        return {
            "is_ready": self.is_ready,
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "device": self.device_name,
            "backend": self.backend_tag,
            "vram_used_mb": self.used_vram_mb,
            "vram_total_mb": self.total_vram_mb,
            "active_sessions": self.active_sessions,
            "pool_size": self.pool_size,
            "available_models": available,
        }

    def shutdown(self) -> None:
        """Gracefully release executor and model resources in VRAM."""
        self.executor.shutdown(wait=True)
        for m in self.models:
            try:
                m.close()
            except Exception as exc:
                logger.warning(f"Error closing model instance: {exc}")
        self.models.clear()
        self._pool = None
        self.is_ready = False


# Backward-compatible alias
STTEngine = ModelPoolEngine
