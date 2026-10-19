"""Server configuration module loading from environment and .env."""

import logging
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_transcribe_environment() -> Path:
    """Configure DLL search paths, TRANSCRIBE_LIBRARY env var, and Python bindings path."""
    dll_candidates = [
        REPO_ROOT / "transcribe_cpp" / "build" / "bin" / "Release" / "transcribe.dll",
        REPO_ROOT / "transcribe_cpp" / "build" / "bin" / "transcribe.dll",
        REPO_ROOT / "transcribe_cpp" / "build" / "src" / "Release" / "transcribe.dll",
        REPO_ROOT / "transcribe_cpp" / "build" / "src" / "transcribe.dll",
        REPO_ROOT / "transcribe_cpp" / "build-shared" / "bin" / "Release" / "transcribe.dll",
    ]
    resolved_dll: Path | None = None
    for cand in dll_candidates:
        if cand.is_file():
            resolved_dll = cand.resolve()
            break

    if resolved_dll:
        os.environ["TRANSCRIBE_LIBRARY"] = str(resolved_dll)
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(resolved_dll.parent))
            except Exception as exc:
                logger.debug(f"Could not add DLL dir {resolved_dll.parent}: {exc}")

    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        cuda_dirs = []
        if "CUDA_PATH" in os.environ:
            cuda_dirs.append(Path(os.environ["CUDA_PATH"]) / "bin")
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            cuda_dirs.append(Path(nvcc_path).parent)
        cuda_dirs.extend(
            [
                Path(r"D:\Programs\cuda128\bin"),
                Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
            ]
        )
        for cdir in cuda_dirs:
            if cdir.is_dir():
                try:
                    os.add_dll_directory(str(cdir.resolve()))
                except Exception as exc:
                    logger.debug(f"Could not add CUDA DLL dir {cdir}: {exc}")

    bindings_src = REPO_ROOT / "transcribe_cpp" / "bindings" / "python" / "src"
    if bindings_src.is_dir() and str(bindings_src) not in sys.path:
        sys.path.insert(0, str(bindings_src.resolve()))

    return resolved_dll or dll_candidates[0]


NATIVE_DLL_PATH = setup_transcribe_environment()


class ServerConfig(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8001
    models_dir: str = os.getenv("MODELS_DIR", "models")
    model_file: str = "nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"
    model_folder: str = "handy-computer--nemotron-3.5-asr-streaming-0.6b-gguf"
    backend: Literal["auto", "cuda", "cpu"] = "auto"
    max_workers: int = 2
    default_language: str = "en-US"
    min_segment_duration_s: float = 0.3
    log_level: str = "INFO"
    model_pool_size: int = 3
    max_chunk_size_bytes: int = 1_048_576  # 1 MB
    max_queue_size: int = 10

    @property
    def model_path(self) -> Path:
        candidates = [
            Path(self.models_dir) / self.model_folder / self.model_file,
            REPO_ROOT / "models" / self.model_file,
            Path(self.model_file),
        ]
        for c in candidates:
            if c.is_file():
                return c.resolve()
        return candidates[0]


@lru_cache
def get_config() -> ServerConfig:
    return ServerConfig()
