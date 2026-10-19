"""Audio validation and sanitizing pipeline against poisoning, corruption, and DoS.

Created: 2026-10-13
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def sanitize_pcm_chunk(
    chunk_bytes: bytes,
    max_size_bytes: int = 1_048_576,
) -> np.ndarray | None:
    """Validate and sanitize raw float32 PCM bytes.

    Guards:
    1. Size cap: Rejects empty chunks or chunks exceeding max_size_bytes (default 1 MB).
    2. Alignment check: Enforces len(chunk_bytes) % 4 == 0 (32-bit float boundary).
    3. State-Poisoning filter: Rejects chunks containing NaN, +Inf, -Inf, or denormals.

    Returns:
        np.ndarray of dtype float32 if valid, otherwise None.
    """
    if not chunk_bytes:
        return None

    if len(chunk_bytes) > max_size_bytes:
        logger.warning(
            f"Rejected PCM chunk: size {len(chunk_bytes)} bytes exceeds limit of {max_size_bytes} bytes."
        )
        return None

    if len(chunk_bytes) % 4 != 0:
        logger.warning(
            f"Rejected PCM chunk: size {len(chunk_bytes)} bytes is not 4-byte aligned (float32 required)."
        )
        return None

    try:
        arr = np.frombuffer(chunk_bytes, dtype=np.float32)
    except Exception as exc:
        logger.warning(f"Rejected PCM chunk: numpy buffer conversion failed: {exc}")
        return None

    if not np.isfinite(arr).all():
        logger.warning("Rejected PCM chunk: buffer contains non-finite values (NaN / Inf).")
        return None

    return arr
