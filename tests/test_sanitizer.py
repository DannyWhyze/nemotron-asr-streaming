"""Unit tests for audio sanitizing module against corruption and state poisoning.

Created: 2026-10-13
"""

import numpy as np

from server.sanitizer import sanitize_pcm_chunk


def test_sanitize_valid_chunk():
    data = np.zeros(1600, dtype=np.float32)
    chunk = sanitize_pcm_chunk(data.tobytes())
    assert chunk is not None
    assert len(chunk) == 1600
    assert chunk.dtype == np.float32


def test_sanitize_empty_chunk():
    assert sanitize_pcm_chunk(b"") is None


def test_sanitize_oversized_chunk():
    large_bytes = b"\x00" * (1_048_576 + 4)
    assert sanitize_pcm_chunk(large_bytes, max_size_bytes=1_048_576) is None


def test_sanitize_unaligned_chunk():
    unaligned = b"\x00\x00\x00"  # 3 bytes
    assert sanitize_pcm_chunk(unaligned) is None


def test_sanitize_nan_chunk():
    data = np.array([0.0, float("nan"), 0.5], dtype=np.float32)
    assert sanitize_pcm_chunk(data.tobytes()) is None


def test_sanitize_inf_chunk():
    data = np.array([0.0, float("inf"), 0.5], dtype=np.float32)
    assert sanitize_pcm_chunk(data.tobytes()) is None
    data_neg = np.array([0.0, float("-inf"), 0.5], dtype=np.float32)
    assert sanitize_pcm_chunk(data_neg.tobytes()) is None
