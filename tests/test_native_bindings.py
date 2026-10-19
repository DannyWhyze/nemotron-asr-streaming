"""Isolated native bindings smoke test for transcribe.cpp FFI on GPU.

Created: 2026-10-13
"""

import time

import numpy as np

from server.config import get_config


def test_native_bindings_smoke():
    import transcribe_cpp

    assert transcribe_cpp.__version__ == "0.3.1"
    config = get_config()
    assert config.model_path.is_file(), f"Model file not found: {config.model_path}"

    model = transcribe_cpp.Model(str(config.model_path), backend="cuda")
    try:
        session = model.session()
        try:
            # 1. Warmup inference
            silent_chunk = np.zeros(5120, dtype=np.float32)  # 320 ms at 16 kHz
            t0 = time.perf_counter()
            _ = session.run(silent_chunk)
            warmup_ms = (time.perf_counter() - t0) * 1000.0
            assert warmup_ms < 500.0, f"Warmup took too long: {warmup_ms:.2f}ms"

            # 2. Second inference should be fast (< 500 ms on RTX 5070 Ti)
            t1 = time.perf_counter()
            _ = session.run(silent_chunk)
            infer_ms = (time.perf_counter() - t1) * 1000.0
            assert infer_ms < 500.0, f"Inference took too long: {infer_ms:.2f}ms"

            # 3. Stream with ParakeetStreamOptions(att_context_right=3)
            options = transcribe_cpp.ParakeetStreamOptions(att_context_right=3)
            stream = session.stream(family=options)
            stream.feed(silent_chunk)
            stream.finalize()
            text_result = stream.text()
            assert hasattr(text_result, "full")
        finally:
            session.close()
    finally:
        model.close()
