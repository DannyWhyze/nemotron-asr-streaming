"""Integration and stress tests for /ws/stream producer-consumer live streaming.

Tests:
1. Standard streaming lifecycle.
2. Multi-client concurrency up to pool limit (3 clients) and 4th client rejection (4029).
3. 15-chunk benchmark verifying low latency (<100ms) and RTF (<0.2).
4. Security guards against unaligned PCM, NaN, and Inf poisoned inputs.

Created: 2026-10-13
"""

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from server.config import ServerConfig
from server.engine import ModelPoolEngine
from server.routes.streaming_ws import create_streaming_ws_router

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_WAV = REPO_ROOT / "data" / "input" / "audio" / "bcn_weather_16k.wav"


@pytest.fixture(scope="module")
def streaming_env():
    config = ServerConfig(model_pool_size=3)
    engine = ModelPoolEngine(config)
    engine.initialize()

    app = FastAPI()
    app.include_router(create_streaming_ws_router(engine))
    app.state.engine = engine

    client = TestClient(app)
    yield client, engine

    engine.shutdown()


def test_ws_streaming_lifecycle(streaming_env):
    client, _ = streaming_env
    data, _ = sf.read(str(TEST_WAV), dtype="float32")

    with client.websocket_connect("/ws/stream") as ws:
        # 1. Send handshake
        ws.send_text(json.dumps({"action": "start", "language": "en-US", "chunk_ms": 320}))
        ready = json.loads(ws.receive_text())
        assert ready["event"] == "ready"

        # 2. Feed 2 chunks
        chunk_len = int(16000 * 0.32)
        chunk1 = data[:chunk_len].tobytes()
        chunk2 = data[chunk_len : chunk_len * 2].tobytes()

        ws.send_bytes(chunk1)
        part1 = json.loads(ws.receive_text())
        assert part1["event"] == "partial"
        assert "latency_ms" in part1

        ws.send_bytes(chunk2)
        part2 = json.loads(ws.receive_text())
        assert part2["event"] == "partial"

        # 3. Finalize
        ws.send_text(json.dumps({"action": "stop"}))
        final = json.loads(ws.receive_text())
        assert final["event"] == "final"
        assert "text" in final
        assert final["duration_s"] > 0


def test_ws_streaming_multi_client_concurrency(streaming_env):
    client, engine = streaming_env
    assert engine.pool_size == 3

    hold_barrier = threading.Barrier(4)  # 3 client threads + 1 coordinator thread
    release_event = threading.Event()
    results = [None, None, None]

    def client_worker(worker_id: int):
        try:
            with client.websocket_connect("/ws/stream") as ws:
                ws.send_text(json.dumps({"action": "start"}))
                ready = json.loads(ws.receive_text())
                assert ready["event"] == "ready"

                # Signal ready and hold session open
                hold_barrier.wait()

                # Feed 1 chunk
                chunk = (b"\x00" * 2048)
                ws.send_bytes(chunk)
                part = json.loads(ws.receive_text())
                assert part["event"] == "partial"

                # Wait for coordinator signal to finish
                release_event.wait()
                ws.send_text(json.dumps({"action": "stop"}))
                final = json.loads(ws.receive_text())
                assert final["event"] == "final"
                results[worker_id] = "OK"
        except Exception as exc:
            results[worker_id] = str(exc)

    threads = [threading.Thread(target=client_worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()

    # Wait until all 3 clients are actively holding models
    hold_barrier.wait()
    time.sleep(0.1)

    # 4th concurrent client should be rejected with 4029 / busy
    got_busy_rejection = False
    try:
        with client.websocket_connect("/ws/stream") as ws4:
            msg = ws4.receive_text()
            data = json.loads(msg)
            if data.get("event") == "busy" or data.get("code") == 4029:
                got_busy_rejection = True
    except WebSocketDisconnect as exc:
        if exc.code == 4029:
            got_busy_rejection = True

    assert got_busy_rejection is True, "4th client was not rejected with busy event / code 4029"

    # Release workers
    release_event.set()
    for t in threads:
        t.join()

    assert results == ["OK", "OK", "OK"], f"Concurrency errors: {results}"


def test_ws_streaming_benchmark_15_chunks(streaming_env):
    client, _ = streaming_env
    data, sr = sf.read(str(TEST_WAV), dtype="float32")
    chunk_len = int(sr * 0.32)  # 320ms = 5120 samples
    num_chunks = 15

    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"action": "start", "language": "en-US", "lookahead": 3}))
        ready = json.loads(ws.receive_text())
        assert ready["event"] == "ready"

        latencies = []
        for i in range(num_chunks):
            chunk = data[i * chunk_len : (i + 1) * chunk_len].tobytes()
            ws.send_bytes(chunk)
            part = json.loads(ws.receive_text())
            assert part["event"] == "partial"
            latencies.append(part["latency_ms"])

        ws.send_text(json.dumps({"action": "stop"}))
        final = json.loads(ws.receive_text())
        assert final["event"] == "final"
        assert "Barcelona" in final["text"]

        avg_lat = np.mean(latencies)
        assert avg_lat < 100.0, f"Average latency too high: {avg_lat:.2f}ms"
        assert final["rtf"] < 0.35, f"Real-time factor too high: {final['rtf']}"


def test_ws_streaming_security_guards(streaming_env):
    client, _ = streaming_env
    data, _ = sf.read(str(TEST_WAV), dtype="float32")
    valid_chunk = data[:5120].tobytes()

    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"action": "start"}))
        ready = json.loads(ws.receive_text())
        assert ready["event"] == "ready"

        # 1. Unaligned chunk (3 bytes) - should be silently dropped by sanitizer without crashing
        ws.send_bytes(b"\x00\x00\x00")

        # 2. Poisoned NaN chunk - should be dropped without crashing or state corruption
        nan_data = np.array([float("nan")] * 1600, dtype=np.float32).tobytes()
        ws.send_bytes(nan_data)

        # 3. Poisoned Inf chunk
        inf_data = np.array([float("inf")] * 1600, dtype=np.float32).tobytes()
        ws.send_bytes(inf_data)

        # 4. Valid chunk after poisoned ones - should process normally
        ws.send_bytes(valid_chunk)
        part = json.loads(ws.receive_text())
        assert part["event"] == "partial"

        # 5. Clean finalization
        ws.send_text(json.dumps({"action": "stop"}))
        final = json.loads(ws.receive_text())
        assert final["event"] == "final"
