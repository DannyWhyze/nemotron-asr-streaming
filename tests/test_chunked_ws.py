"""Integration test for /ws/chunked and /ws/stt chunked audio WebSocket endpoint."""

from pathlib import Path

import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.config import ServerConfig
from server.engine import STTEngine
from server.routes.chunked_ws import create_chunked_ws_router

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_WAV = REPO_ROOT / "data" / "input" / "audio" / "bcn_weather_16k.wav"


@pytest.fixture(scope="module")
def client():
    config = ServerConfig()
    engine = STTEngine(config)
    engine.initialize()

    app = FastAPI()
    app.include_router(create_chunked_ws_router(engine))
    app.state.engine = engine

    with TestClient(app) as c:
        yield c

    engine.shutdown()


def test_chunked_ws_stt_endpoint(client):
    data, _ = sf.read(str(TEST_WAV), dtype="float32")
    pcm_bytes = data.tobytes()

    with client.websocket_connect("/ws/stt") as ws:
        ws.send_bytes(pcm_bytes)
        response_text = ws.receive_text()
        assert "Barcelona" in response_text


def test_chunked_ws_chunked_endpoint(client):
    data, _ = sf.read(str(TEST_WAV), dtype="float32")
    pcm_bytes = data.tobytes()

    with client.websocket_connect("/ws/chunked") as ws:
        ws.send_bytes(pcm_bytes)
        response_text = ws.receive_text()
        assert "Barcelona" in response_text
