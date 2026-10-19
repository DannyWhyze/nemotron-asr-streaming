"""Comprehensive end-to-end client test suite verifying all server endpoints."""

import json
from pathlib import Path

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from server.main import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_WAV = REPO_ROOT / "data" / "input" / "audio" / "bcn_weather_16k.wav"


@pytest.fixture(scope="module")
def live_client():
    app = create_app()
    with TestClient(app) as client:
        yield client


def test_full_pipeline(live_client):
    # 1. Health
    h = live_client.get("/health").json()
    assert h["status"] == "healthy"

    # 2. REST Transcription
    with open(TEST_WAV, "rb") as f:
        rest_resp = live_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("bcn.wav", f, "audio/wav")},
            data={"language": "en-US"},
        ).json()
    assert "Barcelona" in rest_resp["text"]

    # 3. Legacy VAD WebSocket
    data, _ = sf.read(str(TEST_WAV), dtype="float32")
    pcm_bytes = data.tobytes()

    with live_client.websocket_connect("/ws/stt") as ws:
        ws.send_bytes(pcm_bytes)
        ws_text = ws.receive_text()
        assert "Barcelona" in ws_text

    # 4. Incremental Stream WebSocket
    with live_client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"action": "start", "language": "en-US"}))
        assert json.loads(ws.receive_text())["event"] == "ready"

        ws.send_bytes(pcm_bytes)
        partial = json.loads(ws.receive_text())
        assert partial["event"] == "partial"

        ws.send_text(json.dumps({"action": "stop"}))
        final = json.loads(ws.receive_text())
        assert final["event"] == "final"
        assert "Barcelona" in final["text"]
