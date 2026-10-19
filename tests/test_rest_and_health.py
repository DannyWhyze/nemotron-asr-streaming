"""Integration tests for OpenAI-compatible REST API and health diagnostics."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.config import ServerConfig
from server.engine import STTEngine
from server.routes.health import create_health_router
from server.routes.rest_api import create_rest_router

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_WAV = REPO_ROOT / "data" / "input" / "audio" / "bcn_weather_16k.wav"


@pytest.fixture(scope="module")
def client():
    config = ServerConfig()
    engine = STTEngine(config)
    engine.initialize()

    app = FastAPI()
    app.include_router(create_rest_router(engine))
    app.include_router(create_health_router(engine))
    app.state.engine = engine

    with TestClient(app) as c:
        yield c

    engine.shutdown()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["vram_used_mb"] > 0


def test_models_endpoint(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) >= 1


def test_openai_transcriptions_json(client):
    with open(TEST_WAV, "rb") as f:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", f, "audio/wav")},
            data={"language": "en", "response_format": "json"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "Barcelona" in data["text"]


def test_openai_transcriptions_verbose_json(client):
    with open(TEST_WAV, "rb") as f:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", f, "audio/wav")},
            data={"language": "en-US", "response_format": "verbose_json"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "Barcelona" in data["text"]
    assert "segments" in data
    assert "words" in data
