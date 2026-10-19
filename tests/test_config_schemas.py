"""Unit tests for server configuration and Pydantic schemas."""

from server.config import ServerConfig
from server.schemas import (
    HealthResponse,
    StreamFinalEvent,
    StreamInitAction,
    StreamPartialEvent,
    TranscriptionResponse,
)


def test_server_config_defaults():
    config = ServerConfig()
    assert config.host == "127.0.0.1"
    assert config.port == 8001
    assert config.backend in ("auto", "cuda", "cpu")
    assert config.default_language == "en-US"
    assert config.max_workers >= 1


def test_schemas_serialization():
    resp = TranscriptionResponse(text="Hello world")
    assert resp.model_dump() == {"text": "Hello world"}

    stream_init = StreamInitAction(action="start", language="en-US", chunk_ms=320)
    assert stream_init.chunk_ms == 320

    partial = StreamPartialEvent(event="partial", committed="Hello", tentative=" world", latency_ms=15.2)
    assert partial.committed == "Hello"

    final = StreamFinalEvent(event="final", text="Hello world", duration_s=2.5, rtf=0.05)
    assert final.rtf == 0.05

    health = HealthResponse(
        status="healthy",
        uptime_seconds=10.0,
        backend="CUDA0",
        vram_used_mb=716.3,
        vram_total_mb=16302.0,
        active_sessions=0,
    )
    assert health.status == "healthy"
