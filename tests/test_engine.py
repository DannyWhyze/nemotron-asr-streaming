"""Integration test for STTEngine with reference audio."""

from pathlib import Path

import pytest
import soundfile as sf

from server.config import ServerConfig
from server.engine import STTEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_WAV = REPO_ROOT / "data" / "input" / "audio" / "bcn_weather_16k.wav"


@pytest.fixture(scope="module")
def engine():
    cfg = ServerConfig(backend="auto")
    eng = STTEngine(cfg)
    eng.initialize()
    yield eng
    eng.shutdown()


def test_engine_telemetry(engine):
    telemetry = engine.get_telemetry()
    assert telemetry["backend"] is not None
    assert telemetry["is_ready"] is True
    assert "RTX" in telemetry["device"] or "Ryzen" in telemetry["device"] or "CPU" in telemetry["device"]


def test_engine_transcribe_pcm(engine):
    data, sr = sf.read(str(TEST_WAV), dtype="float32")
    assert sr == 16000
    pcm_bytes = data.tobytes()

    text, duration = engine.transcribe_pcm(pcm_bytes, language="en-US")
    assert "Barcelona" in text
    assert duration > 10.0


def test_engine_transcribe_file(engine):
    result = engine.transcribe_audio_file(TEST_WAV, language="en-US")
    assert "Barcelona" in result["text"]
    assert result["duration"] > 10.0
