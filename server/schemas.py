"""Pydantic schemas for REST endpoints and WebSocket events."""

from typing import Literal

from pydantic import BaseModel, Field


class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float


class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    text: str


class VerboseTranscriptionResponse(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    words: list[TranscriptionWord] = Field(default_factory=list)


class StreamInitAction(BaseModel):
    action: Literal["start"] = "start"
    language: str = "en-US"
    chunk_ms: int = 320
    lookahead: int = 3


class StreamReadyEvent(BaseModel):
    event: Literal["ready"] = "ready"
    session_id: str
    sample_rate: int = 16000


class StreamPartialEvent(BaseModel):
    event: Literal["partial"] = "partial"
    committed: str
    tentative: str
    latency_ms: float = 0.0


class StreamFinalEvent(BaseModel):
    event: Literal["final"] = "final"
    text: str
    duration_s: float
    rtf: float


class StreamBusyEvent(BaseModel):
    event: Literal["busy"] = "busy"
    code: int = 4029
    message: str = "Model pool exhausted; server is busy. Please retry shortly."


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "initializing"]
    uptime_seconds: float
    backend: str
    device: str | None = None
    vram_used_mb: float
    vram_total_mb: float
    active_sessions: int
    pool_size: int = 1
    available_models: int = 1
