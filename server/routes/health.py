"""Health check and model catalog diagnostic endpoints."""

from fastapi import APIRouter

from server.engine import STTEngine
from server.schemas import HealthResponse


def create_health_router(engine: STTEngine) -> APIRouter:
    router = APIRouter(tags=["diagnostics"])

    @router.get("/health", response_model=HealthResponse)
    async def get_health():
        t = engine.get_telemetry()
        return HealthResponse(
            status="healthy" if t["is_ready"] else "initializing",
            uptime_seconds=t["uptime_seconds"],
            backend=t["backend"],
            device=t.get("device"),
            vram_used_mb=t["vram_used_mb"],
            vram_total_mb=t["vram_total_mb"],
            active_sessions=t["active_sessions"],
            pool_size=t.get("pool_size", 1),
            available_models=t.get("available_models", 1),
        )

    @router.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_path.name,
                    "object": "model",
                    "owned_by": "nvidia",
                    "permission": [],
                }
            ],
        }

    return router
