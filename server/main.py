"""FastAPI Application Factory with lifespan management and router registration."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import get_config
from server.engine import STTEngine
from server.routes.chunked_ws import create_chunked_ws_router
from server.routes.health import create_health_router
from server.routes.rest_api import create_rest_router
from server.routes.streaming_ws import create_streaming_ws_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing transcribe.cpp STT Engine...")
    engine: STTEngine = app.state.engine
    engine.initialize()
    logger.info(f"STT Engine ready on backend: {engine.backend_tag} ({engine.device_name})")
    yield
    logger.info("Shutting down STT Engine...")
    engine.shutdown()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    config = get_config()
    engine = STTEngine(config)

    app = FastAPI(
        title="Nemotron 3.5 ASR Enterprise STT Server",
        version="0.1.0",
        description="Low-latency ASR server powered by transcribe.cpp (CUDA/AVX2).",
        lifespan=lifespan,
    )
    app.state.engine = engine

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(create_health_router(engine))
    app.include_router(create_rest_router(engine))
    app.include_router(create_chunked_ws_router(engine))
    app.include_router(create_streaming_ws_router(engine))

    return app


app = create_app()
