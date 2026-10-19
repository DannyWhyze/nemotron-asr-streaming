"""Chunked audio WebSocket endpoint for discrete audio segment transcription."""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.engine import ModelPoolEngine, ModelPoolExhaustedError
from server.sanitizer import sanitize_pcm_chunk

logger = logging.getLogger(__name__)


def create_chunked_ws_router(engine: ModelPoolEngine) -> APIRouter:
    router = APIRouter(tags=["chunked-ws"])

    @router.websocket("/ws/stt")
    @router.websocket("/ws/chunked")
    async def chunked_ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        engine.active_sessions += 1
        logger.info("Client connected to chunked audio WebSocket")
        loop = asyncio.get_running_loop()

        try:
            while True:
                data = await websocket.receive_bytes()
                if not data:
                    continue

                # 1. Sanitize incoming PCM bytes
                sanitized_arr = sanitize_pcm_chunk(
                    data,
                    max_size_bytes=engine.config.max_chunk_size_bytes,
                )
                if sanitized_arr is None:
                    # Invalid or poisoned chunk: ignore safely
                    await websocket.send_text("")
                    continue

                # 2. Acquire a pooled model for chunk inference
                try:
                    async with engine.acquire_model(timeout=2.0) as pooled_model:
                        text, duration = await loop.run_in_executor(
                            engine.executor,
                            engine.transcribe_pcm,
                            sanitized_arr.tobytes(),
                            engine.config.default_language,
                            pooled_model,
                        )
                except ModelPoolExhaustedError:
                    logger.warning("Chunked STT rejected: model pool exhausted (4029 Busy)")
                    await websocket.send_text(
                        json.dumps({"error": "busy", "code": 4029, "message": "Model pool exhausted"})
                    )
                    continue

                if text:
                    logger.debug(f"Transcribed {duration:.2f}s segment: '{text}'")
                    await websocket.send_text(text)
                else:
                    # Emits empty text for silence/short chunks under duration threshold
                    await websocket.send_text("")

        except WebSocketDisconnect:
            logger.info("Client disconnected from chunked audio WebSocket")
        except Exception as exc:
            logger.error(f"Error in chunked WebSocket session: {exc}", exc_info=True)
        finally:
            engine.active_sessions = max(0, engine.active_sessions - 1)

    return router
