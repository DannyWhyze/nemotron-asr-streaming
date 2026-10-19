"""Enterprise asynchronous producer-consumer streaming WebSocket endpoint."""

import asyncio
import json
import logging
import time
import uuid

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import transcribe_cpp
from server.engine import ModelPoolEngine, ModelPoolExhaustedError
from server.sanitizer import sanitize_pcm_chunk
from server.schemas import (
    StreamBusyEvent,
    StreamFinalEvent,
    StreamPartialEvent,
    StreamReadyEvent,
)

logger = logging.getLogger(__name__)


def create_streaming_ws_router(engine: ModelPoolEngine) -> APIRouter:
    router = APIRouter(tags=["streaming-ws"])

    @router.websocket("/ws/stream")
    async def streaming_ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        session_id = str(uuid.uuid4())
        logger.info(f"Stream client connected: session_id={session_id}")

        # 1. Acquire resident model from the pool with timeout
        try:
            async with engine.acquire_model(timeout=2.0) as model:
                engine.active_sessions += 1
                try:
                    await _handle_streaming_session(websocket, engine, model, session_id)
                finally:
                    engine.active_sessions = max(0, engine.active_sessions - 1)
        except ModelPoolExhaustedError:
            logger.warning(f"Stream rejected: model pool exhausted (4029 Busy) for session_id={session_id}")
            try:
                await websocket.send_text(StreamBusyEvent().model_dump_json())
                await websocket.close(code=4029, reason="Model pool exhausted")
            except Exception:
                pass

    return router


async def _handle_streaming_session(
    websocket: WebSocket,
    engine: ModelPoolEngine,
    model: transcribe_cpp.Model,
    session_id: str,
):
    loop = asyncio.get_running_loop()
    language = engine.config.default_language
    lookahead = 3

    # Handshake
    try:
        init_msg = await websocket.receive_text()
        try:
            init_data = json.loads(init_msg)
            if init_data.get("language"):
                language = init_data["language"]
            if "lookahead" in init_data:
                lookahead = int(init_data["lookahead"])
        except Exception:
            pass
    except Exception as exc:
        logger.warning(f"Handshake failed for session {session_id}: {exc}")
        return

    # Initialize Parakeet Conformer streaming session
    session = model.session()
    try:
        family_options = transcribe_cpp.ParakeetStreamOptions(att_context_right=lookahead)
        stream = session.stream(family=family_options, language=language)

        await websocket.send_text(StreamReadyEvent(session_id=session_id).model_dump_json())

        audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=engine.config.max_queue_size)
        total_samples = 0
        start_t = time.perf_counter()

        # Producer Task
        async def producer():
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"]:
                    chunk_bytes = message["bytes"]
                    arr = sanitize_pcm_chunk(
                        chunk_bytes,
                        max_size_bytes=engine.config.max_chunk_size_bytes,
                    )
                    if arr is None:
                        continue

                    # Queue with drop-oldest on congestion
                    if audio_queue.full():
                        try:
                            _ = audio_queue.get_nowait()
                            logger.warning(f"Queue full in session {session_id}; dropped oldest chunk.")
                        except asyncio.QueueEmpty:
                            pass
                    await audio_queue.put(arr)

                elif "text" in message and message["text"]:
                    try:
                        data = json.loads(message["text"])
                        if data.get("action") == "stop":
                            await audio_queue.put(None)  # Sentinel
                            break
                    except Exception:
                        pass

        # Consumer Task
        async def consumer():
            nonlocal total_samples
            while True:
                item = await audio_queue.get()
                if item is None:
                    break

                # Batch drain any queued chunks for low latency
                chunks = [item]
                stop_signal = False
                while not audio_queue.empty():
                    try:
                        nxt = audio_queue.get_nowait()
                        if nxt is None:
                            stop_signal = True
                            break
                        chunks.append(nxt)
                    except asyncio.QueueEmpty:
                        break

                combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
                total_samples += len(combined)

                t0 = time.perf_counter()
                await loop.run_in_executor(engine.executor, stream.feed, combined)
                text_res = stream.text()
                latency_ms = (time.perf_counter() - t0) * 1000.0

                committed = getattr(text_res, "committed", "") or ""
                tentative = getattr(text_res, "tentative", "") or ""

                partial_event = StreamPartialEvent(
                    committed=committed,
                    tentative=tentative,
                    latency_ms=round(latency_ms, 1),
                )
                await websocket.send_text(partial_event.model_dump_json())

                if stop_signal:
                    break

        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())

        # Wait until producer completes or error occurs
        done, pending = await asyncio.wait(
            [producer_task, consumer_task],
            return_when=asyncio.FIRST_EXCEPTION,
        )

        for task in done:
            if task.exception():
                raise task.exception()

        # If producer finished normally, wait for consumer to drain
        if not consumer_task.done():
            await consumer_task

        # Finalize inference
        await loop.run_in_executor(engine.executor, stream.finalize)
        final_text_res = stream.text()
        final_str = getattr(final_text_res, "full", "") or str(final_text_res or "")
        duration_s = total_samples / 16000.0
        wall_time = time.perf_counter() - start_t
        rtf = wall_time / duration_s if duration_s > 0 else 0.0

        final_event = StreamFinalEvent(
            text=final_str,
            duration_s=round(duration_s, 2),
            rtf=round(rtf, 4),
        )
        await websocket.send_text(final_event.model_dump_json())

    except (WebSocketDisconnect, asyncio.CancelledError):
        logger.info(f"Stream client disconnected: session_id={session_id}")
        session.cancel()
    except Exception as exc:
        logger.error(f"Error in streaming session {session_id}: {exc}", exc_info=True)
        session.cancel()
    finally:
        session.close()
