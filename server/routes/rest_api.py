"""OpenAI-compatible audio transcriptions REST endpoint with model-pool integration."""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from server.engine import ModelPoolEngine, ModelPoolExhaustedError
from server.schemas import TranscriptionResponse, VerboseTranscriptionResponse

logger = logging.getLogger(__name__)


def create_rest_router(engine: ModelPoolEngine) -> APIRouter:
    router = APIRouter(tags=["openai-rest"])

    @router.post("/v1/audio/transcriptions")
    async def transcribe_audio(
        file: UploadFile = File(...),
        model: str = Form(default="nemotron-3.5-asr-streaming-0.6b"),
        language: str = Form(default="en-US"),
        response_format: Literal["json", "text", "verbose_json"] = Form(default="json"),
        temperature: float = Form(default=0.0),
    ):
        suffix = Path(file.filename or "input.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_f:
            tmp_path = Path(tmp_f.name)
            content = await file.read()
            tmp_f.write(content)
            tmp_f.close()

        loop = asyncio.get_running_loop()
        try:
            # Map simple 2-letter ISO to standard BCP-47 locale
            if language in ("en", "en-US"):
                norm_lang = "en-US"
            elif language in ("de", "de-DE"):
                norm_lang = "de-DE"
            else:
                norm_lang = language

            try:
                async with engine.acquire_model(timeout=2.0) as pooled_model:
                    result = await loop.run_in_executor(
                        engine.executor,
                        engine.transcribe_audio_file,
                        tmp_path,
                        norm_lang,
                        pooled_model,
                    )
            except ModelPoolExhaustedError as exc:
                logger.warning(f"REST request rejected: {exc}")
                raise HTTPException(
                    status_code=503,
                    detail="Model pool exhausted; server is busy. Please retry.",
                    headers={"Retry-After": "1"},
                )

            if response_format == "text":
                return PlainTextResponse(result["text"])
            if response_format == "verbose_json":
                return VerboseTranscriptionResponse(**result)
            return TranscriptionResponse(text=result["text"])

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Error during audio transcription: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            if tmp_path.is_file():
                tmp_path.unlink()

    return router
