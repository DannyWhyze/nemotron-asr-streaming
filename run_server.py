"""CLI entrypoint to run the Enterprise transcribe.cpp STT Server."""

import argparse

import uvicorn

from server.config import get_config


def main():
    parser = argparse.ArgumentParser(description="Start the Enterprise STT Server.")
    parser.add_argument("--host", default=None, help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default: 8001)")
    parser.add_argument("--backend", choices=["auto", "cuda", "cpu"], default=None, help="Compute backend")
    parser.add_argument("--pool-size", type=int, default=None, help="Resident model pool size in VRAM (default: 3)")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload for development")
    args = parser.parse_args()

    cfg = get_config()
    host = args.host or cfg.host
    port = args.port or cfg.port

    if args.backend:
        cfg.backend = args.backend
    if args.pool_size:
        cfg.model_pool_size = args.pool_size

    print("=" * 70)
    print("  Nemotron 3.5 ASR Enterprise STT Server (transcribe.cpp)")
    print("=" * 70)
    print(f"  Bind Address: http://{host}:{port}")
    print(f"  Backend:      {cfg.backend.upper()}")
    print(f"  Model:        {cfg.model_file}")
    print(f"  Model Pool:   {cfg.model_pool_size} resident instances")
    print(f"  Chunked WS:   ws://{host}:{port}/ws/chunked")
    print(f"  Legacy WS:    ws://{host}:{port}/ws/stt")
    print(f"  Stream WS:    ws://{host}:{port}/ws/stream")
    print(f"  OpenAI REST:  http://{host}:{port}/v1/audio/transcriptions")
    print(f"  Docs:         http://{host}:{port}/docs")
    print("=" * 70)

    uvicorn.run("server.main:app", host=host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
