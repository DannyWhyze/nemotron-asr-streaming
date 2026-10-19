"""Test model pool leasing, exhaustion handling, and concurrency limits.

Created: 2026-10-13
"""

import pytest

from server.config import ServerConfig
from server.engine import ModelPoolEngine, ModelPoolExhaustedError


@pytest.mark.anyio
async def test_model_pool_concurrency_and_exhaustion():
    config = ServerConfig(model_pool_size=2)
    engine = ModelPoolEngine(config)
    engine.initialize()

    try:
        assert len(engine.models) == 2

        # 1. Acquire both models
        async with engine.acquire_model() as m1:
            async with engine.acquire_model() as m2:
                assert m1 is not m2

                # 2. Attempting to acquire a 3rd model should raise ModelPoolExhaustedError
                with pytest.raises(ModelPoolExhaustedError):
                    async with engine.acquire_model(timeout=0.2):
                        pass

        # 3. Both models returned, can acquire again
        async with engine.acquire_model(timeout=0.1) as m:
            assert m is not None

    finally:
        engine.shutdown()
