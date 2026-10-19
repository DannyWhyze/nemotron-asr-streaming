"""Central pytest fixtures and early native environment bootstrapping.

Created: 2026-10-13
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Initialize native bindings & CUDA environment before test collection
from server.config import setup_transcribe_environment  # noqa: E402

setup_transcribe_environment()
