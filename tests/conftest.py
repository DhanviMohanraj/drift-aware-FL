import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402


@pytest.fixture
def cfg():
    """Core config shrunk for fast CPU tests (same code paths)."""
    return load_config(ROOT / "configs" / "core.yaml", {
        "horizon": 50,
        "synthetic": {"warm_per_client": 128, "calib_chunks": 8},
        "warm_start": {"rounds": 2, "local_epochs": 2},
        "clinical": {"min_client_final": 150, "min_client_warm": 30},
    })
