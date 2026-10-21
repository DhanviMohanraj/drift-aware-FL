"""Small shared helpers: config loading, hashing, atomic writes, environment info."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import platform
import subprocess
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike, overrides: dict | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(memoryview(a.tobytes()))
    return h.hexdigest()[:16]


def atomic_write_bytes(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, sort_keys=True, default=str).encode())


def atomic_write_pickle(path, obj) -> None:
    atomic_write_bytes(path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def atomic_write_csv(path, df) -> None:
    atomic_write_bytes(path, df.to_csv(index=False).encode())


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def environment_info() -> dict:
    import numpy, pandas, scipy, sklearn, torch, matplotlib
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cuda": torch.cuda.is_available(),
        "versions": {
            "torch": torch.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__,
        },
    }


def rng_for(*key: int):
    """Deterministic numpy Generator keyed by a tuple of non-negative ints."""
    import numpy as np
    return np.random.default_rng([int(k) for k in key])


# Separate RNG streams (plan: data, drift, initialization, policy never share draws).
STREAM_DATA, STREAM_DRIFT, STREAM_INIT, STREAM_POLICY, STREAM_TRAIN, STREAM_SPLIT = 1, 2, 3, 4, 5, 6
SPLIT_CODES = {"dev": 0, "warm": 1, "calib": 2, "final": 3}
