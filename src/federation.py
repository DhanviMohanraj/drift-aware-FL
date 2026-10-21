"""Weighted FedAvg, client FIFO buffers, explicit deployed model versions, warm-start."""
from __future__ import annotations

from collections import deque

import numpy as np

from .model import get_state, mean_bce, payload_bytes, train_local


def aggregate(states: list[dict], examples: list[int]) -> dict:
    """aggregate(states, examples) -> global_state, weighted by examples actually trained."""
    if not states:
        raise ValueError("aggregate() needs at least one state; callers keep the old state when none are admitted")
    w = np.asarray(examples, dtype=np.float64)
    if w.sum() <= 0:
        raise ValueError("total examples must be positive")
    w = w / w.sum()
    out = {}
    for k in states[0]:
        acc = sum(float(wi) * s[k].double() for wi, s in zip(w, states))
        out[k] = acc.to(states[0][k].dtype)
    return out


class FIFOBuffer:
    """Keeps at most `capacity` most recent labeled examples for one clinic."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.x, self.y = deque(), deque()

    def append(self, x: np.ndarray, y: np.ndarray):
        for xi, yi in zip(x, y):
            self.x.append(xi)
            self.y.append(yi)
            if len(self.x) > self.capacity:
                self.x.popleft()
                self.y.popleft()

    def arrays(self):
        if not self.x:
            return np.zeros((0, 0), np.float32), np.zeros(0, np.float32)
        return np.stack(self.x).astype(np.float32), np.asarray(self.y, np.float32)

    def __len__(self):
        return len(self.x)

    def state(self):
        return self.arrays()

    def load(self, st):
        self.x, self.y = deque(st[0]), deque(st[1])


class Deployment:
    """Model version actually deployed at each clinic. Changes only via a charged broadcast."""

    def __init__(self, num_clients: int, state: dict):
        self.versions = [0] * num_clients
        self.states = [state] * num_clients
        self.server_version = 0

    def broadcast(self, state: dict, ledger, t: int, model_bytes: int, kind: str = "model_broadcast"):
        self.server_version += 1
        for c in range(len(self.versions)):
            ledger.record(t, "server", c, kind, model_bytes)
            self.states[c] = state
            self.versions[c] = self.server_version


def warm_start(model, init_state, warm: list, cfg: dict, seed: int, ledger) -> tuple[dict, dict]:
    """Five shared FedAvg rounds on client-local warm-start records (all clinics participate).

    Ledger records use time = -1 and kinds warm_download / warm_upload / warm_broadcast.
    Returns (state, info).
    """
    P = payload_bytes(init_state)
    state = init_state
    wcfg = dict(cfg, model=dict(cfg["model"], local_epochs=cfg["warm_start"].get("local_epochs", 1)))
    K = len(warm)
    for r in range(cfg["warm_start"]["rounds"]):
        states, ns = [], []
        for c, b in enumerate(warm):
            ledger.record(-1, "server", c, "warm_download", P)
            s, n, _ = train_local(model, state, b.x, b.y, wcfg, (seed, 10_000 + r, c))
            ledger.record(-1, c, "server", "warm_upload", P)
            states.append(s)
            ns.append(n)
        state = aggregate(states, ns)
    for c in range(K):
        ledger.record(-1, "server", c, "warm_broadcast", P)
    model.load_state_dict(state)
    info = {"warm_losses": [mean_bce(model, b.x, b.y) for b in warm], "rounds": cfg["warm_start"]["rounds"]}
    return get_state(model), info
