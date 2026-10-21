"""MLP factory, local training (fresh Adam per call), prediction and model payload size."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .utils import STREAM_INIT, STREAM_TRAIN


def make_model(in_dim: int, hidden=(64, 32), seed: int = 0) -> nn.Module:
    torch.manual_seed(int(np.random.default_rng([seed, STREAM_INIT]).integers(2**31)))
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def get_state(model: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def payload_bytes(state: dict) -> int:
    """Transmitted full-model payload: sum of numel * element_size over all tensors (incl. buffers)."""
    return int(sum(v.numel() * v.element_size() for v in state.values()))


@torch.no_grad()
def predict_proba(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(x)).squeeze(-1)).numpy().astype(np.float64)


@torch.no_grad()
def mean_bce(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    logits = model(torch.from_numpy(x)).squeeze(-1)
    return float(nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(y)).item())


def train_local(model: nn.Module, state: dict, x: np.ndarray, y: np.ndarray, cfg: dict,
                seed_key: tuple) -> tuple[dict, int, int]:
    """train_local(state, buffer, config) -> (state, examples, steps).

    Starts from `state` (the current server model), resets Adam, one local epoch with
    batch size from cfg. Shuffle order depends only on (seed, time, client) via seed_key,
    never on the policy RNG.
    """
    m = cfg["model"]
    model.load_state_dict(state)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=m["lr"])
    n = len(y)
    if n == 0:
        return get_state(model), 0, 0
    g = torch.Generator().manual_seed(int(np.random.default_rng([*seed_key, STREAM_TRAIN]).integers(2**31)))
    X, Y = torch.from_numpy(x), torch.from_numpy(y)
    steps = 0
    for _ in range(m["local_epochs"]):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, m["batch_size"]):
            b = perm[s:s + m["batch_size"]]
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(X[b]).squeeze(-1), Y[b])
            loss.backward()
            opt.step()
            steps += 1
    return get_state(model), n, steps
