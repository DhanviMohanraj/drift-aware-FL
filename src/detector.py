"""Frozen-sentinel one-sided CUSUM. An engineering detector: no false-alarm guarantee is claimed."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Signal:
    flag: bool
    severity: float      # g / threshold at flag time, clipped; 0 when no flag
    score: float         # CUSUM statistic g after the update (before any reset)
    z: float
    loss: float
    in_cooldown: bool


class CusumDetector:
    def __init__(self, ref_mean: float, ref_sd: float, dcfg: dict):
        self.allowance = dcfg["allowance"]
        self.threshold = dcfg["threshold"]
        self.cooldown_len = dcfg["cooldown"]
        self.sd_floor = dcfg["sd_floor"]
        self.clip = dcfg["loss_clip"]
        self.sev_clip = dcfg["severity_clip"]
        self.ref_mean, self.ref_sd = float(ref_mean), float(ref_sd)
        self.g = 0.0
        self.cooldown = 0
        self.history = deque(maxlen=dcfg["refresh_window"])

    @classmethod
    def from_calibration(cls, chunk_losses, dcfg):
        l = np.clip(np.asarray(chunk_losses, float), 0, dcfg["loss_clip"])
        return cls(l.mean(), l.std(ddof=1) if len(l) > 1 else 0.0, dcfg)

    def update(self, loss: float, time: int | None = None) -> Signal:
        loss = float(np.clip(loss, 0.0, self.clip))
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite sentinel loss")
        self.history.append(loss)
        z = (loss - self.ref_mean) / max(self.ref_sd, self.sd_floor)
        self.g = max(0.0, self.g + z - self.allowance)
        score = self.g
        in_cd = self.cooldown > 0
        if in_cd:
            self.cooldown -= 1
        if self.g > self.threshold and not in_cd:
            sev = min(self.g / self.threshold, self.sev_clip)
            self.g = 0.0
            h = np.asarray(self.history)
            self.ref_mean = float(h.mean())
            self.ref_sd = float(h.std(ddof=1)) if len(h) > 1 else self.ref_sd
            self.cooldown = self.cooldown_len
            return Signal(True, sev, score, z, loss, False)
        return Signal(False, 0.0, score, z, loss, in_cd)

    def state(self):
        return dict(ref_mean=self.ref_mean, ref_sd=self.ref_sd, g=self.g, cooldown=self.cooldown,
                    history=list(self.history))

    def load(self, st):
        self.ref_mean, self.ref_sd, self.g, self.cooldown = st["ref_mean"], st["ref_sd"], st["g"], st["cooldown"]
        self.history.clear()
        self.history.extend(st["history"])
