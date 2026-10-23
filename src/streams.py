"""Client data streams. Stream.next(client_id, time) -> Batch(x, y, sample_ids).

Synthetic: deterministic sample(client, time, seed) with separate draws for dev /
warm-start / calibration / final splits. Pre-drift draws are identical across
paired scenarios because drift is applied as a transform of the same base draws
(S5 is the exception: it uses class-conditional resampling, logged in its stream info).

Clinical: finite, non-overlapping, seeded chunks of each artificial clinic's final
records. Patients are never recycled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .drift import Scenario, is_active, prevalence, rotation_angle
from .utils import SPLIT_CODES, STREAM_DATA, rng_for


@dataclass
class Batch:
    x: np.ndarray          # float32 [n, d]
    y: np.ndarray          # float32 [n]
    sample_ids: np.ndarray  # int64 [n]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _sid(split: int, client: int, t: int, n: int) -> np.ndarray:
    return (np.int64(split) * 10**9 + np.int64(client) * 10**8 + np.int64(t) * 10**5
            + np.arange(n, dtype=np.int64))


class SyntheticWorld:
    """Seeded logistic generator for four heterogeneous clinics."""

    def __init__(self, cfg: dict, seed: int, scenario: Scenario):
        s = cfg["synthetic"]
        self.cfg, self.seed, self.scenario = cfg, seed, scenario
        self.F = s["n_features"]
        self.n = s["batch_per_step"]
        self.flip = s["label_flip"]
        g = rng_for(seed, STREAM_DATA, 999)
        w = g.normal(0.0, s["weight_sd"], self.F)
        ang = g.uniform(0, 2 * np.pi)
        w[:2] = s["lead_weight_norm"] * np.array([np.cos(ang), np.sin(ang)])
        self.w = w
        self.offsets = g.normal(0.0, s["client_offset_sd"], (cfg["num_clients"], self.F))
        if s.get("offset_exclude_rotated", False):
            # Optional variant (evaluated on dev streams, NOT used in core): no clinic offset on the
            # two rotated coordinates.
            self.offsets[:, :2] = 0.0
        self.b0 = self._fit_intercept(s["base_prevalence"])
        self.num_clients = cfg["num_clients"]
        self.info = {"generator_seed": seed, "weights": self.w.tolist(), "intercept": self.b0,
                     "offsets": self.offsets.tolist(),
                     "resampling": "class-conditional (S5 only)" if prevalence(scenario, 0) is not None else "none"}

    def _fit_intercept(self, target):
        g = rng_for(self.seed, STREAM_DATA, 998)
        xs = [g.normal(size=(4000, self.F)) + o for o in self.offsets]
        x = np.vstack(xs)
        lo, hi = -10.0, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if _sigmoid(x @ self.w + mid).mean() < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def weights_at(self, client: int, t: int) -> np.ndarray:
        a = np.deg2rad(rotation_angle(self.scenario, client, t))
        w = self.w.copy()
        c, s = np.cos(a), np.sin(a)
        w[0], w[1] = c * self.w[0] - s * self.w[1], s * self.w[0] + c * self.w[1]
        return w

    def _joint(self, rng, client, t, n, apply_drift):
        z = rng.normal(size=(n, self.F))
        u = rng.random(n)
        flip = rng.random(n) < self.flip
        x = z + self.offsets[client]
        if apply_drift and is_active(self.scenario, "covariate", client, t):
            idx = self.scenario.params["features"]
            x[:, idx] += self.scenario.params["shift"]      # feature SD is 1 by construction
        w = self.weights_at(client, t) if apply_drift else self.w
        y = ((u < _sigmoid(x @ w + self.b0)) ^ flip).astype(np.float32)
        return x.astype(np.float32), y

    def sample(self, client: int, t: int, split: str = "final", n: int | None = None) -> Batch:
        n = n or self.n
        code = SPLIT_CODES[split]
        rng = rng_for(self.seed, STREAM_DATA, code, client, t)
        drift = split in ("final", "dev")
        pi = prevalence(self.scenario, t) if drift else None
        if pi is None:
            x, y = self._joint(rng, client, t, n, drift)
        else:  # class-conditional sampling: p(x|y) fixed, prior changed
            k = 8
            while True:
                xp, yp = self._joint(rng, client, t, k * n, drift)
                npos = int(round(n * pi))
                pos, neg = np.flatnonzero(yp == 1), np.flatnonzero(yp == 0)
                if len(pos) >= npos and len(neg) >= n - npos:
                    sel = np.sort(np.concatenate([pos[:npos], neg[:n - npos]]))
                    x, y = xp[sel], yp[sel]
                    break
                k *= 2
        return Batch(x, y, _sid(code, client, t, n))

    # -- splits used outside the final stream --
    def warm_data(self, client: int) -> Batch:
        m = self.cfg["synthetic"]["warm_per_client"]
        parts = [self.sample(client, t, "warm") for t in range(int(np.ceil(m / self.n)))]
        return _concat(parts, m)

    def calib_chunks(self, client: int) -> list[Batch]:
        return [self.sample(client, t, "calib") for t in range(self.cfg["synthetic"]["calib_chunks"])]


def _concat(parts, limit=None):
    x = np.concatenate([p.x for p in parts])
    y = np.concatenate([p.y for p in parts])
    i = np.concatenate([p.sample_ids for p in parts])
    if limit is not None:
        x, y, i = x[:limit], y[:limit], i[:limit]
    return Batch(x, y, i)


class Stream:
    """Uniform stream interface over a world for one split."""

    def __init__(self, world, split: str = "final"):
        self.world, self.split = world, split

    def next(self, client_id: int, time: int) -> Batch:
        if isinstance(self.world, SyntheticWorld):
            return self.world.sample(client_id, time, self.split)
        return self.world.chunk(client_id, time, self.split)


class ClinicalWorld:
    """Finite chunked streams over preprocessed clinical arrays (see preprocess.py)."""

    def __init__(self, cfg: dict, seed: int, scenario: Scenario, prepared: dict):
        self.cfg, self.seed, self.scenario = cfg, seed, scenario
        self.p = prepared
        self.num_clients = cfg["num_clients"]
        self.horizon = prepared["horizon"]
        c = cfg["clinical"]
        feats = c.get("shift_features") or [prepared["shift_feature"]]
        numeric = prepared["audit"]["numeric"]
        bad = [f for f in feats if f not in numeric]
        if bad:
            raise ValueError(f"shift_features must be numeric columns {numeric}; got {bad}")
        self.shift_features = list(feats)
        self.shift_cols = [prepared["feature_names"].index(f) for f in feats]
        # magnitude in warm-start SD units (features are standardized); default = scenario value (+1 SD)
        self.shift_magnitude = c.get("shift_magnitude")
        self.info = {"chunk_sizes": {s: [[len(ch) for ch in cl] for cl in prepared["chunks"][s]]
                                      for s in prepared["chunks"]},
                     "horizon": self.horizon, "shift_features": self.shift_features,
                     "shift_magnitude": self._magnitude()}

    def chunk(self, client: int, t: int, split: str = "final") -> Batch:
        idx = self.p["chunks"][split][client][t]
        X, y, ids = self.p["X"][split], self.p["y"][split], self.p["ids"][split]
        x = X[idx].copy()
        if split in ("final", "dev") and is_active(self.scenario, "measurement", client, t):
            x[:, self.shift_cols] += self._magnitude()   # standardized space: warm-start SD units
        return Batch(x, y[idx].copy(), ids[idx].copy())

    def _magnitude(self) -> float:
        if self.shift_magnitude is not None:
            return float(self.shift_magnitude)
        return float(self.scenario.params.get("shift", 0.0))

    def warm_data(self, client: int) -> Batch:
        idx = self.p["client_idx"]["warm"][client]
        return Batch(self.p["X"]["warm"][idx], self.p["y"]["warm"][idx], self.p["ids"]["warm"][idx])

    def calib_chunks(self, client: int) -> list[Batch]:
        idx = self.p["client_idx"]["calib"][client]
        size = max(8, int(np.median([len(c) for c in self.p["chunks"]["final"][client]])))
        out = []
        for s in range(0, len(idx) - size + 1, size):
            j = idx[s:s + size]
            out.append(Batch(self.p["X"]["calib"][j], self.p["y"]["calib"][j], self.p["ids"]["calib"][j]))
        return out


_SEMI_FIT_CACHE: dict = {}   # (preprocessing checksum, split checksum) -> fitted (w, b); identical for every run of a seed


_SEMI_FIT_CACHE: dict = {}   # (preprocessing, split checksum) -> fitted (w, b); identical for every run of a seed


class SemiSyntheticWorld(ClinicalWorld):
    """Real UCI patients and features; outcomes drawn from a logistic risk model fitted on the warm-start pool.

    - base model (w, b): sklearn LogisticRegression on warm-start X and real readmission labels (warm only).
    - semi_concept drift: the 5 largest-|w| coefficients change sign ("strongest risk factors reverse"); the
      intercept is re-solved on warm-start X so the overall outcome rate is unchanged.
    - semi_rate drift: the intercept is re-solved so the outcome rate doubles; w unchanged.
    - each record has ONE fixed uniform draw u (seeded by run seed and split), y = u < p, so outcomes are paired
      across scenarios and methods: before a drift every scenario sees identical labels.
    Drift parameters were fixed before any semi-synthetic run.
    """
    TOP_K = 5

    def __init__(self, cfg, seed, scenario, prepared):
        super().__init__(cfg, seed, scenario, prepared)
        from sklearn.linear_model import LogisticRegression
        Xw, yw = prepared["X"]["warm"], prepared["y"]["warm"]
        key = (prepared["checksums"]["preprocessing"], prepared["checksums"]["split"])
        if key not in _SEMI_FIT_CACHE:   # the fit depends only on the warm-start pool, so it is reused
            lr = LogisticRegression(C=1.0, max_iter=3000).fit(Xw, yw)
            _SEMI_FIT_CACHE[key] = (lr.coef_[0].astype(np.float64), float(lr.intercept_[0]))
        self.w, self.b = _SEMI_FIT_CACHE[key]
        base_rate = float(_sigmoid(Xw @ self.w + self.b).mean())
        top = np.argsort(-np.abs(self.w))[:self.TOP_K]
        self.w_concept = self.w.copy()
        self.w_concept[top] *= -1
        self.b_concept = self._solve_b(Xw, self.w_concept, base_rate)
        self.b_rate = self._solve_b(Xw, self.w, min(0.95, 2 * base_rate))
        self.u = {s: rng_for(seed, STREAM_DATA, 4242, SPLIT_CODES[s]).random(len(prepared["y"][s]))
                  for s in prepared["y"]}
        names = prepared["feature_names"]
        self.info.update({"semi_base_rate": base_rate, "semi_top_features": [names[i] for i in top],
                          "semi_rate_after": min(0.95, 2 * base_rate)})

    @staticmethod
    def _solve_b(X, w, target):
        lo, hi = -20.0, 20.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if _sigmoid(X @ w + mid).mean() < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def _labels(self, x, split, idx, client, t):
        w, b = self.w, self.b
        if split in ("final", "dev"):
            if is_active(self.scenario, "semi_concept", client, t):
                w, b = self.w_concept, self.b_concept
            elif is_active(self.scenario, "semi_rate", client, t):
                b = self.b_rate
        p = _sigmoid(x.astype(np.float64) @ w + b)
        return (self.u[split][idx] < p).astype(np.float32)

    def chunk(self, client, t, split="final"):
        idx = self.p["chunks"][split][client][t]
        x = self.p["X"][split][idx].copy()
        return Batch(x, self._labels(x, split, idx, client, t), self.p["ids"][split][idx].copy())

    def warm_data(self, client):
        idx = self.p["client_idx"]["warm"][client]
        x = self.p["X"]["warm"][idx]
        return Batch(x, self._labels(x, "warm", idx, client, -1), self.p["ids"]["warm"][idx])

    def calib_chunks(self, client):
        out = []
        pos = {sid: i for i, sid in enumerate(self.p["ids"]["calib"])}
        for b in super().calib_chunks(client):
            idx = np.array([pos[s] for s in b.sample_ids])
            out.append(Batch(b.x, self._labels(b.x, "calib", idx, client, -1), b.sample_ids))
        return out
