"""Token-bucket budget, pending request queue and the six scheduling methods (+ ablations).

All policies share one interface: Policy.select(signals, budget, time) -> admitted client ids.
Only KnownTimePolicy receives event metadata (privileged reference, not an upper bound).
"""
from __future__ import annotations

import numpy as np

from .utils import STREAM_POLICY, rng_for


class TokenBucket:
    """Starts empty, refills uniformly, capacity = `bucket_full_rounds` full transactions,
    hard cumulative cap. Feasibility: spending only ever deducts affordable non-negative
    costs, so cumulative spend <= cumulative refill <= cap (see tests/test_accounting.py)."""

    def __init__(self, refill: float, capacity: float, cap_total: float, cost_fn):
        self.refill_rate, self.capacity, self.cap_total = float(refill), float(capacity), float(cap_total)
        self.cost = cost_fn
        self.tokens = 0.0
        self.refilled = 0.0
        self.spent = 0.0
        self.unlimited = False

    def refill(self):
        add = max(0.0, min(self.refill_rate, self.cap_total - self.refilled))
        self.refilled += add
        self.tokens = min(self.capacity, self.tokens + add)

    def affordable(self, m: int) -> bool:
        c = self.cost(m)
        return c <= self.tokens + 1e-9 and self.spent + c <= self.cap_total + 1e-9

    def admit(self, candidates: list[int]) -> list[int]:
        """Largest affordable prefix of the ordered candidate list, charged once."""
        m = len(candidates)
        while m > 0 and not self.affordable(m):
            m -= 1
        if m == 0:
            return []
        c = self.cost(m)
        self.tokens -= c
        self.spent += c
        assert self.tokens >= -1e-6
        return list(candidates[:m])

    def state(self):
        return dict(tokens=self.tokens, refilled=self.refilled, spent=self.spent)

    def load(self, st):
        self.tokens, self.refilled, self.spent = st["tokens"], st["refilled"], st["spent"]


class UnlimitedBudget(TokenBucket):
    def __init__(self, cost_fn):
        super().__init__(0, 0, float("inf"), cost_fn)
        self.unlimited = True

    def refill(self):
        pass

    def affordable(self, m):
        return True

    def admit(self, candidates):
        c = self.cost(len(candidates))
        self.spent += c
        return list(candidates)


class PendingQueue:
    """Drift service requests: severity, remaining credits, waiting clock and expiry clock."""

    def __init__(self, credits: int, expiry: int):
        self.credits, self.expiry = credits, expiry
        self.req: dict[int, dict] = {}
        self.expired_log: list[tuple] = []

    def flag(self, c: int, severity: float, t: int):
        if c in self.req:          # coalesce: keep larger severity, reset credits to at most `credits`
            r = self.req[c]
            r["severity"] = max(r["severity"], severity)
            r["credits"] = self.credits
            r["activity"] = t
        else:
            self.req[c] = dict(severity=severity, credits=self.credits, since=t, activity=t, flag_time=t)

    def expire(self, t: int):
        out = []
        for c in sorted(self.req):
            if t - self.req[c]["activity"] >= self.expiry:
                self.expired_log.append((t, c, self.req[c]["credits"]))
                out.append(c)
                del self.req[c]
        return out

    def served(self, c: int, t: int):
        r = self.req.get(c)
        if r is None:
            return
        r["credits"] -= 1
        r["since"] = t
        r["activity"] = t
        if r["credits"] <= 0:
            del self.req[c]

    def wait(self, c: int, t: int) -> int:
        return t - self.req[c]["since"]

    def credits_of(self, c):
        return self.req[c]["credits"] if c in self.req else 0

    def state(self):
        return dict(req={k: dict(v) for k, v in self.req.items()}, expired_log=list(self.expired_log))

    def load(self, st):
        self.req = {int(k): dict(v) for k, v in st["req"].items()}
        self.expired_log = list(st["expired_log"])


class Policy:
    name = "base"
    uses_detector = False
    budgeted = True

    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.K = cfg["num_clients"]
        self.rng = rng_for(seed, STREAM_POLICY)
        self.pending = PendingQueue(cfg["detector"]["credits"], cfg["detector"]["expiry"])
        self.just_expired: list[int] = []

    def observe(self, t, signals):  # update request state
        pass

    def candidates(self, t) -> list[int]:
        raise NotImplementedError

    def on_served(self, t, admitted):
        for c in admitted:
            self.pending.served(c, t)

    def select(self, signals, budget, time: int) -> list[int]:
        self.observe(time, signals)
        self.just_expired = self.pending.expire(time)
        cand = self.candidates(time)
        admitted = budget.admit(cand) if cand else []
        self.on_served(time, admitted)
        return admitted

    def credits(self, c):
        return self.pending.credits_of(c)

    def state(self):
        return dict(pending=self.pending.state(), rng=self.rng.bit_generator.state, extra=self._extra_state())

    def load(self, st):
        self.pending.load(st["pending"])
        self.rng.bit_generator.state = st["rng"]
        self._load_extra(st["extra"])

    def _extra_state(self):
        return {}

    def _load_extra(self, st):
        pass


class AlwaysOn(Policy):
    name, budgeted = "always_on", False

    def candidates(self, t):
        return list(range(self.K))

    def credits(self, c):
        return 0


class Periodic(Policy):
    """Every `interval` steps request all clinics in a seeded cyclic order; unserved requests persist."""
    name = "periodic"

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.interval = cfg["policy"]["periodic_interval"]
        self.offset = int(self.rng.integers(self.K))
        self.queue: list[int] = []
        self.cycle = 0

    def observe(self, t, signals):
        if t % self.interval == self.interval - 1:
            order = [(self.offset + self.cycle + i) % self.K for i in range(self.K)]
            self.cycle += 1
            self.queue = order

    def candidates(self, t):
        return list(self.queue)

    def on_served(self, t, admitted):
        self.queue = [c for c in self.queue if c not in admitted]

    def credits(self, c):
        return int(c in self.queue)

    def _extra_state(self):
        return dict(queue=list(self.queue), cycle=self.cycle, offset=self.offset)

    def _load_extra(self, st):
        self.queue, self.cycle, self.offset = list(st["queue"]), st["cycle"], st["offset"]


class RandomPolicy(Policy):
    name = "random"

    def candidates(self, t):
        return [int(c) for c in self.rng.permutation(self.K)]


class DriftOnly(Policy):
    """Internal control: same detector and credits as proposed, FIFO order. Not FedConD/FedDrift."""
    name, uses_detector = "drift_only", True

    def observe(self, t, signals):
        for c, s in enumerate(signals):
            if s.flag:
                self.pending.flag(c, s.severity, t)

    def candidates(self, t):
        return sorted(self.pending.req, key=lambda c: (self.pending.req[c]["since"], c))


class KnownTime(DriftOnly):
    """Privileged: flags at true onsets (severity 1). Same credits, FIFO order, budget and training."""
    name, uses_detector = "known_time", False

    def __init__(self, cfg, seed, event_manifest: list[dict]):
        super().__init__(cfg, seed)
        self.onsets = {}
        for e in event_manifest:
            self.onsets.setdefault(e["onset"], []).append(e["client"])

    def observe(self, t, signals):
        for c in self.onsets.get(t, []):
            self.pending.flag(c, 1.0, t)


class Proposed(DriftOnly):
    """priority = min(severity, 3) + eta * min(wait / wait_norm, 1); ties broken by client id."""
    name = "proposed"

    def __init__(self, cfg, seed, use_severity=True, use_wait=True):
        super().__init__(cfg, seed)
        p = cfg["policy"]
        self.eta = p["eta"] if use_wait else 0.0
        self.wsev = 1.0 if use_severity else 0.0
        self.wait_norm = p["wait_norm"]
        self.sev_clip = cfg["detector"]["severity_clip"]

    def priority(self, c, t):
        r = self.pending.req[c]
        return self.wsev * min(r["severity"], self.sev_clip) + self.eta * min(self.pending.wait(c, t) / self.wait_norm, 1.0)

    def candidates(self, t):
        return sorted(self.pending.req, key=lambda c: (-self.priority(c, t), c))


METHODS = ["always_on", "periodic", "random", "drift_only", "known_time", "proposed"]
ABLATIONS = ["proposed_nosev", "proposed_nowait"]


def make_policy(method: str, cfg: dict, seed: int, event_manifest: list[dict]) -> Policy:
    if method == "always_on":
        return AlwaysOn(cfg, seed)
    if method == "periodic":
        return Periodic(cfg, seed)
    if method == "random":
        return RandomPolicy(cfg, seed)
    if method == "drift_only":
        return DriftOnly(cfg, seed)
    if method == "known_time":
        return KnownTime(cfg, seed, event_manifest)
    if method == "proposed":
        return Proposed(cfg, seed)
    if method == "proposed_nosev":
        p = Proposed(cfg, seed, use_severity=False)
        p.name = method
        return p
    if method == "proposed_nowait":
        p = Proposed(cfg, seed, use_wait=False)
        p.name = method
        return p
    raise KeyError(method)
