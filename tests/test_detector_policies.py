"""Gates: detector behaviour; pending queue, priority, tie-breaking, expiry, coalescing."""
import numpy as np

from src.detector import CusumDetector, Signal
from src.policies import PendingQueue, Proposed, TokenBucket, DriftOnly

D = dict(allowance=0.25, threshold=6.0, cooldown=5, sd_floor=0.05, loss_clip=5.0, refresh_window=5,
         credits=3, expiry=15, severity_clip=3.0)


def test_constant_signal_no_alarm():
    d = CusumDetector(0.5, 0.05, D)
    assert not any(d.update(0.5).flag for _ in range(1000))


def test_jump_alarms_after_onset_then_cooldown():
    d = CusumDetector(0.5, 0.05, D)
    flags = [d.update(0.5 if t < 50 else 0.8).flag for t in range(80)]
    first = flags.index(True)
    assert 50 <= first <= 52
    assert not any(flags[first + 1:first + 6])             # cooldown 5 steps
    # after a flag the reference refreshes to recent losses, so a persistent regime stops alarming
    assert sum(flags) <= 3


def test_severity_clipped():
    d = CusumDetector(0.5, 0.05, D)
    s = d.update(5.0)                                      # z = 90 -> g/threshold = 14.9
    assert s.flag and s.severity == 3.0


def test_nonfinite_loss_fails_clearly():
    import pytest
    d = CusumDetector(0.5, 0.05, D)
    with pytest.raises(FloatingPointError):
        d.update(float("nan"))


def _cfg():
    return {"num_clients": 4, "detector": D, "policy": {"eta": 1.0, "wait_norm": 10, "periodic_interval": 2}}


def sig(flags, sev):
    return [Signal(bool(f), s, 0, 0, 0, False) for f, s in zip(flags, sev)]


def test_priority_tiebreak_and_simultaneous_flags():
    p = Proposed(_cfg(), 0)
    p.observe(0, sig([1, 1, 1, 0], [1.5, 2.0, 1.5, 0]))
    assert p.candidates(0) == [1, 0, 2]                    # severity desc, ties by client id


def test_waiting_time_raises_priority():
    p = Proposed(_cfg(), 0)
    p.observe(0, sig([1, 0, 0, 0], [1.0, 0, 0, 0]))
    p.observe(10, sig([0, 1, 0, 0], [0, 1.8, 0, 0]))
    # client 0: 1.0 + 1.0*min(10/10,1) = 2.0 ; client 1: 1.8 + 0
    assert p.candidates(10) == [0, 1]
    q = Proposed(_cfg(), 0, use_wait=False)
    q.observe(0, sig([1, 0, 0, 0], [1.0, 0, 0, 0]))
    q.observe(10, sig([0, 1, 0, 0], [0, 1.8, 0, 0]))
    assert q.candidates(10) == [1, 0]


def test_drift_only_fifo():
    p = DriftOnly(_cfg(), 0)
    p.observe(0, sig([0, 0, 1, 0], [0, 0, 3.0, 0]))
    p.observe(1, sig([1, 0, 0, 0], [1.0, 0, 0, 0]))
    assert p.candidates(1) == [2, 0]


def test_credits_consumed_and_expiry_and_coalesce():
    q = PendingQueue(3, 15)
    q.flag(0, 1.2, 0)
    q.served(0, 1); q.served(0, 2)
    assert q.credits_of(0) == 1
    q.flag(0, 2.5, 3)                                      # coalesce: larger severity, credits reset to 3
    assert q.req[0]["severity"] == 2.5 and q.credits_of(0) == 3
    q.flag(1, 1.0, 5)
    q.served(0, 4)
    assert q.credits_of(0) == 2
    assert q.expire(18) == []                              # client 0 last active at 4, client 1 at 5
    assert q.expire(19) == [0] and q.expire(20) == [1]     # unserved for 15 steps -> expired
    assert q.expired_log == [(19, 0, 2), (20, 1, 3)]       # expiry logs remaining credits
    q.flag(2, 1.0, 30)
    for t in (31, 32, 33):
        q.served(2, t)
    assert 2 not in q.req                                  # three services consume all credits


def test_select_admits_prefix_only():
    p = Proposed(_cfg(), 0)
    cost = lambda m: 0 if m == 0 else (2 * m + 4) * 10
    tb = TokenBucket(80, 240, 1e9, cost)
    tb.refill()
    got = p.select(sig([1, 1, 0, 1], [2.0, 1.0, 0, 3.0]), tb, 0)
    # order by priority: 3 (sev 3), 0 (sev 2), 1 (sev 1); cost(2)=80 <= 80 tokens < cost(3)=100
    assert got == [3, 0]
    assert tb.tokens == 0
