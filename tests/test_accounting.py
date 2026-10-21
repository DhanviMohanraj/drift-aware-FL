"""Gates: hand-computed transaction cost, ledger == prediction, token bucket feasibility lemma."""
import numpy as np

from src.accounting import MODEL_KINDS, Accounting
from src.policies import TokenBucket
from src.simulation import Simulation


def test_hand_computed_two_client_transaction(cfg):
    # MLP 20-64-32-1 float32: params = 20*64+64 + 64*32+32 + 32+1 = 3457 -> P = 13828 bytes
    acct = Accounting(cfg, 13828)
    assert acct.transaction_cost(2) == (2 * 2 + 4) * 13828 + 2 * 8 == 110640
    assert acct.transaction_cost(4) == 12 * 13828 + 32
    assert acct.transaction_cost(0) == 0
    assert acct.monitoring_per_step() == 64


def test_model_payload_matches_hand_count(cfg):
    sim = Simulation(cfg, "synthetic", "S0", "periodic", 11)
    assert sim.P == 13828


def test_ledger_equals_predicted_cost(cfg):
    sim = Simulation(cfg, "synthetic", "S2", "random", 11)
    sim.run()
    comm, ev = sim.ledger.frame(), sim.evaluator  # step() itself asserts equality every step
    sel = {t: g.selected.sum() for t, g in __import__("pandas").DataFrame(sim.event_rows).groupby("time")}
    for t, m in sel.items():
        got = comm[(comm.time == t) & comm.message_kind.isin(MODEL_KINDS + ("selection_meta",))].payload_bytes.sum()
        assert got == sim.acct.transaction_cost(int(m))
    mon = comm[comm.message_kind == "monitor"]
    assert len(mon) == 4 * sim.H and mon.payload_bytes.eq(16).all()
    assert sim.budget.spent <= sim.budget.cap_total


def test_bucket_feasibility_randomised():
    g = np.random.default_rng(0)
    for trial in range(200):
        unit = g.uniform(1, 100)
        cost = lambda m, u=unit: (2 * m + 4) * u
        full = cost(4)
        b = g.choice([0.1, 0.25, 0.5])
        H = int(g.integers(5, 60))
        tb = TokenBucket(b * full, 2 * full, b * H * full, cost)
        for t in range(H + 10):
            tb.refill()
            k = int(g.integers(0, 5))
            tb.admit(list(g.permutation(4)[:k]))
            assert tb.tokens >= -1e-6
            assert tb.spent <= tb.refilled + 1e-6 <= tb.cap_total + 1e-6


def test_no_affordable_then_one(cfg):
    cost = lambda m: 0 if m == 0 else (2 * m + 4) * 10
    tb = TokenBucket(30, 240, 10_000, cost)
    tb.refill()
    assert tb.admit([2, 0, 1]) == []                       # 30 < cost(1)=60
    tb.refill()
    assert tb.admit([2, 0, 1]) == [2]                      # 60 affordable for exactly one
    assert tb.tokens == 0
