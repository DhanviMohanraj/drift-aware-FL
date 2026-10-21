"""Gates: oracle isolation, seed isolation, resume equivalence, metric edge cases."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import Evaluator, auc_pair, detection_summary
from src.simulation import Simulation


def _sel(sim):
    return pd.DataFrame(sim.event_rows)[["client_id", "time", "selected", "flag"]]


def test_oracle_isolation(cfg):
    a = Simulation(cfg, "synthetic", "S2", "proposed", 11)
    b = Simulation(cfg, "synthetic", "S2", "proposed", 11,
                   manifest_override=[{"client": 3, "onset": 5, "end": 5, "kind": "concept"}])
    a.run(); b.run()
    pd.testing.assert_frame_equal(_sel(a), _sel(b))
    np.testing.assert_array_equal(a.evaluator.predictions().probability, b.evaluator.predictions().probability)


def test_policy_seed_isolation(cfg):
    a = Simulation(cfg, "synthetic", "S2", "random", 11, policy_seed=1)
    b = Simulation(cfg, "synthetic", "S2", "random", 11, policy_seed=2)
    assert a.warm_checksum == b.warm_checksum                       # initial weights unchanged
    for c in range(4):
        for t in (0, 13, 40):
            np.testing.assert_array_equal(a.stream.next(c, t).x, b.stream.next(c, t).x)
    a.run(); b.run()
    assert not _sel(a).selected.equals(_sel(b).selected)            # policy RNG does matter
    pa, pb = a.evaluator.predictions(), b.evaluator.predictions()
    np.testing.assert_array_equal(pa.sample_id, pb.sample_id)       # identical sample order


def test_resume_matches_uninterrupted(cfg, tmp_path):
    ck = tmp_path / "ck.pkl"
    full = Simulation(cfg, "synthetic", "S2", "proposed", 22)
    full.run()
    part = Simulation(cfg, "synthetic", "S2", "proposed", 22)
    assert part.run(checkpoint_path=ck, stop_at=20) is False and ck.exists()
    resumed = Simulation(cfg, "synthetic", "S2", "proposed", 22)    # fresh process-equivalent object
    assert resumed.run(checkpoint_path=ck) is True
    pd.testing.assert_frame_equal(full.ledger.frame(), resumed.ledger.frame())
    pd.testing.assert_frame_equal(pd.DataFrame(full.event_rows), pd.DataFrame(resumed.event_rows))
    np.testing.assert_array_equal(full.evaluator.predictions().probability,
                                  resumed.evaluator.predictions().probability)
    assert full.policy.pending.req == resumed.policy.pending.req


def test_single_class_auc_missing_with_reason():
    a, ap, why = auc_pair([0, 0, 0], [0.1, 0.2, 0.3])
    assert np.isnan(a) and np.isnan(ap) and why == "single_class"
    assert auc_pair([], [])[2] == "empty"


def test_logloss_brier_hand_values():
    e = Evaluator()
    e.record([0.8, 0.4], [1, 0], client=0, time=0)
    r = e.step_rows[0]
    assert r["log_loss"] == pytest.approx((-np.log(0.8) - np.log(0.6)) / 2)
    assert r["Brier"] == pytest.approx((0.04 + 0.16) / 2)


def test_nan_predictions_fail_clearly():
    with pytest.raises(FloatingPointError):
        Evaluator().record([np.nan, 0.2], [1, 0], client=0, time=0)


def test_empty_event_set():
    ev = pd.DataFrame({"client_id": [0, 1], "time": [3, 7], "flag": [1, 0]})
    d = detection_summary(ev, [], horizon=10, K=2, match_window=15)
    assert d["n_events"] == 0 and d["misses"] == 0 and d["false_alarms"] == 1
    assert d["false_alarms_per_100_client_steps"] == pytest.approx(5.0)
