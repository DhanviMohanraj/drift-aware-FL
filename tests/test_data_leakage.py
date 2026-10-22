"""Gates: patient leakage, preprocessing leakage, unseen categories, finite non-overlapping chunks."""
import numpy as np
import pandas as pd
import pytest

from src.data import make_fixture
from src.preprocess import UNKNOWN, TabularPreprocessor, prepare_clinical


@pytest.fixture
def prepared(cfg):
    return prepare_clinical(cfg, 11, df=make_fixture(6000, seed=3), cache=False)


def test_patient_pools_disjoint(prepared):
    P = {k: set(v.tolist()) for k, v in prepared["patients"].items()}
    names = list(P)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (P[a] & P[b]), f"patient overlap between {a} and {b}"
    ids = {k: set(v.tolist()) for k, v in prepared["ids"].items()}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (ids[a] & ids[b]), f"sample overlap between {a} and {b}"


def test_one_encounter_per_patient(prepared):
    for split, n in prepared["counts"].items():
        assert n["rows"] == n["patients"], split


def test_preprocessor_fit_only_on_warm(prepared):
    fit = set(prepared["preprocessor_fit_ids"].tolist())
    assert fit == set(prepared["ids"]["warm"].tolist())
    for other in ("dev", "calib", "final"):
        assert not fit & set(prepared["ids"][other].tolist())


def test_identifiers_not_features(prepared):
    names = prepared["feature_names"]
    assert not any(n.startswith(("encounter_id", "patient_nbr", "readmitted", "diag_")) for n in names)
    assert all(np.isfinite(prepared["X"]["final"]).all() for _ in [0])


def test_unseen_category_and_missing():
    tr = pd.DataFrame({"a": [1.0, np.nan, 3.0] * 10, "c": ["x", "y", np.nan] * 10})
    pre = TabularPreprocessor(["a"], ["c"], min_count=5).fit(tr, np.arange(30))
    te = pd.DataFrame({"a": [np.nan, 2.0], "c": ["never_seen", "x"]})
    X = pre.transform(te)
    assert np.isfinite(X).all()
    names = pre.feature_names
    assert X[0, names.index(f"c={UNKNOWN}")] == 1.0
    assert X[1, names.index("c=x")] == 1.0
    assert X[0, names.index("a__isna")] == 1.0


def test_chunks_non_overlapping_and_complete(prepared, cfg):
    for split in ("final", "dev"):
        for c, chunks in enumerate(prepared["chunks"][split]):
            assert len(chunks) == cfg["horizon"]
            allidx = np.concatenate(chunks)
            assert len(allidx) == len(set(allidx.tolist())), "a record appears in two chunks"
            assert set(allidx.tolist()) == set(prepared["client_idx"][split][c].tolist()), "records dropped/recycled"


def test_clients_disjoint(prepared):
    for split, lists in prepared["client_idx"].items():
        seen = set()
        for l in lists:
            assert not seen & set(l.tolist())
            seen |= set(l.tolist())


def test_clinical_measurement_shift_only_intended(prepared, cfg):
    from src.drift import get_scenario
    from src.streams import ClinicalWorld
    w0 = ClinicalWorld(cfg, 11, get_scenario("C0"), prepared)
    w1 = ClinicalWorld(cfg, 11, get_scenario("C1"), prepared)
    j = prepared["shift_col_index"]
    for c in range(4):
        for t in (10, 39, 40, 45):
            a, b = w0.chunk(c, t), w1.chunk(c, t)
            np.testing.assert_array_equal(a.sample_ids, b.sample_ids)
            np.testing.assert_array_equal(a.y, b.y)
            d = b.x - a.x
            if c in (0, 1) and t >= 40:
                np.testing.assert_allclose(d[:, j], 1.0, rtol=1e-6)
                d[:, j] = 0
            assert np.abs(d).max() < 1e-6


def test_configurable_multi_feature_shift(prepared, cfg):
    """v2 option: several numeric features shifted by a configured magnitude; v1 default unchanged."""
    from src.drift import get_scenario
    from src.streams import ClinicalWorld
    from src.utils import deep_update
    feats = ["num_lab_procedures", "num_medications", "number_inpatient"]
    c2 = deep_update(cfg, {"clinical": {"shift_features": feats, "shift_magnitude": 3.0}})
    w0 = ClinicalWorld(cfg, 11, get_scenario("C0"), prepared)
    w2 = ClinicalWorld(c2, 11, get_scenario("C2"), prepared)
    cols = [prepared["feature_names"].index(f) for f in feats]
    for c, t in [(0, 29), (0, 30), (1, 49), (2, 49), (3, 49)]:
        d = w2.chunk(c, t).x - w0.chunk(c, t).x
        affected = {0: 30, 1: 50, 2: 70}.get(c)   # test horizon is 50
        if affected is not None and t >= affected:
            np.testing.assert_allclose(d[:, cols], 3.0, rtol=1e-6)
            d[:, cols] = 0
        assert np.abs(d).max() < 1e-6
    w1 = ClinicalWorld(cfg, 11, get_scenario("C1"), prepared)       # default stays +1 SD on one feature
    assert w1.shift_features == ["num_lab_procedures"] and w1._magnitude() == 1.0
    import pytest
    with pytest.raises(ValueError):
        ClinicalWorld(deep_update(cfg, {"clinical": {"shift_features": ["race"]}}), 11, get_scenario("C1"), prepared)


def test_fixed_dev_pool_disjoint_across_seeds(cfg):
    """With fixed_dev_pool, dev patients are identical for every seed and never in any seed's warm/final pool."""
    from src.utils import deep_update
    c = deep_update(cfg, {"clinical": {"fixed_dev_pool": True, "dev_pool_seed": 2026}})
    df = make_fixture(6000, seed=3)
    preps = [prepare_clinical(c, s, df=df, cache=False) for s in (11, 22, 101)]
    dev = [set(p["patients"]["dev"].tolist()) for p in preps]
    assert dev[0] == dev[1] == dev[2]
    used = set()
    for p in preps:
        for pool in ("warm", "calib", "final"):
            used |= set(p["patients"][pool].tolist())
    assert not dev[0] & used
    assert set(preps[0]["patients"]["final"].tolist()) != set(preps[1]["patients"]["final"].tolist())
    for p in preps:   # pool sizes unchanged vs the default split
        n = sum(len(v) for v in p["patients"].values())
        assert abs(len(p["patients"]["dev"]) - 0.2 * n) <= 1
