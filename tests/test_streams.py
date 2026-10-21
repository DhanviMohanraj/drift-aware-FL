"""Gates: deterministic sampling, paired pre-drift draws, events alter only intended clients/times."""
import numpy as np

from src.drift import get_scenario
from src.streams import SyntheticWorld


def world(cfg, sid, seed=11):
    return SyntheticWorld(cfg, seed, get_scenario(sid))


def test_deterministic(cfg):
    a, b = world(cfg, "S0").sample(2, 7), world(cfg, "S0").sample(2, 7)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.y, b.y)


def test_splits_use_separate_draws(cfg):
    w = world(cfg, "S0")
    assert not np.allclose(w.sample(0, 0, "final").x, w.sample(0, 0, "dev").x)
    assert not np.allclose(w.sample(0, 0, "final").x, w.sample(0, 0, "warm").x)


def test_concept_shift_pairing(cfg):
    w0, w2 = world(cfg, "S0"), world(cfg, "S2")
    for c in range(4):
        for t in (0, 39, 40, 60):
            a, b = w0.sample(c, t), w2.sample(c, t)
            np.testing.assert_array_equal(a.x, b.x)                  # features never change
            if c in (2, 3) or t < 40:
                np.testing.assert_array_equal(a.y, b.y)
    w = w2.weights_at(0, 45)
    ang = np.degrees(np.arctan2(w[1], w[0]) - np.arctan2(w2.w[1], w2.w[0])) % 360
    assert abs(ang - 60) < 1e-6
    np.testing.assert_allclose(w[2:], w2.w[2:])


def test_covariate_shift(cfg):
    w0, w1 = world(cfg, "S0"), world(cfg, "S1")
    a, b = w0.sample(1, 50), w1.sample(1, 50)
    d = b.x - a.x
    np.testing.assert_allclose(d[:, :4], 1.0, rtol=1e-5)
    assert np.abs(d[:, 4:]).max() == 0
    np.testing.assert_array_equal(w0.sample(1, 39).x, w1.sample(1, 39).x)


def test_staggered_and_gradual(cfg):
    w3 = world(cfg, "S3")
    assert w3.weights_at(0, 30)[0] != w3.w[0] and np.allclose(w3.weights_at(1, 49), w3.w)
    assert np.allclose(w3.weights_at(3, 99), w3.w)                 # clinic 4 never affected
    w4 = world(cfg, "S4")
    from src.drift import rotation_angle
    sc = get_scenario("S4")
    assert rotation_angle(sc, 0, 34) == 0 and abs(rotation_angle(sc, 0, 45) - 30) < 1e-9
    assert rotation_angle(sc, 0, 60) == 60


def test_prior_shift_prevalence(cfg):
    w = world(cfg, "S5")
    pre = np.mean([w.sample(c, t).y.mean() for c in range(4) for t in range(30, 40)])
    post = np.mean([w.sample(c, t).y.mean() for c in range(4) for t in range(40, 50)])
    assert abs(pre - 0.3) < 0.02 and abs(post - 0.6) < 0.02
