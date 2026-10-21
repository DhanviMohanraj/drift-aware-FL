"""Gates: aggregation, prequential order, deployment only via charged broadcast."""
import numpy as np
import torch

from src.federation import aggregate
from src.simulation import Simulation


def test_weighted_mean_exact():
    s1 = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.0])}
    s2 = {"w": torch.tensor([4.0, 8.0]), "b": torch.tensor([3.0])}
    out = aggregate([s1, s2], [1, 2])          # weights 1/3, 2/3
    torch.testing.assert_close(out["w"], torch.tensor([3.0, 6.0]))
    torch.testing.assert_close(out["b"], torch.tensor([2.0]))


def test_zero_admitted_leaves_state(cfg):
    sim = Simulation(cfg, "synthetic", "S0", "proposed", 11)
    before = {k: v.clone() for k, v in sim.server_state.items()}
    sim.step()                                  # bucket starts empty -> nothing affordable at t=0
    assert sim.n_transactions == 0 and sim.deploy.versions == [0, 0, 0, 0]
    for k in before:
        assert torch.equal(before[k], sim.server_state[k])


def test_prequential_order(cfg):
    sim = Simulation(cfg, "synthetic", "S2", "always_on", 11, trace=True)
    for _ in range(5):
        sim.step()
    order = {"predict": 0, "label": 1, "buffer": 2, "train": 3}
    for t in range(5):
        for c in range(4):
            phases = [i for i, (tt, ph, cc) in enumerate(sim.trace) if tt == t and cc == c]
            seq = [sim.trace[i][1] for i in phases]
            assert seq == sorted(seq, key=order.get), (t, c, seq)
            assert seq[0] == "predict" and "train" in seq
    # every client's predictions at t precede ANY training at t
    for t in range(5):
        last_pred = max(i for i, (tt, ph, _) in enumerate(sim.trace) if tt == t and ph == "predict")
        first_train = min(i for i, (tt, ph, _) in enumerate(sim.trace) if tt == t and ph == "train")
        assert last_pred < first_train


def test_deployment_requires_broadcast(cfg):
    sim = Simulation(cfg, "synthetic", "S2", "random", 11)
    sim.run()
    ev = sim.evaluator.metrics_frame(10)
    comm = sim.ledger.frame()
    bc = comm[comm.message_kind == "model_broadcast"]
    for c in range(4):
        v = ev[ev.client_id == c].sort_values("time").deployed_version.to_numpy()
        changes = np.flatnonzero(np.diff(v)) + 1           # step whose prediction used a new version
        times = ev[ev.client_id == c].sort_values("time").time.to_numpy()
        for i in changes:
            t_used = times[i]
            # a charged broadcast to this client must exist at some t < t_used, after the previous version
            assert ((bc.receiver == str(c)) & (bc.time < t_used) & (bc.time >= times[i - 1])).any()
    assert len(bc) == 4 * sim.n_transactions
