"""Regime study: where does drift-triggered retraining beat a fixed schedule?

Predeclared grid (written before running): budgets b in {0.05, 0.10, 0.25}; synthetic scenarios
S2 (concept, 2 clinics), S3 (staggered), S5 (prior shift), S6 (one clinic, sudden).
Fairness: the periodic interval is re-chosen PER BUDGET on development seeds (101-104) from {2, 5, 10}
by mean dev post-change log loss; final seeds 11-55 are never used to choose anything.

python -m src.regime --config configs/v2.yaml --workers 2
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

from .simulation import Simulation
from .utils import ROOT, atomic_write_csv, atomic_write_json, deep_update, load_config

BUDGETS = [0.05, 0.10, 0.25]
SCENARIOS = ["S2", "S3", "S5", "S6"]
INTERVALS = [2, 5, 10]
METHODS = ["periodic", "random", "drift_only", "proposed", "known_time"]
DEV_SEEDS = [101, 102, 103, 104]


def _job(args):
    cfg, sc, method, seed, b, interval, split = args
    c = deep_update(cfg, {"budget": {"b": b}, "policy": {"periodic_interval": interval}})
    sim = Simulation(c, "synthetic", sc, method, seed, split=split)
    sim.run()
    s = sim.results()["summary"]
    return dict(scenario=sc, method=method, seed=seed, b=b, interval=interval, split=split,
                post_logloss=s["post_logloss_pooled"], worst_client=s["post_logloss_worst_client"],
                auroc=s["post_auroc_pooled"], bytes_adaptation=s["bytes_adaptation"],
                trainings=s["n_client_trainings"])


def _map(workers, fn, jobs):
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as p:
            return p.map(fn, jobs, chunksize=4)
    return list(map(fn, jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = ROOT / "results" / "regime"
    # 1) dev-only choice of the periodic interval per budget
    dev = _map(a.workers, _job, [(cfg, sc, "periodic", s, b, k, "dev")
                                 for b in BUDGETS for k in INTERVALS for sc in SCENARIOS for s in DEV_SEEDS])
    dev = pd.DataFrame(dev)
    chosen = {float(b): int(g.groupby("interval").post_logloss.mean().idxmin()) for b, g in dev.groupby("b")}
    print("[regime] periodic interval chosen on dev seeds:", chosen)
    # 2) final runs
    seeds = cfg["seeds"]
    jobs = [(cfg, sc, m, s, b, chosen[b], "final") for b in BUDGETS for sc in SCENARIOS for m in METHODS for s in seeds]
    jobs += [(cfg, sc, "always_on", s, BUDGETS[0], chosen[BUDGETS[0]], "final") for sc in SCENARIOS for s in seeds]
    res = pd.DataFrame(_map(a.workers, _job, jobs))
    atomic_write_csv(out / "dev_periodic.csv", dev)
    atomic_write_csv(out / "summary.csv", res)
    # 3) tables
    lines = ["# Regime study (synthetic, configs/v2.yaml)", "",
             f"Periodic interval chosen per budget on dev seeds {DEV_SEEDS}: {chosen}. Final seeds {seeds}.",
             "Post-change pooled log loss, mean of 5 seeds (lower is better). 'better' = seeds where the method "
             "beats periodic on the same seed.", ""]
    ao = res[res.method == "always_on"].groupby("scenario").post_logloss.mean()
    for b in BUDGETS:
        lines += [f"## Budget b = {b:g}", "",
                  "| Scenario | always-on* | periodic | random | drift-only | proposed | known-time | proposed better than periodic | proposed MB vs periodic MB |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for sc in SCENARIOS:
            g = res[(res.b == b) & (res.scenario == sc)]
            m = g.groupby("method").post_logloss.mean()
            mb = g.groupby("method").bytes_adaptation.mean() / 1e6
            p = g[g.method == "periodic"].set_index("seed").post_logloss
            q = g[g.method == "proposed"].set_index("seed").post_logloss
            better = int((q < p).sum())
            best = min(m.loc[["periodic", "random", "drift_only", "proposed"]])
            fmt = lambda k: (f"**{m[k]:.4f}**" if m[k] == best else f"{m[k]:.4f}")
            lines.append(f"| {sc} | {ao[sc]:.4f} | {fmt('periodic')} | {fmt('random')} | {fmt('drift_only')} | "
                         f"{fmt('proposed')} | {m['known_time']:.4f} | {better}/5 | {mb['proposed']:.2f} vs {mb['periodic']:.2f} |")
        lines.append("")
    lines += ["*always-on is unbudgeted (16.6 MB) and shown only as a reference.", ""]
    (out / "results.md").write_text("\n".join(lines))
    atomic_write_json(out / "chosen_intervals.json", chosen)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
