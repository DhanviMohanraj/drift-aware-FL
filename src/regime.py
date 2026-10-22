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
METHODS = ["periodic", "random", "drift_only", "proposed", "proposed_v2", "known_time"]
DEV_SEEDS = [101, 102, 103, 104]


DATASET = "synthetic"
GRIDS = {   # clinical grid is smaller so it fits a Colab session (~1 h with 2 workers)
    "synthetic": dict(budgets=[0.05, 0.10, 0.25], scenarios=["S2", "S3", "S5", "S6"], dev_seeds=[101, 102, 103, 104],
                      methods=METHODS),
    "clinical": dict(budgets=[0.05, 0.10], scenarios=["C1", "C2"], dev_seeds=[101, 102],
                     methods=["periodic", "drift_only", "proposed", "proposed_v2"]),
}


def _job(args):
    cfg, sc, method, seed, b, interval, split = args
    c = deep_update(cfg, {"budget": {"b": b}, "policy": {"periodic_interval": interval}})
    if DATASET == "clinical":   # per-dataset overrides would replace the interval, so set it there too
        c = deep_update(c, {"dataset_overrides": {"clinical": {"policy": {"periodic_interval": interval}}}})
    sim = Simulation(c, DATASET, sc, method, seed, split=split)
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
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "clinical"])
    a = ap.parse_args()
    global DATASET, BUDGETS, SCENARIOS, DEV_SEEDS, METHODS
    DATASET = a.dataset
    g = GRIDS[a.dataset]
    BUDGETS, SCENARIOS, DEV_SEEDS, METHODS = g["budgets"], g["scenarios"], g["dev_seeds"], g["methods"]
    cfg = load_config(a.config)
    import json
    choice = ROOT / "results" / "regime" / "tune_need_choice.json"
    if choice.exists():   # proposed_v2's one setting, chosen on dev seeds by src/tune_need.py
        cfg = deep_update(cfg, {"policy": {"need_ema": json.loads(choice.read_text())["need_ema"]}})
    out = ROOT / "results" / ("regime" if a.dataset == "synthetic" else "regime_clinical")
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
    lines = [f"# Regime study ({a.dataset}, configs/v2.yaml)", "",
             f"Periodic interval chosen per budget on dev seeds {DEV_SEEDS}: {chosen}. Final seeds {seeds}.",
             f"Post-change pooled log loss, mean of {len(seeds)} seeds (lower is better). 'better' = seeds where the method "
             "beats periodic on the same seed.", ""]
    ao = res[res.method == "always_on"].groupby("scenario").post_logloss.mean()
    names = {"periodic": "periodic", "random": "random", "drift_only": "drift-only", "proposed": "proposed",
             "proposed_v2": "proposed v2", "known_time": "known-time"}
    budgeted = [m for m in METHODS if m != "known_time"]
    for b in BUDGETS:
        head = ["Scenario", "always-on*"] + [names[m] for m in METHODS] + \
               ["proposed better than periodic", "v2 better than periodic", "proposed MB vs periodic MB"]
        lines += [f"## Budget b = {b:g}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for sc in SCENARIOS:
            g = res[(res.b == b) & (res.scenario == sc)]
            m = g.groupby("method").post_logloss.mean()
            mb = g.groupby("method").bytes_adaptation.mean() / 1e6
            p = g[g.method == "periodic"].set_index("seed").post_logloss
            beat = lambda k: int((g[g.method == k].set_index("seed").post_logloss < p).sum())
            best = min(m.loc[budgeted])
            cell = lambda k: (f"**{m[k]:.4f}**" if k in budgeted and m[k] == best else f"{m[k]:.4f}")
            row = [sc, f"{ao[sc]:.4f}"] + [cell(k) for k in METHODS] + \
                  [f"{beat('proposed')}/{len(seeds)}", f"{beat('proposed_v2')}/{len(seeds)}", f"{mb['proposed']:.2f} vs {mb['periodic']:.2f}"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += ["*always-on is unbudgeted and shown only as a reference.", ""]
    (out / "results.md").write_text("\n".join(lines))
    atomic_write_json(out / "chosen_intervals.json", chosen)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
