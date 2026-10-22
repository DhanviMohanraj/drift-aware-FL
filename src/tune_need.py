"""Dev-only tuning of proposed_v2's single setting (EMA weight of the current-need signal).

Grid predeclared: need_ema in {0.3, 0.5, 0.8}; budgets {0.05, 0.10, 0.25}; scenarios S2, S3, S5, S6;
dev seeds 101-104 (split='dev'). Chooses the value with the lowest mean dev post-change log loss and records
whether proposed_v2 beats proposed on dev. Final seeds are never touched here.
"""
from __future__ import annotations

import multiprocessing as mp

import pandas as pd

from .regime import BUDGETS, DEV_SEEDS, SCENARIOS
from .simulation import Simulation
from .utils import ROOT, atomic_write_csv, atomic_write_json, deep_update, load_config

EMAS = [0.3, 0.5, 0.8]


def _job(a):
    cfg, sc, method, seed, b, ema = a
    c = deep_update(cfg, {"budget": {"b": b}, "policy": {"need_ema": ema}})
    sim = Simulation(c, "synthetic", sc, method, seed, split="dev")
    sim.run()
    return dict(scenario=sc, method=method, seed=seed, b=b, ema=ema,
                post_logloss=sim.results()["summary"]["post_logloss_pooled"])


if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "v2.yaml")
    jobs = [(cfg, sc, "proposed_v2", s, b, e) for e in EMAS for b in BUDGETS for sc in SCENARIOS for s in DEV_SEEDS]
    jobs += [(cfg, sc, "proposed", s, b, 0.5) for b in BUDGETS for sc in SCENARIOS for s in DEV_SEEDS]
    with mp.get_context("fork").Pool(2) as p:
        df = pd.DataFrame(p.map(_job, jobs, chunksize=4))
    v2 = df[df.method == "proposed_v2"].groupby("ema").post_logloss.mean()
    best = float(v2.idxmin())
    base = df[df.method == "proposed"].post_logloss.mean()
    per_b = df[(df.method == "proposed") | (df.ema == best)].pivot_table(index="b", columns="method", values="post_logloss")
    print("dev mean loss by ema:", v2.round(4).to_dict(), "| proposed:", round(base, 4))
    print("chosen need_ema:", best)
    print(per_b.round(4))
    atomic_write_csv(ROOT / "results" / "regime" / "tune_need_dev.csv", df)
    atomic_write_json(ROOT / "results" / "regime" / "tune_need_choice.json",
                      {"need_ema": best, "dev_mean_v2": float(v2[best]), "dev_mean_proposed": float(base)})
