"""Dev-only tuning of proposed_v3's fairness floor K (steps without training before a clinic is forced in).

Predeclared: K in {10, 20, 30}; budgets {0.05, 0.10, 0.25}; scenarios S2, S3, S5, S6; dev seeds 101-104.
Criterion: lowest mean of (pooled post-change loss + worst-clinic post-change loss) / 2, because v3 exists to
protect the worst clinic without giving up average accuracy. Final seeds are never touched here.
"""
from __future__ import annotations

import json
import multiprocessing as mp

import pandas as pd

from .regime import BUDGETS, DEV_SEEDS, SCENARIOS
from .simulation import Simulation
from .utils import ROOT, atomic_write_csv, atomic_write_json, deep_update, load_config

KS = [10, 20, 30]
CLIN = dict(budgets=[0.05, 0.10], scenarios=["C1", "C2"], dev_seeds=[101, 102])


def _job(a):
    cfg, sc, method, seed, b, k, ds = a
    c = deep_update(cfg, {"budget": {"b": b}, "policy": {"fair_k": k}})
    sim = Simulation(c, ds, sc, method, seed, split="dev")
    sim.run()
    s = sim.results()["summary"]
    return dict(scenario=sc, method=method, seed=seed, b=b, k=k, pooled=s["post_logloss_pooled"],
                worst=s["post_logloss_worst_client"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "clinical"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "v2.yaml"))
    args = ap.parse_args()
    ds = args.dataset
    budgets, scenarios, seeds = (BUDGETS, SCENARIOS, DEV_SEEDS) if ds == "synthetic" else \
        (CLIN["budgets"], CLIN["scenarios"], CLIN["dev_seeds"])
    cfg = load_config(args.config)
    ema = json.loads((ROOT / "results" / "regime" / "tune_need_choice.json").read_text())["need_ema"]
    cfg = deep_update(cfg, {"policy": {"need_ema": ema}})
    jobs = [(cfg, sc, "proposed_v3", s, b, k, ds) for k in KS for b in budgets for sc in scenarios for s in seeds]
    jobs += [(cfg, sc, "proposed_v2", s, b, 0, ds) for b in budgets for sc in scenarios for s in seeds]
    with mp.get_context("fork").Pool(2) as p:
        df = pd.DataFrame(p.map(_job, jobs, chunksize=2))
    df["score"] = (df.pooled + df.worst) / 2
    v3 = df[df.method == "proposed_v3"].groupby("k")[["pooled", "worst", "score"]].mean()
    v2 = df[df.method == "proposed_v2"][["pooled", "worst", "score"]].mean()
    best = int(v3.score.idxmin())
    helps = bool(v3.loc[best, "score"] < v2["score"])
    print(f"dev ({ds}), proposed_v3 by K:\n", v3.round(4))
    print("dev, proposed_v2:", v2.round(4).to_dict())
    print("chosen fair_k:", best, "| v3 better than v2 on dev:", helps)
    sfx = "" if ds == "synthetic" else "_clinical"
    atomic_write_csv(ROOT / "results" / "regime" / f"tune_fair_dev{sfx}.csv", df)
    atomic_write_json(ROOT / "results" / "regime" / f"tune_fair_choice{sfx}.json",
                      {"fair_k": best, "v3_better_than_v2_on_dev": helps,
                       "dev_v3": v3.loc[best].to_dict(), "dev_v2": v2.to_dict()})
