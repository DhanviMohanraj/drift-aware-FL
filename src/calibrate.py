"""Development-stream calibration (never touches final streams).

1. CUSUM threshold from {3, 6, 12}: target <= 1 false alarm per 100 client-steps on no-drift dev
   streams; among those, maximise detection rate then minimise delay on drift dev streams.
   The detector depends only on the frozen sentinel and the data, so it is method-independent.
2. Periodic interval from {2, 5, 10}: lowest dev post-change log loss under the budget.

python -m src.calibrate --dataset synthetic
"""
from __future__ import annotations

import argparse

import numpy as np

from .detector import CusumDetector
from .simulation import Simulation
from .utils import ROOT, atomic_write_json, deep_update, load_config

PLAN = {"synthetic": {"null": ["S0"], "drift": ["S2", "S3"]},
        "clinical": {"null": ["C0"], "drift": ["C1", "C2"]}}


def sentinel_losses(cfg, dataset, scenario, seed):
    sim = Simulation(cfg, dataset, scenario, "drift_only", seed, split="dev")
    L = np.zeros((sim.K, sim.H))
    from .model import mean_bce
    for t in range(sim.H):
        for c in range(sim.K):
            b = sim.stream.next(c, t)
            L[c, t] = mean_bce(sim.sentinel, b.x, b.y)
    return sim, L


def detector_eval(cfg, sim, L, thr):
    dcfg = deep_update(cfg["detector"], {"threshold": thr})
    flags = np.zeros_like(L, dtype=bool)
    for c in range(sim.K):
        m, s, _ = sim.calib_ref[c]
        d = CusumDetector(m, s, dcfg)
        for t in range(sim.H):
            flags[c, t] = d.update(L[c, t]).flag
    mw = cfg["evaluation"]["match_window"]
    inwin = np.zeros_like(flags)
    delays, miss = [], 0
    for e in sim.manifest:
        inwin[e["client"], e["onset"]:min(sim.H, e["end"] + mw + 1)] = True
        hits = np.flatnonzero(flags[e["client"], e["onset"]:e["end"] + mw + 1])
        if len(hits):
            delays.append(int(hits[0]))
        else:
            miss += 1
    elig = (~inwin).sum()
    return dict(false_alarms=int((flags & ~inwin).sum()), eligible=int(elig),
                fa_per_100=float(100 * (flags & ~inwin).sum() / elig), detected=len(delays),
                events=len(sim.manifest), delays=delays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/core.yaml")
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "clinical"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg = deep_update(cfg, cfg.get("dataset_overrides", {}).get(a.dataset, {}))
    out = {"dataset": a.dataset, "dev_seeds": cfg["dev_seeds"], "thresholds": {}, "periodic": {}}
    traces = {}
    for kind in ("null", "drift"):
        for sc in PLAN[a.dataset][kind]:
            for s in cfg["dev_seeds"]:
                traces[(kind, sc, s)] = sentinel_losses(cfg, a.dataset, sc, s)
    for thr in (3.0, 6.0, 12.0):
        res = {"null": [], "drift": []}
        for (kind, sc, s), (sim, L) in traces.items():
            res[kind].append(dict(scenario=sc, seed=s, **detector_eval(cfg, sim, L, thr)))
        fa_null = float(np.mean([r["fa_per_100"] for r in res["null"]]))
        fa_drift = float(np.mean([r["fa_per_100"] for r in res["drift"]]))
        det = sum(r["detected"] for r in res["drift"]) / max(1, sum(r["events"] for r in res["drift"]))
        dl = [d for r in res["drift"] for d in r["delays"]]
        out["thresholds"][str(thr)] = dict(fa_per_100_null=fa_null, fa_per_100_drift_streams=fa_drift,
                                           detection_rate=det, mean_delay=float(np.mean(dl)) if dl else None,
                                           runs=res)
        print(f"threshold {thr:5.1f}: FA/100 null={fa_null:.2f} (drift-stream outside windows {fa_drift:.2f}) "
              f"detect={det:.2f} delay={np.mean(dl) if dl else float('nan'):.1f}")
    ok = [t for t, v in out["thresholds"].items() if v["fa_per_100_null"] <= 1.0]
    if ok:
        best = max(ok, key=lambda t: (out["thresholds"][t]["detection_rate"],
                                      -(out["thresholds"][t]["mean_delay"] or 1e9)))
        out["threshold_target_met"] = True
    else:
        best = "12.0"
        out["threshold_target_met"] = False
    out["selected_threshold"] = float(best)
    print(f"selected threshold = {best} (target met: {out['threshold_target_met']})")
    cfg_t = deep_update(cfg, {"detector": {"threshold": float(best)}})
    for k in (2, 5, 10):
        c2 = deep_update(cfg_t, {"policy": {"periodic_interval": k}})
        vals = {"null": [], "drift": []}
        for kind in ("null", "drift"):
            for sc in PLAN[a.dataset][kind]:
                for s in cfg["dev_seeds"]:
                    sim = Simulation(c2, a.dataset, sc, "periodic", s, split="dev")
                    sim.run()
                    vals[kind].append(sim.results()["summary"]["post_logloss_pooled"])
        out["periodic"][str(k)] = {kk: float(np.mean(v)) for kk, v in vals.items()}
        print(f"periodic interval {k:2d}: dev post logloss null={np.mean(vals['null']):.4f} "
              f"drift={np.mean(vals['drift']):.4f}")
    bestk = min(out["periodic"], key=lambda k: out["periodic"][k]["drift"] + out["periodic"][k]["null"])
    out["selected_periodic_interval"] = int(bestk)
    print(f"selected periodic interval = {bestk}")
    atomic_write_json(ROOT / "results" / f"calibration_{a.dataset}.json", out)


if __name__ == "__main__":
    main()
