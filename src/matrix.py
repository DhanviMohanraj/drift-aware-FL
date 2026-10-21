"""Experiment matrix launcher (priority tiers) with manifests, status file and runtime estimate.

python -m src.matrix --tier smoke
python -m src.matrix --tier minimum --workers 2
python -m src.matrix --tier full --only-dataset synthetic --dry-run
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .drift import SCENARIOS
from .policies import METHODS
from .run import run_id_for, run_one
from .utils import ROOT, atomic_write_csv, atomic_write_json, deep_update, load_config

SYN = ["S0", "S1", "S2", "S3", "S4", "S5"]
CLIN = ["C0", "C1", "C2"]


def tier_specs(tier: str, cfg: dict) -> list[dict]:
    seeds = cfg["seeds"]
    b0 = cfg["budget"]["b"]
    specs = []

    def add(scs, methods, sds, b=b0):
        for sc in scs:
            for m in methods:
                for s in sds:
                    specs.append(dict(dataset=SCENARIOS[sc].dataset, scenario=sc, method=m, seed=s, b=b))

    if tier == "smoke":
        add(["S0", "S2"], METHODS, seeds[:1])
    elif tier == "minimum":
        add(["S0", "S2", "S3", "C0", "C1"], METHODS, seeds)
    elif tier == "full":
        add(SYN + CLIN, METHODS, seeds)
    elif tier == "ablation":
        add(["S3", "C1"], ["proposed_nosev", "proposed_nowait"], seeds)
    elif tier == "sweep":
        for b in (0.10, 0.50):
            add(["S3", "C1"], ["proposed", "periodic", "drift_only"], seeds, b)
    else:
        raise KeyError(tier)
    return specs


def _job(cfg_path, spec, out):
    cfg = load_config(cfg_path, {"budget": {"b": spec["b"]}})
    return run_one(cfg, spec["dataset"], spec["scenario"], spec["method"], spec["seed"], out, quiet=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/core.yaml")
    ap.add_argument("--tier", required=True, choices=["smoke", "minimum", "full", "ablation", "sweep"])
    ap.add_argument("--only-dataset", choices=["synthetic", "clinical"], default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    specs = tier_specs(a.tier, cfg)
    if a.only_dataset:
        specs = [s for s in specs if s["dataset"] == a.only_dataset]
    for s in specs:
        s["run_id"] = run_id_for(s["dataset"], s["scenario"], s["method"], s["seed"], s["b"])
    out = Path(a.out)
    tag = a.tier + (f"_{a.only_dataset}" if a.only_dataset else "")
    manifest_path = ROOT / "results" / f"manifest_{tag}.json"

    def status_of(rid):
        p = out / rid / "run.json"
        return json.loads(p.read_text()).get("status", "unknown") if p.exists() else "missing"

    done = [s for s in specs if status_of(s["run_id"]) == "complete"]
    todo = [s for s in specs if status_of(s["run_id"]) != "complete"]
    walls = [json.loads((out / s["run_id"] / "run.json").read_text()).get("wall_clock_s", np.nan) for s in done]
    med = float(np.nanmedian(walls)) if walls else np.nan
    print(f"[matrix] tier={tag}: {len(specs)} expected, {len(done)} complete, {len(todo)} to run")
    if np.isfinite(med):
        est = len(todo) * med * 1.3 / max(1, a.workers) / 60
        print(f"[matrix] median run {med:.1f}s -> estimated {est:.1f} min remaining (incl. 30% contingency)")
    if not a.dry_run and todo:
        if a.workers > 1:
            with ProcessPoolExecutor(a.workers) as ex:
                futs = {ex.submit(_job, a.config, s, str(out)): s for s in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    m = f.result()
                    print(f"[matrix] {i}/{len(todo)} {m['run_id']} {m['status']}")
        else:
            for i, s in enumerate(todo, 1):
                m = _job(a.config, s, str(out))
                print(f"[matrix] {i}/{len(todo)} {m['run_id']} {m['status']}")
    status = pd.DataFrame([dict(run_id=s["run_id"], **{k: s[k] for k in ("dataset", "scenario", "method", "seed", "b")},
                                status=status_of(s["run_id"])) for s in specs])
    atomic_write_csv(ROOT / "results" / f"status_{tag}.csv", status)
    atomic_write_json(manifest_path, {"tier": tag, "runs_dir": str(Path(a.out).resolve().relative_to(ROOT))
                                      if Path(a.out).resolve().is_relative_to(ROOT) else str(Path(a.out).resolve()),
                                      "expected": [s["run_id"] for s in specs],
                                      "complete": status[status.status == "complete"].run_id.tolist()})
    print(status.status.value_counts().to_string())


if __name__ == "__main__":
    main()
