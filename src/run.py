"""Single-run CLI with immutable run directories, checkpoint/resume and run manifests.

python -m src.run --config configs/core.yaml --dataset synthetic --scenario S3 --method proposed --seed 11
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import traceback
from pathlib import Path

import yaml

from .simulation import Simulation
from .utils import ROOT, atomic_write_bytes, atomic_write_csv, atomic_write_json, environment_info, git_commit, stable_hash, load_config


def run_id_for(dataset, scenario, method, seed, b, split="final"):
    pre = "dev-" if split == "dev" else ""
    return f"{pre}{dataset}-{scenario}-{method}-b{b:g}-s{seed}"


def run_one(cfg: dict, dataset: str, scenario: str, method: str, seed: int, out_root=None,
            split: str = "final", prepared=None, quiet=False) -> dict:
    out_root = Path(out_root or ROOT / "results" / "runs")
    b = cfg["budget"]["b"]
    rid = run_id_for(dataset, scenario, method, seed, b, split)
    d = out_root / rid
    meta_path = d / "run.json"
    cfg_hash = stable_hash(cfg)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("status") == "complete":
            if meta.get("config_hash") != cfg_hash:
                raise RuntimeError(f"{rid} is complete under a DIFFERENT config; refusing to overwrite. "
                                   f"Move it aside or use another output folder.")
            if not quiet:
                print(f"[run] {rid} already complete; skipping")
            return meta
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(run_id=rid, dataset=dataset, scenario=scenario, method=method, seed=seed, split=split,
                budget_b=b, git_commit=git_commit(), config_hash=cfg_hash, env=environment_info(),
                start_time=dt.datetime.now(dt.timezone.utc).isoformat(), status="running")
    atomic_write_bytes(d / "config.yaml", yaml.safe_dump(cfg).encode())
    atomic_write_json(meta_path, meta)
    try:
        sim = Simulation(cfg, dataset, scenario, method, seed, split=split, prepared=prepared)
        sim.run(checkpoint_path=d / "checkpoint.pkl")
        res = sim.results()
        metrics = res["metrics"].copy()
        metrics.insert(0, "run_id", rid)
        atomic_write_csv(d / "metrics.csv", metrics)
        atomic_write_csv(d / "events.csv", res["events"])
        atomic_write_csv(d / "communication.csv", res["communication"])
        atomic_write_bytes(d / "predictions.csv.gz", _gz(res["predictions"]))
        summ = dict(run_id=rid, dataset=dataset, scenario=scenario, method=method, seed=seed,
                    split=split, budget_b=b, **res["summary"])
        atomic_write_json(d / "summary.json", summ)
        atomic_write_json(d / "detail.json", res["detail"])
        if dataset == "clinical":
            meta["dataset_checksum"] = sim.world.p["checksums"]["dataset"]
            meta["split_checksum"] = sim.world.p["checksums"]["split"]
            meta["preprocessing_checksum"] = sim.world.p["checksums"]["preprocessing"]
        else:
            meta["dataset_checksum"] = stable_hash(sim.world.info["weights"])
            meta["split_checksum"] = "synthetic-deterministic"
            meta["preprocessing_checksum"] = "none"
        meta.update(status="complete", end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                    warm_checksum=sim.warm_checksum, wall_clock_s=sim.wall)
        (d / "checkpoint.pkl").unlink(missing_ok=True)
        if not quiet:
            print(f"[run] {rid} complete  post_logloss={summ['post_logloss_pooled']:.4f} "
                  f"adapt_bytes={summ['bytes_adaptation']:,}  {sim.wall:.1f}s")
    except Exception as e:
        meta.update(status="failed", error=repr(e), traceback=traceback.format_exc(),
                    end_time=dt.datetime.now(dt.timezone.utc).isoformat())
        print(f"[run] {rid} FAILED: {e!r}")
    atomic_write_json(meta_path, meta)
    return meta


def _gz(df) -> bytes:
    import gzip, io
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as f:
        f.write(df.to_csv(index=False).encode())
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/core.yaml")
    ap.add_argument("--dataset", required=True, choices=["synthetic", "clinical"])
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--b", type=float, default=None, help="override budget fraction")
    ap.add_argument("--split", default="final", choices=["final", "dev"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, {"budget": {"b": a.b}} if a.b is not None else None)
    run_one(cfg, a.dataset, a.scenario, a.method, a.seed, a.out, a.split)


if __name__ == "__main__":
    main()
