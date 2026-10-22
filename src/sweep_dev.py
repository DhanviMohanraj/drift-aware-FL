"""Development-only protocol sweep for v2 (never touches final streams).

Stage A (clinical only): clinic split (Dirichlet alpha) x drift option x CUSUM threshold.
    Measures clinic balance, whether the injected drift raises the frozen-sentinel loss, detection rate,
    delay and no-drift false alarms on DEV patients. Picks a setting by a predeclared rule.
Stage B (clinical and synthetic): service credits x budget fraction b.
    Measures whether the budget is binding and whether the proposed priority changes any decision
    relative to drift-only. Picks by that DESIGN rule - never by which method has lower loss
    (dev losses are reported for information only).

    python -m src.sweep_dev --dataset clinical --workers 2
    python -m src.sweep_dev --dataset synthetic --workers 2
    python -m src.sweep_dev --write-candidate        # merges both into configs/v2_candidate.yaml

Nothing here edits configs/core.yaml. The candidate must be reviewed (supervisor) before freezing.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .calibrate import detector_eval
from .model import mean_bce
from .simulation import Simulation
from .utils import ROOT, atomic_write_json, deep_update, load_config

ALPHAS = [0.5, 1.0, 5.0]
DRIFTS = {   # name -> (features, magnitude in warm-start SD); order = distance from the original plan
    "v1_lab_1sd": (["num_lab_procedures"], 1.0),
    "lab_med_3sd": (["num_lab_procedures", "num_medications", "time_in_hospital"], 3.0),
    "utilisation_3sd": (["number_inpatient", "number_emergency", "number_diagnoses"], 3.0),
}
THRESHOLDS = [3.0, 6.0, 12.0]
CREDITS = [3, 6, 10]
BUDGETS = [0.25, 0.10]          # order = distance from the original plan
FA_TARGET = 1.0                 # false alarms per 100 client-steps on no-drift dev streams
SKEW_LIMIT = 3.0                # clinic prevalence must stay within [pooled/3, 3 x pooled] (symmetric)
DIFF_TARGET = 0.02              # proposed must change >= 2% of client-step decisions vs drift-only
UTIL_TARGET = 0.5               # proposed must spend >= 50% of its byte allowance (budget can bind)

_PREP: dict = {}                # (alpha, seed) -> prepared clinical arrays; shared with forked workers


# ----------------------------------------------------------------------------- helpers
def base_cfg(config_path, fixed_dev_pool=True):
    cfg = load_config(config_path)
    if fixed_dev_pool:   # dev patients identical across seeds and disjoint from every seed's final stream
        cfg = deep_update(cfg, {"clinical": {"fixed_dev_pool": True,
                                             "dev_pool_seed": cfg["clinical"].get("dev_pool_seed", 2026)}})
    return cfg


def clinical_cfg(cfg, alpha, drift=None):
    upd = {"clinical": {"dirichlet_alpha": alpha}}
    if drift is not None:
        feats, mag = DRIFTS[drift]
        upd["clinical"].update(shift_features=feats, shift_magnitude=mag)
    return deep_update(cfg, upd)


def get_prepared(cfg, alpha, seed):
    key = (alpha, seed)
    if key not in _PREP:
        from .preprocess import prepare_clinical
        _PREP[key] = prepare_clinical(clinical_cfg(cfg, alpha), seed)
    return _PREP[key]


def _sentinel_trace(sim):
    L = np.zeros((sim.K, sim.H))
    for t in range(sim.H):
        for c in range(sim.K):
            b = sim.stream.next(c, t)
            L[c, t] = mean_bce(sim.sentinel, b.x, b.y)
    return L


# ----------------------------------------------------------------------------- stage A
def _stage_a_job(args):
    cfg, alpha, drift, seed = args
    c = clinical_cfg(cfg, alpha, drift)
    p = get_prepared(cfg, alpha, seed)
    out = {"alpha": alpha, "drift": drift, "seed": seed}
    dev = p["counts"]["dev"]
    prev = [x for x in dev["per_client_prevalence"] if x is not None]
    out["dev_prevalence"] = dev["prevalence"]
    out["dev_client_prevalence"] = prev
    pooled = max(dev["prevalence"], 1e-9)
    # symmetric skew: a clinic dominated by positives (max/pooled) or almost without them (pooled/min)
    out["dev_max_skew"] = float(max(max(prev) / pooled, pooled / max(min(prev), 1e-9)))
    out["dev_client_sizes"] = dev["per_client"]
    traces = {}
    for sc in ("C0", "C1", "C2"):
        sim = Simulation(c, "clinical", sc, "drift_only", seed, split="dev", prepared=p)
        traces[sc] = (sim, _sentinel_trace(sim))
    sim0 = traces["C0"][0]
    out["warm_loss_per_clinic"] = [float(x) for x in sim0.warm_info["warm_losses"]]
    # effect size: post-onset minus pre-onset sentinel loss, in calibration-SD units
    eff = []
    for sc in ("C1", "C2"):
        sim, L = traces[sc]
        for e in sim.manifest:
            cl, o = e["client"], e["onset"]
            sd = max(sim.calib_ref[cl][1], c["detector"]["sd_floor"])
            eff.append(float((L[cl, o:o + 10].mean() - L[cl, max(0, o - 10):o].mean()) / sd))
    out["effect_z"] = eff
    out["thresholds"] = {}
    for thr in THRESHOLDS:
        r0 = detector_eval(c, *traces["C0"], thr)
        r = [detector_eval(c, *traces[sc], thr) for sc in ("C1", "C2")]
        out["thresholds"][str(thr)] = {
            "fa_per_100_null": r0["fa_per_100"],
            "detected": sum(x["detected"] for x in r), "events": sum(x["events"] for x in r),
            "delays": [d for x in r for d in x["delays"]]}
    return out


def summarise_a(rows):
    df = []
    for (alpha, drift), g in itertools.groupby(sorted(rows, key=lambda r: (r["alpha"], r["drift"])),
                                               key=lambda r: (r["alpha"], r["drift"])):
        g = list(g)
        best = None
        for thr in THRESHOLDS:
            fa = float(np.mean([r["thresholds"][str(thr)]["fa_per_100_null"] for r in g]))
            det = sum(r["thresholds"][str(thr)]["detected"] for r in g) / max(1, sum(r["thresholds"][str(thr)]["events"] for r in g))
            dl = [d for r in g for d in r["thresholds"][str(thr)]["delays"]]
            cand = dict(threshold=thr, fa_per_100_null=fa, detection_rate=det,
                        mean_delay=float(np.mean(dl)) if dl else np.nan, fa_ok=fa <= FA_TARGET)
            if cand["fa_ok"] and (best is None or (det, -np.nan_to_num(cand["mean_delay"], nan=99))
                                  > (best["detection_rate"], -np.nan_to_num(best["mean_delay"], nan=99))):
                best = cand
        if best is None:
            best = dict(threshold=max(THRESHOLDS), fa_per_100_null=np.nan, detection_rate=0.0,
                        mean_delay=np.nan, fa_ok=False)
        skew = float(np.max([r["dev_max_skew"] for r in g]))     # worst seed, not the average
        df.append(dict(alpha=alpha, drift=drift, **best,
                       mean_effect_z=float(np.mean([z for r in g for z in r["effect_z"]])),
                       max_clinic_skew=skew, balanced=skew <= SKEW_LIMIT,
                       worst_warm_loss=float(np.mean([max(r["warm_loss_per_clinic"]) for r in g])),
                       plan_distance=ALPHAS.index(alpha) + list(DRIFTS).index(drift)))
    df = pd.DataFrame(df)
    # detection_rate rounded so tiny differences do not beat a smaller protocol change
    ranked = df.assign(det_r=df.detection_rate.round(1)).sort_values(
        ["fa_ok", "balanced", "det_r", "plan_distance", "mean_delay"],
        ascending=[False, False, False, True, True]).drop(columns="det_r")
    return ranked


# ----------------------------------------------------------------------------- stage B
def _stage_b_job(args):
    cfg, dataset, sc, method, seed, credits, b, alpha, drift, thr = args
    c = deep_update(cfg, {"detector": {"credits": credits}, "budget": {"b": b}})
    prepared = None
    if dataset == "clinical":
        c = clinical_cfg(c, alpha, drift)
        c = deep_update(c, {"dataset_overrides": {"clinical": {"detector": {"threshold": thr, "credits": credits}}}})
        prepared = get_prepared(cfg, alpha, seed)
    sim = Simulation(c, dataset, sc, method, seed, split="dev", prepared=prepared)
    sim.run()
    res = sim.results()
    s = res["summary"]
    ev = res["events"][["client_id", "time", "selected"]]
    return dict(scenario=sc, method=method, seed=seed, credits=credits, b=b,
                post_logloss=s["post_logloss_pooled"], worst_client=s["post_logloss_worst_client"],
                utilisation=(s["budget_spent"] / s["budget_cap"]) if s["budget_cap"] else np.nan,
                trainings=s["n_client_trainings"], expired=s["expired_requests"],
                alarm_to_service=s["mean_alarm_to_service"], selected=ev.selected.to_numpy().tolist())


def summarise_b(rows):
    df = pd.DataFrame(rows)
    out = []
    for (credits, b), g in df.groupby(["credits", "b"]):
        diffs = []
        for (sc, seed), h in g.groupby(["scenario", "seed"]):
            p = h[h.method == "proposed"].selected
            d = h[h.method == "drift_only"].selected
            if len(p) and len(d):
                diffs.append(float(np.mean(np.array(p.iloc[0]) != np.array(d.iloc[0]))))
        pr = g[g.method == "proposed"]
        row = dict(credits=credits, b=b,
                   decision_difference=float(np.mean(diffs)) if diffs else np.nan,
                   proposed_utilisation=float(pr.utilisation.mean()),
                   proposed_trainings=float(pr.trainings.mean()),
                   proposed_expired=float(pr.expired.mean()),
                   plan_distance=CREDITS.index(credits) + BUDGETS.index(b))
        for m in ("proposed", "drift_only", "periodic"):
            row[f"dev_logloss_{m}"] = float(g[g.method == m].post_logloss.mean())   # information only
        out.append(row)
    t = pd.DataFrame(out)
    t["priority_matters"] = t.decision_difference >= DIFF_TARGET
    t["budget_binds"] = t.proposed_utilisation >= UTIL_TARGET
    return t.sort_values(["priority_matters", "budget_binds", "plan_distance", "decision_difference"],
                         ascending=[False, False, True, False])


# ----------------------------------------------------------------------------- main
def _pool(workers):
    return mp.get_context("fork").Pool(workers) if workers > 1 else None


def _map(pool, fn, jobs):
    t0 = time.time()
    it = pool.imap_unordered(fn, jobs) if pool else map(fn, jobs)
    out = []
    for i, r in enumerate(it, 1):
        out.append(r)
        if i % max(1, len(jobs) // 10) == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
    return out


def run(dataset, config, workers, stage_a_choice=None, dev_seeds=None, tag="", stage_a_only_fixed=False):
    cfg = base_cfg(config)
    if dev_seeds:
        cfg["dev_seeds"] = list(dev_seeds)
    report = {"dataset": dataset, "config": str(config), "dev_seeds": cfg["dev_seeds"],
              "rules": dict(FA_TARGET=FA_TARGET, SKEW_LIMIT=SKEW_LIMIT, DIFF_TARGET=DIFF_TARGET,
                            UTIL_TARGET=UTIL_TARGET)}
    alpha = drift = thr = None
    if dataset == "clinical" and stage_a_only_fixed and stage_a_choice:
        alpha, drift, thr = stage_a_choice
        for s_ in cfg["dev_seeds"]:
            get_prepared(cfg, alpha, s_)
        print(f"[sweep] stage A skipped; stage B uses alpha={alpha}, drift={drift}, threshold={thr}")
        report["stage_a"] = {"skipped": True, "choice": {"alpha": alpha, "drift": drift,
                             "features": DRIFTS[drift][0], "magnitude": DRIFTS[drift][1], "threshold": thr}}
        scenarios = ["C1", "C2"]
    elif dataset == "clinical":
        # prepare once per (alpha, seed) in the parent so forked workers share it
        for a_, s_ in itertools.product(ALPHAS, cfg["dev_seeds"]):
            get_prepared(cfg, a_, s_)
        jobs = [(cfg, a_, d_, s_) for a_ in ALPHAS for d_ in DRIFTS for s_ in cfg["dev_seeds"]]
        print(f"[sweep] stage A: {len(jobs)} jobs (alpha x drift x dev seed), each = C0+C1+C2 dev streams")
        pool = _pool(workers)
        rows = _map(pool, _stage_a_job, jobs)
        if pool:
            pool.close()
        ta = summarise_a(rows)
        print("\n[sweep] STAGE A (dev patients only; best threshold per row):")
        print(ta.round(3).to_string(index=False))
        top = ta.iloc[0]
        if not (top.fa_ok and top.detection_rate > 0):
            print("\n[sweep] WARNING: no setting detects injected drift within the false-alarm target.")
        alpha, drift, thr = float(top.alpha), str(top.drift), float(top.threshold)
        if stage_a_choice:
            alpha, drift, thr = stage_a_choice
        print(f"\n[sweep] stage A choice: alpha={alpha}, drift={drift} {DRIFTS[drift]}, threshold={thr}")
        report["stage_a"] = {"table": ta.to_dict("records"), "raw": rows,
                             "choice": {"alpha": alpha, "drift": drift, "features": DRIFTS[drift][0],
                                        "magnitude": DRIFTS[drift][1], "threshold": thr}}
        scenarios = ["C1", "C2"]
    else:
        scenarios = ["S2", "S3"]
    jobs = [(cfg, dataset, sc, m, s, cr, b, alpha, drift, thr)
            for cr in CREDITS for b in BUDGETS for sc in scenarios
            for m in ("proposed", "drift_only", "periodic") for s in cfg["dev_seeds"]]
    print(f"\n[sweep] stage B: {len(jobs)} dev runs (credits x b x scenario x method x seed)")
    pool = _pool(workers)
    rows = _map(pool, _stage_b_job, jobs)
    if pool:
        pool.close()
    tb = summarise_b(rows)
    print("\n[sweep] STAGE B (dev only; chosen by design rule, dev log loss shown for information):")
    print(tb.round(4).to_string(index=False))
    top = tb.iloc[0]
    if not (top.priority_matters and top.budget_binds):
        print("\n[sweep] WARNING: no tested setting makes the budget bind AND the priority change decisions. "
              "Report this honestly; consider tighter budgets or more simultaneous drift.")
    report["stage_b"] = {"table": tb.to_dict("records"),
                         "choice": {"credits": int(top.credits), "b": float(top.b),
                                    "priority_matters": bool(top.priority_matters),
                                    "budget_binds": bool(top.budget_binds)}}
    print(f"\n[sweep] stage B choice: credits={int(top.credits)}, b={top.b}")
    path = ROOT / "results" / f"sweep_dev_{dataset}{'_' + tag if tag else ''}.json"
    atomic_write_json(path, report)
    print(f"[sweep] wrote {path.relative_to(ROOT)}")


def merge(paths, tag):
    """Combine sweeps run on different dev seeds. Stage A: pool raw per-seed rows and re-rank.
    Stage B: average the per-setting tables (each file has the same scenarios x seeds count),
    valid only if every file ran stage B with the same stage-A setting."""
    reps = [json.loads(Path(p).read_text()) for p in paths]
    ds = {r["dataset"] for r in reps}
    assert len(ds) == 1, "merge files from one dataset only"
    dataset = ds.pop()
    seeds = sorted({s for r in reps for s in r["dev_seeds"]})
    out = {"dataset": dataset, "merged_from": [str(p) for p in paths], "dev_seeds": seeds,
           "rules": reps[0]["rules"]}
    print(f"[merge] {dataset}: seeds {seeds}")
    if dataset == "clinical":
        rows = [x for r in reps for x in r["stage_a"]["raw"]]
        for x in rows:   # recompute balance with the current (symmetric) rule
            pooled = max(x["dev_prevalence"], 1e-9)
            prev = x["dev_client_prevalence"]
            x["dev_max_skew"] = float(max(max(prev) / pooled, pooled / max(min(prev), 1e-9)))
        ta = summarise_a(rows)
        print("\n[merge] STAGE A (all seeds pooled):")
        print(ta.round(3).to_string(index=False))
        top = ta.iloc[0]
        choice = {"alpha": float(top.alpha), "drift": str(top.drift), "features": DRIFTS[str(top.drift)][0],
                  "magnitude": DRIFTS[str(top.drift)][1], "threshold": float(top.threshold)}
        used = {json.dumps({k: r["stage_a"]["choice"][k] for k in ("alpha", "drift", "threshold")}, sort_keys=True)
                for r in reps}
        same = used == {json.dumps({k: choice[k] for k in ("alpha", "drift", "threshold")}, sort_keys=True)}
        print(f"\n[merge] stage A choice: {choice}")
        if not same:
            print("[merge] WARNING: the pooled stage-A choice differs from the setting stage B was run with "
                  f"({used}). Stage B below does NOT apply to the new choice - rerun stage B before freezing.")
        out["stage_a"] = {"table": ta.to_dict("records"), "raw": rows, "choice": choice,
                          "stage_b_valid_for_choice": same}
    tabs = [pd.DataFrame(r["stage_b"]["table"]) for r in reps]
    cols = ["decision_difference", "proposed_utilisation", "proposed_trainings", "proposed_expired",
            "dev_logloss_proposed", "dev_logloss_drift_only", "dev_logloss_periodic"]
    tb = pd.concat(tabs).groupby(["credits", "b"], as_index=False)[cols].mean()
    tb["plan_distance"] = [CREDITS.index(int(c)) + BUDGETS.index(float(b)) for c, b in zip(tb.credits, tb.b)]
    tb["priority_matters"] = tb.decision_difference >= DIFF_TARGET
    tb["budget_binds"] = tb.proposed_utilisation >= UTIL_TARGET
    tb = tb.sort_values(["priority_matters", "budget_binds", "plan_distance", "decision_difference"],
                        ascending=[False, False, True, False])
    print("\n[merge] STAGE B (averaged over files; dev log loss for information only):")
    print(tb.round(4).to_string(index=False))
    top = tb.iloc[0]
    out["stage_b"] = {"table": tb.to_dict("records"),
                      "choice": {"credits": int(top.credits), "b": float(top.b),
                                 "priority_matters": bool(top.priority_matters),
                                 "budget_binds": bool(top.budget_binds)}}
    print(f"\n[merge] stage B choice: credits={int(top.credits)}, b={top.b}")
    path = ROOT / "results" / f"sweep_dev_{dataset}_{tag}.json"
    atomic_write_json(path, out)
    print(f"[merge] wrote {path.relative_to(ROOT)}")


def write_candidate(config, tag=""):
    cfg = base_cfg(config)
    sfx = f"_{tag}" if tag else ""
    rc = ROOT / "results" / f"sweep_dev_clinical{sfx}.json"
    rs = ROOT / "results" / f"sweep_dev_synthetic{sfx}.json"
    for alt in ("sweep_dev_synthetic_4seeds.json", "sweep_dev_synthetic.json"):   # most seeds first
        if not rs.exists():
            rs = ROOT / "results" / alt
    if not rc.exists():
        raise SystemExit("run `python -m src.sweep_dev --dataset clinical` first")
    c = json.loads(rc.read_text())
    a, bc = c["stage_a"]["choice"], c["stage_b"]["choice"]
    cand = deep_update(cfg, {"clinical": {"dirichlet_alpha": a["alpha"], "shift_features": a["features"],
                                          "shift_magnitude": a["magnitude"]}})
    ov = cand.setdefault("dataset_overrides", {})
    ov["clinical"] = deep_update(ov.get("clinical") or {}, {"detector": {"threshold": a["threshold"],
                                                                         "credits": bc["credits"]}})
    b = bc["b"]
    notes = [f"clinical: alpha {a['alpha']}, drift {a['drift']} ({a['features']} +{a['magnitude']} SD), "
             f"threshold {a['threshold']}, credits {bc['credits']}, b {bc['b']}"]
    if rs.exists():
        s = json.loads(rs.read_text())["stage_b"]["choice"]
        ov["synthetic"] = deep_update(ov.get("synthetic") or {}, {"detector": {"credits": s["credits"]}})
        notes.append(f"synthetic: credits {s['credits']}, b {s['b']}")
        if s["b"] != b:
            notes.append(f"NOTE: synthetic preferred b={s['b']} but one shared b is required; using clinical b={b}")
    cand["budget"]["b"] = b
    cand["protocol_version"] = "v2-candidate (dev sweep; NOT frozen until supervisor review)"
    out = ROOT / "configs" / f"v2_candidate{sfx}.yaml"
    out.write_text("# Generated by `python -m src.sweep_dev --write-candidate` from dev-only sweeps.\n# "
                   + "\n# ".join(notes) + "\n# Next: re-run `python -m src.calibrate --dataset clinical "
                   "--config configs/v2.yaml --tag v2` after freezing (periodic interval).\n"
                   + yaml.safe_dump(cand, sort_keys=False))
    print("\n".join(notes))
    print(f"[sweep] wrote {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/core.yaml")
    ap.add_argument("--dataset", choices=["clinical", "synthetic"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--write-candidate", action="store_true")
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=None, help="override dev seeds, e.g. 101 102 103 104")
    ap.add_argument("--tag", default="", help="suffix for outputs, e.g. 4seeds (keeps earlier sweep files)")
    ap.add_argument("--fix-stage-a", nargs=3, metavar=("ALPHA", "DRIFT", "THRESHOLD"), default=None,
                    help="run stage B with this stage-A setting (so extra seeds match an earlier sweep)")
    ap.add_argument("--stage-b-only", action="store_true",
                    help="with --fix-stage-a: skip stage A entirely (rerun stage B for a new choice)")
    ap.add_argument("--merge", nargs="+", default=None, metavar="JSON",
                    help="merge earlier sweep JSON files (same dataset) into one, re-applying the rules")
    a = ap.parse_args()
    if a.merge:
        merge(a.merge, a.tag or "merged")
    elif a.write_candidate:
        write_candidate(a.config, a.tag)
    elif a.dataset:
        fix = (float(a.fix_stage_a[0]), a.fix_stage_a[1], float(a.fix_stage_a[2])) if a.fix_stage_a else None
        run(a.dataset, a.config, a.workers, stage_a_choice=fix, dev_seeds=a.dev_seeds, tag=a.tag,
            stage_a_only_fixed=a.stage_b_only)
    else:
        ap.error("give --dataset or --write-candidate")


if __name__ == "__main__":
    main()
