"""Tables and figures from DECLARED complete runs only (manifests), with pairing checks.

python -m src.analyze --manifests results/manifest_full_synthetic.json results/manifest_ablation_synthetic.json \
    results/manifest_sweep_synthetic.json
No significance stars: five seeds cannot support formal tests (see plan section 9).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .drift import get_scenario  # noqa: E402
from .utils import ROOT, atomic_write_csv  # noqa: E402

METHOD_ORDER = ["always_on", "periodic", "random", "drift_only", "known_time", "proposed",
                "proposed_nosev", "proposed_nowait"]
# Validated categorical palette (light): fixed order, never cycled. Line style is a secondary encoding.
COLORS = {"always_on": "#2a78d6", "periodic": "#eb6834", "random": "#1baf7a", "drift_only": "#eda100",
          "known_time": "#e87ba4", "proposed": "#008300", "proposed_nosev": "#6b6a64", "proposed_nowait": "#3d3c38"}
STYLES = {"always_on": "-", "periodic": "--", "random": ":", "drift_only": "-.", "known_time": (0, (5, 1, 1, 1)),
          "proposed": "-", "proposed_nosev": "--", "proposed_nowait": ":"}
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e6e5e0"


def load(manifests):
    expected, rows, missing = [], [], []
    for mp in manifests:
        m = json.loads(Path(mp).read_text())
        runs_dir = Path(m["runs_dir"])
        runs_dir = runs_dir if runs_dir.is_absolute() else ROOT / runs_dir
        for rid in m["expected"]:
            expected.append(rid)
            meta_p = runs_dir / rid / "run.json"
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            if meta.get("status") != "complete":
                missing.append((rid, meta.get("status", "missing")))
                continue
            s = json.loads((runs_dir / rid / "summary.json").read_text())
            s["run_dir"] = str(runs_dir / rid)
            s["config_hash"] = meta["config_hash"]
            rows.append(s)
    dup = pd.Series(expected)[pd.Series(expected).duplicated()].tolist()
    df = pd.DataFrame(rows).drop_duplicates("run_id")
    return df, missing, dup


def check_pairing(df):
    issues = []
    for (ds, sc, seed), g in df.groupby(["dataset", "scenario", "seed"]):
        if g.warm_checksum.nunique() > 1:
            issues.append(f"{ds}/{sc}/seed {seed}: warm-start weights differ across methods")
    for b, g in df.groupby("budget_b"):
        if g.config_hash.nunique() > 1:
            issues.append(f"b={b}: {g.config_hash.nunique()} different config hashes")
    return issues


def ms(x):
    x = pd.Series(x).dropna()
    if not len(x):
        return "n/a"
    return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}" if len(x) > 1 else f"{x.mean():.4f}"


def main_table(df, b):
    d = df[df.budget_b == b]
    cols = ["post_logloss_pooled", "post_logloss_macro", "post_logloss_worst_client", "post_auroc_pooled",
            "post_auprc_pooled", "post_brier_pooled", "bytes_adaptation", "bytes_monitoring", "bytes_end_to_end",
            "n_client_trainings"]
    agg = d.groupby(["dataset", "scenario", "method"])[cols].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b_}" for a, b_ in agg.columns]
    return agg.reset_index()


def paired(df, b, metric="post_logloss_pooled", base="proposed",
           against=("periodic", "drift_only", "random", "always_on", "known_time")):
    d = df[df.budget_b == b]
    out = []
    for (ds, sc), g in d.groupby(["dataset", "scenario"]):
        p = g[g.method == base].set_index("seed")[metric]
        for other in against:
            q = g[g.method == other].set_index("seed")[metric]
            common = p.index.intersection(q.index)
            if not len(common):
                continue
            diff = (p.loc[common] - q.loc[common])
            out.append(dict(dataset=ds, scenario=sc, comparison=f"{base} - {other}", metric=metric,
                            n_seeds=len(common), mean_diff=diff.mean(), sd_diff=diff.std(ddof=1),
                            seeds_proposed_better=int((diff < 0).sum()),
                            per_seed=";".join(f"{s}:{v:+.4f}" for s, v in diff.items())))
    return pd.DataFrame(out)


def drift_table(df, b):
    d = df[(df.budget_b == b) & df.method.isin(["drift_only", "proposed", "known_time"])]
    cols = ["detected", "n_events", "misses", "mean_detection_delay", "false_alarms_per_100_client_steps",
            "recovered", "censored", "mean_recovery_time", "expired_requests", "mean_alarm_to_service",
            "n_client_trainings"]
    return d.groupby(["dataset", "scenario", "method"])[cols].mean().reset_index()


def excess_vs_always_on(df, b):
    d = df[df.budget_b == b]
    ao = d[d.method == "always_on"].set_index(["dataset", "scenario", "seed"]).post_logloss_pooled
    d = d[d.method != "always_on"].copy()
    d["excess_loss_vs_always_on"] = [r.post_logloss_pooled - ao.get((r.dataset, r.scenario, r.seed), np.nan)
                                     for r in d.itertuples()]
    return d.groupby(["dataset", "scenario", "method"]).excess_loss_vs_always_on.agg(["mean", "std"]).reset_index()


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def loss_over_time(df, scenario, b, out, methods=("always_on", "periodic", "random", "drift_only", "known_time", "proposed")):
    d = df[(df.scenario == scenario) & (df.budget_b == b)]
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for m in methods:
        g = d[d.method == m]
        if g.empty:
            continue
        curves = []
        for r in g.itertuples():
            mt = pd.read_csv(Path(r.run_dir) / "metrics.csv")
            pooled = mt.groupby("time").apply(lambda x: np.average(x.log_loss, weights=x.sample_count),
                                              include_groups=False)
            curves.append(pooled.rolling(5, min_periods=1).mean().to_numpy())
        C = np.vstack(curves)
        mu, sd = C.mean(0), C.std(0, ddof=1) if len(C) > 1 else np.zeros(C.shape[1])
        t = np.arange(C.shape[1])
        ax.plot(t, mu, color=COLORS[m], linestyle=STYLES[m], linewidth=2, label=m.replace("_", "-"))
        ax.fill_between(t, mu - sd, mu + sd, color=COLORS[m], alpha=0.10, linewidth=0)
    for e in get_scenario(scenario).events:
        ax.axvline(e.onset, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.text(e.onset + 0.5, ax.get_ylim()[1], f"clinic {e.client + 1}", fontsize=7, color=MUTED, va="top")
    _style(ax, f"{scenario}: pooled prequential log loss (5-step rolling, mean ± sd over {d.seed.nunique()} seeds)",
           "stream step", "log loss")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def quality_vs_bytes(df, scenario, out):
    d = df[(df.scenario == scenario) & df.method.isin(["proposed", "periodic", "drift_only"])]
    if d.empty or d.budget_b.nunique() < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    for m in ["periodic", "drift_only", "proposed"]:
        g = d[d.method == m].groupby("budget_b").agg(x=("bytes_adaptation", "mean"), y=("post_logloss_pooled", "mean"),
                                                      ysd=("post_logloss_pooled", "std")).reset_index()
        ax.errorbar(g.x / 1e6, g.y, yerr=g.ysd, color=COLORS[m], linestyle=STYLES[m], marker="o", markersize=6,
                    linewidth=2, capsize=3, label=m.replace("_", "-"))
        for r in g.itertuples():
            ax.annotate(f"b={r.budget_b:g}", (r.x / 1e6, r.y), textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color=MUTED)
    _style(ax, f"{scenario}: post-change log loss vs actual adaptation bytes", "adaptation payload (MB, counted)",
           "post-change log loss")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def per_client(df, scenario, b, out):
    d = df[(df.scenario == scenario) & (df.budget_b == b)]
    if d.empty:
        return None
    methods = [m for m in METHOD_ORDER[:6] if m in set(d.method)]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        g = d[d.method == m]
        vals = [g[f"post_logloss_client{c}"].mean() for c in range(4)]
        errs = [g[f"post_logloss_client{c}"].std(ddof=1) for c in range(4)]
        ax.bar(np.arange(4) + (i - (len(methods) - 1) / 2) * w, vals, w * 0.9, yerr=errs, color=COLORS[m],
               label=m.replace("_", "-"), capsize=2, error_kw=dict(ecolor=MUTED, lw=0.8))
    affected = {e.client: e.onset for e in get_scenario(scenario).events}
    ax.set_xticks(range(4), [f"Clinic {c + 1}\n" + (f"drift t={affected[c]}" if c in affected else "unaffected")
                             for c in range(4)])
    lo = np.nanmin([d[f"post_logloss_client{c}"].min() for c in range(4)])
    ax.set_ylim(max(0, lo - 0.05), None)
    _style(ax, f"{scenario}: per-clinic post-change log loss (mean ± sd over seeds)", "", "log loss")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--b", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df, missing, dup = load(a.manifests)
    issues = check_pairing(df)
    print(f"[analyze] {len(df)} complete runs, {len(missing)} missing/failed, {len(dup)} duplicate ids")
    for i in issues:
        print("[analyze] PAIRING ISSUE:", i)
    summary_cols = [c for c in df.columns if c != "run_dir"]
    atomic_write_csv(out / "tables" / "summary.csv", df[summary_cols])
    atomic_write_csv(out / "tables" / "missing_runs.csv", pd.DataFrame(missing, columns=["run_id", "status"]))
    t2 = main_table(df, a.b)
    atomic_write_csv(out / "tables" / "table2_main.csv", t2)
    pd_ll = paired(df, a.b)
    pd_w = paired(df, a.b, "post_logloss_worst_client")
    atomic_write_csv(out / "tables" / "paired_differences.csv", pd.concat([pd_ll, pd_w]))
    t3 = drift_table(df, a.b)
    atomic_write_csv(out / "tables" / "table3_drift.csv", t3)
    ex = excess_vs_always_on(df, a.b)
    atomic_write_csv(out / "tables" / "excess_vs_always_on.csv", ex)
    abl = df[df.method.isin(["proposed", "proposed_nosev", "proposed_nowait"]) & (df.budget_b == a.b)]
    abl_t = abl.groupby(["dataset", "scenario", "method"])[["post_logloss_pooled", "post_logloss_worst_client",
                                                           "mean_alarm_to_service", "expired_requests",
                                                           "bytes_adaptation"]].agg(["mean", "std"])
    sweep = df[df.method.isin(["proposed", "periodic", "drift_only"])].groupby(
        ["dataset", "scenario", "method", "budget_b"])[["post_logloss_pooled", "bytes_adaptation", "n_client_trainings"]].mean()
    figs = []
    for sc in sorted(df.scenario.unique()):
        figs.append(loss_over_time(df, sc, a.b, out / "figures" / f"fig2_loss_over_time_{sc}.png"))
        figs.append(per_client(df, sc, a.b, out / "figures" / f"fig4_per_client_{sc}.png"))
        figs.append(quality_vs_bytes(df, sc, out / "figures" / f"fig3_quality_vs_bytes_{sc}.png"))
    figs = [f for f in figs if f]
    # markdown report
    lines = ["# Results (auto-generated from declared complete runs)", "",
             f"Runs: {len(df)} complete; missing/failed: {len(missing)}; duplicates: {len(dup)}; "
             f"pairing issues: {len(issues) or 'none'}.", "",
             "Five seeds: descriptive only, no significance tests.", "", f"## Table 2 — post-change quality and bytes (b={a.b:g})", "",
             "| data | scen. | method | post log loss (pooled) | macro | worst clinic | AUROC | adapt. MB | end-to-end MB | client trainings |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    d = df[df.budget_b == a.b]
    for (ds, sc), g in d.groupby(["dataset", "scenario"]):
        for m in METHOD_ORDER:
            h = g[g.method == m]
            if h.empty:
                continue
            lines.append(f"| {ds} | {sc} | {m} | {ms(h.post_logloss_pooled)} | {ms(h.post_logloss_macro)} | "
                         f"{ms(h.post_logloss_worst_client)} | {ms(h.post_auroc_pooled)} | "
                         f"{h.bytes_adaptation.mean() / 1e6:.2f} | {h.bytes_end_to_end.mean() / 1e6:.2f} | "
                         f"{h.n_client_trainings.mean():.0f} |")
    lines += ["", "## Paired differences (proposed − other), post log loss pooled; negative = proposed better", "",
              "| scen. | comparison | mean diff | sd | seeds proposed better |", "|---|---|---|---|---|"]
    for r in pd_ll.itertuples():
        lines.append(f"| {r.scenario} | {r.comparison} | {r.mean_diff:+.4f} | {r.sd_diff:.4f} | "
                     f"{r.seeds_proposed_better}/{r.n_seeds} |")
    lines += ["", "## Drift outcomes (means over seeds)", "", t3.round(3).to_markdown(index=False)
              if hasattr(t3, "to_markdown") and _has_tabulate() else t3.round(3).to_string(index=False), ""]
    lines += ["## Priority ablation", "", "```", abl_t.round(4).to_string() if len(abl_t) else "not yet measured", "```",
              "", "## Budget sweep", "", "```", sweep.round(4).to_string() if len(sweep) else "not yet measured", "```", ""]
    if missing:
        lines += ["## Missing / failed runs", ""] + [f"- {r} ({s})" for r, s in missing]
    (out / "tables" / "results.md").write_text("\n".join(lines))
    print(f"[analyze] wrote {out / 'tables'} and {len(figs)} figures")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
