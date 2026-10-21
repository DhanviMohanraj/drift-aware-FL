"""Prequential predictive metrics and drift outcomes (detection, false alarms, recovery)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-7


def _ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auc_pair(y, p) -> tuple[float, float, str]:
    """(AUROC, AUPRC, reason). Single-class sets are undefined (NaN), never 0.5 or 0."""
    y = np.asarray(y)
    if len(y) == 0:
        return np.nan, np.nan, "empty"
    if np.unique(y).size < 2:
        return np.nan, np.nan, "single_class"
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p)), ""


class Evaluator:
    """Evaluator.record(predictions, labels, client, time). Only the evaluator sees event metadata."""

    def __init__(self, event_manifest: list[dict] | None = None):
        self.events = list(event_manifest or [])
        self.rows: list[pd.DataFrame] = []
        self.step_rows: list[dict] = []

    def record(self, predictions, labels, client, time, sample_ids=None, deployed_version=0, cumulative_bytes=0):
        p = np.asarray(predictions, float)
        y = np.asarray(labels, float)
        if not np.all(np.isfinite(p)):
            raise FloatingPointError(f"non-finite predictions at client {client} t={time}")
        sid = np.asarray(sample_ids) if sample_ids is not None else np.arange(len(y))
        self.rows.append(pd.DataFrame({"sample_id": sid, "client_id": client, "time": time,
                                       "target": y, "probability": p}))
        self.step_rows.append(dict(client_id=client, time=time, sample_count=len(y),
                                   positive_count=int(y.sum()), log_loss=float(_ll(y, p).mean()),
                                   Brier=float(((p - y) ** 2).mean()), deployed_version=deployed_version,
                                   cumulative_bytes=cumulative_bytes))

    def predictions(self) -> pd.DataFrame:
        return pd.concat(self.rows, ignore_index=True)

    def metrics_frame(self, window: int) -> pd.DataFrame:
        m = pd.DataFrame(self.step_rows)
        pred = self.predictions()
        m["window_AUROC"] = np.nan
        m["window_AUPRC"] = np.nan
        m["window_reason"] = ""
        for (c, w), g in pred.groupby([pred.client_id, pred.time // window]):
            a, ap, why = auc_pair(g.target, g.probability)
            sel = (m.client_id == c) & (m.time == g.time.max())
            m.loc[sel, ["window_AUROC", "window_AUPRC"]] = [a, ap]
            m.loc[sel, "window_reason"] = why or "ok"
        return m

    # ---------------- summaries ----------------
    def predictive_summary(self, post_start: int, K: int) -> dict:
        pred = self.predictions()
        post = pred[pred.time >= post_start]
        out = {"post_start": post_start}
        for tag, d in (("post", post), ("full", pred)):
            ll = _ll(d.target.to_numpy(), d.probability.to_numpy())
            out[f"{tag}_logloss_pooled"] = float(ll.mean())
            per = [float(_ll(g.target.to_numpy(), g.probability.to_numpy()).mean()) for _, g in d.groupby("client_id")]
            out[f"{tag}_logloss_macro"] = float(np.mean(per))
            out[f"{tag}_logloss_worst_client"] = float(np.max(per))
            out[f"{tag}_brier_pooled"] = float(((d.probability - d.target) ** 2).mean())
            a, ap, why = auc_pair(d.target, d.probability)
            out[f"{tag}_auroc_pooled"], out[f"{tag}_auprc_pooled"] = a, ap
            aucs = [auc_pair(g.target, g.probability)[0] for _, g in d.groupby("client_id")]
            valid = [x for x in aucs if np.isfinite(x)]
            out[f"{tag}_auroc_worst_client"] = float(min(valid)) if valid else np.nan
        for c, g in post.groupby("client_id"):
            out[f"post_logloss_client{c}"] = float(_ll(g.target.to_numpy(), g.probability.to_numpy()).mean())
        return out


def detection_summary(events_df: pd.DataFrame, manifest: list[dict], horizon: int, K: int,
                      match_window: int) -> dict:
    """Match first alarm within [onset, end + match_window] to each event; others are false alarms."""
    alarms = events_df[events_df.flag == 1][["client_id", "time"]].to_numpy()
    matched = set()
    delays, misses = [], 0
    windows = []
    for e in manifest:
        lo, hi = e["onset"], e["end"] + match_window
        windows.append((e["client"], lo, hi))
        cand = [(i, t) for i, (c, t) in enumerate(alarms) if c == e["client"] and lo <= t <= hi]
        if cand:
            i, t = cand[0]
            matched.add(i)
            delays.append(int(t - e["onset"]))
        else:
            misses += 1
    in_window = np.zeros((K, horizon), bool)
    for c, lo, hi in windows:
        in_window[c, lo:min(hi + 1, horizon)] = True
    fa = [i for i, (c, t) in enumerate(alarms) if not in_window[c, t]]
    eligible = int((~in_window).sum())
    return {"n_alarms": int(len(alarms)), "n_events": len(manifest), "detected": len(delays), "misses": misses,
            "mean_detection_delay": float(np.mean(delays)) if delays else np.nan,
            "detection_delays": delays, "false_alarms": len(fa), "eligible_client_steps": eligible,
            "false_alarms_per_100_client_steps": 100.0 * len(fa) / eligible if eligible else np.nan}


def recovery_summary(metrics_df: pd.DataFrame, manifest: list[dict], ev: dict) -> dict:
    """First of `consecutive` rolling `window`-step loss windows within tolerance of pre-event loss.
    Unrecovered events are censored (NaN time, counted)."""
    times, censored = [], 0
    detail = []
    for e in manifest:
        m = metrics_df[metrics_df.client_id == e["client"]].sort_values("time")
        loss = m.set_index("time")["log_loss"]
        pre = loss.loc[max(0, e["onset"] - ev["pre_event_steps"]):e["onset"] - 1]
        ref = float(pre.mean()) if len(pre) else np.nan
        thr = max(ref * (1 + ev["recovery_tolerance"]), ref + ev["recovery_floor"])
        roll = loss.rolling(ev["recovery_window"]).mean()
        ok = (roll <= thr)
        rec = None
        ts = [t for t in roll.index if t >= e["end"] + ev["recovery_window"] - 1]
        for t in ts:
            seq = [t + i for i in range(ev["recovery_consecutive"])]
            if all(s in ok.index and bool(ok.loc[s]) for s in seq):
                rec = t - e["onset"]
                break
        if rec is None:
            censored += 1
        else:
            times.append(rec)
        detail.append({"client": e["client"], "onset": e["onset"], "ref_loss": ref, "recovery_time": rec})
    return {"recovered": len(times), "censored": censored,
            "mean_recovery_time": float(np.mean(times)) if times else np.nan, "recovery_detail": detail}
