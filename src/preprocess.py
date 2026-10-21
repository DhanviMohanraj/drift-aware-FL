"""Clinical preprocessing with patient-grouped pools and warm-start-only fitting.

Pools (by patient): development 20% | warm-start 20% (of which `calib_frac` is the
disjoint sentinel-calibration subset) | final stream 60%.
The transformer is fit on warm-start TRAINING records only.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .utils import ROOT, STREAM_SPLIT, atomic_write_json, atomic_write_pickle, rng_for, sha256_array, sha256_file, stable_hash

UNKNOWN, OTHER, MISSING = "__unknown__", "__other__", "__missing__"


class TabularPreprocessor:
    def __init__(self, numeric, categorical, min_count=10):
        self.numeric, self.categorical, self.min_count = list(numeric), list(categorical), min_count
        self.fitted_ids: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, sample_ids: np.ndarray):
        self.fitted_ids = np.asarray(sample_ids).copy()
        self.median = {c: float(pd.to_numeric(df[c], errors="coerce").median()) for c in self.numeric}
        self.indicator = [c for c in self.numeric if df[c].isna().any()]
        num = self._numeric(df)
        self.mean = num.mean(0)
        self.sd = num.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        self.vocab, self.warm_levels = {}, {}
        for c in self.categorical:
            vc = self._cat(df, c).value_counts()
            self.warm_levels[c] = set(vc.index.tolist())
            keep = sorted(vc[vc >= self.min_count].index.tolist())
            self.vocab[c] = keep + [OTHER, UNKNOWN]
        self.feature_names = (list(self.numeric) + [f"{c}__isna" for c in self.indicator]
                              + [f"{c}={v}" for c in self.categorical for v in self.vocab[c]])
        return self

    def _numeric(self, df):
        cols = [pd.to_numeric(df[c], errors="coerce").fillna(self.median[c]).to_numpy(float) for c in self.numeric]
        return np.stack(cols, 1)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = (self._numeric(df) - self.mean) / self.sd
        ind = np.stack([df[c].isna().to_numpy(float) for c in self.indicator], 1) if self.indicator \
            else np.zeros((len(df), 0))
        blocks = [num, ind]
        for c in self.categorical:
            voc = self.vocab[c]
            pos = {v: i for i, v in enumerate(voc)}
            raw = self._cat(df, c).to_numpy()
            idx = np.array([pos[v] if v in pos and v not in (OTHER, UNKNOWN)
                            else pos[OTHER] if v in self.warm_levels[c] else pos[UNKNOWN] for v in raw],
                           dtype=np.int64)
            oh = np.zeros((len(df), len(voc)))
            oh[np.arange(len(df)), idx] = 1.0
            blocks.append(oh)
        return np.concatenate(blocks, 1).astype(np.float32)

    @staticmethod
    def _cat(df, c):
        col = df[c].astype(object)
        return col.where(col.notna(), MISSING).astype(str)


def _feature_lists(df):
    numeric = [c for c in D.NUMERIC_COLS if c in df.columns]
    categorical = [c for c in D.ID_CODED_CATEGORICAL + D.OTHER_CATEGORICAL + D.MEDICATIONS if c in df.columns]
    return numeric, categorical


def audit_columns(df) -> dict:
    numeric, categorical = _feature_lists(df)
    used = set(numeric) | set(categorical)
    dropped = [c for c in df.columns if c not in used]
    return {"n_rows": int(len(df)), "n_patients": int(df["patient_nbr"].nunique()),
            "numeric": numeric, "categorical": categorical, "dropped": dropped,
            "dropped_reason": {**{c: "identifier (grouping only)" for c in D.ID_COLS},
                               D.TARGET_COL: "target",
                               **{c: "high cardinality" for c in D.DROPPED_HIGH_CARDINALITY},
                               **{c: "mostly missing" for c in D.DROPPED_MOSTLY_MISSING}},
            "outcome_derived_check": "only `readmitted` encodes the outcome; it is the target and excluded. "
                                     "discharge_disposition_id (incl. expired) is a discharge-time field and is kept "
                                     "because prediction is defined at discharge."}


def dirichlet_allocate(y_pools: dict, alpha: float, k: int, seed: int, min_sizes: dict, max_retries: int):
    """Per-class Dirichlet proportions shared across pools. Returns (client_idx, proportions, retries)."""
    for retry in range(max_retries):
        g = rng_for(seed, STREAM_SPLIT, 77, retry)
        props = {cls: g.dirichlet(alpha * np.ones(k)) for cls in (0, 1)}
        out = {}
        for pool, y in y_pools.items():
            lists = [[] for _ in range(k)]
            for cls in (0, 1):
                idx = np.flatnonzero(y == cls)
                idx = idx[g.permutation(len(idx))]
                cuts = (np.cumsum(props[cls])[:-1] * len(idx)).astype(int)
                for c, part in enumerate(np.split(idx, cuts)):
                    lists[c].extend(part.tolist())
            out[pool] = [np.sort(np.array(l, dtype=np.int64)) for l in lists]
        ok = all(min(len(v) for v in out[p]) >= m for p, m in min_sizes.items())
        if ok:
            return out, {cls: p.tolist() for cls, p in props.items()}, retry
    raise RuntimeError("Dirichlet allocation failed minimum client sizes; reduce min sizes or alpha")


def prepare_clinical(cfg: dict, seed: int, df: pd.DataFrame | None = None, cache: bool = True) -> dict:
    c = cfg["clinical"]
    csv = Path(c["csv_path"])
    if not csv.is_absolute():
        csv = ROOT / csv
    src_hash = sha256_file(csv) if df is None else stable_hash(len(df))
    key = stable_hash({"c": c, "seed": seed, "src": src_hash, "H": cfg["horizon"], "k": cfg["num_clients"], "v": 2})
    cpath = ROOT / "results" / "cache" / f"clinical_{seed}_{key}.pkl"
    from_csv = df is None
    if cache and from_csv and cpath.exists():
        return pickle.loads(cpath.read_bytes())
    if from_csv:
        df = D.load_raw(csv)
    audit = audit_columns(df)
    df = df.copy()
    df["_y"] = (df[D.TARGET_COL].astype(str) == "<30").astype(np.float32)
    # one seeded (non-chronological) encounter per patient
    g = rng_for(seed, STREAM_SPLIT, 1)
    df = df.iloc[g.permutation(len(df))].drop_duplicates("patient_nbr", keep="first").reset_index(drop=True)
    df["_sid"] = df["encounter_id"].astype(np.int64)
    patients = np.array(sorted(df["patient_nbr"].unique()))
    patients = patients[rng_for(seed, STREAM_SPLIT, 2).permutation(len(patients))]
    n = len(patients)
    a, b = int(c["pools"][0] * n), int((c["pools"][0] + c["pools"][1]) * n)
    dev_p, warm_all_p, final_p = patients[:a], patients[a:b], patients[b:]
    ncal = int(c["calib_frac"] * len(warm_all_p))
    calib_p, warm_p = warm_all_p[:ncal], warm_all_p[ncal:]
    pools = {}
    for name, ps in (("dev", dev_p), ("warm", warm_p), ("calib", calib_p), ("final", final_p)):
        pools[name] = df[df["patient_nbr"].isin(set(ps.tolist()))].reset_index(drop=True)
    numeric, categorical = _feature_lists(df)
    pre = TabularPreprocessor(numeric, categorical, c["min_category_count"])
    pre.fit(pools["warm"], pools["warm"]["_sid"].to_numpy())
    X = {k: pre.transform(v) for k, v in pools.items()}
    y = {k: v["_y"].to_numpy(np.float32) for k, v in pools.items()}
    ids = {k: v["_sid"].to_numpy(np.int64) for k, v in pools.items()}
    shift_feature = c["shift_feature"]
    if shift_feature not in pre.feature_names or pre.sd[numeric.index(shift_feature)] <= 1e-8:
        raise RuntimeError(f"shift feature {shift_feature} missing or has zero scale")
    k = cfg["num_clients"]
    alloc, props, retries = dirichlet_allocate(
        y, c["dirichlet_alpha"], k, seed,
        {"final": c["min_client_final"], "warm": c["min_client_warm"]}, c["max_retries"])
    H = cfg["horizon"]
    chunks = {}
    for split in ("final", "dev"):
        chunks[split] = []
        for cl in range(k):
            idx = alloc[split][cl]
            idx = idx[rng_for(seed, STREAM_SPLIT, 3, cl, 0 if split == "final" else 1).permutation(len(idx))]
            if len(idx) < H:
                raise RuntimeError(f"client {cl} {split} has {len(idx)} < horizon {H} records")
            chunks[split].append(np.array_split(idx, H))
    prepared = {
        "X": X, "y": y, "ids": ids, "client_idx": alloc, "chunks": chunks, "horizon": H,
        "feature_names": pre.feature_names, "shift_feature": shift_feature,
        "shift_col_index": pre.feature_names.index(shift_feature), "audit": audit,
        "preprocessor_fit_ids": pre.fitted_ids,
        "patients": {"dev": dev_p, "warm": warm_p, "calib": calib_p, "final": final_p},
        "counts": {s: {"rows": int(len(pools[s])), "patients": int(pools[s]["patient_nbr"].nunique()),
                       "prevalence": float(y[s].mean()),
                       "per_client": [int(len(v)) for v in alloc[s]],
                       "per_client_prevalence": [float(y[s][v].mean()) if len(v) else None for v in alloc[s]]}
                   for s in pools},
        "dirichlet": {"alpha": c["dirichlet_alpha"], "proportions": props, "retries": retries},
        "checksums": {"dataset": src_hash,
                      "split": sha256_array(*[ids[s] for s in ("dev", "warm", "calib", "final")]),
                      "preprocessing": stable_hash({"f": pre.feature_names, "m": pre.mean.tolist(), "s": pre.sd.tolist()})},
    }
    if cache and from_csv:
        atomic_write_pickle(cpath, prepared)
        atomic_write_json(cpath.with_suffix(".json"), {"counts": prepared["counts"], "audit": audit,
                                                       "dirichlet": prepared["dirichlet"],
                                                       "checksums": prepared["checksums"],
                                                       "n_features": len(pre.feature_names)})
    return prepared
