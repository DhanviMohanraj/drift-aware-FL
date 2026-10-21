"""UCI Diabetes 130-US Hospitals (dataset 296): download, manifest and loading.

Run once (e.g. in Colab):  python -m src.data --out data/raw
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import atomic_write_json, sha256_file

UCI_PAGE = "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
CITATION = ("Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for "
            "Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J")
LICENSE_NOTE = "As displayed on the UCI dataset page (CC BY 4.0 at time of writing) - verify before release."

ID_COLS = ["encounter_id", "patient_nbr"]
TARGET_COL = "readmitted"
NUMERIC_COLS = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
                "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
ID_CODED_CATEGORICAL = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
DROPPED_HIGH_CARDINALITY = ["diag_1", "diag_2", "diag_3"]   # ~700+ ICD-9 codes each
DROPPED_MOSTLY_MISSING = ["weight"]                          # ~97% missing
MEDICATIONS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
               "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
               "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
               "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
               "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
OTHER_CATEGORICAL = ["race", "gender", "age", "payer_code", "medical_specialty", "max_glu_serum",
                     "A1Cresult", "change", "diabetesMed"]


def download(out_dir: str = "data/raw") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "diabetic_data.csv"
    source = UCI_ZIP
    try:
        with urllib.request.urlopen(UCI_ZIP, timeout=120) as r:
            blob = r.read()
        (out / "dataset296.zip").write_bytes(blob)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            inner = [n for n in z.namelist() if n.endswith("diabetic_data.csv")]
            if not inner:   # some UCI zips nest a second zip
                nested = [n for n in z.namelist() if n.endswith(".zip")]
                z = zipfile.ZipFile(io.BytesIO(z.read(nested[0])))
                inner = [n for n in z.namelist() if n.endswith("diabetic_data.csv")]
            with z.open(inner[0]) as src, open(csv, "wb") as dst:
                shutil.copyfileobj(src, dst)
    except Exception as e:  # fall back to ucimlrepo
        print(f"[data] zip download failed ({e}); trying ucimlrepo")
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=296)
        parts = [p for p in (d.data.ids, d.data.features, d.data.targets) if p is not None]
        df = pd.concat(parts, axis=1)
        missing = [c for c in ID_COLS + [TARGET_COL] if c not in df.columns]
        if missing:
            raise RuntimeError(f"ucimlrepo frame lacks {missing}; download the zip manually from {UCI_PAGE}")
        df.to_csv(csv, index=False)
        source = "ucimlrepo id=296"
    manifest = {
        "dataset": "UCI Diabetes 130-US Hospitals for Years 1999-2008 (id 296)",
        "page_url": UCI_PAGE, "download_source": source,
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "citation": CITATION, "license": LICENSE_NOTE,
        "file": str(csv), "sha256": sha256_file(csv),
    }
    atomic_write_json(out / "data_manifest.json", manifest)
    print(f"[data] wrote {csv} sha256={manifest['sha256']}")
    return csv


def load_raw(csv_path) -> pd.DataFrame:
    # keep_default_na=False: the literal string "None" in A1Cresult/max_glu_serum is a valid level
    df = pd.read_csv(csv_path, na_values=["?"], keep_default_na=False, low_memory=False)
    df = df.replace({"": np.nan})
    return df


def make_fixture(n_patients: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Schema-faithful FAKE data for tests and offline smoke checks. Not clinical data."""
    g = np.random.default_rng(seed)
    n_enc = int(n_patients * 1.4)
    pat = g.integers(0, n_patients, n_enc)
    df = pd.DataFrame({"encounter_id": np.arange(n_enc) * 7 + 1000, "patient_nbr": pat * 13 + 5})
    for c in NUMERIC_COLS:
        df[c] = g.poisson({"num_lab_procedures": 43, "num_medications": 16}.get(c, 3), n_enc)
    lev = {"race": ["Caucasian", "AfricanAmerican", "Hispanic", "Other", np.nan],
           "gender": ["Female", "Male"], "age": [f"[{a}-{a+10})" for a in range(0, 100, 10)],
           "payer_code": ["MC", "HM", "SP", np.nan], "medical_specialty": ["InternalMedicine", "Cardiology", np.nan],
           "max_glu_serum": ["None", ">200", ">300", "Norm"], "A1Cresult": ["None", ">7", ">8", "Norm"],
           "change": ["No", "Ch"], "diabetesMed": ["Yes", "No"], "weight": [np.nan] * 30 + ["[75-100)"]}
    for c, v in lev.items():
        df[c] = g.choice(np.array(v, dtype=object), n_enc)
    for c in ID_CODED_CATEGORICAL:
        df[c] = g.integers(1, 8, n_enc)
    for c in DROPPED_HIGH_CARDINALITY:
        df[c] = g.integers(1, 900, n_enc).astype(str)
    for c in MEDICATIONS:
        df[c] = g.choice(np.array(["No", "Steady", "Up", "Down"], dtype=object), n_enc, p=[0.7, 0.2, 0.05, 0.05])
    logit = -2.3 + 0.25 * (df["number_inpatient"] - 3) + 0.02 * (df["num_lab_procedures"] - 43)
    p = 1 / (1 + np.exp(-logit))
    r = g.random(n_enc)
    df[TARGET_COL] = np.where(r < p, "<30", np.where(r < p + 0.35, ">30", "NO"))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    download(ap.parse_args().out)
