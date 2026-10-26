"""
extract_mimic.py — Build one clean tabular CSV from the MIMIC-IV demo tables.

Joins admissions.csv + patients.csv, builds the in-hospital mortality target, and
derives a modeling-ready feature table. MIMIC's own informative missingness
(marital_status, language, discharge_location, etc.) is KEPT as-is — those blanks
are non-random and correlate with the outcome, which is the real-world version of
the phenomenon this project studies.

Usage:
    python -m src.extract_mimic --adm data/raw/admissions.csv \
                                --pat data/raw/patients.csv \
                                --out data/raw/mimic.csv

Then run the benchmark on it:
    python -m src.run_experiment --real mimic --path data/raw/mimic.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def build(adm_path, pat_path):
    adm = pd.read_csv(adm_path)
    pat = pd.read_csv(pat_path)

    df = adm.merge(pat[["subject_id", "gender", "anchor_age"]],
                   on="subject_id", how="left")

    # --- target: in-hospital mortality ---
    y = df["hospital_expire_flag"].astype(int)

    # --- derived numeric features ---
    admit = pd.to_datetime(df["admittime"], errors="coerce")
    disch = pd.to_datetime(df["dischtime"], errors="coerce")
    ed_in = pd.to_datetime(df["edregtime"], errors="coerce")
    ed_out = pd.to_datetime(df["edouttime"], errors="coerce")

    los_days = (disch - admit).dt.total_seconds() / 86400.0        # length of stay
    ed_hours = (ed_out - ed_in).dt.total_seconds() / 3600.0        # ED time (often NaN)

    out = pd.DataFrame({
        "age": pd.to_numeric(df["anchor_age"], errors="coerce"),
        "gender": df["gender"].astype("category").cat.codes.replace(-1, np.nan),
        "los_days": los_days.round(3),
        "ed_hours": ed_hours.round(3),
        # categorical context (encoded; native missingness preserved as NaN)
        "admission_type": df["admission_type"].astype("category").cat.codes.replace(-1, np.nan),
        "admission_location": df["admission_location"].astype("category").cat.codes.replace(-1, np.nan),
        "insurance": df["insurance"].astype("category").cat.codes.replace(-1, np.nan),
        "language": df["language"].astype("category").cat.codes.replace(-1, np.nan),
        "marital_status": df["marital_status"].astype("category").cat.codes.replace(-1, np.nan),
        "race": df["race"].astype("category").cat.codes.replace(-1, np.nan),
    })
    # NOTE: do NOT include discharge_location/deathtime/dischtime as features —
    # they are post-outcome and would be trivial target leakage themselves.

    out["target"] = y.values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adm", default="data/raw/admissions.csv")
    ap.add_argument("--pat", default="data/raw/patients.csv")
    ap.add_argument("--out", default="data/raw/mimic.csv")
    a = ap.parse_args()
    df = build(a.adm, a.pat)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: shape={df.shape}, target rate={df['target'].mean():.3f}")
    miss = df.drop(columns=["target"]).isna().mean()
    miss = miss[miss > 0].round(3)
    if len(miss):
        print("native missingness (kept):")
        print(miss.to_string())
    else:
        print("note: no native missingness in the derived columns "
              "(demo is small; the injected features still drive the benchmark)")


if __name__ == "__main__":
    main()