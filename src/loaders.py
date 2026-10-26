"""
loaders.py — Real-dataset loaders. Each returns (X, y, p) matching the synthetic
interface so inject() works unchanged.

  X : DataFrame of base features (keep native missingness as-is; it is pre-screened
      separately in run_experiment.py so it doesn't confound injected features)
  y : binary target (0/1 numpy array)
  p : per-row risk probability used to build a LEGITIMATE risk tier for miss_L.
      We fit a quick logistic model on the complete-case features to get p.

Download instructions are in data/README.md.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _risk_proba(X, y):
    """Quick, leakage-free p estimate for building the legitimate risk tier."""
    num = X.select_dtypes(include=[np.number]).copy()
    pipe = make_pipeline(SimpleImputer(strategy="median"),
                         StandardScaler(),
                         LogisticRegression(max_iter=1000))
    pipe.fit(num, y)
    return pipe.predict_proba(num)[:, 1]


def load_give_me_some_credit(path):
    """
    Kaggle 'Give Me Some Credit' training CSV (cs-training.csv).
    Target: SeriousDlqin2yrs. Has native missingness in MonthlyIncome and
    NumberOfDependents (~20%) — this is real informative missingness; we keep it.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")],
                 errors="ignore")
    y = df["SeriousDlqin2yrs"].astype(int).values
    X = df.drop(columns=["SeriousDlqin2yrs"])
    p = _risk_proba(X, y)
    return X, y, p


def load_mimic_demo(path_admissions, path_labevents=None):
    """
    MIMIC-IV demo -> a simple tabular mortality task.
    Minimal version: use hospital_expire_flag (if present) as target and a few
    admission-level numeric columns. For a richer version, join lab means per
    admission (left as a TODO so you control the clinical preprocessing).

    Returns (X, y, p). Edit column names to match your extracted CSV.
    """
    adm = pd.read_csv(path_admissions)
    # --- adapt these column names to your extraction ---
    target_col = "hospital_expire_flag"
    if target_col not in adm.columns:
        raise ValueError(f"expected '{target_col}' in admissions; got {list(adm.columns)[:10]}")
    y = adm[target_col].astype(int).values
    drop = [target_col]
    X = adm.drop(columns=[c for c in drop if c in adm.columns])
    X = X.select_dtypes(include=[np.number])     # keep numeric for the minimal demo
    p = _risk_proba(X, y)
    return X, y, p


def load_adult(path=None):
    """
    UCI Adult / Census Income. Binary target: income >50K.
    Real informative missingness: workclass / occupation / native-country are
    recorded as '?' for a non-random subset whose income distribution differs
    (people with no recorded occupation skew lower-income). We convert '?' -> NaN
    and KEEP those columns as native informative missingness.

    `path` : local adult.data / adult.csv. If None, fetches from OpenML.
             UCI raw has no header; we assign the standard column names.
    """
    cols = ["age","workclass","fnlwgt","education","education_num","marital_status",
            "occupation","relationship","race","sex","capital_gain","capital_loss",
            "hours_per_week","native_country","income"]
    if path:
        df = pd.read_csv(path, header=None, names=cols, skipinitialspace=True,
                         na_values=["?"])
        # some mirrors already have a header row; detect and redo if so
        if str(df.iloc[0]["age"]).strip().lower() in ("age", "nan"):
            df = pd.read_csv(path, skipinitialspace=True, na_values=["?"])
    else:
        from sklearn.datasets import fetch_openml
        d = fetch_openml("adult", version=2, as_frame=True)
        df = d.frame.copy()
        df.columns = [c.replace("-", "_") for c in df.columns]
        # openml target col is 'class'
        if "class" in df.columns and "income" not in df.columns:
            df = df.rename(columns={"class": "income"})
        df = df.replace("?", np.nan)

    # target: >50K -> 1
    inc = df["income"].astype(str).str.strip().str.replace(".", "", regex=False)
    y = (inc.str.contains(">50K")).astype(int).values
    X = df.drop(columns=["income"])

    # numeric encode for the risk-tier model only (X keeps raw cols incl. NaNs)
    num = X.copy()
    for c in num.select_dtypes(exclude=[np.number]).columns:
        num[c] = num[c].astype("category").cat.codes.replace(-1, np.nan)  # -1 == NaN
    p = _risk_proba(num, y)
    return X, y, p


def load_mimic(path):
    """
    Clean MIMIC CSV produced by extract_mimic.py (columns: features... + target).
    Target: in-hospital mortality. Native missingness (language, marital_status,
    ed_hours, ...) is kept as-is. Returns (X, y, p).
    """
    df = pd.read_csv(path)
    y = df["target"].astype(int).values
    X = df.drop(columns=["target"])
    p = _risk_proba(X, y)
    return X, y, p


REAL_LOADERS = {
    "adult": load_adult,
    "mimic": load_mimic,
    "gmc": load_give_me_some_credit,
    "mimic_demo": load_mimic_demo,
}