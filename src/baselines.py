"""
baselines.py — Adapters for each leakage detector. Each returns, per injected
feature, {'flagged': bool, 'label': str|None}. Tools wrapped in try/except by the
caller. The four injected features are always miss_L / miss_K / miss_S / miss_N.

Installed APIs verified:
  featureleak : featureleak.rules.missingness_leakage.detect_missingness_leakage
  leaklint    : leaklint.detect_leakage(X, y, ...)
  dataleaks   : dataleaks.DataLeaks(df, target=...).run()
  cleanlab    : row/dataset issues only -> no per-feature verdict (capability gap)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

FEATS = ["miss_L", "miss_K", "miss_S", "miss_N"]

def _blank():
    return {f: {"flagged": False, "label": None} for f in FEATS}

# --- canonical association-threshold baseline (the obvious missingness detector) ---
def run_assoc_threshold(df, target="target", thr=0.10):
    y = df[target].values
    res = _blank()
    for f in FEATS:
        a = (abs(np.corrcoef(df[f].isna().astype(int), y)[0, 1])
             if df[f].isna().any() else 0.0)
        if a >= thr:
            res[f] = {"flagged": True, "label": f"assoc={a:.2f}"}
    return res

# --- featureleak: dedicated missingness rule (thresholds the same association) ---
def run_featureleak(df, target="target", thr=0.20):
    from featureleak.rules.missingness_leakage import detect_missingness_leakage
    d = df if len(df) >= 10000 else pd.concat([df]*(10000//len(df)+1), ignore_index=True)
    issues = detect_missingness_leakage(d, target=target, association_threshold=thr)
    res = _blank()
    for iss in issues:
        for c in iss.affected_columns:
            if c in res:
                res[c] = {"flagged": True, "label": iss.severity}
    return res

# --- leaklint: value-based leakage (won't see missingness; documents the gap) ---
def run_leaklint(df, target="target"):
    import leaklint
    X = df.drop(columns=[target]); y = df[target].values
    rep = leaklint.detect_leakage(X, y, feature_names=list(X.columns),
                                  use_mutual_info=True, leaky_auc_threshold=0.80)
    res = _blank()
    for f in (getattr(rep, "findings", None) or []):
        d = f.to_dict() if hasattr(f, "to_dict") else (f if isinstance(f, dict) else {})
        cols = d.get("columns") or d.get("features") or \
               ([d.get("column")] if d.get("column") else []) or \
               ([d.get("feature")] if d.get("feature") else [])
        lab = d.get("severity") or d.get("kind") or d.get("code") or "flagged"
        for c in cols:
            if c in res:
                res[c] = {"flagged": True, "label": str(lab)}
    return res

# --- dataleaks: 11 detectors, none missingness-based (documents the gap) ---
def run_dataleaks(df, target="target"):
    import dataleaks
    rep = dataleaks.DataLeaks(df, target=target).run()
    res = _blank()
    for it in (getattr(rep, "findings", None) or []):
        d = it if isinstance(it, dict) else getattr(it, "__dict__", {})
        cols = d.get("columns") or d.get("features") or \
               ([d.get("column")] if d.get("column") else []) or \
               ([d.get("feature")] if d.get("feature") else [])
        lab = d.get("severity") or d.get("type") or d.get("detector") or "flagged"
        for c in (cols or []):
            if c in res:
                res[c] = {"flagged": True, "label": str(lab)}
    return res

# --- cleanlab: no per-feature missingness-leakage verdict (structural gap) ---
def run_cleanlab(df, target="target"):
    raise RuntimeError("cleanlab Datalab offers no per-feature missingness-leakage verdict")

BASELINES = {
    "assoc_threshold": run_assoc_threshold,
    "featureleak":     run_featureleak,
    "leaklint":        run_leaklint,
    "dataleaks":       run_dataleaks,
    "cleanlab":        run_cleanlab,
}

def run_all(df, target="target"):
    out = {}
    for name, fn in BASELINES.items():
        try:
            out[name] = {"status": "ok", "verdicts": fn(df, target)}
        except Exception as e:
            out[name] = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:140]}"}
    return out
