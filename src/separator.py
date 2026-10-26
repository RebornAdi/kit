"""
separator.py — Our method: classify a missingness feature's TYPE (L/K/S/N),
not just flag it. Baselines read in-sample association magnitude and therefore
cannot separate legitimate (L) from leakage (K/S). The separator brings in
information the raw sample lacks:

  f1  informativeness      I(Y; M_j) via point-biserial |corr|      (detection)
  f2  bootstrap instability std of f1 across bootstrap resamples    -> S
  f3  env-split sensitivity max |f1_env_a - f1_env_b| over proxy envs -> S
  f4  post-outcome flag     name heuristic + "present-only-with-resolved" check -> K
  f5  class-gap             |P(Y=1|missing) - P(Y=1|observed)|       (supporting)

Then map (f1..f5) -> predicted type, two ways:
  - rule-based  : transparent thresholds (no training)
  - supervised  : small DecisionTree trained on benchmark ground-truth labels

Why this can work where in-sample magnitude cannot:
  L is driven by a STABLE mechanism  -> low f2, low f3
  S is driven by a FRAGILE entanglement -> high f2 / high f3
  K is a post-outcome field -> f4 fires (metadata), otherwise looks like L/S
  N is not informative -> low f1
"""
from __future__ import annotations
import numpy as np
import pandas as pd

MISS_FEATS = ["miss_L", "miss_K", "miss_S", "miss_N"]

# ---------------------------------------------------------------- signals ---
def _assoc(mask_int, y):
    if mask_int.std() == 0 or np.std(y) == 0:
        return 0.0
    return abs(np.corrcoef(mask_int, y)[0, 1])

def _informativeness(mask_int, y):
    return _assoc(mask_int, y)

def _bootstrap_instability(mask_int, y, B=40, frac=0.7, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(y); m = int(frac*n)
    vals = []
    for _ in range(B):
        idx = rng.choice(n, m, replace=True)
        vals.append(_assoc(mask_int[idx], y[idx]))
    return float(np.std(vals))

def _split_divergence(mask_int, y, hi):
    if hi.sum() < 30 or (~hi).sum() < 30:
        return None
    return abs(_assoc(mask_int[hi], y[hi]) - _assoc(mask_int[~hi], y[~hi]))

def _env_sensitivity(mask_int, y, X, rng=None, n_random=60):
    """
    NULL-CALIBRATED environment sensitivity.

    The separator searches candidate environment splits (each numeric base feature
    at its median) and also builds a NULL distribution from many RANDOM splits.
    The signal returned is how much the best *structured* (feature-based) split
    EXCEEDS the random-split null for THIS feature:

        f3 = max_feature_divergence  -  (mean_random + 2*std_random)

    Why this fixes real data: on datasets with many target-correlated features, a
    legitimate (L) feature also shows some divergence when split on those features
    -- but no more than its own random splits produce. A true stability-leak (S)
    diverges along a real axis FAR beyond its random null. Subtracting the null
    removes the base-feature inflation that made L/K look unstable.
    """
    rng = rng or np.random.default_rng(0)
    n = len(y)
    num = X.select_dtypes(include=[np.number])

    # structured splits: base features (env_* first) at the median
    env_cols = [c for c in num.columns if c.lower().startswith("env")]
    other = [c for c in num.columns if c not in env_cols]
    feat_divs = []
    for c in env_cols + other:
        v = num[c].values.astype(float)
        v = np.nan_to_num(v, nan=np.nanmedian(v) if np.isfinite(np.nanmedian(v)) else 0.0)
        if np.std(v) == 0:
            continue
        d = _split_divergence(mask_int, y, v >= np.median(v))
        if d is not None:
            feat_divs.append(d)

    # null distribution from random splits
    null = []
    for _ in range(n_random):
        d = _split_divergence(mask_int, y, rng.binomial(1, 0.5, n).astype(bool))
        if d is not None:
            null.append(d)

    if not feat_divs or not null:
        return 0.0
    best_feat = max(feat_divs)
    null_hi = float(np.mean(null) + 2.5 * np.std(null))   # upper band of the null
    return float(max(0.0, best_feat - null_hi))

POST_OUTCOME_TOKENS = ("discharge", "death", "outcome", "final", "result",
                       "disposition", "dx_code", "reason", "resolved", "_post")

def _post_outcome_flag(col_name, mask_int, y):
    """Name heuristic + a weak data check: does the feature's PRESENCE align
    almost perfectly with one class (as a post-outcome field would)?"""
    name_hit = any(tok in col_name.lower() for tok in POST_OUTCOME_TOKENS)
    present = 1 - mask_int
    if present.sum() == 0 or present.sum() == len(present):
        data_hit = False
    else:
        rate_present = y[present == 1].mean()
        rate_absent = y[present == 0].mean()
        data_hit = abs(rate_present - rate_absent) > 0.6   # extreme class split
    return float(name_hit or data_hit), float(name_hit)

def _class_gap(mask_int, y):
    miss = mask_int == 1
    if miss.sum() == 0 or (~miss).sum() == 0:
        return 0.0
    return float(abs(y[miss].mean() - y[~miss].mean()))


def signals_for_feature(df, col, target="target", B=40, seed=0):
    rng = np.random.default_rng(seed)
    y = df[target].values.astype(int)
    mask = df[col].isna().astype(int).values
    X = df.drop(columns=[c for c in MISS_FEATS if c in df.columns] + [target],
                errors="ignore")
    f1 = _informativeness(mask, y)
    f2 = _bootstrap_instability(mask, y, B=B, rng=rng)
    f3 = _env_sensitivity(mask, y, X, rng=rng)
    f4, f4_name = _post_outcome_flag(col, mask, y)
    f5 = _class_gap(mask, y)
    return dict(f1_info=f1, f2_boot=f2, f3_env=f3, f4_post=f4,
                f4_name=f4_name, f5_gap=f5)


# ---------------------------------------------------------- rule-based map ---
def rule_based_type(s, info_thr=0.08, env_thr=0.25, gap_thr=0.20):
    """
    Transparent mapping (no training). Order matters:
      not informative                         -> N
      informative + env-unstable              -> S   (diverges across environments)
      informative + name/extreme post-outcome -> K
      informative + stable + large class gap  -> K   (presence ~ a function of y)
      informative + stable + moderate gap     -> L
    Thresholds are defaults; tune on a dev split (or use the supervised version).
    """
    if s["f1_info"] < info_thr:
        return "N"
    if s["f3_env"] >= env_thr:
        return "S"
    # K requires deployment/timing metadata to separate from L (post-outcome name
    # or an extreme presence-outcome alignment). Without it, K is indistinguishable
    # from L in-sample -- the paper's central claim. Default to L when no K evidence.
    if s["f4_name"] >= 1.0 or s["f4_post"] >= 1.0:
        return "K"
    return "L"


# ---------------------------------------------------------- supervised map ---
FEATURE_COLS = ["f1_info", "f2_boot", "f3_env", "f4_post", "f5_gap"]

def build_training_table(instances, target="target", B=40):
    """
    instances: iterable of (df, gt_type_dict) where gt_type_dict maps
               miss_L/K/S/N -> 'L'/'K'/'S'/'N'.
    Returns a DataFrame of signal features + label, one row per missingness feature.
    """
    rows = []
    for inst_id, (df, gt) in enumerate(instances):
        for col in MISS_FEATS:
            if col not in df.columns:
                continue
            s = signals_for_feature(df, col, target=target, B=B, seed=inst_id)
            s["label"] = gt[col]
            s["inst"] = inst_id
            rows.append(s)
    return pd.DataFrame(rows)

def train_supervised(train_df, max_depth=4, seed=0):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed,
                                 class_weight="balanced")
    clf.fit(train_df[FEATURE_COLS], train_df["label"])
    return clf

def predict_supervised(clf, df, target="target", B=40, seed=0):
    out = {}
    for col in MISS_FEATS:
        if col not in df.columns:
            continue
        s = signals_for_feature(df, col, target=target, B=B, seed=seed)
        out[col] = clf.predict(pd.DataFrame([s])[FEATURE_COLS])[0]
    return out

def predict_rule(df, target="target", B=40, seed=0, **thr):
    out = {}
    for col in MISS_FEATS:
        if col not in df.columns:
            continue
        s = signals_for_feature(df, col, target=target, B=B, seed=seed)
        out[col] = rule_based_type(s, **thr)
    return out


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from benchmark import build_synthetic
    df, man = build_synthetic("clinical", 12000, 0.3, 1)
    print("ground truth:", {k: v["type"] for k, v in man.items()})
    print("\nsignals + rule prediction:")
    for col in MISS_FEATS:
        s = signals_for_feature(df, col, seed=1)
        print(f"  {col} gt={man[col]['type']}  "
              f"f1={s['f1_info']:.3f} f2={s['f2_boot']:.3f} f3={s['f3_env']:.3f} "
              f"f4={s['f4_post']:.0f} -> {rule_based_type(s)}")