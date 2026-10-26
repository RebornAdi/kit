"""
benchmark.py — Base datasets + typed missingness injection.

Injects FOUR features of KNOWN type into any base (X, y):
  L  legitimate          presence driven by an available risk tier (keep)
  K  leakage-availability presence driven by the outcome itself      (remove)
  S  leakage-stability    presence driven by a train-only env signal  (remove)
  N  neutral              presence independent of y (MCAR)            (keep)

All of L/K/S are tuned to the SAME mask-target association, so a detector that
reads only association magnitude cannot separate them. That is the whole point.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- base synth ---
def make_clinical(n, seed):
    rng = np.random.default_rng(seed)
    age = rng.normal(55, 15, n).clip(18, 95)
    bmi = rng.normal(28, 6, n).clip(15, 55)
    sbp = rng.normal(130, 18, n).clip(85, 220)
    glucose = rng.normal(105, 30, n).clip(60, 300)
    logit = (-1.1 + 0.045*(age-55) + 0.06*(bmi-28)
             + 0.02*(glucose-105) + 0.015*(sbp-130))
    p = 1/(1+np.exp(-logit))
    y = rng.binomial(1, p)
    X = pd.DataFrame({"age": age.round(1), "bmi": bmi.round(1),
                      "systolic_bp": sbp.round(0), "glucose": glucose.round(0)})
    return X, y, p

def make_credit(n, seed):
    rng = np.random.default_rng(seed)
    util = rng.beta(2, 5, n)
    age = rng.normal(45, 13, n).clip(21, 90)
    debt = rng.lognormal(-1.0, 0.8, n).clip(0, 5)
    income = rng.lognormal(8.6, 0.6, n)
    lines = rng.poisson(8, n)
    logit = (-1.4 + 2.2*(util-0.3) + 0.9*(debt-0.5)
             - 0.000018*(income-5000) - 0.02*(age-45))
    p = 1/(1+np.exp(-logit))
    y = rng.binomial(1, p)
    X = pd.DataFrame({"util": util.round(3), "age": age.round(0),
                      "debt_ratio": debt.round(3), "income": income.round(0),
                      "open_lines": lines})
    return X, y, p

BASES = {"clinical": make_clinical, "credit": make_credit}

# ------------------------------------------------------------ injection core ---
def _assoc(mask_bool, y):
    m = np.asarray(mask_bool, dtype=int)
    y = np.asarray(y)
    if m.std() == 0 or y.std() == 0:
        return 0.0
    return abs(np.corrcoef(m, y)[0, 1])

def _tune_presence(driver_high, y, target, rng, grid=60):
    """Find (p_hi, p_lo) so |corr(missing_indicator, y)| ~= target."""
    best = None
    for gap in np.linspace(0.02, 0.48, grid):
        present = np.where(driver_high == 1,
                           rng.binomial(1, 0.5+gap, len(y)),
                           rng.binomial(1, 0.5-gap, len(y)))
        a = _assoc(1-present, y)                      # missing = 1 - present
        if best is None or abs(a-target) < abs(best[0]-target):
            best = (a, present.copy())
    return best

def inject(X, y, p, target_assoc=0.3, seed=0, noise_frac=0.18):
    """
    Add miss_L / miss_K / miss_S / miss_N to a copy of X. Returns (df, manifest).
    `p` is the per-row target probability (used to build a legitimate risk tier);
    for a real dataset without p, pass p = a model's predicted proba, or p=y.
    """
    rng = np.random.default_rng(seed + 10_000)
    y = np.asarray(y).astype(int)
    df = X.reset_index(drop=True).copy()
    n = len(df)

    # L: risk tier that tracks y with controlled disagreement (stays legitimate)
    flip_l = rng.binomial(1, noise_frac, n)
    driver_L = np.where(flip_l == 1, 1-y, y)
    aL, pres_L = _tune_presence(driver_L, y, target_assoc, rng)
    df["miss_L"] = np.where(pres_L == 1, rng.normal(0, 1, n).round(3), np.nan)

    # K: POST-OUTCOME field. Present (recorded) almost only for resolved/positive
    # cases, i.e. its PRESENCE aligns strongly with the outcome -> a value that
    # would not exist at prediction time. This gives K a detectable post-outcome
    # structure (large class gap on presence) while matching |corr(mask,y)|.
    aK, pres_K = _tune_presence(y, y, target_assoc, rng)
    df["miss_K"] = np.where(pres_K == 1, rng.integers(1, 9, n).astype(float), np.nan)

    # S: STABILITY leak. Missingness driven by a site/environment latent that is
    # entangled with y ONLY in part of the data. Across environment splits the
    # mask-target relationship diverges sharply (high env-sensitivity) even though
    # the pooled |corr(mask,y)| matches the others -> genuinely unstable.
    # In env A the mask tracks y; in env B it is (near) random. Pooled stays
    # informative and matched, but f1 measured within each env differs sharply
    # (strong in A, ~0 in B) -> high env-sensitivity, the signature of S.
    env_axis = rng.binomial(1, 0.5, n)
    driver_S = np.where(env_axis == 1, y, rng.binomial(1, 0.5, n))
    aS, pres_S = _tune_presence(driver_S, y, target_assoc, rng)
    df["miss_S"] = np.where(pres_S == 1, rng.normal(5, 2, n).round(2), np.nan)
    # Record the environment axis as an OBSERVED column (like a site id / year).
    # Realistic: deployment-stability is always assessed relative to a known
    # environment variable. The separator may use env_* columns as split candidates.
    df["env_site"] = env_axis.astype(int)

    # N: MCAR
    pres_N = rng.binomial(1, 0.5, n)
    df["miss_N"] = np.where(pres_N == 1, rng.normal(0, 1, n).round(3), np.nan)

    df["target"] = y
    manifest = {
        "miss_L": {"type": "L", "action": "keep",   "assoc": round(aL, 3),
                   "scenario": "deployed in the same workflow that gates the feature"},
        "miss_K": {"type": "K", "action": "remove", "assoc": round(aK, 3),
                   "scenario": "field populated only after the outcome is known"},
        "miss_S": {"type": "S", "action": "remove", "assoc": round(aS, 3),
                   "scenario": "missingness pattern will not transfer across sites/time"},
        "miss_N": {"type": "N", "action": "keep",
                   "assoc": round(_assoc(np.isnan(df['miss_N']), y), 3),
                   "scenario": "missingness carries no target information"},
    }
    return df, manifest

def build_synthetic(base, n=12000, target_assoc=0.3, seed=0):
    X, y, p = BASES[base](n, seed)
    return inject(X, y, p, target_assoc, seed)

if __name__ == "__main__":
    for b in BASES:
        df, man = build_synthetic(b, 12000, 0.3, 1)
        a = {k: v["assoc"] for k, v in man.items()}
        print(f"{b:9s} L={a['miss_L']:.3f} K={a['miss_K']:.3f} "
              f"S={a['miss_S']:.3f} N={a['miss_N']:.3f}  shape={df.shape}")