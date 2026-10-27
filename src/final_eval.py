"""
final_eval.py — Honest end-to-end evaluation for the project.

Fixes the one methodological hole: the S-detection threshold (env_thr) is TUNED
on a dev split and REPORTED on a held-out test split, so it is never fit on the
data it is scored on. Runs on synthetic + any real dataset, writes a single
results table and the data for figures.

Usage:
  python -m src.final_eval                                  # synthetic only
  python -m src.final_eval --real adult --path data/raw/adult.data
  python -m src.final_eval --real mimic --path data/raw/mimic.csv
"""
from __future__ import annotations
import argparse, itertools, os
import numpy as np, pandas as pd

from .benchmark import build_synthetic, inject, BASES
from .separator import signals_for_feature, rule_based_type, MISS_FEATS

GT = {"miss_L": "L", "miss_K": "K", "miss_S": "S", "miss_N": "N"}
DIFFS = [0.2, 0.3, 0.4, 0.5]
TYPES = ["L", "K", "S", "N"]
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def make_instances(args):
    insts = []
    if args.real:
        from .loaders import REAL_LOADERS
        loader = REAL_LOADERS[args.real]
        Xb, yb, pb = (loader(args.path) if args.path else loader())
        if args.n and len(Xb) > args.n:
            idx = np.random.RandomState(0).choice(len(Xb), args.n, replace=False)
            Xb = Xb.iloc[idx].reset_index(drop=True); yb = yb[idx]; pb = pb[idx]
        for diff, seed in itertools.product(DIFFS, range(1, args.seeds + 1)):
            df, man = inject(Xb, yb, pb, target_assoc=diff, seed=seed)
            insts.append((df, {k: v["type"] for k, v in man.items()}, seed))
    else:
        for domain, diff, seed in itertools.product(BASES, DIFFS, range(1, args.seeds + 1)):
            df, man = build_synthetic(domain, args.n, diff, seed)
            insts.append((df, {k: v["type"] for k, v in man.items()}, seed))
    return insts


def signals_table(insts):
    """Precompute signals once per feature per instance (the slow part)."""
    rows = []
    for i, (df, gt, seed) in enumerate(insts):
        for col in MISS_FEATS:
            if col not in df.columns:
                continue
            s = signals_for_feature(df, col, seed=i)
            s["gt"] = gt[col]; s["seed"] = seed; s["inst"] = i
            rows.append(s)
    return pd.DataFrame(rows)


def balanced_acc(true, pred):
    classes = sorted(set(true))
    return float(np.mean([
        np.mean([pred[i] == c for i, t in enumerate(true) if t == c])
        for c in classes
    ]))


def predict_from_signals(tbl, env_thr):
    preds = []
    for _, s in tbl.iterrows():
        preds.append(rule_based_type(s.to_dict(), env_thr=env_thr))
    return preds


def confusion(true, pred, labels=TYPES):
    M = pd.DataFrame(0, index=labels, columns=labels)
    for t, p in zip(true, pred):
        M.loc[t, p] += 1
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", choices=["adult", "mimic", "gmc"], default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    tag = args.real or "synthetic"

    insts = make_instances(args)
    print(f"[{tag}] {len(insts)} instances, computing signals...")
    tbl = signals_table(insts)

    # ---- honest split: dev seeds vs test seeds (odd=dev, even=test) ----
    dev = tbl[tbl["seed"] % 2 == 1].copy()
    test = tbl[tbl["seed"] % 2 == 0].copy()

    # tune env_thr on DEV only
    grid = np.round(np.arange(0.05, 0.60, 0.02), 3)
    best_thr, best_dev = 0.25, -1
    for thr in grid:
        ba = balanced_acc(list(dev["gt"]), predict_from_signals(dev, thr))
        if ba > best_dev:
            best_dev, best_thr = ba, float(thr)

    # report on TEST with the dev-tuned threshold
    test_pred = predict_from_signals(test, best_thr)
    ba_test = balanced_acc(list(test["gt"]), test_pred)

    # baseline ceiling: flag-only -> N right, L/K/S collapse to one class
    base_pred = ["K" if t in ("L", "K", "S") else "N" for t in test["gt"]]
    ba_base = balanced_acc(list(test["gt"]), base_pred)

    print(f"\n=== {tag.upper()} — honest held-out result ===")
    print(f"env_thr tuned on dev = {best_thr}  (dev balanced acc {best_dev:.3f})")
    print(f"TEST balanced typing accuracy : separator {ba_test:.3f}  |  baseline {ba_base:.3f}")
    print("\nTEST confusion (rows=true, cols=pred):")
    print(confusion(list(test["gt"]), test_pred).to_string())

    # per-class recall on test
    for c in TYPES:
        idx = [i for i, t in enumerate(test["gt"]) if t == c]
        if idx:
            r = np.mean([test_pred[i] == c for i in idx])
            print(f"  recall[{c}] = {r:.2f}")

    # save
    out = pd.DataFrame([dict(dataset=tag, env_thr=best_thr,
                             separator_acc=round(ba_test, 3),
                             baseline_acc=round(ba_base, 3))])
    path = os.path.join(RESULTS, f"final_{tag}.csv")
    out.to_csv(path, index=False)
    confusion(list(test["gt"]), test_pred).to_csv(
        os.path.join(RESULTS, f"final_confusion_{tag}.csv"))
    print(f"\nsaved {path} and final_confusion_{tag}.csv")


if __name__ == "__main__":
    main()
