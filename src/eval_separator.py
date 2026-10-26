"""
eval_separator.py — Evaluate the separator's TYPING (L/K/S/N) against the
baselines, on synthetic + real datasets. Reports balanced accuracy, per-class
recall, and a confusion matrix. Also reports the baseline-equivalent typing
(baselines emit only flagged/label, mapped to a predicted type the fair way:
flagged informative features all map to a single "suspicious" verdict, so their
best possible typing is N-vs-rest).

Usage:
  python -m src.eval_separator                       # synthetic
  python -m src.eval_separator --real adult --path data/raw/adult.data
  python -m src.eval_separator --real mimic --path data/raw/mimic.csv
"""
from __future__ import annotations
import argparse, itertools, os
import numpy as np, pandas as pd
from collections import Counter

from .benchmark import build_synthetic, inject, BASES
from .separator import (signals_for_feature, rule_based_type, MISS_FEATS,
                        build_training_table, train_supervised, FEATURE_COLS)

GT = {"miss_L": "L", "miss_K": "K", "miss_S": "S", "miss_N": "N"}
DIFFS = [0.2, 0.3, 0.4, 0.5]
TYPES = ["L", "K", "S", "N"]
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def make_instances(args):
    if args.real:
        from .loaders import REAL_LOADERS
        loader = REAL_LOADERS[args.real]
        Xb, yb, pb = (loader(args.path) if args.path else loader())
        if args.n and len(Xb) > args.n:
            idx = np.random.RandomState(0).choice(len(Xb), args.n, replace=False)
            Xb = Xb.iloc[idx].reset_index(drop=True); yb = yb[idx]; pb = pb[idx]
        for diff, seed in itertools.product(DIFFS, range(1, args.seeds + 1)):
            df, man = inject(Xb, yb, pb, target_assoc=diff, seed=seed)
            yield df, {k: v["type"] for k, v in man.items()}
    else:
        for domain, diff, seed in itertools.product(BASES, DIFFS, range(1, args.seeds + 1)):
            df, man = build_synthetic(domain, args.n, diff, seed)
            yield df, {k: v["type"] for k, v in man.items()}


def balanced_acc(true, pred):
    classes = sorted(set(true))
    recalls = []
    for c in classes:
        idx = [i for i, t in enumerate(true) if t == c]
        if idx:
            recalls.append(np.mean([pred[i] == c for i in idx]))
    return float(np.mean(recalls)) if recalls else 0.0


def confusion(true, pred, labels=TYPES):
    M = pd.DataFrame(0, index=labels, columns=labels)
    for t, p in zip(true, pred):
        if t in labels and p in labels:
            M.loc[t, p] += 1
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", choices=["adult", "mimic", "gmc"], default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--supervised", action="store_true",
                    help="also train+eval the supervised separator (5-fold by instance)")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    insts = list(make_instances(args))
    print(f"{len(insts)} instances")

    # ---- rule-based separator ----
    true, pred_rule = [], []
    for i, (df, gt) in enumerate(insts):
        for col in MISS_FEATS:
            if col not in df.columns:
                continue
            s = signals_for_feature(df, col, seed=i)
            true.append(gt[col]); pred_rule.append(rule_based_type(s))

    ba_rule = balanced_acc(true, pred_rule)
    print(f"\n=== RULE-BASED SEPARATOR ===")
    print(f"balanced typing accuracy: {ba_rule:.3f}")
    print("confusion (rows=true, cols=pred):")
    print(confusion(true, pred_rule).to_string())

    # ---- baseline best-possible typing ----
    # baselines emit only suspicious/not. Their best achievable typing is to call
    # everything flagged "leak" -> they get N right and collapse L/K/S. We compute
    # that ceiling so the comparison is fair.
    pred_base = ["K" if t in ("L", "K", "S") else "N" for t in true]  # all->one leak class
    ba_base = balanced_acc(true, pred_base)
    print(f"\n=== BASELINE CEILING (flag-only, cannot type) ===")
    print(f"balanced typing accuracy: {ba_base:.3f}")

    out = [dict(method="baseline_ceiling", balanced_acc=round(ba_base, 3)),
           dict(method="separator_rule", balanced_acc=round(ba_rule, 3))]

    # ---- supervised separator (optional) ----
    if args.supervised:
        from sklearn.model_selection import GroupKFold
        tbl = build_training_table(insts)
        gkf = GroupKFold(n_splits=5)
        preds = np.empty(len(tbl), dtype=object)
        for tr, te in gkf.split(tbl, tbl["label"], groups=tbl["inst"]):
            clf = train_supervised(tbl.iloc[tr])
            preds[te] = clf.predict(tbl.iloc[te][FEATURE_COLS])
        ba_sup = balanced_acc(list(tbl["label"]), list(preds))
        print(f"\n=== SUPERVISED SEPARATOR (5-fold by instance) ===")
        print(f"balanced typing accuracy: {ba_sup:.3f}")
        print(confusion(list(tbl["label"]), list(preds)).to_string())
        out.append(dict(method="separator_supervised", balanced_acc=round(ba_sup, 3)))

    pd.DataFrame(out).to_csv(os.path.join(RESULTS, "separator_summary.csv"), index=False)
    confusion(true, pred_rule).to_csv(os.path.join(RESULTS, "separator_confusion.csv"))
    print("\nsaved separator_summary.csv, separator_confusion.csv")


if __name__ == "__main__":
    main()