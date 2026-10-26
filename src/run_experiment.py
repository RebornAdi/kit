"""
run_experiment.py — Full sweep + scoring.

Synthetic mode (default): sweeps 2 domains x 4 difficulties x N seeds.
Real mode: pass --real gmc --path data/raw/cs-training.csv to inject typed
features on top of a real base and run the same sweep over difficulties/seeds.

Outputs to results/: raw_results.csv, summary.csv, by_difficulty.csv, flag_by_type.csv

Usage:
  python -m src.run_experiment                              # synthetic
  python -m src.run_experiment --seeds 10 --n 12000
  python -m src.run_experiment --real gmc --path data/raw/cs-training.csv
"""
from __future__ import annotations
import argparse, itertools, os
import numpy as np
import pandas as pd

from .benchmark import build_synthetic, BASES, inject
from .baselines import run_all, FEATS

GT_TYPE   = {"miss_L": "L", "miss_K": "K", "miss_S": "S", "miss_N": "N"}
GT_ACTION = {"miss_L": "keep", "miss_K": "remove", "miss_S": "remove", "miss_N": "keep"}
INFORMATIVE = ["miss_L", "miss_K", "miss_S"]
DIFFS = [0.2, 0.3, 0.4, 0.5]

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def instances(args):
    """Yield (domain_label, diff, seed, df) for each benchmark instance."""
    if args.real:
        from .loaders import REAL_LOADERS
        loader = REAL_LOADERS[args.real]
        # load once; re-inject per (diff, seed) so native base is constant.
        # adult can load from OpenML when no --path is given; others need --path.
        if args.path:
            Xb, yb, pb = loader(args.path)
        elif args.real == "adult":
            Xb, yb, pb = loader()            # OpenML fallback
        else:
            raise SystemExit(f"--path required for --real {args.real}")
        if args.n and len(Xb) > args.n:
            idx = np.random.RandomState(0).choice(len(Xb), args.n, replace=False)
            Xb = Xb.iloc[idx].reset_index(drop=True); yb = yb[idx]; pb = pb[idx]
        for diff, seed in itertools.product(DIFFS, range(1, args.seeds+1)):
            df, _ = inject(Xb, yb, pb, target_assoc=diff, seed=seed)
            yield args.real, diff, seed, df
    else:
        for domain, diff, seed in itertools.product(BASES, DIFFS, range(1, args.seeds+1)):
            df, _ = build_synthetic(domain, args.n, diff, seed)
            yield domain, diff, seed, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", choices=["adult", "mimic", "gmc", "mimic_demo"], default=None)
    ap.add_argument("--path", default=None, help="CSV path for --real")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=12000)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    rows, done = [], 0
    insts = list(instances(args))
    for domain, diff, seed, df in insts:
        for tool, r in run_all(df).items():
            if r["status"] != "ok":
                for f in FEATS:
                    rows.append(dict(domain=domain, diff=diff, seed=seed, tool=tool,
                                     feature=f, gt_type=GT_TYPE[f], gt_action=GT_ACTION[f],
                                     flagged=None, label=None, status="error"))
            else:
                for f in FEATS:
                    v = r["verdicts"][f]
                    rows.append(dict(domain=domain, diff=diff, seed=seed, tool=tool,
                                     feature=f, gt_type=GT_TYPE[f], gt_action=GT_ACTION[f],
                                     flagged=bool(v["flagged"]), label=v["label"], status="ok"))
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{len(insts)} instances", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RESULTS, "raw_results.csv"), index=False)
    print("saved raw_results.csv", R.shape)
    score(R)


def score(R):
    ok = R[R.status == "ok"].copy()
    summary = []
    for tool in R.tool.unique():
        t = ok[ok.tool == tool]
        if t.empty:
            summary.append(dict(tool=tool, status="unavailable",
                                detect_recall=np.nan, detect_fpr=np.nan,
                                type_separation=np.nan,
                                note=R[R.tool == tool].iloc[0].get("label") or "failed"))
            continue
        inf = t[t.feature.isin(INFORMATIVE)]
        neu = t[t.feature == "miss_N"]
        detect_recall = inf.flagged.mean()
        detect_fpr = neu.flagged.mean()

        sep_hits = sep_total = 0
        for _, g in t.groupby(["domain", "diff", "seed"]):
            gL, gK, gS = (g[g.feature == c] for c in ("miss_L", "miss_K", "miss_S"))
            if gL.empty or gK.empty or gS.empty:
                continue
            vL = (gL.iloc[0].flagged, gL.iloc[0].label)
            vK = (gK.iloc[0].flagged, gK.iloc[0].label)
            vS = (gS.iloc[0].flagged, gS.iloc[0].label)
            if not (vL[0] or vK[0] or vS[0]):
                sep_total += 1; continue
            sep_hits += int((vL != vK) and (vL != vS)); sep_total += 1
        type_sep = sep_hits/sep_total if sep_total else np.nan

        summary.append(dict(tool=tool, status="ok",
                            detect_recall=round(float(detect_recall), 3),
                            detect_fpr=round(float(detect_fpr), 3),
                            type_separation=round(float(type_sep), 3), note=""))

    S = pd.DataFrame(summary); S.to_csv(os.path.join(RESULTS, "summary.csv"), index=False)
    print("\n=== SUMMARY ==="); print(S.to_string(index=False))

    byd = (ok[ok.feature.isin(INFORMATIVE)].groupby(["tool", "diff"])
           .flagged.mean().reset_index().rename(columns={"flagged": "detect_recall"}))
    byd.to_csv(os.path.join(RESULTS, "by_difficulty.csv"), index=False)

    conf = (ok[ok.tool.isin(["featureleak", "assoc_threshold"])]
            .groupby(["tool", "gt_type", "flagged"]).size().reset_index(name="count"))
    conf.to_csv(os.path.join(RESULTS, "flag_by_type.csv"), index=False)
    print("saved summary.csv, by_difficulty.csv, flag_by_type.csv")


if __name__ == "__main__":
    main()
