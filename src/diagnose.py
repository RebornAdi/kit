"""
diagnose.py — Print the separator's raw signal values on a real dataset, so we
can see WHY typing fails. Run:

  python -m src.diagnose --real adult --path data/raw/adult.data
  python -m src.diagnose --real mimic --path data/raw/mimic.csv
"""
import argparse
import numpy as np
from .loaders import REAL_LOADERS
from .benchmark import inject
from .separator import signals_for_feature, MISS_FEATS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, choices=["adult","mimic","gmc"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--assoc", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()

    loader = REAL_LOADERS[a.real]
    X, y, p = (loader(a.path) if a.path else loader())
    if a.n and len(X) > a.n:
        idx = np.random.RandomState(0).choice(len(X), a.n, replace=False)
        X = X.iloc[idx].reset_index(drop=True); y = y[idx]; p = p[idx]

    print(f"dataset: {a.real}  n={len(X)}  pos_rate={y.mean():.3f}  "
          f"p-y corr={np.corrcoef(p,y)[0,1]:.3f}  n_features={X.shape[1]}")
    print(f"base numeric features: {list(X.select_dtypes(include=[np.number]).columns)}")

    df, man = inject(X, y, p, target_assoc=a.assoc, seed=1)
    print(f"\n{'feat':8s}{'gt':4s}{'f1_info':9s}{'f2_boot':9s}{'f3_env':9s}{'f5_gap':9s}")
    for col in MISS_FEATS:
        s = signals_for_feature(df, col, seed=1)
        print(f"{col:8s}{man[col]['type']:4s}"
              f"{s['f1_info']:<9.3f}{s['f2_boot']:<9.3f}{s['f3_env']:<9.3f}{s['f5_gap']:<9.3f}")
    print("\nIf f3_env is HIGH for L/K (not just S), the env-search is finding "
          "spurious splits -> that's the bug to fix.")

if __name__ == "__main__":
    main()