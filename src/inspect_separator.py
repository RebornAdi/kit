
from src.benchmark import build_synthetic
from src.separator import MISS_FEATS, signals_for_feature, rule_based_type

def main():
    df, manifest = build_synthetic(
        base="clinical",
        n=12000,
        target_assoc=0.3,
        seed=42
    )

    print("\nMISSINGNESS SIGNAL DIAGNOSTICS")
    print("-" * 90)
    print(
        f"{'Feature':<12}"
        f"{'True':<8}"
        f"{'Info':<10}"
        f"{'Bootstrap':<12}"
        f"{'Environment':<13}"
        f"{'Post-outcome':<14}"
        f"{'Gap':<10}"
        f"{'Prediction':<12}"
    )

    for col in MISS_FEATS:
        signals = signals_for_feature(
            df,
            col,
            target="target",
            B=40,
            seed=42
        )

        prediction = rule_based_type(signals)

        print(
            f"{col:<12}"
            f"{manifest[col]['type']:<8}"
            f"{signals['f1_info']:<10.3f}"
            f"{signals['f2_boot']:<12.3f}"
            f"{signals['f3_env']:<13.3f}"
            f"{signals['f4_post']:<14.1f}"
            f"{signals['f5_gap']:<10.3f}"
            f"{prediction:<12}"
        )

    print("\nInterpretation:")
    print("- Info: strength of the missingness-target association.")
    print("- Bootstrap: variability in that association.")
    print("- Environment: instability across observed feature splits.")
    print("- Post-outcome: current heuristic for detecting outcome leakage.")
    print("- Gap: difference in target rates for missing vs observed values.")

if __name__ == "__main__":
    main()
