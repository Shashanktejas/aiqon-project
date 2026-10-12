from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "model_dataset_v2.csv"
MODEL = ROOT / "models" / "LogisticRegression_v2.joblib"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2020-10-01")

FEATURES = [
    "MonthSin",
    "MonthCos",
    "Year",
    "MeanDwellDays_roll3",
    "MeanDwellDays_roll6",
    "MedianDwellDays_roll3",
    "MedianDwellDays_roll6",
    "UniqueVessels_roll3",
    "UniqueVessels_roll6",
    "DwellChange1",
    "DwellChange3",
    "Port",
]


def main():
    df = pd.read_csv(DATA)
    df["YearMonth"] = pd.to_datetime(df["YearMonth"])

    df = df.dropna(
        subset=["TargetHighDwellNextMonth"]
    ).copy()

    df["TargetHighDwellNextMonth"] = (
        df["TargetHighDwellNextMonth"].astype(int)
    )

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    test = df[df["YearMonth"] >= CUTOFF].copy()

    if test.empty:
        raise ValueError("No test rows found after the cutoff.")

    model = joblib.load(MODEL)

    X_test = test[FEATURES]
    y_test = test["TargetHighDwellNextMonth"].to_numpy()
    prob = model.predict_proba(X_test)[:, 1]

    brier = brier_score_loss(y_test, prob)
    loss = log_loss(y_test, prob)
    auc = roc_auc_score(y_test, prob)

    print("=== V2 PROBABILITY EVALUATION ===")
    print(f"Test rows: {len(test)}")
    print(f"Test period starts: {CUTOFF:%Y-%m}")
    print(f"Test period ends: {test['YearMonth'].max():%Y-%m}")
    print(f"Actual high-dwell rate: {y_test.mean():.3f}")
    print(f"Mean predicted probability: {prob.mean():.3f}")
    print(f"Brier score: {brier:.4f} (lower is better)")
    print(f"Log loss: {loss:.4f} (lower is better)")
    print(f"ROC-AUC: {auc:.4f}")

    fraction_positive, mean_predicted = calibration_curve(
        y_test,
        prob,
        n_bins=8,
        strategy="quantile",
    )

    print("\n=== CALIBRATION BINS ===")
    for predicted, actual in zip(mean_predicted, fraction_positive):
        print(
            f"Mean predicted={predicted:.3f} | "
            f"Observed positive rate={actual:.3f}"
        )

    results = pd.DataFrame({
        "YearMonth": test["YearMonth"].to_numpy(),
        "Port": test["Port"].to_numpy(),
        "ActualHighDwell": y_test,
        "PredictedProbability": prob,
    })

    results.to_csv(
        OUT / "probability_evaluation_v2.csv",
        index=False,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="Logistic Regression v2",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed high-dwell rate")
    plt.title("V2 Probability Calibration — Future Holdout")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    chart_path = OUT / "calibration_curve_v2.png"
    plt.savefig(chart_path, dpi=160)
    plt.close()

    print(f"\nSaved predictions: {OUT / 'probability_evaluation_v2.csv'}")
    print(f"Saved calibration plot: {chart_path}")


if __name__ == "__main__":
    main()