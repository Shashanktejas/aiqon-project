from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "model_dataset_v2.csv"
MODELS = ROOT / "models"
OUT = ROOT / "data" / "processed"

FEATURES = [
    "MonthSin", "MonthCos", "Year",
    "MeanDwellDays_roll3", "MeanDwellDays_roll6",
    "MedianDwellDays_roll3", "MedianDwellDays_roll6",
    "UniqueVessels_roll3", "UniqueVessels_roll6",
    "DwellChange1", "DwellChange3", "Port",
]

NUMERIC = [c for c in FEATURES if c != "Port"]
CATEGORICAL = ["Port"]

TRAIN_END = pd.Timestamp("2022-01-01")
CALIBRATION_END = pd.Timestamp("2024-01-01")
FINAL_END = pd.Timestamp("2026-01-01")


def main():
    df = pd.read_csv(DATA)
    df["YearMonth"] = pd.to_datetime(df["YearMonth"])
    df = df.dropna(subset=["TargetHighDwellNextMonth"]).copy()
    df["TargetHighDwellNextMonth"] = (
        df["TargetHighDwellNextMonth"].astype(int)
    )

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    train = df[df["YearMonth"] < TRAIN_END]
    calibration = df[
        (df["YearMonth"] >= TRAIN_END)
        & (df["YearMonth"] < CALIBRATION_END)
    ]
    final = df[
        (df["YearMonth"] >= CALIBRATION_END)
        & (df["YearMonth"] < FINAL_END)
    ]

    for name, part in [
        ("training", train),
        ("calibration", calibration),
        ("final evaluation", final),
    ]:
        if part.empty or part["TargetHighDwellNextMonth"].nunique() < 2:
            raise ValueError(
                f"{name} period is empty or has only one target class."
            )

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(
                strategy="median", add_indicator=True
            )),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
        ("categorical", OneHotEncoder(
            handle_unknown="ignore"
        ), CATEGORICAL),
    ])

    base_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
        )),
    ])

    X_train, y_train = train[FEATURES], train["TargetHighDwellNextMonth"]
    X_cal, y_cal = calibration[FEATURES], calibration["TargetHighDwellNextMonth"]
    X_final = final[FEATURES]
    y_final = final["TargetHighDwellNextMonth"].to_numpy()

    # Fit the predictive model on the earliest period only.
    base_model.fit(X_train, y_train)

    # Fit the sigmoid calibrator using a separate, later period.
    calibration_prob = base_model.predict_proba(X_cal)[:, 1]
    calibrator = LogisticRegression(max_iter=2000)
    calibrator.fit(
        calibration_prob.reshape(-1, 1),
        y_cal,
    )

    raw_prob = base_model.predict_proba(X_final)[:, 1]
    calibrated_prob = calibrator.predict_proba(
        raw_prob.reshape(-1, 1)
    )[:, 1]

    def report(name, prob):
        pred = (prob >= 0.45).astype(int)
        print(f"\n=== {name} ===")
        print(f"Accuracy:  {accuracy_score(y_final, pred):.4f}")
        print(f"Precision: {precision_score(y_final, pred, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_final, pred, zero_division=0):.4f}")
        print(f"F1:        {f1_score(y_final, pred, zero_division=0):.4f}")
        print(f"ROC-AUC:   {roc_auc_score(y_final, prob):.4f}")
        print(f"Brier:     {brier_score_loss(y_final, prob):.4f}")
        print(f"Log loss:  {log_loss(y_final, prob):.4f}")

    print("Training rows:", len(train))
    print("Calibration rows:", len(calibration))
    print("Final evaluation rows:", len(final))
    print("Calibration period:", "2022-01 to 2023-12")
    print("Final evaluation period:", "2024-01 to 2025-12")

    report("Raw probabilities", raw_prob)
    report("Sigmoid-calibrated probabilities", calibrated_prob)

    # Save both versions for further analysis.
    predictions = final[["YearMonth", "Port"]].copy()
    predictions["ActualHighDwell"] = y_final
    predictions["RawProbability"] = raw_prob
    predictions["CalibratedProbability"] = calibrated_prob
    predictions.to_csv(
        OUT / "calibration_comparison_v2.csv",
        index=False,
    )

    bundle = {
        "model": base_model,
        "calibrator": calibrator,
        "features": FEATURES,
        "threshold": 0.45,
        "calibration_method": "sigmoid",
    }
    model_path = MODELS / "calibrated_logistic_regression_v2.joblib"
    joblib.dump(bundle, model_path)

    print("\nSaved predictions:", OUT / "calibration_comparison_v2.csv")
    print("Saved calibrated model:", model_path)


if __name__ == "__main__":
    main()
