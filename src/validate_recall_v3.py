
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_dataset_v2.csv"
OUTPUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TARGET = "TargetHighDwellNextMonth"
MIN_RECALL = 0.90

BASE_FEATURES = [
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

# Candidate adds only current-month dwell information.
V3_FEATURES = BASE_FEATURES + [
    "MeanDwellDays",
    "DwellMargin",
]
OPERATIONAL_FEATURES = [
    "anchor_hours",
    "berth_hours",
    "manoeuvre_hours",
    "transit_hours",
    "unknown_hoteling_hours",
    "HasModeData",
]

V4_FEATURES = V3_FEATURES + OPERATIONAL_FEATURES

def make_model(features):
    numeric = [c for c in features if c != "Port"]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True),
        ),
        ("scaler", StandardScaler()),
    ])

    port_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("port", port_pipeline, ["Port"]),
    ])

    return Pipeline([
        ("preprocess", preprocess),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
            ),
        ),
    ])


def select_threshold(y_true, probabilities):
    """On validation data, meet recall >= 90%, then maximize precision."""
    candidates = []

    for threshold in np.arange(0.05, 0.951, 0.005):
        predictions = (probabilities >= threshold).astype(int)

        rec = recall_score(
            y_true, predictions, zero_division=0
        )

        if rec >= MIN_RECALL:
            candidates.append({
                "threshold": float(threshold),
                "precision": precision_score(
                    y_true, predictions, zero_division=0
                ),
                "recall": rec,
                "f1": f1_score(
                    y_true, predictions, zero_division=0
                ),
            })

    if not candidates:
        raise ValueError(
            "No validation threshold achieved 90% recall. "
            "Do not claim the recall target was met."
        )

    return max(
        candidates,
        key=lambda row: (
            row["precision"],
            row["f1"],
            row["threshold"],
        ),
    )


def evaluate(name, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "PR_AUC": average_precision_score(
            y_true, probabilities
        ),
        "Brier": brier_score_loss(y_true, probabilities),
        "LogLoss": log_loss(
            y_true, probabilities, labels=[0, 1]
        ),
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
    }


def run_experiment(df, features, name):
    train = df[df["Date"] < "2022-01-01"].copy()
    validation = df[
        (df["Date"] >= "2022-01-01")
        & (df["Date"] < "2024-01-01")
    ].copy()
    test = df[
        (df["Date"] >= "2024-01-01")
        & (df["Date"] < "2026-01-01")
    ].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            f"Empty split: train={len(train)}, "
            f"validation={len(validation)}, test={len(test)}"
        )

    model = make_model(features)
    model.fit(train[features], train[TARGET])

    validation_probabilities = model.predict_proba(
        validation[features]
    )[:, 1]

    selected = select_threshold(
        validation[TARGET], validation_probabilities
    )
    threshold = selected["threshold"]

    # Final test is evaluated only after the threshold is selected.
    test_probabilities = model.predict_proba(
        test[features]
    )[:, 1]

    metrics = evaluate(
        name, test[TARGET], test_probabilities, threshold
    )

    predictions = test[
        ["Date", "Port", TARGET]
    ].copy()
    predictions["PredictedProbability"] = test_probabilities
    predictions["PredictedHighDwell"] = (
        test_probabilities >= threshold
    ).astype(int)
    predictions["Threshold"] = threshold

    print(f"\n--- {name} ---")
    print(f"Features: {len(features)}")
    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Final test rows: {len(test)}")
    print(f"Validation-selected threshold: {threshold:.3f}")
    print(
        f"Validation precision: {selected['precision']:.4f}"
    )
    print(f"Validation recall: {selected['recall']:.4f}")
    print(f"Validation F1: {selected['f1']:.4f}")
    print("\nFinal test metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics, predictions


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)

    required = list(set(
    BASE_FEATURES
    + ["MeanDwellDays", "HighDwellThreshold",
       TARGET, "YearMonth"]
    + OPERATIONAL_FEATURES
))
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(
        df["YearMonth"], errors="coerce"
    )

    if df["Date"].isna().any():
        raise ValueError("Could not parse some YearMonth values.")

    # DwellMargin uses current-month values only.
    # HighDwellThreshold must have been established from training-era data.
    df["DwellMargin"] = (
        pd.to_numeric(df["MeanDwellDays"], errors="coerce")
        - pd.to_numeric(
            df["HighDwellThreshold"], errors="coerce"
        )
    )

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    df = df.sort_values(["Date", "Port"]).reset_index(drop=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Same chronological splits for both models.
    
baseline_metrics, baseline_predictions = run_experiment(
    df, BASE_FEATURES, "Baseline_LR"
)

v3_metrics, v3_predictions = run_experiment(
    df, V3_FEATURES, "V3_DwellFeatures_LR"
)

v4_metrics, v4_predictions = run_experiment(
    df, V4_FEATURES, "V4_Dwell_Operational_LR"
)

# Save v4 separately.
v4_path = OUTPUT / "recall_priority_v4_test_results.csv"
v4_predictions.to_csv(v4_path, index=False)

comparison = pd.DataFrame([
    baseline_metrics,
    v3_metrics,
    v4_metrics,
])

comparison_path = REPORTS / "baseline_v3_v4_comparison.csv"
comparison.to_csv(comparison_path, index=False)

print("\n--- V4 minus V3 ---")
for metric in [
    "Precision", "Recall", "F1", "ROC_AUC",
    "PR_AUC", "Brier", "FN", "FP",
]:
    delta = v4_metrics[metric] - v3_metrics[metric]
    print(f"{metric}: {delta:+.4f}")

print(f"\nSaved v4 predictions: {v4_path}")
print(f"Saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
