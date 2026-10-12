
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_dataset_v2.csv"

FEATURES = [
    "MonthSin", "MonthCos", "Year",
    "MeanDwellDays_roll3", "MeanDwellDays_roll6",
    "MedianDwellDays_roll3", "MedianDwellDays_roll6",
    "UniqueVessels_roll3", "UniqueVessels_roll6",
    "DwellChange1", "DwellChange3", "Port",
]

TARGET = "TargetHighDwellNextMonth"
MIN_RECALL = 0.90

df = pd.read_csv(DATA)

# Confirm the expected columns exist before training.

required = FEATURES + [TARGET, "YearMonth"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

df["Date"] = pd.to_datetime(
    df["YearMonth"].astype(str),
    errors="coerce"
)

if df["Date"].isna().any():
    raise ValueError("Some YearMonth values could not be parsed as dates.")

df = df.dropna(subset=[TARGET]).sort_values("Date").reset_index(drop=True)
df[TARGET] = df[TARGET].astype(int)

# Train only on dates before January 2022.
train = df[df["Date"] < "2022-01-01"].copy()

# Use 2022–2023 to select the operating threshold.
validation = df[
    (df["Date"] >= "2022-01-01") &
    (df["Date"] < "2024-01-01")
].copy()

# Keep 2024–2025 untouched until the final evaluation.
test = df[
    (df["Date"] >= "2024-01-01") &
    (df["Date"] < "2026-01-01")
].copy()

if min(len(train), len(validation), len(test)) == 0:
    raise ValueError(
        f"Empty split: train={len(train)}, "
        f"validation={len(validation)}, test={len(test)}"
    )

numeric_features = [c for c in FEATURES if c != "Port"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_features),
    ("port", OneHotEncoder(handle_unknown="ignore"), ["Port"]),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )),
])

X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = validation[FEATURES], validation[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

model.fit(X_train, y_train)

# Select threshold using validation data only.
val_prob = model.predict_proba(X_val)[:, 1]
candidates = []

for threshold in np.arange(0.05, 0.951, 0.005):
    pred = (val_prob >= threshold).astype(int)
    recall = recall_score(y_val, pred, zero_division=0)
    precision = precision_score(y_val, pred, zero_division=0)

    if recall >= MIN_RECALL:
        candidates.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1_score(y_val, pred, zero_division=0),
        })

if not candidates:
    print(
        f"No threshold reached {MIN_RECALL:.0%} recall "
        "on validation data. Do not claim the target is met."
    )
    print("Maximum validation recall:", max(
        recall_score(y_val, (val_prob >= t).astype(int), zero_division=0)
        for t in np.arange(0.05, 0.951, 0.005)
    ))
    raise SystemExit(1)

# Among thresholds meeting the recall target, maximize precision.
best = max(
    candidates,
    key=lambda x: (x["precision"], x["f1"], x["threshold"])
)
threshold = best["threshold"]

print("\n--- THRESHOLD SELECTED ON 2022–2023 ONLY ---")
print(f"Threshold: {threshold:.3f}")
print(f"Validation precision: {best['precision']:.4f}")
print(f"Validation recall:    {best['recall']:.4f}")
print(f"Validation F1:        {best['f1']:.4f}")

# Evaluate exactly once on the untouched 2024–2025 test period.
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
).ravel()

print("\n--- FINAL UNTOUCHED TEST: 2024–2025 ---")
print(f"Rows:      {len(test)}")
print(f"Threshold: {threshold:.3f}")
print(f"Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, test_pred, zero_division=0):.4f}")
print(f"F1:        {f1_score(y_test, test_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, test_prob):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, test_prob):.4f}")
print(f"True positives (high dwell found): {tp}")
print(f"False negatives (high dwell missed): {fn}")
print(f"False positives (false alarms): {fp}")
print(f"True negatives: {tn}")

results = test[["Date", "Port", TARGET]].copy()
results["PredictedProbability"] = test_prob
results["PredictedHighDwell"] = test_pred
results["Threshold"] = threshold
results.to_csv(
    ROOT / "data" / "processed" / "recall_priority_test_results.csv",
    index=False,
)

print("\nSaved final predictions to:")
print(ROOT / "data" / "processed" / "recall_priority_test_results.csv")
