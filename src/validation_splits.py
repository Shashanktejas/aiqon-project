from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "model_dataset_v2.csv"

df = pd.read_csv(DATA)
df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.dropna(subset=["TargetHighDwellNextMonth"]).copy()
df["TargetHighDwellNextMonth"] = (
    df["TargetHighDwellNextMonth"].astype(int)
)

numeric = [
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
]
numeric = [c for c in numeric if c in df.columns]
categorical = ["Port"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])

# Expanding-window tests: each test period is later than its training data.
periods = [
    ("2020-10-01", "2022-01-01"),
    ("2022-01-01", "2024-01-01"),
    ("2024-01-01", "2026-01-01"),
]

rows = []

for start, end in periods:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    train = df[df["YearMonth"] < start].copy()
    test = df[
        (df["YearMonth"] >= start)
        & (df["YearMonth"] < end)
    ].copy()

    if train.empty or test.empty or train["TargetHighDwellNextMonth"].nunique() < 2:
        print(f"Skipping {start:%Y-%m} to {end:%Y-%m}: insufficient data")
        continue

    model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
        )),
    ])

    features = numeric + categorical
    X_train, y_train = train[features], train["TargetHighDwellNextMonth"]
    X_test, y_test = test[features], test["TargetHighDwellNextMonth"]

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    result = {
        "period": f"{start:%Y-%m} to {end:%Y-%m}",
        "train_rows": len(train),
        "test_rows": len(test),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
    }
    rows.append(result)
    print("\n", result)

out = pd.DataFrame(rows)
out.to_csv(
    ROOT / "data" / "processed" / "time_validation_v2.csv",
    index=False,
)
print("\n=== TIME VALIDATION SUMMARY ===")
print(out.round(4).to_string(index=False))