from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

MODELS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROCESSED / "model_dataset.csv")
df["YearMonth"] = pd.to_datetime(df["YearMonth"])

# Only rows with an available target can be evaluated
df = df.dropna(subset=["TargetHighDwellNextMonth"]).copy()
df["TargetHighDwellNextMonth"] = (
    df["TargetHighDwellNextMonth"].astype(int)
)

# Strict chronological split using unique calendar months
months = np.sort(df["YearMonth"].unique())
split_index = int(len(months) * 0.70)
cutoff = pd.Timestamp(months[split_index - 1])

# Each row predicts the following month's outcome.
# Keep training labels within the training period.
train = df[df["YearMonth"] < cutoff].copy()
test = df[df["YearMonth"] >= cutoff].copy()

assert (
    train["YearMonth"].max() + pd.offsets.MonthBegin(1)
    <= cutoff
), "Training labels may extend beyond the cutoff"

assert (
    test["YearMonth"].min() >= cutoff
), "Unexpected test split"

# The target threshold was estimated from the training period.
# Exclude identifiers, current/future outcomes, and raw same-month
# measurements to focus on lagged features for this first experiment.
numeric_features = [
    "MeanDwellDays_lag1",
    "MedianDwellDays_lag1",
    "UniqueVessels_lag1",
    "TotalObservations_lag1",
    "anchor_hours_lag1",
    "berth_hours_lag1",
    "total_mode_hours_lag1",
    "DwellChange_lag1",
]

numeric_features = [
    c for c in numeric_features if c in df.columns
]

categorical_features = ["Port"]

X_train = train[numeric_features + categorical_features]
y_train = train["TargetHighDwellNextMonth"]

X_test = test[numeric_features + categorical_features]
y_test = test["TargetHighDwellNextMonth"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

models = {
    "MajorityBaseline": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
fitted_models = {}

print("Training cutoff:", cutoff.date())
print("Training rows:", len(train))
print("Test rows:", len(test))
print("Training months:", train["YearMonth"].nunique())
print("Test months:", test["YearMonth"].nunique())
print("Features:", numeric_features + categorical_features)

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    if hasattr(pipeline, "predict_proba") and y_test.nunique() == 2:
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        row["roc_auc"] = roc_auc_score(y_test, probabilities)
        row["pr_auc"] = average_precision_score(
            y_test, probabilities
        )
    else:
        probabilities = None
        row["roc_auc"] = np.nan
        row["pr_auc"] = np.nan

    results.append(row)
    fitted_models[name] = pipeline

    print(f"\n=== {name} ===")
    print(json.dumps(row, indent=2))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

# Save comparison table
results_df = pd.DataFrame(results).sort_values(
    "pr_auc", ascending=False, na_position="last"
)
results_path = PROCESSED / "model_comparison.csv"
results_df.to_csv(results_path, index=False)

# Save the random forest for later explanation and dashboard work
joblib.dump(
    fitted_models["RandomForest"],
    MODELS / "random_forest.joblib",
)

joblib.dump(
    fitted_models["LogisticRegression"],
    MODELS / "logistic_regression.joblib",
)

print("\n=== MODEL COMPARISON ===")
print(results_df.round(4).to_string(index=False))
print("\nSaved:", results_path)
print("Saved models in:", MODELS)