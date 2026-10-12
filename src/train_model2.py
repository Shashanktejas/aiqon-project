from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "model_dataset_v2.csv"
OUT = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.dropna(subset=["TargetHighDwellNextMonth"]).copy()
df["TargetHighDwellNextMonth"] = (
    df["TargetHighDwellNextMonth"].astype(int)
)

# Same chronological cutoff as the previous experiment.
months = np.sort(df["YearMonth"].unique())
split_index = int(len(months) * 0.70)
cutoff = pd.Timestamp(months[split_index - 1])

train = df[df["YearMonth"] < cutoff].copy()
test = df[df["YearMonth"] >= cutoff].copy()

# Exclude contemporaneous outcome measurements and metadata.
# Use only known features for predicting the following month.
numeric_features = [
    "MeanDwellDays_lag1",
    "MedianDwellDays_lag1",
    "UniqueVessels_lag1",
    "TotalObservations_lag1",
    "anchor_hours_lag1",
    "berth_hours_lag1",
    "total_mode_hours_lag1",
    "DwellChange_lag1",
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
numeric_features = [c for c in numeric_features if c in df.columns]
categorical_features = ["Port"]
features = numeric_features + categorical_features

X_train, y_train = train[features], train["TargetHighDwellNextMonth"]
X_test, y_test = test[features], test["TargetHighDwellNextMonth"]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

models = {
    "LogisticRegression_v2": Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", LogisticRegression(
            max_iter=3000, class_weight="balanced"
        )),
    ]),
    "RandomForest_v2": Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        )),
    ]),
    "HistGradientBoosting_v2": Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )),
    ]),
}

results = []
print("Training cutoff:", cutoff.strftime("%Y-%m"))
print("Train rows:", len(train), "| Test rows:", len(test))
print("Features:", features)

for name, model in models.items():
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob),
    }
    results.append(metrics)
    print(f"\n=== {name} ===")
    print(json.dumps(metrics, indent=2))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    joblib.dump(model, MODELS / f"{name}.joblib")

comparison = pd.DataFrame(results).sort_values(
    "pr_auc", ascending=False
)
comparison.to_csv(OUT / "model_comparison_v2.csv", index=False)

print("\n=== COMPARISON ===")
print(comparison.round(4).to_string(index=False))
print("\nSaved:", OUT / "model_comparison_v2.csv")
print("Saved models to:", MODELS)