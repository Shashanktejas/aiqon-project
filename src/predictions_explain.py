from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

df = pd.read_csv(PROCESSED / "model_dataset.csv")
df["YearMonth"] = pd.to_datetime(df["YearMonth"])

df = df.dropna(subset=["TargetHighDwellNextMonth"]).copy()
df["TargetHighDwellNextMonth"] = (
    df["TargetHighDwellNextMonth"].astype(int)
)

months = np.sort(df["YearMonth"].unique())
split_index = int(len(months) * 0.70)
cutoff = pd.Timestamp(months[split_index - 1])

# Only generate evaluation predictions for the held-out period
test = df[df["YearMonth"] >= cutoff].copy()

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
features = numeric_features + categorical_features

model = joblib.load(MODELS / "logistic_regression.joblib")

X_test = test[features]
probabilities = model.predict_proba(X_test)[:, 1]

# Explain individual predictions using each transformed feature's
# contribution to the fitted logistic regression score.
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]

X_transformed = preprocessor.transform(X_test)
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()
contributions = X_transformed * classifier.coef_[0]

# Risk categories are provisional display thresholds, not calibrated
# probabilities or validated operational alert levels.
def risk_category(p):
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Moderate"
    return "Low"

rows = []

for i, (_, record) in enumerate(test.iterrows()):
    contribution = contributions[i]
    top_indices = np.argsort(contribution)[::-1]

    # Positive contributions increase the predicted high-dwell score.
    positive = [
        f"{feature_names[j]} (+{contribution[j]:.2f})"
        for j in top_indices
        if contribution[j] > 0
    ][:3]

    rows.append({
        "Port": record["Port"],
        "PredictionForMonth": (
            record["YearMonth"] + pd.offsets.MonthBegin(1)
        ).strftime("%Y-%m"),
        "EstimatedHighDwellProbability": round(
            float(probabilities[i]), 4
        ),
        "RiskCategory": risk_category(float(probabilities[i])),
        "ActualHighDwell": int(
            record["TargetHighDwellNextMonth"]
        ),
        "TopPositiveFactors": "; ".join(positive),
    })

output = pd.DataFrame(rows)
path = PROCESSED / "test_predictions_explained.csv"
output.to_csv(path, index=False)

print("Saved:", path)
print("\nSample predictions:")
print(output.head(15).to_string(index=False))

print("\nRisk categories:")
print(output["RiskCategory"].value_counts().to_string())
