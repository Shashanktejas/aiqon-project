from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "probability_evaluation_v2.csv"

df = pd.read_csv(DATA)
y = df["ActualHighDwell"].astype(int)
prob = df["PredictedProbability"]

print("=== THRESHOLD ANALYSIS ===")
print(f"Rows evaluated: {len(df)}")
print()

results = []

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    row = {
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "f1": f1_score(y, pred, zero_division=0),
        "false_alarms": fp,
        "missed_high_dwell": fn,
        "true_positives": tp,
        "true_negatives": tn,
    }
    results.append(row)

result_df = pd.DataFrame(results)
print(result_df.round(3).to_string(index=False))

output = ROOT / "data" / "processed" / "threshold_analysis_v2.csv"
result_df.to_csv(output, index=False)
print(f"\nSaved: {output}")