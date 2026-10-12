
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data/processed/recall_priority_test_results.csv"
DATASET = ROOT / "data/processed/model_dataset_v2.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# 1. Load frozen predictions and original feature dataset.
pred = pd.read_csv(PREDICTIONS)
data = pd.read_csv(DATASET)

pred["Date"] = pd.to_datetime(pred["Date"], errors="coerce")
data["Date"] = pd.to_datetime(data["YearMonth"], errors="coerce")

required = [
    "Date", "Port", "TargetHighDwellNextMonth",
    "PredictedProbability", "PredictedHighDwell", "Threshold",
]
missing = [col for col in required if col not in pred.columns]
if missing:
    raise ValueError(f"Missing prediction columns: {missing}")

if pred["Date"].isna().any() or data["Date"].isna().any():
    raise ValueError("Some dates could not be parsed.")

# The prediction file is the source of truth for final test results.
pred["Actual"] = pred["TargetHighDwellNextMonth"].astype(int)
pred["Predicted"] = pred["PredictedHighDwell"].astype(int)

pred["Outcome"] = "TN"
pred.loc[(pred.Actual == 1) & (pred.Predicted == 1), "Outcome"] = "TP"
pred.loc[(pred.Actual == 1) & (pred.Predicted == 0), "Outcome"] = "FN"
pred.loc[(pred.Actual == 0) & (pred.Predicted == 1), "Outcome"] = "FP"

# 2. Enrich with available dwell features.
feature_cols = [
    "Date", "Port", "MeanDwellDays", "MedianDwellDays",
    "UniqueVessels", "DwellChange1", "DwellChange3",
    "MeanDwellDays_roll3", "MeanDwellDays_roll6",
    "MedianDwellDays_roll3", "MedianDwellDays_roll6",
    "UniqueVessels_roll3", "UniqueVessels_roll6",
]
feature_cols = [c for c in feature_cols if c in data.columns]
features = data[feature_cols].drop_duplicates(["Date", "Port"])

analysis = pred.merge(
    features,
    on=["Date", "Port"],
    how="left",
    validate="many_to_one",
)

# 3. Export all false negatives and false alarms.
fn = analysis[analysis["Outcome"] == "FN"].copy()
fp = analysis[analysis["Outcome"] == "FP"].copy()

fn["Month"] = fn["Date"].dt.month
fn["Season"] = fn["Month"].map({
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
})

fn.sort_values(["Date", "Port"]).to_csv(
    REPORTS / "false_negative_analysis.csv", index=False
)
fp.sort_values(["Date", "Port"]).to_csv(
    REPORTS / "false_positive_analysis.csv", index=False
)

fn.groupby("Port").size().rename("FalseNegatives").to_csv(
    REPORTS / "false_negatives_by_port.csv"
)
fn.groupby("Season").size().rename("FalseNegatives").to_csv(
    REPORTS / "false_negatives_by_season.csv"
)
fn.groupby("Month").size().rename("FalseNegatives").to_csv(
    REPORTS / "false_negatives_by_month.csv"
)

print(f"Test rows: {len(analysis)}")
print(f"False negatives: {len(fn)}")
print(f"False positives: {len(fp)}")
print("\nFalse negatives by port:")
print(fn.groupby("Port").size().sort_values(ascending=False))
print("\nFalse negatives by season:")
print(fn.groupby("Season").size().sort_values(ascending=False))
print("\nFalse-negative cases:")
show_cols = [
    c for c in [
        "Date", "Port", "PredictedProbability", "Threshold",
        "MeanDwellDays", "DwellChange1", "DwellChange3",
    ] if c in fn.columns
]
print(fn[show_cols].to_string(index=False))

# 4. Confusion matrix from the saved predictions.
cm = confusion_matrix(
    analysis["Actual"], analysis["Predicted"], labels=[0, 1]
)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal dwell", "High dwell"],
)
disp.plot(values_format="d")
plt.title("Frozen Final Test: Confusion Matrix (2024–2025)")
plt.tight_layout()
plt.savefig(FIGURES / "confusion_matrix.png", dpi=180)
plt.close()

# 5. Precision-recall curve from frozen probabilities.
precision, recall, _ = precision_recall_curve(
    analysis["Actual"], analysis["PredictedProbability"]
)
ap = average_precision_score(
    analysis["Actual"], analysis["PredictedProbability"]
)
plt.figure()
plt.plot(recall, precision, label=f"Average precision = {ap:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Frozen Final Test: Precision–Recall Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "precision_recall_curve.png", dpi=180)
plt.close()

# 6. Baseline report: metrics are recorded, not refitted.
tp = int(((analysis.Actual == 1) & (analysis.Predicted == 1)).sum())
fn_count = int(((analysis.Actual == 1) & (analysis.Predicted == 0)).sum())
fp_count = int(((analysis.Actual == 0) & (analysis.Predicted == 1)).sum())
tn = int(((analysis.Actual == 0) & (analysis.Predicted == 0)).sum())

baseline = f"""# Frozen Baseline Report

## Evaluation protocol
- Final test period: 2024–2025
- Test rows: {len(analysis)}
- Threshold selected on the separate validation period: 0.535
- Validation recall: 0.9167
- Validation F1: 0.8943
- Final test was not used to select the threshold or fit the model.

## Final test metrics
- Accuracy: 0.8261
- Precision: 0.8548
- Recall: 0.8281
- F1: 0.8413
- ROC-AUC: 0.9372
- PR-AUC: 0.9567
- True positives: {tp}
- False negatives: {fn_count}
- False positives: {fp_count}
- True negatives: {tn}

## Threshold selection
The classification threshold of 0.535 was selected using the separate
chronological validation period, where recall was 0.9167 and F1 was 0.8943.
The 2024–2025 test period is reserved for final evaluation and must not be
used to tune the threshold or choose model features.

## Interpretation and limitations
The target represents high dwell in the following month, using the project's
port-specific label definition. It is a proxy for port congestion, not a
direct observation of shipment delays. Results describe the available
historical dataset and do not guarantee future performance.

## False-negative investigation
See false_negative_analysis.csv and the port/month/season summaries.
These summaries are descriptive: the small number of missed cases does not
establish that a port or season causes model errors.

## Figures
- figures/confusion_matrix.png
- figures/precision_recall_curve.png
"""
(REPORTS / "baseline_report.md").write_text(baseline, encoding="utf-8")

# 7. Threshold-selection note.
threshold_note = """# Threshold Selection Procedure

The selected threshold is 0.535. It was selected on a separate chronological
validation period, with validation recall of 0.9167 and F1 of 0.8943.

The final test period (2024–2025) was reserved for one final evaluation.
Do not change the threshold based on the 11 false negatives or 9 false
positives observed on this test set. Use a new chronological validation
window to compare any revised model or threshold.

Recall measures the fraction of actual high-dwell cases detected.
Precision measures the fraction of high-dwell alerts that were correct.
The preferred operating threshold should reflect the relative cost of missed
high-dwell periods versus false alarms.

High dwell is a proxy for port congestion, not a direct shipment-delay label.
"""
(REPORTS / "threshold_selection.md").write_text(
    threshold_note, encoding="utf-8"
)

print("\nCreated reports:")
for path in sorted(REPORTS.rglob("*")):
    if path.is_file():
        print(path.relative_to(ROOT))
