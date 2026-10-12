# Frozen Baseline Report

## Evaluation protocol
- Final test period: 2024–2025
- Test rows: 115
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
- True positives: 53
- False negatives: 11
- False positives: 9
- True negatives: 42

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
