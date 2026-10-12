# Threshold Selection Procedure

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
