# AIQON Project: Port Congestion Prediction

## Overview

This project uses machine learning to predict the likelihood of high port dwell time in the following month using historical port activity data.

It analyzes historical dwell-time patterns, vessel activity, seasonal trends, and port-specific characteristics to identify potential congestion risks.

**Objective:** Predict next-month high-dwell risk, estimate prediction confidence, and identify the factors influencing the prediction.

## Problem Statement

Port congestion can cause operational inefficiencies and shipment delays. Early identification of elevated dwell-time risk can help stakeholders plan operations and allocate resources more effectively.

This project frames the task as a binary classification problem:

- **High risk (1):** The following month's average dwell time exceeds the defined port-specific threshold.
- **Lower risk (0):** The following month's average dwell time does not exceed that threshold.

The target is a high-dwell congestion proxy, not a direct measurement of shipment delays.

## Dataset

The project uses historical AIS-derived port activity data for the following US port groups:

- Houston
- Los Angeles / Long Beach
- New York / New Jersey
- Savannah
- Seattle

The data includes historical dwell-time measurements and vessel activity. Operational mode features are also evaluated where available.

## Features

The feature engineering pipeline includes:

- Seasonal features using month sine and cosine.
- Year and port identifiers.
- Rolling 3-month and 6-month dwell-time statistics.
- Rolling vessel activity statistics.
- Changes in dwell time over recent months.
- Current-month dwell time and its margin above the port-specific threshold.
- Operational activity features evaluated in an additional experiment.

## Models and Evaluation

Logistic Regression models were evaluated using chronological data splits to reduce temporal leakage.

The leading recorded experiment was **V3 — Dwell Features Logistic Regression**.

| Metric | V3 test result |
|---|---:|
| Precision | 88.52% |
| Recall | 84.38% |
| F1-score | 86.40% |
| ROC-AUC | 94.52% |
| PR-AUC | 96.28% |
| Brier score | 0.106 |

These results reflect the recorded evaluation and should be reproduced from the final code and data before being treated as independently verified.

## Explainability and Prediction Confidence

The project includes prediction and explanation utilities to help interpret model outputs and the factors contributing to congestion risk.

Predicted probabilities represent model-estimated risk, not guaranteed outcomes. Their reliability depends on model calibration and the data distribution.

## Project Structure

```text
aiqon-project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── src/
├── README.md
└── .gitignore
```

The actual repository may contain additional scripts and files.

## Setup

Requirements: Python and the libraries used by the project scripts.

1. Clone the repository:

   ```bash
   git clone https://github.com/Shashanktejas/aiqon-project.git
   cd aiqon-project
   ```

2. Create and activate a virtual environment:

   **Windows PowerShell**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install the dependencies required by the project. Use the package list established in the development environment.

## Running the Project

Run the data preparation, feature engineering, training, validation, and prediction scripts in the order required by the implementation.

The exact execution commands should be confirmed against the final source files before running the pipeline from a clean environment.

## Limitations

- High dwell time is used as a proxy for congestion risk; shipment delays are not directly measured.
- Predictions depend on historical data quality and feature availability.
- Performance may change across ports and future time periods.
- Model confidence should not be interpreted as certainty.

## Author

**Shashank Sanda**

Developed as part of the AIQON internship assignment: *Predicting Port Congestion and Shipment Delays Using AI*.
