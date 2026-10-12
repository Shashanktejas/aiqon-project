from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

df = pd.read_csv(PROCESSED / "port_month_combined.csv")

df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.sort_values(["Port", "YearMonth"]).reset_index(drop=True)

# Confirm one record per port-month
assert not df.duplicated(["Port", "YearMonth"]).any()

# Convert relevant columns to numeric where present
numeric_cols = [
    "MeanDwellDays",
    "MedianDwellDays",
    "StdDwellDays",
    "UniqueVessels",
    "TotalObservations",
    "AvgVesselLength",
    "anchor_hours",
    "berth_hours",
    "manoeuvre_hours",
    "transit_hours",
    "unknown_hoteling_hours",
    "total_mode_hours",
    "ModeVesselCount",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Features available at the end of the current month
for col in [
    "MeanDwellDays",
    "MedianDwellDays",
    "UniqueVessels",
    "TotalObservations",
    "anchor_hours",
    "berth_hours",
    "total_mode_hours",
]:
    if col in df.columns:
        df[f"{col}_lag1"] = df.groupby("Port")[col].shift(1)

# Previous month's dwell trend
df["DwellChange_lag1"] = (
    df.groupby("Port")["MeanDwellDays"].diff()
)

# Target: next month's mean dwell time
df["NextMonthMeanDwellDays"] = (
    df.groupby("Port")["MeanDwellDays"].shift(-1)
)

# Time-based split: first 70% of unique months for training
months = np.sort(df["YearMonth"].dropna().unique())
split_index = int(len(months) * 0.70)
train_end = pd.Timestamp(months[split_index - 1])

# Define high dwell separately for each port using training data only
train_rows = df["YearMonth"] <= train_end

thresholds = (
    df.loc[train_rows]
    .groupby("Port")["MeanDwellDays"]
    .quantile(0.75)
    .to_dict()
)

df["HighDwellThreshold"] = df["Port"].map(thresholds)

# Label the following month, using the threshold fixed from training
df["TargetHighDwellNextMonth"] = (
    df["NextMonthMeanDwellDays"] > df["HighDwellThreshold"]
).astype("Int64")

# The final month cannot have a next-month label
df.loc[
    df["NextMonthMeanDwellDays"].isna(),
    "TargetHighDwellNextMonth",
] = pd.NA

# Save target thresholds for reporting
threshold_table = pd.DataFrame(
    thresholds.items(),
    columns=["Port", "HighDwellThresholdDays"],
)
threshold_table.to_csv(
    PROCESSED / "high_dwell_thresholds.csv",
    index=False,
)

# Save feature-ready dataset
output = PROCESSED / "model_dataset.csv"
df.to_csv(output, index=False)

print("=== FEATURE DATASET CREATED ===")
print("Rows:", len(df))
print("Ports:", df["Port"].nunique())
print("Months:", df["YearMonth"].min(), "to", df["YearMonth"].max())
print("Training cutoff:", train_end.date())

print("\n=== PORT-SPECIFIC HIGH-DWELL THRESHOLDS ===")
print(threshold_table.to_string(index=False))

print("\n=== TARGET DISTRIBUTION ===")
print(
    df["TargetHighDwellNextMonth"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\n=== OPERATIONAL DATA AVAILABILITY ===")
if "HasModeData" in df.columns:
    print(df["HasModeData"].value_counts(dropna=False).to_string())

print("\nSaved:", output)
