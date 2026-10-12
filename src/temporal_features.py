from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "processed" / "port_month_combined.csv"
OUTPUT = ROOT / "data" / "processed" / "model_dataset_v2.csv"
THRESHOLD_OUTPUT = ROOT / "data" / "processed" / "high_dwell_thresholds_v2.csv"

df = pd.read_csv(SOURCE)
df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.sort_values(["Port", "YearMonth"]).copy()

# Calculate thresholds using only the earliest 70% of unique months.
months = np.sort(df["YearMonth"].dropna().unique())
split_index = int(len(months) * 0.70)
cutoff = pd.Timestamp(months[split_index - 1])

threshold_rows = []
for port, group in df.groupby("Port"):
    train_values = group.loc[
        group["YearMonth"] <= cutoff, "MeanDwellDays"
    ].dropna()

    threshold_rows.append({
        "Port": port,
        "HighDwellThreshold": train_values.quantile(0.75),
        "TrainingCutoff": cutoff,
    })

thresholds = pd.DataFrame(threshold_rows)
df = df.merge(
    thresholds[["Port", "HighDwellThreshold"]],
    on="Port",
    how="left",
    validate="many_to_one",
)
thresholds.to_csv(THRESHOLD_OUTPUT, index=False)

# Calendar features.
df["MonthOfYear"] = df["YearMonth"].dt.month
df["MonthSin"] = np.sin(2 * np.pi * df["MonthOfYear"] / 12)
df["MonthCos"] = np.cos(2 * np.pi * df["MonthOfYear"] / 12)
df["Year"] = df["YearMonth"].dt.year

# Rolling features use current and earlier observed months only.
for column in ["MeanDwellDays", "MedianDwellDays", "UniqueVessels"]:
    grouped = df.groupby("Port")[column]

    df[f"{column}_roll3"] = grouped.transform(
        lambda s: s.rolling(3, min_periods=2).mean()
    )
    df[f"{column}_roll6"] = grouped.transform(
        lambda s: s.rolling(6, min_periods=3).mean()
    )

df["DwellChange1"] = (
    df.groupby("Port")["MeanDwellDays"].diff(1)
)
df["DwellChange3"] = (
    df.groupby("Port")["MeanDwellDays"].diff(3)
)

# Target: next month's mean dwell exceeds this port's threshold.
df["NextMonthMeanDwellDays"] = (
    df.groupby("Port")["MeanDwellDays"].shift(-1)
)

df["TargetHighDwellNextMonth"] = pd.Series(
    pd.NA, index=df.index, dtype="Int64"
)

valid = df["NextMonthMeanDwellDays"].notna()
df.loc[valid, "TargetHighDwellNextMonth"] = (
    df.loc[valid, "NextMonthMeanDwellDays"]
    > df.loc[valid, "HighDwellThreshold"]
).astype(int)

df.to_csv(OUTPUT, index=False)

print("Saved dataset:", OUTPUT)
print("Saved thresholds:", THRESHOLD_OUTPUT)
print("Training cutoff:", cutoff.strftime("%Y-%m"))
print("\nThresholds:")
print(thresholds.to_string(index=False))
print("\nTarget distribution:")
print(df["TargetHighDwellNextMonth"].value_counts(dropna=False))