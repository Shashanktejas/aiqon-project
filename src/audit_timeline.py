from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent
df = pd.read_csv(root / "data" / "processed" / "model_dataset_v2.csv")
df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.sort_values(["Port", "YearMonth"])

# Confirm that target month follows feature month by exactly one month.
df["ExpectedNextMonth"] = (
    df["YearMonth"] + pd.offsets.MonthBegin(1)
)

print("Target timeline checks:")
print("Rows:", len(df))
print("Rows with next-month labels:", df["NextMonthMeanDwellDays"].notna().sum())
print("Missing target labels:", df["TargetHighDwellNextMonth"].isna().sum())

# Check target/threshold consistency.
valid = df["NextMonthMeanDwellDays"].notna()
expected = (
    df.loc[valid, "NextMonthMeanDwellDays"]
    > df.loc[valid, "HighDwellThreshold"]
).astype(int)

actual = df.loc[valid, "TargetHighDwellNextMonth"].astype(int)

print("Target matches threshold:", (expected == actual).all())

print("\nThresholds:")
print(
    df.groupby("Port")["HighDwellThreshold"]
    .first()
    .to_string()
)

print("\nDate range by port:")
print(
    df.groupby("Port")["YearMonth"]
    .agg(["min", "max", "count"])
    .to_string()
)

