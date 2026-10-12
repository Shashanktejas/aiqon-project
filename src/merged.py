from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"

# Load the dwell datasets
dwell_early = pd.read_csv(DATA / "monthly_dwell_2009_2014.csv")
dwell_recent = pd.read_csv(DATA / "monthly_dwell_2015_2025.csv")

dwell = pd.concat(
    [dwell_early, dwell_recent],
    ignore_index=True,
)

# Load aggregated operational features
mode = pd.read_csv(
    PROCESSED / "port_month_mode_features.csv"
)

# Normalize dates before merging
for df in (dwell, mode):
    df["YearMonth"] = (
        pd.to_datetime(df["YearMonth"])
        .dt.to_period("M")
        .astype(str)
    )

keys = ["Port", "YearMonth"]

# Validate the merge keys
assert not dwell.duplicated(keys).any(), (
    "Duplicate port-month keys in dwell data"
)
assert not mode.duplicated(keys).any(), (
    "Duplicate port-month keys in operational data"
)

# Keep every dwell record; operational features may be missing
merged = dwell.merge(
    mode,
    on=keys,
    how="left",
    indicator=True,
    validate="one_to_one",
)

merged["HasModeData"] = merged["_merge"].eq("both")

print("=== MERGE SUMMARY ===")
print("Dwell rows:", len(dwell))
print("Operational port-month rows:", len(mode))
print("Merged rows:", len(merged))
print("Rows with operational data:", merged["HasModeData"].sum())
print("Rows without operational data:", (~merged["HasModeData"]).sum())
print(
    "Operational coverage:",
    round(100 * merged["HasModeData"].mean(), 2),
    "%",
)

print("\n=== COVERAGE BY PORT ===")
coverage = (
    merged.groupby("Port")["HasModeData"]
    .agg(["sum", "count", "mean"])
)
coverage["coverage_percent"] = (coverage["mean"] * 100).round(2)
print(coverage.drop(columns="mean").to_string())

print("\n=== MISSING PORT-MONTHS ===")
missing = merged.loc[~merged["HasModeData"], keys]
print(missing.to_string(index=False))

print("\n=== DWELL DATE RANGE ===")
print(dwell["YearMonth"].min(), "to", dwell["YearMonth"].max())

# Save the merged dataset
merged = merged.drop(columns="_merge")
output = PROCESSED / "port_month_combined.csv"
merged.to_csv(output, index=False)

print("\nSaved:", output)