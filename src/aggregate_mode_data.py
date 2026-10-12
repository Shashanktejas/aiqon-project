from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "processed"

OUT.mkdir(parents=True, exist_ok=True)

key = ["MMSI", "Port", "YearMonth"]

mode_cols = [
    "anchor_hours",
    "berth_hours",
    "manoeuvre_hours",
    "transit_hours",
    "unknown_hoteling_hours",
    "total_mode_hours",
]

# Load both source periods
early = pd.read_csv(DATA / "monthly_mode_time_2009_2014.csv")
recent = pd.read_csv(DATA / "monthly_mode_time_2015_2025.csv")

# Identify every key repeated in the older dataset
counts = early.groupby(key, dropna=False).size()
repeated_keys = counts[counts > 1].index

early_key_index = pd.MultiIndex.from_frame(early[key])
early["duplicate_key"] = early_key_index.isin(repeated_keys)

# Save excluded records for transparency and review
excluded = early.loc[early["duplicate_key"]].copy()
excluded.to_csv(OUT / "excluded_duplicate_mode_records.csv", index=False)

# Exclude all rows belonging to repeated keys, not just one row
early_clean = early.loc[~early["duplicate_key"]].drop(
    columns="duplicate_key"
)

# Recent data has already been checked for duplicate keys
assert not recent.duplicated(key).any(), (
    "Unexpected duplicate keys in recent dataset"
)

# Combine the cleaned older data with the recent data
mode = pd.concat(
    [early_clean, recent],
    ignore_index=True,
)

assert not mode.duplicated(key).any(), (
    "Duplicate vessel-port-month keys remain after cleaning"
)

# Aggregate vessel-level records to port-month features
port_month_mode = (
    mode.groupby(["Port", "YearMonth"], as_index=False)[mode_cols]
    .sum()
)

# Add counts and operational coverage information
vessel_counts = (
    mode.groupby(["Port", "YearMonth"])["MMSI"]
    .nunique()
    .rename("ModeVesselCount")
    .reset_index()
)

port_month_mode = port_month_mode.merge(
    vessel_counts,
    on=["Port", "YearMonth"],
    how="left",
)

# Save cleaned intermediate dataset
port_month_mode.to_csv(
    OUT / "port_month_mode_features.csv",
    index=False,
)

print("=== AGGREGATION COMPLETE ===")
print("Older rows originally:", len(early))
print("Older rows excluded:", len(excluded))
print("Older rows retained:", len(early_clean))
print("Recent rows retained:", len(recent))
print("Combined clean vessel-month rows:", len(mode))
print("Port-month feature rows:", len(port_month_mode))
print(
    "Port-months covered:",
    port_month_mode[["Port", "YearMonth"]].drop_duplicates().shape[0],
)
print("\nDate range:", mode["YearMonth"].min(), "to", mode["YearMonth"].max())
print("\nRows by port:")
print(port_month_mode.groupby("Port").size().to_string())
print("\nSaved:", OUT / "port_month_mode_features.csv")
print("Excluded records saved:", OUT / "excluded_duplicate_mode_records.csv")