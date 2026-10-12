from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

early = pd.read_csv(DATA / "monthly_mode_time_2009_2014.csv")

key = ["MMSI", "Port", "YearMonth"]
mode_cols = [
    "anchor_hours",
    "berth_hours",
    "manoeuvre_hours",
    "transit_hours",
    "unknown_hoteling_hours",
    "total_mode_hours",
]

# Identify repeated keys
counts = early.groupby(key, dropna=False).size()
repeated_keys = counts[counts > 1].index

early["has_duplicate_key"] = pd.MultiIndex.from_frame(
    early[key]
).isin(repeated_keys)

print("=== DUPLICATE IMPACT ===")
print("Total rows:", len(early))
print("Rows with repeated keys:", early["has_duplicate_key"].sum())
print("Unique repeated keys:", len(repeated_keys))
print(
    "Share of rows affected:",
    round(100 * early["has_duplicate_key"].mean(), 3),
    "%",
)

print("\n=== MODE HOURS: ALL ROWS VS DUPLICATE-KEY ROWS ===")

print("\nAll rows:")
print(early[mode_cols].sum().round(2).to_string())

print("\nRows belonging to repeated keys:")
print(
    early.loc[early["has_duplicate_key"], mode_cols]
    .sum()
    .round(2)
    .to_string()
)

print("\n=== PORT-LEVEL IMPACT ===")

all_port = early.groupby("Port")[mode_cols].sum()
duplicate_port = (
    early.loc[early["has_duplicate_key"]]
    .groupby("Port")[mode_cols]
    .sum()
)

impact = (
    duplicate_port
    .div(all_port.replace(0, float("nan")))
    .mul(100)
    .round(2)
)

print("\nPercentage of each port's recorded hours from duplicate-key rows:")
print(impact.to_string())

print("\n=== DUPLICATES BY YEAR ===")
dup_df = early.loc[early["has_duplicate_key"]].copy()
dup_df["Year"] = pd.to_datetime(
    dup_df["YearMonth"]
).dt.year

print(dup_df.groupby("Year").size().to_string())

print("\n=== DUPLICATES BY PORT ===")
print(dup_df.groupby("Port").size().to_string())