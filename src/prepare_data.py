
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = DATA / "processed"

OUTPUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load and combine dwell datasets
# --------------------------------------------------

dwell_early = pd.read_csv(DATA / "monthly_dwell_2009_2014.csv")
dwell_recent = pd.read_csv(DATA / "monthly_dwell_2015_2025.csv")

dwell = pd.concat(
    [dwell_early, dwell_recent],
    ignore_index=True,
    sort=False,
)

dwell["Port"] = dwell["Port"].astype("string").str.strip()
dwell["YearMonth"] = pd.to_datetime(
    dwell["YearMonth"], format="%Y-%m", errors="raise"
)

print("Dwell records:", len(dwell))
print("Duplicate port-month keys:",
      dwell.duplicated(["Port", "YearMonth"]).sum())

# The port-month key must be unique.
if dwell.duplicated(["Port", "YearMonth"]).any():
    raise ValueError("Duplicate dwell keys found. Inspect before continuing.")

dwell = dwell.sort_values(["Port", "YearMonth"]).reset_index(drop=True)

# --------------------------------------------------
# 2. Load and combine vessel mode-time datasets
# --------------------------------------------------

mode_early = pd.read_csv(DATA / "monthly_mode_time_2009_2014.csv")
mode_recent = pd.read_csv(DATA / "monthly_mode_time_2015_2025.csv")

mode = pd.concat(
    [mode_early, mode_recent],
    ignore_index=True,
    sort=False,
)

mode["Port"] = mode["Port"].astype("string").str.strip()
mode["YearMonth"] = pd.to_datetime(
    mode["YearMonth"], format="%Y-%m", errors="raise"
)

key = ["MMSI", "Port", "YearMonth"]
hour_columns = [
    "anchor_hours",
    "berth_hours",
    "manoeuvre_hours",
    "transit_hours",
    "unknown_hoteling_hours",
    "total_mode_hours",
]

# Remove only completely identical duplicate rows.
exact_duplicates = int(mode.duplicated().sum())
print("\nExact duplicate rows:", exact_duplicates)

mode = mode.drop_duplicates().copy()

# Investigate repeated keys that have different values.
conflicting = mode.loc[
    mode.duplicated(key, keep=False)
].sort_values(key)

print("Rows involved in repeated vessel-month keys:",
      len(conflicting))

if not conflicting.empty:
    conflicting.to_csv(
        OUTPUT / "duplicate_mode_keys_review.csv", index=False
    )
    raise ValueError(
        "Repeated vessel-month keys remain. Review "
        "data/processed/duplicate_mode_keys_review.csv "
        "before deciding how to resolve them."
    )

# Validate operational hours.
if mode[hour_columns].isna().any().any():
    raise ValueError("Missing operational-hour values found.")

if (mode[hour_columns] < 0).any().any():
    raise ValueError("Negative operational-hour values found.")

print("Mode records after exact deduplication:", len(mode))

# --------------------------------------------------
# 3. Identify missing port-month observations
# --------------------------------------------------

all_months = pd.date_range(
    start="2009-01-01",
    end="2025-12-01",
    freq="MS",
)

ports = sorted(dwell["Port"].dropna().unique())

expected = pd.MultiIndex.from_product(
    [ports, all_months],
    names=["Port", "YearMonth"],
).to_frame(index=False)

observed = (
    mode[["Port", "YearMonth"]]
    .drop_duplicates()
)

missing_mode_months = (
    expected.merge(
        observed,
        on=["Port", "YearMonth"],
        how="left",
        indicator=True,
    )
    .query('_merge == "left_only"')
    .drop(columns="_merge")
)

missing_mode_months.to_csv(
    OUTPUT / "missing_mode_port_months.csv", index=False
)

print("\nMissing mode port-month combinations:",
      len(missing_mode_months))
print("Saved missing-month report.")

# --------------------------------------------------
# 4. Aggregate vessel records to port-month features
# --------------------------------------------------

port_month = (
    mode.groupby(["Port", "YearMonth"], as_index=False)
    .agg(
        observed_vessels=("MMSI", "nunique"),
        total_anchor_hours=("anchor_hours", "sum"),
        mean_anchor_hours=("anchor_hours", "mean"),
        total_berth_hours=("berth_hours", "sum"),
        mean_berth_hours=("berth_hours", "mean"),
        total_manoeuvre_hours=("manoeuvre_hours", "sum"),
        total_transit_hours=("transit_hours", "sum"),
        total_unknown_hours=("unknown_hoteling_hours", "sum"),
        total_mode_hours=("total_mode_hours", "sum"),
    )
)

port_month["anchor_time_share"] = (
    port_month["total_anchor_hours"]
    / port_month["total_mode_hours"].replace(0, float("nan"))
)

# --------------------------------------------------
# 5. Join observed features to dwell outcomes
# --------------------------------------------------

combined = dwell.merge(
    port_month,
    on=["Port", "YearMonth"],
    how="left",
    validate="one_to_one",
    indicator=True,
)

print("\nDwell-to-mode join results:")
print(combined["_merge"].value_counts().to_string())

combined = combined.drop(columns="_merge")
combined = combined.sort_values(["Port", "YearMonth"])

# Do not fill missing mode features with zero:
# missing observations are not proof of zero vessel activity.
combined.to_csv(
    OUTPUT / "port_month_combined.csv", index=False
)

print("\nSaved:", OUTPUT / "port_month_combined.csv")
print("Combined shape:", combined.shape)
print("\nColumns:")
print(combined.columns.tolist())
print("\nFirst 5 rows:")
print(combined.head().to_string(index=False))
