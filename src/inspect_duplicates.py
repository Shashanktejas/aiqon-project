
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "data" / "processed" / "duplicate_mode_keys_review.csv"

df = pd.read_csv(path)

key = ["MMSI", "Port", "YearMonth"]
value_columns = [
    "anchor_hours",
    "berth_hours",
    "manoeuvre_hours",
    "transit_hours",
    "unknown_hoteling_hours",
    "total_mode_hours",
]

print("Rows in duplicate review:", len(df))
print("Unique repeated keys:", df.groupby(key).ngroups)

# Show the first repeated keys and all their associated records.
print("\nFIRST REPEATED RECORDS:")
print(
    df.sort_values(key)
      .groupby(key, sort=False)
      .head(3)
      .head(20)
      .to_string(index=False)
)

# Count groups with identical measurements versus conflicting measurements.
group_stats = (
    df.groupby(key)[value_columns]
      .nunique(dropna=False)
)

identical_measurements = group_stats.eq(1).all(axis=1)
conflicting_measurements = ~identical_measurements

print("\nKEY GROUPS WITH IDENTICAL MEASUREMENTS:",
      int(identical_measurements.sum()))

print("KEY GROUPS WITH DIFFERENT MEASUREMENTS:",
      int(conflicting_measurements.sum()))

print("\nNUMBER OF ROWS PER REPEATED KEY:")
print(df.groupby(key).size().value_counts().sort_index())

# Save only the groups that need closer inspection.
conflict_keys = group_stats.index[conflicting_measurements]
conflicts = (
    df.set_index(key)
      .loc[conflict_keys]
      .reset_index()
)

output = ROOT / "data" / "processed" / "conflicting_mode_records.csv"
conflicts.to_csv(output, index=False)

print("\nSaved conflicting records to:", output)
