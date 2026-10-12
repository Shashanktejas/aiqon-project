
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

early = pd.read_csv(DATA / "monthly_mode_time_2009_2014.csv")
recent = pd.read_csv(DATA / "monthly_mode_time_2015_2025.csv")

key = ["MMSI", "Port", "YearMonth"]

print("EARLY DATASET")
print("Rows:", len(early))
print("MMSI zero rows:", (early["MMSI"] == 0).sum())
print("MMSI missing rows:", early["MMSI"].isna().sum())

print("\nREPEATED KEY COUNTS")
counts = early.groupby(key, dropna=False).size()
print(counts.value_counts().sort_index())

print("\nDUPLICATE GROUPS WITH MMSI = 0")
zero = early.loc[early["MMSI"] == 0]
print("Rows:", len(zero))
print(zero.head(15).to_string(index=False))

print("\nDUPLICATE ROWS: SAMPLE")
repeated_keys = counts[counts > 1].index
sample_keys = repeated_keys[:10]

for values in sample_keys:
    mask = pd.Series(True, index=early.index)
    for col, value in zip(key, values):
        mask &= early[col].eq(value)

    print("\nKey:", values)
    print(early.loc[mask].to_string(index=False))

print("\nRECENT DATASET")
print("Rows:", len(recent))
print("MMSI zero rows:", (recent["MMSI"] == 0).sum())
print("Duplicate keys:", recent.duplicated(key).sum())
