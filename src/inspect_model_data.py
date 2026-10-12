from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "data" / "processed" / "model_dataset.csv"

df = pd.read_csv(path)
df["YearMonth"] = pd.to_datetime(df["YearMonth"])

print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== TARGET DISTRIBUTION BY PORT ===")
print(pd.crosstab(
    df["Port"],
    df["TargetHighDwellNextMonth"],
    normalize="index"
).round(3))

print("\n=== TARGET RATE BY YEAR ===")
df["Year"] = df["YearMonth"].dt.year
print(df.groupby("Year")["TargetHighDwellNextMonth"].mean().round(3))

print("\n=== FEATURE CORRELATIONS WITH TARGET ===")
numeric = df.select_dtypes(include="number")
print(
    numeric.corr()["TargetHighDwellNextMonth"]
    .drop("TargetHighDwellNextMonth")
    .sort_values(ascending=False)
    .round(3)
)

print("\n=== MISSING FEATURES ===")
print(
    df.isna().sum()
    .sort_values(ascending=False)
    .head(15)
)

print("\n=== TARGET AND FEATURE TIMELINE ===")
print(
    df.groupby("YearMonth")
    .agg(
        ports=("Port", "nunique"),
        mean_dwell=("MeanDwellDays", "mean"),
        next_month_high_rate=("TargetHighDwellNextMonth", "mean"),
    )
    .tail(18)
    .round(3)
)
