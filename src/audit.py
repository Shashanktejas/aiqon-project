
from pathlib import Path

import pandas as pd

# Project root is the parent directory of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

FILES = {
    "dwell_early": "monthly_dwell_2009_2014.csv",
    "dwell_recent": "monthly_dwell_2015_2025.csv",
    "mode_early": "monthly_mode_time_2009_2014.csv",
    "mode_recent": "monthly_mode_time_2015_2025.csv",
}

DWELL_COLUMNS = {
    "Port",
    "YearMonth",
    "MeanDwellDays",
    "MedianDwellDays",
    "UniqueVessels",
}

MODE_COLUMNS = {
    "MMSI",
    "Port",
    "YearMonth",
    "anchor_hours",
    "berth_hours",
    "manoeuvre_hours",
    "transit_hours",
    "unknown_hoteling_hours",
    "total_mode_hours",
}


def inspect_file(name, filename):
    path = DATA_DIR / filename

    print("\n" + "=" * 75)
    print(f"DATASET: {filename}")
    print("=" * 75)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return None

    df = pd.read_csv(path, low_memory=False)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Column names:", df.columns.tolist())

    # Check the expected schema.
    expected = DWELL_COLUMNS if name.startswith("dwell") else MODE_COLUMNS
    missing_columns = expected - set(df.columns)

    if missing_columns:
        print("WARNING: Missing expected columns:", sorted(missing_columns))
    else:
        print("Expected columns: OK")

    # Check missing data.
    print("\nMissing values:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "None")

    # Check the month format and date coverage.
    if "YearMonth" in df.columns:
        df["YearMonth"] = df["YearMonth"].astype("string").str.strip()
        parsed_dates = pd.to_datetime(
            df["YearMonth"], format="%Y-%m", errors="coerce"
        )

        print("\nDate range:", parsed_dates.min(), "to", parsed_dates.max())
        print("Invalid month values:", int(parsed_dates.isna().sum()))
        print("Distinct months:", parsed_dates.nunique())

    # Check duplicate records using the expected key.
    if name.startswith("dwell"):
        key = ["Port", "YearMonth"]
    else:
        key = ["MMSI", "Port", "YearMonth"]

    if set(key).issubset(df.columns):
        duplicates = df.duplicated(subset=key).sum()
        print(f"\nDuplicate records by {key}: {duplicates}")

    # Check port coverage.
    if "Port" in df.columns:
        print("\nRecords by port:")
        print(df["Port"].value_counts(dropna=False).to_string())

    # Check numeric values.
    numeric_columns = (
        [
            "MeanDwellDays",
            "MedianDwellDays",
            "UniqueVessels",
        ]
        if name.startswith("dwell")
        else [
            "anchor_hours",
            "berth_hours",
            "manoeuvre_hours",
            "transit_hours",
            "unknown_hoteling_hours",
            "total_mode_hours",
        ]
    )

    existing_numeric = [
        column for column in numeric_columns if column in df.columns
    ]

    print("\nNumeric summary:")
    print(df[existing_numeric].describe().round(3).to_string())

    if existing_numeric:
        negative_counts = (df[existing_numeric] < 0).sum()
        print("\nNegative values:")
        print(negative_counts[negative_counts > 0].to_string()
              if (negative_counts > 0).any() else "None")

    return df


def main():
    datasets = {}

    for name, filename in FILES.items():
        datasets[name] = inspect_file(name, filename)

    print("\n" + "=" * 75)
    print("SCHEMA COMPARISON")
    print("=" * 75)

    for early_name, recent_name in [
        ("dwell_early", "dwell_recent"),
        ("mode_early", "mode_recent"),
    ]:
        early = datasets[early_name]
        recent = datasets[recent_name]

        if early is None or recent is None:
            continue

        early_columns = set(early.columns)
        recent_columns = set(recent.columns)

        print(f"\n{early_name} vs {recent_name}")
        print("Same columns:", early_columns == recent_columns)
        print("Only in earlier file:", sorted(early_columns - recent_columns))
        print("Only in recent file:", sorted(recent_columns - early_columns))

    print("\nAudit complete. Review warnings before modeling.")


if __name__ == "__main__":
    main()
