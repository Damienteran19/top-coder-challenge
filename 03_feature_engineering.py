"""
Session 3 – Feature engineering.

Creates a few simple, business‑motivated features and saves:
- train_features.csv
- test_features.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Filepath Constants
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"

# List of Features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    days = df["trip_duration_days"].replace(0, 1)

    df["receipts_per_day"] = df["total_receipts_amount"] / days
    df["miles_per_day"] = df["miles_traveled"] / days
    df["log_receipts"] = np.log1p(df["total_receipts_amount"])
    df["log_miles"] = np.log1p(df["miles_traveled"])
    df["is_week_plus"] = (df["trip_duration_days"] >= 7).astype(int)
    df["is_long_miles"] = (df["miles_traveled"] > 500).astype(int)

    return df


def main() -> None:
    # Filepaths
    train_path = PROC / "train_data.csv"
    test_path = PROC / "test_data.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run 01_project_setup.py first to create train/test CSVs")

    # Load raw features
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Add derived features
    train_fe = add_features(train)
    test_fe = add_features(test)

    # Save derived features
    train_fe.to_csv(PROC / "train_features.csv", index=False)
    test_fe.to_csv(PROC / "test_features.csv", index=False)

    # Display results
    print("Saved train_features.csv and test_features.csv in data/processed/")
    print("Feature columns:", [c for c in train_fe.columns if c != "reimbursement"])

# Main Execution
if __name__ == "__main__":
    main()
