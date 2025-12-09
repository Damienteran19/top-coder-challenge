#!/usr/bin/env python3
"""
Command‑line prediction entry point.

Usage:
    ./predict.py <trip_duration_days> <miles_traveled> <total_receipts_amount>

This script is designed to be called by `run.sh` and by the instructor's
evaluation scripts. It simply parses the three inputs, loads the production
predictor, and prints a single reimbursement value.
"""
import sys

from pathlib import Path

from _07_production_pipeline import ReimbursementPredictor


def main(argv) -> None:
    # Ensure correct number of arguments were passed
    if len(argv) != 4:
        sys.stderr.write(
            "Usage: predict.py <trip_duration_days> <miles_traveled> <total_receipts_amount>\n"
        )
        # Still print something numeric so automated scripts do not crash
        print("0.00")
        return

    # Ensure arguments are of the correct data type 
    try:
        days = float(argv[1])
        miles = float(argv[2])
        receipts = float(argv[3])
    except ValueError:
        sys.stderr.write("All three arguments must be numeric.\n")
        print("0.00")
        return

    # Make a prdiction based on arguments
    predictor = ReimbursementPredictor()
    value = predictor.predict_one(days, miles, receipts)
    # Ensure exactly two decimals
    print(f"{value:.2f}")

# Main Execution
if __name__ == "__main__":
    main(sys.argv)
