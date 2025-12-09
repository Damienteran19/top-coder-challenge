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

# Calculate amount of added reimbursement based on a 2-tiered system
def calc_added_reimbursement(feature, threshold, tier_1_rate, tier_2_rate) -> float:
    # Initialize
    added_reimb = 0
    # Store total value of feature
    feat_remaining = feature
    # Find how much of the feature lies in the first tier
    feat_in_T1 = min([feat_remaining, threshold])
    # Calculae reimbursement from values in first tier
    added_reimb += feat_in_T1 * tier_1_rate
    # Reconsider remaining feature values
    feat_remaining -= feat_in_T1
    # Calculae reimbursement from values in first tier
    added_reimb += feat_remaining * tier_2_rate
    # Return additional reimbursement
    return added_reimb

def main(argv) -> None:
    # Ensure correct number of arguments were passed
    if len(argv) != 4:
        sys.stderr.write(
            "Usage: business_logic.py <trip_duration_days> <miles_traveled> <total_receipts_amount>\n"
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

    # Progrssive Reimbursement Rate Constants
    RECEIPT_THRESHOLD = 800
    RECEIPT_TIER_1_RATE = 1.0
    RECEIPT_TIER_2_RATE = .075

    MILE_THRESHOLD = 500
    MILE_TIER_1_RATE = .58
    MILE_TIER_2_RATE = .14

    DAY_THRESHOLD = 7
    DAY_TIER_1_RATE = 100.0
    DAY_TIER_2_RATE = 25.0

    # Make a prediction based on business logic inference
    reimburse = 0
    reimburse += calc_added_reimbursement(receipts, RECEIPT_THRESHOLD, RECEIPT_TIER_1_RATE, RECEIPT_TIER_2_RATE)
    reimburse += calc_added_reimbursement(miles, MILE_THRESHOLD, MILE_TIER_1_RATE, MILE_TIER_2_RATE)
    reimburse += calc_added_reimbursement(days, DAY_THRESHOLD, DAY_TIER_1_RATE, DAY_TIER_2_RATE)
    
    # Ensure exactly two decimals
    print(f"{reimburse:.2f}")


if __name__ == "__main__":
    main(sys.argv)