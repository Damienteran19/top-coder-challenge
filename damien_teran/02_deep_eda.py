# 02_deep_eda.py
# Deep Exploratory Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
ROOT = Path(__file__).parent
train = pd.read_csv(ROOT / "data" / "processed" / "train_data.csv")

# Create results folder
(ROOT / "results").mkdir(exist_ok=True)

print("Train data shape:", train.shape)
print("\nFirst 5 rows:")
print(train.head())

# ------------------------------------------------------------------
# 1. Basic Statistics
# ------------------------------------------------------------------
desc = train.describe()
desc.to_csv(ROOT / "results" / "eda_summary_stats.csv")
print("\nSummary statistics saved to results/eda_summary_stats.csv")

# ------------------------------------------------------------------
# 2. Distribution Plots
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
cols = train.columns.tolist()

for i, col in enumerate(cols):
    if i >= 4:
        break
    sns.histplot(train[col], kde=True, ax=axes[i], bins=30, color="skyblue")
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel("")
plt.tight_layout()
plt.savefig(ROOT / "results" / "distribution_plots.png", dpi=150)
plt.close()
print("Distribution plots saved to results/distribution_plots.png")

# ------------------------------------------------------------------
# 3. Correlation Analysis
# ------------------------------------------------------------------
corr = train.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True, fmt=".3f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(ROOT / "results" / "correlation_heatmap.png", dpi=150)
plt.close()
print("Correlation heatmap saved to results/correlation_heatmap.png")

# Save correlation values sorted
corr_with_target = corr["reimbursement"].drop("reimbursement").abs().sort_values(ascending=False)
corr_with_target.to_csv(ROOT / "results" / "top_correlations.csv")
print("\nTop correlations with reimbursement:")
print(corr_with_target)

# ------------------------------------------------------------------
# 4. Pattern Discovery (Thresholds, Binning, Outliers)
# ------------------------------------------------------------------
print("\n" + "="*50)
print("PATTERN DISCOVERY")
print("="*50)

# Mileage vs Reimbursement scatter
plt.figure(figsize=(8, 6))
sns.scatterplot(data=train, x="miles_traveled", y="reimbursement", alpha=0.6)
plt.title("Reimbursement vs Miles Traveled")
plt.tight_layout()
plt.savefig(ROOT / "results" / "scatter_miles_vs_reimb.png", dpi=150)
plt.close()

# Receipts vs Reimbursement
plt.figure(figsize=(8, 6))
sns.scatterplot(data=train, x="total_receipts_amount", y="reimbursement", alpha=0.6, color="orange")
plt.title("Reimbursement vs Total Receipts")
plt.tight_layout()
plt.savefig(ROOT / "results" / "scatter_receipts_vs_reimb.png", dpi=150)
plt.close()

# Duration vs Reimbursement
plt.figure(figsize=(8, 6))
sns.boxplot(data=train, x="trip_duration_days", y="reimbursement")
plt.title("Reimbursement by Trip Duration (Days)")
plt.tight_layout()
plt.savefig(ROOT / "results" / "box_duration_vs_reimb.png", dpi=150)
plt.close()

print("Scatter & box plots saved.")

# ------------------------------------------------------------------
# 5. EDA Summary Report (Text)
# ------------------------------------------------------------------
report = f"""
LEGACY REIMBURSEMENT - EDA SUMMARY
==================================

Dataset: {len(train)} training samples
Features: {', '.join(train.columns.drop('reimbursement').tolist())}
Target: reimbursement

Key Insights:
-------------
1. Strongest predictor: {corr_with_target.index[0]} (r = {corr_with_target.iloc[0]:.3f})
2. Data range:
   - Miles: {train['miles_traveled'].min()} to {train['miles_traveled'].max()}
   - Receipts: ${train['total_receipts_amount'].min():.2f} to ${train['total_receipts_amount'].max():.2f}
   - Duration: {train['trip_duration_days'].min()} to {train['trip_duration_days'].max()} days
3. Potential non-linearity? Check scatter plots for thresholds.

Next Steps:
-----------
- Read PRD.md & INTERVIEWS.md for business rules
- Engineer features: rate per mile, per diem flags, receipt ratios
- Look for approval thresholds (e.g $500 cap?)

Generated on: {pd.Timestamp('now').strftime('%Y-%m-%d %H:%M')}
"""

with open(ROOT / "results" / "EDA_SUMMARY_REPORT.txt", "w") as f:
    f.write(report)

print("\nEDA SUMMARY REPORT saved to results/EDA_SUMMARY_REPORT.txt")
print("\nWeek 1 Session 2 COMPLETE!")
print("Review all files in results/ folder before moving to feature engineering.")