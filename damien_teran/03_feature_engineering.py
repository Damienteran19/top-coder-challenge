# 03_feature_engineering.py
# Feature Engineering for Legacy Reimbursement System
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
FEAT = ROOT / "data" / "features"
FEAT.mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)

# Load data
train = pd.read_csv(PROC / "train_data.csv")
test = pd.read_csv(PROC / "test_data.csv")

print(f"Loaded train: {train.shape}, test: {test.shape}")

# ------------------------------------------------------------------
# 1. Base Features (from input)
# ------------------------------------------------------------------
def add_base_features(df):
    df = df.copy()
    # Ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

train = add_base_features(train)
test = add_base_features(test)

# ------------------------------------------------------------------
# 2. Business-Derived Features (PRD & INTERVIEWS)
# ------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    
    # --- Mileage Features ---
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days'].replace(0, 1)
    df['high_mileage'] = (df['miles_traveled'] > 300).astype(int)
    df['long_trip'] = (df['trip_duration_days'] > 7).astype(int)
    
    # --- Receipt Features ---
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days'].replace(0, 1)
    df['high_receipts'] = (df['total_receipts_amount'] > 800).astype(int)
    df['receipts_to_miles_ratio'] = df['total_receipts_amount'] / df['miles_traveled'].replace(0, 1)
    
    # --- Rate-Based Features ---
    df['reimb_per_mile'] = df['reimbursement'] / df['miles_traveled'].replace(0, 1)
    df['reimb_per_day'] = df['reimbursement'] / df['trip_duration_days'].replace(0, 1)
    df['reimb_per_receipt'] = df['reimbursement'] / df['total_receipts_amount'].replace(0, 1)
    
    # --- Efficiency & Compliance ---
    df['mileage_efficiency'] = df['miles_traveled'] / df['reimbursement'].replace(0, 1)
    df['receipt_efficiency'] = df['total_receipts_amount'] / df['reimbursement'].replace(0, 1)
    
    # --- Threshold & Cap Features ---
    df['near_receipt_cap'] = ((df['total_receipts_amount'] > 700) & (df['total_receipts_amount'] <= 800)).astype(int)
    df['over_receipt_cap'] = (df['total_receipts_amount'] > 800).astype(int)
    df['under_100_miles'] = (df['miles_traveled'] < 100).astype(int)
    
    # --- Interaction Features ---
    df['miles_x_duration'] = df['miles_traveled'] * df['trip_duration_days']
    df['receipts_x_duration'] = df['total_receipts_amount'] * df['trip_duration_days']
    df['miles_x_receipts'] = df['miles_traveled'] * df['total_receipts_amount']
    
    # --- Binned Features ---
    df['miles_bin'] = pd.cut(df['miles_traveled'], bins=[0, 100, 300, 500, 1000], labels=['<100', '100-300', '300-500', '>500'])
    df['duration_bin'] = pd.cut(df['trip_duration_days'], bins=[0, 3, 7, 14], labels=['Short', 'Medium', 'Long'])
    df['receipts_bin'] = pd.cut(df['total_receipts_amount'], bins=[0, 300, 800, 1500, 3000], labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # One-hot encode bins
    df = pd.get_dummies(df, columns=['miles_bin', 'duration_bin', 'receipts_bin'], prefix=['miles', 'dur', 'rec'])
    
    # --- Statistical Features (from distribution) ---
    df['reimb_zscore'] = (df['reimbursement'] - df['reimbursement'].mean()) / df['reimbursement'].std()
    df['miles_zscore'] = (df['miles_traveled'] - df['miles_traveled'].mean()) / df['miles_traveled'].std()
    
    # --- Anomaly Flags ---
    df['anomaly_miles'] = ((df['miles_zscore'] > 2) | (df['miles_zscore'] < -2)).astype(int)
    df['anomaly_reimb'] = ((df['reimb_zscore'] > 2) | (df['reimb_zscore'] < -2)).astype(int)
    
    return df

print("Engineering features...")
train_fe = engineer_features(train)
test_fe = engineer_features(test)

print(f"Features created: {train_fe.shape[1] - 4} new + 4 original = {train_fe.shape[1]} total")

# ------------------------------------------------------------------
# 3. Save Feature Sets
# ------------------------------------------------------------------
train_fe.to_csv(FEAT / "train_features.csv", index=False)
test_fe.to_csv(FEAT / "test_features.csv", index=False)
print(f"Feature CSVs saved to data/features/")

# ------------------------------------------------------------------
# 4. Feature Importance Preview (Random Forest)
# ------------------------------------------------------------------
X_train = train_fe.drop(columns=['reimbursement'])
y_train = train_fe['reimbursement']

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X_train.columns)
top15 = importances.abs().sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(x=top15.values, y=top15.index, palette="viridis")
plt.title("Top 15 Feature Importances (Preview)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(ROOT / "results" / "feature_importance_preview.png", dpi=150)
plt.close()

# Save top features
top15.to_csv(ROOT / "results" / "top_features.csv")
print("Feature importance plot saved.")

# ------------------------------------------------------------------
# 5. Correlation Heatmap of Top Features
# ------------------------------------------------------------------
top_cols = top15.index.tolist() + ['reimbursement']
corr_top = train_fe[top_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_top, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True)
plt.title("Correlation of Top 15 Features + Target")
plt.tight_layout()
plt.savefig(ROOT / "results" / "feature_correlation_heatmap.png", dpi=150)
plt.close()
print("Feature correlation heatmap saved.")

# ------------------------------------------------------------------
# 6. Summary Report
# ------------------------------------------------------------------
report = f"""
FEATURE ENGINEERING SUMMARY
==========================

Total Features: {train_fe.shape[1]}
Original: 4
Engineered: {train_fe.shape[1] - 4}

Top 3 Features (RF Importance):
1. {top15.index[0]} ({top15.iloc[0]:.4f})
2. {top15.index[1]} ({top15.iloc[1]:.4f})
3. {top15.index[2]} ({top15.iloc[2]:.4f})

Key Engineered Features:
- reimb_per_mile, reimb_per_day
- receipts_per_day, high_receipts
- interaction terms, binned categories
- anomaly detection

Next Steps:
- Run 04_baseline_models.py
- Compare MAE: baseline vs engineered

Generated: {pd.Timestamp('now').strftime('%Y-%m-%d %H:%M')}
"""

with open(ROOT / "results" / "FEATURE_ENGINEERING_REPORT.txt", "w") as f:
    f.write(report)

print("\nFEATURE ENGINEERING COMPLETE!")
print("Check results/ for plots and reports.")
print("Next: python 04_baseline_models.py")