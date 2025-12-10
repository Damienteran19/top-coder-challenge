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
# 2. Business-Derived Features (NO TARGET LEAKAGE)
# ------------------------------------------------------------------
def engineer_features(df):
    """
    Create features using ONLY the 3 input variables.
    NO features derived from the target (output/reimbursement).
    """
    df = df.copy()
    
    # --- Basic Rate Features ---
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days'].replace(0, 1)
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days'].replace(0, 1)
    df['receipts_per_mile'] = df['total_receipts_amount'] / df['miles_traveled'].replace(0, 1)
    
    # --- Standard Reimbursement Estimates ---
    # Based on typical IRS rates and per diem
    IRS_MILEAGE_RATE = 0.655  # 2024 standard
    TYPICAL_PER_DIEM = 150.0
    
    df['est_mileage_reimb'] = df['miles_traveled'] * IRS_MILEAGE_RATE
    df['est_per_diem'] = df['trip_duration_days'] * TYPICAL_PER_DIEM
    df['est_total_simple'] = df['total_receipts_amount'] + df['est_mileage_reimb'] + df['est_per_diem']
    
    # --- Threshold Features ---
    df['high_mileage'] = (df['miles_traveled'] > 300).astype(int)
    df['very_high_mileage'] = (df['miles_traveled'] > 500).astype(int)
    df['long_trip'] = (df['trip_duration_days'] > 7).astype(int)
    df['medium_trip'] = ((df['trip_duration_days'] > 3) & (df['trip_duration_days'] <= 7)).astype(int)
    df['overnight'] = (df['trip_duration_days'] > 1).astype(int)
    
    # --- Receipt Thresholds ---
    df['high_receipts'] = (df['total_receipts_amount'] > 800).astype(int)
    df['very_high_receipts'] = (df['total_receipts_amount'] > 1200).astype(int)
    df['low_receipts'] = (df['total_receipts_amount'] < 300).astype(int)
    df['near_receipt_cap'] = ((df['total_receipts_amount'] > 700) & (df['total_receipts_amount'] <= 800)).astype(int)
    df['over_receipt_cap'] = (df['total_receipts_amount'] > 800).astype(int)
    
    # --- Distance Categories ---
    df['under_100_miles'] = (df['miles_traveled'] < 100).astype(int)
    df['local_trip'] = (df['miles_traveled'] < 50).astype(int)
    df['regional_trip'] = ((df['miles_traveled'] >= 100) & (df['miles_traveled'] < 300)).astype(int)
    
    # --- Interaction Features ---
    df['miles_x_duration'] = df['miles_traveled'] * df['trip_duration_days']
    df['receipts_x_duration'] = df['total_receipts_amount'] * df['trip_duration_days']
    df['miles_x_receipts'] = df['miles_traveled'] * df['total_receipts_amount']
    
    # --- Ratio Features ---
    df['receipts_to_miles_ratio'] = df['total_receipts_amount'] / df['miles_traveled'].replace(0, 1)
    df['miles_to_days_ratio'] = df['miles_traveled'] / df['trip_duration_days'].replace(0, 1)
    
    # --- Polynomial Features ---
    df['miles_squared'] = df['miles_traveled'] ** 2
    df['duration_squared'] = df['trip_duration_days'] ** 2
    df['receipts_squared'] = df['total_receipts_amount'] ** 2
    
    df['miles_sqrt'] = np.sqrt(df['miles_traveled'])
    df['duration_sqrt'] = np.sqrt(df['trip_duration_days'])
    df['receipts_sqrt'] = np.sqrt(df['total_receipts_amount'])
    
    # --- Log Features (for exponential relationships) ---
    df['miles_log'] = np.log1p(df['miles_traveled'])
    df['duration_log'] = np.log1p(df['trip_duration_days'])
    df['receipts_log'] = np.log1p(df['total_receipts_amount'])
    
    # --- Binned Features (categorical) ---
    df['miles_bin'] = pd.cut(df['miles_traveled'], 
                              bins=[0, 100, 300, 500, 1000], 
                              labels=['<100', '100-300', '300-500', '>500'])
    df['duration_bin'] = pd.cut(df['trip_duration_days'], 
                                 bins=[0, 3, 7, 14], 
                                 labels=['Short', 'Medium', 'Long'])
    df['receipts_bin'] = pd.cut(df['total_receipts_amount'], 
                                 bins=[0, 300, 800, 1500, 3000], 
                                 labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # One-hot encode bins
    df = pd.get_dummies(df, columns=['miles_bin', 'duration_bin', 'receipts_bin'], 
                        prefix=['miles', 'dur', 'rec'])
    
    # --- Z-scores of INPUT features (not target) ---
    df['miles_zscore'] = (df['miles_traveled'] - df['miles_traveled'].mean()) / df['miles_traveled'].std()
    df['duration_zscore'] = (df['trip_duration_days'] - df['trip_duration_days'].mean()) / df['trip_duration_days'].std()
    df['receipts_zscore'] = (df['total_receipts_amount'] - df['total_receipts_amount'].mean()) / df['total_receipts_amount'].std()
    
    # --- Anomaly Flags (on INPUT features, not target) ---
    df['anomaly_miles'] = ((df['miles_zscore'] > 2) | (df['miles_zscore'] < -2)).astype(int)
    df['anomaly_duration'] = ((df['duration_zscore'] > 2) | (df['duration_zscore'] < -2)).astype(int)
    df['anomaly_receipts'] = ((df['receipts_zscore'] > 2) | (df['receipts_zscore'] < -2)).astype(int)
    
    # --- Business Logic Combinations ---
    df['weighted_estimate_v1'] = (df['total_receipts_amount'] * 0.8 + 
                                  df['est_mileage_reimb'] + 
                                  df['trip_duration_days'] * 50)
    
    df['weighted_estimate_v2'] = (df['total_receipts_amount'] * 1.0 + 
                                  df['miles_traveled'] * 0.5 + 
                                  df['trip_duration_days'] * 100)
    
    return df

print("Engineering features (NO target leakage)...")
train_fe = engineer_features(train)
test_fe = engineer_features(test)

print(f"Features created: {train_fe.shape[1] - 4} new + 4 original = {train_fe.shape[1]} total")

# ------------------------------------------------------------------
# 3. Save Feature Sets
# ------------------------------------------------------------------
train_fe.to_csv(FEAT / "train_features.csv", index=False)
test_fe.to_csv(FEAT / "test_features.csv", index=False)
train_fe.to_csv(PROC / "train_features.csv", index=False)
test_fe.to_csv(PROC / "test_features.csv", index=False)
print(f"Feature CSVs saved to data/features/ and data/processed/")

# ------------------------------------------------------------------
# 4. Feature Importance Preview (Random Forest)
# ------------------------------------------------------------------
X_train = train_fe.drop(columns=['output'])
y_train = train_fe['output']

# Select only numeric columns
X_train_numeric = X_train.select_dtypes(include=[np.number])

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_numeric, y_train)

importances = pd.Series(rf.feature_importances_, index=X_train_numeric.columns)
top15 = importances.abs().sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(x=top15.values, y=top15.index, palette="viridis")
plt.title("Top 15 Feature Importances (Preview - No Target Leakage)")
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
top_cols = top15.index.tolist() + ['output']
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
FEATURE ENGINEERING SUMMARY (FIXED - NO TARGET LEAKAGE)
========================================================

Total Features: {train_fe.shape[1]}
Original: 4
Engineered: {train_fe.shape[1] - 4}

Top 3 Features (RF Importance):
1. {top15.index[0]} ({top15.iloc[0]:.4f})
2. {top15.index[1]} ({top15.iloc[1]:.4f})
3. {top15.index[2]} ({top15.iloc[2]:.4f})

Key Engineered Features:
[OK] Rate features (per day, per mile)
[OK] Standard estimates (IRS rate, per diem)
[OK] Threshold indicators (high receipts, long trips)
[OK] Interaction terms (miles × days, receipts × duration)
[OK] Polynomial features (squared, sqrt, log)
[OK] Z-scores and anomaly detection on INPUTS only

IMPORTANT: All features use ONLY the 3 input variables.
NO features derived from target (output/reimbursement).

This ensures the model can make predictions on new data!

Next Steps:
- Run 04_baseline_models.py
- Run 05_advanced_models.py
- Run 06_tuning_and_ensembles.py
- Session 07 will now work!

Generated: {pd.Timestamp('now').strftime('%Y-%m-%d %H:%M')}
"""

with open(ROOT / "results" / "FEATURE_ENGINEERING_REPORT.txt", "w") as f:
    f.write(report)

print("\n" + "="*70)
print("FEATURE ENGINEERING COMPLETE!")
print("="*70)
print("[OK] NO target leakage")
print("[OK] All features use only 3 inputs")
print("[OK] Production-ready feature engineering")
print(f"[OK] Created {train_fe.shape[1] - 4} features")
print("\nCheck results/ for plots and reports.")
print("Next: python 04_baseline_models.py")