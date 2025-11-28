# 01_project_setup.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Load and flatten JSON
# ------------------------------------------------------------------
with open(RAW / "public_cases.json") as f:
    data = json.load(f)

# Normalize nested structure
df = pd.json_normalize(data, sep="_")

print("Columns after flattening:")
print(df.columns.tolist())

# ------------------------------------------------------------------
# Expected columns (based on your sample)
# ------------------------------------------------------------------
# From your snippet: input_trip_duration_days, input_miles_traveled
# Target is likely: output_reimbursement or just reimbursement
# Let's find the reimbursement column
target_candidates = [col for col in df.columns if "reimb" in col.lower() or "output" in col]
print(f"Possible target columns: {target_candidates}")

# AUTO DETECT TARGET (first match)
target_col = None
for candidate in ["output_reimbursement", "reimbursement", "total_reimbursement"]:
    if candidate in df.columns:
        target_col = candidate
        break
if not target_col and target_candidates:
    target_col = target_candidates[0]

if not target_col:
    raise ValueError("Could not find reimbursement column! Check JSON structure.")

print(f"Using target column: {target_col}")

# ------------------------------------------------------------------
# Feature columns (from input_)
# ------------------------------------------------------------------
feature_cols = [col for col in df.columns if col.startswith("input_")]
print(f"Using {len(feature_cols)} input features: {feature_cols[:5]}...")

# ------------------------------------------------------------------
# Clean and prepare
# ------------------------------------------------------------------
df = df[feature_cols + [target_col]].copy()

# Rename input_ columns to clean names
rename_dict = {}
for col in feature_cols:
    clean_name = col.replace("input_", "").lower()
    rename_dict[col] = clean_name

df.rename(columns=rename_dict, inplace=True)
df.rename(columns={target_col: "reimbursement"}, inplace=True)

# Convert to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with any missing values
df.dropna(inplace=True)
print(f"After cleaning: {len(df)} rows")

# ------------------------------------------------------------------
# Train / Test Split
# ------------------------------------------------------------------
train, test = train_test_split(df, test_size=250, train_size=750, random_state=42, shuffle=True)

train.to_csv(PROC / "train_data.csv", index=False)
test.to_csv(PROC / "test_data.csv", index=False)

print(f"Train saved: {len(train)} rows")
print(f"Test saved: {len(test)} rows")

# ------------------------------------------------------------------
# Initial Plots
# ------------------------------------------------------------------
plot_cols = [c for c in ["miles_traveled", "trip_duration_days", "reimbursement"] if c in df.columns]
if len(plot_cols) >= 3:
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(plot_cols, 1):
        plt.subplot(2, 3, i)
        sns.histplot(train[col], kde=True, bins=30)
        plt.title(col)
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "initial_distributions.png")
    plt.close()
    print("Plots saved to results/initial_distributions.png")
else:
    print("Not enough columns for plotting.")