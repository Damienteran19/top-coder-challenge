"""
Session 5 – A couple of stronger models.

Trains:
- RandomForestRegressor
- GradientBoostingRegressor

Saves their raw performance so we can decide what to tune later.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Filepath Constants
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS_DIR = ROOT / "models" / "saved"

# Ensure RESULTS and MODEL_DIR paths exist
RESULTS.mkdir(exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Calculation of error metrics between y_true and y_pred for advanced models
def project_metrics(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    exact = np.mean(diff <= 0.01) * 100
    close = np.mean(diff <= 1.00) * 100
    return {
        "mae": mae,
        "rmse": rmse,
        "exact_pct": exact,
        "close_pct": close,
    }


def main() -> None:
    # Filepaths
    train_path = PROC / "train_features.csv"
    test_path = PROC / "test_features.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run 03_feature_engineering.py first to create feature CSVs")

    # Load derived features
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Isolate non-target features
    feature_cols = [c for c in train.columns if c != "reimbursement"]

    # Break both train and test partitions into X (descriptive) and y (target)
    X_train = train[feature_cols]
    y_train = train["reimbursement"]
    X_test = test[feature_cols]
    y_test = test["reimbursement"]

    # Initialize advanced model results to print
    rows = []

    # Random Forest (moderate size)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rows.append({"model": "random_forest", **project_metrics(y_test, y_pred_rf)})

    # Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    rows.append({"model": "gradient_boosting", **project_metrics(y_test, y_pred_gb)})

    # Display results and save to 05_advanced_results.csv
    df_results = pd.DataFrame(rows)
    df_results.to_csv(RESULTS / "05_advanced_results.csv", index=False)
    print(df_results)

# Main Execution
if __name__ == "__main__":
    main()
