"""
Session 6: Hyperparameter Tuning and Ensemble Methods
CSCI/DASC 6020 - Machine Learning Team Project

This script performs hyperparameter tuning and creates ensemble models
to maximize prediction accuracy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import joblib
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EVALUATION (reuse)
# ============================================================================

class ReimbursementEvaluator:
    """Custom evaluator for reimbursement predictions"""
    
    def __init__(self, tolerance_exact=0.01, tolerance_close=1.00):
        self.tolerance_exact = tolerance_exact
        self.tolerance_close = tolerance_close
    
    def evaluate(self, y_true, y_pred):
        """Evaluate predictions using project metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        diff = np.abs(y_true - y_pred)
        
        exact_matches = np.sum(diff <= self.tolerance_exact)
        exact_pct = exact_matches / len(y_true) * 100
        
        close_matches = np.sum(diff <= self.tolerance_close)
        close_pct = close_matches / len(y_true) * 100
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            'exact_matches': exact_matches,
            'exact_pct': exact_pct,
            'close_matches': close_matches,
            'close_pct': close_pct,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'avg_error': np.mean(diff),
            'max_error': np.max(diff),
            'n_samples': len(y_true)
        }
    
    def print_results(self, results, model_name="Model"):
        """Print evaluation results"""
        print(f"\n{model_name} Results:")
        print(f"  Exact (±$0.01): {results['exact_matches']}/{results['n_samples']} ({results['exact_pct']:.2f}%)")
        print(f"  Close (±$1.00): {results['close_matches']}/{results['n_samples']} ({results['close_pct']:.2f}%)")
        print(f"  MAE: ${results['mae']:.2f}")
        print(f"  RMSE: ${results['rmse']:.2f}")
        print(f"  R²: {results['r2']:.4f}")

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def tune_random_forest(X_train, y_train):
    """
    Hyperparameter tuning for Random Forest
    """
    print("\n" + "="*80)
    print("TUNING RANDOM FOREST")
    print("="*80)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    print(f"\nParameter grid: {sum([len(v) for v in param_grid.values()])} combinations")
    
    # Base model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Grid search with CV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    print("Running grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV MAE: ${-grid_search.best_score_:.2f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def tune_xgboost(X_train, y_train):
    """
    Hyperparameter tuning for XGBoost
    """
    print("\n" + "="*80)
    print("TUNING XGBOOST")
    print("="*80)
    
    # Define parameter grid (smaller for speed)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    print(f"\nParameter grid: {sum([len(v) for v in param_grid.values()])} combinations")
    print("Using randomized search for efficiency...")
    
    # Base model
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    # Randomized search (faster than grid search)
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=20,  # Try 20 random combinations
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("Running randomized search...")
    random_search.fit(X_train, y_train)
    
    print(f"\nBest parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV MAE: ${-random_search.best_score_:.2f}")
    
    return random_search.best_estimator_, random_search.best_params_

def tune_gradient_boosting(X_train, y_train):
    """
    Hyperparameter tuning for Gradient Boosting
    """
    print("\n" + "="*80)
    print("TUNING GRADIENT BOOSTING")
    print("="*80)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [5, 10, 15]
    }
    
    print(f"\nParameter grid: {sum([len(v) for v in param_grid.values()])} combinations")
    print("Using randomized search for efficiency...")
    
    # Base model
    gb = GradientBoostingRegressor(random_state=42)
    
    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("Running randomized search...")
    random_search.fit(X_train, y_train)
    
    print(f"\nBest parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV MAE: ${-random_search.best_score_:.2f}")
    
    return random_search.best_estimator_, random_search.best_params_

# ============================================================================
# ENSEMBLE METHODS
# ============================================================================

def create_voting_ensemble(tuned_models, X_train, y_train):
    """
    Create a voting ensemble of tuned models
    """
    print("\n" + "="*80)
    print("CREATING VOTING ENSEMBLE")
    print("="*80)
    
    # Create list of (name, model) tuples
    estimators = [
        ('rf', tuned_models['random_forest']),
        ('xgb', tuned_models['xgboost']),
        ('gb', tuned_models['gradient_boosting'])
    ]
    
    # Create voting regressor
    voting = VotingRegressor(estimators=estimators, n_jobs=-1)
    
    print("Training voting ensemble...")
    voting.fit(X_train, y_train)
    
    print("✓ Voting ensemble trained")
    
    return voting

def create_stacking_ensemble(tuned_models, X_train, y_train):
    """
    Create a stacking ensemble with Ridge meta-learner
    """
    print("\n" + "="*80)
    print("CREATING STACKING ENSEMBLE")
    print("="*80)
    
    # Base models
    estimators = [
        ('rf', tuned_models['random_forest']),
        ('xgb', tuned_models['xgboost']),
        ('gb', tuned_models['gradient_boosting'])
    ]
    
    # Meta-learner (final estimator)
    meta_learner = Ridge(alpha=1.0)
    
    # Create stacking regressor
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    print("Training stacking ensemble...")
    stacking.fit(X_train, y_train)
    
    print("✓ Stacking ensemble trained")
    
    return stacking

def create_weighted_average_ensemble(tuned_models, X_train, y_train, X_test, y_test):
    """
    Create a weighted average ensemble based on validation performance
    """
    print("\n" + "="*80)
    print("CREATING WEIGHTED AVERAGE ENSEMBLE")
    print("="*80)
    
    evaluator = ReimbursementEvaluator()
    
    # Get predictions from each model
    predictions = {}
    weights = {}
    
    for name, model in tuned_models.items():
        y_pred_val = model.predict(X_test)
        results = evaluator.evaluate(y_test, y_pred_val)
        
        predictions[name] = y_pred_val
        # Weight inversely proportional to MAE
        weights[name] = 1.0 / results['mae']
        
        print(f"{name}: MAE = ${results['mae']:.2f}, Weight = {weights[name]:.4f}")
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    print("\nNormalized weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    
    # Create weighted prediction function
    def weighted_predict(X):
        pred = np.zeros(len(X))
        for name, model in tuned_models.items():
            pred += weights[name] * model.predict(X)
        return pred
    
    return weighted_predict, weights

# ============================================================================
# COMPARE ALL MODELS
# ============================================================================

def compare_all_models(tuned_models, ensembles, X_test, y_test):
    """
    Compare all tuned and ensemble models
    """
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON")
    print("="*80)
    
    evaluator = ReimbursementEvaluator()
    results_summary = []
    
    # Evaluate tuned individual models
    print("\nTUNED INDIVIDUAL MODELS:")
    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        results = evaluator.evaluate(y_test, y_pred)
        evaluator.print_results(results, f"Tuned {name.replace('_', ' ').title()}")
        
        results_summary.append({
            'model': f'Tuned {name.replace("_", " ").title()}',
            'test_mae': results['mae'],
            'test_rmse': results['rmse'],
            'test_exact_pct': results['exact_pct'],
            'test_close_pct': results['close_pct'],
            'test_r2': results['r2']
        })
    
    # Evaluate ensemble models
    print("\nENSEMBLE MODELS:")
    for name, model in ensembles.items():
        if name == 'weighted_average':
            # Special case: weighted average returns a function
            y_pred = model(X_test)
        else:
            y_pred = model.predict(X_test)
        
        results = evaluator.evaluate(y_test, y_pred)
        evaluator.print_results(results, f"{name.replace('_', ' ').title()} Ensemble")
        
        results_summary.append({
            'model': f'{name.replace("_", " ").title()} Ensemble',
            'test_mae': results['mae'],
            'test_rmse': results['rmse'],
            'test_exact_pct': results['exact_pct'],
            'test_close_pct': results['close_pct'],
            'test_r2': results['r2']
        })
    
    return results_summary

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_final_comparison(results_summary):
    """
    Create comprehensive visualization of final model comparison
    """
    df_results = pd.DataFrame(results_summary)
    df_results = df_results.sort_values('test_mae')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # MAE comparison
    axes[0, 0].barh(range(len(df_results)), df_results['test_mae'], color='steelblue')
    axes[0, 0].set_yticks(range(len(df_results)))
    axes[0, 0].set_yticklabels(df_results['model'], fontsize=9)
    axes[0, 0].set_xlabel('Mean Absolute Error ($)', fontsize=12)
    axes[0, 0].set_title('Test MAE - Lower is Better', fontsize=14, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Exact match percentage
    axes[0, 1].barh(range(len(df_results)), df_results['test_exact_pct'], color='coral')
    axes[0, 1].set_yticks(range(len(df_results)))
    axes[0, 1].set_yticklabels(df_results['model'], fontsize=9)
    axes[0, 1].set_xlabel('Exact Matches (%)', fontsize=12)
    axes[0, 1].set_title('Exact Match Rate (±$0.01)', fontsize=14, fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Close match percentage
    axes[1, 0].barh(range(len(df_results)), df_results['test_close_pct'], color='mediumseagreen')
    axes[1, 0].set_yticks(range(len(df_results)))
    axes[1, 0].set_yticklabels(df_results['model'], fontsize=9)
    axes[1, 0].set_xlabel('Close Matches (%)', fontsize=12)
    axes[1, 0].set_title('Close Match Rate (±$1.00)', fontsize=14, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # R² comparison
    axes[1, 1].barh(range(len(df_results)), df_results['test_r2'], color='mediumpurple')
    axes[1, 1].set_yticks(range(len(df_results)))
    axes[1, 1].set_yticklabels(df_results['model'], fontsize=9)
    axes[1, 1].set_xlabel('R² Score', fontsize=12)
    axes[1, 1].set_title('R² Score - Higher is Better', fontsize=14, fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/06_final_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_final_models(tuned_models, ensembles, results_summary, weights=None):
    """
    Save all final models and results
    """
    print("\n" + "="*80)
    print("SAVING FINAL MODELS AND RESULTS")
    print("="*80)
    
    # Save tuned models
    for name, model in tuned_models.items():
        joblib.dump(model, f'models/saved/tuned_{name}.pkl')
    print(f"✓ Tuned models saved")
    
    # Save ensembles
    for name, model in ensembles.items():
        if name != 'weighted_average':  # Can't pickle functions easily
            joblib.dump(model, f'models/saved/ensemble_{name}.pkl')
    print(f"✓ Ensemble models saved")
    
    # Save weights if available
    if weights:
        joblib.dump(weights, 'models/saved/ensemble_weights.pkl')
        print(f"✓ Ensemble weights saved")
    
    # Save results
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv('results/final_model_results.csv', index=False)
    print(f"✓ Results saved to results/final_model_results.csv")
    
    # Identify and save best model
    best_idx = df_results['test_mae'].idxmin()
    best_model_name = df_results.loc[best_idx, 'model']
    best_mae = df_results.loc[best_idx, 'test_mae']
    
    with open('results/best_model.txt', 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Test MAE: ${best_mae:.2f}\n")
        f.write(f"\nFull Results:\n")
        f.write(df_results.to_string())
    
    print(f"✓ Best model info saved")
    print(f"\nBest Model: {best_model_name} (MAE: ${best_mae:.2f})")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("SESSION 6: HYPERPARAMETER TUNING & ENSEMBLE METHODS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('data/processed/train_features.csv')
    test_df = pd.read_csv('data/processed/test_features.csv')
    
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'reimbursement']
    
    X_train = train_df[numeric_cols]
    y_train = train_df['reimbursement']
    X_test = test_df[numeric_cols]
    y_test = test_df['reimbursement']
    
    print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Hyperparameter tuning
    tuned_models = {}
    
    # Tune Random Forest
    tuned_rf, params_rf = tune_random_forest(X_train, y_train)
    tuned_models['random_forest'] = tuned_rf
    
    # Tune XGBoost
    tuned_xgb, params_xgb = tune_xgboost(X_train, y_train)
    tuned_models['xgboost'] = tuned_xgb
    
    # Tune Gradient Boosting
    tuned_gb, params_gb = tune_gradient_boosting(X_train, y_train)
    tuned_models['gradient_boosting'] = tuned_gb
    
    # Create ensembles
    ensembles = {}
    
    voting_ensemble = create_voting_ensemble(tuned_models, X_train, y_train)
    ensembles['voting'] = voting_ensemble
    
    stacking_ensemble = create_stacking_ensemble(tuned_models, X_train, y_train)
    ensembles['stacking'] = stacking_ensemble
    
    weighted_pred, weights = create_weighted_average_ensemble(
        tuned_models, X_train, y_train, X_test, y_test
    )
    ensembles['weighted_average'] = weighted_pred
    
    # Compare all models
    results_summary = compare_all_models(tuned_models, ensembles, X_test, y_test)
    
    # Visualize
    plot_final_comparison(results_summary)
    
    # Save everything
    save_final_models(tuned_models, ensembles, results_summary, weights)
    
    print("\n" + "="*80)
    print("SESSION 6 COMPLETE!")
    print("="*80)
    print("\nAll models tuned and evaluated!")
    print("Check results/final_model_results.csv for complete comparison")
    print("\nNext Steps:")
    print("1. Create production prediction script")
    print("2. Build final report and presentation")

if __name__ == "__main__":
    main()