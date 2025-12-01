"""
Session 5: Advanced Model Development
CSCI/DASC 6020 - Machine Learning Team Project

This script implements advanced ML models including tree-based methods,
neural networks, and ensemble techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import evaluator from previous session
import sys
sys.path.append('.')

# ============================================================================
# EVALUATION (reuse from Session 4)
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
        
        avg_error = np.mean(diff)
        max_error = np.max(diff)
        
        return {
            'exact_matches': exact_matches,
            'exact_pct': exact_pct,
            'close_matches': close_matches,
            'close_pct': close_pct,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'avg_error': avg_error,
            'max_error': max_error,
            'n_samples': len(y_true)
        }
    
    def print_results(self, results, model_name="Model"):
        """Print evaluation results"""
        print(f"\n{model_name} Results:")
        print(f"  Exact matches (±${self.tolerance_exact}): {results['exact_matches']}/{results['n_samples']} ({results['exact_pct']:.2f}%)")
        print(f"  Close matches (±${self.tolerance_close}): {results['close_matches']}/{results['n_samples']} ({results['close_pct']:.2f}%)")
        print(f"  MAE: ${results['mae']:.2f}")
        print(f"  RMSE: ${results['rmse']:.2f}")
        print(f"  R²: {results['r2']:.4f}")

# ============================================================================
# TREE-BASED MODELS
# ============================================================================

def train_decision_tree(X_train, y_train, max_depth=10):
    """
    Train a Decision Tree with interpretability
    """
    print(f"\nTraining Decision Tree (max_depth={max_depth})...")
    
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Feature importance
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"  Top 10 important features:")
    for idx, row in importances.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return model, importances

def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Train Random Forest Regressor
    """
    print(f"\nTraining Random Forest (n_estimators={n_estimators})...")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Feature importance
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"  Top 10 important features:")
    for idx, row in importances.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return model, importances

def train_gradient_boosting(X_train, y_train, n_estimators=100):
    """
    Train Gradient Boosting Regressor
    """
    print(f"\nTraining Gradient Boosting (n_estimators={n_estimators})...")
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Feature importance
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"  Top 10 important features:")
    for idx, row in importances.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return model, importances

def train_xgboost(X_train, y_train, n_estimators=100):
    """
    Train XGBoost Regressor
    """
    print(f"\nTraining XGBoost (n_estimators={n_estimators})...")
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Feature importance
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"  Top 10 important features:")
    for idx, row in importances.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return model, importances

# ============================================================================
# NEURAL NETWORK
# ============================================================================

def train_neural_network(X_train, y_train, hidden_layers=(100, 50)):
    """
    Train Multi-layer Perceptron Regressor
    """
    print(f"\nTraining Neural Network (hidden_layers={hidden_layers})...")
    
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    model.fit(X_train, y_train)
    
    print(f"  Training iterations: {model.n_iter_}")
    print(f"  Training loss: {model.loss_:.4f}")
    
    return model

# ============================================================================
# SUPPORT VECTOR REGRESSION
# ============================================================================

def train_svr(X_train, y_train, kernel='rbf'):
    """
    Train Support Vector Regressor
    Note: SVR can be slow on large datasets
    """
    print(f"\nTraining SVR (kernel={kernel})...")
    print("  Warning: This may take several minutes...")
    
    # Sample data if too large
    if len(X_train) > 2000:
        print(f"  Sampling 2000 cases for SVR training...")
        sample_idx = np.random.choice(len(X_train), 2000, replace=False)
        X_sample = X_train.iloc[sample_idx]
        y_sample = y_train.iloc[sample_idx]
    else:
        X_sample = X_train
        y_sample = y_train
    
    model = SVR(
        kernel=kernel,
        C=100.0,
        epsilon=0.1,
        gamma='scale'
    )
    
    model.fit(X_sample, y_sample)
    
    return model

# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_advanced_models(X_train, X_test, y_train, y_test):
    """
    Train and compare all advanced models
    """
    print("\n" + "="*80)
    print("ADVANCED MODEL COMPARISON")
    print("="*80)
    
    evaluator = ReimbursementEvaluator()
    results_summary = []
    models = {}
    feature_importances = {}
    
    # Decision Tree
    print("\n" + "-"*80)
    print("1. DECISION TREE")
    print("-"*80)
    model_dt, imp_dt = train_decision_tree(X_train, y_train, max_depth=15)
    y_pred_train_dt = model_dt.predict(X_train)
    y_pred_test_dt = model_dt.predict(X_test)
    
    results_train = evaluator.evaluate(y_train, y_pred_train_dt)
    results_test = evaluator.evaluate(y_test, y_pred_test_dt)
    evaluator.print_results(results_train, "Decision Tree (Train)")
    evaluator.print_results(results_test, "Decision Tree (Test)")
    
    models['decision_tree'] = model_dt
    feature_importances['decision_tree'] = imp_dt
    results_summary.append({
        'model': 'Decision Tree',
        'train_mae': results_train['mae'],
        'test_mae': results_test['mae'],
        'test_exact_pct': results_test['exact_pct'],
        'test_close_pct': results_test['close_pct'],
        'test_r2': results_test['r2']
    })
    
    # Random Forest
    print("\n" + "-"*80)
    print("2. RANDOM FOREST")
    print("-"*80)
    model_rf, imp_rf = train_random_forest(X_train, y_train, n_estimators=200)
    y_pred_train_rf = model_rf.predict(X_train)
    y_pred_test_rf = model_rf.predict(X_test)
    
    results_train = evaluator.evaluate(y_train, y_pred_train_rf)
    results_test = evaluator.evaluate(y_test, y_pred_test_rf)
    evaluator.print_results(results_train, "Random Forest (Train)")
    evaluator.print_results(results_test, "Random Forest (Test)")
    
    models['random_forest'] = model_rf
    feature_importances['random_forest'] = imp_rf
    results_summary.append({
        'model': 'Random Forest',
        'train_mae': results_train['mae'],
        'test_mae': results_test['mae'],
        'test_exact_pct': results_test['exact_pct'],
        'test_close_pct': results_test['close_pct'],
        'test_r2': results_test['r2']
    })
    
    # Gradient Boosting
    print("\n" + "-"*80)
    print("3. GRADIENT BOOSTING")
    print("-"*80)
    model_gb, imp_gb = train_gradient_boosting(X_train, y_train, n_estimators=200)
    y_pred_train_gb = model_gb.predict(X_train)
    y_pred_test_gb = model_gb.predict(X_test)
    
    results_train = evaluator.evaluate(y_train, y_pred_train_gb)
    results_test = evaluator.evaluate(y_test, y_pred_test_gb)
    evaluator.print_results(results_train, "Gradient Boosting (Train)")
    evaluator.print_results(results_test, "Gradient Boosting (Test)")
    
    models['gradient_boosting'] = model_gb
    feature_importances['gradient_boosting'] = imp_gb
    results_summary.append({
        'model': 'Gradient Boosting',
        'train_mae': results_train['mae'],
        'test_mae': results_test['mae'],
        'test_exact_pct': results_test['exact_pct'],
        'test_close_pct': results_test['close_pct'],
        'test_r2': results_test['r2']
    })
    
    # XGBoost
    print("\n" + "-"*80)
    print("4. XGBOOST")
    print("-"*80)
    model_xgb, imp_xgb = train_xgboost(X_train, y_train, n_estimators=200)
    y_pred_train_xgb = model_xgb.predict(X_train)
    y_pred_test_xgb = model_xgb.predict(X_test)
    
    results_train = evaluator.evaluate(y_train, y_pred_train_xgb)
    results_test = evaluator.evaluate(y_test, y_pred_test_xgb)
    evaluator.print_results(results_train, "XGBoost (Train)")
    evaluator.print_results(results_test, "XGBoost (Test)")
    
    models['xgboost'] = model_xgb
    feature_importances['xgboost'] = imp_xgb
    results_summary.append({
        'model': 'XGBoost',
        'train_mae': results_train['mae'],
        'test_mae': results_test['mae'],
        'test_exact_pct': results_test['exact_pct'],
        'test_close_pct': results_test['close_pct'],
        'test_r2': results_test['r2']
    })
    
    # Neural Network
    print("\n" + "-"*80)
    print("5. NEURAL NETWORK")
    print("-"*80)
    model_nn = train_neural_network(X_train, y_train, hidden_layers=(100, 50, 25))
    y_pred_train_nn = model_nn.predict(X_train)
    y_pred_test_nn = model_nn.predict(X_test)
    
    results_train = evaluator.evaluate(y_train, y_pred_train_nn)
    results_test = evaluator.evaluate(y_test, y_pred_test_nn)
    evaluator.print_results(results_train, "Neural Network (Train)")
    evaluator.print_results(results_test, "Neural Network (Test)")
    
    models['neural_network'] = model_nn
    results_summary.append({
        'model': 'Neural Network',
        'train_mae': results_train['mae'],
        'test_mae': results_test['mae'],
        'test_exact_pct': results_test['exact_pct'],
        'test_close_pct': results_test['close_pct'],
        'test_r2': results_test['r2']
    })
    
    return results_summary, models, feature_importances

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_model_comparison(results_summary):
    """Visualize model performance comparison"""
    df_results = pd.DataFrame(results_summary)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Test MAE
    axes[0, 0].barh(range(len(df_results)), df_results['test_mae'])
    axes[0, 0].set_yticks(range(len(df_results)))
    axes[0, 0].set_yticklabels(df_results['model'])
    axes[0, 0].set_xlabel('Mean Absolute Error ($)')
    axes[0, 0].set_title('Test MAE Comparison (Lower is Better)')
    axes[0, 0].invert_yaxis()
    
    # Exact match percentage
    axes[0, 1].barh(range(len(df_results)), df_results['test_exact_pct'])
    axes[0, 1].set_yticks(range(len(df_results)))
    axes[0, 1].set_yticklabels(df_results['model'])
    axes[0, 1].set_xlabel('Exact Matches (%)')
    axes[0, 1].set_title('Exact Match % (±$0.01)')
    axes[0, 1].invert_yaxis()
    
    # Close match percentage
    axes[1, 0].barh(range(len(df_results)), df_results['test_close_pct'])
    axes[1, 0].set_yticks(range(len(df_results)))
    axes[1, 0].set_yticklabels(df_results['model'])
    axes[1, 0].set_xlabel('Close Matches (%)')
    axes[1, 0].set_title('Close Match % (±$1.00)')
    axes[1, 0].invert_yaxis()
    
    # R² Score
    axes[1, 1].barh(range(len(df_results)), df_results['test_r2'])
    axes[1, 1].set_yticks(range(len(df_results)))
    axes[1, 1].set_yticklabels(df_results['model'])
    axes[1, 1].set_xlabel('R² Score')
    axes[1, 1].set_title('R² Score (Higher is Better)')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('reports/figures/05_advanced_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(feature_importances):
    """Plot feature importance from tree-based models"""
    n_models = len(feature_importances)
    fig, axes = plt.subplots(1, min(n_models, 4), figsize=(20, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, importances) in enumerate(list(feature_importances.items())[:4]):
        top_features = importances.head(15)
        
        axes[idx].barh(range(len(top_features)), top_features['importance'])
        axes[idx].set_yticks(range(len(top_features)))
        axes[idx].set_yticklabels(top_features['feature'])
        axes[idx].set_xlabel('Importance')
        axes[idx].set_title(f'{model_name.replace("_", " ").title()}')
        axes[idx].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('reports/figures/05_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_advanced_results(results_summary, models, feature_importances):
    """Save results and models"""
    # Save results
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv('results/advanced_model_results.csv', index=False)
    print(f"\n✓ Results saved to results/advanced_model_results.csv")
    
    # Save models
    for name, model in models.items():
        joblib.dump(model, f'models/saved/{name}.pkl')
    print(f"✓ Models saved to models/saved/")
    
    # Save feature importances
    for name, importances in feature_importances.items():
        importances.to_csv(f'results/{name}_feature_importance.csv', index=False)
    print(f"✓ Feature importances saved")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("SESSION 5: ADVANCED MODEL DEVELOPMENT")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('data/processed/train_features.csv')
    test_df = pd.read_csv('data/processed/test_features.csv')
    
    # Use numeric features only
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'reimbursement']
    
    X_train = train_df[numeric_cols]
    y_train = train_df['reimbursement']
    X_test = test_df[numeric_cols]
    y_test = test_df['reimbursement']
    
    print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train and compare models
    results_summary, models, feature_importances = compare_advanced_models(
        X_train, X_test, y_train, y_test
    )
    
    # Visualizations
    print("\n Creating visualizations...")
    plot_model_comparison(results_summary)
    plot_feature_importance(feature_importances)
    
    # Save results
    save_advanced_results(results_summary, models, feature_importances)
    
    # Summary
    df_results = pd.DataFrame(results_summary)
    best_model = df_results.loc[df_results['test_mae'].idxmin(), 'model']
    best_mae = df_results['test_mae'].min()
    
    print("\n" + "="*80)
    print("SESSION 5 COMPLETE!")
    print("="*80)
    print(f"\nBest Model: {best_model}")
    print(f"Test MAE: ${best_mae:.2f}")
    print("\nAll Models Trained:")
    print(df_results[['model', 'test_mae', 'test_exact_pct']].to_string(index=False))
    print("\nNext Steps:")
    print("1. Hyperparameter tuning for best models")
    print("2. Ensemble methods (stacking, voting)")
    print("3. Model interpretability analysis")

if __name__ == "__main__":
    main()
