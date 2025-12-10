"""
Session 7: Production Pipeline Creation
CSCI/DASC 6020 - Machine Learning Team Project

This script creates the production-ready prediction pipeline
that meets the project requirements:
- Takes 3 parameters as input
- Outputs a single reimbursement amount
- Runs in under 5 seconds
- No external dependencies (database, network, etc.)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================================
# FEATURE ENGINEERING PIPELINE
# ============================================================================

class ProductionFeatureEngineer:
    """
    Production-ready feature engineering pipeline
    Must match training feature engineering exactly
    """
    
    def __init__(self):
        self.STANDARD_MILEAGE_RATE = 0.655
        self.PER_DIEM_RATE = 150.0
    
    def create_features(self, trip_duration_days, miles_traveled, total_receipts_amount):
        """
        Create all features from the three inputs
        
        Parameters:
        -----------
        trip_duration_days : int or float
        miles_traveled : int or float
        total_receipts_amount : float
        
        Returns:
        --------
        pd.DataFrame : Single-row dataframe with all features
        """
        # Create base dictionary
        features = {
            'trip_duration_days': trip_duration_days,
            'miles_traveled': miles_traveled,
            'total_receipts_amount': total_receipts_amount
        }
        
        # Protect against division by zero
        days_safe = trip_duration_days if trip_duration_days > 0 else 1
        miles_safe = miles_traveled if miles_traveled > 0 else 1
        
        # Rate-based features
        features['receipts_per_day'] = total_receipts_amount / days_safe
        features['receipts_per_mile'] = total_receipts_amount / miles_safe
        features['miles_per_day'] = miles_traveled / days_safe
        
        # Standard reimbursement estimates
        features['est_mileage_reimb'] = miles_traveled * self.STANDARD_MILEAGE_RATE
        features['est_per_diem'] = trip_duration_days * self.PER_DIEM_RATE
        
        # Interaction features
        features['days_times_miles'] = trip_duration_days * miles_traveled
        features['days_times_receipts'] = trip_duration_days * total_receipts_amount
        features['miles_times_receipts'] = miles_traveled * total_receipts_amount
        
        # Polynomial features
        features['days_squared'] = trip_duration_days ** 2
        features['miles_squared'] = miles_traveled ** 2
        features['receipts_squared'] = total_receipts_amount ** 2
        
        # Root features
        features['days_sqrt'] = np.sqrt(trip_duration_days)
        features['miles_sqrt'] = np.sqrt(miles_traveled)
        features['receipts_sqrt'] = np.sqrt(total_receipts_amount)
        
        # Log features
        features['days_log'] = np.log1p(trip_duration_days)
        features['miles_log'] = np.log1p(miles_traveled)
        features['receipts_log'] = np.log1p(total_receipts_amount)
        
        # Threshold features
        features['is_overnight'] = int(trip_duration_days > 1)
        features['is_long_trip'] = int(trip_duration_days > 3)
        features['is_week_plus'] = int(trip_duration_days >= 7)
        features['is_long_distance'] = int(miles_traveled > 250)
        features['is_very_long_distance'] = int(miles_traveled > 500)
        features['high_receipts'] = int(total_receipts_amount > 500)
        features['very_high_receipts'] = int(total_receipts_amount > 1000)
        
        # Business logic features
        features['simple_sum'] = (total_receipts_amount + 
                                  features['est_mileage_reimb'] + 
                                  features['est_per_diem'])
        
        features['weighted_v1'] = (total_receipts_amount * 0.8 + 
                                   features['est_mileage_reimb'] + 
                                   trip_duration_days * 50)
        
        features['weighted_v2'] = (total_receipts_amount * 1.0 + 
                                   miles_traveled * 0.5 + 
                                   trip_duration_days * 100)
        
        features['conditional_v1'] = (total_receipts_amount + 
                                      features['est_mileage_reimb'] + 
                                      (features['est_per_diem'] if trip_duration_days > 1 else 0))
        
        # Receipt percentages
        for pct in [0.7, 0.8, 0.9, 1.0]:
            features[f'receipts_{int(pct*100)}pct'] = total_receipts_amount * pct
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        return df

# ============================================================================
# PRODUCTION PREDICTOR
# ============================================================================

class ReimbursementPredictor:
    """
    Production-ready reimbursement predictor
    """
    
    def __init__(self, model_path='models/saved/tuned_xgboost.pkl'):
        """
        Initialize predictor with trained model
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        """
        self.feature_engineer = ProductionFeatureEngineer()
        self.model = None
        self.feature_names = None
        self.model_path = model_path
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        if Path(self.model_path).exists():
            self.model = joblib.load(self.model_path)
            print(f"[OK] Model loaded from {self.model_path}")
            
            # Load feature names if available
            feature_names_path = 'data/processed/feature_names.txt'
            if Path(feature_names_path).exists():
                with open(feature_names_path, 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
    def predict(self, trip_duration_days, miles_traveled, total_receipts_amount):
        """
        Make a prediction
        
        Parameters:
        -----------
        trip_duration_days : int or float
        miles_traveled : int or float
        total_receipts_amount : float
        
        Returns:
        --------
        float : Predicted reimbursement amount (rounded to 2 decimal places)
        """
        # Input validation
        if trip_duration_days < 0 or miles_traveled < 0 or total_receipts_amount < 0:
            raise ValueError("All inputs must be non-negative")
        
        # Create features
        X = self.feature_engineer.create_features(
            trip_duration_days, 
            miles_traveled, 
            total_receipts_amount
        )
        
        # Ensure we have all required features
        if self.feature_names:
            # Reorder columns to match training data
            missing_cols = set(self.feature_names) - set(X.columns)
            if missing_cols:
                print(f"Warning: Missing features: {missing_cols}")
            
            # Select only features that model expects
            available_features = [f for f in self.feature_names if f in X.columns]
            X = X[available_features]
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Round to 2 decimal places
        prediction = round(prediction, 2)
        
        # Ensure non-negative
        prediction = max(0.0, prediction)
        
        return prediction
    
    def predict_batch(self, data):
        """
        Make predictions for multiple cases
        
        Parameters:
        -----------
        data : pd.DataFrame or list of tuples
            If DataFrame, must have columns: trip_duration_days, miles_traveled, total_receipts_amount
            If list, each tuple should be (days, miles, receipts)
        
        Returns:
        --------
        np.array : Array of predictions
        """
        if isinstance(data, pd.DataFrame):
            predictions = []
            for _, row in data.iterrows():
                pred = self.predict(
                    row['trip_duration_days'],
                    row['miles_traveled'],
                    row['total_receipts_amount']
                )
                predictions.append(pred)
            return np.array(predictions)
        
        elif isinstance(data, list):
            predictions = []
            for days, miles, receipts in data:
                pred = self.predict(days, miles, receipts)
                predictions.append(pred)
            return np.array(predictions)
        
        else:
            raise ValueError("Data must be DataFrame or list of tuples")

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_production_pipeline():
    """
    Test the production pipeline with sample cases
    """
    print("\n" + "="*80)
    print("TESTING PRODUCTION PIPELINE")
    print("="*80)
    
    # Initialize predictor
    predictor = ReimbursementPredictor()
    
    # Test cases
    test_cases = [
        (1, 50, 100.00, "Short day trip"),
        (3, 250, 500.00, "Multi-day regional trip"),
        (7, 1000, 1500.00, "Week-long distance trip"),
        (14, 2500, 3000.00, "Extended travel"),
        (0, 0, 0.00, "Edge case: all zeros"),
    ]
    
    print("\nTest Cases:")
    print("-" * 80)
    print(f"{'Days':>6} {'Miles':>6} {'Receipts':>10} {'Prediction':>12} {'Description'}")
    print("-" * 80)
    
    for days, miles, receipts, description in test_cases:
        try:
            prediction = predictor.predict(days, miles, receipts)
            print(f"{days:6d} {miles:6d} ${receipts:9.2f} ${prediction:11.2f} {description}")
        except Exception as e:
            print(f"{days:6d} {miles:6d} ${receipts:9.2f} {'ERROR':>12} {str(e)}")
    
    print("-" * 80)
    print("[OK] Production pipeline tested successfully")

def validate_on_test_set():
    """
    Validate the production pipeline on the test set
    """
    print("\n" + "="*80)
    print("VALIDATING ON TEST SET")
    print("="*80)
    
    # Load test data
    test_df = pd.read_csv('data/processed/test_data.csv')
    
    # Initialize predictor
    predictor = ReimbursementPredictor()
    
    # Make predictions
    print("Making predictions on test set...")
    predictions = predictor.predict_batch(test_df)
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    y_true = test_df['output'].values
    diff = np.abs(y_true - predictions)
    
    exact_matches = np.sum(diff <= 0.01)
    close_matches = np.sum(diff <= 1.00)
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)
    
    print(f"\nProduction Pipeline Performance:")
    print(f"  Exact matches (±$0.01): {exact_matches}/{len(y_true)} ({exact_matches/len(y_true)*100:.2f}%)")
    print(f"  Close matches (±$1.00): {close_matches}/{len(y_true)} ({close_matches/len(y_true)*100:.2f}%)")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    
    # Save predictions
    results_df = test_df.copy()
    results_df['predicted'] = predictions
    results_df['error'] = diff
    results_df.to_csv('results/production_predictions.csv', index=False)
    
    print(f"\n[OK] Predictions saved to results/production_predictions.csv")

def benchmark_performance():
    """
    Benchmark prediction speed
    """
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK")
    print("="*80)
    
    import time
    
    predictor = ReimbursementPredictor()
    
    # Single prediction timing
    start = time.time()
    for _ in range(100):
        _ = predictor.predict(5, 250, 500.00)
    elapsed = time.time() - start
    
    avg_time = elapsed / 100
    
    print(f"\nAverage prediction time: {avg_time*1000:.2f} ms")
    print(f"Predictions per second: {1/avg_time:.0f}")
    
    if avg_time < 5.0:
        print(f"[OK] Meets requirement: < 5 seconds per prediction")
    else:
        print(f"✗ WARNING: Exceeds 5 second limit!")

# ============================================================================
# CREATE STANDALONE SCRIPT
# ============================================================================

def create_standalone_script():
    """
    Create the final standalone predict.py script
    """
    print("\n" + "="*80)
    print("CREATING STANDALONE PREDICTION SCRIPT")
    print("="*80)
    
    script_content = '''#!/usr/bin/env python3
"""
Reimbursement Prediction Script
CSCI/DASC 6020 - Machine Learning Team Project

Usage:
    python predict.py <trip_duration_days> <miles_traveled> <total_receipts_amount>

Example:
    python predict.py 5 250 500.00
"""

import sys
import pandas as pd
import numpy as np
import joblib

class FeatureEngineer:
    def __init__(self):
        self.STANDARD_MILEAGE_RATE = 0.655
        self.PER_DIEM_RATE = 150.0
    
    def create_features(self, trip_duration_days, miles_traveled, total_receipts_amount):
        features = {
            'trip_duration_days': trip_duration_days,
            'miles_traveled': miles_traveled,
            'total_receipts_amount': total_receipts_amount
        }
        
        days_safe = max(trip_duration_days, 1)
        miles_safe = max(miles_traveled, 1)
        
        features['receipts_per_day'] = total_receipts_amount / days_safe
        features['receipts_per_mile'] = total_receipts_amount / miles_safe
        features['miles_per_day'] = miles_traveled / days_safe
        features['est_mileage_reimb'] = miles_traveled * self.STANDARD_MILEAGE_RATE
        features['est_per_diem'] = trip_duration_days * self.PER_DIEM_RATE
        features['days_times_miles'] = trip_duration_days * miles_traveled
        features['days_times_receipts'] = trip_duration_days * total_receipts_amount
        features['miles_times_receipts'] = miles_traveled * total_receipts_amount
        features['days_squared'] = trip_duration_days ** 2
        features['miles_squared'] = miles_traveled ** 2
        features['receipts_squared'] = total_receipts_amount ** 2
        features['days_sqrt'] = np.sqrt(trip_duration_days)
        features['miles_sqrt'] = np.sqrt(miles_traveled)
        features['receipts_sqrt'] = np.sqrt(total_receipts_amount)
        features['days_log'] = np.log1p(trip_duration_days)
        features['miles_log'] = np.log1p(miles_traveled)
        features['receipts_log'] = np.log1p(total_receipts_amount)
        features['is_overnight'] = int(trip_duration_days > 1)
        features['is_long_trip'] = int(trip_duration_days > 3)
        features['is_week_plus'] = int(trip_duration_days >= 7)
        features['is_long_distance'] = int(miles_traveled > 250)
        features['is_very_long_distance'] = int(miles_traveled > 500)
        features['high_receipts'] = int(total_receipts_amount > 500)
        features['very_high_receipts'] = int(total_receipts_amount > 1000)
        features['simple_sum'] = (total_receipts_amount + features['est_mileage_reimb'] + features['est_per_diem'])
        features['weighted_v1'] = (total_receipts_amount * 0.8 + features['est_mileage_reimb'] + trip_duration_days * 50)
        features['weighted_v2'] = (total_receipts_amount * 1.0 + miles_traveled * 0.5 + trip_duration_days * 100)
        features['conditional_v1'] = (total_receipts_amount + features['est_mileage_reimb'] + 
                                      (features['est_per_diem'] if trip_duration_days > 1 else 0))
        
        for pct in [0.7, 0.8, 0.9, 1.0]:
            features[f'receipts_{int(pct*100)}pct'] = total_receipts_amount * pct
        
        return pd.DataFrame([features])

def predict_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """Make reimbursement prediction"""
    # Load model
    model = joblib.load('models/saved/tuned_xgboost.pkl')
    
    # Create features
    engineer = FeatureEngineer()
    X = engineer.create_features(trip_duration_days, miles_traveled, total_receipts_amount)
    
    # Predict
    prediction = model.predict(X)[0]
    prediction = round(max(0.0, prediction), 2)
    
    return prediction

def main():
    if len(sys.argv) != 4:
        print("Usage: python predict.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        print("Example: python predict.py 5 250 500.00")
        sys.exit(1)
    
    try:
        days = float(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        prediction = predict_reimbursement(days, miles, receipts)
        
        print(prediction)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open('predict.py', 'w') as f:
        f.write(script_content)
    
    print("[OK] Created predict.py")
    print("\nUsage:")
    print("  python predict.py 5 250 500.00")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("SESSION 7: PRODUCTION PIPELINE CREATION")
    print("="*80)
    
    # Test production pipeline
    test_production_pipeline()
    
    # Validate on test set
    validate_on_test_set()
    
    # Benchmark performance
    benchmark_performance()
    
    # Create standalone script
    create_standalone_script()
    
    print("\n" + "="*80)
    print("SESSION 7 COMPLETE!")
    print("="*80)
    print("\nProduction pipeline ready!")
    print("\nFiles created:")
    print("- predict.py (standalone prediction script)")
    print("- results/production_predictions.csv")
    print("\nNext Steps:")
    print("1. Review production predictions")
    print("2. Create final technical report")
    print("3. Prepare business presentation")

if __name__ == "__main__":
    main()
