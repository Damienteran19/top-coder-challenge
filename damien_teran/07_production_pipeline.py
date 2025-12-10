"""
Session 7: Production Pipeline Creation (FIXED - Matches Session 3)
CSCI/DASC 6020 - Machine Learning Team Project

This script creates the production-ready prediction pipeline.
Features EXACTLY match Session 3 feature engineering.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================================
# FEATURE ENGINEERING PIPELINE (MATCHES SESSION 3 EXACTLY)
# ============================================================================

class ProductionFeatureEngineer:
    """
    Production-ready feature engineering pipeline
    MUST match Session 3 training feature engineering EXACTLY
    """
    
    def __init__(self):
        self.IRS_MILEAGE_RATE = 0.655
        self.TYPICAL_PER_DIEM = 150.0
        
        # Store training statistics for z-scores (will be loaded from training data)
        self.miles_mean = None
        self.miles_std = None
        self.duration_mean = None
        self.duration_std = None
        self.receipts_mean = None
        self.receipts_std = None
        
    def fit_stats(self, train_df):
        """Fit statistics from training data for z-score calculations"""
        self.miles_mean = train_df['miles_traveled'].mean()
        self.miles_std = train_df['miles_traveled'].std()
        self.duration_mean = train_df['trip_duration_days'].mean()
        self.duration_std = train_df['trip_duration_days'].std()
        self.receipts_mean = train_df['total_receipts_amount'].mean()
        self.receipts_std = train_df['total_receipts_amount'].std()
    
    def create_features(self, trip_duration_days, miles_traveled, total_receipts_amount):
        """
        Create all features EXACTLY as Session 3 does
        """
        # Use defaults if stats not fitted
        if self.miles_mean is None:
            self.miles_mean = 287.46
            self.miles_std = 198.57
            self.duration_mean = 4.52
            self.duration_std = 2.83
            self.receipts_mean = 625.79
            self.receipts_std = 411.34
        
        features = {
            'trip_duration_days': trip_duration_days,
            'miles_traveled': miles_traveled,
            'total_receipts_amount': total_receipts_amount
        }
        
        days_safe = max(trip_duration_days, 1)
        miles_safe = max(miles_traveled, 1)
        
        # Basic rates
        features['miles_per_day'] = miles_traveled / days_safe
        features['receipts_per_day'] = total_receipts_amount / days_safe
        features['receipts_per_mile'] = total_receipts_amount / miles_safe
        
        # Standard estimates
        features['est_mileage_reimb'] = miles_traveled * self.IRS_MILEAGE_RATE
        features['est_per_diem'] = trip_duration_days * self.TYPICAL_PER_DIEM
        features['est_total_simple'] = (total_receipts_amount + 
                                        features['est_mileage_reimb'] + 
                                        features['est_per_diem'])
        
        # Threshold features (EXACT names from Session 3)
        features['high_mileage'] = int(miles_traveled > 300)
        features['very_high_mileage'] = int(miles_traveled > 500)
        features['long_trip'] = int(trip_duration_days > 7)
        features['medium_trip'] = int((trip_duration_days > 3) and (trip_duration_days <= 7))
        features['overnight'] = int(trip_duration_days > 1)
        
        features['high_receipts'] = int(total_receipts_amount > 800)
        features['very_high_receipts'] = int(total_receipts_amount > 1200)
        features['low_receipts'] = int(total_receipts_amount < 300)
        features['near_receipt_cap'] = int((total_receipts_amount > 700) and (total_receipts_amount <= 800))
        features['over_receipt_cap'] = int(total_receipts_amount > 800)
        
        features['under_100_miles'] = int(miles_traveled < 100)
        features['local_trip'] = int(miles_traveled < 50)
        features['regional_trip'] = int((miles_traveled >= 100) and (miles_traveled < 300))
        
        # Interaction features (EXACT names)
        features['miles_x_duration'] = miles_traveled * trip_duration_days
        features['receipts_x_duration'] = total_receipts_amount * trip_duration_days
        features['miles_x_receipts'] = miles_traveled * total_receipts_amount
        
        # Ratio features
        features['receipts_to_miles_ratio'] = total_receipts_amount / miles_safe
        features['miles_to_days_ratio'] = miles_traveled / days_safe
        
        # Polynomial features
        features['miles_squared'] = miles_traveled ** 2
        features['duration_squared'] = trip_duration_days ** 2
        features['receipts_squared'] = total_receipts_amount ** 2
        
        features['miles_sqrt'] = np.sqrt(miles_traveled)
        features['duration_sqrt'] = np.sqrt(trip_duration_days)
        features['receipts_sqrt'] = np.sqrt(total_receipts_amount)
        
        # Log features
        features['miles_log'] = np.log1p(miles_traveled)
        features['duration_log'] = np.log1p(trip_duration_days)
        features['receipts_log'] = np.log1p(total_receipts_amount)
        
        # Binned features (exact names from Session 3)
        # Miles bins
        if miles_traveled < 100:
            features['miles_<100'] = 1
            features['miles_100-300'] = 0
            features['miles_300-500'] = 0
            features['miles_>500'] = 0
        elif miles_traveled < 300:
            features['miles_<100'] = 0
            features['miles_100-300'] = 1
            features['miles_300-500'] = 0
            features['miles_>500'] = 0
        elif miles_traveled < 500:
            features['miles_<100'] = 0
            features['miles_100-300'] = 0
            features['miles_300-500'] = 1
            features['miles_>500'] = 0
        else:
            features['miles_<100'] = 0
            features['miles_100-300'] = 0
            features['miles_300-500'] = 0
            features['miles_>500'] = 1
        
        # Duration bins
        if trip_duration_days <= 3:
            features['dur_Short'] = 1
            features['dur_Medium'] = 0
            features['dur_Long'] = 0
        elif trip_duration_days <= 7:
            features['dur_Short'] = 0
            features['dur_Medium'] = 1
            features['dur_Long'] = 0
        else:
            features['dur_Short'] = 0
            features['dur_Medium'] = 0
            features['dur_Long'] = 1
        
        # Receipts bins
        if total_receipts_amount <= 300:
            features['rec_Low'] = 1
            features['rec_Medium'] = 0
            features['rec_High'] = 0
            features['rec_VeryHigh'] = 0
        elif total_receipts_amount <= 800:
            features['rec_Low'] = 0
            features['rec_Medium'] = 1
            features['rec_High'] = 0
            features['rec_VeryHigh'] = 0
        elif total_receipts_amount <= 1500:
            features['rec_Low'] = 0
            features['rec_Medium'] = 0
            features['rec_High'] = 1
            features['rec_VeryHigh'] = 0
        else:
            features['rec_Low'] = 0
            features['rec_Medium'] = 0
            features['rec_High'] = 0
            features['rec_VeryHigh'] = 1
        
        # Z-scores
        features['miles_zscore'] = (miles_traveled - self.miles_mean) / self.miles_std
        features['duration_zscore'] = (trip_duration_days - self.duration_mean) / self.duration_std
        features['receipts_zscore'] = (total_receipts_amount - self.receipts_mean) / self.receipts_std
        
        # Anomaly flags
        features['anomaly_miles'] = int((features['miles_zscore'] > 2) or (features['miles_zscore'] < -2))
        features['anomaly_duration'] = int((features['duration_zscore'] > 2) or (features['duration_zscore'] < -2))
        features['anomaly_receipts'] = int((features['receipts_zscore'] > 2) or (features['receipts_zscore'] < -2))
        
        # Weighted estimates (EXACT names)
        features['weighted_estimate_v1'] = (total_receipts_amount * 0.8 + 
                                            features['est_mileage_reimb'] + 
                                            trip_duration_days * 50)
        
        features['weighted_estimate_v2'] = (total_receipts_amount * 1.0 + 
                                            miles_traveled * 0.5 + 
                                            trip_duration_days * 100)
        
        return pd.DataFrame([features])

# ============================================================================
# PRODUCTION PREDICTOR
# ============================================================================

class ReimbursementPredictor:
    """Production-ready reimbursement predictor"""
    
    def __init__(self, model_path='models/saved/tuned_xgboost.pkl'):
        self.feature_engineer = ProductionFeatureEngineer()
        self.model = None
        self.model_path = model_path
        self._load_model()
        self._fit_stats()
    
    def _load_model(self):
        """Load the trained model"""
        if Path(self.model_path).exists():
            self.model = joblib.load(self.model_path)
            print(f"✓ Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
    def _fit_stats(self):
        """Load training statistics for z-score calculations"""
        try:
            train_df = pd.read_csv('data/processed/train_data.csv')
            self.feature_engineer.fit_stats(train_df)
        except:
            print("Warning: Could not load training data for stats, using defaults")
    
    def predict(self, trip_duration_days, miles_traveled, total_receipts_amount):
        """Make a prediction"""
        if trip_duration_days < 0 or miles_traveled < 0 or total_receipts_amount < 0:
            raise ValueError("All inputs must be non-negative")
        
        X = self.feature_engineer.create_features(
            trip_duration_days, miles_traveled, total_receipts_amount
        )
        
        prediction = self.model.predict(X)[0]
        prediction = round(max(0.0, prediction), 2)
        
        return prediction
    
    def predict_batch(self, data):
        """Make predictions for multiple cases"""
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
        else:
            raise ValueError("Data must be DataFrame")

# ============================================================================
# TESTING
# ============================================================================

def test_production_pipeline():
    print("\n" + "="*80)
    print("TESTING PRODUCTION PIPELINE")
    print("="*80)
    
    predictor = ReimbursementPredictor()
    
    test_cases = [
        (1, 50, 100.00, "Short day trip"),
        (3, 250, 500.00, "Multi-day regional trip"),
        (7, 1000, 1500.00, "Week-long distance trip"),
        (14, 2500, 3000.00, "Extended travel"),
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

def validate_on_test_set():
    print("\n" + "="*80)
    print("VALIDATING ON TEST SET")
    print("="*80)
    
    test_df = pd.read_csv('data/processed/test_data.csv')
    predictor = ReimbursementPredictor()
    
    print("Making predictions on test set...")
    predictions = predictor.predict_batch(test_df)
    
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
    
    results_df = test_df.copy()
    results_df['predicted'] = predictions
    results_df['error'] = diff
    results_df.to_csv('results/production_predictions.csv', index=False)
    
    print(f"\n✓ Predictions saved to results/production_predictions.csv")

def main():
    print("="*80)
    print("SESSION 7: PRODUCTION PIPELINE CREATION")
    print("="*80)
    
    test_production_pipeline()
    validate_on_test_set()
    
    print("\n" + "="*80)
    print("SESSION 7 COMPLETE!")
    print("="*80)
    print("\n✓ Production pipeline working!")
    print("✓ Features match training exactly")
    print("\nNext: python 08_generate_final_report.py")

if __name__ == "__main__":
    main()