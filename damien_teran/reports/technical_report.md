# TECHNICAL REPORT
## Legacy Reimbursement System Reverse Engineering
**Project:** CSCI/DASC 6020 Machine Learning Team Project  
**Team:** Damien Teran, Sidney Land, Yasmine Scotland  
**Date:** November 20, 2025

## 1. INTRODUCTION
This project applies machine learning techniques to reverse-engineer ACME Corporation's legacy travel reimbursement system. The system has operated for 60 years without documentation, creating operational risks and limiting system transparency.

### Project Objectives
- Understand the hidden business logic in the legacy system
- Create a predictive model that replicates system behavior
- Provide explainable AI solutions for stakeholders
- Generate actionable business insights

## 2. PROBLEM STATEMENT
### Input Variables
- `trip_duration_days`: Number of days traveling (integer)
- `miles_traveled`: Total miles traveled (integer)
- `total_receipts_amount`: Total receipt amount (dollars)
### Output Variable
- Single reimbursement amount (dollars, 2 decimal places)
### Success Criteria
- Exact matches: Predictions within ±$0.01
- Close matches: Predictions within ±$1.00
- Minimize Mean Absolute Error (MAE)

## 3. DATA ANALYSIS
### Dataset Characteristics
- Total cases: 1,000
- Training set: 750 cases (75%)
- Test set: 250 cases (25%)
- Missing values: 0
- Outliers: Long trips >14 days, miles >2,500, receipts >$2,500 (kept to preserve legacy behavior)

### Descriptive Statistics
**Trip Duration (Days)**
- Mean: 4.8
- Median: 4
- Range: 1–14

**Miles Traveled**
- Mean: 412
- Median: 378
- Range: 5–1,204

**Total Receipts**
- Mean: $487.32
- Median: $412.50
- Range: $1.42–$2,503.46

**Reimbursement Amount**
- Mean: $512.18
- Median: $448.67
- Range: $8.50–$2,187.33

### Correlation Analysis
Strongest correlation: total_receipts_amount with reimbursement (Pearson r = 0.695)  
Moderate correlations: trip_duration_days (r = 0.42), miles_traveled (r = 0.38)  
Scatter plots revealed clear threshold effects around 500 miles and 7 days.

## 4. METHODOLOGY
### Feature Engineering
We created 32 derived features including:
1. Rate-based Features: receipts_per_day, miles_per_day, reimb_per_mile
2. Domain-specific Features: estimated mileage (0.655 × miles), estimated per diem (150 × days)
3. Interaction Features: days × miles, days × receipts, miles × receipts
4. Polynomial & Transform Features: log_receipts, sqrt_miles, squared terms
5. Threshold Features: is_week_plus (>=7 days), is_long_miles (>500 miles), high_receipts (>1000)

### Model Selection
Evaluated Linear Models, Tree-based Models (Random Forest, Gradient Boosting, XGBoost), Neural Networks, and Ensemble Methods.

### Evaluation Framework
- 5-fold cross-validation
- Fixed 75/25 train-test split
- Primary metrics: MAE, Exact Match % (±$0.01), Close Match % (±$1.00)

## 5. MODEL DEVELOPMENT
### Baseline Models
| Model               | MAE   | Exact % | Close % |
|-------------------|-------|---------|---------|
| Mean Predictor    | $182  | 0%      | 2%      |
| Linear Regression | $48   | 3%      | 20%     |

### Advanced Models (Test Set)
| Model                        | MAE   | RMSE  | Exact Match | Close Match |
|------------------------------|-------|-------|-------------|-------------|
| Random Forest                | $18.2 | $28.1 | 15%         | 68%         |
| Gradient Boosting (tuned)    | $16.4 | $25.3 | 18%         | 71%         |
| **Final Ensemble (RF + GB)** | **$14.1** | **$22.4** | **22%**     | **78%**     |

### Hyperparameter Tuning
- Gradient Boosting best: n_estimators=300, learning_rate=0.05, max_depth=3

### Final Model Selection
Best performing model: Ensemble of tuned Random Forest + Gradient Boosting  
- Test MAE: $14.10  
- Exact match rate: 22%  
- Close match rate: 78%

## 6. RESULTS
### Feature Importance (Ensemble Average)
1. total_receipts_amount        0.412
2. log_receipts                0.168
3. receipts_per_day            0.092
4. is_long_miles (>500)         0.071
5. trip_duration_days          0.058
6. miles_traveled              0.049
7. is_week_plus (>=7 days)     0.043
8. days × receipts             0.031
9. estimated_mileage           0.028
10. miles_per_day              0.022

### Prediction Analysis
- Average prediction error: $14.10
- 90% of errors fall within ±$38
- Largest errors on very high-receipt or extreme-mileage cases (legacy caps suspected)

## 7. BUSINESS INSIGHTS
### Discovered Business Logic
1. **Base Calculation**: Reimbursement ≈ total_receipts + mileage allowance, but receipts dominate
2. **Threshold Effects**: Clear rule changes at ~500 miles and ≥7 days (different per-diem/mileage logic)
3. **Special Cases**: Evidence of hard caps near $2,000–$2,200 and manual-override-like behavior on outliers

### Cost Drivers
1. Receipt amounts: ~41% of variance
2. Trip duration & mileage thresholds: ~20% combined
3. Interaction effects: ~15%

### Policy Insights
The legacy system effectively reimburses nearly 100% of receipts up to a soft cap, then adds tiered mileage and per-diem only on longer trips — inconsistent with modern IRS guidelines.

## 8. RECOMMENDATIONS
### Immediate Actions
1. Deploy the ensemble model as a “legacy matcher” for discrepancy audits
2. Flag all cases where new policy differs >$50 from legacy for manual review
3. Document discovered thresholds (500 miles, 7 days) in new policy

### Long-term Strategy
1. Migrate to transparent IRS-based policy (0.655/mile + fixed per diem)
2. Retire legacy system after 12-month parallel run

### Risk Mitigation
1. Regular model retraining with new submissions
2. SHAP explanations for every disputed claim

## 9. CONCLUSION
We successfully reverse-engineered ACME Corporation's legacy reimbursement system using machine learning. Our ensemble model achieves 22% exact and 78% close-match accuracy and provides interpretable insights into the system's business logic.

Key achievements:
- Replicated 60-year-old undocumented logic with <$15 MAE
- Uncovered hidden thresholds and receipt-dominant behavior
- Delivered production-ready predictor and full interpretability pipeline

## 10. APPENDICES
### Appendix A: Complete Feature List
36 total (4 original + 32 engineered (see feature_engineering.py)
### Appendix B: Model Hyperparameters
Available in models/final_ensemble.joblib metadata
### Appendix C: Error Analysis
Detailed in notebooks/06_error_analysis.ipynb
### Appendix D: Code Repository
https://github.com/dteran/acme-legacy-reimbursement

**References**
1. Scikit-learn Documentation
2. XGBoost Documentation
3. IRS Standard Mileage Rates 2025
4. Corporate Travel Policy Best Practices