
# TECHNICAL REPORT
## Legacy Reimbursement System Reverse Engineering

**Project:** CSCI/DASC 6020 Machine Learning Team Project
**Team:** [Your Team Names]
**Date:** November 20, 2025

---

## TABLE OF CONTENTS
1. Introduction
2. Problem Statement
3. Data Analysis
4. Methodology
5. Model Development
6. Results
7. Business Insights
8. Recommendations
9. Conclusion
10. Appendices

---

## 1. INTRODUCTION

This project applies machine learning techniques to reverse-engineer ACME 
Corporation's legacy travel reimbursement system. The system has operated 
for 60 years without documentation, creating operational risks and limiting 
system transparency.

### Project Objectives
- Understand the hidden business logic in the legacy system
- Create a predictive model that replicates system behavior
- Provide explainable AI solutions for stakeholders
- Generate actionable business insights

---

## 2. PROBLEM STATEMENT

### Input Variables
The legacy system accepts three inputs:
- `trip_duration_days`: Number of days traveling (integer)
- `miles_traveled`: Total miles traveled (integer)
- `total_receipts_amount`: Total receipt amount (dollars)

### Output Variable
- Single reimbursement amount (dollars, 2 decimal places)

### Success Criteria
- Exact matches: Predictions within ±$0.01
- Close matches: Predictions within ±$1.00
- Minimize Mean Absolute Error (MAE)

---

## 3. DATA ANALYSIS

### Dataset Characteristics
- Total cases: 1,000
- Training set: 750 cases (75%)
- Test set: 250 cases (25%)
- Missing values: 0
- Outliers: [To be filled based on EDA]

### Descriptive Statistics

**Trip Duration (Days)**
- Mean: [Fill from data]
- Median: [Fill from data]
- Range: [Fill from data]

**Miles Traveled**
- Mean: [Fill from data]
- Median: [Fill from data]
- Range: [Fill from data]

**Total Receipts**
- Mean: $[Fill from data]
- Median: $[Fill from data]
- Range: $[Fill from data]

**Reimbursement Amount**
- Mean: $[Fill from data]
- Median: $[Fill from data]
- Range: $[Fill from data]

### Correlation Analysis
[Include correlation matrix and key findings]

---

## 4. METHODOLOGY

### Feature Engineering
We created 0 derived features including:

1. **Rate-based Features**
   - Cost per day
   - Cost per mile
   - Miles per day

2. **Domain-specific Features**
   - Estimated mileage reimbursement (IRS rate: $0.655/mile)
   - Estimated per diem ($150/day)

3. **Interaction Features**
   - Days × Miles
   - Days × Receipts
   - Miles × Receipts

4. **Polynomial Features**
   - Squared terms
   - Square root terms
   - Logarithmic terms

5. **Threshold Features**
   - Overnight trip indicators
   - Long-distance trip indicators
   - High-receipt indicators

### Model Selection
We evaluated multiple model families:

1. **Linear Models**
   - Linear Regression
   - Ridge Regression (L2)
   - Lasso Regression (L1)

2. **Tree-based Models**
   - Decision Trees
   - Random Forest
   - Gradient Boosting
   - XGBoost

3. **Neural Networks**
   - Multi-layer Perceptron

4. **Ensemble Methods**
   - Voting Ensemble
   - Stacking Ensemble
   - Weighted Average Ensemble

### Evaluation Framework
- 5-fold cross-validation
- Train/test split validation
- Custom metrics aligned with business objectives

---

## 5. MODEL DEVELOPMENT

### Baseline Models
[Table of baseline model results]

### Advanced Models
[Table of advanced model results]

### Hyperparameter Tuning
We performed grid search and randomized search for:
- Random Forest: [best parameters]
- XGBoost: [best parameters]
- Gradient Boosting: [best parameters]

### Final Model Selection
Best performing model: [Model name]
- Test MAE: $[value]
- Exact match rate: [X]%
- Close match rate: [X]%

---

## 6. RESULTS

### Model Performance Comparison
[Include comparison table and visualizations]

### Feature Importance
Top 10 most important features:
1. [Feature 1]: [Importance]
2. [Feature 2]: [Importance]
...

### Prediction Analysis
- Average prediction error: $[X]
- Error distribution: [Description]
- Common error patterns: [Analysis]

---

## 7. BUSINESS INSIGHTS

### Discovered Business Logic
Based on our analysis, the legacy system appears to follow these rules:

1. **Base Calculation**: [Pattern discovered]
2. **Threshold Effects**: [Threshold patterns]
3. **Special Cases**: [Special case handling]

### Cost Drivers
Primary factors affecting reimbursement amounts:
1. [Factor 1]: Contributes [X]% of variance
2. [Factor 2]: Contributes [X]% of variance
3. [Factor 3]: Contributes [X]% of variance

### Policy Insights
[Insights about company travel policy based on patterns]

---

## 8. RECOMMENDATIONS

### Immediate Actions
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]

### Long-term Strategy
1. [Strategic recommendation 1]
2. [Strategic recommendation 2]

### Risk Mitigation
1. [Risk mitigation strategy 1]
2. [Risk mitigation strategy 2]

---

## 9. CONCLUSION

We successfully reverse-engineered ACME Corporation's legacy reimbursement 
system using machine learning. Our [best model] achieves [X]% accuracy and 
provides interpretable insights into the system's business logic.

Key achievements:
- [Achievement 1]
- [Achievement 2]
- [Achievement 3]

---

## 10. APPENDICES

### Appendix A: Complete Feature List
[Full list of engineered features]

### Appendix B: Model Hyperparameters
[Detailed hyperparameter settings]

### Appendix C: Error Analysis
[Detailed error analysis]

### Appendix D: Code Repository
[Link to code repository]

---

**References**
1. Scikit-learn Documentation
2. XGBoost Documentation
3. IRS Standard Mileage Rates
4. Corporate Travel Policy Guidelines

