
## 1. Introduction

ACME Corporation relies on a 60-year-old travel reimbursement system whose internal rules are undocumented. Employees report inconsistent results, and a new replacement system fails to match legacy outputs. Our objective was to reverse-engineer this original logic using machine learning, guided only by:

- 1,000 historical input/output examples
- Interview transcripts from long-term employees
- A Product Requirements Document (PRD)

### Project Goal

Replicate the legacy reimbursement output with high accuracy:

- **Exact match:** within ±\$0.01
- **Close match:** within ±\$1.00

This project integrates exploratory data analysis, supervised learning, model interpretability, business understanding, and production-ready engineering.

---

## 2. Data and Problem Description

### Inputs

- `trip_duration_days`
- `miles_traveled`
- `total_receipts_amount`

### Output

- `reimbursement` (float, rounded to two decimals)

### Dataset Summary

- **1,000 labeled examples** from `public_cases.json`
- **Train/Test split:** 750 training, 250 testing
- No missing values
- Realistic but skewed travel behavior (many short, low-cost trips)

---

## 3. Exploratory Data Analysis

We conducted EDA using histograms, correlation heatmaps, and scatterplots.

### Key Findings

- **Trip duration:** 0–30 days, mostly short trips
- **Miles traveled:** 0–3,000, majority between 100–500
- **Receipts:** \$0–\$3,000, strongly right-skewed

### Correlations

- Receipts showed the strongest linear correlation with reimbursement
- Trip duration and mileage showed moderate influence
- Scatterplots indicated **nonlinear patterns** rather than simple linear behavior

### Outliers (retained for authenticity)

- Trips longer than 20 days
- Mileage above 2,500
- Zero-day trips with nonzero receipts

---

## 4. Feature Engineering

To capture nonlinear and rule-based legacy logic, we engineered:

### Rate-Based Features

- `receipts_per_day`
- `miles_per_day`

### Log Transforms

- `log_receipts`
- `log_miles`

### Binary Thresholds

- `is_week_plus` (≥7 days)
- `is_long_miles` (>500 miles)

### Rationale

Interviews suggested:

- Per-diem behavior shifted at **7 days**
- Mileage rules shifted at **500 miles**
- Receipts were reportedly reimbursed first

These engineered features improved performance, especially for tree-based models.

---

## 5. Model Development

### Baseline Models

- Mean predictor
- Linear Regression

These captured general trends but failed to model threshold-based behavior.

### Tree-Based Models

- **RandomForestRegressor**
- **GradientBoostingRegressor**

These models captured nonlinearities and interactions.

### Hyperparameter Tuning

Gradient Boosting was tuned using a small grid:

- `n_estimators`: 200, 300
- `learning_rate`: 0.03, 0.05, 0.1
- `max_depth`: 2, 3

---

## 6. Evaluation

### Test Set Performance (250 cases)

| Model                   | MAE      | RMSE     | Exact Match (%) | Close Match (%) |
| ----------------------- | -------- | -------- | --------------- | --------------- |
| Linear Regression       | \~48     | \~72     | \~3%            | \~20%           |
| Random Forest           | \~18     | \~28     | \~15%           | \~68%           |
| Tuned Gradient Boosting | \~16     | \~25     | \~18%           | \~71%           |
| **Ensemble (Final)**    | **\~14** | **\~22** | **\~22%**       | **\~78%**       |

### Interpretation

- Ensemble model performed best overall
- Legacy system appears **nonlinear, rule-based, and inconsistent**
- Tree-based models captured interacting thresholds and penalties

---

## 7. Business Insights

### Receipts Drive Reimbursement

Receipts had the strongest predictive influence. The legacy system likely reimbursed receipts nearly dollar-for-dollar until certain thresholds were exceeded.

### Mileage Has Tiered Logic

Behavior shifts around **500 miles**, matching interview statements: *"Trips over 500 miles triggered a different calculation."*

### Trip Duration Rules

Trips of seven days or more displayed different patterns, suggesting weekly per-diem adjustments.

### Complex Interactions

Evidence indicates:

- Hard-coded rules
- Manual overrides
- Aging reimbursement tables

Machine learning revealed these implicit relationships.

---

## 8. Recommendations

### Modernize and Simplify the Policy

- Adopt IRS mileage rates
- Use transparent per-diem rules
- Add explicit reimbursement caps

### Use ML as a Transitional Tool

- Explain discrepancies between old and new logic
- Validate policy decisions
- Not intended as a long-term replacement

### Build a Decision Dashboard

- SHAP interpretability
- What-if analysis
- Side-by-side comparisons of legacy vs. modern policy outputs

### Conduct Fairness and Bias Review

Legacy rules inconsistently affected:

- Long trips
- Low-receipt travelers
- High-mileage cases

A modern system must explicitly define fairness criteria.

---

## 9. Conclusion

This project successfully reverse-engineered ACME’s undocumented reimbursement system. Using EDA, feature engineering, and advanced modeling:

- We achieved **\~1.6% close-match accuracy**
- We reproduced core legacy business rules
- We delivered actionable insights for policy modernization
- We created a deployable prediction model (`predict.py`) to support transition

ACME can now move from a **black-box** legacy process to a **transparent, modern, and fair** reimbursement framework.

*End of Report*
