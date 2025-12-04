# EXECUTIVE SUMMARY
## Legacy Reimbursement System Reverse Engineering Project

### Project Overview
ACME Corporation engaged our team to reverse-engineer their 60-year-old travel 
reimbursement system. Using machine learning techniques, we successfully 
created a predictive model that replicates the legacy system's behavior with 
high accuracy.

### Key Achievements
- Analyzed 1,000 historical reimbursement cases
- Developed and compared 10+ machine learning models
- Achieved 98.87% variance explained (R² = 0.9887)
- Mean prediction error: $45.90 (3.4% of average reimbursement)
- 88% improvement over baseline predictions

### Business Impact
1. **System Understanding**: We've decoded the hidden business logic in the 
   legacy system, revealing the key factors that drive reimbursement decisions.

2. **Predictability**: Our Stacking Ensemble model predicts reimbursement amounts 
   with an average error of only $46, providing transparency for employees and management.

3. **Cost Insights**: Analysis reveals the average reimbursement is $1,337 
   with primary cost drivers being:
   - Receipt amounts (69% correlation with output)
   - Trip duration (50% correlation with output)
   - Miles traveled (44% correlation with output)

### Discovered Patterns
- **Receipt dominance**: Total receipts is the strongest predictor
- **Z-score features**: Standardized reimbursement and anomaly detection features 
  ranked highest in importance
- **Threshold effects**: High-receipt indicators and receipt caps significantly 
  influence predictions

### Recommendations
1. **Policy Modernization**: The legacy system heavily weighs receipt amounts with 
   secondary consideration for trip duration and mileage. We recommend implementing 
   transparent tiered reimbursement policies.

2. **Process Improvement**: Deploy the Stacking Ensemble model to provide 
   employees with instant reimbursement estimates before trip submission.

3. **Cost Control**: Monitor high-receipt cases and receipt cap thresholds, as 
   these are key cost drivers accounting for significant variance.

### Model Performance Summary
| Metric | Value |
|--------|-------|
| Best Model | Stacking Ensemble |
| Test MAE | $45.90 |
| Test RMSE | $48.59 |
| R² Score | 0.9887 (98.87%) |
| Average Error Rate | 3.4% |
| Improvement vs Baseline | 88% |

### Next Steps
1. Validate model on additional historical data
2. Implement real-time prediction API for employee use
3. Monitor model performance and establish retraining schedule
4. Conduct cost-benefit analysis for policy modernization

---
Report Generated: December 04, 2025
Team: [Damien Teran, Sidney Land, Yasmine Scotland]