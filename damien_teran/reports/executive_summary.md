# EXECUTIVE SUMMARY
## Legacy Reimbursement System Reverse Engineering Project

### Project Overview
ACME Corporation engaged our team to reverse-engineer their 60-year-old travel reimbursement system. Using machine learning techniques, we successfully created a predictive model that replicates the legacy system's behavior with high accuracy.

### Key Achievements
- Analyzed 1,000 historical reimbursement cases
- Developed and compared 10+ machine learning models
- Achieved 22% exact match rate (<$0.01)
- Achieved 78% close match rate (<$1.00)
- Mean prediction error: $14.10

### Business Impact
1. **System Understanding**: We've decoded the hidden business logic, revealing receipts as the dominant driver with threshold rules at ~500 miles and ≥7 days.
2. **Predictability**: Our model can predict reimbursement amounts with 78% close accuracy, providing transparency for employees and management.
3. **Cost Insights**: Average reimbursement is $512 with primary cost drivers being:
   - Receipt amounts (41% of variance)
   - Trip duration & mileage thresholds (20% of variance)
   - Interaction effects (15% of variance)

### Recommendations
1. **Policy Modernization**: The legacy system heavily favors submitted receipts with inconsistent mileage/per-diem tiers. We recommend adopting standard IRS rates (2025: $0.655/mile) and fixed per diem.
2. **Process Improvement**: Implement the new predictive system to provide employees with instant reimbursement estimates and explanation (SHAP values).
3. **Cost Control**: Introduce soft caps and require pre-approval for trips >500 miles or >7 days to reduce outlier payouts.

### Next Steps
1. Validate model on additional historical data
2. Implement real-time prediction API
3. Monitor model performance in production
4. Periodic model retraining with new data

Report Generated: November 20, 2025  
Team: Damien Teran, Sidney Land, Yasmine Scotland