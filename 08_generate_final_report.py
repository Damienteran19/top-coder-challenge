"""
Session 8: Final Report and Documentation Generation
CSCI/DASC 6020 - Machine Learning Team Project

This script generates the final technical report and presentation materials.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_executive_summary():
    """
    Generate executive summary for business stakeholders
    """
    summary = """
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
- Achieved XX% exact match rate (±$0.01)
- Achieved XX% close match rate (±$1.00)
- Mean prediction error: $XX.XX

### Business Impact
1. **System Understanding**: We've decoded the hidden business logic in the 
   legacy system, revealing the key factors that drive reimbursement decisions.

2. **Predictability**: Our model can predict reimbursement amounts with 
   XX% accuracy, providing transparency for employees and management.

3. **Cost Insights**: Analysis reveals the average reimbursement is $XXX 
   with primary cost drivers being:
   - Receipt amounts (XX% of variance)
   - Trip duration (XX% of variance)
   - Miles traveled (XX% of variance)

### Recommendations
1. **Policy Modernization**: The legacy system appears to use [identified pattern].
   We recommend [specific recommendation].

2. **Process Improvement**: Implement the new predictive system to provide 
   employees with instant reimbursement estimates.

3. **Cost Control**: [Specific recommendations based on patterns discovered]

### Next Steps
1. Validate model on additional historical data
2. Implement real-time prediction API
3. Monitor model performance in production
4. Periodic model retraining with new data

---
Report Generated: {date}
Team: [Your Team Names]
"""
    
    return summary.format(date=datetime.now().strftime("%B %d, %Y"))

def load_all_results():
    """
    Load all results from previous sessions
    """
    results = {}
    
    # Load model results
    try:
        results['baseline'] = pd.read_csv('results/baseline_model_results.csv')
    except:
        print("Warning: baseline results not found")
    
    try:
        results['advanced'] = pd.read_csv('results/advanced_model_results.csv')
    except:
        print("Warning: advanced results not found")
    
    try:
        results['final'] = pd.read_csv('results/final_model_results.csv')
    except:
        print("Warning: final results not found")
    
    # Load production predictions
    try:
        results['predictions'] = pd.read_csv('results/production_predictions.csv')
    except:
        print("Warning: production predictions not found")
    
    return results

def generate_technical_report(results):
    """
    Generate detailed technical report
    """
    report = f"""
# TECHNICAL REPORT
## Legacy Reimbursement System Reverse Engineering

**Project:** CSCI/DASC 6020 Machine Learning Team Project
**Team:** [Your Team Names]
**Date:** {datetime.now().strftime("%B %d, %Y")}

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
We created {len(results.get('features', []))} derived features including:

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

"""
    
    return report

def create_presentation_outline():
    """
    Create outline for business presentation
    """
    outline = """
# PRESENTATION OUTLINE
## Legacy Reimbursement System Reverse Engineering

**Duration:** 20 minutes + 5 minutes Q&A

---

## SLIDE 1: Title Slide
- Project Title
- Team Members
- Date
- Company Logo

---

## SLIDE 2: Problem Statement (2 minutes)
- ACME's 60-year-old legacy system
- No documentation, hidden business logic
- Need for transparency and modernization
- Project objectives

**Key Message:** "We're decoding a black box system using machine learning"

---

## SLIDE 3: Our Approach (2 minutes)
- Data-driven methodology
- 1,000 historical cases analyzed
- Multiple ML models tested
- Rigorous validation process

**Visual:** Process flowchart

---

## SLIDE 4: Data Insights (3 minutes)
- Input variables: Days, Miles, Receipts
- Output: Reimbursement amount
- Data distribution and patterns
- Key correlations discovered

**Visuals:** 
- Distribution histograms
- Correlation heatmap

---

## SLIDE 5: Feature Engineering (2 minutes)
- Created XX derived features
- Rate-based calculations
- Business rule simulations
- Interaction effects

**Key Message:** "Smart features = Better predictions"

---

## SLIDE 6: Model Development (3 minutes)
- Tested 10+ different models
- From simple (linear) to complex (neural networks)
- Rigorous cross-validation
- Ensemble methods for robustness

**Visual:** Model comparison chart

---

## SLIDE 7: Results (3 minutes)
- Best model: [Model Name]
- XX% exact match rate (±$0.01)
- XX% close match rate (±$1.00)
- Average error: $XX

**Visual:** 
- Performance metrics bar chart
- Predicted vs Actual scatter plot

---

## SLIDE 8: Business Logic Discovered (3 minutes)
- Primary cost drivers identified
- Threshold effects revealed
- Special case handling decoded
- Policy implications

**Key Message:** "We've decoded the hidden formula"

---

## SLIDE 9: Recommendations (2 minutes)
1. Implement prediction system for transparency
2. Modernize policy based on insights
3. Monitor and update regularly

**Key Message:** "From black box to glass box"

---

## SLIDE 10: Conclusion & Next Steps (1 minute)
- Successfully reverse-engineered system
- High-accuracy predictions achieved
- Actionable insights delivered
- Ready for deployment

**Call to Action:** "Let's modernize ACME's reimbursement system"

---

## BACKUP SLIDES
- Technical details
- Additional visualizations
- Error analysis
- Feature importance details
- Implementation timeline
- ROI analysis

"""
    
    return outline

def generate_all_visualizations(results):
    """
    Generate all visualizations for the final report
    """
    print("\n" + "="*80)
    print("GENERATING FINAL VISUALIZATIONS")
    print("="*80)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Legacy Reimbursement System - Complete Analysis', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # Load data for visualizations
    try:
        test_df = pd.read_csv('data/processed/test_data.csv')
        pred_df = pd.read_csv('results/production_predictions.csv')
        
        # 1. Input distributions
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(test_df['trip_duration_days'], bins=30, edgecolor='black', alpha=0.7)
        ax1.set_title('Trip Duration Distribution')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Frequency')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(test_df['miles_traveled'], bins=30, edgecolor='black', alpha=0.7)
        ax2.set_title('Miles Traveled Distribution')
        ax2.set_xlabel('Miles')
        ax2.set_ylabel('Frequency')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(test_df['total_receipts_amount'], bins=30, edgecolor='black', alpha=0.7)
        ax3.set_title('Receipts Distribution')
        ax3.set_xlabel('Amount ($)')
        ax3.set_ylabel('Frequency')
        
        # 2. Model comparison
        if 'final' in results and results['final'] is not None:
            ax4 = fig.add_subplot(gs[1, :])
            models = results['final']['model'].values
            maes = results['final']['test_mae'].values
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
            bars = ax4.barh(models, maes, color=colors)
            ax4.set_xlabel('Mean Absolute Error ($)')
            ax4.set_title('Model Performance Comparison')
            ax4.invert_yaxis()
            
            # Add values on bars
            for bar in bars:
                width = bar.get_width()
                ax4.text(width, bar.get_y() + bar.get_height()/2, 
                        f'${width:.2f}', ha='left', va='center')
        
        # 3. Predicted vs Actual
        ax5 = fig.add_subplot(gs[2, :2])
        ax5.scatter(pred_df['output'], pred_df['predicted'], alpha=0.5, s=30)
        ax5.plot([pred_df['output'].min(), pred_df['output'].max()],
                [pred_df['output'].min(), pred_df['output'].max()],
                'r--', lw=2, label='Perfect Prediction')
        ax5.set_xlabel('Actual Reimbursement ($)')
        ax5.set_ylabel('Predicted Reimbursement ($)')
        ax5.set_title('Predicted vs Actual (Test Set)')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 4. Error distribution
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.hist(pred_df['error'], bins=50, edgecolor='black', alpha=0.7)
        ax6.set_xlabel('Absolute Error ($)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Prediction Error Distribution')
        ax6.axvline(1.0, color='r', linestyle='--', label='±$1.00 threshold')
        ax6.legend()
        
        # 5. Error vs inputs
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.scatter(pred_df['trip_duration_days'], pred_df['error'], alpha=0.5, s=20)
        ax7.set_xlabel('Trip Duration (Days)')
        ax6.set_ylabel('Absolute Error ($)')
        ax7.set_title('Error vs Trip Duration')
        ax7.grid(alpha=0.3)
        
        ax8 = fig.add_subplot(gs[3, 1])
        ax8.scatter(pred_df['miles_traveled'], pred_df['error'], alpha=0.5, s=20)
        ax8.set_xlabel('Miles Traveled')
        ax8.set_ylabel('Absolute Error ($)')
        ax8.set_title('Error vs Miles')
        ax8.grid(alpha=0.3)
        
        ax9 = fig.add_subplot(gs[3, 2])
        ax9.scatter(pred_df['total_receipts_amount'], pred_df['error'], alpha=0.5, s=20)
        ax9.set_xlabel('Receipt Amount ($)')
        ax9.set_ylabel('Absolute Error ($)')
        ax9.set_title('Error vs Receipts')
        ax9.grid(alpha=0.3)
        
        # 6. Performance metrics summary
        ax10 = fig.add_subplot(gs[4, :])
        metrics = ['MAE', 'RMSE', 'Exact Match %', 'Close Match %', 'R²']
        values = [
            pred_df['error'].mean(),
            np.sqrt((pred_df['error'] ** 2).mean()),
            (pred_df['error'] <= 0.01).mean() * 100,
            (pred_df['error'] <= 1.00).mean() * 100,
            1 - (pred_df['error'] ** 2).sum() / ((pred_df['output'] - pred_df['output'].mean()) ** 2).sum()
        ]
        
        colors_metric = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'gold']
        bars = ax10.bar(metrics, values, color=colors_metric, alpha=0.7, edgecolor='black')
        ax10.set_ylabel('Value')
        ax10.set_title('Final Model Performance Metrics')
        ax10.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width()/2., height,
                     f'{val:.2f}', ha='center', va='bottom')
        
        # 7. Summary statistics table
        ax11 = fig.add_subplot(gs[5, :])
        ax11.axis('off')
        
        summary_data = [
            ['Total Test Cases', f"{len(pred_df)}"],
            ['Mean Absolute Error', f"${pred_df['error'].mean():.2f}"],
            ['Median Error', f"${pred_df['error'].median():.2f}"],
            ['Exact Matches (±$0.01)', f"{(pred_df['error'] <= 0.01).sum()} ({(pred_df['error'] <= 0.01).mean()*100:.1f}%)"],
            ['Close Matches (±$1.00)', f"{(pred_df['error'] <= 1.00).sum()} ({(pred_df['error'] <= 1.00).mean()*100:.1f}%)"],
            ['Max Error', f"${pred_df['error'].max():.2f}"],
            ['Avg Reimbursement', f"${pred_df['output'].mean():.2f}"],
        ]
        
        table = ax11.table(cellText=summary_data, cellLoc='left',
                          colWidths=[0.5, 0.5],
                          loc='center', bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_data)):
            table[(i, 0)].set_facecolor('#E8E8E8')
            table[(i, 0)].set_text_props(weight='bold')
        
        ax11.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
    except Exception as e:
        print(f"Warning: Could not generate some visualizations: {e}")
    
    plt.savefig('reports/figures/08_comprehensive_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Comprehensive visualization created")

def save_final_report(exec_summary, tech_report, pres_outline):
    """
    Save all report documents
    """
    print("\n" + "="*80)
    print("SAVING FINAL REPORTS")
    print("="*80)
    
    # Save executive summary
    with open('reports/executive_summary.md', 'w') as f:
        f.write(exec_summary)
    print("✓ Executive summary saved")
    
    # Save technical report
    with open('reports/technical_report.md', 'w') as f:
        f.write(tech_report)
    print("✓ Technical report saved")
    
    # Save presentation outline
    with open('reports/presentation_outline.md', 'w') as f:
        f.write(pres_outline)
    print("✓ Presentation outline saved")
    
    print("\nReports saved to reports/ directory")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("SESSION 8: FINAL REPORT GENERATION")
    print("="*80)
    
    # Load all results
    print("\nLoading results from previous sessions...")
    results = load_all_results()
    
    # Generate executive summary
    print("\nGenerating executive summary...")
    exec_summary = generate_executive_summary()
    
    # Generate technical report
    print("Generating technical report...")
    tech_report = generate_technical_report(results)
    
    # Generate presentation outline
    print("Generating presentation outline...")
    pres_outline = create_presentation_outline()
    
    # Generate visualizations
    generate_all_visualizations(results)
    
    # Save everything
    save_final_report(exec_summary, tech_report, pres_outline)
    
    print("\n" + "="*80)
    print("SESSION 8 COMPLETE!")
    print("="*80)
    print("\nAll reports generated successfully!")
    print("\nGenerated Files:")
    print("- reports/executive_summary.md")
    print("- reports/technical_report.md")
    print("- reports/presentation_outline.md")
    print("- reports/figures/08_comprehensive_analysis.png")
    print("\nProject Complete! 🎉")
    print("\nFinal Checklist:")
    print("☐ Review all reports")
    print("☐ Update README with final results")
    print("☐ Create presentation slides")
    print("☐ Practice presentation")
    print("☐ Prepare for Q&A")

if __name__ == "__main__":
    main()
