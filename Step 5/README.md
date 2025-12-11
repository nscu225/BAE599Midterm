Step 5 - Model performance analysis and diagnostics

This folder contains a script that analyzes model performance and residual diagnostics.

Files:
- analysis_step5.py : Runs diagnostics (residuals, QQ, ACF), computes OLS and Newey-West robust results, and saves plots and a written analysis.
- analysis.txt : Written analysis produced by the script (created when script runs).
- plots/ : contains diagnostic figures:
  - residuals_vs_pred_test.png
  - resid_hist_test.png
  - qq_ols_resid.png
  - acf_ols_resid.png
  - pred_vs_actual_test_step5.png

How to run:

python3 "Step 5/analysis_step5.py"

Notes:
- The script expects `Step 2/cleaned_yield_data.csv` and `Step 4/best_model.joblib` to exist.
- The script uses statsmodels for inference and diagnostics.
