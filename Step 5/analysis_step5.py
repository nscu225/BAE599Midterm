import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# statsmodels for diagnostics
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

cwd = os.getcwd()
clean_path = os.path.join(cwd, 'Step 2', 'cleaned_yield_data.csv')
best_model_path = os.path.join(cwd, 'Step 4', 'best_model.joblib')
output_dir = os.path.join(cwd, 'Step 5')
plots_dir = os.path.join(output_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

if not os.path.exists(clean_path):
    print('ERROR: cleaned data not found:', clean_path)
    sys.exit(1)
if not os.path.exists(best_model_path):
    print('ERROR: best model not found:', best_model_path)
    sys.exit(1)

# Load data
df = pd.read_csv(clean_path)
mask = (df['is_us_total'] == True) & (df['is_all_classes'] == True)
df_us = df[mask].copy()
if df_us.empty:
    df_us = df[(df['LOCATION_DESCRIPTION'].str.upper()=='U.S. TOTAL') & (df['CLASS_DESCRIPTION'].str.upper()=='ALL CLASSES')].copy()

df_year = df_us.sort_values('YEAR').groupby('YEAR').agg({'VALUE_clipped':'mean'}).reset_index()
X = df_year[['YEAR']].values
y = df_year['VALUE_clipped'].values
n = len(X)

# reproduce time-based split
test_size = max(1, int(np.ceil(0.2 * n)))
train_end = n - test_size
X_train, X_test = X[:train_end], X[train_end:]
y_train, y_test = y[:train_end], y[train_end:]

# load model
best = joblib.load(best_model_path)

# predictions
y_pred_train = best.predict(X_train)
y_pred_test = best.predict(X_test)

# metrics
r2_train = r2_score(y_train, y_pred_train) if len(y_train)>1 else float('nan')
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train)) if len(y_train)>1 else float('nan')
mae_train = mean_absolute_error(y_train, y_pred_train) if len(y_train)>1 else float('nan')

r2_test = r2_score(y_test, y_pred_test) if len(y_test)>1 else float('nan')
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test)) if len(y_test)>1 else float('nan')
mae_test = mean_absolute_error(y_test, y_pred_test) if len(y_test)>1 else float('nan')

# Residuals on test
resid_test = y_test - y_pred_test

# OLS on full series for inference
X_sm = sm.add_constant(X)
ols = sm.OLS(y, X_sm).fit()
# Newey-West (HAC) robust covariance
try:
    ols_nw = ols.get_robustcov_results(cov_type='HAC', maxlags=4)
    nw_text = ols_nw.summary().as_text()
except Exception as e:
    ols_nw = None
    nw_text = f'Newey-West robust covariance failed: {e}'

# Durbin-Watson on OLS residuals
dw = durbin_watson(ols.resid)

# Plots
# 1) Residuals vs fitted (test)
plt.figure(figsize=(6,4))
plt.scatter(y_pred_test, resid_test, color='tab:blue')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Predicted (test)')
plt.ylabel('Residual (test)')
plt.title('Residuals vs Predicted (Test set)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'residuals_vs_pred_test.png'))
plt.close()

# 2) Residual histogram (test)
plt.figure(figsize=(6,4))
sns.histplot(resid_test, kde=True)
plt.xlabel('Residual (test)')
plt.title('Residual distribution (Test set)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'resid_hist_test.png'))
plt.close()

# 3) QQ plot of OLS residuals
qq = sm.qqplot(ols.resid, line='s')
plt.title('QQ plot of OLS residuals (full series)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'qq_ols_resid.png'))
plt.close()

# 4) ACF plot of OLS residuals
from statsmodels.graphics.tsaplots import plot_acf
plt.figure(figsize=(6,4))
plot_acf(ols.resid, lags=20, alpha=0.05)
plt.title('ACF of OLS residuals')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'acf_ols_resid.png'))
plt.close()

# 5) Predicted vs Actual (test) plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_test, color='tab:green')
lims = [min(min(y_test), min(y_pred_test)), max(max(y_test), max(y_pred_test))]
plt.plot(lims, lims, 'k--', alpha=0.7)
plt.xlabel('Actual yield (pounds per acre)')
plt.ylabel('Predicted yield (pounds per acre)')
plt.title('Predicted vs Actual (Test set)')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'pred_vs_actual_test_step5.png'))
plt.close()

# Write analysis text
analysis_path = os.path.join(output_dir, 'analysis.txt')
with open(analysis_path, 'w') as f:
    f.write(f'Analysis generated on {datetime.now().isoformat()}\n\n')
    f.write('Data split:\n')
    f.write(f'  Total years: {n}, train: {len(y_train)}, test: {len(y_test)}\n\n')
    f.write('Performance metrics:\n')
    f.write(f'  Train R^2: {r2_train:.4f}, RMSE: {rmse_train:.3f}, MAE: {mae_train:.3f}\n')
    f.write(f'  Test R^2: {r2_test:.4f}, RMSE: {rmse_test:.3f}, MAE: {mae_test:.3f}\n\n')
    f.write('OLS (full series) results:\n')
    f.write(ols.summary().as_text())
    f.write('\n\n')
    f.write('Newey-West (HAC) robust summary (maxlags=4):\n')
    f.write(nw_text + '\n\n')
    f.write(f'Durbin-Watson statistic (OLS residuals): {dw:.4f}\n\n')
    f.write('Interpretation:\n')
    f.write('  - Slope from OLS indicates average change in yield per year (pounds/acre per year).\n')
    f.write('  - The model shows a positive long-term trend (slope ~), but test R^2 is negative in earlier evaluation,\n')
    f.write('    indicating forecasting performance on the held-out recent years was poor relative to predicting the test mean.\n')
    f.write('  - Durbin-Watson < 2 indicates positive autocorrelation in residuals; this violates OLS independence assumption and\n')
    f.write('    can make standard errors optimistic. The Newey-West results provide HAC-robust standard errors to account for this.\n')
    f.write('\nRecommendations:\n')
    f.write('  - Use time-series models (ARIMA/SARIMAX or GLS with AR errors) to capture autocorrelation and potential nonstationarity.\n')
    f.write('  - Use rolling-origin (expanding window) evaluation to better estimate forecasting performance.\n')
    f.write('  - Investigate structural breaks (e.g., via Chow test) or include covariates (technology, inputs, weather) if available.\n')

print('Step 5 analysis complete.')
print('Saved analysis text to', analysis_path)
print('Saved plots to', plots_dir)
