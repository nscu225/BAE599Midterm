import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Attempt to import statsmodels for inference
try:
    import statsmodels.api as sm
    statsmodels_available = True
except Exception:
    statsmodels_available = False

cwd = os.getcwd()
clean_path = os.path.join(cwd, 'Step 2', 'cleaned_yield_data.csv')
output_dir = os.path.join(cwd, 'Step 3')
plots_dir = os.path.join(output_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

if not os.path.exists(clean_path):
    print('ERROR: cleaned data not found at', clean_path)
    sys.exit(1)

# Read cleaned data
df = pd.read_csv(clean_path)

# Focus: national yield time series (U.S. TOTAL, ALL CLASSES)
# The cleaned CSV produced by the in-app minimal pipeline may not contain the boolean flags.
# Handle multiple possible formats gracefully.
if 'is_us_total' in df.columns and 'is_all_classes' in df.columns:
    mask = (df['is_us_total'] == True) & (df['is_all_classes'] == True)
    df_us = df[mask].copy()
else:
    # Try fallback filters using descriptive columns if present
    if 'LOCATION_DESCRIPTION' in df.columns and 'CLASS_DESCRIPTION' in df.columns:
        df_us = df[(df['LOCATION_DESCRIPTION'].str.upper()=='U.S. TOTAL') & (df['CLASS_DESCRIPTION'].str.upper()=='ALL CLASSES')].copy()
    else:
        # If the dataframe appears to be already aggregated to YEAR/VALUE (minimal cleaned), use it directly
        if set(['YEAR','VALUE','VALUE_clipped']).intersection(df.columns):
            df_us = df.copy()
        else:
            # As a last resort, try to detect a location or class column that suggests national totals
            df_us = df.copy()

# Ensure year order and uniqueness
df_us = df_us.sort_values('YEAR')
# If there are duplicates per year, average
# Some cleaned files may not have VALUE_clipped (winsorized). Fall back to VALUE if missing.
if 'VALUE_clipped' in df_us.columns:
    df_us_agg = df_us.groupby('YEAR').agg({'VALUE':'mean', 'VALUE_clipped':'mean'}).reset_index()
    df_us_agg = df_us_agg.rename(columns={'VALUE_clipped':'VALUE_used'})
else:
    df_us_agg = df_us.groupby('YEAR').agg({'VALUE':'mean'}).reset_index()
    df_us_agg = df_us_agg.rename(columns={'VALUE':'VALUE_used'})

X = df_us_agg[['YEAR']].values
# Use the value column chosen above (VALUE_clipped when available, otherwise VALUE)
y = df_us_agg['VALUE_used'].values

# Fit linear regression
lr = LinearRegression()
lr.fit(X, y)
pred = lr.predict(X)

# Metrics
r2 = r2_score(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))
mae = mean_absolute_error(y, pred)

# Save model
model_path = os.path.join(output_dir, 'linear_model_year_to_yield.joblib')
joblib.dump(lr, model_path)

# Save metrics
metrics_path = os.path.join(output_dir, 'metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(f'Transformation and model run at {datetime.now().isoformat()}\n')
    f.write('Model: LinearRegression predicting VALUE_clipped from YEAR\n')
    f.write(f'Number of years: {len(X)}\n')
    f.write(f'Coefficients: slope={lr.coef_[0]:.6f}, intercept={lr.intercept_:.4f}\n')
    f.write(f'R^2: {r2:.4f}\n')
    f.write(f'RMSE: {rmse:.3f}\n')
    f.write(f'MAE: {mae:.3f}\n')

# If statsmodels available, fit OLS to get p-value and summary
if statsmodels_available:
    X_sm = sm.add_constant(X)
    ols_res = sm.OLS(y, X_sm).fit()
    with open(metrics_path, 'a') as f:
        f.write('\nStatsmodels OLS summary:\n')
        f.write(ols_res.summary().as_text())
else:
    with open(metrics_path, 'a') as f:
        f.write('\nstatsmodels not available; OLS summary skipped.\n')

# Predicted vs Actual plot
plt.figure(figsize=(6,6))
plt.scatter(y, pred, color='tab:blue')
lims = [min(min(y), min(pred)), max(max(y), max(pred))]
plt.plot(lims, lims, 'k--', alpha=0.7)
plt.xlabel('Actual yield (pounds per acre)')
plt.ylabel('Predicted yield (pounds per acre)')
plt.title('Predicted vs Actual - Linear Regression (Year -> Yield)')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plot_path = os.path.join(plots_dir, 'pred_vs_actual.png')
plt.savefig(plot_path)
plt.close()

# Also save a plot of time series with fitted line
plt.figure(figsize=(10,5))
plt.scatter(df_us_agg['YEAR'], df_us_agg['VALUE_used'], label='Actual (clipped or raw)', color='tab:gray')
plt.plot(df_us_agg['YEAR'], pred, label='Linear fit', color='tab:orange')
plt.xlabel('Year')
plt.ylabel('Yield (pounds per acre)')
plt.title('U.S. Mean Yield by Year with Linear Fit')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'yield_trend_with_fit.png'))
plt.close()

print('Training and evaluation complete.')
print('Saved model to', model_path)
print('Saved plots to', plots_dir)
print('Saved metrics to', metrics_path)
