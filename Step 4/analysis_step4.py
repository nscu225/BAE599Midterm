import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, learning_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

cwd = os.getcwd()
clean_path = os.path.join(cwd, 'Step 2', 'cleaned_yield_data.csv')
output_dir = os.path.join(cwd, 'Step 4')
plots_dir = os.path.join(output_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

if not os.path.exists(clean_path):
    print('ERROR: cleaned data not found:', clean_path)
    sys.exit(1)

# Read cleaned data
df = pd.read_csv(clean_path)
# Filter to national series
mask = (df['is_us_total'] == True) & (df['is_all_classes'] == True)
df_us = df[mask].copy()
if df_us.empty:
    df_us = df[(df['LOCATION_DESCRIPTION'].str.upper()=='U.S. TOTAL') & (df['CLASS_DESCRIPTION'].str.upper()=='ALL CLASSES')].copy()

# Aggregate by year (mean)
df_us = df_us.sort_values('YEAR')
df_year = df_us.groupby('YEAR').agg({'VALUE_clipped':'mean'}).reset_index()
X = df_year[['YEAR']].values
y = df_year['VALUE_clipped'].values

# Train/test split: time-based (last 20% years as test)
n = len(X)
test_size = max(1, int(np.ceil(0.2 * n)))
train_end = n - test_size
X_train, X_test = X[:train_end], X[train_end:]
y_train, y_test = y[:train_end], y[train_end:]

# Cross-validation strategy: TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Pipeline: scaler -> poly features -> ridge
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False)),
    ('clf', Ridge())
])

param_grid = {
    'poly__degree': [1, 2],
    'clf__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}

grid = GridSearchCV(pipe, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)

best = grid.best_estimator_
best_params = grid.best_params_
best_score_cv = -grid.best_score_

# Evaluate on test set
y_pred_test = best.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

# Save model
model_path = os.path.join(output_dir, 'best_model.joblib')
joblib.dump(best, model_path)

# Save metrics
metrics_path = os.path.join(output_dir, 'metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(f'Model selection run at {datetime.now().isoformat()}\n')
    f.write('Time-based train/test split; cross-validated on training set with TimeSeriesSplit (5 splits)\n')
    f.write(f'Number of years total: {n}; train: {len(X_train)}; test: {len(X_test)}\n')
    f.write(f'GridSearch best params: {best_params}\n')
    f.write(f'CV best MSE (neg->pos): {best_score_cv:.4f}\n')
    f.write('Test set performance:\n')
    f.write(f'R^2: {r2_test:.4f}\n')
    f.write(f'RMSE: {rmse_test:.3f}\n')
    f.write(f'MAE: {mae_test:.3f}\n')

# Predicted vs Actual on test
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_test, color='tab:blue')
lims = [min(min(y_test), min(y_pred_test)), max(max(y_test), max(y_pred_test))]
plt.plot(lims, lims, 'k--', alpha=0.7)
plt.xlabel('Actual yield (pounds per acre)')
plt.ylabel('Predicted yield (pounds per acre)')
plt.title('Predicted vs Actual (Test set)')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'pred_vs_actual_test.png'))
plt.close()

# Learning curve (training vs validation) using TimeSeriesSplit
train_sizes = np.linspace(0.1, 1.0, 10)
# learning_curve will use cv=tscv; set shuffle=False (default)
train_sizes_abs, train_scores, valid_scores = learning_curve(best, X_train, y_train, cv=tscv, train_sizes=train_sizes, scoring='r2', n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.figure(figsize=(8,5))
plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='tab:blue', label='Training score')
plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='tab:blue')
plt.plot(train_sizes_abs, valid_scores_mean, 'o-', color='tab:orange', label='Cross-validation score')
plt.fill_between(train_sizes_abs, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='tab:orange')
plt.xlabel('Training set size (n)')
plt.ylabel('R^2')
plt.title('Learning Curve (TimeSeriesSplit CV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'learning_curve.png'))
plt.close()

print('Step 4 complete.')
print('Saved best model to', model_path)
print('Saved metrics to', metrics_path)
print('Saved plots to', plots_dir)
