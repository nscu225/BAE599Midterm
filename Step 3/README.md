Step 3 - Model selection, training, and evaluation

This folder trains a simple linear regression to answer: is U.S. rice yield (pounds per acre) changing over time?

Files:
- train_and_eval.py : trains a LinearRegression model (YEAR -> VALUE_clipped), saves model and metrics, and produces plots:
  - `plots/pred_vs_actual.png` : Predicted vs Actual scatter with y=x reference line
  - `plots/yield_trend_with_fit.png` : Time series of mean yield with fitted linear line
- metrics.txt : model coefficients and performance metrics (created when script runs)
- linear_model_year_to_yield.joblib : saved trained model (created when script runs)

Rationale summary (full explanation also included in the report):
- Problem type: Regression (predicting continuous yield values).
- Model choice: Ordinary least squares / Linear Regression.
  - Good because the research question focuses on trend over time; slope gives a direct interpretable rate of change (pounds per acre per year).
  - Assumptions: linearity, independence of errors, homoscedasticity, approximate normality of residuals — these can be checked with residual plots and (if statsmodels is available) OLS diagnostics.
  - Chosen over alternatives (e.g., RandomForest, SVR) because of interpretability, simplicity, and small sample size (one observation per year for national series).

How to run:

python3 "Step 3/train_and_eval.py"

Outputs will be written to `Step 3/` and `Step 3/plots/`.
