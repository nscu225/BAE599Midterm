Step 4 - Model implementation, training, cross-validation, and learning curve

This folder contains a script to implement and evaluate models using the prepared data from Step 2.

Script: `analysis_step4.py`
- Loads `Step 2/cleaned_yield_data.csv` and filters national series (U.S. TOTAL, ALL CLASSES).
- Aggregates to one observation per year (mean of `VALUE_clipped`).
- Creates a time-based train/test split (last 20% of years used as test set).
- Uses `TimeSeriesSplit` (5 splits) for cross-validation on the training data.
- Runs `GridSearchCV` over a pipeline: StandardScaler -> PolynomialFeatures (degree 1 or 2) -> Ridge (alpha grid).
- Saves the best model, metrics, and two plots:
  - `plots/pred_vs_actual_test.png` : Predicted vs Actual on the test set.
  - `plots/learning_curve.png` : Training vs cross-validation performance vs training size.

How to run:

python3 "Step 4/analysis_step4.py"

Notes on methods:
- TimeSeriesSplit keeps temporal order in CV folds so training folds are earlier in time than validation folds — prevents lookahead leakage.
- GridSearchCV searches hyperparameters (polynomial degree and ridge alpha) and selects the model minimizing CV mean squared error.
- Learning curve shows how training and CV R^2 change as training size increases; useful to detect high bias or variance.
