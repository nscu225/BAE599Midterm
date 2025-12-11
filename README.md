# US Rice Production & Yield Analysis

**Research Question:** Is U.S. rice production becoming more efficient (yield per acre) over time?

## Project Overview

This project analyzes historical U.S. rice yield data from the USDA to understand productivity trends and build predictive models. The analysis follows a complete machine learning workflow from data inspection through model deployment.

## Dataset

- **Source:** USDA National Agricultural Statistics Service (NASS)
- **File:** `US-Rice-Acreage-Production-and-yield copy.csv`
- **Coverage:** U.S. rice acreage, production, and yield statistics by state, year, and rice class
- **Time Period:** 1959-2024 (66 years)

## Project Structure

```
├── Step 1/           # Data inspection and exploration
├── Step 2/           # Data cleaning and preprocessing
├── Step 3/           # Model selection and training
├── Step 4/           # Model application and evaluation
├── Step 5/           # Results interpretation and diagnostics
├── Step 6/           # Interactive visualizations
├── app/              # Streamlit web application
├── scripts/          # Helper scripts
└── requirements.txt  # Python dependencies
```

## Key Findings

- **Trend:** U.S. mean rice yield shows a strong upward trend of approximately 60-70 pounds/acre/year
- **Efficiency:** Rice production efficiency has steadily increased over the 66-year period
- **Model Performance:** 
  - Cross-sectional models achieve R² ≈ 0.998
  - Time-forward forecasting performs poorly (R² ≈ -0.95) due to temporal shifts
- **Limitations:** Simple linear regression inadequate for year-ahead forecasting; time-series models recommended

## Technologies Used

- **Python 3.9+**
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn
- **Web App:** Streamlit

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt
```

## Running the Streamlit App

```bash
streamlit run app/app.py
```

Or visit the deployed app: [https://bae599midterm-3agjvjkurtdrcptkmxmvrv.streamlit.app](https://bae599midterm-3agjvjkurtdrcptkmxmvrv.streamlit.app)

## Models Tested

1. **Linear Regression** (OLS) - Baseline model
2. **Ridge Regression** - L2 regularization
3. **Gradient Boosting** - Ensemble method
4. **Random Forest** - Ensemble method

## Results Summary

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 37.30 | 14.72 | 0.998 |
| Ridge (α=1) | 37.30 | 14.71 | 0.998 |
| Gradient Boosting | 57.99 | 31.53 | 0.995 |
| Random Forest | 58.13 | 27.05 | 0.995 |

*Note: Metrics are for cross-sectional pooled data. Time-forward performance differs significantly.*

## Future Improvements

- Implement ARIMA or SARIMAX for time-series forecasting
- Add exogenous variables (weather, economic indicators)
- Use rolling-origin cross-validation
- Incorporate GLS with Newey-West standard errors

## Author

Natalie Cupples  
BAE 599 - Mid-term Project  
University of Kentucky, Fall 2025

## License

This project is for educational purposes.
