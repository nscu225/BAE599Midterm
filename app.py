import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pathlib
import streamlit.components.v1 as components

BASE = Path.cwd()
STEP1 = BASE / 'Step 1'
STEP2 = BASE / 'Step 2'
STEP3 = BASE / 'Step 3'
STEP4 = BASE / 'Step 4'
STEP5 = BASE / 'Step 5'
STEP6 = BASE / 'Step 6'

st.set_page_config(page_title='US Rice Yield Analysis', layout='wide')

st.title('US Rice Production & Yield — Analysis Report')
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select step', ['Overview', 'Dataset Inspection & Description', 'Dataset Preparation', 'ML Model Selection', 'ML Model Application', 'Interpretation of Results'])

# helper to read text files
def read_text(p:pathlib.Path):
    try:
        return p.read_text()
    except Exception as e:
        return None

import pathlib

def show_text_file(path, header=True):
    if path.exists():
        text = path.read_text()
        if header:
            st.subheader(path.name)
        st.text(text)
    else:
        st.warning(f'File not found: {path}')

def list_images(folder):
    if folder.exists():
        return sorted([p for p in folder.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg']])
    return []

if page == 'Overview':
    st.header('Overview')
    st.markdown(
        """
        The dataset contains U.S. rice acreage, production, and yield data collected by the USDA National Agricultural
        Statistics Service (NASS). It includes yearly information by state and rice class, showing how much rice was
        planted, harvested, and produced.

        My research question is: “Is U.S. rice production becoming more efficient (yield per acre) over time?”
        """
    )
    st.markdown('- Use the sidebar to navigate each step. Plots and artifacts are loaded from the repository `Step 1` to `Step 6` folders.')
    # Data Inspection Summary moved to the Dataset Inspection page per request

if page == 'Dataset Inspection & Description':
    st.header('Step 1 — Dataset Inspection & Description')
    st.markdown(
        """
        The dataset contains U.S. rice acreage, production, and yield data collected by the USDA National Agricultural
        Statistics Service (NASS). It includes yearly information by state and rice class, showing how much rice was
        planted, harvested, and produced.

        My research question is: “Is U.S. rice production becoming more efficient (yield per acre) over time?”
        """
    )
    st.subheader('Data Inspection Summary')
    st.markdown(
        """
        - Created a Step 1 folder with an inspection pipeline.
        - Loaded the US-Rice-Acreage-Production-and-yield.csv dataset.
        - Generated dataset summaries (info, describe, and value_counts) and saved outputs.
        - Converted the VALUE column to numeric and checked for missing data.
        - Created visualizations: histograms, boxplot (top 12 states), and yield/area trends over time.
        - Added requirements.txt and README.md, fixed a script bug, and re-ran successfully.
        """
    )
    # Show a small snapshot of the original dataset (safe nrows read)
    st.subheader('Original dataset snapshot')
    orig_csv = BASE / 'US-Rice-Acreage-Production-and-yield copy.csv'
    if orig_csv.exists():
        try:
            df_orig = pd.read_csv(orig_csv, nrows=50)
            st.dataframe(df_orig)
            st.caption(f'Preview of `{orig_csv.name}` — showing up to {len(df_orig)} rows')
        except Exception as e:
            st.warning(f'Could not read `{orig_csv.name}` for preview: {e}')
    else:
        st.info(f'Original dataset not found at `{orig_csv}`. Place the CSV in the project root to preview it here.')
    summary = STEP1 / 'summary.txt'
    if summary.exists():
        # Show the summary inside a collapsible expander to avoid dumping long text onto the page
        with st.expander('Show Step 1/summary.txt'):
            show_text_file(summary)
    

    st.subheader('Diagnostic plots')
    st.markdown('Showing selected inspection plots: histogram of yield, boxplot of VALUE, and scatter/time trend of yield.')
    plots_dir = STEP1 / 'plots'
    # Preferred filenames
    hist_plot = plots_dir / 'hist_VALUE.png'
    box_plot = plots_dir / 'box_VALUE.png'
    # For scatter/time trend, try several fallbacks
    scatter_candidates = [
        plots_dir / 'scatter_VALUE_vs_YEAR.png',
        STEP3 / 'plots' / 'yield_trend_with_fit.png',
        STEP6 / 'screenshots' / 'yield_full_range.png',
        plots_dir / 'hist_YEAR.png'  # last resort: year histogram if nothing else
    ]

    cols = st.columns(3)
    # Histogram
    with cols[0]:
        if hist_plot.exists():
            st.image(str(hist_plot), use_container_width=True, caption='Histogram of VALUE (yield)')
        else:
            st.info(f'Histogram not found: {hist_plot.name}')
        # Brief explanation
        st.markdown(
            """
            **Histogram summary:** Shows the distribution of reported yields (pounds per acre). The distribution
            centers in the low-to-mid thousands of pounds per acre with a noticeable spread and some higher-value
            observations (e.g., high-yielding regions). This helps identify skew and where most observations fall.
            """
        )
    # Boxplot
    with cols[1]:
        if box_plot.exists():
            st.image(str(box_plot), use_container_width=True, caption='Boxplot of VALUE')
        else:
            st.info(f'Boxplot not found: {box_plot.name}')
        # Brief explanation
        st.markdown(
            """
            **Boxplot summary:** Compares the distribution of yield values across locations. Medians, interquartile ranges,
            and outliers highlight which states typically report higher or lower yields. For example, California generally
            shows higher median yields and a wider spread compared with several other states; the U.S. total sits near the
            center of the distribution.
            """
        )
    # Scatter / time trend
    with cols[2]:
        found = False
        for cand in scatter_candidates:
            if cand.exists():
                st.image(str(cand), use_container_width=True, caption=cand.name)
                found = True
                break
        if not found:
            st.info('No scatter/time-trend plot found in Step 1/Step 3/Step 6; run plotting scripts to generate one.')
        # Brief explanation
        st.markdown(
            """
            **Scatter / trend summary:** Shows yield observations over time (and where available a fitted trend).
            The long-term pattern indicates an upward trend in U.S. mean yield. A simple linear fit estimates an increase
            of roughly 60–70 pounds per acre per year over the historical period, though time-series diagnostics
            (autocorrelation, structural changes) affect forecasting performance.
            """
        )

if page == 'Dataset Preparation':
    st.header('Step 2 — Dataset Preparation')
    st.subheader('Data Cleaning and Preprocessing Summary')
    st.markdown(
        """
        - Created a Step 2 folder with preprocess_data.py, requirements.txt, and README.md.
        - Filtered the dataset to include only relevant statistics: YIELD, AREA HARVESTED, AREA PLANTED, and PRODUCTION — these directly relate to rice production efficiency.
        - Converted the VALUE column to numeric and standardized units (e.g., thousand acres → acres, thousand hundredweight → pounds) to ensure consistency across all records.
        - Pivoted the data so each row represents a unique combination of year × state × rice class, with separate columns for each statistic — simplifying analysis and model input.
        - Imputed missing values using each location’s median (or overall median if unavailable), justified by inspection results from Step 1 showing some incomplete records.
        - Detected and corrected outliers (|z| > 3) using location-based z-scores, replacing extreme values with the median to prevent distortion in the model.
        - One-hot encoded categorical features (state and rice class) to prepare them for machine learning models.
        - Scaled numeric features using StandardScaler to normalize ranges and improve model performance.
        - Saved final outputs: cleaned_dataset.csv, pivot_stats.csv, filtered_raw_stats.csv, and documentation files (processing_log.txt, processing_summary.json).
        """
    )
    changelog = STEP2 / 'changelog.txt'
    # changelog file presence intentionally not shown inline

    st.subheader('Cleaned data sample')
    cleaned = STEP2 / 'cleaned_yield_data.csv'
    if cleaned.exists():
        df = pd.read_csv(cleaned)
        st.dataframe(df.head(50))
        st.caption(f'Cleaned data: {cleaned} — {df.shape[0]} rows, {df.shape[1]} cols')
    else:
        st.warning('Run Step 2 to generate cleaned data')

    # Before vs After outlier removal visualization (histogram only)
    st.subheader('Before vs. After Outlier Removal')
    st.markdown('Compare the distribution of YIELD values before cleaning (raw CSV) and after cleaning (Step 2 output).')

    orig_csv = BASE / 'US-Rice-Acreage-Production-and-yield copy.csv'
    before_vals = None
    after_vals = None
    if orig_csv.exists():
        try:
            df_orig_full = pd.read_csv(orig_csv)
            # Filter to yield rows if present
            if 'STATISTIC_DESCRIPTION' in df_orig_full.columns:
                df_before = df_orig_full[df_orig_full['STATISTIC_DESCRIPTION'].str.contains('YIELD', na=False)]
            else:
                df_before = df_orig_full
            if 'VALUE' in df_before.columns:
                before_vals = pd.to_numeric(df_before['VALUE'], errors='coerce').dropna()
        except Exception as e:
            st.warning(f'Could not read original CSV for before-values: {e}')
    else:
        st.info('Original CSV not found; cannot show "before" distribution.')

    if cleaned.exists():
        try:
            df_after = pd.read_csv(cleaned)
            # prefer winsorized / clipped value if present
            if 'VALUE_clipped' in df_after.columns:
                after_vals = pd.to_numeric(df_after['VALUE_clipped'], errors='coerce').dropna()
            elif 'VALUE' in df_after.columns:
                after_vals = pd.to_numeric(df_after['VALUE'], errors='coerce').dropna()
        except Exception as e:
            st.warning(f'Could not read cleaned CSV for after-values: {e}')

    if (before_vals is None or len(before_vals) == 0) and (after_vals is None or len(after_vals) == 0):
        st.info('No numeric VALUE data available to plot for before or after.')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.set_theme(style='whitegrid')
        # Left: before (histogram)
        if before_vals is not None and len(before_vals) > 0:
            sns.histplot(before_vals, bins=40, kde=False, ax=axes[0])
            axes[0].set_title('Before (raw CSV)')
            axes[0].set_xlabel('VALUE (yield)')
        else:
            axes[0].text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
            axes[0].set_title('Before (raw CSV)')
            axes[0].set_xticks([])
            axes[0].set_yticks([])

        # Right: after (histogram)
        if after_vals is not None and len(after_vals) > 0:
            sns.histplot(after_vals, bins=40, kde=False, ax=axes[1])
            axes[1].set_title('After (cleaned Step 2)')
            axes[1].set_xlabel('VALUE (yield)')
        else:
            axes[1].text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
            axes[1].set_title('After (cleaned Step 2)')
            axes[1].set_xticks([])
            axes[1].set_yticks([])

        plt.tight_layout()
        st.pyplot(fig)
        # Short explanation under the plot
        st.markdown(
            """
            **Plot summary:** The left panel shows the distribution of reported yields from the original (raw) CSV.
            The right panel shows the distribution after data cleaning and outlier handling (winsorization or replacement).
            Note how extreme high/low values are reduced or removed in the cleaned data, which helps stabilize
            modeling and prevents a few extreme observations from dominating model fit.
            """
        )

    # Summary numbers (from processing comparison)
    st.subheader('Summary numbers')
    st.markdown(
        """
        - Rows before (original CSV): 1869

        - Rows after (cleaned CSV): 615

        - Rows removed: 1254

        - Total missing values before: 0

        - Total missing values after: 0

        - Duplicates before: 0

        - Duplicates after: 0
        """
    )

    st.subheader('Dropped-rows detection')
    st.markdown(
        """
        - Dropped rows found by comparing common columns (left-only from a merge): 1254
        - A sample of up to 50 dropped rows was saved to: `Step 2/dropped_rows_sample.csv`
        """
    )

if page == 'ML Model Selection':
    st.header('Step 3 — ML Model Selection')
    st.subheader('Model Selection and Training Summary')
    st.markdown(
        """
        - Created a Step 3 folder with train_and_evaluate.py and requirements.txt.
        - Selected Model: Ordinary Least Squares (OLS) Linear Regression
        - Type of problem: Regression (predicting a continuous variable — YIELD).
        - Rationale:
          - Linear regression is interpretable and provides a clear baseline for understanding how each feature impacts rice yield.
          - The data inspection from Step 1 showed relatively linear relationships between yield and variables such as area planted and production, making OLS a logical first choice.
          - It assumes a linear relationship, independent errors, and normally distributed residuals — all reasonable starting assumptions for this dataset.
        - Data preparation for modeling:
          - Prevented target leakage by excluding YIELD_scaled from predictors.
          - Features used: YEAR, scaled AREA HARVESTED, AREA PLANTED, PRODUCTION, and one-hot-encoded categorical variables (LOC_* and CLASS_*).
        - Model training and evaluation:
          - Used an 80/20 train-test split for fair performance assessment.
          - Model achieved RMSE ≈ 658.6 and R² ≈ 0.732, indicating a reasonably strong linear fit.
        - Saved outputs:
          - metrics.json (evaluation results)
          - model_coefficients.csv (feature weights)
          - test_predictions.csv
          - predicted_vs_actual.png and residuals_vs_predicted.png (visual performance checks).
        """
    )
    metrics = STEP3 / 'metrics.txt'
    if metrics.exists():
        # metrics file present (not displayed inline)
        pass
    else:
        st.warning('Run Step 3 to generate metrics.txt')

    # Show compact model comparison table for requested models if available
    comp_csv = STEP3 / 'model_comparison.csv'
    if comp_csv.exists():
        try:
            df_comp = pd.read_csv(comp_csv)
            # Select the three requested models and present rounded metrics
            wanted = ['LinearRegression', 'GradientBoosting', 'RandomForest']
            df_show = df_comp[df_comp['model'].isin(wanted)].copy()
            name_map = {
                'LinearRegression': 'Linear Regression',
                'GradientBoosting': 'Gradient Boosting',
                'RandomForest': 'Random Forest'
            }
            df_show['Model'] = df_show['model'].map(name_map).fillna(df_show['model'])
            df_show = df_show[['Model', 'rmse', 'mae', 'r2']]
            df_show[['rmse', 'mae', 'r2']] = df_show[['rmse', 'mae', 'r2']].round(3)
            st.subheader('Selected model comparison')
            st.table(df_show.rename(columns={'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R2'}).set_index('Model'))
        except Exception as e:
            st.warning(f'Could not read model comparison CSV: {e}')
    else:
        st.info('Model comparison CSV not found at Step 3/model_comparison.csv')

    st.subheader('Plots')
    imgs = list_images(STEP3 / 'plots')
    # Prefer showing the yield trend with fit and skip the predicted vs actual plot per request
    for img in imgs:
        if img.name == 'pred_vs_actual.png':
            # skip this plot (deleted from UI)
            continue
        st.image(str(img), caption=img.name, use_container_width=True)
        # Add a short summary specifically under the yield_trend_with_fit plot
        if img.name == 'yield_trend_with_fit.png':
            st.markdown(
                """
                **Short summary:** The plot shows U.S. mean yield by year with a fitted linear trend.
                The fitted slope indicates a positive, steady increase in mean yield over the observed
                period — consistent with gradual productivity improvements. Use diagnostics in Step 5
                to assess residual autocorrelation and structural change before using the linear fit for long-term forecasts.
                """
            )

if page == 'ML Model Application':
    st.header('Step 4 — ML Model Application')
    # Model Training and Evaluation Summary (user-provided)
    st.subheader('Model Training and Evaluation Summary')
    st.markdown(
        """
        - Loaded the cleaned U.S. yield data (66 years total)
        - Combined data to one record per year
        - Split the dataset by time: first 52 years for training, last 14 for testing
        - Used 5-fold time-based cross-validation with TimeSeriesSplit
        - Tested a pipeline with scaling, polynomial features (degree 1 or 2), and Ridge regression (different alpha values)
        - The best model turned out to be a simple linear Ridge model (degree = 1, alpha = 1.0)
        - Saved the model, evaluation metrics, and plots:
          - Metrics: R² = −0.95, RMSE ≈ 260.7, MAE ≈ 203.8
          - Plots: Predicted vs. Actual and Learning Curve
        - Noticed a few minor issues — some small validation folds caused undefined R² values, and multiprocessing produced harmless teardown warnings
        - Takeaway: The model preferred a simple linear approach, but performance on the test set was poor, likely due to time-based shifts. Future steps could include exploring time-series models like ARIMA or GLS.
        """
    )

    st.subheader('Plots')
    imgs = list_images(STEP4 / 'plots')
    for img in imgs:
        st.image(str(img), caption=img.name, use_container_width=True)
        # Add short summaries for known Step 4 plots
        if img.name == 'pred_vs_actual_test.png':
            st.markdown(
                """
                **Predicted vs Actual (test set):** This scatter plot compares model predictions to actual test-set yields.
                Points near the diagonal indicate good prediction agreement; deviations show prediction errors and bias.
                Large spread away from the diagonal (as seen here) explains the high RMSE and low/negative R² on the test set.
                """
            )
        if img.name == 'learning_curve.png':
            st.markdown(
                """
                **Learning curve:** Shows training and cross-validation error versus training set size or complexity.
                If the validation curve stays well above the training curve, the model may be underfitting; if the two
                curves converge with low error, the model is well-fit. This plot helps determine whether more data,
                higher model complexity, or regularization changes are needed.
                """
            )

if page == 'Interpretation of Results':
    st.header('Step 5 — Interpretation of Results')

    # Interpretation summary added per user request
    st.subheader('Interpretation of Results: Key Findings and Model Limitations')
    st.markdown(
        """
        - **Conclusion on Efficiency (The Research Question):** U.S. mean rice yield shows a strong long-term upward trend (visually, ≈60–70 pounds/acre/year), suggesting increased efficiency over time.

        - **Model Performance Varies by Task:**

          - *Cross-Sectional/Pooled Data:* Simple linear models (OLS/Ridge) performed extremely well (R² ≈ 0.998), indicating that available features capture most variation in the aggregated dataset.

          - *Time-Forward Forecasting:* The best simple linear model (Ridge) performed very poorly on the 14-year test set (R² ≈ −0.95, RMSE ≈ 260.7), meaning it could not reliably predict future yields.

        - **Major Model Limitations / Diagnostics:**

          - *Autocorrelation is Violated:* Time-series residual diagnostics showed serial correlation, violating a key OLS assumption and making statistical inferences (like significance of trends) unreliable.

          - *Poor Generalization:* The model failed to generalize to holdout years, likely due to nonstationarity and temporal shifts in the data.

        - **Practical Takeaway:** While the data shows a clear trend of increasing efficiency, simple linear regression is not a reliable method for year-ahead forecasting of the aggregated yield series.

        - **Recommended Next Steps:** To improve forecasts and validate the time trend, you should:

          - Use time-series-specific models (e.g., ARIMA, SARIMAX, or GLS with Newey–West standard errors).
          - Re-evaluate the forecast using rolling-origin (walk-forward) cross-validation.
          - Add relevant time-based features like lagged yields or exogenous variables (e.g., weather).
        """
    )

    analysis = STEP5 / 'analysis.txt'
    st.subheader('Diagnostic plots')

    # Show only the QQ plot for OLS residuals (as requested)
    qq_plot = STEP5 / 'plots' / 'qq_ols_resid.png'
    if qq_plot.exists():
        st.image(str(qq_plot), caption='QQ Plot of OLS Residuals', use_container_width=False, width=500)
        st.markdown(
            """
            **QQ plot summary:** The QQ plot compares the distribution of OLS residuals to a normal distribution.
            Points following the diagonal indicate approximately normal residuals; systematic deviations suggest
            skewness or heavy tails. Non-normal residuals weaken classical inference (p-values/confidence
            intervals) though they do not necessarily invalidate predictive performance.
            """
        )
    else:
        st.info('QQ plot not found at `Step 5/plots/qq_ols_resid.png`')

    # Embed the interactive visualization for convenient interpretation
    st.subheader('Interactive visualization')
    interactive = STEP6 / 'interactive_yield.html'
    if interactive.exists():
        try:
            with open(interactive, 'r', encoding='utf-8') as f:
                html = f.read()
            components.html(html, height=700, scrolling=True)
            # Compact caption immediately below the embedded graph (closer spacing)
            st.caption("Interactive plot summary: This interactive visualization shows annual yields with filters for location and class. Use hover to inspect points and zoom to focus on time ranges. It's useful for detecting outliers, local trends, and structural changes that may explain why simple linear forecasts perform poorly for some holdout periods.")
        except Exception as e:
            st.warning('Could not embed Step 6 interactive HTML here. You can open the file directly: ' + str(interactive))
    else:
        st.info('Interactive HTML not found at `Step 6/interactive_yield.html`. Run Step 6 to create it.')

# Note: Step 6 interactive page removed per user request. Interactive content is embedded in Step 5 for interpretation.

st.sidebar.markdown('---')
st.sidebar.markdown('Generated: 2025-10-20')
