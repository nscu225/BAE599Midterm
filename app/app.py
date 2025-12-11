import streamlit as st
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit Cloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import re
from pathlib import Path
import pathlib
import streamlit.components.v1 as components

# Determine project root relative to this file so the app finds Step folders regardless of cwd
try:
    BASE = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback if __file__ is not defined
    BASE = Path.cwd()
    
STEP1 = BASE / 'Step 1'
STEP2 = BASE / 'Step 2'
STEP3 = BASE / 'Step 3'
STEP4 = BASE / 'Step 4'
STEP5 = BASE / 'Step 5'
STEP6 = BASE / 'Step 6'

st.set_page_config(
    page_title='US Rice Yield Analysis', 
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title('US Rice Production & Yield — Analysis Report')
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select step', ['Overview', 'Dataset Inspection & Description', 'Dataset Preparation', 'ML Model Selection', 'ML Model Application', 'Interpretation of Results'])

# Note: uploader removed per user request. The app will read the original CSV from the repo when available.


def get_original_df():
    """Return the original dataframe either from the repo file or uploaded file in session state."""
    # Try multiple possible filenames
    candidates = [
        'US-Rice-Acreage-Production-and-yield.csv',
        'US-Rice-Acreage-Production-and-yield copy.csv',
        'US-Rice-Acreage-Production-and-yield_copy.csv'
    ]
    for name in candidates:
        orig_csv = BASE / name
        if orig_csv.exists():
            try:
                return pd.read_csv(orig_csv)
            except Exception:
                continue
    # fallback to uploaded
    return st.session_state.get('uploaded_orig_df')


def find_original_csv():
    """Return a Path to the original CSV if it exists in common locations, plus a list of tried locations.

    Checks (in order):
      - BASE / common filenames
      - Path.cwd() / common filenames (in case Streamlit's cwd differs)
      - BASE / 'data' directory for any .csv
    """
    candidates = [
        'US-Rice-Acreage-Production-and-yield.csv',
        'US-Rice-Acreage-Production-and-yield copy.csv',
        'US-Rice-Acreage-Production-and-yield_copy.csv'
    ]
    tried = []
    for name in candidates:
        p = BASE / name
        tried.append(str(p))
        if p.exists():
            return p, tried
        p2 = Path.cwd() / name
        tried.append(str(p2))
        if p2.exists():
            return p2, tried

    data_dir = BASE / 'data'
    tried.append(str(data_dir))
    if data_dir.exists() and data_dir.is_dir():
        for p in data_dir.iterdir():
            if p.suffix.lower() == '.csv':
                tried.append(str(p))
                return p, tried

    return None, tried


def ensure_step2_dir():
    p = STEP2
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p


def create_cleaned_from_original(df_orig):
    """Simple in-app cleaning to produce a cleaned dataframe suitable for plotting/modeling.

    This is a minimal pipeline: filter to YIELD rows if STATISTIC_DESCRIPTION exists, coerce VALUE to numeric,
    drop NA, and aggregate to one record per year (mean VALUE). Returns cleaned dataframe and saves to Step 2.
    """
    df = df_orig.copy()
    # If statistic descriptor present, filter to yield
    if 'STATISTIC_DESCRIPTION' in df.columns:
        df = df[df['STATISTIC_DESCRIPTION'].str.contains('YIELD', na=False)]
    # Normalize column names for year and location/class
    if 'Year' in df.columns and 'YEAR' not in df.columns:
        df = df.rename(columns={'Year': 'YEAR'})
    if 'LOCATION' in df.columns and 'LOCATION_DESCRIPTION' not in df.columns:
        df = df.rename(columns={'LOCATION': 'LOCATION_DESCRIPTION'})
    if 'CLASS' in df.columns and 'CLASS_DESCRIPTION' not in df.columns:
        df = df.rename(columns={'CLASS': 'CLASS_DESCRIPTION'})

    # Coerce VALUE to numeric
    if 'VALUE' in df.columns:
        df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')

    # Create helpful flags if columns exist
    if 'LOCATION_DESCRIPTION' in df.columns:
        df['is_us_total'] = df['LOCATION_DESCRIPTION'].astype(str).str.upper().str.contains('U.S') | df['LOCATION_DESCRIPTION'].astype(str).str.upper().str.contains('U.S. TOTAL')
    else:
        df['is_us_total'] = False

    if 'CLASS_DESCRIPTION' in df.columns:
        df['is_all_classes'] = df['CLASS_DESCRIPTION'].astype(str).str.upper().str.contains('ALL')
    else:
        df['is_all_classes'] = False

    # Simple winsorization: clip at 1st and 99th percentile (global)
    if 'VALUE' in df.columns:
        lower = df['VALUE'].quantile(0.01)
        upper = df['VALUE'].quantile(0.99)
        df['VALUE_clipped'] = df['VALUE'].clip(lower=lower, upper=upper)

    # Save a detailed cleaned file (retains location/class rows) and an aggregated by-year file
    ensure_step2_dir()
    detailed_path = STEP2 / 'cleaned_yield_data.csv'
    df.to_csv(detailed_path, index=False)

    # Also create an aggregated YEAR-level file for quick plotting/legacy code
    if 'YEAR' in df.columns and 'VALUE' in df.columns:
        df_year = df.groupby('YEAR', as_index=False).agg({'VALUE':'mean', 'VALUE_clipped': 'mean' if 'VALUE_clipped' in df.columns else ('VALUE','mean')})
        # normalize aggregated column names
        if 'VALUE_clipped' in df.columns:
            df_year = df_year.rename(columns={'VALUE_clipped':'VALUE_clipped', 'VALUE':'VALUE'})
        else:
            df_year = df_year.rename(columns={('VALUE','mean'):'VALUE'}) if isinstance(df_year.columns[1], tuple) else df_year
        year_path = STEP2 / 'cleaned_yield_by_year.csv'
        df_year.to_csv(year_path, index=False)
        return df_year

    return df

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

        My research question is: "Is U.S. rice production becoming more efficient (yield per acre) over time?"
        """
    )
    st.markdown('- Use the sidebar to navigate each step. Plots and artifacts are loaded from the repository `Step 1` to `Step 6` folders.')
    
    # GitHub repository link
    st.markdown('---')
    st.subheader('📂 Project Repository')
    st.markdown('**GitHub Repository:** [https://github.com/nscu225/BAE599Midterm](https://github.com/nscu225/BAE599Midterm)')
    st.markdown('View the complete source code, data files, and analysis scripts.')
    
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
    orig_csv, tried_paths = find_original_csv()
    if orig_csv is not None and orig_csv.exists():
        try:
            df_orig = pd.read_csv(orig_csv, nrows=50)
            st.dataframe(df_orig)
            st.caption(f'Preview of `{orig_csv.name}` — showing up to {len(df_orig)} rows')
        except Exception as e:
            st.warning(f'Could not read `{orig_csv.name}` for preview: {e}')
    else:
        # If the user uploaded a CSV in-session, show that instead
        if 'uploaded_orig_df' in st.session_state and st.session_state['uploaded_orig_df'] is not None:
            st.success('Using uploaded CSV (in-session).')
            st.dataframe(st.session_state['uploaded_orig_df'].head(50))
        else:
            st.info('Original dataset not found. Place the CSV in the project root or use the sidebar uploader to provide it for the app.')
            # show a compact diagnostics expander listing attempted locations
            with st.expander('Paths checked for original CSV'):
                for p in tried_paths:
                    st.text(p)
    summary = STEP1 / 'summary.txt'
    if summary.exists():
        # Show the summary inside a collapsible expander to avoid dumping long text onto the page
        with st.expander('Show Step 1/summary.txt'):
            show_text_file(summary)
    

    st.subheader('Diagnostic plots')
    st.markdown('Showing selected inspection plots: histogram of yield, boxplot of VALUE, and scatter/time trend of yield.')
    orig_csv_path, _ = find_original_csv()
    orig_csv = orig_csv_path if orig_csv_path else BASE / 'US-Rice-Acreage-Production-and-yield.csv'

    cols = st.columns(3)
    # Histogram (in-memory if possible)
    with cols[0]:
        if orig_csv.exists():
            try:
                df_orig_full = pd.read_csv(orig_csv)
                # Filter to yield rows if present
                if 'STATISTIC_DESCRIPTION' in df_orig_full.columns:
                    df_before = df_orig_full[df_orig_full['STATISTIC_DESCRIPTION'].str.contains('YIELD', na=False)]
                else:
                    df_before = df_orig_full
                if 'VALUE' in df_before.columns:
                    vals = pd.to_numeric(df_before['VALUE'], errors='coerce').dropna()
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.histplot(vals, bins=40, kde=False, ax=ax)
                    ax.set_title('Histogram of VALUE (yield)')
                    st.pyplot(fig)
                else:
                    st.info('No VALUE column found in original CSV to plot histogram')
            except Exception as e:
                st.warning(f'Could not generate histogram from CSV: {e}')
        else:
            st.info('Original CSV not found; histogram not available')
        st.markdown(
            """
            **Histogram summary:** Shows the distribution of reported yields (pounds per acre). The distribution
            centers in the low-to-mid thousands of pounds per acre with a noticeable spread and some higher-value
            observations. This helps identify skew and where most observations fall.
            """
        )

    # Boxplot (in-memory if possible)
    with cols[1]:
        if orig_csv.exists():
            try:
                df_orig_full = pd.read_csv(orig_csv)
                # flexible column detection for state and value
                # flexible detection for state/location and value columns — include USDA-style column names
                state_candidates = ['STATE','STATE_NAME','State','LOCATION','Location', 'LOCATION_DESCRIPTION', 'LOCATION_NAME']
                state_col = next((c for c in state_candidates if c in df_orig_full.columns), None)
                value_col = next((c for c in ['VALUE','YIELD','Value'] if c in df_orig_full.columns), None)
                if value_col and state_col:
                    df_box = df_orig_full.copy()
                    df_box['VALUE_num'] = pd.to_numeric(df_box[value_col], errors='coerce')
                    top_states = df_box.groupby(state_col)['VALUE_num'].median().dropna().nlargest(12).index
                    df_box_top = df_box[df_box[state_col].isin(top_states)]
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.boxplot(x=state_col, y='VALUE_num', data=df_box_top, ax=ax)
                    ax.set_title('Boxplot of VALUE (top 12 states)')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    st.pyplot(fig)
                else:
                    st.info('STATE or VALUE column missing; cannot create boxplot')
            except Exception as e:
                st.warning(f'Could not generate boxplot from CSV: {e}')
        else:
            st.info('Original CSV not found; boxplot not available')
        st.markdown(
            """
            **Boxplot summary:** Compares the distribution of yield values across the top states by median yield.
            Medians, interquartile ranges, and outliers highlight which states typically report higher or lower yields.
            """
        )

    # Scatter / time trend (in-memory if possible)
    with cols[2]:
        # Prefer to generate a mean-yield-by-year trend with linear fit from cleaned data if available
        cleaned = STEP2 / 'cleaned_yield_data.csv'
        try:
            plotted = False
            if cleaned.exists():
                df_after = pd.read_csv(cleaned)
                # Expect a 'YEAR' and 'VALUE' or similar target column
                year_col = 'YEAR' if 'YEAR' in df_after.columns else 'Year' if 'Year' in df_after.columns else None
                val_col = 'VALUE' if 'VALUE' in df_after.columns else 'YIELD' if 'YIELD' in df_after.columns else None
                if year_col and val_col:
                    df_agg = df_after.groupby(year_col)[val_col].mean().reset_index()
                    df_agg = df_agg.dropna()
                    x = df_agg[year_col].astype(float).values
                    y = pd.to_numeric(df_agg[val_col], errors='coerce').values
                    coeffs = np.polyfit(x, y, deg=1)
                    trend = np.poly1d(coeffs)
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.scatter(x, y, s=20)
                    ax.plot(x, trend(x), color='red')
                    ax.set_title('US Mean Yield by Year with Linear Fit')
                    ax.set_xlabel('Year')
                    ax.set_ylabel(val_col)
                    st.pyplot(fig)
                    plotted = True
            # If not plotted yet, try to create cleaned data from uploaded/original CSV on the fly
            if not plotted:
                # Try to locate the original CSV from common locations first
                orig_path, _ = find_original_csv()
                if orig_path is not None:
                    try:
                        orig_df = pd.read_csv(orig_path)
                    except Exception:
                        orig_df = get_original_df()
                else:
                    orig_df = get_original_df()
                if orig_df is not None:
                    df_year = create_cleaned_from_original(orig_df)
                    if df_year is not None:
                        x = df_year['YEAR'].astype(float).values
                        y = pd.to_numeric(df_year['VALUE'], errors='coerce').values
                        coeffs = np.polyfit(x, y, deg=1)
                        trend = np.poly1d(coeffs)
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.scatter(x, y, s=20)
                        ax.plot(x, trend(x), color='red')
                        ax.set_title('US Mean Yield by Year with Linear Fit')
                        ax.set_xlabel('Year')
                        ax.set_ylabel('VALUE')
                        st.pyplot(fig)
                        plotted = True
                else:
                        # Prefer the exact saved plot from Step 3 if present (this restores the earlier plot)
                        s3_plot = STEP3 / 'plots' / 'yield_trend_with_fit.png'
                        if s3_plot.exists():
                            st.image(str(s3_plot), use_container_width=True, caption='Yield trend with fit (saved)')
                            plotted = True
                        else:
                            # Fall back to the aggregated by-year CSV we now create: cleaned_yield_by_year.csv
                            agg_path = STEP2 / 'cleaned_yield_by_year.csv'
                            if agg_path.exists():
                                try:
                                    df_agg = pd.read_csv(agg_path)
                                    year_col = 'YEAR' if 'YEAR' in df_agg.columns else ('Year' if 'Year' in df_agg.columns else None)
                                    val_col = 'VALUE_clipped' if 'VALUE_clipped' in df_agg.columns else ('VALUE' if 'VALUE' in df_agg.columns else None)
                                    if year_col and val_col:
                                        x = df_agg[year_col].astype(float).values
                                        y = pd.to_numeric(df_agg[val_col], errors='coerce').values
                                        coeffs = np.polyfit(x, y, deg=1)
                                        trend = np.poly1d(coeffs)
                                        fig, ax = plt.subplots(figsize=(4, 3))
                                        ax.scatter(x, y, s=20)
                                        ax.plot(x, trend(x), color='red')
                                        ax.set_title('US Mean Yield by Year with Linear Fit')
                                        ax.set_xlabel('Year')
                                        ax.set_ylabel(val_col)
                                        st.pyplot(fig)
                                        plotted = True
                                except Exception:
                                    pass
            if not plotted:
                st.info('No cleaned data or plot available for trend')
        except Exception as e:
            st.warning(f'Could not generate trend plot: {e}')
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

    # In-app preprocessing controls
    orig_df_status = get_original_df()

    if st.button('Run in-app preprocessing (create Step 2/cleaned_yield_data.csv)'):
        if orig_df_status is None:
            st.warning('No original CSV available to process. Add the CSV to the repository.')
        else:
            with st.spinner('Running minimal cleaning and aggregation...'):
                cleaned_df = create_cleaned_from_original(orig_df_status)
            if cleaned_df is not None:
                st.success(f'Created cleaned data with {len(cleaned_df)} rows and saved to `Step 2/cleaned_yield_data.csv`.')
                st.dataframe(cleaned_df.head(20))
            else:
                st.error('Could not create cleaned data — input appears to lack YEAR or VALUE columns.')

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

    orig_csv, _ = find_original_csv()
    before_vals = None
    after_vals = None
    if orig_csv is not None and orig_csv.exists():
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
        # If there's an uploaded CSV in-session, prefer that
        if 'uploaded_orig_df' in st.session_state and st.session_state['uploaded_orig_df'] is not None:
            df_orig_full = st.session_state['uploaded_orig_df']
            if 'VALUE' in df_orig_full.columns:
                before_vals = pd.to_numeric(df_orig_full['VALUE'], errors='coerce').dropna()
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
        # Attempt a lightweight on-the-fly model comparison using the aggregated Step 2 data.
        # This helps Streamlit Community Cloud deployments where precomputed CSVs aren't committed.
        agg_path = STEP2 / 'cleaned_yield_by_year.csv'
        if not agg_path.exists():
            # fall back to detailed cleaned file and aggregate
            detailed = STEP2 / 'cleaned_yield_data.csv'
            if detailed.exists():
                try:
                    df_det = pd.read_csv(detailed)
                    if 'YEAR' in df_det.columns and ('VALUE_clipped' in df_det.columns or 'VALUE' in df_det.columns):
                        val_col = 'VALUE_clipped' if 'VALUE_clipped' in df_det.columns else 'VALUE'
                        df_agg = df_det.groupby('YEAR', as_index=False)[val_col].mean().rename(columns={val_col: 'VALUE'})
                        df_agg['YEAR'] = pd.to_numeric(df_agg['YEAR'], errors='coerce')
                    else:
                        df_agg = None
                except Exception:
                    df_agg = None
            else:
                df_agg = None
        else:
            try:
                df_agg = pd.read_csv(agg_path)
                # normalize columns
                if 'VALUE_clipped' in df_agg.columns:
                    df_agg = df_agg.rename(columns={'VALUE_clipped':'VALUE'})
                df_agg['YEAR'] = pd.to_numeric(df_agg['YEAR'], errors='coerce')
            except Exception:
                df_agg = None

        if df_agg is None or df_agg.empty:
            st.info('Model comparison CSV not found at Step 3/model_comparison.csv')
        else:
            st.info('Model comparison CSV not found — computing a lightweight comparison from Step 2 data')
            # Compute a lightweight comparison WITHOUT scikit-learn (use numpy) to avoid dependency issues
            if df_agg is None or df_agg.empty:
                st.info('Not enough data to compute comparisons.')
            else:
                try:
                    # Prepare X and y robustly (ensure 1-d numeric arrays)
                    df_agg = df_agg.dropna(subset=['YEAR', 'VALUE'])
                    X = pd.to_numeric(df_agg['YEAR'], errors='coerce')
                    y = pd.to_numeric(df_agg['VALUE'], errors='coerce')

                    # Convert to numpy 1-d arrays and filter finite values
                    # If any cells are list-like (e.g., ['1970']) flatten them to scalars
                    def _unwrap_first(v):
                        # If it's already a list/tuple/ndarray, take first element
                        if isinstance(v, (list, tuple, np.ndarray)):
                            try:
                                return v[0]
                            except Exception:
                                return np.nan
                        # If it's a string that looks like a list (e.g. "['1970']"), try to parse it
                        if isinstance(v, str) and v.strip().startswith('[') and v.strip().endswith(']'):
                            try:
                                parsed = ast.literal_eval(v)
                                if isinstance(parsed, (list, tuple, np.ndarray)) and len(parsed) > 0:
                                    return parsed[0]
                            except Exception:
                                # fall through and return original string to be coerced (may become NaN)
                                pass
                        return v

                    X_series = pd.Series([_unwrap_first(v) for v in X])
                    y_series = pd.Series([_unwrap_first(v) for v in y])

                    # Aggressive numeric extraction: try to parse numbers from strings (e.g. "['1970']" or "1970\n")
                    def _extract_number(val):
                        if pd.isna(val):
                            return np.nan
                        if isinstance(val, (int, float, np.integer, np.floating)):
                            return float(val)
                        s = str(val).strip()
                        if s == '':
                            return np.nan
                        # quick try: direct float conversion
                        try:
                            return float(s)
                        except Exception:
                            pass
                        # regex search for the first number-like token
                        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
                        if m:
                            try:
                                return float(m.group(0))
                            except Exception:
                                return np.nan
                        return np.nan

                    X = np.asarray([_extract_number(v) for v in X_series]).ravel().astype(float)
                    y = np.asarray([_extract_number(v) for v in y_series]).ravel().astype(float)

                    mask = np.isfinite(X) & np.isfinite(y)
                    X = X[mask]
                    y = y[mask]

                    if X.size < 2 or y.size < 2:
                        raise ValueError('Not enough numeric YEAR/VALUE pairs to fit a linear model (need >=2).')

                    # Linear fit using numpy.polyfit (degree 1)
                    polyfit_failed = False
                    try:
                        coeffs = np.polyfit(X, y, deg=1)
                        pred_lr = np.poly1d(coeffs)(X)
                    except Exception as e_poly:
                        polyfit_failed = True
                        st.warning(f"Could not fit a linear trend: {e_poly}. Using mean baseline only.")
                        pred_lr = None

                    # Baseline: mean predictor
                    y_mean = np.nanmean(y)
                    pred_mean = np.full_like(y, y_mean)

                    def metrics_from_preds(y_true, y_pred):
                        # both inputs are 1-d numpy arrays
                        if y_true.size == 0:
                            return (None, None, None)
                        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                        mae = float(np.mean(np.abs(y_true - y_pred)))
                        ss_res = float(np.sum((y_true - y_pred) ** 2))
                        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else None
                        return (rmse, mae, r2)

                    if pred_lr is not None:
                        lr_rmse, lr_mae, lr_r2 = metrics_from_preds(y, pred_lr)
                    else:
                        lr_rmse = lr_mae = lr_r2 = None
                    mean_rmse, mean_mae, mean_r2 = metrics_from_preds(y, pred_mean)

                    rows = [
                        {'model':'LinearRegression','rmse':lr_rmse,'mae':lr_mae,'r2':lr_r2},
                        {'model':'MeanBaseline','rmse':mean_rmse,'mae':mean_mae,'r2':mean_r2}
                    ]
                    df_comp = pd.DataFrame(rows)
                    # Try to save computed comparison file so future loads succeed in-session
                    try:
                        comp_path = STEP3 / 'model_comparison.csv'
                        comp_path.parent.mkdir(parents=True, exist_ok=True)
                        df_comp.to_csv(comp_path, index=False)
                    except Exception:
                        pass

                    df_show = df_comp.copy()
                    name_map = {
                        'LinearRegression': 'Linear Regression',
                        'MeanBaseline': 'Mean baseline'
                    }
                    df_show['Model'] = df_show['model'].map(name_map).fillna(df_show['model'])
                    df_show = df_show[['Model','rmse','mae','r2']]
                    df_show[['rmse','mae','r2']] = df_show[['rmse','mae','r2']].round(3)
                    st.subheader('Selected model comparison (computed)')
                    st.table(df_show.rename(columns={'rmse':'RMSE','mae':'MAE','r2':'R2'}).set_index('Model'))
                except Exception as e:
                    st.warning(f'Could not compute model comparison: {e}')

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
