import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Try to import sklearn; if missing, instruct user
try:
    from sklearn.preprocessing import StandardScaler
    import joblib
except Exception:
    StandardScaler = None
    joblib = None

cwd = os.getcwd()
csv_filename = "US-Rice-Acreage-Production-and-yield copy.csv"
csv_path = os.path.join(cwd, csv_filename)
output_dir = os.path.join(cwd, "Step 2")
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(csv_path):
    print(f"ERROR: CSV not found at {csv_path}")
    sys.exit(1)

# Read raw CSV
df = pd.read_csv(csv_path)
original_shape = df.shape

changelog = []
changelog.append(f"Preprocessing log generated on {datetime.now().isoformat()}")
changelog.append(f"Original data shape: {original_shape}")

# Step 1: standardize column names (strip)
df.columns = [c.strip() for c in df.columns]
changelog.append("Stripped whitespace from column names.")

# Step 2: Filter to STATISTIC_DESCRIPTION == 'YIELD'
# Rationale: Step 1 identified many STATISTIC_DESCRIPTION values; we only want yield-per-acre rows for yield trend analysis.
mask_yield = df['STATISTIC_DESCRIPTION'].astype(str).str.strip().str.upper() == 'YIELD'
df_y = df[mask_yield].copy()
changelog.append(f"Filtered to STATISTIC_DESCRIPTION == 'YIELD' -> {df_y.shape[0]} rows")

# Step 3: Restrict to unit 'POUNDS PER ACRE' for consistency
mask_unit = df_y['UNIT_DESCRIPTION'].astype(str).str.upper().str.contains('POUND')
df_y = df_y[mask_unit].copy()
changelog.append(f"Filtered to UNIT_DESCRIPTION containing 'POUND' -> {df_y.shape[0]} rows")

# Step 4: Handle missing values
# Rationale: Step 1 showed no nulls in overall data, but re-check after filtering
missing_summary = df_y.isna().sum().to_dict()
changelog.append(f"Missing values by column after filtering: {missing_summary}")
# For critical columns YEAR and VALUE, drop rows missing these
before_drop = df_y.shape[0]
df_y = df_y.dropna(subset=['YEAR', 'VALUE'])
after_drop = df_y.shape[0]
changelog.append(f"Dropped {before_drop - after_drop} rows missing YEAR or VALUE (kept {after_drop})")

# Step 5: Coerce YEAR to integer and VALUE to numeric
df_y['YEAR'] = pd.to_numeric(df_y['YEAR'], errors='coerce').astype('Int64')
df_y['VALUE'] = pd.to_numeric(df_y['VALUE'], errors='coerce')

# Re-check missing after coercion
after_coerce_missing = df_y[['YEAR','VALUE']].isna().sum().to_dict()
changelog.append(f"Missing after coercion (YEAR, VALUE): {after_coerce_missing}")
# Drop any rows still missing YEAR or VALUE
before_drop2 = df_y.shape[0]
df_y = df_y.dropna(subset=['YEAR','VALUE'])
after_drop2 = df_y.shape[0]
changelog.append(f"Dropped {before_drop2 - after_drop2} rows still missing YEAR or VALUE after coercion")

# Step 6: Create useful flags
# is_us_total and is_all_classes to allow filtering for national series
df_y['is_us_total'] = df_y['LOCATION_DESCRIPTION'].astype(str).str.strip().str.upper() == 'U.S. TOTAL'
df_y['is_all_classes'] = df_y['CLASS_DESCRIPTION'].astype(str).str.strip().str.upper() == 'ALL CLASSES'
changelog.append("Added boolean flags 'is_us_total' and 'is_all_classes'.")

# Step 7: Encode categorical variables
# Rationale: Many ML/stat methods require numeric encodings. We will one-hot encode CLASS_DESCRIPTION (small cardinality)
# and factorize LOCATION_DESCRIPTION (many states) while keeping 'U.S. TOTAL' flag separately.

df_y['CLASS_DESCRIPTION'] = df_y['CLASS_DESCRIPTION'].astype(str).str.strip()
df_y['LOCATION_DESCRIPTION'] = df_y['LOCATION_DESCRIPTION'].astype(str).str.strip()

class_dummies = pd.get_dummies(df_y['CLASS_DESCRIPTION'], prefix='CLASS')
# Merge dummies (keep original col too for transparency)
df_y = pd.concat([df_y, class_dummies], axis=1)
changelog.append(f"One-hot encoded CLASS_DESCRIPTION into columns: {list(class_dummies.columns)}")

# Factorize LOCATION_DESCRIPTION into integer codes (preserves mapping)
df_y['LOCATION_CODE'], location_index = pd.factorize(df_y['LOCATION_DESCRIPTION'])
# Create mapping dict
location_map = dict(enumerate(location_index))
changelog.append(f"Factorized LOCATION_DESCRIPTION into LOCATION_CODE ({len(location_map)} unique locations)")

# Step 8: Scale numeric features (YEAR, VALUE)
# Rationale: Scaling helps models converge and makes z-score based outlier detection comparable.
numeric_cols = ['YEAR', 'VALUE']
if StandardScaler is None:
    changelog.append("scikit-learn not available: numeric scaling skipped. To enable scaling, install scikit-learn and joblib.")
    df_y['YEAR_scaled'] = df_y['YEAR']
    df_y['VALUE_scaled'] = df_y['VALUE']
else:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_y[numeric_cols])
    df_y['YEAR_scaled'] = scaled[:,0]
    df_y['VALUE_scaled'] = scaled[:,1]
    # save scaler for downstream reproducibility
    scaler_path = os.path.join(output_dir, 'scaler_year_value.joblib')
    joblib.dump(scaler, scaler_path)
    changelog.append(f"Scaled numeric columns {numeric_cols} with StandardScaler and saved scaler to {scaler_path}")

# Step 9: Outlier handling for VALUE (yield)
# Rationale: Step 1 showed VALUE had large spread because of mixed stats; after filtering to yields, we still need to detect outliers.
Q1 = df_y['VALUE'].quantile(0.25)
Q3 = df_y['VALUE'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Flag outliers using IQR rule
df_y['is_outlier_iqr'] = (df_y['VALUE'] < lower_bound) | (df_y['VALUE'] > upper_bound)
num_outliers = df_y['is_outlier_iqr'].sum()
changelog.append(f"Outlier detection using IQR: Q1={Q1}, Q3={Q3}, IQR={IQR}; flagged {int(num_outliers)} outliers")

# Also compute z-score outliers (absolute z > 3)
mean_val = df_y['VALUE'].mean()
std_val = df_y['VALUE'].std()
df_y['value_zscore'] = (df_y['VALUE'] - mean_val) / std_val
num_z_outliers = (df_y['value_zscore'].abs() > 3).sum()
changelog.append(f"Z-score method: mean={mean_val:.2f}, std={std_val:.2f}; {int(num_z_outliers)} points with |z|>3")

# Create Winsorized/clipped VALUE to reduce effect of extreme outliers for modeling
lower_pct = df_y['VALUE'].quantile(0.01)
upper_pct = df_y['VALUE'].quantile(0.99)
df_y['VALUE_clipped'] = df_y['VALUE'].clip(lower=lower_pct, upper=upper_pct)
changelog.append(f"Created 'VALUE_clipped' by winsorizing at 1st and 99th percentiles: ({lower_pct}, {upper_pct})")

# Step 10: Save cleaned data and changelog
clean_path = os.path.join(output_dir, 'cleaned_yield_data.csv')
df_y.to_csv(clean_path, index=False)
changelog.append(f"Saved cleaned data to {clean_path} (rows: {df_y.shape[0]}, cols: {df_y.shape[1]})")

changelog_path = os.path.join(output_dir, 'changelog.txt')
with open(changelog_path, 'w') as f:
    for line in changelog:
        f.write(line + '\n')

print('Step 2 preprocessing complete.')
print('Cleaned data saved to', clean_path)
print('Changelog saved to', changelog_path)
