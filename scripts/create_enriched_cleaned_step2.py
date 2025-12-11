import pandas as pd
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
orig = BASE / 'US-Rice-Acreage-Production-and-yield copy.csv'
step2 = BASE / 'Step 2'
step2.mkdir(parents=True, exist_ok=True)
if not orig.exists():
    print('Original CSV not found at', orig)
    raise SystemExit(1)

df = pd.read_csv(orig)
if 'STATISTIC_DESCRIPTION' in df.columns:
    df = df[df['STATISTIC_DESCRIPTION'].str.contains('YIELD', na=False)]
if 'Year' in df.columns and 'YEAR' not in df.columns:
    df = df.rename(columns={'Year':'YEAR'})
if 'LOCATION' in df.columns and 'LOCATION_DESCRIPTION' not in df.columns:
    df = df.rename(columns={'LOCATION':'LOCATION_DESCRIPTION'})
if 'CLASS' in df.columns and 'CLASS_DESCRIPTION' not in df.columns:
    df = df.rename(columns={'CLASS':'CLASS_DESCRIPTION'})

if 'VALUE' in df.columns:
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')

if 'LOCATION_DESCRIPTION' in df.columns:
    df['is_us_total'] = df['LOCATION_DESCRIPTION'].astype(str).str.upper().str.contains('U.S') | df['LOCATION_DESCRIPTION'].astype(str).str.upper().str.contains('U.S. TOTAL')
else:
    df['is_us_total'] = False

if 'CLASS_DESCRIPTION' in df.columns:
    df['is_all_classes'] = df['CLASS_DESCRIPTION'].astype(str).str.upper().str.contains('ALL')
else:
    df['is_all_classes'] = False

if 'VALUE' in df.columns:
    lower = df['VALUE'].quantile(0.01)
    upper = df['VALUE'].quantile(0.99)
    df['VALUE_clipped'] = df['VALUE'].clip(lower=lower, upper=upper)

# save detailed cleaned
detailed_path = step2 / 'cleaned_yield_data.csv'
df.to_csv(detailed_path, index=False)
# save aggregated by year
if 'YEAR' in df.columns and 'VALUE' in df.columns:
    df_year = df.groupby('YEAR', as_index=False).agg({'VALUE':'mean', 'VALUE_clipped':'mean' if 'VALUE_clipped' in df.columns else 'mean'})
    year_path = step2 / 'cleaned_yield_by_year.csv'
    df_year.to_csv(year_path, index=False)

print('Wrote enriched cleaned files to Step 2:')
print('-', detailed_path)
print('-', year_path)
