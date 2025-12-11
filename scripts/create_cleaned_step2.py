import pandas as pd
from pathlib import Path

BASE = Path.cwd()
orig = BASE / 'US-Rice-Acreage-Production-and-yield copy.csv'
step2 = BASE / 'Step 2'
step2.mkdir(parents=True, exist_ok=True)
out = step2 / 'cleaned_yield_data.csv'

if not orig.exists():
    print('Original CSV not found:', orig)
else:
    df = pd.read_csv(orig)
    if 'STATISTIC_DESCRIPTION' in df.columns:
        df = df[df['STATISTIC_DESCRIPTION'].str.contains('YIELD', na=False)]
    # normalize year
    if 'YEAR' not in df.columns and 'Year' in df.columns:
        df = df.rename(columns={'Year':'YEAR'})
    # value column
    val_col = None
    for c in ['VALUE','YIELD','Value']:
        if c in df.columns:
            val_col = c
            break
    if val_col is None or 'YEAR' not in df.columns:
        print('Required columns not found. YEAR present?', 'YEAR' in df.columns, 'value column found?', val_col)
    else:
        df['VALUE'] = pd.to_numeric(df[val_col], errors='coerce')
        df_year = df.groupby('YEAR', as_index=False)['VALUE'].mean()
        df_year.to_csv(out, index=False)
        print('Wrote cleaned file to', out)
