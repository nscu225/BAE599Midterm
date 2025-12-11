import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()
clean_path = os.path.join(cwd, 'Step 2', 'cleaned_yield_data.csv')
out_dir = os.path.join(cwd, 'Step 6', 'screenshots')
os.makedirs(out_dir, exist_ok=True)

if not os.path.exists(clean_path):
    raise FileNotFoundError(clean_path)

# Load and aggregate
df = pd.read_csv(clean_path)
mask = (df['is_us_total'] == True) & (df['is_all_classes'] == True)
df_us = df[mask].copy()
if df_us.empty:
    df_us = df[(df['LOCATION_DESCRIPTION'].str.upper()=='U.S. TOTAL') & (df['CLASS_DESCRIPTION'].str.upper()=='ALL CLASSES')].copy()

agg = df_us.sort_values('YEAR').groupby('YEAR').agg({'VALUE':'mean','VALUE_clipped':'mean'}).reset_index()
years = agg['YEAR'].values
raw = agg['VALUE'].values
clipped = agg['VALUE_clipped'].values

# Linear fit
coef = np.polyfit(years, clipped, 1)
fit = coef[0]*years + coef[1]

# Full range screenshot
plt.figure(figsize=(10,5))
plt.plot(years, clipped, marker='o', color='blue', label='Yield (clipped)')
plt.plot(years, raw, marker='o', color='lightblue', linestyle='None', label='Yield (raw)')
plt.plot(years, fit, color='orange', linestyle='--', label=f'Linear fit (slope={coef[0]:.2f})')
plt.xlabel('Year')
plt.ylabel('Yield (pounds per acre)')
plt.title('U.S. Mean Rice Yield by Year')
plt.legend()
plt.grid(True)
plt.tight_layout()
full_path = os.path.join(out_dir, 'yield_full_range.png')
plt.savefig(full_path)
plt.close()

# Zoom recent years (last 20 years)
recent_years = years[-20:]
recent_clipped = clipped[-20:]
recent_raw = raw[-20:]
recent_fit = fit[-20:]

plt.figure(figsize=(10,5))
plt.plot(recent_years, recent_clipped, marker='o', color='blue', label='Yield (clipped)')
plt.plot(recent_years, recent_raw, marker='o', color='lightblue', linestyle='None', label='Yield (raw)')
plt.plot(recent_years, recent_fit, color='orange', linestyle='--', label=f'Linear fit (slope={coef[0]:.2f})')
plt.xlabel('Year')
plt.ylabel('Yield (pounds per acre)')
plt.title('U.S. Mean Rice Yield — Recent 20 Years')
plt.legend()
plt.grid(True)
plt.tight_layout()
zoom_path = os.path.join(out_dir, 'yield_recent_20yrs.png')
plt.savefig(zoom_path)
plt.close()

print('Saved screenshots:', full_path, zoom_path)
