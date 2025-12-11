import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

cwd = os.getcwd()
clean_path = os.path.join(cwd, 'Step 2', 'cleaned_yield_data.csv')
output_dir = os.path.join(cwd, 'Step 6')
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(clean_path):
    raise FileNotFoundError(f'Cleaned data not found at {clean_path}')

# Read and aggregate national series
df = pd.read_csv(clean_path)
mask = (df['is_us_total'] == True) & (df['is_all_classes'] == True)
df_us = df[mask].copy()
if df_us.empty:
    df_us = df[(df['LOCATION_DESCRIPTION'].str.upper()=='U.S. TOTAL') & (df['CLASS_DESCRIPTION'].str.upper()=='ALL CLASSES')].copy()

agg = df_us.sort_values('YEAR').groupby('YEAR').agg({'VALUE':'mean','VALUE_clipped':'mean'}).reset_index()
years = agg['YEAR'].values
y = agg['VALUE_clipped'].values
raw_y = agg['VALUE'].values

# Linear fit on winsorized values
coef = np.polyfit(years, y, 1)
slope, intercept = coef[0], coef[1]
fit_vals = slope * years + intercept

# Create interactive Plotly figure
fig = go.Figure()
# Actual (winsorized)
fig.add_trace(go.Scatter(x=years, y=y, mode='markers+lines', name='Yield (winsorized)',
                         marker=dict(color='blue'),
                         hovertemplate='Year: %{x}<br>Yield (clipped): %{y:.1f} lb/acre'))
# Raw actual
fig.add_trace(go.Scatter(x=years, y=raw_y, mode='markers', name='Yield (raw)',
                         marker=dict(color='lightblue', symbol='circle-open'),
                         hovertemplate='Year: %{x}<br>Yield (raw): %{y:.1f} lb/acre'))
# Linear fit
fig.add_trace(go.Scatter(x=years, y=fit_vals, mode='lines', name=f'Linear fit (slope={slope:.2f} lb/acre/year)',
                         line=dict(color='orange', dash='dash')))

# Add annotation for slope
fig.add_annotation(x=years[int(len(years)*0.7)], y=fit_vals[int(len(years)*0.7)],
                   text=f'Slope ≈ {slope:.2f} lb/acre/year', showarrow=True, arrowhead=2)

fig.update_layout(title='U.S. Mean Rice Yield by Year — Interactive',
                  xaxis_title='Year',
                  yaxis_title='Yield (pounds per acre)',
                  hovermode='x unified',
                  template='plotly_white')

out_html = os.path.join(output_dir, 'interactive_yield.html')
fig.write_html(out_html, include_plotlyjs='cdn', full_html=True)
print('Interactive HTML saved to', out_html)
