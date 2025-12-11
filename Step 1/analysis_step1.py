import os
import sys
import io
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
cwd = os.getcwd()
csv_filename = "US-Rice-Acreage-Production-and-yield copy.csv"
csv_path = os.path.join(cwd, csv_filename)
output_dir = os.path.join(cwd, "Step 1")
plots_dir = os.path.join(output_dir, "plots")

os.makedirs(plots_dir, exist_ok=True)

if not os.path.exists(csv_path):
    print(f"ERROR: CSV not found at {csv_path}")
    sys.exit(1)

# Read CSV
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print("Failed to read CSV:", e)
    sys.exit(1)

# Capture info and basic summaries
buf = io.StringIO()
df.info(buf=buf)
info_str = buf.getvalue()

# Use describe without datetime_is_numeric for compatibility
describe_all = df.describe(include='all').to_string()

# Value counts for non-numeric cols
value_counts_str = []
for col in df.columns:
    if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
        vc = df[col].value_counts(dropna=False)
        vc_top = vc.head(20).to_string()
        value_counts_str.append(f"--- Value counts for {col} (top 20) ---\n{vc_top}\n")

# Save summary
summary_path = os.path.join(output_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Data inspection generated on {datetime.now().isoformat()}\n\n")
    f.write("--- DataFrame.info() ---\n")
    f.write(info_str + "\n\n")
    f.write("--- describe(include='all') ---\n")
    f.write(describe_all + "\n\n")
    for s in value_counts_str:
        f.write(s + "\n")

print("Saved text summary to", summary_path)

# Numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns:", num_cols)

# Histograms
for col in num_cols:
    try:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col].dropna(), kde=False, bins=30)
        plt.title(f"Histogram of {col}")
        plt.tight_layout()
        out = os.path.join(plots_dir, f"hist_{col}.png")
        plt.savefig(out)
        plt.close()
    except Exception as e:
        print(f"Failed histogram for {col}:", e)

# Boxplots
for col in num_cols:
    try:
        plt.figure(figsize=(6,3))
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        out = os.path.join(plots_dir, f"box_{col}.png")
        plt.savefig(out)
        plt.close()
    except Exception as e:
        print(f"Failed boxplot for {col}:", e)

# Try to locate likely columns: year, acreage, production, yield
cols_lower = {c.lower(): c for c in df.columns}

def find_col(keywords):
    for kw in keywords:
        for lower, orig in cols_lower.items():
            if kw in lower:
                return orig
    return None

year_col = find_col(["year"]) 
acre_col = find_col(["acre", "acres", "area"])
prod_col = find_col(["production", "prod"])
yield_col = find_col(["yield"]) 

pairs = []
if year_col and yield_col:
    pairs.append((year_col, yield_col))
if acre_col and yield_col:
    pairs.append((acre_col, yield_col))
if prod_col and yield_col:
    pairs.append((prod_col, yield_col))
if acre_col and prod_col:
    pairs.append((acre_col, prod_col))

# Scatter plots for identified pairs
for xcol, ycol in pairs:
    try:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[xcol], y=df[ycol])
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        plt.title(f"Scatter {ycol} vs {xcol}")
        plt.tight_layout()
        out = os.path.join(plots_dir, f"scatter_{ycol}_vs_{xcol}.png")
        plt.savefig(out)
        plt.close()
    except Exception as e:
        print(f"Failed scatter {ycol} vs {xcol}:", e)

# If year and yield exist, also compute yearly aggregate mean yield and plot trendline
if year_col and yield_col:
    try:
        # coerce year to numeric if possible
        df_year = df.copy()
        df_year[year_col] = pd.to_numeric(df_year[year_col], errors='coerce')
        yearly = df_year.groupby(year_col)[yield_col].mean().reset_index()
        plt.figure(figsize=(8,4))
        sns.lineplot(x=yearly[year_col], y=yearly[yield_col], marker='o')
        plt.xlabel(year_col)
        plt.ylabel(f"Mean {yield_col}")
        plt.title(f"Mean {yield_col} by {year_col}")
        plt.tight_layout()
        out = os.path.join(plots_dir, f"mean_{yield_col}_by_{year_col}.png")
        plt.savefig(out)
        plt.close()
    except Exception as e:
        print("Failed yearly mean plot:", e)

print("Plots saved to", plots_dir)
print("Done.")
