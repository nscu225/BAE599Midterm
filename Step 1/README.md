Step 1 - Initial data inspection

This folder contains the automated script and outputs for the first step of the analysis: a thorough initial inspection of the "US-Rice-Acreage-Production-and-yield copy.csv" dataset.

Files:
- analysis_step1.py : Python script that reads the CSV, generates summary statistics, and saves visualizations.
- summary.txt : Generated data inspection text (created when the script is run).
- plots/ : PNG files with histograms, boxplots, scatter plots, and a yearly mean trend if available.

How to run:
- Ensure you have Python 3 and the packages in requirements.txt installed.
- From the repository root run:

python3 "Step 1/analysis_step1.py"

Outputs will be saved into the Step 1 folder.