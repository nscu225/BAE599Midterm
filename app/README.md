Streamlit app for the US Rice Yield analysis

Files:
- `app.py` : main Streamlit app. Use `streamlit run app/app.py` from the repository root to start.

How to run:

1. Install packages:

pip install -r app/requirements.txt

2. From repository root run:

streamlit run app/app.py

Notes:
- The app reads artifacts from Step 1..Step 6 folders created by earlier scripts. Run those steps first if files are missing.
- Embeds the interactive HTML from `Step 6/interactive_yield.html` when available.
