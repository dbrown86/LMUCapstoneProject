#!/usr/bin/env bash
set -euo pipefail

# Disable pagers for cleaner output
export PAGER=; export GIT_PAGER=

echo "== DonorAI Dashboard Data Debugger =="

echo ""
echo "[1/4] Python environment check"
python - <<'PY'
import sys
print("Python:", sys.version.replace("\n"," "))
try:
    import pandas as pd; print("pandas:", pd.__version__)
except Exception as e:
    print("pandas unavailable:", e)
try:
    import plotly; print("plotly:", plotly.__version__)
except Exception as e:
    print("plotly unavailable:", e)
PY

echo ""
echo "[2/4] Column overview from load_full_dataset"
python - <<'PY'
import pandas as pd
from dashboard.data.loader import load_full_dataset

df = load_full_dataset(use_cache=False)
print("Rows:", len(df))
print("Columns:", list(df.columns))
# basic stats
for col in ["segment","region","predicted_prob","Will_Give_Again_Probability","donor_type","avg_gift","total_giving","days_since_last","Last_Gift_Date","actual_gave"]:
    if col in df.columns:
        obj = df[col]
        if isinstance(obj, pd.DataFrame):
            series = pd.to_numeric(obj.iloc[:, 0], errors='coerce')
        else:
            series = pd.to_numeric(obj, errors='coerce')
        print(f"{col!r}: non-null {series.notna().sum()}, unique {series.nunique()}")
    else:
        print(f"Missing column -> {col}")
# duplicates?
dupes = df.columns[df.columns.duplicated()]
print("Duplicate columns:", list(dupes))
PY

echo ""
echo "[3/4] Sample values (head)"
python - <<'PY'
from dashboard.data.loader import load_full_dataset
cols = ["donor_id","segment","region","predicted_prob","Will_Give_Again_Probability","avg_gift","total_giving"]
df = load_full_dataset(use_cache=False)
available_cols = [c for c in cols if c in df.columns]
print("Showing columns:", available_cols)
print(df[available_cols].head(10))
PY

echo ""
echo "[4/4] Dashboard render smoke test (console only)"
python - <<'PY'
import pandas as pd
from dashboard.data.loader import load_full_dataset
from dashboard.components.sidebar import render_sidebar
from dashboard.pages.dashboard import render as render_dashboard

try:
    import streamlit as st
    print("Streamlit import ok, this script should be run via 'streamlit run' to view visuals.")
except Exception as e:
    print("Streamlit import failed:", e)

df = load_full_dataset(use_cache=False)
missing = [c for c in ["segment","region","predicted_prob"] if c not in df.columns]
print("Missing columns for dashboard charts:", missing)

# mimic filters: pass empty selections
try:
    render_dashboard(df, [], [], [], 0.5)
    print("Dashboard render() executed without raising exceptions.")
except Exception as e:
    import traceback
    print("Dashboard render() raised an exception:")
    traceback.print_exc()
PY

echo ""
echo "== Done. Review the output above for missing columns or errors. =="

