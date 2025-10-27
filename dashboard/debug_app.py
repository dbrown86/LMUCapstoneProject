#!/usr/bin/env python3
"""
Debug Streamlit App - Check page imports
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="🎯 Debug Dashboard",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Debug Dashboard")

# Test page imports
st.subheader("🔍 Testing Page Imports")

try:
    from pages import overview
    st.success("✅ Overview page imported successfully")
except Exception as e:
    st.error(f"❌ Error importing overview: {e}")

try:
    from pages import search
    st.success("✅ Search page imported successfully")
except Exception as e:
    st.error(f"❌ Error importing search: {e}")

try:
    from pages import predictions
    st.success("✅ Predictions page imported successfully")
except Exception as e:
    st.error(f"❌ Error importing predictions: {e}")

try:
    from pages import explanations
    st.success("✅ Explanations page imported successfully")
except Exception as e:
    st.error(f"❌ Error importing explanations: {e}")

try:
    from pages import analytics
    st.success("✅ Analytics page imported successfully")
except Exception as e:
    st.error(f"❌ Error importing analytics: {e}")

# Test data loading
st.subheader("📊 Testing Data Loading")

try:
    import pandas as pd
    donors_df = pd.read_csv('data/synthetic_donor_dataset/donors.csv')
    st.success(f"✅ Data loaded: {len(donors_df)} donors")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")

# Test page functions
st.subheader("🧪 Testing Page Functions")

if st.button("Test Overview Page"):
    try:
        overview.show()
    except Exception as e:
        st.error(f"❌ Error in overview.show(): {e}")

if st.button("Test Search Page"):
    try:
        search.show()
    except Exception as e:
        st.error(f"❌ Error in search.show(): {e}")

