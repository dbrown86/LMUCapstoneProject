#!/usr/bin/env python3
"""
Test Streamlit App - Simplified version to debug
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="🎯 Test Dashboard",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Test Dashboard")
st.write("This is a test to see if Streamlit is working properly.")

# Test data loading
try:
    donors_df = pd.read_csv('data/synthetic_donor_dataset/donors.csv')
    st.success(f"✅ Data loaded successfully: {len(donors_df)} donors")
    
    # Show basic info
    st.subheader("📊 Basic Data Info")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Donors", f"{len(donors_df):,}")
    
    with col2:
        legacy_count = len(donors_df[donors_df['Legacy_Intent_Binary'] == True])
        st.metric("Legacy Intent", f"{legacy_count:,}")
    
    with col3:
        avg_age = donors_df['Estimated_Age'].mean()
        st.metric("Average Age", f"{avg_age:.0f}")
    
    # Show sample data
    st.subheader("📋 Sample Data")
    sample_data = donors_df.head(5)[['ID', 'Full_Name', 'Estimated_Age', 'Lifetime_Giving', 'Legacy_Intent_Binary']]
    st.dataframe(sample_data, width='stretch')
    
except Exception as e:
    st.error(f"❌ Error loading data: {e}")

st.write("If you can see this message, Streamlit is working correctly!")

