#!/usr/bin/env python3
"""
Simple Streamlit App - Working version
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="🎯 Donor Legacy Prediction Dashboard",
    page_icon="🎯",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Overview", "🔍 Search & Filter", "📊 Predictions", "🧠 Model Explanations", "📈 Analytics"]
)

# Load data
@st.cache_data
def load_data():
    try:
        donors_df = pd.read_csv('data/synthetic_donor_dataset/donors.csv')
        return donors_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main content
if page == "🏠 Overview":
    st.title("🏠 Overview")
    
    donors_df = load_data()
    if donors_df is not None:
        st.success(f"✅ Data loaded successfully: {len(donors_df)} donors")
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Donors", f"{len(donors_df):,}")
        
        with col2:
            legacy_count = len(donors_df[donors_df['Legacy_Intent_Binary'] == True])
            st.metric("Legacy Intent", f"{legacy_count:,}")
        
        with col3:
            avg_age = donors_df['Estimated_Age'].mean()
            st.metric("Average Age", f"{avg_age:.0f}")
        
        with col4:
            total_giving = donors_df['Lifetime_Giving'].sum()
            st.metric("Total Giving", f"${total_giving:,.0f}")
        
        # Sample data
        st.subheader("📋 Sample Data")
        sample_data = donors_df.head(10)[['ID', 'Full_Name', 'Estimated_Age', 'Lifetime_Giving', 'Legacy_Intent_Binary']]
        st.dataframe(sample_data, width='stretch')
        
    else:
        st.error("❌ Could not load data")

elif page == "🔍 Search & Filter":
    st.title("🔍 Search & Filter")
    
    donors_df = load_data()
    if donors_df is not None:
        st.write("Search and filter functionality will be implemented here.")
        
        # Simple search
        search_term = st.text_input("Search by name:")
        if search_term:
            filtered_df = donors_df[donors_df['Full_Name'].str.contains(search_term, case=False, na=False)]
            st.write(f"Found {len(filtered_df)} donors")
            if len(filtered_df) > 0:
                st.dataframe(filtered_df[['ID', 'Full_Name', 'Estimated_Age', 'Lifetime_Giving']].head(20), width='stretch')
    else:
        st.error("❌ Could not load data")

elif page == "📊 Predictions":
    st.title("📊 Predictions")
    st.info("Run the model pipeline first to see predictions")
    st.code("python scripts/advanced_multimodal_ensemble.py")

elif page == "🧠 Model Explanations":
    st.title("🧠 Model Explanations")
    st.info("Run the model pipeline first to see explanations")
    st.code("python scripts/advanced_multimodal_ensemble.py")

elif page == "📈 Analytics":
    st.title("📈 Analytics")
    st.info("Run the model pipeline first to see analytics")
    st.code("python scripts/advanced_multimodal_ensemble.py")

# Footer
st.markdown("---")
st.markdown("🎯 **Donor Legacy Prediction Dashboard** - Built with Streamlit")

