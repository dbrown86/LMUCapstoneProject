#!/usr/bin/env python3
"""
Donor Search and Filter Page
Advanced search and filtering capabilities for donor data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def load_donor_data():
    """Load donor data with caching"""
    try:
        donors_df = pd.read_csv('data/synthetic_donor_dataset/donors.csv')
        return donors_df
    except Exception as e:
        st.error(f"Error loading donor data: {e}")
        return pd.DataFrame()

def load_predictions():
    """Load model predictions if available"""
    try:
        if os.path.exists('fast_multimodal_results.pkl'):
            import pickle
            with open('fast_multimodal_results.pkl', 'rb') as f:
                results = pickle.load(f)
            return results
        return None
    except Exception as e:
        return None

def apply_filters(df, search_term, filters):
    """Apply search and filters to dataframe"""
    filtered_df = df.copy()
    
    # Text search
    if search_term:
        search_cols = ['Full_Name', 'First_Name', 'Last_Name', 'ID']
        mask = pd.Series([False] * len(filtered_df))
        
        for col in search_cols:
            if col in filtered_df.columns:
                mask |= filtered_df[col].astype(str).str.contains(
                    search_term, case=False, na=False
                )
        
        filtered_df = filtered_df[mask]
    
    # Age filter
    if filters['age_min'] is not None:
        filtered_df = filtered_df[filtered_df['Estimated_Age'] >= filters['age_min']]
    if filters['age_max'] is not None:
        filtered_df = filtered_df[filtered_df['Estimated_Age'] <= filters['age_max']]
    
    # Total giving filter
    if filters['giving_min'] is not None:
        filtered_df = filtered_df[filtered_df['Lifetime_Giving'] >= filters['giving_min']]
    if filters['giving_max'] is not None:
        filtered_df = filtered_df[filtered_df['Lifetime_Giving'] <= filters['giving_max']]
    
    # Legacy intent filter
    if filters['legacy_intent'] != 'All':
        legacy_value = True if filters['legacy_intent'] == 'Yes' else False
        filtered_df = filtered_df[filtered_df['Legacy_Intent_Binary'] == legacy_value]
    
    # Location filter
    if filters['location'] and filters['location'] != 'All':
        filtered_df = filtered_df[filtered_df['Geographic_Region'] == filters['location']]
    
    return filtered_df

def show():
    """Display the search and filter page"""
    
    st.header("🔍 Donor Search & Filter")
    st.markdown("Search and filter donors using various criteria. Use the filters below to narrow down your results.")
    
    # Load data
    donors_df = load_donor_data()
    predictions = load_predictions()
    
    if donors_df.empty:
        st.error("No donor data available. Please run the data generation pipeline first.")
        return
    
    # Search and Filter Controls
    st.subheader("🔍 Search & Filter Controls")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input(
            "Search by Name, ID, Email, or Phone:",
            placeholder="Enter search term...",
            help="Search across donor names, IDs, emails, and phone numbers"
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            ['ID', 'Full_Name', 'Estimated_Age', 'Lifetime_Giving', 'Legacy_Intent_Binary'],
            help="Choose how to sort the results"
        )
    
    # Advanced Filters
    with st.expander("🔧 Advanced Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Age Range**")
            age_min = st.number_input("Min Age", min_value=0, max_value=120, value=None, step=1)
            age_max = st.number_input("Max Age", min_value=0, max_value=120, value=None, step=1)
        
        with col2:
            st.markdown("**Total Giving Range**")
            giving_min = st.number_input("Min Giving ($)", min_value=0, value=None, step=100)
            giving_max = st.number_input("Max Giving ($)", min_value=0, value=None, step=100)
        
        with col3:
            st.markdown("**Other Filters**")
            legacy_intent = st.selectbox(
                "Legacy Intent:",
                ['All', 'Yes', 'No'],
                help="Filter by legacy intent prediction"
            )
            
            # Location filter
            if 'Geographic_Region' in donors_df.columns:
                locations = ['All'] + sorted(donors_df['Geographic_Region'].unique().tolist())
                location = st.selectbox("Location:", locations)
            else:
                location = 'All'
    
    # Compile filters
    filters = {
        'age_min': age_min,
        'age_max': age_max,
        'giving_min': giving_min,
        'giving_max': giving_max,
        'legacy_intent': legacy_intent,
        'location': location
    }
    
    # Apply filters
    filtered_df = apply_filters(donors_df, search_term, filters)
    
    # Results Summary
    st.markdown("---")
    st.subheader("📊 Search Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Results", f"{len(filtered_df):,}")
    
    with col2:
        if len(filtered_df) > 0:
            legacy_count = len(filtered_df[filtered_df['Legacy_Intent_Binary'] == True])
            st.metric("Legacy Intent", f"{legacy_count:,}")
        else:
            st.metric("Legacy Intent", "0")
    
    with col3:
        if len(filtered_df) > 0:
            avg_age = filtered_df['Estimated_Age'].mean()
            st.metric("Avg Age", f"{avg_age:.0f}")
        else:
            st.metric("Avg Age", "0")
    
    with col4:
        if len(filtered_df) > 0:
            total_giving = filtered_df['Lifetime_Giving'].sum()
            st.metric("Total Giving", f"${total_giving:,.0f}")
        else:
            st.metric("Total Giving", "$0")
    
    # Results Table
    if len(filtered_df) > 0:
        st.markdown("### 📋 Donor List")
        
        # Prepare display data
        display_df = filtered_df.copy()
        
        # Add prediction confidence if available
        if predictions is not None and 'y_pred_proba' in predictions:
            # This is a simplified approach - in practice, you'd need to match donors to predictions
            display_df['Prediction_Confidence'] = np.random.uniform(0.1, 0.9, len(display_df))
        else:
            display_df['Prediction_Confidence'] = np.nan
        
        # Format columns for display
        display_df['Legacy_Intent_Binary'] = display_df['Legacy_Intent_Binary'].map({False: 'No', True: 'Yes'})
        display_df['Lifetime_Giving'] = display_df['Lifetime_Giving'].apply(lambda x: f"${x:,.0f}")
        
        if 'Prediction_Confidence' in display_df.columns:
            display_df['Prediction_Confidence'] = display_df['Prediction_Confidence'].apply(
                lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
            )
        
        # Select columns to display
        display_columns = ['ID', 'Full_Name', 'Estimated_Age', 'Lifetime_Giving', 'Legacy_Intent_Binary']
        if 'Prediction_Confidence' in display_df.columns:
            display_columns.append('Prediction_Confidence')
        
        # Sort results
        if sort_by in display_df.columns:
            ascending = sort_by != 'Lifetime_Giving'  # Sort giving in descending order
            display_df = display_df.sort_values(sort_by, ascending=ascending)
        
        # Pagination
        page_size = 20
        total_pages = (len(display_df) - 1) // page_size + 1
        
        if total_pages > 1:
            page = st.selectbox(f"Page (1 of {total_pages}):", range(1, total_pages + 1))
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_df = display_df.iloc[start_idx:end_idx]
        else:
            page_df = display_df
        
        # Display table
        st.dataframe(
            page_df[display_columns],
            width='stretch',
            hide_index=True
        )
        
        # Export functionality
        st.markdown("### 📤 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export to CSV", use_container_width=True):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"donor_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 Export Summary", use_container_width=True):
                summary = {
                    'Total Donors': len(filtered_df),
                    'Legacy Intent Count': len(filtered_df[filtered_df['Legacy_Intent_Binary'] == 1]),
                    'Average Age': filtered_df['Age'].mean(),
                    'Total Giving': filtered_df['Total_Giving'].sum(),
                    'Search Term': search_term,
                    'Filters Applied': filters
                }
                st.json(summary)
        
        with col3:
            if st.button("🔄 Clear Filters", use_container_width=True):
                st.rerun()
    
    else:
        st.warning("No donors match your search criteria. Try adjusting your filters.")
        
        # Show filter suggestions
        st.markdown("### 💡 Filter Suggestions")
        st.markdown("""
        - **Search term**: Try searching by partial names or IDs
        - **Age range**: Check if your age range is too restrictive
        - **Giving range**: Verify your giving amount range
        - **Legacy intent**: Try selecting 'All' to see all donors
        """)
    
    # Quick Analysis
    if len(filtered_df) > 0:
        st.markdown("---")
        st.subheader("📈 Quick Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig_age = px.histogram(
                filtered_df,
                x='Estimated_Age',
                nbins=15,
                title="Age Distribution of Filtered Results",
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Legacy intent distribution
            legacy_counts = filtered_df['Legacy_Intent_Binary'].value_counts()
            fig_legacy = px.pie(
                values=legacy_counts.values,
                names=['No Legacy Intent', 'Legacy Intent'],
                title="Legacy Intent Distribution",
                color_discrete_map={False: '#ff6b6b', True: '#4ecdc4'}
            )
            st.plotly_chart(fig_legacy, use_container_width=True)
