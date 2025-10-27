#!/usr/bin/env python3
"""
Simplified Streamlit Dashboard - Robust Version
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import page modules
from page_modules.gift_officers import show_gift_officers

st.set_page_config(
    page_title="🎯 Donor Legacy Prediction Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading with error handling
@st.cache_data(ttl=300)
def load_donor_data():
    """Load donor data with caching and error handling"""
    try:
        # Try multiple possible paths
        possible_paths = [
            'data/synthetic_donor_dataset/donors.csv',
            'synthetic_donor_dataset/donors.csv',
            '../data/synthetic_donor_dataset/donors.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                donors_df = pd.read_csv(path)
                return donors_df
        
        st.error("❌ Could not find donor data file")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"❌ Error loading donor data: {str(e)}")
        return pd.DataFrame()

def show_overview():
    """Overview page content with error handling"""
    try:
        # Centered title
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #1f77b4; font-size: 2.5rem; margin-bottom: 0.5rem;">🎯 Donor Legacy Prediction Dashboard</h1>
            <h2 style="color: #666; font-size: 1.2rem; font-weight: normal;">Comprehensive Analytics & Insights</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.title("🏠 Overview")
        
        # Refresh and Export buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("🔄 Refresh Data", type="secondary"):
                st.cache_data.clear()
                st.rerun()
        with col3:
            # Export data button (will be populated after data loads)
            st.markdown("📊 Export Data")
        
        # Load data
        donors_df = load_donor_data()
        if donors_df.empty:
            st.error("❌ Could not load donor data")
            return
        
        # Update export button with actual data
        with col3:
            csv_data = donors_df.to_csv(index=False)
            st.download_button(
                label="📊 Export Data",
                data=csv_data,
                file_name=f"donor_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
        
        # Key metrics
        st.subheader("📊 Key Metrics")
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
        
        # Additional formatted metrics
        st.subheader("💰 Financial Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_giving = donors_df['Lifetime_Giving'].mean()
            st.metric("Average Giving", f"${avg_giving:,.0f}")
        
        with col2:
            median_giving = donors_df['Lifetime_Giving'].median()
            st.metric("Median Giving", f"${median_giving:,.0f}")
        
        with col3:
            max_giving = donors_df['Lifetime_Giving'].max()
            st.metric("Highest Gift", f"${max_giving:,.0f}")
        
        with col4:
            legacy_donors = donors_df[donors_df['Legacy_Intent_Binary'] == True]
            if len(legacy_donors) > 0:
                legacy_avg_giving = legacy_donors['Lifetime_Giving'].mean()
                st.metric("Legacy Donor Avg", f"${legacy_avg_giving:,.0f}")
            else:
                st.metric("Legacy Donor Avg", "N/A")
        
        # Age distribution
        st.subheader("📊 Age Distribution")
        
        # Age distribution - Line graph with age groups for smoother visualization
        age_groups = pd.cut(donors_df['Estimated_Age'], 
                          bins=range(18, 101, 2), 
                          labels=range(19, 100, 2),
                          include_lowest=True)
        age_group_counts = age_groups.value_counts().sort_index()
        
        fig_age = px.line(
            x=age_group_counts.index,
            y=age_group_counts.values,
            title="Donor Age Distribution",
            labels={'x': 'Age', 'y': 'Number of Donors'},
            markers=True
        )
        fig_age.update_traces(
            line=dict(width=3, color='#4ecdc4'),
            marker=dict(size=6, color='#4ecdc4'),
            fill='tonexty'
        )
        fig_age.update_layout(
            xaxis_title="Age",
            yaxis_title="Number of Donors",
            hovermode='x unified',
            showlegend=False
        )
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Gift Officer Quick Summary
        if 'Primary_Manager' in donors_df.columns:
            st.subheader("👨‍💼 Gift Officer Quick Summary")
            
            # Filter out unassigned donors
            assigned_donors = donors_df[donors_df['Primary_Manager'] != 'Unassigned']
            
            if len(assigned_donors) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Gift Officers", assigned_donors['Primary_Manager'].nunique())
                
                with col2:
                    st.metric("Assigned Donors", f"{len(assigned_donors):,}")
                
                with col3:
                    assignment_rate = len(assigned_donors) / len(donors_df) * 100
                    st.metric("Assignment Rate", f"{assignment_rate:.1f}%")
                
                # Link to detailed analysis
                st.info("📊 For detailed gift officer analytics, portfolio analysis, and performance metrics, visit the **Gift Officers** page in the sidebar.")
            else:
                st.info("ℹ️ No donors are currently assigned to gift officers")
        
        # Sample data
        st.subheader("📋 Sample Data")
        sample_cols = ['ID', 'Full_Name', 'Estimated_Age', 'Lifetime_Giving', 'Legacy_Intent_Binary']
        if 'Primary_Manager' in donors_df.columns:
            sample_cols.append('Primary_Manager')
        
        sample_data = donors_df[sample_cols].head(10).copy()
        
        # Format the data for better display
        if 'Lifetime_Giving' in sample_data.columns:
            sample_data['Lifetime_Giving'] = sample_data['Lifetime_Giving'].apply(lambda x: f"${x:,.0f}")
        
        if 'Legacy_Intent_Binary' in sample_data.columns:
            sample_data['Legacy_Intent_Binary'] = sample_data['Legacy_Intent_Binary'].map({True: 'Yes', False: 'No'})
        
        # Primary_Manager already has 'Unassigned' values, no need to fill NaN
        
        # Rename columns for better display
        column_mapping = {
            'ID': 'Donor ID',
            'Full_Name': 'Full Name',
            'Estimated_Age': 'Age',
            'Lifetime_Giving': 'Lifetime Giving',
            'Legacy_Intent_Binary': 'Legacy Intent',
            'Primary_Manager': 'Gift Officer'
        }
        sample_data = sample_data.rename(columns=column_mapping)
        
        st.dataframe(sample_data, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error in overview page: {str(e)}")
        st.write("Please check the data file and try refreshing.")

def show_search():
    """Search page with error handling"""
    try:
        st.title("🔍 Search & Filter")
        
        donors_df = load_donor_data()
        if donors_df.empty:
            st.error("❌ Could not load donor data")
            return
        
        # Search
        search_term = st.text_input("🔍 Search by name or ID:", placeholder="Enter name or ID...")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age_min, age_max = st.slider(
                "Age Range",
                min_value=int(donors_df['Estimated_Age'].min()),
                max_value=int(donors_df['Estimated_Age'].max()),
                value=(int(donors_df['Estimated_Age'].min()), int(donors_df['Estimated_Age'].max()))
            )
        
        with col2:
            giving_min, giving_max = st.slider(
                "Lifetime Giving Range",
                min_value=float(donors_df['Lifetime_Giving'].min()),
                max_value=float(donors_df['Lifetime_Giving'].max()),
                value=(float(donors_df['Lifetime_Giving'].min()), float(donors_df['Lifetime_Giving'].max())),
                format="$%.0f"
            )
        
        with col3:
            legacy_intent = st.selectbox("Legacy Intent", ["All", "Yes", "No"])
        
        # Apply filters
        filtered_df = donors_df.copy()
        
        # Text search
        if search_term:
            search_cols = ['Full_Name', 'First_Name', 'Last_Name', 'ID']
            mask = pd.Series([False] * len(filtered_df))
            for col in search_cols:
                if col in filtered_df.columns:
                    col_data = filtered_df[col].fillna('').astype(str)
                    mask |= col_data.str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
        
        # Age filter
        filtered_df = filtered_df[filtered_df['Estimated_Age'] >= age_min]
        filtered_df = filtered_df[filtered_df['Estimated_Age'] <= age_max]
        
        # Giving filter
        filtered_df = filtered_df[filtered_df['Lifetime_Giving'] >= giving_min]
        filtered_df = filtered_df[filtered_df['Lifetime_Giving'] <= giving_max]
        
        # Legacy intent filter
        if legacy_intent == "Yes":
            filtered_df = filtered_df[filtered_df['Legacy_Intent_Binary'] == True]
        elif legacy_intent == "No":
            filtered_df = filtered_df[filtered_df['Legacy_Intent_Binary'] == False]
        
        # Results
        st.subheader(f"📊 Results ({len(filtered_df):,} donors)")
        
        if len(filtered_df) > 0:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_age = filtered_df['Estimated_Age'].mean()
                st.metric("Average Age", f"{avg_age:.0f}")
            
            with col2:
                total_giving = filtered_df['Lifetime_Giving'].sum()
                st.metric("Total Giving", f"${total_giving:,.0f}")
            
            with col3:
                legacy_rate = filtered_df['Legacy_Intent_Binary'].mean()
                st.metric("Legacy Intent Rate", f"{legacy_rate:.1%}")
            
            # Display results
            display_cols = ['ID', 'Full_Name', 'Estimated_Age', 'Lifetime_Giving', 'Legacy_Intent_Binary']
            if 'Primary_Manager' in filtered_df.columns:
                display_cols.append('Primary_Manager')
            
            display_data = filtered_df[display_cols].copy()
            
            # Format the data for better display
            if 'Lifetime_Giving' in display_data.columns:
                display_data['Lifetime_Giving'] = display_data['Lifetime_Giving'].apply(lambda x: f"${x:,.0f}")
            
            if 'Legacy_Intent_Binary' in display_data.columns:
                display_data['Legacy_Intent_Binary'] = display_data['Legacy_Intent_Binary'].map({True: 'Yes', False: 'No'})
            
            # Primary_Manager already has 'Unassigned' values, no need to fill NaN
            
            # Rename columns for better display
            column_mapping = {
                'ID': 'Donor ID',
                'Full_Name': 'Full Name',
                'Estimated_Age': 'Age',
                'Lifetime_Giving': 'Lifetime Giving',
                'Legacy_Intent_Binary': 'Legacy Intent',
                'Primary_Manager': 'Gift Officer'
            }
            display_data = display_data.rename(columns=column_mapping)
            
            st.dataframe(display_data, use_container_width=True)
        else:
            st.warning("No donors match your search criteria")
            
    except Exception as e:
        st.error(f"❌ Error in search page: {str(e)}")

def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Overview", "Search & Filter", "Gift Officers"])
    
    # Page routing
    if page == "Overview":
        show_overview()
    elif page == "Search & Filter":
        show_search()
    elif page == "Gift Officers":
        show_gift_officers()
    else:
        st.error("Page not found")

if __name__ == "__main__":
    main()
