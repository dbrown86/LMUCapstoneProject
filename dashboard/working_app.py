#!/usr/bin/env python3
"""
Working Streamlit App - All pages in one file
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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add page_modules to path for imports
page_modules_path = Path(__file__).parent / "page_modules"
sys.path.insert(0, str(page_modules_path))

st.set_page_config(
    page_title="🎯 Donor Legacy Prediction Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Cache data loading
@st.cache_data(ttl=300)  # Cache for 5 minutes to allow for data updates
def load_donor_data():
    """Load donor data with caching"""
    try:
        donors_df = pd.read_csv('data/synthetic_donor_dataset/donors.csv')
        return donors_df
    except Exception as e:
        st.error(f"Error loading donor data: {e}")
        return pd.DataFrame()

@st.cache_data
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

def show_overview():
    """Overview page content"""
    # Centered title at the top
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #1f77b4; font-size: 2.5rem; margin-bottom: 0.5rem;">🎯 Donor Legacy Prediction Dashboard</h1>
        <h2 style="color: #666; font-size: 1.2rem; font-weight: normal;">Comprehensive Analytics & Insights</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("🏠 Overview")
    
    # Add cache clear button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("")  # Spacing
    with col2:
        if st.button("🔄 Refresh Data", type="secondary"):
            st.cache_data.clear()
            st.rerun()
    
    donors_df = load_donor_data()
    if donors_df.empty:
        st.error("❌ Could not load donor data")
        return
    
    st.success(f"✅ Data loaded successfully: {len(donors_df)} donors")
    
    # Key metrics
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
    
    # Row 1: Key Distribution Charts
    st.subheader("📊 Key Distributions")
    col1, col2 = st.columns(2)
    
    with col1:
        # Legacy intent distribution
        legacy_counts = donors_df['Legacy_Intent_Binary'].value_counts()
        fig_pie = px.pie(
            values=legacy_counts.values,
            names=['No Legacy Intent', 'Legacy Intent'],
            color_discrete_map={False: '#ff6b6b', True: '#4ecdc4'},
            title="Legacy Intent Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Gender distribution
        if 'Gender' in donors_df.columns:
            gender_counts = donors_df['Gender'].value_counts()
            fig_gender = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Gender Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_gender, use_container_width=True)
    
    # Row 2: Age and Giving Analysis
    st.subheader("📈 Age & Giving Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.histogram(
            donors_df, 
            x='Estimated_Age', 
            nbins=30,
            title="Donor Age Distribution",
            color_discrete_sequence=['#4ecdc4']
        )
        fig_age.update_layout(xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Lifetime giving distribution
        fig_giving = px.histogram(
            donors_df, 
            x='Lifetime_Giving', 
            nbins=30,
            title="Lifetime Giving Distribution",
            color_discrete_sequence=['#ff6b6b']
        )
        fig_giving.update_layout(xaxis_title="Lifetime Giving ($)", yaxis_title="Count")
        st.plotly_chart(fig_giving, use_container_width=True)
    
    # Row 3: Advanced Analytics
    st.subheader("🔍 Advanced Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        # Age vs Legacy Intent
        fig_age_legacy = px.box(
            donors_df, 
            x='Legacy_Intent_Binary', 
            y='Estimated_Age',
            title="Age Distribution by Legacy Intent",
            color='Legacy_Intent_Binary',
            color_discrete_map={False: '#ff6b6b', True: '#4ecdc4'}
        )
        fig_age_legacy.update_layout(xaxis_title="Legacy Intent", yaxis_title="Age")
        st.plotly_chart(fig_age_legacy, use_container_width=True)
    
    with col2:
        # Giving vs Legacy Intent
        fig_giving_legacy = px.box(
            donors_df, 
            x='Legacy_Intent_Binary', 
            y='Lifetime_Giving',
            title="Giving Distribution by Legacy Intent",
            color='Legacy_Intent_Binary',
            color_discrete_map={False: '#ff6b6b', True: '#4ecdc4'}
        )
        fig_giving_legacy.update_layout(xaxis_title="Legacy Intent", yaxis_title="Lifetime Giving ($)")
        st.plotly_chart(fig_giving_legacy, use_container_width=True)
    
    # Row 4: Scatter Plot Analysis
    st.subheader("📊 Relationship Analysis")
    fig_scatter = px.scatter(
        donors_df,
        x='Estimated_Age',
        y='Lifetime_Giving',
        color='Legacy_Intent_Binary',
        size='Lifetime_Giving',
        hover_data=['Full_Name', 'ID'],
        title="Age vs Lifetime Giving (Size = Giving Amount)",
        color_discrete_map={False: '#ff6b6b', True: '#4ecdc4'},
        labels={'Estimated_Age': 'Age', 'Lifetime_Giving': 'Lifetime Giving ($)'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Row 5: Gift Officer Analysis
    if 'Primary_Manager' in donors_df.columns:
        st.subheader("👨‍💼 Gift Officer Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Gift officer portfolio sizes
            officer_counts = donors_df[donors_df['Primary_Manager'].notna()]['Primary_Manager'].value_counts()
            fig_officer_portfolio = px.bar(
                x=officer_counts.index,
                y=officer_counts.values,
                title="Donor Portfolio Size by Gift Officer",
                color=officer_counts.values,
                color_continuous_scale='viridis'
            )
            fig_officer_portfolio.update_layout(
                xaxis_title="Gift Officer", 
                yaxis_title="Number of Donors",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig_officer_portfolio, use_container_width=True)
        
        with col2:
            # Legacy intent rate by gift officer
            officer_legacy = donors_df.groupby('Primary_Manager')['Legacy_Intent_Binary'].agg(['count', 'sum', 'mean']).reset_index()
            officer_legacy.columns = ['Gift_Officer', 'Total_Donors', 'Legacy_Donors', 'Legacy_Rate']
            officer_legacy = officer_legacy.sort_values('Legacy_Rate', ascending=False)
            
            fig_officer_legacy = px.bar(
                officer_legacy,
                x='Gift_Officer',
                y='Legacy_Rate',
                title="Legacy Intent Rate by Gift Officer",
                color='Legacy_Rate',
                color_continuous_scale='RdYlGn',
                text='Legacy_Rate'
            )
            fig_officer_legacy.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig_officer_legacy.update_layout(
                xaxis_title="Gift Officer", 
                yaxis_title="Legacy Intent Rate",
                xaxis_tickangle=45,
                yaxis_tickformat='.0%'
            )
            st.plotly_chart(fig_officer_legacy, use_container_width=True)
        
        # Gift Officer Performance Table
        st.subheader("📊 Gift Officer Performance Metrics")
        officer_metrics = donors_df.groupby('Primary_Manager').agg({
            'ID': 'count',
            'Legacy_Intent_Binary': ['sum', 'mean'],
            'Lifetime_Giving': ['sum', 'mean', 'median'],
            'Estimated_Age': 'mean'
        }).round(2)
        
        officer_metrics.columns = ['Total_Donors', 'Legacy_Donors', 'Legacy_Rate', 
                                 'Total_Giving', 'Avg_Giving', 'Median_Giving', 'Avg_Age']
        officer_metrics['Legacy_Rate'] = officer_metrics['Legacy_Rate'].apply(lambda x: f"{x:.1%}")
        officer_metrics['Total_Giving'] = officer_metrics['Total_Giving'].apply(lambda x: f"${x:,.0f}")
        officer_metrics['Avg_Giving'] = officer_metrics['Avg_Giving'].apply(lambda x: f"${x:,.0f}")
        officer_metrics['Median_Giving'] = officer_metrics['Median_Giving'].apply(lambda x: f"${x:,.0f}")
        officer_metrics['Avg_Age'] = officer_metrics['Avg_Age'].apply(lambda x: f"{x:.0f}")
        
        st.dataframe(officer_metrics, use_container_width=True)
        
        # Gift Officer vs Rating Analysis
        if 'Rating' in donors_df.columns:
            st.subheader("⭐ Gift Officer Performance by Donor Rating")
            col1, col2 = st.columns(2)
            
            with col1:
                # Rating distribution by gift officer
                assigned_donors = donors_df[donors_df['Primary_Manager'].notna()]
                rating_officer = pd.crosstab(assigned_donors['Primary_Manager'], assigned_donors['Rating'])
                fig_rating_officer = px.bar(
                    rating_officer,
                    title="Donor Rating Distribution by Gift Officer",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_rating_officer.update_layout(
                    xaxis_title="Gift Officer", 
                    yaxis_title="Number of Donors",
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_rating_officer, use_container_width=True)
            
            with col2:
                # Legacy intent rate by rating and officer
                rating_legacy = donors_df.groupby(['Primary_Manager', 'Rating'])['Legacy_Intent_Binary'].mean().unstack(fill_value=0)
                fig_rating_legacy = px.bar(
                    rating_legacy,
                    title="Legacy Intent Rate by Gift Officer & Rating",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_rating_legacy.update_layout(
                    xaxis_title="Gift Officer", 
                    yaxis_title="Legacy Intent Rate",
                    xaxis_tickangle=45,
                    yaxis_tickformat='.0%'
                )
                st.plotly_chart(fig_rating_legacy, use_container_width=True)
    
    # Row 6: Constituent Type Analysis
    if 'Primary_Constituent_Type' in donors_df.columns:
        st.subheader("👥 Constituent Type Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Constituent type distribution
            constituent_counts = donors_df['Primary_Constituent_Type'].value_counts()
            fig_constituent = px.bar(
                x=constituent_counts.index,
                y=constituent_counts.values,
                title="Donors by Constituent Type",
                color=constituent_counts.values,
                color_continuous_scale='viridis'
            )
            fig_constituent.update_layout(xaxis_title="Constituent Type", yaxis_title="Count")
            st.plotly_chart(fig_constituent, use_container_width=True)
        
        with col2:
            # Constituent type vs Legacy Intent
            constituent_legacy = donors_df.groupby(['Primary_Constituent_Type', 'Legacy_Intent_Binary']).size().unstack(fill_value=0)
            fig_constituent_legacy = px.bar(
                constituent_legacy,
                title="Legacy Intent by Constituent Type",
                color_discrete_map={False: '#ff6b6b', True: '#4ecdc4'}
            )
            fig_constituent_legacy.update_layout(xaxis_title="Constituent Type", yaxis_title="Count")
            st.plotly_chart(fig_constituent_legacy, use_container_width=True)
    
    # Row 6: Top Performers
    st.subheader("🏆 Top Performers")
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 donors by giving
        top_donors = donors_df.nlargest(10, 'Lifetime_Giving')[['Full_Name', 'Lifetime_Giving', 'Legacy_Intent_Binary']]
        fig_top_donors = px.bar(
            top_donors,
            x='Lifetime_Giving',
            y='Full_Name',
            orientation='h',
            title="Top 10 Donors by Lifetime Giving",
            color='Legacy_Intent_Binary',
            color_discrete_map={False: '#ff6b6b', True: '#4ecdc4'}
        )
        fig_top_donors.update_layout(xaxis_title="Lifetime Giving ($)", yaxis_title="Donor")
        st.plotly_chart(fig_top_donors, use_container_width=True)
    
    with col2:
        # Legacy intent rate by age group
        donors_df['Age_Group'] = pd.cut(donors_df['Estimated_Age'], 
                                      bins=[0, 30, 40, 50, 60, 70, 80, 100], 
                                      labels=['<30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+'])
        age_legacy_rate = donors_df.groupby('Age_Group', observed=True)['Legacy_Intent_Binary'].mean().reset_index()
        fig_age_rate = px.bar(
            age_legacy_rate,
            x='Age_Group',
            y='Legacy_Intent_Binary',
            title="Legacy Intent Rate by Age Group",
            color='Legacy_Intent_Binary',
            color_continuous_scale='viridis'
        )
        fig_age_rate.update_layout(xaxis_title="Age Group", yaxis_title="Legacy Intent Rate")
        st.plotly_chart(fig_age_rate, use_container_width=True)
    
    # Row 7: Gift Officer Correlation Analysis
    if 'Primary_Manager' in donors_df.columns:
        st.subheader("🔗 Gift Officer Correlation Analysis")
        
        # Calculate correlation metrics
        officer_legacy_corr = donors_df.groupby('Primary_Manager')['Legacy_Intent_Binary'].mean().sort_values(ascending=False)
        officer_giving_corr = donors_df.groupby('Primary_Manager')['Lifetime_Giving'].mean().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Legacy Intent Rate by Gift Officer (Ranked)**")
            for i, (officer, rate) in enumerate(officer_legacy_corr.items(), 1):
                color = "🟢" if rate > donors_df['Legacy_Intent_Binary'].mean() else "🔴"
                st.write(f"{i}. {officer}: {rate:.1%} {color}")
        
        with col2:
            st.write("**Average Giving by Gift Officer (Ranked)**")
            for i, (officer, giving) in enumerate(officer_giving_corr.items(), 1):
                color = "🟢" if giving > donors_df['Lifetime_Giving'].mean() else "🔴"
                st.write(f"{i}. {officer}: ${giving:,.0f} {color}")
        
        # Statistical correlation
        st.write("**📊 Statistical Insights:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Chi-square test for gift officer vs legacy intent
            try:
                from scipy.stats import chi2_contingency
                assigned_donors = donors_df[donors_df['Primary_Manager'].notna() & (donors_df['Primary_Manager'] != '')]
                if len(assigned_donors) > 0:
                    contingency_table = pd.crosstab(assigned_donors['Primary_Manager'], assigned_donors['Legacy_Intent_Binary'])
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    st.metric("Chi-Square p-value", f"{p_value:.4f}")
                    st.caption("Lower p-value = stronger correlation")
                else:
                    st.metric("Chi-Square p-value", "N/A")
                    st.caption("No assigned donors")
            except Exception as e:
                st.metric("Chi-Square p-value", "Error")
                st.caption(f"Error: {str(e)[:50]}...")
        
        with col2:
            # Cramér's V (effect size)
            try:
                if len(assigned_donors) > 0 and 'contingency_table' in locals():
                    n = contingency_table.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                    st.metric("Cramér's V", f"{cramers_v:.3f}")
                    st.caption("0.1=weak, 0.3=moderate, 0.5=strong")
                else:
                    st.metric("Cramér's V", "N/A")
                    st.caption("No assigned donors")
            except Exception as e:
                st.metric("Cramér's V", "Error")
                st.caption(f"Error: {str(e)[:50]}...")
        
        with col3:
            # Best performing officer
            try:
                if len(assigned_donors) > 0:
                    officer_legacy_corr = assigned_donors.groupby('Primary_Manager')['Legacy_Intent_Binary'].mean().sort_values(ascending=False)
                    best_officer = officer_legacy_corr.index[0]
                    best_rate = officer_legacy_corr.iloc[0]
                    st.metric("Top Performer", best_officer)
                    st.caption(f"Rate: {best_rate:.1%}")
                else:
                    st.metric("Top Performer", "N/A")
                    st.caption("No assigned donors")
            except Exception as e:
                st.metric("Top Performer", "Error")
                st.caption(f"Error: {str(e)[:50]}...")
    
    # Row 8: Summary Statistics
    st.subheader("📈 Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Legacy Intent Rate", f"{donors_df['Legacy_Intent_Binary'].mean():.1%}")
        st.metric("Median Age", f"{donors_df['Estimated_Age'].median():.0f}")
    
    with col2:
        st.metric("Median Lifetime Giving", f"${donors_df['Lifetime_Giving'].median():,.0f}")
        st.metric("Max Lifetime Giving", f"${donors_df['Lifetime_Giving'].max():,.0f}")
    
    with col3:
        legacy_donors = donors_df[donors_df['Legacy_Intent_Binary'] == True]
        if len(legacy_donors) > 0:
            st.metric("Avg Age (Legacy Intent)", f"{legacy_donors['Estimated_Age'].mean():.0f}")
            st.metric("Avg Giving (Legacy Intent)", f"${legacy_donors['Lifetime_Giving'].mean():,.0f}")
    
    # Sample data
    st.subheader("📋 Sample Data")
    sample_data = donors_df.head(10)[['ID', 'Full_Name', 'Estimated_Age', 'Lifetime_Giving', 'Legacy_Intent_Binary']]
    st.dataframe(sample_data, width='stretch')

def show_search():
    """Search and filter page content"""
    st.title("🔍 Search & Filter")
    
    donors_df = load_donor_data()
    if donors_df.empty:
        st.error("❌ Could not load donor data")
        return
    
    # Search and filters
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search by name or ID:", placeholder="Enter name or ID...", key="search_input")
        
        # Clear search button
        if search_term:
            if st.button("Clear Search", type="secondary"):
                st.rerun()
    
    with col2:
        # Show search suggestions
        if search_term and len(search_term) >= 2:
            suggestions = []
            for col in ['Full_Name', 'First_Name', 'Last_Name']:
                if col in donors_df.columns:
                    matches = donors_df[col].dropna().str.contains(search_term, case=False, na=False)
                    if matches.any():
                        suggestions.extend(donors_df[matches][col].head(3).tolist())
            
            if suggestions:
                unique_suggestions = list(set(suggestions))[:5]
                st.write("**Suggestions:**")
                for suggestion in unique_suggestions:
                    st.write(f"• {suggestion}")
        else:
            st.write("")  # Spacing
    
    # Filter options
    st.subheader("🎛️ Filters")
    
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
                # Convert to string and handle NaN values properly
                col_data = filtered_df[col].fillna('').astype(str)
                mask |= col_data.str.contains(search_term, case=False, na=False)
        filtered_df = filtered_df[mask]
        
        # Debug info (remove in production)
        if len(filtered_df) == 0:
            st.warning(f"No results found for '{search_term}'. Try searching for partial names or IDs.")
    
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
    
    # Results summary
    st.subheader(f"📊 Results ({len(filtered_df):,} donors)")
    
    # Show search examples if no search term
    if not search_term:
        st.info("💡 **Search Tips:** Try searching for names like 'Ashley', 'Williams', or IDs like '1', '100'")
    
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
            legacy_count = len(filtered_df[filtered_df['Legacy_Intent_Binary'] == True])
            st.metric("Legacy Intent", f"{legacy_count:,}")
        
        # Results table
        st.subheader("📋 Filtered Results")
        
        # Sort options
        sort_by = st.selectbox("Sort by:", ['ID', 'Full_Name', 'Estimated_Age', 'Lifetime_Giving', 'Legacy_Intent_Binary'])
        ascending = sort_by != 'Lifetime_Giving'
        
        display_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
        
        # Format for display
        display_df = display_df.copy()
        display_df['Legacy_Intent_Binary'] = display_df['Legacy_Intent_Binary'].map({False: 'No', True: 'Yes'})
        display_df['Lifetime_Giving'] = display_df['Lifetime_Giving'].apply(lambda x: f"${x:,.0f}")
        
        display_columns = ['ID', 'Full_Name', 'Estimated_Age', 'Lifetime_Giving', 'Legacy_Intent_Binary']
        st.dataframe(display_df[display_columns].head(100), width='stretch')
        
        # Visualizations
        if len(filtered_df) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Age Distribution")
                fig_age = px.histogram(
                    filtered_df, 
                    x='Estimated_Age', 
                    nbins=20,
                    title="Filtered Age Distribution"
                )
                st.plotly_chart(fig_age, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Legacy Intent")
                legacy_counts = filtered_df['Legacy_Intent_Binary'].value_counts()
                fig_legacy = px.pie(
                    values=legacy_counts.values,
                    names=['No', 'Yes'],
                    color_discrete_map={False: '#ff6b6b', True: '#4ecdc4'},
                    title="Legacy Intent Distribution"
                )
                st.plotly_chart(fig_legacy, use_container_width=True)
    else:
        st.warning("No donors match the current filters. Try adjusting your search criteria.")

def show_predictions():
    """Predictions page content"""
    st.title("📊 Predictions")
    
    # Check if model results are available
    predictions = load_predictions()
    
    if predictions is None:
        st.info("🤖 Model predictions not available. Run the model pipeline first:")
        st.code("python scripts/advanced_multimodal_ensemble.py")
        return
    
    st.success("✅ Model predictions loaded successfully!")
    
    # Display prediction summary
    if 'y_pred' in predictions and 'y_proba' in predictions:
        st.subheader("📈 Prediction Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            positive_predictions = np.sum(predictions['y_pred'])
            st.metric("Positive Predictions", f"{positive_predictions:,}")
        
        with col2:
            avg_confidence = np.mean(predictions['y_proba'])
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        with col3:
            high_confidence = np.sum(predictions['y_proba'] > 0.8)
            st.metric("High Confidence (>0.8)", f"{high_confidence:,}")
        
        # Confidence distribution
        st.subheader("📊 Confidence Distribution")
        fig_conf = px.histogram(
            x=predictions['y_proba'],
            nbins=30,
            title="Prediction Confidence Distribution"
        )
        fig_conf.update_layout(xaxis_title="Confidence Score", yaxis_title="Count")
        st.plotly_chart(fig_conf, use_container_width=True)
    
    else:
        st.warning("Prediction data format not recognized.")

def show_explanations():
    """Model explanations page content"""
    st.title("🧠 Model Explanations")
    
    st.info("🤖 Model explanations will be available after running the model pipeline.")
    st.code("python scripts/advanced_multimodal_ensemble.py")

def show_analytics():
    """Analytics page content"""
    st.title("📈 Analytics")
    
    st.info("📊 Advanced analytics will be available after running the model pipeline.")
    st.code("python scripts/advanced_multimodal_ensemble.py")

# Main app
def main():
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "🔍 Search & Filter", "📊 Predictions", "🧠 Model Explanations", "📈 Analytics"]
    )
    
    # Model status
    def check_model_status():
        try:
            model_files = [
                'fast_multimodal_results.pkl',
                'data/bert_embeddings_real.npy',
                'data/gnn_embeddings_real.npy'
            ]
            return all(os.path.exists(f) for f in model_files)
        except:
            return False
    
    model_ready = check_model_status()
    if model_ready:
        st.sidebar.success("✅ Model Ready")
    else:
        st.sidebar.error("❌ Model Not Ready")
        st.sidebar.markdown("Run the model pipeline first:")
        st.sidebar.code("python scripts/advanced_multimodal_ensemble.py")
    
    # Page routing
    if page == "🏠 Overview":
        show_overview()
    elif page == "🔍 Search & Filter":
        show_search()
    elif page == "📊 Predictions":
        show_predictions()
    elif page == "🧠 Model Explanations":
        show_explanations()
    elif page == "📈 Analytics":
        show_analytics()
    
    # Footer
    st.markdown("---")
    st.markdown("🎯 **Donor Legacy Prediction Dashboard** - Built with Streamlit")

if __name__ == "__main__":
    main()
