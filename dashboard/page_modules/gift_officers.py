import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
import numpy as np
import os

def show_gift_officers():
    """Gift Officers page content"""
    
    # Load data
    @st.cache_data(ttl=300)
    def load_donor_data():
        """Load donor data with caching"""
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
        
    donors_df = load_donor_data()
    if donors_df.empty:
        st.error("❌ Could not load donor data")
        return
    
    # Page header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #1f77b4; font-size: 2.5rem; margin-bottom: 0.5rem;">👨‍💼 Gift Officer Analytics</h1>
        <h2 style="color: #666; font-size: 1.2rem; font-weight: normal;">Portfolio Management & Performance Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if gift officer data exists
    if 'Primary_Manager' not in donors_df.columns:
        st.warning("⚠️ No gift officer data available in the dataset")
        return
    
    # Filter out unassigned donors
    assigned_donors = donors_df[donors_df['Primary_Manager'] != 'Unassigned']
    
    if len(assigned_donors) == 0:
        st.warning("⚠️ No donors are assigned to gift officers")
        return
    
    # Key metrics
    st.subheader("📊 Gift Officer Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Gift Officers", assigned_donors['Primary_Manager'].nunique())
    
    with col2:
        st.metric("Assigned Donors", f"{len(assigned_donors):,}")
    
    with col3:
        assignment_rate = len(assigned_donors) / len(donors_df) * 100
        st.metric("Assignment Rate", f"{assignment_rate:.1f}%")
    
    with col4:
        avg_portfolio = len(assigned_donors) / assigned_donors['Primary_Manager'].nunique()
        st.metric("Avg Portfolio Size", f"{avg_portfolio:.0f}")
    
    # Portfolio Analysis
    st.subheader("📈 Portfolio Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gift officer portfolio sizes
        officer_counts = assigned_donors['Primary_Manager'].value_counts()
        fig_portfolio = px.bar(
            x=officer_counts.index,
            y=officer_counts.values,
            title="Donor Portfolio Size by Gift Officer",
            color=officer_counts.values,
            color_continuous_scale='viridis'
        )
        fig_portfolio.update_layout(
            xaxis_title="Gift Officer", 
            yaxis_title="Number of Donors",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig_portfolio, use_container_width=True)
    
    with col2:
        # Portfolio size distribution
        fig_dist = px.histogram(
            officer_counts.values,
            nbins=10,
            title="Portfolio Size Distribution",
            labels={'x': 'Portfolio Size', 'y': 'Number of Officers'}
        )
        fig_dist.update_layout(
            xaxis_title="Portfolio Size",
            yaxis_title="Number of Officers"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Performance Analysis
    st.subheader("🎯 Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Legacy intent rate by gift officer
        officer_legacy = assigned_donors.groupby('Primary_Manager', observed=True)['Legacy_Intent_Binary'].agg(['count', 'sum', 'mean']).reset_index()
        officer_legacy.columns = ['Gift_Officer', 'Total_Donors', 'Legacy_Donors', 'Legacy_Rate']
        officer_legacy = officer_legacy.sort_values('Legacy_Rate', ascending=False)
        
        fig_legacy = px.bar(
            officer_legacy,
            x='Gift_Officer',
            y='Legacy_Rate',
            title="Legacy Intent Rate by Gift Officer",
            color='Legacy_Rate',
            color_continuous_scale='RdYlGn',
            text='Legacy_Rate'
        )
        fig_legacy.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_legacy.update_layout(
            xaxis_title="Gift Officer", 
            yaxis_title="Legacy Intent Rate",
            xaxis_tickangle=45,
            yaxis_tickformat='.0%'
        )
        st.plotly_chart(fig_legacy, use_container_width=True)
    
    with col2:
        # Total giving by gift officer
        officer_giving = assigned_donors.groupby('Primary_Manager', observed=True)['Lifetime_Giving'].sum().reset_index()
        officer_giving = officer_giving.sort_values('Lifetime_Giving', ascending=False)
        
        fig_giving = px.bar(
            officer_giving,
            x='Primary_Manager',
            y='Lifetime_Giving',
            title="Total Lifetime Giving by Gift Officer",
            color='Lifetime_Giving',
            color_continuous_scale='Blues'
        )
        fig_giving.update_layout(
            xaxis_title="Gift Officer",
            yaxis_title="Total Lifetime Giving ($)",
            xaxis_tickangle=45,
            yaxis_tickformat='$,.0f'
        )
        st.plotly_chart(fig_giving, use_container_width=True)
    
    # Performance Metrics Table
    st.subheader("📋 Performance Metrics Table")
    
    officer_metrics = assigned_donors.groupby('Primary_Manager', observed=True).agg({
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
    
    # Rating Analysis
    if 'Rating' in assigned_donors.columns:
        st.subheader("⭐ Donor Rating Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution by gift officer
            rating_officer = pd.crosstab(assigned_donors['Primary_Manager'], assigned_donors['Rating'])
            fig_rating = px.bar(
                rating_officer,
                title="Donor Rating Distribution by Gift Officer",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_rating.update_layout(
                xaxis_title="Gift Officer", 
                yaxis_title="Number of Donors",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig_rating, use_container_width=True)
        
        with col2:
            # Legacy intent rate by rating and officer
            rating_legacy = assigned_donors.groupby(['Primary_Manager', 'Rating'], observed=True)['Legacy_Intent_Binary'].mean().unstack(fill_value=0)
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
    
    # Statistical Analysis
    st.subheader("📊 Statistical Correlation Analysis")
    
    try:
        # Chi-square test for gift officer vs legacy intent
        contingency_table = pd.crosstab(assigned_donors['Primary_Manager'], assigned_donors['Legacy_Intent_Binary'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Cramér's V
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Chi-Square p-value", f"{p_value:.4f}")
        
        with col2:
            st.metric("Cramér's V", f"{cramers_v:.4f}")
        
        with col3:
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            st.metric("Correlation", significance)
        
        # Top performers
        best_performer = officer_legacy.iloc[0]
        st.info(f"🏆 **Top Performer**: {best_performer['Gift_Officer']} with {best_performer['Legacy_Rate']:.1%} legacy intent rate")
        
    except Exception as e:
        st.warning(f"⚠️ Could not perform statistical analysis: {e}")
    
    # Export functionality
    st.subheader("📤 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Performance Metrics", type="secondary"):
            csv = officer_metrics.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="gift_officer_performance.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("👥 Export Assigned Donors", type="secondary"):
            csv = assigned_donors.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="assigned_donors.csv",
                mime="text/csv"
            )
