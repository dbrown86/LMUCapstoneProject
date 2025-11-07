"""
Minimal test to verify dashboard charts render correctly.
Run with: streamlit run test_dashboard_charts.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.data.loader import load_full_dataset
from dashboard.components.charts import plotly_chart_silent

st.set_page_config(page_title="Chart Test", layout="wide")

st.title("Dashboard Chart Rendering Test")

# Load data
df = load_full_dataset(use_cache=False)
st.write(f"✅ Data loaded: {len(df)} rows")

# Remove duplicate columns
df_work = df.loc[:, ~df.columns.duplicated()].copy()

# Test 1: Segment Distribution
st.header("Test 1: Segment Distribution")
if {'segment', 'predicted_prob'}.issubset(df_work.columns):
    seg_df = df_work[['segment', 'predicted_prob']].dropna(subset=['segment'])
    st.write(f"Segment data: {len(seg_df)} rows")
    
    if not seg_df.empty:
        summary = seg_df.groupby('segment', observed=False).size().reset_index(name='Count')
        st.write("Summary data:")
        st.dataframe(summary)
        
        fig_segment = px.bar(summary, x='segment', y='Count', color='segment',
                             category_orders={'segment': ['Recent (0-6mo)', 'Recent (6-12mo)', 'Lapsed (1-2yr)', 'Very Lapsed (2yr+)', 'Prospects/New']},
                             color_discrete_sequence=['#4caf50', '#8bc34a', '#ffc107', '#ff5722', '#9e9e9e'])
        fig_segment.update_traces(texttemplate='%{y:,}', textposition='outside')
        fig_segment.update_layout(height=350, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        
        st.write("Rendering with plotly_chart_silent:")
        plotly_chart_silent(fig_segment)
        
        st.write("Rendering with st.plotly_chart:")
        st.plotly_chart(fig_segment, use_container_width=True)
    else:
        st.error("No segment data available")
else:
    st.error("Segment or prediction columns missing")

# Test 2: Region Distribution
st.header("Test 2: Region Distribution")
if 'region' in df_work.columns:
    reg_series = df_work['region'].dropna()
    st.write(f"Region data: {len(reg_series)} rows")
    
    if not reg_series.empty:
        summary = reg_series.value_counts().reset_index()
        summary.columns = ['Region', 'Count']
        st.write("Summary data:")
        st.dataframe(summary)
        
        fig_region = px.pie(summary, names='Region', values='Count', hole=0.4,
                            color_discrete_sequence=['#2196f3', '#4caf50', '#ff9800', '#9c27b0', '#e91e63'])
        fig_region.update_layout(height=350)
        
        st.write("Rendering with plotly_chart_silent:")
        plotly_chart_silent(fig_region)
        
        st.write("Rendering with st.plotly_chart:")
        st.plotly_chart(fig_region, use_container_width=True)
    else:
        st.error("No region data available")
else:
    st.error("Region column missing")

# Test 3: Confidence Tiers
st.header("Test 3: Confidence Tiers")
if 'predicted_prob' in df_work.columns:
    probs = pd.to_numeric(df_work['predicted_prob'], errors='coerce').dropna()
    st.write(f"Probability data: {len(probs)} rows")
    
    if len(probs):
        bins = [0.0, 0.4, 0.7, 1.0]
        labels = ['Low', 'Medium', 'High']
        tiers = pd.cut(probs, bins=bins, labels=labels, include_lowest=True)
        summary = tiers.value_counts().reindex(labels, fill_value=0).reset_index()
        summary.columns = ['Tier', 'Count']
        st.write("Summary data:")
        st.dataframe(summary)
        
        fig_tiers = px.bar(summary, x='Tier', y='Count', color='Tier',
                           color_discrete_map={'Low': '#f44336', 'Medium': '#ffc107', 'High': '#4caf50'})
        fig_tiers.update_traces(texttemplate='%{y:,}', textposition='outside')
        fig_tiers.update_layout(height=250, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        
        st.write("Rendering with plotly_chart_silent:")
        plotly_chart_silent(fig_tiers)
        
        st.write("Rendering with st.plotly_chart:")
        st.plotly_chart(fig_tiers, use_container_width=True)
    else:
        st.error("Prediction probabilities are present but all values are NaN")
else:
    st.error("Prediction probabilities not available")

st.success("✅ Test complete!")

