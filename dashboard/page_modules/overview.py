#!/usr/bin/env python3
"""
Dashboard Overview Page
Shows key metrics, recent predictions, and model status
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
        # Try to load cached predictions
        if os.path.exists('fast_multimodal_results.pkl'):
            import pickle
            with open('fast_multimodal_results.pkl', 'rb') as f:
                results = pickle.load(f)
            return results
        return None
    except Exception as e:
        st.warning(f"Predictions not available: {e}")
        return None

def show():
    """Display the overview page"""
    
    st.header("🏠 Dashboard Overview")
    st.markdown("Welcome to the Donor Prediction Dashboard! Here's a summary of your donor data and model performance.")
    
    # Load data
    donors_df = load_donor_data()
    predictions = load_predictions()
    
    if donors_df.empty:
        st.error("No donor data available. Please run the data generation pipeline first.")
        return
    
    # Key Metrics Row
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_donors = len(donors_df)
        st.metric(
            label="Total Donors",
            value=f"{total_donors:,}",
            delta=None
        )
    
    with col2:
        legacy_donors = len(donors_df[donors_df['Legacy_Intent_Binary'] == 1])
        legacy_rate = (legacy_donors / total_donors) * 100
        st.metric(
            label="Legacy Intent",
            value=f"{legacy_donors:,}",
            delta=f"{legacy_rate:.1f}%"
        )
    
    with col3:
        avg_age = donors_df['Estimated_Age'].mean()
        st.metric(
            label="Average Age",
            value=f"{avg_age:.0f}",
            delta="years"
        )
    
    with col4:
        total_giving = donors_df['Lifetime_Giving'].sum()
        st.metric(
            label="Total Giving",
            value=f"${total_giving:,.0f}",
            delta="lifetime"
        )
    
    st.markdown("---")
    
    # Data Distribution Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Legacy Intent Distribution")
        
        # Legacy intent pie chart
        legacy_counts = donors_df['Legacy_Intent_Binary'].value_counts()
        fig_pie = px.pie(
            values=legacy_counts.values,
            names=['No Legacy Intent', 'Legacy Intent'],
            color_discrete_map={0: '#ff6b6b', 1: '#4ecdc4'},
            title="Donor Legacy Intent Distribution"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("👥 Age Distribution")
        
        # Age distribution histogram
        fig_age = px.histogram(
            donors_df,
            x='Estimated_Age',
            nbins=20,
            title="Donor Age Distribution",
            color_discrete_sequence=['#1f77b4']
        )
        fig_age.update_layout(
            xaxis_title="Age",
            yaxis_title="Number of Donors",
            showlegend=False
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Model Performance Section
    st.markdown("---")
    st.subheader("🤖 Model Performance")
    
    if predictions is not None:
        # Extract performance metrics from predictions
        if 'test_metrics' in predictions:
            metrics = predictions['test_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="AUC-ROC",
                    value=f"{metrics.get('auc_roc', 0):.3f}",
                    delta="Area Under Curve"
                )
            
            with col2:
                st.metric(
                    label="Accuracy",
                    value=f"{metrics.get('accuracy', 0):.3f}",
                    delta="Overall Accuracy"
                )
            
            with col3:
                st.metric(
                    label="Precision",
                    value=f"{metrics.get('precision', 0):.3f}",
                    delta="Positive Predictions"
                )
            
            with col4:
                st.metric(
                    label="F1-Score",
                    value=f"{metrics.get('f1_score', 0):.3f}",
                    delta="Balanced Metric"
                )
        
        # Model performance visualization
        if 'y_test' in predictions and 'y_pred_proba' in predictions:
            st.subheader("📊 Prediction Confidence Distribution")
            
            y_test = predictions['y_test']
            y_pred_proba = predictions['y_pred_proba']
            
            # Create confidence distribution
            confidence_data = pd.DataFrame({
                'Confidence': y_pred_proba,
                'Actual': y_test,
                'Prediction': (y_pred_proba > 0.5).astype(int)
            })
            
            fig_conf = px.histogram(
                confidence_data,
                x='Confidence',
                color='Actual',
                nbins=20,
                title="Prediction Confidence Distribution",
                color_discrete_map={0: '#ff6b6b', 1: '#4ecdc4'}
            )
            fig_conf.update_layout(
                xaxis_title="Prediction Confidence",
                yaxis_title="Number of Predictions"
            )
            st.plotly_chart(fig_conf, use_container_width=True)
    
    else:
        st.info("💡 No model predictions available. Run the model pipeline to see performance metrics.")
        
        # Show instructions
        st.markdown("### 🚀 Get Started")
        st.markdown("""
        To see model predictions and performance metrics:
        
        1. **Run the model pipeline**:
           ```bash
           python scripts/advanced_multimodal_ensemble.py
           ```
        
        2. **Or run the quick pipeline**:
           ```bash
           python scripts/quick_ml_pipeline.py
           ```
        
        3. **Refresh this page** to see the results!
        """)
    
    # Recent Activity Section
    st.markdown("---")
    st.subheader("📋 Recent Activity")
    
    # Show sample of recent donors
    st.markdown("### Sample Donor Data")
    
    # Display sample data
    sample_donors = donors_df.head(10)[['ID', 'Full_Name', 'Estimated_Age', 'Lifetime_Giving', 'Legacy_Intent_Binary']].copy()
    sample_donors['Legacy_Intent_Binary'] = sample_donors['Legacy_Intent_Binary'].map({False: 'No', True: 'Yes'})
    sample_donors.columns = ['ID', 'Name', 'Age', 'Total Giving', 'Legacy Intent']
    
    st.dataframe(
        sample_donors,
        width='stretch',
        hide_index=True
    )
    
    # Quick Actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Search Donors", use_container_width=True):
            st.info("Use the sidebar to navigate to the Search & Filter page")
    
    with col2:
        if st.button("📊 View Predictions", use_container_width=True):
            st.info("Use the sidebar to navigate to the Predictions page")
    
    with col3:
        if st.button("🧠 Model Explanations", use_container_width=True):
            st.info("Use the sidebar to navigate to the Model Explanations page")
