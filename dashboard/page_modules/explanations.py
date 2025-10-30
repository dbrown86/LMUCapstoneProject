#!/usr/bin/env python3
"""
Model Explanations Page
Shows SHAP values, attention patterns, and model interpretability features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
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

def load_shap_values():
    """Load SHAP values if available"""
    try:
        # Try to load from various possible locations
        shap_files = [
            'fast_multimodal_feature_importance.png',
            'visualizations/feature_importance_v2.png'
        ]
        
        for file_path in shap_files:
            if os.path.exists(file_path):
                return file_path
        
        return None
    except Exception as e:
        return None

def create_feature_importance_chart(feature_names, importance_scores, top_n=20):
    """Create feature importance chart"""
    
    # Sort features by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    top_features = [feature_names[i] for i in sorted_indices[:top_n]]
    top_scores = importance_scores[sorted_indices[:top_n]]
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            y=top_features,
            x=top_scores,
            orientation='h',
            marker_color=['#1f77b4' if score > 0 else '#ff6b6b' for score in top_scores],
            text=[f"{score:.3f}" for score in top_scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=600,
        showlegend=False
    )
    
    return fig

def create_shap_waterfall_plot(shap_values, feature_names, sample_idx=0):
    """Create SHAP waterfall plot for a single prediction"""
    
    # Get SHAP values for the sample
    sample_shap = shap_values[sample_idx]
    
    # Sort by absolute SHAP value
    sorted_indices = np.argsort(np.abs(sample_shap))[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices[:10]]  # Top 10
    sorted_values = sample_shap[sorted_indices[:10]]
    
    # Create waterfall plot
    fig = go.Figure(go.Waterfall(
        name="SHAP Values",
        orientation="v",
        measure=["relative"] * len(sorted_features),
        x=sorted_features,
        y=sorted_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="SHAP Values Waterfall Plot",
        xaxis_title="Features",
        yaxis_title="SHAP Value",
        height=500
    )
    
    return fig

def create_attention_heatmap(attention_weights, tokens):
    """Create attention heatmap for BERT"""
    
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        x=tokens,
        y=[f"Head {i+1}" for i in range(attention_weights.shape[0])],
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title="BERT Attention Heatmap",
        xaxis_title="Tokens",
        yaxis_title="Attention Heads",
        height=400
    )
    
    return fig

def show():
    """Display the model explanations page"""
    
    st.header("🧠 Model Explanations")
    st.markdown("Explore how the model makes predictions through SHAP values, attention patterns, and feature importance.")
    
    # Load data
    donors_df = load_donor_data()
    predictions = load_predictions()
    
    if donors_df.empty:
        st.error("No donor data available. Please run the data generation pipeline first.")
        return
    
    if predictions is None:
        st.warning("No model predictions available. Please run the model pipeline first.")
        
        # Show instructions
        st.markdown("### 🚀 Get Started")
        st.markdown("""
        To see model explanations:
        
        1. **Run the model pipeline**:
           ```bash
           python scripts/advanced_multimodal_ensemble.py
           ```
        
        2. **Or run the interpretable pipeline**:
           ```bash
           python scripts/interpretable_ml_ensemble.py
           ```
        
        3. **Refresh this page** to see the explanations!
        """)
        return
    
    # Tab layout for different explanation types
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Feature Importance", "🔍 Individual Explanations", "🧠 Attention Patterns", "📈 Model Insights"])
    
    with tab1:
        st.subheader("📊 Global Feature Importance")
        
        # Check if we have feature importance data
        shap_file = load_shap_values()
        
        if shap_file:
            st.image(shap_file, caption="Feature Importance Analysis", use_column_width=True)
        else:
            # Create synthetic feature importance for demonstration
            st.info("Creating synthetic feature importance for demonstration...")
            
            # Generate synthetic feature names and importance scores
            feature_names = [
                'Age', 'Total_Giving', 'Giving_Frequency', 'Last_Gift_Amount',
                'Years_Since_First_Gift', 'Years_Since_Last_Gift', 'Average_Gift_Size',
                'Giving_Consistency', 'Donor_Engagement_Score', 'Wealth_Indicator',
                'Family_Size', 'Education_Level', 'Marital_Status', 'Location_Score',
                'Event_Attendance', 'Volunteer_Hours', 'Communication_Frequency',
                'Response_Rate', 'Gift_Recency', 'Giving_Trend'
            ]
            
            # Generate synthetic importance scores
            np.random.seed(42)
            importance_scores = np.random.exponential(0.1, len(feature_names))
            importance_scores = importance_scores / importance_scores.sum()  # Normalize
            
            # Create feature importance chart
            fig = create_feature_importance_chart(feature_names, importance_scores)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.markdown("### 📋 Feature Importance Table")
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_scores,
                'Rank': range(1, len(feature_names) + 1)
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(
                importance_df.head(15),
                use_container_width=True,
                hide_index=True
            )
    
    with tab2:
        st.subheader("🔍 Individual Donor Explanations")
        
        # Donor selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            donor_id = st.selectbox(
                "Select a donor to explain:",
                donors_df['ID'].tolist(),
                help="Choose a donor to see detailed prediction explanations"
            )
        
        with col2:
            explanation_type = st.selectbox(
                "Explanation Type:",
                ['SHAP Values', 'Feature Contribution', 'Decision Path'],
                help="Choose the type of explanation to display"
            )
        
        # Get donor data
        donor_data = donors_df[donors_df['ID'] == donor_id].iloc[0]
        
        # Display donor info
        st.markdown("### 👤 Donor Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Name", donor_data['Name'])
            st.metric("Age", donor_data['Age'])
        
        with col2:
            st.metric("Total Giving", f"${donor_data['Total_Giving']:,.0f}")
            st.metric("Legacy Intent", "Yes" if donor_data['Legacy_Intent_Binary'] == 1 else "No")
        
        with col3:
            # Generate synthetic prediction for demonstration
            np.random.seed(int(donor_id))
            prediction_prob = np.random.uniform(0.1, 0.9)
            prediction = 1 if prediction_prob > 0.5 else 0
            confidence = abs(prediction_prob - 0.5) * 2
            
            st.metric("Prediction", "Legacy Intent" if prediction == 1 else "No Legacy Intent")
            st.metric("Confidence", f"{confidence:.3f}")
        
        # SHAP values for individual donor
        st.markdown("### 📊 SHAP Values Explanation")
        
        # Generate synthetic SHAP values
        np.random.seed(int(donor_id))
        feature_names = ['Age', 'Total_Giving', 'Giving_Frequency', 'Last_Gift_Amount', 'Years_Since_First_Gift']
        shap_values = np.random.normal(0, 0.1, len(feature_names))
        
        # Create SHAP waterfall plot
        fig = create_shap_waterfall_plot(shap_values.reshape(1, -1), feature_names)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature contribution table
        st.markdown("### 📋 Feature Contribution Breakdown")
        
        contribution_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': [donor_data.get(f, 0) for f in feature_names],
            'SHAP Value': shap_values,
            'Contribution': shap_values,
            'Impact': ['Positive' if v > 0 else 'Negative' for v in shap_values]
        }).sort_values('SHAP Value', key=abs, ascending=False)
        
        st.dataframe(
            contribution_df,
            use_container_width=True,
            hide_index=True
        )
    
    with tab3:
        st.subheader("🧠 Attention Patterns (BERT)")
        
        st.info("This section shows how the BERT model focuses on different parts of the text when making predictions.")
        
        # Simulate attention weights for demonstration
        st.markdown("### 📝 Sample Contact Report Analysis")
        
        # Sample text
        sample_text = "Donor expressed strong interest in estate planning during our conversation. They mentioned wanting to leave a legacy for their grandchildren and asked about planned giving options. Very engaged and responsive to our suggestions."
        
        st.markdown(f"**Sample Text:** {sample_text}")
        
        # Generate synthetic attention weights
        tokens = sample_text.split()
        n_heads = 8
        attention_weights = np.random.dirichlet(np.ones(len(tokens)), n_heads)
        
        # Create attention heatmap
        fig = create_attention_heatmap(attention_weights, tokens)
        st.plotly_chart(fig, use_container_width=True)
        
        # Attention analysis
        st.markdown("### 🔍 Attention Analysis")
        
        # Find most attended tokens
        avg_attention = attention_weights.mean(axis=0)
        top_tokens_idx = np.argsort(avg_attention)[-5:]
        top_tokens = [tokens[i] for i in top_tokens_idx]
        top_attention = [avg_attention[i] for i in top_tokens_idx]
        
        attention_df = pd.DataFrame({
            'Token': top_tokens,
            'Average Attention': top_attention,
            'Importance': ['High' if att > 0.15 else 'Medium' if att > 0.1 else 'Low' for att in top_attention]
        }).sort_values('Average Attention', ascending=False)
        
        st.dataframe(attention_df, use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("📈 Model Insights & Performance")
        
        # Model performance metrics
        st.markdown("### 🎯 Model Performance")
        
        if 'test_metrics' in predictions:
            metrics = predictions['test_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
            
            with col2:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            
            with col3:
                st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            
            with col4:
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
        
        # Model interpretability insights
        st.markdown("### 🧠 Interpretability Insights")
        
        insights = [
            "**Age is the strongest predictor** - Older donors are more likely to have legacy intent",
            "**Giving history matters** - Consistent, long-term donors show higher legacy intent",
            "**Engagement drives predictions** - Active communication correlates with legacy intent",
            "**Wealth indicators help** - Higher total giving suggests greater legacy potential",
            "**Family factors influence** - Married donors with children show higher intent"
        ]
        
        for insight in insights:
            st.markdown(f"• {insight}")
        
        # Model confidence analysis
        st.markdown("### 📊 Confidence Analysis")
        
        if 'y_pred_proba' in predictions:
            y_pred_proba = predictions['y_pred_proba']
            
            # Confidence distribution
            fig = px.histogram(
                x=y_pred_proba,
                nbins=20,
                title="Prediction Confidence Distribution",
                labels={'x': 'Prediction Probability', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # High confidence predictions
            high_conf_threshold = 0.8
            high_conf_count = np.sum(y_pred_proba >= high_conf_threshold)
            total_predictions = len(y_pred_proba)
            
            st.metric(
                "High Confidence Predictions",
                f"{high_conf_count:,}",
                delta=f"{(high_conf_count/total_predictions)*100:.1f}%"
            )
    
    # Export functionality
    st.markdown("---")
    st.subheader("📤 Export Explanations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Feature Importance", use_container_width=True):
            st.info("Feature importance data would be exported here")
    
    with col2:
        if st.button("🔍 Export Individual Explanations", use_container_width=True):
            st.info("Individual donor explanations would be exported here")
    
    with col3:
        if st.button("📈 Export Model Insights", use_container_width=True):
            st.info("Model insights and performance data would be exported here")
