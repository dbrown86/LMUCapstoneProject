#!/usr/bin/env python3
"""
Predictions Display Page
Shows donor predictions with confidence scores and visualizations
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

def get_confidence_level(confidence):
    """Get confidence level category"""
    if confidence >= 0.8:
        return "High", "confidence-high"
    elif confidence >= 0.6:
        return "Medium", "confidence-medium"
    else:
        return "Low", "confidence-low"

def create_donor_card(donor_data, prediction_data=None):
    """Create a donor prediction card"""
    
    # Extract donor info
    donor_id = donor_data.get('ID', 'N/A')
    name = donor_data.get('Name', 'Unknown')
    age = donor_data.get('Age', 0)
    total_giving = donor_data.get('Total_Giving', 0)
    legacy_intent = donor_data.get('Legacy_Intent_Binary', False)
    
    # Get prediction data
    if prediction_data is not None:
        confidence = prediction_data.get('confidence', 0.5)
        prediction = prediction_data.get('prediction', 0)
        probability = prediction_data.get('probability', 0.5)
    else:
        confidence = 0.5
        prediction = 0
        probability = 0.5
    
    # Determine confidence level
    conf_level, conf_class = get_confidence_level(confidence)
    
    # Create card HTML
    card_html = f"""
    <div class="donor-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #1f77b4;">{name}</h4>
            <span class="{conf_class}">{conf_level} Confidence</span>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
            <div>
                <strong>ID:</strong> {donor_id}<br>
                <strong>Age:</strong> {age}<br>
                <strong>Total Giving:</strong> ${total_giving:,.0f}
            </div>
            <div>
                <strong>Prediction:</strong> {'Legacy Intent' if prediction == 1 else 'No Legacy Intent'}<br>
                <strong>Probability:</strong> {probability:.3f}<br>
                <strong>Actual:</strong> {'Legacy Intent' if legacy_intent == 1 else 'No Legacy Intent'}
            </div>
        </div>
        
        <div style="background-color: #f8f9fa; padding: 0.5rem; border-radius: 0.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>Confidence Score:</span>
                <div style="flex: 1; margin: 0 1rem;">
                    <div style="background-color: #e9ecef; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background-color: {'#28a745' if confidence >= 0.8 else '#ffc107' if confidence >= 0.6 else '#dc3545'}; 
                                    height: 100%; width: {confidence * 100}%; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                <span class="{conf_class}">{confidence:.3f}</span>
            </div>
        </div>
    </div>
    """
    
    return card_html

def show():
    """Display the predictions page"""
    
    st.header("📊 Donor Predictions")
    st.markdown("View model predictions with confidence scores and detailed analysis.")
    
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
        To see model predictions:
        
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
        return
    
    # Extract prediction data
    y_test = predictions.get('y_test', [])
    y_pred = predictions.get('y_pred', [])
    y_pred_proba = predictions.get('y_pred_proba', [])
    
    if len(y_test) == 0:
        st.error("No prediction data available in the results file.")
        return
    
    # Create predictions dataframe
    pred_df = pd.DataFrame({
        'ID': donors_df['ID'].iloc[:len(y_test)],
        'Name': donors_df['Full_Name'].iloc[:len(y_test)],
        'Age': donors_df['Estimated_Age'].iloc[:len(y_test)],
        'Total_Giving': donors_df['Lifetime_Giving'].iloc[:len(y_test)],
        'Actual': y_test,
        'Prediction': y_pred,
        'Probability': y_pred_proba,
        'Confidence': np.abs(y_pred_proba - 0.5) * 2  # Convert to 0-1 confidence scale
    })
    
    # Prediction Summary
    st.subheader("📈 Prediction Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_predictions = len(pred_df)
        st.metric("Total Predictions", f"{total_predictions:,}")
    
    with col2:
        correct_predictions = (pred_df['Actual'] == pred_df['Prediction']).sum()
        accuracy = correct_predictions / total_predictions
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    with col3:
        high_confidence = len(pred_df[pred_df['Confidence'] >= 0.8])
        st.metric("High Confidence", f"{high_confidence:,}")
    
    with col4:
        legacy_predictions = len(pred_df[pred_df['Prediction'] == 1])
        st.metric("Legacy Predictions", f"{legacy_predictions:,}")
    
    # Confidence Distribution
    st.markdown("---")
    st.subheader("📊 Confidence Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence histogram
        fig_conf = px.histogram(
            pred_df,
            x='Confidence',
            nbins=20,
            title="Prediction Confidence Distribution",
            color_discrete_sequence=['#1f77b4']
        )
        fig_conf.update_layout(
            xaxis_title="Confidence Score",
            yaxis_title="Number of Predictions"
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        # Confidence vs Accuracy
        pred_df['Correct'] = (pred_df['Actual'] == pred_df['Prediction']).astype(int)
        conf_bins = pd.cut(pred_df['Confidence'], bins=5, labels=['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
        conf_accuracy = pred_df.groupby(conf_bins)['Correct'].mean().reset_index()
        
        fig_acc = px.bar(
            conf_accuracy,
            x='Confidence',
            y='Correct',
            title="Accuracy by Confidence Level",
            color='Correct',
            color_continuous_scale='RdYlGn'
        )
        fig_acc.update_layout(
            xaxis_title="Confidence Range",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    # Individual Predictions
    st.markdown("---")
    st.subheader("👥 Individual Predictions")
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_filter = st.selectbox(
            "Filter by Confidence:",
            ['All', 'High (≥0.8)', 'Medium (0.6-0.8)', 'Low (<0.6)']
        )
    
    with col2:
        prediction_filter = st.selectbox(
            "Filter by Prediction:",
            ['All', 'Legacy Intent', 'No Legacy Intent']
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by:",
            ['Confidence (High to Low)', 'Confidence (Low to High)', 'Name', 'Age', 'Total Giving']
        )
    
    # Apply filters
    filtered_pred_df = pred_df.copy()
    
    if confidence_filter == 'High (≥0.8)':
        filtered_pred_df = filtered_pred_df[filtered_pred_df['Confidence'] >= 0.8]
    elif confidence_filter == 'Medium (0.6-0.8)':
        filtered_pred_df = filtered_pred_df[(filtered_pred_df['Confidence'] >= 0.6) & (filtered_pred_df['Confidence'] < 0.8)]
    elif confidence_filter == 'Low (<0.6)':
        filtered_pred_df = filtered_pred_df[filtered_pred_df['Confidence'] < 0.6]
    
    if prediction_filter == 'Legacy Intent':
        filtered_pred_df = filtered_pred_df[filtered_pred_df['Prediction'] == 1]
    elif prediction_filter == 'No Legacy Intent':
        filtered_pred_df = filtered_pred_df[filtered_pred_df['Prediction'] == 0]
    
    # Sort data
    if sort_by == 'Confidence (High to Low)':
        filtered_pred_df = filtered_pred_df.sort_values('Confidence', ascending=False)
    elif sort_by == 'Confidence (Low to High)':
        filtered_pred_df = filtered_pred_df.sort_values('Confidence', ascending=True)
    elif sort_by == 'Name':
        filtered_pred_df = filtered_pred_df.sort_values('Name')
    elif sort_by == 'Age':
        filtered_pred_df = filtered_pred_df.sort_values('Age')
    elif sort_by == 'Total Giving':
        filtered_pred_df = filtered_pred_df.sort_values('Total_Giving', ascending=False)
    
    # Display results
    st.markdown(f"### Showing {len(filtered_pred_df)} predictions")
    
    # Pagination
    page_size = 10
    total_pages = (len(filtered_pred_df) - 1) // page_size + 1
    
    if total_pages > 1:
        page = st.selectbox(f"Page (1 of {total_pages}):", range(1, total_pages + 1))
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = filtered_pred_df.iloc[start_idx:end_idx]
    else:
        page_df = filtered_pred_df
    
    # Display donor cards
    for idx, row in page_df.iterrows():
        donor_data = {
            'ID': row['ID'],
            'Name': row['Name'],
            'Age': row['Age'],
            'Total_Giving': row['Total_Giving'],
            'Legacy_Intent_Binary': row['Actual']
        }
        
        prediction_data = {
            'confidence': row['Confidence'],
            'prediction': row['Prediction'],
            'probability': row['Probability']
        }
        
        card_html = create_donor_card(donor_data, prediction_data)
        st.markdown(card_html, unsafe_allow_html=True)
    
    # Export functionality
    st.markdown("---")
    st.subheader("📤 Export Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export All Predictions", use_container_width=True):
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"donor_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📋 Export Filtered Results", use_container_width=True):
            csv = filtered_pred_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered CSV",
                data=csv,
                file_name=f"filtered_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📈 Export Summary Report", use_container_width=True):
            summary = {
                'Total Predictions': len(pred_df),
                'Accuracy': accuracy,
                'High Confidence Count': high_confidence,
                'Legacy Predictions': legacy_predictions,
                'Filters Applied': {
                    'Confidence': confidence_filter,
                    'Prediction': prediction_filter,
                    'Sort By': sort_by
                }
            }
            st.json(summary)
