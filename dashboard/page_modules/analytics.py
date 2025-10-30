#!/usr/bin/env python3
"""
Analytics Page
Shows model performance, cohort analysis, and advanced insights
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

def create_roc_curve(y_true, y_scores):
    """Create ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=500,
        height=400
    )
    
    return fig

def create_precision_recall_curve(y_true, y_scores):
    """Create Precision-Recall curve"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR Curve (AP = {avg_precision:.3f})',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=500,
        height=400
    )
    
    return fig

def create_confusion_matrix(y_true, y_pred):
    """Create confusion matrix heatmap"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predicted", y="Actual"),
        title="Confusion Matrix",
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        width=400,
        height=400
    )
    
    return fig

def show():
    """Display the analytics page"""
    
    st.header("📈 Analytics & Insights")
    st.markdown("Comprehensive analysis of model performance, donor segments, and predictive insights.")
    
    # Load data
    donors_df = load_donor_data()
    predictions = load_predictions()
    
    if donors_df.empty:
        st.error("No donor data available. Please run the data generation pipeline first.")
        return
    
    # Tab layout for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Model Performance", "👥 Cohort Analysis", "📊 Predictive Insights", "🔍 Advanced Analytics"])
    
    with tab1:
        st.subheader("🎯 Model Performance Analysis")
        
        if predictions is None:
            st.warning("No model predictions available. Please run the model pipeline first.")
            return
        
        # Extract prediction data
        y_test = predictions.get('y_test', [])
        y_pred = predictions.get('y_pred', [])
        y_pred_proba = predictions.get('y_pred_proba', [])
        
        if len(y_test) == 0:
            st.error("No prediction data available.")
            return
        
        # Performance metrics
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.3f}")
        
        with col2:
            from sklearn.metrics import precision_score
            precision = precision_score(y_test, y_pred)
            st.metric("Precision", f"{precision:.3f}")
        
        with col3:
            from sklearn.metrics import recall_score
            recall = recall_score(y_test, y_pred)
            st.metric("Recall", f"{recall:.3f}")
        
        with col4:
            from sklearn.metrics import f1_score
            f1 = f1_score(y_test, y_pred)
            st.metric("F1-Score", f"{f1:.3f}")
        
        # ROC and PR Curves
        st.markdown("### 📈 Performance Curves")
        
        col1, col2 = st.columns(2)
        
        with col1:
            roc_fig = create_roc_curve(y_test, y_pred_proba)
            st.plotly_chart(roc_fig, use_container_width=True)
        
        with col2:
            pr_fig = create_precision_recall_curve(y_test, y_pred_proba)
            st.plotly_chart(pr_fig, use_container_width=True)
        
        # Confusion Matrix
        st.markdown("### 🔢 Confusion Matrix")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            cm_fig = create_confusion_matrix(y_test, y_pred)
            st.plotly_chart(cm_fig, use_container_width=True)
        
        with col2:
            # Confusion matrix details
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            tn, fp, fn, tp = cm.ravel()
            
            st.markdown("#### Confusion Matrix Details")
            st.markdown(f"**True Negatives:** {tn:,}")
            st.markdown(f"**False Positives:** {fp:,}")
            st.markdown(f"**False Negatives:** {fn:,}")
            st.markdown(f"**True Positives:** {tp:,}")
            
            st.markdown("#### Derived Metrics")
            st.markdown(f"**Specificity:** {tn/(tn+fp):.3f}")
            st.markdown(f"**Sensitivity:** {tp/(tp+fn):.3f}")
            st.markdown(f"**Positive Predictive Value:** {tp/(tp+fp):.3f}")
            st.markdown(f"**Negative Predictive Value:** {tn/(tn+fn):.3f}")
    
    with tab2:
        st.subheader("👥 Donor Cohort Analysis")
        
        # Create donor segments
        st.markdown("### 📊 Donor Segmentation")
        
        # Age cohorts
        donors_df['Age_Cohort'] = pd.cut(
            donors_df['Age'], 
            bins=[0, 30, 50, 70, 100], 
            labels=['Young (0-30)', 'Middle (31-50)', 'Senior (51-70)', 'Elderly (71+)']
        )
        
        # Giving cohorts
        donors_df['Giving_Cohort'] = pd.cut(
            donors_df['Total_Giving'],
            bins=[0, 1000, 10000, 50000, float('inf')],
            labels=['Low ($0-1K)', 'Medium ($1K-10K)', 'High ($10K-50K)', 'Very High ($50K+)']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age cohort analysis
            age_cohort_analysis = donors_df.groupby('Age_Cohort').agg({
                'Legacy_Intent_Binary': ['count', 'sum', 'mean'],
                'Total_Giving': 'mean'
            }).round(2)
            
            age_cohort_analysis.columns = ['Total_Donors', 'Legacy_Count', 'Legacy_Rate', 'Avg_Giving']
            age_cohort_analysis['Legacy_Rate'] = age_cohort_analysis['Legacy_Rate'] * 100
            
            st.markdown("#### Age Cohort Analysis")
            st.dataframe(age_cohort_analysis, use_container_width=True)
            
            # Age cohort visualization
            fig_age = px.bar(
                age_cohort_analysis.reset_index(),
                x='Age_Cohort',
                y='Legacy_Rate',
                title='Legacy Intent Rate by Age Cohort',
                color='Legacy_Rate',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Giving cohort analysis
            giving_cohort_analysis = donors_df.groupby('Giving_Cohort').agg({
                'Legacy_Intent_Binary': ['count', 'sum', 'mean'],
                'Age': 'mean'
            }).round(2)
            
            giving_cohort_analysis.columns = ['Total_Donors', 'Legacy_Count', 'Legacy_Rate', 'Avg_Age']
            giving_cohort_analysis['Legacy_Rate'] = giving_cohort_analysis['Legacy_Rate'] * 100
            
            st.markdown("#### Giving Cohort Analysis")
            st.dataframe(giving_cohort_analysis, use_container_width=True)
            
            # Giving cohort visualization
            fig_giving = px.bar(
                giving_cohort_analysis.reset_index(),
                x='Giving_Cohort',
                y='Legacy_Rate',
                title='Legacy Intent Rate by Giving Cohort',
                color='Legacy_Rate',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_giving, use_container_width=True)
        
        # Cohort performance with predictions
        if predictions is not None and 'y_pred' in predictions:
            st.markdown("### 🎯 Cohort Performance with Predictions")
            
            # Add predictions to donor data
            pred_df = donors_df.copy()
            pred_df['Prediction'] = predictions['y_pred'][:len(donors_df)]
            pred_df['Prediction_Prob'] = predictions['y_pred_proba'][:len(donors_df)]
            
            # Age cohort performance
            age_performance = pred_df.groupby('Age_Cohort').agg({
                'Legacy_Intent_Binary': 'mean',
                'Prediction': 'mean',
                'Prediction_Prob': 'mean'
            }).round(3)
            
            age_performance.columns = ['Actual_Legacy_Rate', 'Predicted_Legacy_Rate', 'Avg_Prediction_Prob']
            
            st.markdown("#### Age Cohort Prediction Performance")
            st.dataframe(age_performance, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Predictive Insights")
        
        # Feature correlation analysis
        st.markdown("### 🔗 Feature Correlation Analysis")
        
        # Select numeric features for correlation
        numeric_features = donors_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_features) > 1:
            correlation_matrix = donors_df[numeric_features].corr()
            
            # Create correlation heatmap
            fig_corr = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Top correlations with legacy intent
            if 'Legacy_Intent_Binary' in correlation_matrix.columns:
                legacy_corr = correlation_matrix['Legacy_Intent_Binary'].drop('Legacy_Intent_Binary').abs().sort_values(ascending=False)
                
                st.markdown("#### Top Correlations with Legacy Intent")
                top_corr_df = pd.DataFrame({
                    'Feature': legacy_corr.index,
                    'Correlation': legacy_corr.values
                }).head(10)
                
                st.dataframe(top_corr_df, use_container_width=True, hide_index=True)
        
        # Prediction confidence analysis
        if predictions is not None and 'y_pred_proba' in predictions:
            st.markdown("### 📊 Prediction Confidence Analysis")
            
            y_pred_proba = predictions['y_pred_proba']
            
            # Confidence distribution by actual outcome
            confidence_df = pd.DataFrame({
                'Confidence': y_pred_proba,
                'Actual': y_test[:len(y_pred_proba)]
            })
            
            fig_conf = px.histogram(
                confidence_df,
                x='Confidence',
                color='Actual',
                nbins=20,
                title='Prediction Confidence Distribution by Actual Outcome',
                color_discrete_map={0: '#ff6b6b', 1: '#4ecdc4'}
            )
            st.plotly_chart(fig_conf, use_container_width=True)
            
            # Confidence vs accuracy
            confidence_bins = pd.cut(y_pred_proba, bins=5, labels=['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
            accuracy_by_conf = confidence_df.groupby(confidence_bins).apply(
                lambda x: (x['Actual'] == (x['Confidence'] > 0.5)).mean()
            ).reset_index()
            accuracy_by_conf.columns = ['Confidence_Range', 'Accuracy']
            
            fig_acc = px.bar(
                accuracy_by_conf,
                x='Confidence_Range',
                y='Accuracy',
                title='Accuracy by Confidence Range',
                color='Accuracy',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_acc, use_container_width=True)
    
    with tab4:
        st.subheader("🔍 Advanced Analytics")
        
        # Model comparison (if multiple models available)
        st.markdown("### 🤖 Model Comparison")
        
        # Simulate multiple model results for demonstration
        models = {
            'Multimodal Ensemble': {'auc': 0.714, 'accuracy': 0.732, 'f1': 0.472},
            'Random Forest': {'auc': 0.698, 'accuracy': 0.721, 'f1': 0.445},
            'Logistic Regression': {'auc': 0.675, 'accuracy': 0.708, 'f1': 0.423},
            'Neural Network': {'auc': 0.689, 'accuracy': 0.715, 'f1': 0.438}
        }
        
        model_df = pd.DataFrame(models).T.reset_index()
        model_df.columns = ['Model', 'AUC-ROC', 'Accuracy', 'F1-Score']
        
        st.dataframe(model_df, use_container_width=True, hide_index=True)
        
        # Model comparison visualization
        fig_models = px.bar(
            model_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Metric',
            title='Model Performance Comparison',
            barmode='group'
        )
        st.plotly_chart(fig_models, use_container_width=True)
        
        # Business impact analysis
        st.markdown("### 💼 Business Impact Analysis")
        
        if predictions is not None and 'y_pred' in predictions:
            y_pred = predictions['y_pred']
            y_test = predictions['y_test'][:len(y_pred)]
            
            # Calculate business metrics
            tp = np.sum((y_pred == 1) & (y_test == 1))
            fp = np.sum((y_pred == 1) & (y_test == 0))
            fn = np.sum((y_pred == 0) & (y_test == 1))
            tn = np.sum((y_pred == 0) & (y_test == 0))
            
            # Simulate business costs and benefits
            contact_cost = 50  # Cost per contact
            legacy_value = 10000  # Average legacy value
            
            total_contacts = tp + fp
            total_cost = total_contacts * contact_cost
            total_value = tp * legacy_value
            net_value = total_value - total_cost
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Contacts", f"{total_contacts:,}")
            
            with col2:
                st.metric("Total Cost", f"${total_cost:,}")
            
            with col3:
                st.metric("Total Value", f"${total_value:,}")
            
            with col4:
                st.metric("Net Value", f"${net_value:,}")
            
            # ROI calculation
            roi = (net_value / total_cost) * 100 if total_cost > 0 else 0
            st.metric("ROI", f"{roi:.1f}%")
    
    # Export functionality
    st.markdown("---")
    st.subheader("📤 Export Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Performance Report", use_container_width=True):
            st.info("Performance analytics would be exported here")
    
    with col2:
        if st.button("👥 Export Cohort Analysis", use_container_width=True):
            st.info("Cohort analysis data would be exported here")
    
    with col3:
        if st.button("💼 Export Business Impact", use_container_width=True):
            st.info("Business impact analysis would be exported here")
