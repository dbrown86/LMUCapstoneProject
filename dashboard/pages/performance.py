"""
Performance page for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

# Optional Streamlit import
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

# Import modules
try:
    from dashboard.models.metrics import get_model_metrics, try_load_saved_metrics
except ImportError:
    # Fallbacks for testing
    def get_model_metrics(df):
        return {'auc': None, 'f1': None, 'baseline_auc': None, 'lift': None}
    def try_load_saved_metrics():
        return None

# Import chart wrapper (with fallback)
try:
    from dashboard.components.charts import plotly_chart_silent
except ImportError:
    # Fallback: use st.plotly_chart directly with config (filter kwargs)
    def plotly_chart_silent(fig, width='stretch', config=None, **kwargs):
        if config is None:
            config = {'displayModeBar': True, 'displaylogo': False}
        if STREAMLIT_AVAILABLE:
            # Filter to only recognized parameters to avoid deprecation warnings
            recognized = {'theme', 'key', 'on_select', 'selection_mode'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in recognized}
            return st.plotly_chart(fig, width=width, config=config, **filtered_kwargs)
        return None


def render(df: pd.DataFrame):
    """
    Render the performance page.
    
    Args:
        df: Dataframe with model predictions and outcomes
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    st.markdown('<p class="page-title">📈 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Comprehensive model evaluation metrics</p>', unsafe_allow_html=True)
    
    # Use the same metrics function as sidebar for consistency
    metrics = get_model_metrics(df)
    
    # Get threshold from saved metrics or default
    saved_meta = try_load_saved_metrics() or {}
    threshold = saved_meta.get('optimal_threshold', 0.5)
    metrics['optimal_threshold'] = threshold
    
    # CRITICAL: Use 2025 columns - prioritize Gave_Again_In_2025 and Will_Give_Again_Probability
    outcome_col = 'Gave_Again_In_2025' if 'Gave_Again_In_2025' in df.columns else ('Gave_Again_In_2024' if 'Gave_Again_In_2024' in df.columns else 'actual_gave')
    prob_col = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df.columns else 'predicted_prob'
    
    # Compute accuracy and precision/recall if not in saved metrics and we have data
    if metrics.get('accuracy') is None and outcome_col in df.columns and prob_col in df.columns:
        y_true_series = pd.to_numeric(df[outcome_col], errors='coerce')
        y_prob_series = pd.to_numeric(df[prob_col], errors='coerce')
        valid_mask = y_true_series.notna() & y_prob_series.notna()
        y_true = y_true_series.loc[valid_mask].astype(int).values
        y_prob = np.clip(y_prob_series.loc[valid_mask].astype(float).values, 0, 1)
        if y_prob.size and np.unique(y_true).size >= 2:
            y_pred = (y_prob >= float(threshold)).astype(int)
            if metrics.get('accuracy') is None:
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
            if metrics.get('precision') is None:
                metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            if metrics.get('recall') is None:
                metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    
    # Metric cards (AUC, F1, Accuracy)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auc_display = f"{metrics['auc']:.2%}" if metrics['auc'] is not None else "N/A"
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-left: none; min-height: 160px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div class="metric-label" style="color: white; margin-bottom: 10px;">AUC Score</div>
            <div class="metric-value" style="color: white; text-align: center;">{auc_display}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        f1_display = f"{metrics['f1']:.2%}" if metrics['f1'] is not None else "N/A"
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; border: none; border-left: none; min-height: 160px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div class="metric-label" style="color: white; margin-bottom: 10px;">F1 Score</div>
            <div class="metric-value" style="color: white; text-align: center;">{f1_display}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        acc_display = f"{metrics['accuracy']:.2%}" if metrics['accuracy'] is not None else "N/A"
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; border: none; border-left: none; min-height: 160px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div class="metric-label" style="color: white; margin-bottom: 10px;">Accuracy</div>
            <div class="metric-value" style="color: white; text-align: center;">{acc_display}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Only show ROC curve if metrics are available
    if metrics['auc'] is not None and metrics['baseline_auc'] is not None:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 📊 ROC Curve Analysis")
        
        # Generate ROC curve data from actual predictions (using 2025 columns)
        if outcome_col in df.columns and prob_col in df.columns:
            y_true_series = pd.to_numeric(df[outcome_col], errors='coerce')
            y_prob_series = pd.to_numeric(df[prob_col], errors='coerce')
            valid_mask = y_true_series.notna() & y_prob_series.notna()
            y_true = y_true_series.loc[valid_mask].astype(int)
            y_prob = np.clip(y_prob_series.loc[valid_mask].astype(float), 0, 1)

            # Need at least two classes to compute ROC
            if len(y_true) >= 2 and np.unique(y_true).size >= 2:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
            else:
                fpr, tpr = np.linspace(0, 1, 2), np.linspace(0, 1, 2)
                st.info("Insufficient class variety to compute ROC; showing placeholder line.")
            
            # Baseline (recency-based)
            if 'days_since_last' in df.columns:
                days_series = pd.to_numeric(df['days_since_last'], errors='coerce')
                base_mask = valid_mask & days_series.notna()
                y_true_base = y_true_series.loc[base_mask].astype(int)
                days_valid = days_series.loc[base_mask].astype(float)
                if len(y_true_base) >= 2 and np.unique(y_true_base).size >= 2:
                    baseline_pred = 1 / (1 + days_valid / 365)
                    fpr_baseline, tpr_baseline, _ = roc_curve(y_true_base, baseline_pred)
                else:
                    fpr_baseline = np.linspace(0, 1, 2)
                    tpr_baseline = fpr_baseline
            else:
                fpr_baseline = np.linspace(0, 1, 100)
                tpr_baseline = fpr_baseline
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f'Multimodal Fusion Model (AUC = {metrics["auc"]:.4f})', 
                line=dict(color='#2E86AB', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=fpr_baseline, y=tpr_baseline, mode='lines',
                name=f'Baseline (AUC = {metrics["baseline_auc"]:.4f})', 
                line=dict(color='#F18F01', width=2, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                name='Random (AUC = 0.50)', 
                line=dict(color='gray', dash='dot')
            ))
            
            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#e0e0e0'),
                yaxis=dict(showgrid=True, gridcolor='#e0e0e0')
            )
            plotly_chart_silent(fig, config={'displayModeBar': True, 'displaylogo': False})
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info(f"ROC curve requires '{outcome_col}' and '{prob_col}' columns in the dataset")

    # Precision-Recall Curve
    if metrics.get('precision') is not None and metrics.get('recall') is not None:
        st.markdown("### 📊 Precision-Recall Curve")
        try:
            if outcome_col in df.columns and prob_col in df.columns:
                y_true_series = pd.to_numeric(df[outcome_col], errors='coerce')
                y_prob_series = pd.to_numeric(df[prob_col], errors='coerce')
                valid_mask = y_true_series.notna() & y_prob_series.notna()
                y_true = y_true_series.loc[valid_mask].astype(int).values
                y_prob = np.clip(y_prob_series.loc[valid_mask].astype(float).values, 0, 1)
                
                if len(y_true) >= 2 and np.unique(y_true).size >= 2:
                    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
                    
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(
                        x=recall,
                        y=precision,
                        mode='lines',
                        name='Precision-Recall Curve',
                        line=dict(color='#2E86AB', width=3),
                        fill='tozeroy'
                    ))
                    fig_pr.add_hline(
                        y=metrics.get('precision', 0),
                        line_dash="dash",
                        line_color="gray",
                        annotation_text=f"Current Precision: {metrics.get('precision', 0):.2%}"
                    )
                    fig_pr.update_layout(
                        title='Precision-Recall Curve',
                        xaxis_title='Recall',
                        yaxis_title='Precision',
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    plotly_chart_silent(fig_pr, config={'displayModeBar': True, 'displaylogo': False})
                    st.caption("💡 **What this means**: This curve shows the trade-off between precision and recall at different thresholds. Higher area under the curve is better.")
        except Exception as e:
            st.warning(f"Could not render PR curve: {e}")

    # Confusion Matrix
    st.markdown("### 🎲 Confusion Matrix")
    try:
        if outcome_col in df.columns and prob_col in df.columns:
            y_true_series = pd.to_numeric(df[outcome_col], errors='coerce')
            y_prob_series = pd.to_numeric(df[prob_col], errors='coerce')
            valid_mask = y_true_series.notna() & y_prob_series.notna()
            y_true = y_true_series.loc[valid_mask].astype(int).values
            thresh_val = float(metrics.get('optimal_threshold', 0.5))
            st.caption(f"Threshold ({thresh_val:.2f})")
            y_pred = (np.clip(y_prob_series.loc[valid_mask].astype(float).values, 0, 1) >= thresh_val).astype(int)
            if y_true.size:
                cm = confusion_matrix(y_true, y_pred)
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted No', 'Predicted Yes'],
                    y=['Actually No', 'Actually Yes'],
                    text=cm,
                    texttemplate='%{text:,}',
                    textfont={"size": 20},
                    colorscale='Blues'
                ))
                fig_cm.update_layout(title='Confusion Matrix', height=400)
                plotly_chart_silent(fig_cm, config={'displayModeBar': True, 'displaylogo': False})
        else:
            st.info(f"Confusion matrix requires '{outcome_col}' and '{prob_col}' columns.")
    except Exception as e:
        st.warning(f"Could not render confusion matrix: {e}")
    fusion_precision = metrics.get('precision')
    fusion_recall = metrics.get('recall')
    fusion_specificity = metrics.get('specificity')
    baseline_precision = metrics.get('baseline_precision')
    baseline_recall = metrics.get('baseline_recall')
    baseline_specificity = metrics.get('baseline_specificity')
    if all(value is not None for value in [fusion_precision, fusion_recall, baseline_precision, baseline_recall]):
        st.markdown("### 🎯 Confusion Matrix Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="background: #fee; color: #111; padding: 20px; border-radius: 10px; border-left: 5px solid #e74c3c;">
                <h4 style="color: #c0392b; margin-top: 0;">Baseline Model Performance</h4>
                <ul style="line-height: 2.0;">
                    <li><strong>Correctly identifies {baseline_recall:.1%}</strong> of actual donors</li>
                    <li><strong>Correctly avoids {(baseline_specificity if baseline_specificity is not None else 0.426):.1%}</strong> of non-donors</li>
                    <li><strong>Precision:</strong> {baseline_precision:.1%} of predicted donors actually give</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background: #efe; color: #111; padding: 20px; border-radius: 10px; border-left: 5px solid #2ecc71;">
                <h4 style="color: #27ae60; margin-top: 0;">Fusion Model Performance</h4>
                <ul style="line-height: 2.0;">
                    <li><strong>Correctly identifies {fusion_recall:.1%}</strong> of actual donors</li>
                    <li><strong>Correctly avoids {(fusion_specificity if fusion_specificity is not None else 0.684):.1%}</strong> of non-donors</li>
                    <li><strong>Precision:</strong> {fusion_precision:.1%} of predicted donors actually give</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        recall_improvement = (fusion_recall - baseline_recall) / baseline_recall * 100 if baseline_recall else 0
        specificity_improvement = 0
        if baseline_specificity:
            specificity_improvement = (fusion_specificity - baseline_specificity) / baseline_specificity * 100 if fusion_specificity is not None else 0
        st.info(f"""
        **Key Insights:**
        - **{recall_improvement:.1f}% better at finding donors**: The Fusion model identifies {fusion_recall:.1%} vs {baseline_recall:.1%} with baseline
        - **{specificity_improvement:.1f}% better at avoiding wasted effort**: The Fusion model correctly avoids {(fusion_specificity if fusion_specificity is not None else 0.853):.1%} vs {(baseline_specificity if baseline_specificity is not None else 0.426):.1%} with baseline
        - **Higher precision means less waste**: {fusion_precision:.1%} of our Fusion predictions are correct vs {baseline_precision:.1%} with baseline
        """)
     
