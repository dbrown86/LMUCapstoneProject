"""
Model Comparison page for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
        return {'auc': None, 'f1': None, 'baseline_auc': None}
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
    Render the model comparison page.
    
    Args:
        df: Dataframe with model predictions and outcomes
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    st.markdown('<p class="page-title">🔬 Model Performance Comparison</p>', unsafe_allow_html=True)
    
    # Get actual metrics from saved training or computed from data
    actual_metrics = get_model_metrics(df)
    
    # Use actual Multimodal Fusion metrics - only use if actually available, don't use placeholders
    fusion_auc = actual_metrics.get('auc')  # Keep None if not available
    fusion_f1 = actual_metrics.get('f1')  # Keep None if not available
    fusion_precision = actual_metrics.get('precision')  # Keep None if not available
    fusion_recall = actual_metrics.get('recall')  # Keep None if not available
    fusion_specificity = actual_metrics.get('specificity')  # Keep None if not available
    
    # Try to compute fusion metrics from data if not in saved metrics
    if (fusion_f1 is None or fusion_precision is None or fusion_recall is None) and 'actual_gave' in df.columns and 'predicted_prob' in df.columns:
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
            saved_meta = try_load_saved_metrics() or {}
            threshold = saved_meta.get('optimal_threshold', 0.5)
            
            y_true_series = pd.to_numeric(df['actual_gave'], errors='coerce')
            y_prob_series = pd.to_numeric(df['predicted_prob'], errors='coerce')
            valid_mask = y_true_series.notna() & y_prob_series.notna()
            y_true = y_true_series.loc[valid_mask].astype(int).values
            y_prob = np.clip(y_prob_series.loc[valid_mask].astype(float).values, 0, 1)
            
            if y_prob.size and np.unique(y_true).size >= 2:
                y_pred = (y_prob >= float(threshold)).astype(int)
                if fusion_f1 is None:
                    fusion_f1 = f1_score(y_true, y_pred, zero_division=0)
                if fusion_precision is None:
                    fusion_precision = precision_score(y_true, y_pred, zero_division=0)
                if fusion_recall is None:
                    fusion_recall = recall_score(y_true, y_pred, zero_division=0)
        except Exception:
            pass
    
    # Use actual baseline metrics if available
    baseline_auc = actual_metrics.get('baseline_auc')  # Keep None if not available
    
    # Calculate baseline F1, precision, recall from data if available
    baseline_f1 = actual_metrics.get('baseline_f1')
    baseline_precision = actual_metrics.get('baseline_precision')
    baseline_recall = actual_metrics.get('baseline_recall')
    baseline_specificity = actual_metrics.get('baseline_specificity')
    
    # Recalculate if missing
    if (baseline_f1 is None or baseline_precision is None or baseline_recall is None) and 'days_since_last' in df.columns and 'actual_gave' in df.columns:
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
            y_true_series = pd.to_numeric(df['actual_gave'], errors='coerce')
            days_series = pd.to_numeric(df['days_since_last'], errors='coerce')
            mask = y_true_series.notna() & days_series.notna()
            y_true = y_true_series.loc[mask].astype(int).values
            days_valid = days_series.loc[mask].astype(float).values
            if y_true.size and np.unique(y_true).size >= 2:
                max_days = np.nanpercentile(days_valid, 95) if days_valid.size else np.nan
                if np.isfinite(max_days) and max_days > 0:
                    baseline_pred_proba = 1 - (np.clip(days_valid, 0, max_days) / max_days)
                    baseline_pred = (baseline_pred_proba >= 0.5).astype(int)
                    baseline_f1 = f1_score(y_true, baseline_pred, zero_division=0)
                    baseline_precision = precision_score(y_true, baseline_pred, zero_division=0)
                    baseline_recall = recall_score(y_true, baseline_pred, zero_division=0)
        except Exception:
            pass
    
    # 1. MODEL PERFORMANCE COMPARISON CHART
    st.markdown("### 📊 Performance Comparison: Baseline vs. Multimodal Fusion")
    
    # Primary comparison: Actual Baseline vs Actual Fusion
    # Only show if both AUC values are available
    if baseline_auc is None or fusion_auc is None:
        st.error("❌ **Cannot display comparison**: Missing baseline AUC or fusion AUC. Please ensure both models have been evaluated.")
        return
    
    # 3. KEY INSIGHTS CARDS (moved to top per request)
    st.markdown("### 💡 Key Insights")

    # Calculate actual performance gain vs baseline
    performance_gain = None
    if baseline_auc is not None and fusion_auc is not None and baseline_auc > 0:
        performance_gain = ((fusion_auc - baseline_auc) / baseline_auc) * 100

    # Calculate lift if available
    lift_display = actual_metrics.get('lift')
    lift_display = f"+{lift_display:.1%}" if lift_display is not None else "N/A"

    # Get actual fusion accuracy if available
    fusion_accuracy = actual_metrics.get('accuracy')

    col1, col2, col3 = st.columns(3)
    accuracy_improvement = None
    with col1:
        if performance_gain is not None:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #2ecc71;">
                <div class="metric-icon">🏆</div>
                <div class="metric-label">Performance Gain</div>
                <div class="metric-value" style="color: #2ecc71;">+{performance_gain:.1f}%</div>
                <div class="metric-delta" style="background: #d5f4e6; color: #27ae60;">
                    vs. Baseline
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Performance gain calculation requires baseline metrics")

    with col2:
        if lift_display != "N/A":
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #3498db;">
                <div class="metric-icon">📈</div>
                <div class="metric-label">Lift vs Baseline</div>
                <div class="metric-value" style="color: #3498db;">{lift_display}</div>
                <div class="metric-delta" style="background: #d6eaf8; color: #2874a6;">
                    AUC Improvement
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Lift calculation requires baseline AUC")

    with col3:
        if fusion_accuracy is not None:
            baseline_accuracy = None
            # Try to calculate baseline accuracy if we have the data
            if 'days_since_last' in df.columns and 'actual_gave' in df.columns:
                try:
                    from sklearn.metrics import accuracy_score
                    y_true_series = pd.to_numeric(df['actual_gave'], errors='coerce')
                    days_series = pd.to_numeric(df['days_since_last'], errors='coerce')
                    mask = y_true_series.notna() & days_series.notna()
                    y_true = y_true_series.loc[mask].astype(int).values
                    days_valid = days_series.loc[mask].astype(float).values
                    if y_true.size and np.unique(y_true).size >= 2:
                        max_days = np.nanpercentile(days_valid, 95) if days_valid.size else np.nan
                        if np.isfinite(max_days) and max_days > 0:
                            baseline_pred = ((1 - (np.clip(days_valid, 0, max_days) / max_days)) >= 0.5).astype(int)
                            baseline_accuracy = accuracy_score(y_true, baseline_pred)
                except Exception:
                    pass
            
            if baseline_accuracy is not None:
                accuracy_improvement = ((fusion_accuracy - baseline_accuracy) / baseline_accuracy * 100)
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #9b59b6;">
                    <div class="metric-icon">✅</div>
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value" style="color: #9b59b6;">{fusion_accuracy:.1%}</div>
                    <div class="metric-delta" style="background: #ebdef0; color: #7d3c98;">
                        +{accuracy_improvement:.1f}% vs Baseline
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #9b59b6;">
                    <div class="metric-icon">✅</div>
                    <div class="metric-label">Fusion Accuracy</div>
                    <div class="metric-value" style="color: #9b59b6;">{fusion_accuracy:.1%}</div>
                    <div class="metric-delta" style="background: #ebdef0; color: #7d3c98;">
                        Actual Result
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Accuracy metrics not available")

    col1, col2 = st.columns([2, 1])

    with col1:
        comparison_metrics = []
        baseline_series = []
        fusion_series = []
        improvements = {}

        def add_metric(label, baseline_value, fusion_value):
            if baseline_value is not None and fusion_value is not None:
                comparison_metrics.append(label)
                baseline_series.append(baseline_value)
                fusion_series.append(fusion_value)
                if baseline_value not in (0, None):
                    improvements[label] = (fusion_value - baseline_value) / baseline_value * 100
                else:
                    improvements[label] = None

        add_metric('AUC', baseline_auc, fusion_auc)
        add_metric('F1', baseline_f1, fusion_f1)
        add_metric('Precision', baseline_precision, fusion_precision)
        add_metric('Recall', baseline_recall, fusion_recall)
        add_metric('Specificity', baseline_specificity, fusion_specificity)

        if not comparison_metrics:
            st.error("❌ **Cannot display comparison**: Missing overlapping baseline/fusion metrics.")
        else:
            fig_actual = go.Figure()
            fig_actual.add_trace(go.Bar(
                name='Recency Baseline',
                x=comparison_metrics,
                y=baseline_series,
                marker_color='#e74c3c',
                text=[f"{value:.2%}" for value in baseline_series],
                textposition='outside',
                hovertemplate='<b>%{x} — Baseline</b><br>%{y:.2%}<extra></extra>'
            ))
            text_positions = []
            fusion_text = []
            for label, value in zip(comparison_metrics, fusion_series):
                fusion_text.append(f"{value:.2%}")
                if label == 'AUC':
                    text_positions.append('outside')
                else:
                    text_positions.append('outside')
            fig_actual.add_trace(go.Bar(
                name='Multimodal Fusion',
                x=comparison_metrics,
                y=fusion_series,
                marker_color='#2ecc71',
                text=fusion_text,
                textposition=text_positions,
                hovertemplate='<b>%{x} — Fusion</b><br>%{y:.2%}<extra></extra>'
            ))
            max_value = max(baseline_series + fusion_series) if (baseline_series or fusion_series) else 0
            y_padding = max(0.02, max_value * 0.15)

            fig_actual.update_layout(
                title='Actual Model Performance: Baseline vs. Fusion',
                xaxis_title='Metric',
                yaxis_title='Score',
                height=420,
                barmode='group',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#e0e0e0', tickformat='.0%', range=[0, min(1, max_value + y_padding)]),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
            )

            plotly_chart_silent(fig_actual, width='stretch', config={'displayModeBar': True, 'displaylogo': False})
            improvement = improvements.get('AUC')
    
    with col2:
        st.markdown("### 📈 Key Metrics")
        if fusion_auc is not None and improvement is not None:
            st.metric("Fusion AUC", f"{fusion_auc:.2%}", delta=f"+{improvement:.1f}%")

        if fusion_accuracy is not None and accuracy_improvement is not None:
            st.metric("Fusion Accuracy", f"{fusion_accuracy:.2%}", delta=f"+{accuracy_improvement:.1f}%")

        if fusion_specificity is not None and baseline_specificity not in (None, 0):
            specificity_delta = (fusion_specificity - baseline_specificity) / baseline_specificity * 100
            st.metric("Fusion Specificity", f"{fusion_specificity:.2%}", delta=f"+{specificity_delta:.1f}%")

    # 1b. BEFORE/AFTER SCENARIOS
    st.markdown("### 📊 Real-World Impact: Before & After Scenarios")
    
    if 'actual_gave' in df.columns and 'predicted_prob' in df.columns:
        # Calculate actual response rates
        baseline_response_rate = df['actual_gave'].mean() if 'actual_gave' in df.columns else 0.17
        saved_meta = try_load_saved_metrics() or {}
        threshold = saved_meta.get('optimal_threshold', 0.5)
        
        # High probability group response rate
        high_prob_donors = df[df['predicted_prob'] >= threshold]
        fusion_response_rate = high_prob_donors['actual_gave'].mean() if len(high_prob_donors) > 0 else baseline_response_rate * 1.5
        
        # Scenario: Contact 10,000 donors
        scenario_contacts = 10000
        baseline_responses = int(scenario_contacts * baseline_response_rate)
        fusion_responses = int(scenario_contacts * fusion_response_rate)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: #fee; color: #111; padding: 20px; border-radius: 10px; border-left: 5px solid #e74c3c;">
                <h4 style="color: #c0392b; margin-top: 0;">❌ Old Way (Baseline)</h4>
                <p style="font-size: 16px; line-height: 1.8;">
                    <strong>Contact 10,000 donors</strong> → <strong style="color: #e74c3c;">{:,} respond</strong> ({:.1%})
                </p>
                <p style="font-size: 14px; color: #666;">
                    Traditional approach using recency only
                </p>
            </div>
            """.format(baseline_responses, baseline_response_rate), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #efe; color: #111; padding: 20px; border-radius: 10px; border-left: 5px solid #2ecc71;">
                <h4 style="color: #27ae60; margin-top: 0;">✅ New Way (Fusion Model)</h4>
                <p style="font-size: 16px; line-height: 1.8;">
                    <strong>Contact 10,000 targeted donors</strong> → <strong style="color: #2ecc71;">{:,} respond</strong> ({:.1%})
                </p>
                <p style="font-size: 14px; color: #666;">
                    AI-powered targeting using multiple data sources
                </p>
            </div>
            """.format(fusion_responses, fusion_response_rate), unsafe_allow_html=True)
        
        improvement_factor = (fusion_response_rate / baseline_response_rate) if baseline_response_rate > 0 else 1
        additional_responses = fusion_responses - baseline_responses
        
        st.success(f"""
        **Result**: {improvement_factor:.1f}x response rate improvement! 
        Same outreach effort, but **{additional_responses:,} additional responses** ({fusion_responses:,} vs {baseline_responses:,}).
        """)
    
    # 2. MULTI-METRIC PERFORMANCE COMPARISON
    st.markdown("### 📈 Multi-Metric Performance Comparison")
    
    # Get specificity from actual_metrics
    baseline_specificity = actual_metrics.get('baseline_specificity')
    fusion_specificity = actual_metrics.get('specificity')
    
    # Only show metrics that we have actual data for
    metrics_to_show = []
    baseline_values = []
    fusion_values = []
    
    if baseline_auc is not None and fusion_auc is not None:
        metrics_to_show.append('AUC')
        baseline_values.append(baseline_auc)
        fusion_values.append(fusion_auc)
    
    if baseline_f1 is not None and fusion_f1 is not None:
        metrics_to_show.append('F1')
        baseline_values.append(baseline_f1)
        fusion_values.append(fusion_f1)
    
    if baseline_precision is not None and fusion_precision is not None:
        metrics_to_show.append('Precision')
        baseline_values.append(baseline_precision)
        fusion_values.append(fusion_precision)
    
    if baseline_recall is not None and fusion_recall is not None:
        metrics_to_show.append('Recall')
        baseline_values.append(baseline_recall)
        fusion_values.append(fusion_recall)
    
    if baseline_specificity is not None and fusion_specificity is not None:
        metrics_to_show.append('Specificity')
        baseline_values.append(baseline_specificity)
        fusion_values.append(fusion_specificity)
    
    # Error check
    if len(metrics_to_show) == 0:
        st.error("❌ **No metrics available for radar chart!** Both baseline and fusion values are required for each metric.")
    
    if len(metrics_to_show) > 0 and len(baseline_values) != len(fusion_values) or len(baseline_values) != len(metrics_to_show):
        st.error(f"⚠️ **Data mismatch**: Baseline values: {len(baseline_values)}, Fusion values: {len(fusion_values)}, Metrics: {len(metrics_to_show)}")
    
    # Only create chart if we have at least one metric and both models have values
    if len(metrics_to_show) > 0 and len(baseline_values) == len(fusion_values) == len(metrics_to_show):
        # Verify all values are valid numbers
        if all(isinstance(v, (int, float)) and not np.isnan(v) for v in baseline_values + fusion_values):
            # Create radar chart with actual data only
            fig_radar = go.Figure()
            colors_radar = ['#e74c3c', '#2ecc71']
            
            # Ensure we have the same order for both traces
            baseline_r = list(baseline_values)
            fusion_r = list(fusion_values)
            theta_labels = list(metrics_to_show)
            
            # Create hover text arrays
            baseline_hover = [f"<b>{metric}</b><br>Recency Baseline: {val:.2%}" 
                             for metric, val in zip(theta_labels, baseline_r)]
            fusion_hover = [f"<b>{metric}</b><br>Multimodal Fusion: {val:.2%}" 
                           for metric, val in zip(theta_labels, fusion_r)]
            
            # Add fusion trace first (rendered behind baseline)
            fig_radar.add_trace(go.Scatterpolar(
                r=fusion_r,
                theta=theta_labels,
                fill='toself',
                name='Multimodal Fusion',
                line_color=colors_radar[1],
                fillcolor=colors_radar[1],
                opacity=0.4,
                line=dict(width=3),
                marker=dict(size=10, symbol='circle', line=dict(width=2, color='white')),
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=fusion_hover
            ))
            
            # Add baseline trace second (rendered on top)
            fig_radar.add_trace(go.Scatterpolar(
                r=baseline_r,
                theta=theta_labels,
                fill='toself',
                name='Recency Baseline',
                line_color=colors_radar[0],
                fillcolor=colors_radar[0],
                opacity=0.6,
                line=dict(width=3.5),
                marker=dict(size=12, symbol='circle', line=dict(width=2, color='white')),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=baseline_hover
            ))
            label_radius = max(max(baseline_r, default=0), max(fusion_r, default=0), 1.0) + 0.02
            fig_radar.add_trace(go.Scatterpolar(
                r=[label_radius] * len(theta_labels),
                theta=theta_labels,
                mode='text',
                text=[f"<b>{label}</b>" for label in theta_labels],
                textposition='middle center',
                textfont=dict(color='white', size=13, family='Arial'),
                hoverinfo='skip',
                showlegend=False
            ))
        else:
            st.error("❌ **Invalid data values detected** - Some metrics contain NaN or invalid numbers.")
            fig_radar = None
    else:
        st.warning(f"⚠️ **Insufficient data for radar chart**: Baseline AUC: {baseline_auc is not None}, Fusion AUC: {fusion_auc is not None}, Metrics available: {len(metrics_to_show)}, Baseline values: {len(baseline_values)}, Fusion values: {len(fusion_values)}")
        fig_radar = None
    
    # Only display chart if we have valid data
    if fig_radar is not None:
        # Set dynamic range based on actual values
        all_values = baseline_values + fusion_values
        min_val = min(all_values) - 0.05 if all_values else 0.7
        max_val = max(all_values) + 0.05 if all_values else 1.0
        
        # Ensure range is reasonable
        if min_val < 0.4:
            min_val = 0.4
        if max_val > 1.0:
            max_val = 1.0
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[min_val, max_val],
                    showticklabels=True,
                    tickmode='linear',
                    tick0=min_val,
                    dtick=(max_val - min_val) / 5,
                    tickfont=dict(color='black', size=12)
                ),
                angularaxis=dict(
                    rotation=90,
                    direction="counterclockwise",
                    tickfont=dict(color='black')
                )
            ),
            title='Multi-Metric Performance: Baseline vs. Fusion',
            height=600,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest'
        )
        plotly_chart_silent(fig_radar, config={'displayModeBar': True, 'displaylogo': False})
        st.caption(
            "💡 **How to read**: If the green polygon extends farther than the red shape on a given axis, the fusion model beats the baseline on that metric.",
            unsafe_allow_html=True
        )
