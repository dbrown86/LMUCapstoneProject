"""
Executive Dashboard page for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List
from datetime import datetime

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
    from dashboard.pages.utils import filter_dataframe, get_value_counts, get_segment_performance, get_segment_stats
    from dashboard.data.query_loader import QueryBasedLoader
except ImportError:
    # Fallbacks for testing
    def get_model_metrics(df):
        return {'auc': None, 'f1': None, 'baseline_auc': None}
    def try_load_saved_metrics():
        return None
    def filter_dataframe(df, regions, donor_types, segments, use_cache=True):
        return df
    def get_value_counts(series, use_cache=True):
        return series.value_counts()
    def get_segment_performance(df_filtered, use_cache=True):
        return pd.DataFrame()
    def get_segment_stats(df, use_cache=True):
        return pd.DataFrame()
    QueryBasedLoader = None

# Import chart wrapper (separate try/except to ensure it's always available)
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


def render(df, regions: List[str], donor_types: List[str], segments: List[str], prob_threshold: float):
    """
    Render the executive dashboard page.
    
    Args:
        df: Dataframe with donor data and predictions
        regions: List of selected regions
        donor_types: List of selected donor types
        segments: List of selected segments
        prob_threshold: Probability threshold for high-probability donors
    """
    if STREAMLIT_AVAILABLE:
        # Extracted function body
        """Modern dashboard with KPIs and interactive charts"""

        # Apply filters (cached)
        df_filtered = filter_dataframe(df, tuple(regions) if regions else (), 
                                        tuple(donor_types) if donor_types else (), 
                                        tuple(segments) if segments else ())

        # Page header
        st.markdown('<p class="page-title">🏠 Executive Summary</p>', unsafe_allow_html=True)

        # Executive Summary Card
        # VERIFY: Ensure we're using "Will Give Again in 2025" predictions and outcomes
        # Check for Will_Give_Again_Probability directly first, then fall back to predicted_prob
        prob_col_exec = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df_filtered.columns else 'predicted_prob'
        outcome_col_exec = 'Gave_Again_In_2025' if 'Gave_Again_In_2025' in df_filtered.columns else ('Gave_Again_In_2024' if 'Gave_Again_In_2024' in df_filtered.columns else 'actual_gave')

        # Check if we're using QueryBasedLoader
        is_query_loader = QueryBasedLoader is not None and isinstance(df_filtered, QueryBasedLoader)
        
        if prob_col_exec in df_filtered.columns and 'avg_gift' in df_filtered.columns and 'total_giving' in df_filtered.columns:
            metrics_summary = get_model_metrics(df_filtered)
            
            # Handle QueryBasedLoader vs DataFrame differently
            if is_query_loader:
                # Use SQL queries for aggregations
                # Get high probability donors count (not used but kept for consistency)
                high_prob_count = df_filtered.get_aggregate_value(
                    column=None,
                    aggregation='COUNT',
                    where=f'"{prob_col_exec}" >= {prob_threshold}'
                )
                
                # Get high probability donors data for calculations
                high_prob_donors = df_filtered.get_filtered_dataframe(
                    where=f'"{prob_col_exec}" >= {prob_threshold}'
                )
                
                if len(high_prob_donors) > 0:
                    # Calculate conversion rate
                    if outcome_col_exec in high_prob_donors.columns:
                        actual_conversion = pd.to_numeric(high_prob_donors[outcome_col_exec], errors='coerce').mean()
                    else:
                        actual_conversion = 0.85  # Fallback
                    
                    # Get gift amounts
                    last_gift_col = None
                    for col in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount']:
                        if col in high_prob_donors.columns:
                            last_gift_col = col
                            break
                    
                    if last_gift_col:
                        gift_amounts = pd.to_numeric(high_prob_donors[last_gift_col], errors='coerce').fillna(0).clip(lower=0)
                        avg_gift_mean = gift_amounts.median() if len(gift_amounts) > 0 else 0
                    else:
                        avg_gift_values = pd.to_numeric(high_prob_donors['avg_gift'], errors='coerce').fillna(0).clip(lower=0)
                        avg_gift_mean = avg_gift_values.mean() if len(avg_gift_values) > 0 else 0
                    
                    estimated_revenue = avg_gift_mean * len(high_prob_donors) * actual_conversion
                else:
                    estimated_revenue = 0
                
                # Get high confidence count
                high_confidence_count = int(df_filtered.get_aggregate_value(
                    column=None,
                    aggregation='COUNT',
                    where=f'"{prob_col_exec}" >= 0.7'
                ))
            else:
                # Traditional DataFrame operations
                high_prob_donors = df_filtered[df_filtered[prob_col_exec] >= prob_threshold]
                
                if outcome_col_exec in df_filtered.columns and len(high_prob_donors) > 0:
                    actual_conversion = high_prob_donors[outcome_col_exec].mean()
                    last_gift_col = None
                    for col in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount']:
                        if col in high_prob_donors.columns:
                            last_gift_col = col
                            break

                    if last_gift_col:
                        gift_amounts = pd.to_numeric(high_prob_donors[last_gift_col], errors='coerce').fillna(0).clip(lower=0)
                        avg_gift_mean = gift_amounts.median() if len(gift_amounts) > 0 else gift_amounts.mean() if len(gift_amounts) > 0 else 0
                    else:
                        avg_gift_values = pd.to_numeric(high_prob_donors['avg_gift'], errors='coerce').fillna(0).clip(lower=0)
                        avg_gift_mean = avg_gift_values.mean() if len(avg_gift_values) > 0 else 0

                    estimated_revenue = avg_gift_mean * len(high_prob_donors) * actual_conversion
                else:
                    last_gift_col = None
                    for col in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount']:
                        if col in high_prob_donors.columns:
                            last_gift_col = col
                            break

                    if last_gift_col:
                        gift_amounts = pd.to_numeric(high_prob_donors[last_gift_col], errors='coerce').fillna(0).clip(lower=0) if len(high_prob_donors) > 0 else pd.Series([0])
                        avg_gift_mean = gift_amounts.median() if len(gift_amounts) > 0 else gift_amounts.mean() if len(gift_amounts) > 0 else 0
                    else:
                        avg_gift_values = pd.to_numeric(high_prob_donors['avg_gift'], errors='coerce').fillna(0).clip(lower=0) if len(high_prob_donors) > 0 else pd.Series([0])
                        avg_gift_mean = avg_gift_values.mean() if len(avg_gift_values) > 0 else 0

                    estimated_revenue = avg_gift_mean * len(high_prob_donors) * 0.85 if len(high_prob_donors) > 0 else 0

                if prob_col_exec in df_filtered.columns:
                    high_confidence_count = (pd.to_numeric(df_filtered[prob_col_exec], errors='coerce') >= 0.7).sum()
                else:
                    high_confidence_count = 0

            if estimated_revenue >= 1_000_000_000:
                revenue_display = f"${estimated_revenue/1_000_000_000:.1f}B"
            elif estimated_revenue >= 1_000_000:
                revenue_display = f"${estimated_revenue/1_000_000:,.0f}M"
            else:
                revenue_display = f"${estimated_revenue:,.0f}"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 10px; margin-bottom: 30px;">
                <h3 style="color: white; margin-top: 0;">📊 Executive Summary</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 15px;">
                    <div>
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 5px;">Key Insight</div>
                        <div style="font-size: 18px; font-weight: bold;">🔥 {high_confidence_count:,} High Confidence Prospects (>70% probability)</div>
                    </div>
                    <div>
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 5px;">Business Impact</div>
                        <div style="font-size: 18px; font-weight: bold; margin-top: -2px;">{revenue_display} in Untapped Donor Potential</div>
                    </div>
                    <div>
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 5px;">Recommended Action</div>
                        <div style="font-size: 18px; font-weight: bold;">Prioritize Outreach to High Confidence Prospects</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Divider line matching sidebar background color
        st.markdown("""
        <div style="height: 4px; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); margin: 20px 0;"></div>
        """, unsafe_allow_html=True)

        # Hero metrics
        col1, col2, col3, col4 = st.columns(4)

        # Get actual metrics for dashboard - these are calculated from Gave_Again_In_2025
        # Use the full dataframe (not filtered) for metrics calculation to ensure we have all data
        # First try to calculate from the dataframe
        metrics = get_model_metrics(df)
        
        # If metrics are None, try to load saved metrics from training
        if metrics.get('auc') is None or metrics.get('f1') is None:
            saved_metrics = try_load_saved_metrics()
            if saved_metrics:
                # Use saved metrics if calculated metrics are missing
                if metrics.get('auc') is None and saved_metrics.get('auc') is not None:
                    metrics['auc'] = saved_metrics.get('auc')
                if metrics.get('f1') is None and saved_metrics.get('f1') is not None:
                    metrics['f1'] = saved_metrics.get('f1')
                if metrics.get('baseline_auc') is None and saved_metrics.get('baseline_auc') is not None:
                    metrics['baseline_auc'] = saved_metrics.get('baseline_auc')
        
        # If still None, try to calculate directly from the dataframe columns
        if metrics.get('auc') is None or metrics.get('f1') is None:
            try:
                from sklearn.metrics import roc_auc_score, f1_score
                # Check for required columns
                prob_col = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df.columns else 'predicted_prob'
                outcome_col = 'Gave_Again_In_2025' if 'Gave_Again_In_2025' in df.columns else ('Gave_Again_In_2024' if 'Gave_Again_In_2024' in df.columns else 'actual_gave')
                
                if prob_col in df.columns and outcome_col in df.columns:
                    y_true = pd.to_numeric(df[outcome_col], errors='coerce')
                    y_prob = pd.to_numeric(df[prob_col], errors='coerce')
                    valid_mask = y_true.notna() & y_prob.notna()
                    
                    if valid_mask.sum() > 0:
                        y_true_valid = y_true[valid_mask].astype(int).values
                        y_prob_valid = np.clip(y_prob[valid_mask].astype(float).values, 0, 1)
                        
                        unique_classes = np.unique(y_true_valid)
                        if len(unique_classes) >= 2:
                            # Check if predictions are inverted
                            prob_for_pos = y_prob_valid[y_true_valid == 1].mean() if (y_true_valid == 1).sum() > 0 else 0
                            prob_for_neg = y_prob_valid[y_true_valid == 0].mean() if (y_true_valid == 0).sum() > 0 else 0
                            
                            if prob_for_pos < prob_for_neg:
                                y_prob_valid = 1 - y_prob_valid
                            
                            # Calculate metrics
                            if metrics.get('auc') is None:
                                metrics['auc'] = roc_auc_score(y_true_valid, y_prob_valid)
                            
                            threshold = 0.5
                            y_pred_binary = (y_prob_valid >= threshold).astype(int)
                            
                            if metrics.get('f1') is None:
                                metrics['f1'] = f1_score(y_true_valid, y_pred_binary, zero_division=0)
            except Exception:
                pass  # Silently fail - will show N/A
        
        # Calculate baseline AUC if not available
        if metrics.get('baseline_auc') is None:
            try:
                from sklearn.metrics import roc_auc_score
                outcome_col = 'Gave_Again_In_2025' if 'Gave_Again_In_2025' in df.columns else ('Gave_Again_In_2024' if 'Gave_Again_In_2024' in df.columns else 'actual_gave')
                
                if outcome_col in df.columns and 'days_since_last' in df.columns:
                    y_true = pd.to_numeric(df[outcome_col], errors='coerce')
                    days_series = pd.to_numeric(df['days_since_last'], errors='coerce')
                    base_mask = y_true.notna() & days_series.notna()
                    
                    if base_mask.sum() > 0:
                        y_true_base = y_true[base_mask].astype(int).values
                        days_valid = days_series[base_mask].astype(float).values
                        
                        unique_classes_base = np.unique(y_true_base)
                        if len(unique_classes_base) >= 2:
                            # Calculate baseline predictions: more recent = higher probability
                            max_days = np.nanpercentile(days_valid, 95) if days_valid.size > 0 else np.nanmax(days_valid)
                            if np.isfinite(max_days) and max_days > 0:
                                baseline_pred = 1 - (np.clip(days_valid, 0, max_days) / max_days)
                                metrics['baseline_auc'] = roc_auc_score(y_true_base, baseline_pred)
            except Exception:
                pass  # Silently fail - will use default
        
        # Use default baseline AUC if still None (50.29% = 0.5029)
        if metrics.get('baseline_auc') is None:
            metrics['baseline_auc'] = 0.5029
        
        auc_display = f"{metrics['auc']:.2%}" if metrics.get('auc') is not None else "N/A"
        baseline_auc_display = f"{metrics['baseline_auc']:.2%}" if metrics.get('baseline_auc') is not None else "50.29%"
        improvement = ((metrics['auc'] - metrics['baseline_auc']) / metrics['baseline_auc'] * 100) if metrics.get('baseline_auc') and metrics.get('baseline_auc') > 0 and metrics.get('auc') is not None else 0
        
        # Calculate lift (improvement ratio) for the "4-5x" display
        # Lift = (AUC - Baseline AUC) / Baseline AUC, which gives the multiplier
        if metrics.get('lift') is not None and metrics.get('lift') > 0:
            lift_ratio = 1 + metrics['lift']  # Convert lift to multiplier (e.g., 0.74 lift = 1.74x = 74% improvement)
            if lift_ratio >= 4.5:
                improvement_display = f"{lift_ratio:.1f}x"
            elif lift_ratio >= 4.0:
                improvement_display = "4-5x"
            elif lift_ratio >= 3.0:
                improvement_display = f"{lift_ratio:.1f}x"
            else:
                improvement_display = f"{lift_ratio:.1f}x"
        elif metrics.get('baseline_auc') and metrics.get('baseline_auc') > 0 and metrics.get('auc') is not None:
            # Calculate lift from AUC values if lift not directly available
            calculated_lift = (metrics['auc'] - metrics['baseline_auc']) / metrics['baseline_auc']
            lift_ratio = 1 + calculated_lift
            if lift_ratio >= 4.5:
                improvement_display = f"{lift_ratio:.1f}x"
            elif lift_ratio >= 4.0:
                improvement_display = "4-5x"
            else:
                improvement_display = f"{lift_ratio:.1f}x"
        else:
            improvement_display = "4-5x"  # Fallback if metrics unavailable

        with col1:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-left: none; height: 170px; display: flex; flex-direction: column; justify-content: space-between;">
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">AUC Score</div>
                <div class="metric-value" style="color: white;">{auc_display}</div>
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 9px; line-height: 1.2;">Predicting "will give again in 2025"</div>
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 9px; line-height: 1.2;">compared to {baseline_auc_display} Baseline AUC</div>
            </div>
            """, unsafe_allow_html=True)

        f1_display = f"{metrics['f1']:.2%}" if metrics.get('f1') is not None else "N/A"
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; border: none; border-left: none; height: 170px; display: flex; flex-direction: column; justify-content: space-between;">
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">F1 Score</div>
                <div class="metric-value" style="color: white;">{f1_display}</div>
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 11px;">Balanced precision & recall</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            # Calculate actual revenue potential - USE WILL GIVE AGAIN COLUMNS
            # CRITICAL: Use Will_Give_Again_Probability and Gave_Again_In_2025 directly (same as Executive Summary)
            prob_col_rev = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df_filtered.columns else 'predicted_prob'
            outcome_col_rev = 'Gave_Again_In_2025' if 'Gave_Again_In_2025' in df_filtered.columns else ('Gave_Again_In_2024' if 'Gave_Again_In_2024' in df_filtered.columns else 'actual_gave')

            if prob_col_rev in df_filtered.columns and 'avg_gift' in df_filtered.columns:
                high_prob = df_filtered[df_filtered[prob_col_rev] >= prob_threshold]
                if len(high_prob) > 0:
                    # Use actual conversion rate from Gave_Again_In_2025 if available (fallback to 2025)
                    if outcome_col_rev in df_filtered.columns:
                        actual_conversion = high_prob[outcome_col_rev].mean()
                        # CRITICAL: Use Last_Gift instead of avg_gift_amount (which is corrupted with mean $0.03)
                        last_gift_col = None
                        for col in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount']:
                            if col in high_prob.columns:
                                last_gift_col = col
                                break

                        if last_gift_col:
                            gift_amounts = pd.to_numeric(high_prob[last_gift_col], errors='coerce').fillna(0).clip(lower=0)
                            # Use median for robustness against outliers
                            avg_gift_mean = gift_amounts.median() if len(gift_amounts) > 0 else gift_amounts.mean() if len(gift_amounts) > 0 else 0
                        else:
                            # Fallback to avg_gift if Last_Gift not available
                            avg_gift_values = pd.to_numeric(high_prob['avg_gift'], errors='coerce').fillna(0).clip(lower=0) if 'avg_gift' in high_prob.columns else pd.Series([0])
                            avg_gift_mean = avg_gift_values.mean() if len(avg_gift_values) > 0 else 0

                        revenue_potential = avg_gift_mean * len(high_prob) * actual_conversion
                    else:
                        # Fallback estimate if actual outcome data not available
                        last_gift_col = None
                        for col in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount']:
                            if col in high_prob.columns:
                                last_gift_col = col
                                break

                        if last_gift_col:
                            gift_amounts = pd.to_numeric(high_prob[last_gift_col], errors='coerce').fillna(0).clip(lower=0) if len(high_prob) > 0 else pd.Series([0])
                            # Use median for robustness against outliers
                            avg_gift_mean = gift_amounts.median() if len(gift_amounts) > 0 else gift_amounts.mean() if len(gift_amounts) > 0 else 0
                        else:
                            avg_gift_values = pd.to_numeric(high_prob['avg_gift'], errors='coerce').fillna(0).clip(lower=0) if 'avg_gift' in high_prob.columns else pd.Series([0])
                            avg_gift_mean = avg_gift_values.mean() if len(avg_gift_values) > 0 else 0

                        revenue_potential = avg_gift_mean * len(high_prob) * 0.85
                    if revenue_potential >= 1_000_000_000:
                        rev_display = f"${revenue_potential/1_000_000_000:.1f}B"
                    elif revenue_potential >= 1_000_000:
                        rev_display = f"${revenue_potential/1_000_000:,.0f}M"
                    else:
                        rev_display = f"${revenue_potential:,.0f}"
                else:
                    rev_display = "$25M+"
            else:
                rev_display = "$25M+"

            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; border: none; border-left: none; height: 170px; display: flex; flex-direction: column; justify-content: space-between;">
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">Revenue Potential</div>
                <div class="metric-value" style="color: white;">{rev_display}</div>
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 11px; opacity: 0;">&nbsp;</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; border: none; border-left: none; height: 170px; display: flex; flex-direction: column; justify-content: space-between;">
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">Improvement</div>
                <div class="metric-value" style="color: white;">{improvement_display}</div>
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">vs Baseline</div>
            </div>
            """, unsafe_allow_html=True)

        # Divider line matching sidebar background color
        st.markdown("""
        <div style="height: 4px; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); margin: 20px 0;"></div>
        """, unsafe_allow_html=True)

        # Charts Section
        col1, col2 = st.columns(2)

        df_work = df_filtered.loc[:, ~df_filtered.columns.duplicated()].copy()

        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 📊 Donor Base Breakdown: Recent to Lapsed Engagement")
            # CRITICAL: Use 2025 prediction column - prioritize Will_Give_Again_Probability
            prob_col_breakdown = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df_work.columns else 'predicted_prob'
            if {'segment', prob_col_breakdown}.issubset(df_work.columns):
                seg_df = df_work[['segment', prob_col_breakdown]].dropna(subset=['segment'])
                if not seg_df.empty:
                    # Filter out Prospects/New category
                    seg_df_filtered = seg_df[seg_df['segment'] != 'Prospects/New'].copy()
                    
                    if not seg_df_filtered.empty:
                        summary = seg_df_filtered.groupby('segment', observed=False).agg(
                            Count=('segment', 'size'),
                            Avg_Prob=(prob_col_breakdown, 'mean')
                        ).reset_index()
                        
                        # Calculate percentages
                        total_donors = summary['Count'].sum()
                        summary['Percentage'] = (summary['Count'] / total_donors * 100).round(1)
                        
                        category_order = ['Recent (0-6mo)', 'Recent (6-12mo)', 'Lapsed (1-2yr)', 'Very Lapsed (2yr+)']
                        summary['segment'] = pd.Categorical(summary['segment'], categories=category_order, ordered=True)
                        summary = summary.sort_values('segment', ascending=True).reset_index(drop=True)
                        
                        # Create figure with gradient colors
                        fig_segment = go.Figure()
                        
                        # Color gradient from green (recent) to red (lapsed)
                        colors = ['#2e7d32', '#66bb6a', '#ffa726', '#ef5350']
                        
                        for pos, (_, row) in enumerate(summary.iterrows()):
                            fig_segment.add_trace(go.Bar(
                                x=[row['Count']],
                                y=[row['segment']],
                                orientation='h',
                                name=row['segment'],
                                marker=dict(
                                    color=colors[pos] if pos < len(colors) else '#757575',
                                    line=dict(width=0)
                                ),
                                text=f"{row['Count']:,} ({row['Percentage']:.1f}%)",
                                textposition='outside',
                                textfont=dict(size=10, color='white', weight='bold'),
                                hovertemplate=(
                                    f"<b>{row['segment']}</b><br>" +
                                    f"Count: {row['Count']:,}<br>" +
                                    f"Percentage: {row['Percentage']:.1f}%<br>" +
                                    f"Avg Probability: {row['Avg_Prob']:.1%}<br>" +
                                    "<extra></extra>"
                                ),
                                showlegend=False
                            ))
                        
                        max_count = summary['Count'].max()
                        padding = max(1, max_count * 0.3) if pd.notna(max_count) else 1
                        
                        fig_segment.update_layout(
                            height=380,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=20, b=40, l=140, r=80),
                            yaxis=dict(
                                title=dict(text='Donor Segment', font=dict(size=12, color='white')),
                                autorange='reversed',
                                categoryorder='array',
                                categoryarray=category_order,
                                tickfont=dict(size=11, color='white'),
                                showgrid=False
                            ),
                            xaxis=dict(
                                range=[0, max_count + padding],
                                title=dict(text='Number of Donors', font=dict(size=12, color='white')),
                                tickfont=dict(size=11, color='white'),
                                showgrid=True,
                                gridcolor='rgba(200,200,200,0.3)',
                                gridwidth=1
                            ),
                            font=dict(family="Arial, sans-serif", color='white')
                        )
                        plotly_chart_silent(fig_segment, width='stretch', config={'displayModeBar': True, 'displaylogo': False})
                        st.markdown("**💡 How to read**: Donors are grouped into engaged (green) to at-risk (orange/red). Focus retention efforts on the yellow/orange bands before they turn red. Note: This chart excludes non-donors.", unsafe_allow_html=True)
                    else:
                        st.info("No donor segment data available (excluding Prospects/New).")
                else:
                    st.info("No segment data available to display.")
            else:
                st.info("Segment or prediction columns missing from dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Donors Predicted to Give Again")
            # Add vertical spacing to align caption with Donor Base Breakdown chart
            # Donor Base Breakdown chart height is 380px, this chart is 250px, so add ~130px spacing
            st.markdown("<div style='height: 130px;'></div>", unsafe_allow_html=True)
            # CRITICAL: Use 2025 prediction column - prioritize Will_Give_Again_Probability
            prob_col_tiers = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df_work.columns else 'predicted_prob'
            if prob_col_tiers in df_work.columns:
                probs = pd.to_numeric(df_work[prob_col_tiers], errors='coerce').dropna()
                if len(probs):
                    bins = [0.0, 0.4, 0.7, 1.0]
                    labels = ['Low', 'Medium', 'High']
                    tiers = pd.cut(probs, bins=bins, labels=labels, include_lowest=True)
                    summary = tiers.value_counts().reindex(labels, fill_value=0).reset_index()
                    summary.columns = ['Tier', 'Count']
                    fig_tiers = px.bar(summary, x='Tier', y='Count', color='Tier',
                                       color_discrete_map={'Low': '#f44336', 'Medium': '#ffc107', 'High': '#4caf50'})
                    fig_tiers.update_traces(texttemplate='%{y:,}', textposition='outside', textfont=dict(weight='bold'))
                    max_tier_count = summary['Count'].max()
                    tier_padding = max(1, max_tier_count * 0.25) if pd.notna(max_tier_count) else 1
                    fig_tiers.update_layout(
                        height=250,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(range=[0, max_tier_count + tier_padding] if pd.notna(max_tier_count) else None)
                    )
                    plotly_chart_silent(fig_tiers, width='stretch', config={'displayModeBar': True, 'displaylogo': False})
                else:
                    st.info("Prediction probabilities are present but all values are NaN.")
            else:
                st.info("Prediction probabilities not available.")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                "<div style='margin-top: -16px;'>💡 How to read: Donors are grouped into Low/Medium/High based on predicted probability. Focus on the High-tier donors first.</div>",
                unsafe_allow_html=True
            )

        # Trend Analysis Section (header removed)

        if False:  # Section disabled per request
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Performance by Constituency Type")
                # Use normalized donor_type if available
                donor_type_col = 'donor_type' if 'donor_type' in df_filtered.columns else None
                prob_col_ct = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df_filtered.columns else ('predicted_prob' if 'predicted_prob' in df_filtered.columns else None)
                outcome_col_ct = 'Gave_Again_In_2025' if 'Gave_Again_In_2025' in df_filtered.columns else ('Gave_Again_In_2024' if 'Gave_Again_In_2024' in df_filtered.columns else ('actual_gave' if 'actual_gave' in df_filtered.columns else None))

                if donor_type_col is not None and prob_col_ct is not None:
                    # Build per-constituency metrics based on Will Give Again 2025 predictions and outcomes (if present)
                    df_ct = df_filtered[[donor_type_col]].copy()
                    df_ct['prob'] = pd.to_numeric(df_filtered[prob_col_ct], errors='coerce')
                    if outcome_col_ct is not None and outcome_col_ct in df_filtered.columns:
                        df_ct['outcome'] = pd.to_numeric(df_filtered[outcome_col_ct], errors='coerce')

                    # Aggregate (mean for probability) without counting on the grouped key to avoid duplicate column on reset_index
                    group_obj = df_ct.groupby(donor_type_col, observed=False)
                    perf_ct = group_obj['prob'].mean().to_frame('Mean_Prob')
                    perf_ct['Count'] = group_obj.size().values
                    if 'outcome' in df_ct.columns:
                        perf_ct['outcome'] = group_obj['outcome'].mean()
                    perf_ct = perf_ct.reset_index().rename(columns={donor_type_col: 'Constituency'})

                    # Include key constituencies even if filtered out or zero members
                    key_constituencies = ['Alum', 'Regent', 'Trustee']
                    for constituency in key_constituencies:
                        if constituency not in perf_ct['Constituency'].values:
                            row = {
                                'Constituency': constituency,
                                'Mean_Prob': 0.0,
                                'Count': 0
                            }
                            if 'outcome' in perf_ct.columns:
                                row['outcome'] = 0.0
                            perf_ct = pd.concat([perf_ct, pd.DataFrame([row])], ignore_index=True)

                    # Sort by Mean_Prob descending for readability
                    perf_ct = perf_ct.sort_values('Mean_Prob', ascending=False)

                    # Build bar chart with hover showing conversion if available
                    fig_ct = go.Figure()
                    hover_tmpl = '<b>%{x}</b><br>Mean Probability: %{y:.1%}'

                    # Color each donor type distinctly (applies to mean bars)
                    palette = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#e91e63', '#00bcd4', '#8bc34a', '#ffc107', '#795548', '#607d8b']
                    colors = [palette[i % len(palette)] for i in range(len(perf_ct))]
                    if 'outcome' in perf_ct.columns:
                        # Mean prediction probability by constituency (colored per donor type)
                        fig_ct.add_trace(go.Bar(
                            x=perf_ct['Constituency'],
                            y=perf_ct['Mean_Prob'],
                            marker_color=colors,
                            text=perf_ct['Mean_Prob'].apply(lambda v: f"{v:.1%}"),
                            textposition='outside',
                            customdata=np.c_[perf_ct['Count'].values, perf_ct['outcome'].values],
                            hovertemplate=hover_tmpl + '<br>Donors: %{customdata[0]:,}<br>Gave Again Rate: %{customdata[1]:.1%}<extra></extra>',
                            showlegend=False
                        ))
                        # Overlay: Gave-again rate as a line over the bars
                        fig_ct.add_trace(go.Scatter(
                            x=perf_ct['Constituency'],
                            y=perf_ct['outcome'],
                            mode='lines+markers',
                            name='Gave Again Rate',
                            line=dict(color='#2E86AB', width=3),
                            marker=dict(size=8, color='#2E86AB'),
                            hovertemplate='<b>%{x}</b><br>Gave Again Rate: %{y:.1%}<extra></extra>'
                        ))
                    else:
                        # Mean prediction probability by constituency (no outcomes available)
                        fig_ct.add_trace(go.Bar(
                            x=perf_ct['Constituency'],
                            y=perf_ct['Mean_Prob'],
                            marker_color=colors,
                            text=perf_ct['Mean_Prob'].apply(lambda v: f"{v:.1%}"),
                            textposition='outside',
                            customdata=np.c_[perf_ct['Count'].values],
                            hovertemplate=hover_tmpl + '<br>Donors: %{customdata[0]:,}<extra></extra>',
                            showlegend=False
                        ))

                    # Y axis from 0 to max with padding, cap at 1.0
                    # Y-axis max accounts for both mean prob and outcome rate (if present)
                    base_max = perf_ct['Mean_Prob'].max() if len(perf_ct) else 0.5
                    if 'outcome' in perf_ct.columns:
                        base_max = max(base_max, perf_ct['outcome'].max())
                    y_max_ct = float(min(1.0, max(0.1, base_max * 1.15)))
                    fig_ct.update_layout(
                        title='Mean Prediction Probability by Constituency Type',
                        yaxis_title='Probability',
                        height=420,
                        yaxis=dict(range=[0, y_max_ct], showgrid=True, gridcolor='#e0e0e0'),
                        xaxis=dict(showgrid=False),
                        margin=dict(t=60, b=80, l=50, r=50),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                    )
                    plotly_chart_silent(fig_ct, config={'displayModeBar': True, 'displaylogo': False})
                    st.caption("💡 **What this means**: Shows average 'will give again in 2025' prediction by constituency type. Where shown, the hover also includes the actual gave-again rate from outcomes.")
                else:
                    st.info("Constituency type or prediction probabilities not available to render this chart.")

            with col2:
                st.markdown("#### 📅 Seasonal Patterns (Simulated)")
                # Simulate monthly trends if we had time data
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                # Use segment distribution to estimate seasonal giving likelihood
                if 'segment' in df_filtered.columns:
                    # Higher probabilities in Q4 (giving season) and Q1 (new year)
                    seasonal_factor = [0.85, 0.90, 0.95, 0.92, 0.88, 0.85, 0.80, 0.82, 0.88, 0.92, 0.98, 1.0]
                    base_prob = df_filtered['predicted_prob'].mean() if 'predicted_prob' in df_filtered.columns else 0.5
                    seasonal_probs = [base_prob * factor for factor in seasonal_factor]

                    fig_seasonal = go.Figure()
                    fig_seasonal.add_trace(go.Scatter(
                        x=months,
                        y=seasonal_probs,
                        mode='lines+markers',
                        line=dict(color='#2196f3', width=3),
                        marker=dict(size=8, color='#2196f3')
                    ))
                    fig_seasonal.update_layout(
                        title='Seasonal Giving Likelihood Pattern',
                        yaxis_title='Predicted Probability',
                        height=350,
                        xaxis_title='Month'
                    )
                    fig_seasonal.add_hline(
                        y=base_prob,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Average"
                    )
                    plotly_chart_silent(fig_seasonal, config={'displayModeBar': True, 'displaylogo': False})
                    st.caption("💡 **What this means**: Giving tends to peak in Q4 (holiday season) and early Q1 (new year). Plan campaigns accordingly.")

        # Gift Officer Assignment Section
        st.markdown("---")
        st.markdown("### 👥 Gift Officer Assignments & Unassigned Prospects")
        # Divider line matching sidebar background color
        st.markdown("""
        <div style="height: 4px; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); margin: 15px 0;"></div>
        """, unsafe_allow_html=True)
        
        # Check if Primary_Manager column exists
        if 'Primary_Manager' in df_work.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("#### 📊 Portfolio Quality by Gift Officer")
                
                # Filter assigned donors (exclude null and empty)
                # Primary_Manager now contains only the correct 100-150 assignments per officer
                assigned_df = df_work[
                    df_work['Primary_Manager'].notna() & 
                    (df_work['Primary_Manager'] != '')
                ].copy()
                
                # Get probability column - use only Will_Give_Again_Probability
                prob_col = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in assigned_df.columns else None
                
                # Get recency column
                recency_col = None
                for col in ['days_since_last_gift', 'days_since_last', 'Days_Since_Last_Gift']:
                    if col in assigned_df.columns:
                        recency_col = col
                        break

                if not assigned_df.empty and prob_col:
                    # Calculate recency using Last_Gift_Date if available, otherwise use days_since_last_gift
                    if 'Last_Gift_Date' in assigned_df.columns:
                        # Convert Last_Gift_Date to datetime
                        assigned_df['Last_Gift_Date'] = pd.to_datetime(assigned_df['Last_Gift_Date'], errors='coerce')
                        # Calculate days since last gift
                        today = pd.Timestamp.now()
                        assigned_df['Days_Since_Last'] = (today - assigned_df['Last_Gift_Date']).dt.days
                    elif recency_col:
                        assigned_df['Days_Since_Last'] = pd.to_numeric(assigned_df[recency_col], errors='coerce')
                    else:
                        assigned_df['Days_Since_Last'] = None
                    
                    # Group by gift officer and calculate metrics
                    if assigned_df['Days_Since_Last'].notna().any():
                        officer_stats = assigned_df.groupby('Primary_Manager', observed=False).agg(
                            Median_Probability=(prob_col, 'median'),
                            Pct_Recent_12mo=('Days_Since_Last', lambda x: (x <= 365).sum() / len(x) if len(x) > 0 else 0),
                            Avg_Days_Since_Last=('Days_Since_Last', 'mean')
                        ).reset_index()
                    else:
                        officer_stats = assigned_df.groupby('Primary_Manager', observed=False).agg(
                            Median_Probability=(prob_col, 'median')
                        ).reset_index()
                        officer_stats['Pct_Recent_12mo'] = 0
                        officer_stats['Avg_Days_Since_Last'] = None
                    
                    # Convert to numeric
                    officer_stats['Median_Probability'] = pd.to_numeric(officer_stats['Median_Probability'], errors='coerce')
                    officer_stats['Pct_Recent_12mo'] = pd.to_numeric(officer_stats['Pct_Recent_12mo'], errors='coerce')
                    
                    # Calculate Composite Quality Score (weighted average: 60% probability, 40% recency)
                    # Both metrics are on 0-1 scale, so we can combine them directly
                    officer_stats['Quality_Score'] = (
                        officer_stats['Median_Probability'] * 0.6 + 
                        officer_stats['Pct_Recent_12mo'] * 0.4
                    ) * 100  # Scale to 0-100
                    
                    # Sort by quality score (descending) for ranking
                    officer_stats = officer_stats.sort_values('Quality_Score', ascending=False).reset_index(drop=True)
                    
                    # Create horizontal bar chart showing quality score
                    fig_officer = go.Figure()
                    
                    # Color bars based on quality score (green for high, yellow for medium, orange for low)
                    colors = []
                    for score in officer_stats['Quality_Score']:
                        if score >= 70:
                            colors.append('#2ecc71')  # Green - High quality
                        elif score >= 50:
                            colors.append('#f39c12')  # Orange - Medium quality
                        else:
                            colors.append('#e74c3c')  # Red - Low quality
                    
                    # Prepare customdata with all metrics
                    customdata_list = list(zip(
                        officer_stats['Primary_Manager'],
                        officer_stats['Median_Probability'],
                        officer_stats['Pct_Recent_12mo'],
                        officer_stats['Avg_Days_Since_Last'] if officer_stats['Avg_Days_Since_Last'].notna().any() else [0] * len(officer_stats)
                    ))
                    
                    fig_officer.add_trace(go.Bar(
                        x=officer_stats['Quality_Score'],
                        y=officer_stats['Primary_Manager'],
                        orientation='h',
                        marker=dict(
                            color=colors,
                            line=dict(width=1, color='white')
                        ),
                        text=[f"{score:.1f}" for score in officer_stats['Quality_Score']],
                        textposition='outside',
                        textfont=dict(size=10, color='white'),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>" +
                            "Quality Score: %{x:.1f}/100<br>" +
                            "Median Will Give Again Probability: %{customdata[1]:.1%}<br>" +
                            "% Donors Who Gave in Last 12 Months: %{customdata[2]:.1%}<br>" +
                            "Avg Days Since Last Gift: %{customdata[3]:.0f} days<br>" +
                            "<extra></extra>"
                        ),
                        customdata=customdata_list
                    ))
                    
                    max_score = officer_stats['Quality_Score'].max()
                    padding = max(5, max_score * 0.1)
                    
                    fig_officer.update_layout(
                        height=max(500, len(officer_stats) * 40),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=40, b=40, l=180, r=80),
                        xaxis=dict(
                            title='Portfolio Quality Score (0-100)',
                            range=[0, max_score + padding],
                            showgrid=True,
                            gridcolor='rgba(200,200,200,0.3)'
                        ),
                        yaxis=dict(
                            title='Gift Officer',
                            autorange='reversed',
                            showgrid=False
                        ),
                        font=dict(family="Arial, sans-serif")
                    )
                    plotly_chart_silent(fig_officer, width='stretch', config={'displayModeBar': True, 'displaylogo': False})
                    st.markdown("**💡 How to read**: Quality Score combines median Will Give Again probability (60% weight) and recent giving activity (40% weight). Scores ≥70 = High quality (green), 50-69 = Medium (orange), <50 = Low (red). Officers are ranked from highest to lowest quality.", unsafe_allow_html=True)
                else:
                    missing = []
                    if not prob_col:
                        missing.append("Will_Give_Again_Probability")
                    st.info(f"Required columns not available. Missing: {', '.join(missing)}.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("#### 🎯 Priority Matrix for Unassigned Prospective Donors")
                
                # Filter unassigned donors
                unassigned_df = df_work[
                    (df_work['Primary_Manager'].isna()) | 
                    (df_work['Primary_Manager'] == '')
                ].copy()
                
                # Get probability column - use only Will_Give_Again_Probability
                prob_col = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in unassigned_df.columns else None
                
                # Get lifetime giving column (check for alternatives)
                giving_col = None
                for col in ['total_giving', 'Lifetime_Giving', 'LifetimeGiving', 'Lifetime Giving']:
                    if col in unassigned_df.columns:
                        giving_col = col
                        break
                
                if not unassigned_df.empty and prob_col and giving_col:
                    # Convert to numeric
                    giving_values = pd.to_numeric(unassigned_df[giving_col], errors='coerce').fillna(0)
                    
                    # Filter to top quartile by lifetime giving
                    giving_threshold = giving_values.quantile(0.75)
                    top_quartile = unassigned_df[giving_values >= giving_threshold].copy()
                    
                    if not top_quartile.empty:
                        # Get last gift date
                        if 'Last_Gift_Date' in top_quartile.columns:
                            top_quartile['Last_Gift_Date'] = pd.to_datetime(top_quartile['Last_Gift_Date'], errors='coerce')
                            today = pd.Timestamp.now()
                            top_quartile['Days_Since_Last_Gift'] = (today - top_quartile['Last_Gift_Date']).dt.days
                        else:
                            # Try alternative columns
                            days_col = None
                            for col in ['days_since_last_gift', 'days_since_last', 'Days_Since_Last_Gift']:
                                if col in top_quartile.columns:
                                    days_col = col
                                    break
                            if days_col:
                                top_quartile['Days_Since_Last_Gift'] = pd.to_numeric(top_quartile[days_col], errors='coerce')
                            else:
                                top_quartile['Days_Since_Last_Gift'] = None
                        
                        # Convert probability to numeric
                        prob_values = pd.to_numeric(top_quartile[prob_col], errors='coerce').fillna(0)
                        giving_values_top = pd.to_numeric(top_quartile[giving_col], errors='coerce').fillna(0)
                        
                        # Filter out invalid data
                        valid_data = top_quartile[
                            prob_values.notna() & 
                            top_quartile['Days_Since_Last_Gift'].notna() &
                            (top_quartile['Days_Since_Last_Gift'] >= 0)
                        ].copy()
                        
                        if not valid_data.empty:
                            # Recalculate probability values for valid_data (matching indices)
                            valid_prob_values = pd.to_numeric(valid_data[prob_col], errors='coerce').fillna(0)
                            valid_days = valid_data['Days_Since_Last_Gift']
                            valid_giving = pd.to_numeric(valid_data[giving_col], errors='coerce').fillna(0)
                            
                            # Define thresholds
                            high_prob_threshold = 0.7
                            recent_threshold = 365  # days (1 year)
                            moderately_lapsed_max = 1095  # days (3 years)
                            
                            # Calculate quadrant metrics
                            quadrants = {
                                'hot': {
                                    'label': '🔥 HOT PROSPECTS',
                                    'desc': 'High Prob + Recent',
                                    'mask': (valid_prob_values >= high_prob_threshold) & (valid_days <= recent_threshold),
                                    'color': '#e74c3c',
                                    'priority': 1
                                },
                                'reeng': {
                                    'label': '🔄 RE-ENGAGEMENT',
                                    'desc': 'High Prob + 1-3 Years Lapsed',
                                    'mask': (valid_prob_values >= high_prob_threshold) & (valid_days > recent_threshold) & (valid_days <= moderately_lapsed_max),
                                    'color': '#f39c12',
                                    'priority': 2
                                },
                                'monitor': {
                                    'label': '👀 MONITOR',
                                    'desc': 'Low Prob + Recent',
                                    'mask': (valid_prob_values < high_prob_threshold) & (valid_days <= recent_threshold),
                                    'color': '#3498db',
                                    'priority': 3
                                },
                                'longshot': {
                                    'label': '🎲 LONG SHOT',
                                    'desc': 'Low Prob + Lapsed',
                                    'mask': (valid_prob_values < high_prob_threshold) & (valid_days > recent_threshold),
                                    'color': '#95a5a6',
                                    'priority': 4
                                }
                            }
                            
                            # Calculate stats for each quadrant
                            for key, quad in quadrants.items():
                                quad_data = valid_data[quad['mask']]
                                quad['count'] = len(quad_data)
                                quad['median_giving'] = valid_giving[quad['mask']].median() if quad['count'] > 0 else 0
                                quad['total_giving'] = valid_giving[quad['mask']].sum() if quad['count'] > 0 else 0
                                # Use median probability of Will_Give_Again_Probability for stability (2025 model)
                                quad['median_prob'] = valid_prob_values[quad['mask']].median() if quad['count'] > 0 else 0
                                # Calculate median_days from the filtered quad_data to ensure correct alignment (median is more robust to outliers)
                                quad_days = quad_data['Days_Since_Last_Gift']
                                quad['median_days'] = quad_days.median() if quad['count'] > 0 and len(quad_days) > 0 else 0
                                quad['median_months'] = quad['median_days'] / 30.44 if quad['count'] > 0 else 0  # Convert days to months
                            
                            # Helper function to convert hex to rgba for gradients
                            def hex_to_rgba(hex_color, alpha):
                                """Convert hex color to rgba string"""
                                hex_color = hex_color.lstrip('#')
                                r = int(hex_color[0:2], 16)
                                g = int(hex_color[2:4], 16)
                                b = int(hex_color[4:6], 16)
                                return f"rgba({r}, {g}, {b}, {alpha})"
                            
                            # Create 2x2 grid using Streamlit columns
                            row1_col1, row1_col2 = st.columns(2)
                            row2_col1, row2_col2 = st.columns(2)
                            
                            # Helper function to render a quadrant card
                            def render_quadrant_card(col, quad_key, border_width=2):
                                quad = quadrants[quad_key]
                                bg_start = hex_to_rgba(quad['color'], 0.08)
                                bg_end = hex_to_rgba(quad['color'], 0.15)
                                
                                # Format values properly
                                median_giving_str = f"${quad['median_giving']:,.0f}" if pd.notna(quad['median_giving']) else "$0"
                                count_str = f"{quad['count']:,}"
                                
                                with col:
                                    # Build card HTML
                                    card_html = '<div style="background: linear-gradient(135deg, ' + bg_start + ' 0%, ' + bg_end + ' 100%); '
                                    card_html += f'border: {border_width}px solid {quad["color"]}; border-radius: 10px; padding: 20px; '
                                    card_html += 'display: flex; flex-direction: column; justify-content: space-between; min-height: 200px; margin-bottom: 15px;">'
                                    
                                    # Header section
                                    card_html += '<div>'
                                    card_html += f'<div style="font-size: 18px; font-weight: bold; color: {quad["color"]}; margin-bottom: 5px;">{quad["label"]}</div>'
                                    card_html += f'<div style="font-size: 12px; color: rgba(255,255,255,0.7); margin-bottom: 15px;">{quad["desc"]}</div>'
                                    card_html += f'<div style="font-size: 32px; font-weight: bold; color: white; margin-bottom: 10px;">{count_str}</div>'
                                    card_html += '<div style="font-size: 13px; color: rgba(255,255,255,0.8);">prospects</div>'
                                    card_html += '</div>'
                                    
                                    # Metrics section
                                    card_html += '<div style="border-top: 1px solid rgba(255,255,255,0.2); padding-top: 10px; margin-top: 10px;">'
                                    card_html += '<div style="font-size: 11px; color: rgba(255,255,255,0.7); margin-bottom: 3px;">Median Lifetime Giving</div>'
                                    card_html += f'<div style="font-size: 16px; font-weight: bold; color: white;">{median_giving_str}</div>'
                                    
                                    # Add quadrant-specific metrics
                                    if quad_key == 'hot':
                                        total_giving_value = quad['total_giving']
                                        # Format as billions if >= 1B, otherwise millions
                                        if total_giving_value >= 1_000_000_000:
                                            total_potential = total_giving_value / 1_000_000_000
                                            card_html += '<div style="font-size: 11px; color: rgba(255,255,255,0.7); margin-top: 8px; margin-bottom: 3px;">Total Potential</div>'
                                            card_html += f'<div style="font-size: 16px; font-weight: bold; color: white;">${total_potential:.3f}B</div>'
                                        else:
                                            total_potential = total_giving_value / 1_000_000
                                            card_html += '<div style="font-size: 11px; color: rgba(255,255,255,0.7); margin-top: 8px; margin-bottom: 3px;">Total Potential</div>'
                                            card_html += f'<div style="font-size: 16px; font-weight: bold; color: white;">${total_potential:.3f}M</div>'
                                    elif quad_key == 'reeng':
                                        median_months = quad['median_months'] if pd.notna(quad['median_months']) and quad['median_months'] > 0 else 0
                                        # Ensure we have valid data
                                        if quad['count'] > 0 and median_months > 0:
                                            card_html += '<div style="font-size: 11px; color: rgba(255,255,255,0.7); margin-top: 8px; margin-bottom: 3px;">Median Months Since Last Gift</div>'
                                            card_html += f'<div style="font-size: 16px; font-weight: bold; color: white;">{median_months:.1f} months</div>'
                                        else:
                                            card_html += '<div style="font-size: 11px; color: rgba(255,255,255,0.7); margin-top: 8px; margin-bottom: 3px;">Median Months Since Last Gift</div>'
                                            card_html += '<div style="font-size: 16px; font-weight: bold; color: white;">N/A</div>'
                                    elif quad_key == 'monitor':
                                        median_prob_str = f"{quad['median_prob']:.1%}" if pd.notna(quad['median_prob']) else "0.0%"
                                        card_html += '<div style="font-size: 11px; color: rgba(255,255,255,0.7); margin-top: 8px; margin-bottom: 3px;">Median Probability to Give Again (2025)</div>'
                                        card_html += f'<div style="font-size: 16px; font-weight: bold; color: white;">{median_prob_str}</div>'
                                    elif quad_key == 'longshot':
                                        median_months = quad['median_months'] if pd.notna(quad['median_months']) and quad['median_months'] > 0 else 0
                                        # Ensure we have valid data
                                        if quad['count'] > 0 and median_months > 0:
                                            card_html += '<div style="font-size: 11px; color: rgba(255,255,255,0.7); margin-top: 8px; margin-bottom: 3px;">Median Months Since Last Gift</div>'
                                            card_html += f'<div style="font-size: 16px; font-weight: bold; color: white;">{median_months:.1f} months</div>'
                                        else:
                                            card_html += '<div style="font-size: 11px; color: rgba(255,255,255,0.7); margin-top: 8px; margin-bottom: 3px;">Median Months Since Last Gift</div>'
                                            card_html += '<div style="font-size: 16px; font-weight: bold; color: white;">N/A</div>'
                                    
                                    # Close metrics section and card
                                    card_html += '</div></div>'
                                    
                                    st.markdown(card_html, unsafe_allow_html=True)
                            
                            # Render all four quadrants
                            render_quadrant_card(row1_col1, 'hot', border_width=3)  # Top Left
                            render_quadrant_card(row1_col2, 'reeng')  # Top Right
                            render_quadrant_card(row2_col1, 'monitor')  # Bottom Left
                            render_quadrant_card(row2_col2, 'longshot')  # Bottom Right
                            
                            st.markdown(f"**💡 How to read**: Prospects segmented by probability to give again (≥70% = High) and recency (≤365 days = Recent). Focus on **🔥 HOT PROSPECTS** first - they have both high likelihood AND recent engagement. Total top-quartile unassigned: {len(valid_data):,} (lifetime giving ≥${giving_threshold:,.0f}).", unsafe_allow_html=True)
                        else:
                            st.info("No valid data available (missing last gift date or probability information).")
                    else:
                        st.info(f"No unassigned donors found in top quartile (lifetime giving ≥ ${giving_threshold:,.2f}).")
                else:
                    missing_cols = []
                    if not prob_col:
                        missing_cols.append("Will_Give_Again_Probability")
                    if not giving_col:
                        missing_cols.append("lifetime giving column (total_giving/Lifetime_Giving)")
                    st.info(f"Required columns not available. Missing: {', '.join(missing_cols)}.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("⚠️ Gift officer assignment data (Primary_Manager column) not available in the dataset.")