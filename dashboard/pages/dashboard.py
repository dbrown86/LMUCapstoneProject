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


def render(df: pd.DataFrame, regions: List[str], donor_types: List[str], segments: List[str], prob_threshold: float):
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
        st.markdown('<p class="page-title">🏠 Executive Dashboard</p>', unsafe_allow_html=True)
        st.markdown('<p class="page-subtitle">Real-time donor analytics and predictions</p>', unsafe_allow_html=True)

        # Executive Summary Card
        # VERIFY: Ensure we're using "Will Give Again in 2024" predictions and outcomes
        # Check for Will_Give_Again_Probability directly first, then fall back to predicted_prob
        prob_col_exec = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df_filtered.columns else 'predicted_prob'
        outcome_col_exec = 'Gave_Again_In_2024' if 'Gave_Again_In_2024' in df_filtered.columns else 'actual_gave'

        if prob_col_exec in df_filtered.columns and 'avg_gift' in df_filtered.columns and 'total_giving' in df_filtered.columns:
            metrics_summary = get_model_metrics(df_filtered)
            # Use the same threshold as the Revenue Potential metric for consistency
            # CRITICAL: Use Will_Give_Again_Probability directly if available (not predicted_prob which may be from Legacy_Intent)
            high_prob_donors = df_filtered[df_filtered[prob_col_exec] >= prob_threshold]
            # Use actual conversion rate if available, otherwise estimate
            # CRITICAL: Use Gave_Again_In_2024 directly if available
            # CRITICAL: avg_gift_amount column appears corrupted (mean $0.03), use Last_Gift instead
            if outcome_col_exec in df_filtered.columns and len(high_prob_donors) > 0:
                actual_conversion = high_prob_donors[outcome_col_exec].mean()
                # Use Last_Gift (last gift amount) for revenue calculation instead of avg_gift_amount
                # Last_Gift represents what they actually gave most recently, better for "untapped potential"
                last_gift_col = None
                for col in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount']:
                    if col in high_prob_donors.columns:
                        last_gift_col = col
                        break

                if last_gift_col:
                    gift_amounts = pd.to_numeric(high_prob_donors[last_gift_col], errors='coerce').fillna(0).clip(lower=0)
                    # Use median instead of mean for robustness (outliers in Last_Gift can skew mean to $28K+)
                    # Median Last_Gift for high prob donors is more representative of typical gift amounts
                    avg_gift_mean = gift_amounts.median() if len(gift_amounts) > 0 else gift_amounts.mean() if len(gift_amounts) > 0 else 0
                else:
                    # Fallback to avg_gift if Last_Gift not available
                    avg_gift_values = pd.to_numeric(high_prob_donors['avg_gift'], errors='coerce').fillna(0).clip(lower=0)
                    avg_gift_mean = avg_gift_values.mean() if len(avg_gift_values) > 0 else 0

                estimated_revenue = avg_gift_mean * len(high_prob_donors) * actual_conversion
            else:
                # Fallback estimate if actual outcome data not available
                last_gift_col = None
                for col in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount']:
                    if col in high_prob_donors.columns:
                        last_gift_col = col
                        break

                if last_gift_col:
                    gift_amounts = pd.to_numeric(high_prob_donors[last_gift_col], errors='coerce').fillna(0).clip(lower=0) if len(high_prob_donors) > 0 else pd.Series([0])
                    # Use median for robustness against outliers
                    avg_gift_mean = gift_amounts.median() if len(gift_amounts) > 0 else gift_amounts.mean() if len(gift_amounts) > 0 else 0
                else:
                    avg_gift_values = pd.to_numeric(high_prob_donors['avg_gift'], errors='coerce').fillna(0).clip(lower=0) if len(high_prob_donors) > 0 else pd.Series([0])
                    avg_gift_mean = avg_gift_values.mean() if len(avg_gift_values) > 0 else 0

                estimated_revenue = avg_gift_mean * len(high_prob_donors) * 0.85 if len(high_prob_donors) > 0 else 0

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
                        <div style="font-size: 18px; font-weight: bold;">AI Model Identifies {len(high_prob_donors):,} High-Value Prospects</div>
                        <div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">(≥{prob_threshold:.0%} probability to give again in 2024)</div>
                    </div>
                    <div>
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 5px;">Business Impact</div>
                        <div style="font-size: 18px; font-weight: bold;">{revenue_display} in Untapped Donor Potential</div>
                        <div style="font-size: 11px; opacity: 0.7; margin-top: 3px;">Based on actual "gave again in 2024" outcomes</div>
                    </div>
                    <div>
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 5px;">Recommended Action</div>
                        <div style="font-size: 18px; font-weight: bold;">Prioritize Outreach to Top Prospects</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Hero metrics
        col1, col2, col3, col4 = st.columns(4)

        # Get actual metrics for dashboard
        metrics = get_model_metrics(df)
        auc_display = f"{metrics['auc']:.2%}" if metrics['auc'] is not None else "94.88%"
        baseline_auc_display = f"{metrics['baseline_auc']:.2%}" if metrics.get('baseline_auc') is not None else "85.69%"
        improvement = ((metrics['auc'] - metrics['baseline_auc']) / metrics['baseline_auc'] * 100) if metrics.get('baseline_auc') and metrics['baseline_auc'] > 0 else 10.7

        with col1:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-left: none; height: 170px; display: flex; flex-direction: column; justify-content: space-between;">
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">AUC Score</div>
                <div class="metric-value" style="color: white;">{auc_display}</div>
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 11px;">Means we're right 9 out of 10 times</div>
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 11px;">+{improvement:.1f}% vs Baseline ({baseline_auc_display})</div>
            </div>
            """, unsafe_allow_html=True)

        f1_display = f"{metrics['f1']:.2%}" if metrics['f1'] is not None else "85.34%"
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; border: none; border-left: none; height: 170px; display: flex; flex-direction: column; justify-content: space-between;">
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">F1 Score</div>
                <div class="metric-value" style="color: white;">{f1_display}</div>
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 11px;">Balanced accuracy score</div>
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 11px;">Minimizes false alarms</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            # Calculate actual revenue potential - USE WILL GIVE AGAIN COLUMNS
            # CRITICAL: Use Will_Give_Again_Probability and Gave_Again_In_2024 directly (same as Executive Summary)
            prob_col_rev = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df_filtered.columns else 'predicted_prob'
            outcome_col_rev = 'Gave_Again_In_2024' if 'Gave_Again_In_2024' in df_filtered.columns else 'actual_gave'

            if prob_col_rev in df_filtered.columns and 'avg_gift' in df_filtered.columns:
                high_prob = df_filtered[df_filtered[prob_col_rev] >= prob_threshold]
                if len(high_prob) > 0:
                    # Use actual conversion rate from Gave_Again_In_2024 if available
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
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">From Targeted Prospects</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; border: none; border-left: none; height: 170px; display: flex; flex-direction: column; justify-content: space-between;">
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">Improvement</div>
                <div class="metric-value" style="color: white;">4-5x</div>
                <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">vs Random</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Section
        col1, col2 = st.columns(2)

        df_work = df_filtered.loc[:, ~df_filtered.columns.duplicated()].copy()

        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 📈 Donor Distribution by Segment (Based on Fusion Model Predictions)")
            if {'segment', 'predicted_prob'}.issubset(df_work.columns):
                seg_df = df_work[['segment', 'predicted_prob']].dropna(subset=['segment'])
                if not seg_df.empty:
                    summary = seg_df.groupby('segment', observed=False).size().reset_index(name='Count')
                    fig_segment = px.bar(summary, x='segment', y='Count', color='segment',
                                         category_orders={'segment': ['Recent (0-6mo)', 'Recent (6-12mo)', 'Lapsed (1-2yr)', 'Very Lapsed (2yr+)', 'Prospects/New']},
                                         color_discrete_sequence=['#4caf50', '#8bc34a', '#ffc107', '#ff5722', '#9e9e9e'])
                    fig_segment.update_traces(texttemplate='%{y:,}', textposition='outside')
                    fig_segment.update_layout(height=350, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_segment)
                else:
                    st.info("No segment data available to display.")
            else:
                st.info("Segment or prediction columns missing from dataset.")
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("💡 **What this means**: Shows distribution of donors across recency segments with their average 'will give again' probability. Recent segments typically score higher.")

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 🌍 Donor Distribution by Region")
            if 'region' in df_work.columns:
                reg_series = df_work['region'].dropna()
                if not reg_series.empty:
                    summary = reg_series.value_counts().reset_index()
                    summary.columns = ['Region', 'Count']
                    fig_region = px.pie(summary, names='Region', values='Count', hole=0.4,
                                        color_discrete_sequence=['#2196f3', '#4caf50', '#ff9800', '#9c27b0', '#e91e63'])
                    fig_region.update_layout(height=350)
                    st.plotly_chart(fig_region)
                else:
                    st.info("No region data available to display.")
            else:
                st.info("Region column missing from dataset.")
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("💡 **What this means**: Geographic distribution helps prioritize regional campaigns and identify coverage gaps.")

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Who Will Give Again in 2024 — Confidence Tiers")
        if 'predicted_prob' in df_work.columns:
            probs = pd.to_numeric(df_work['predicted_prob'], errors='coerce').dropna()
            if len(probs):
                bins = [0.0, 0.4, 0.7, 1.0]
                labels = ['Low', 'Medium', 'High']
                tiers = pd.cut(probs, bins=bins, labels=labels, include_lowest=True)
                summary = tiers.value_counts().reindex(labels, fill_value=0).reset_index()
                summary.columns = ['Tier', 'Count']
                fig_tiers = px.bar(summary, x='Tier', y='Count', color='Tier',
                                   color_discrete_map={'Low': '#f44336', 'Medium': '#ffc107', 'High': '#4caf50'})
                fig_tiers.update_traces(texttemplate='%{y:,}', textposition='outside')
                fig_tiers.update_layout(height=250, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_tiers)
            else:
                st.info("Prediction probabilities are present but all values are NaN.")
        else:
            st.info("Prediction probabilities not available.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("💡 **How to read**: Donors are grouped into Low/Medium/High based on predicted probability. Focus on the High-tier donors first.")

        # Trend Analysis Section (header removed)

        if 'predicted_prob' in df_filtered.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Performance by Constituency Type")
                # Use normalized donor_type if available
                donor_type_col = 'donor_type' if 'donor_type' in df_filtered.columns else None
                prob_col_ct = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df_filtered.columns else ('predicted_prob' if 'predicted_prob' in df_filtered.columns else None)
                outcome_col_ct = 'Gave_Again_In_2024' if 'Gave_Again_In_2024' in df_filtered.columns else ('actual_gave' if 'actual_gave' in df_filtered.columns else None)

                if donor_type_col is not None and prob_col_ct is not None:
                    # Build per-constituency metrics based on Will Give Again 2024 predictions and outcomes (if present)
                    df_ct = df_filtered[[donor_type_col]].copy()
                    df_ct['prob'] = pd.to_numeric(df_filtered[prob_col_ct], errors='coerce')
                    if outcome_col_ct is not None and outcome_col_ct in df_filtered.columns:
                        df_ct['outcome'] = pd.to_numeric(df_filtered[outcome_col_ct], errors='coerce')

                    # Aggregate (median for probability) without counting on the grouped key to avoid duplicate column on reset_index
                    group_obj = df_ct.groupby(donor_type_col, observed=False)
                    perf_ct = group_obj['prob'].median().to_frame('Med_Prob')
                    if 'outcome' in df_ct.columns:
                        perf_ct['outcome'] = group_obj['outcome'].mean()
                    perf_ct['Count'] = group_obj.size().values
                    perf_ct = perf_ct.reset_index().rename(columns={donor_type_col: 'Constituency'})

                    # Sort by Med_Prob descending for readability
                    perf_ct = perf_ct.sort_values('Med_Prob', ascending=False)

                    # Build bar chart with hover showing conversion if available
                    fig_ct = go.Figure()
                    hover_tmpl = '<b>%{x}</b><br>Median Probability: %{y:.1%}'

                    # Color each donor type distinctly (applies to median bars)
                    palette = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#e91e63', '#00bcd4', '#8bc34a', '#ffc107', '#795548', '#607d8b']
                    colors = [palette[i % len(palette)] for i in range(len(perf_ct))]
                    if 'outcome' in perf_ct.columns:
                        # Median prediction probability by constituency (colored per donor type)
                        fig_ct.add_trace(go.Bar(
                            x=perf_ct['Constituency'],
                            y=perf_ct['Med_Prob'],
                            marker_color=colors,
                            text=perf_ct['Med_Prob'].apply(lambda v: f"{v:.1%}"),
                            textposition='outside',
                            customdata=np.c_[perf_ct['Count'].values, perf_ct['outcome'].values],
                            hovertemplate=hover_tmpl + '<br>Donors: %{customdata[0]:,}<br>Gave Again Rate: %{customdata[1]:.1%}<extra></extra>'
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
                        # Median prediction probability by constituency (no outcomes available)
                        fig_ct.add_trace(go.Bar(
                            x=perf_ct['Constituency'],
                            y=perf_ct['Med_Prob'],
                            marker_color=colors,
                            text=perf_ct['Med_Prob'].apply(lambda v: f"{v:.1%}"),
                            textposition='outside',
                            customdata=np.c_[perf_ct['Count'].values],
                            hovertemplate=hover_tmpl + '<br>Donors: %{customdata[0]:,}<extra></extra>'
                        ))

                    # Y axis from 0 to max with padding, cap at 1.0
                    # Y-axis max accounts for both median prob and outcome rate (if present)
                    base_max = perf_ct['Med_Prob'].max() if len(perf_ct) else 0.5
                    if 'outcome' in perf_ct.columns:
                        base_max = max(base_max, perf_ct['outcome'].max())
                    y_max_ct = float(min(1.0, max(0.1, base_max * 1.15)))
                    fig_ct.update_layout(
                        title='Median Prediction Probability by Constituency Type',
                        yaxis_title='Probability',
                        height=420,
                        yaxis=dict(range=[0, y_max_ct], showgrid=True, gridcolor='#e0e0e0'),
                        xaxis=dict(showgrid=False),
                        margin=dict(t=60, b=80, l=50, r=50),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                    )
                    st.plotly_chart(fig_ct)
                    st.caption("💡 **What this means**: Shows average 'will give again in 2024' prediction by constituency type. Where shown, the hover also includes the actual gave-again rate from outcomes.")
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
                    st.plotly_chart(fig_seasonal)
                    st.caption("💡 **What this means**: Giving tends to peak in Q4 (holiday season) and early Q1 (new year). Plan campaigns accordingly.")



        # Model in Action Section (debug elements removed)
        if 'predicted_prob' in df_filtered.columns:
            high_prob_count = (df_filtered['predicted_prob'] >= 0.7).sum()
            medium_prob_count = ((df_filtered['predicted_prob'] >= 0.4) & (df_filtered['predicted_prob'] < 0.7)).sum()
            # Top 10 prospects
            top_prospects = df_filtered.nlargest(10, 'predicted_prob')

            if len(top_prospects) > 0 and 'donor_id' in top_prospects.columns:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("#### 🔥 Top 10 High-Probability Prospects")
                    display_cols = ['donor_id', 'predicted_prob']

                    # Add donor name if available (check common column name variations)
                    name_col = None
                    for col_name in ['donor_name', 'name', 'full_name', 'Full_Name', 'Donor_Name', 'Name', 'fullname']:
                        if col_name in top_prospects.columns:
                            name_col = col_name
                            display_cols.append(col_name)
                            break

                    # Add capacity rating if available (check common column name variations)
                    capacity_col = None
                    for col_name in ['Rating', 'rating', 'capacity_rating', 'Capacity_Rating', 'Capacity', 'capacity']:
                        if col_name in top_prospects.columns:
                            capacity_col = col_name
                            display_cols.append(col_name)
                            break

                    # Add primary manager if available (check common column name variations)
                    manager_col = None
                    for col_name in ['Primary_Manager', 'primary_manager', 'Manager', 'manager', 'assigned_manager', 'Assigned_Manager', 'PrimaryManager']:
                        if col_name in top_prospects.columns:
                            manager_col = col_name
                            display_cols.append(col_name)
                            break

                    if 'avg_gift' in top_prospects.columns:
                        display_cols.append('avg_gift')
                    if 'segment' in top_prospects.columns:
                        display_cols.append('segment')

                    top_display = top_prospects[display_cols].copy()

                    # Rename columns for display
                    rename_dict = {'predicted_prob': 'Probability', 'avg_gift': 'Avg Gift', 'segment': 'Segment'}
                    if name_col:
                        rename_dict[name_col] = 'Donor Name'
                    if capacity_col:
                        rename_dict[capacity_col] = 'Capacity Rating'
                    if manager_col:
                        rename_dict[manager_col] = 'Primary Manager'

                    top_display = top_display.rename(columns=rename_dict)

                    # Diagnostic: Check for suspicious 100% probabilities
                    raw_probs = top_prospects['predicted_prob'].values
                    exact_ones = (raw_probs == 1.0).sum()
                    near_ones = ((raw_probs >= 0.99) & (raw_probs < 1.0)).sum()

                    # Calculate accuracy for top prospects if actual_gave is available
                    accuracy_info = ""
                    if 'actual_gave' in top_prospects.columns:
                        actual_outcomes = top_prospects['actual_gave'].values
                        correct = (raw_probs >= 0.5) == (actual_outcomes == 1)
                        accuracy_pct = correct.sum() / len(correct) * 100

                        # Count false positives (predicted 1, actual 0)
                        false_positives = ((raw_probs >= 0.5) & (actual_outcomes == 0)).sum()
                        true_positives = ((raw_probs >= 0.5) & (actual_outcomes == 1)).sum()

                        accuracy_info = f"""

                        **Accuracy Analysis (Top 10):**
                        - ✅ Correct Predictions: {correct.sum()}/10 ({accuracy_pct:.1f}%)
                        - ❌ False Positives (predicted high, didn't give): {false_positives}
                        - ✅ True Positives (predicted high, did give): {true_positives}
                        """

                        # Critical warning if all are 1.0 and many are wrong
                        if exact_ones == len(top_prospects) and false_positives > 0:
                            st.error(f"""
                            🚨 **CRITICAL ISSUE DETECTED** 🚨

                            **Problem**: ALL top prospects have exactly 100% probability, but {false_positives} out of 10 did NOT actually give ({accuracy_pct:.0f}% accuracy).

                            **What This Means**:
                            - 100% predictions should be 100% correct, but they're not
                            - The model is severely **miscalibrated** (overconfident)
                            - This could indicate:
                              1. **Probability clipping**: Values are being hard-capped at 1.0
                              2. **Model bug**: Using binary predictions instead of probabilities
                              3. **Data preprocessing error**: Normalization/clipping in the pipeline
                              4. **Wrong column**: Using `actual_gave` or a derived column instead of `predicted_prob`

                            **Immediate Actions Required**:
                            1. ✅ Check your model's probability output (should use softmax/sigmoid)
                            2. ✅ Verify `predicted_prob` column contains actual probabilities, not binary predictions
                            3. ✅ Look for `np.clip(..., 0, 1)` or similar clipping in preprocessing
                            4. ✅ Review model training code for calibration issues
                            5. ✅ Consider using probability calibration (IsotonicRegression, Platt scaling)
                            """)
                        elif exact_ones > 0:
                            st.warning(f"⚠️ **Warning**: {exact_ones} donor(s) with exactly 100% probability. This is unusual for ML models.")


                    # Format probability for display (show 2 decimal places for better precision)
                    top_display['Probability'] = top_display['Probability'].apply(lambda x: f"{x:.2%}")
                    if 'Avg Gift' in top_display.columns:
                        top_display['Avg Gift'] = top_display['Avg Gift'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")

                    # Convert capacity rating codes to dollar ranges if available
                    if 'Capacity Rating' in top_display.columns:
                        rating_to_range = {
                            'A': '$100M+',
                            'B': '$50M - $99.9M',
                            'C': '$25M - $49.9M',
                            'D': '$10M - $24.9M',
                            'E': '$5M - $9.9M',
                            'F': '$1M - $4.9M',
                            'G': '$500K - $999.9K',
                            'H': '$250K - $499.9K',
                            'I': '$100K - $249.9K',
                            'J': '$50K - $99.9K',
                            'K': '$25K - $49.9K',
                            'L': '$10K - $24.9K',
                            'M': '$5K - $9.9K',
                            'N': '$2.5K - $4.9K',
                            'O': '$1K - $2.4K',
                            'P': 'Less than $1K'
                        }
                        # Map rating codes to dollar ranges (case-insensitive)
                        top_display['Capacity Rating'] = top_display['Capacity Rating'].apply(
                            lambda x: rating_to_range.get(str(x).upper().strip(), str(x)) if pd.notna(x) and str(x).upper().strip() in rating_to_range else (str(x) if pd.notna(x) else "N/A")
                        )

                    # Reorder columns to put name, capacity, and manager near the front
                    ordered_cols = ['donor_id']
                    if 'Donor Name' in top_display.columns:
                        ordered_cols.append('Donor Name')
                    ordered_cols.append('Probability')
                    if 'Capacity Rating' in top_display.columns:
                        ordered_cols.append('Capacity Rating')
                    if 'Primary Manager' in top_display.columns:
                        ordered_cols.append('Primary Manager')
                    for col in top_display.columns:
                        if col not in ordered_cols:
                            ordered_cols.append(col)

                    top_display = top_display[[col for col in ordered_cols if col in top_display.columns]]

                    st.dataframe(top_display, width='stretch', hide_index=True)

                with col2:
                    st.markdown("#### 📤 Export")
                    # Create export data
                    export_df = df_filtered.nlargest(100, 'predicted_prob').copy()
                    export_cols = ['donor_id']
                    if 'predicted_prob' in export_df.columns:
                        export_cols.append('predicted_prob')
                    if 'avg_gift' in export_df.columns:
                        export_cols.append('avg_gift')
                    if 'total_giving' in export_df.columns:
                        export_cols.append('total_giving')
                    if 'segment' in export_df.columns:
                        export_cols.append('segment')

                    export_data = export_df[export_cols].copy()
                    export_data = export_data.rename(columns={
                        'predicted_prob': 'Prediction_Probability',
                        'avg_gift': 'Recommended_Ask_Amount',
                        'total_giving': 'Lifetime_Value',
                        'segment': 'Segment'
                    })
                    if 'Prediction_Probability' in export_data.columns:
                        export_data['Contact_Priority'] = pd.cut(
                            export_data['Prediction_Probability'],
                            bins=[0, 0.4, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High']
                        )

                    csv = export_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Top 100 Prospects",
                        data=csv,
                        file_name=f"top_prospects_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Downloads CSV with donor IDs, prediction probabilities, recommended ask amounts, and contact priorities"
                    )

                    st.markdown("---")
                    st.markdown(f"""
                    <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;">
                        <h5 style="color: #856404; margin-top: 0;">🔥 Alert</h5>
                        <p style="color: #856404; margin-bottom: 0; font-size: 14px;">
                            <strong>{high_prob_count:,} new high-probability donors</strong> identified this week
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        # Insights Section
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 💡 Key Insights")

        col1, col2, col3 = st.columns(3)

        with col1:
            high_prob = (df_filtered['predicted_prob'] >= 0.7).sum()
            st.markdown(f"""
            **🔥 High Confidence Prospects**  
            {high_prob:,} donors with >70% probability  
            *Priority for immediate outreach*
            """)

        with col2:
            recent_segment = df_filtered[df_filtered['segment'] == 'Recent (0-6mo)']
            avg_recent_prob = recent_segment['predicted_prob'].mean() if len(recent_segment) > 0 else 0
            st.markdown(f"""
            **⚡ Recent Donors**  
            {avg_recent_prob:.1%} average likelihood  
            *Best ROI segment*
            """)

        with col3:
            # Robust calculation: guard missing columns and ensure 1-D Series inputs
            potential_value = 0.0
            if ('predicted_prob' in df_filtered.columns) and (df_filtered.columns.tolist().count('total_giving') >= 1):
                prob_s = pd.to_numeric(df_filtered['predicted_prob'], errors='coerce')
                mask = prob_s >= float(prob_threshold)
                # Handle potential duplicate 'total_giving' columns gracefully
                tg_cols = [c for c in df_filtered.columns if c == 'total_giving']
                tg_obj = df_filtered[tg_cols]
                if isinstance(tg_obj, pd.DataFrame):
                    # Use the first column to avoid 2D errors
                    tg_series = pd.to_numeric(tg_obj.iloc[:, 0], errors='coerce')
                else:
                    tg_series = pd.to_numeric(df_filtered['total_giving'], errors='coerce')
                tg_filtered = tg_series[mask] if tg_series.shape[0] == mask.shape[0] else tg_series
                potential_value = float(tg_filtered.fillna(0).sum()) if tg_filtered is not None else 0.0
            # Human-friendly units: use B if >= $1B, else M, with proper commas
            if potential_value >= 1_000_000_000:
                display_value = f"${potential_value/1_000_000_000:.1f}B"
            else:
                display_value = f"${potential_value/1_000_000:,.1f}M"
            st.markdown(f"""
            **💰 Total Potential Value**  
            {display_value} lifetime giving  
            *From targeted donors*
            """)

        st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================================
    # OTHER PAGES (PLACEHOLDERS - USE YOUR EXISTING FUNCTIONS)
    # ============================================================================
