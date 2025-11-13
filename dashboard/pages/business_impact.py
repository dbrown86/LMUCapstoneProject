"""
Business Impact page for the dashboard.
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


def render(df: pd.DataFrame, prob_threshold: float):
    """
    Render the business impact page.
    
    Args:
        df: Dataframe with donor data and predictions
        prob_threshold: Probability threshold for high-probability donors
    """
    if not STREAMLIT_AVAILABLE:
        return

    # Extracted function body
    """Show concrete ROI and business outcomes"""

    st.markdown('<p class="page-title">💰 Business Impact & ROI</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Concrete revenue projections and cost-benefit analysis</p>', unsafe_allow_html=True)

    # Get metrics
    metrics = get_model_metrics(df)
    saved_meta = try_load_saved_metrics() or {}
    threshold = saved_meta.get('optimal_threshold', prob_threshold)

    # CRITICAL: Use Will_Give_Again_Probability directly if available (from generate_will_give_again_predictions.py)
    # Fall back to predicted_prob if Will_Give_Again_Probability doesn't exist
    prob_col = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df.columns else 'predicted_prob'

    # Calculate actual dollars at stake
    if prob_col in df.columns and 'actual_gave' in df.columns and 'total_giving' in df.columns and 'avg_gift' in df.columns:
            # High probability prospects
            high_prob_donors = df[df[prob_col] >= threshold].copy()
            high_prob_count = len(high_prob_donors)

            # Calculate expected conversions
            if 'actual_gave' in df.columns:
                # Baseline conversion (actual rate)
                baseline_rate = df['actual_gave'].mean() if 'actual_gave' in df.columns else 0.17
                # Handle NaN or None
                if pd.isna(baseline_rate) or baseline_rate is None:
                    baseline_rate = 0.17
                # Ensure baseline_rate is reasonable (between 0 and 1)
                if baseline_rate <= 0:
                    st.warning(f"⚠️ Baseline conversion rate is 0 or negative ({baseline_rate:.2%}). Using default 17%.")
                    baseline_rate = 0.17
                elif baseline_rate > 1:
                    st.warning(f"⚠️ Baseline conversion rate is >100% ({baseline_rate:.2%}). Using default 17%.")
                    baseline_rate = 0.17

                # Fusion model conversion (for high probability group) - use actual data
                high_prob_rate = high_prob_donors['actual_gave'].mean() if len(high_prob_donors) > 0 and 'actual_gave' in high_prob_donors.columns else None
                if high_prob_rate is None:
                    st.warning("⚠️ Actual conversion rate for high-probability donors not available. Using baseline rate estimate.")
                    high_prob_rate = baseline_rate

                # Average gift amount
                # CRITICAL: avg_gift_amount column may be corrupted (mean $0.03), use Last_Gift instead
                last_gift_col = None
                for col in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount']:
                    if col in df.columns:
                        last_gift_col = col
                        break

                if last_gift_col:
                    gift_amounts = pd.to_numeric(df[last_gift_col], errors='coerce').fillna(0).clip(lower=0)
                    # Use median for robustness against outliers
                    avg_gift_amount = gift_amounts.median() if len(gift_amounts) > 0 and gift_amounts.median() > 0 else (gift_amounts.mean() if len(gift_amounts) > 0 and gift_amounts.mean() > 0 else 500)
                else:
                    # Fallback to avg_gift if Last_Gift not available
                    avg_gift_values = pd.to_numeric(df['avg_gift'], errors='coerce').fillna(0).clip(lower=0) if 'avg_gift' in df.columns else pd.Series([500])
                    avg_gift_amount = avg_gift_values.median() if len(avg_gift_values) > 0 and avg_gift_values.median() > 0 else (avg_gift_values.mean() if len(avg_gift_values) > 0 and avg_gift_values.mean() > 0 else 500)

                # Debug: Show what we're using
                if avg_gift_amount <= 0 or avg_gift_amount < 1:
                    st.warning(f"⚠️ Average gift amount appears low ({avg_gift_amount:.2f}). Using fallback value of $500.")
                    avg_gift_amount = 500

                # Store last_gift_col for debug section
                _last_gift_col_used = last_gift_col

                # Cost assumptions
                cost_per_contact = st.sidebar.number_input("Cost per Contact ($)", 0.5, 10.0, 2.0, 0.5, help="Average cost per outreach")

                # Scenario: Contact top X% of donors
                contact_percentage = st.slider("Contact Top % of Donors", 1, 100, 20, 1)
                num_to_contact = int(len(df) * contact_percentage / 100)
                top_donors = df.nlargest(num_to_contact, prob_col)

                # Baseline scenario
                baseline_contacts = num_to_contact
                baseline_responses = int(baseline_contacts * baseline_rate)
                baseline_revenue = baseline_responses * avg_gift_amount
                baseline_cost = baseline_contacts * cost_per_contact
                baseline_roi = ((baseline_revenue - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0

                # Fusion scenario (using top predictions) - use actual response rate from top predicted donors
                fusion_contacts = num_to_contact
                if len(top_donors) > 0 and 'actual_gave' in top_donors.columns:
                    fusion_response_rate = top_donors['actual_gave'].mean()
                    # Handle NaN or None
                    if pd.isna(fusion_response_rate) or fusion_response_rate is None:
                        fusion_response_rate = baseline_rate
                else:
                    st.warning("⚠️ Actual response rate for top predicted donors not available. Using baseline rate.")
                    fusion_response_rate = baseline_rate

                # Ensure response rate is valid
                if pd.isna(fusion_response_rate) or fusion_response_rate is None:
                    fusion_response_rate = baseline_rate if not pd.isna(baseline_rate) else 0.17

                fusion_responses = int(fusion_contacts * fusion_response_rate)
                fusion_revenue = fusion_responses * avg_gift_amount
                fusion_cost = fusion_contacts * cost_per_contact
                fusion_roi = ((fusion_revenue - fusion_cost) / fusion_cost * 100) if fusion_cost > 0 else 0

                # Debug information (only show if values are suspicious)
                if baseline_revenue == 0 or fusion_revenue == 0:
                    with st.expander("🔍 Debug Information (Click to see why metrics are 0)"):
                        st.write(f"**Data Check:**")
                        st.write(f"- Donors in dataset: {len(df):,}")
                        st.write(f"- Number to contact: {num_to_contact:,}")
                        st.write(f"- Average gift amount: ${avg_gift_amount:,.2f}")
                        st.write(f"- Baseline rate: {baseline_rate:.2%}")
                        st.write(f"- Fusion response rate: {fusion_response_rate:.2%}")
                        st.write(f"- Baseline responses: {baseline_responses:,}")
                        st.write(f"- Fusion responses: {fusion_responses:,}")
                        st.write(f"- Baseline revenue: ${baseline_revenue:,.2f}")
                        st.write(f"- Fusion revenue: ${fusion_revenue:,.2f}")

                        # Check for missing columns
                        missing_cols = []
                        if prob_col not in df.columns:
                            missing_cols.append(prob_col)
                        if 'actual_gave' not in df.columns:
                            missing_cols.append('actual_gave')
                        if 'avg_gift' not in df.columns and _last_gift_col_used is None:
                            missing_cols.append('avg_gift or Last_Gift')

                        if missing_cols:
                            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                        else:
                            st.info("✅ All required columns present. Check values above for zeros or NaN.")

                # Improvement
                revenue_gain = fusion_revenue - baseline_revenue
                roi_improvement = fusion_roi - baseline_roi

                # Hero metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-left: none; height: 170px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">Revenue (Baseline)</div>
                        <div class="metric-value" style="color: white;">${baseline_revenue:,.0f}</div>
                        <div class="metric-label" style="color: white; white-space: nowrap; font-size: 11px;">{baseline_responses:,} responses</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; border: none; border-left: none; height: 170px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">Revenue (Fusion)</div>
                        <div class="metric-value" style="color: white;">${fusion_revenue:,.0f}</div>
                        <div class="metric-label" style="color: white; white-space: nowrap; font-size: 11px;">{fusion_responses:,} responses</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; border: none; border-left: none; height: 170px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">Revenue Gain</div>
                        <div class="metric-value" style="color: white;">${revenue_gain:,.0f}</div>
                        <div class="metric-label" style="color: white; white-space: nowrap; font-size: 11px;">+{(fusion_response_rate/baseline_rate - 1)*100:.0f}% response rate</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    # Format ROI improvement with comma if 1000 or higher
                    roi_display = f"+{roi_improvement:,.0f}%" if abs(roi_improvement) >= 1000 else f"+{roi_improvement:.0f}%"
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; border: none; border-left: none; height: 170px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">ROI Improvement</div>
                        <div class="metric-value" style="color: white;">{roi_display}</div>
                        <div class="metric-label" style="color: white; white-space: nowrap; font-size: 11px;">vs Baseline</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Verification Section
                with st.expander("✅ Verification & Calculation Details", expanded=False):
                    st.markdown("### 🔍 How to Verify These Calculations")
                    st.markdown("""
                    Use this section to manually verify the hero metrics and before/after chart values.
                    All calculations use the following formulas:
                    """)

                    verification_data = {
                        'Input/Calculation': [
                            '**INPUTS**',
                            'Total donors in dataset',
                            'Contact percentage (slider)',
                            'Number of donors to contact',
                            'Cost per contact ($)',
                            'Average gift amount ($)',
                            'Baseline conversion rate',
                            'Fusion response rate (top predicted donors)',
                            '',
                            '**BASELINE CALCULATIONS**',
                            'Baseline contacts',
                            'Baseline responses',
                            'Baseline revenue',
                            'Baseline cost',
                            'Baseline ROI',
                            '',
                            '**FUSION CALCULATIONS**',
                            'Fusion contacts',
                            'Fusion responses',
                            'Fusion revenue',
                            'Fusion cost',
                            'Fusion ROI',
                            '',
                            '**HERO METRICS**',
                            'Revenue (Baseline)',
                            'Revenue (Fusion)',
                            'Revenue Gain',
                            'ROI Improvement',
                        ],
                        'Value': [
                            '',
                            f"{len(df):,}",
                            f"{contact_percentage}%",
                            f"{num_to_contact:,}",
                            f"${cost_per_contact:.2f}",
                            f"${avg_gift_amount:,.2f}",
                            f"{baseline_rate:.4%}",
                            f"{fusion_response_rate:.4%}",
                            '',
                            '',
                            f"{baseline_contacts:,}",
                            f"{baseline_responses:,}",
                            f"${baseline_revenue:,.2f}",
                            f"${baseline_cost:,.2f}",
                            f"{baseline_roi:.2f}%",
                            '',
                            '',
                            f"{fusion_contacts:,}",
                            f"{fusion_responses:,}",
                            f"${fusion_revenue:,.2f}",
                            f"${fusion_cost:,.2f}",
                            f"{fusion_roi:.2f}%",
                            '',
                            '',
                            f"${baseline_revenue:,.0f}",
                            f"${fusion_revenue:,.0f}",
                            f"${revenue_gain:,.0f}",
                            f"+{roi_improvement:.0f}%",
                        ],
                        'Formula': [
                            '',
                            'Count of rows in dataframe',
                            'User-selected slider value',
                            'len(df) × contact_percentage / 100',
                            'User-selected sidebar input',
                            'Median of Last_Gift column (or avg_gift fallback)',
                            'Mean of actual_gave column (all donors)',
                            'Mean of actual_gave for top predicted donors',
                            '',
                            '',
                            'num_to_contact',
                            'int(baseline_contacts × baseline_rate)',
                            'baseline_responses × avg_gift_amount',
                            'baseline_contacts × cost_per_contact',
                            '((baseline_revenue - baseline_cost) / baseline_cost) × 100',
                            '',
                            '',
                            'num_to_contact',
                            'int(fusion_contacts × fusion_response_rate)',
                            'fusion_responses × avg_gift_amount',
                            'fusion_contacts × cost_per_contact',
                            '((fusion_revenue - fusion_cost) / fusion_cost) × 100',
                            '',
                            '',
                            'baseline_revenue (rounded)',
                            'fusion_revenue (rounded)',
                            'fusion_revenue - baseline_revenue',
                            'fusion_roi - baseline_roi',
                        ]
                    }

                    verification_df = pd.DataFrame(verification_data)
                    st.dataframe(verification_df, width='stretch', hide_index=True)

                    st.markdown("---")
                    st.markdown("### 📝 Manual Verification Steps")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("""
                        **1. Verify Baseline Revenue:**
                        - Baseline contacts: {:,}
                        - Baseline rate: {:.2%}
                        - Baseline responses: {:,} × {:.2%} = {:,}
                        - Revenue: {:,} × ${:.2f} = **${:,.2f}**
                        """.format(
                            baseline_contacts,
                            baseline_rate,
                            baseline_contacts,
                            baseline_rate,
                            baseline_responses,
                            baseline_responses,
                            avg_gift_amount,
                            baseline_revenue
                        ))

                        st.markdown("""
                        **2. Verify Fusion Revenue:**
                        - Fusion contacts: {:,}
                        - Fusion rate: {:.2%}
                        - Fusion responses: {:,} × {:.2%} = {:,}
                        - Revenue: {:,} × ${:.2f} = **${:,.2f}**
                        """.format(
                            fusion_contacts,
                            fusion_response_rate,
                            fusion_contacts,
                            fusion_response_rate,
                            fusion_responses,
                            fusion_responses,
                            avg_gift_amount,
                            fusion_revenue
                        ))

                    with col2:
                        st.markdown("""
                        **3. Verify Revenue Gain:**
                        - Fusion revenue: ${:,.2f}
                        - Baseline revenue: ${:,.2f}
                        - Gain: ${:,.2f} - ${:,.2f} = **${:,.2f}**
                        """.format(
                            fusion_revenue,
                            baseline_revenue,
                            fusion_revenue,
                            baseline_revenue,
                            revenue_gain
                        ))

                        st.markdown("""
                        **4. Verify ROI Improvement:**
                        - Fusion ROI: {:.2f}%
                        - Baseline ROI: {:.2f}%
                        - Improvement: {:.2f}% - {:.2f}% = **{:.2f}%**
                        """.format(
                            fusion_roi,
                            baseline_roi,
                            fusion_roi,
                            baseline_roi,
                            roi_improvement
                        ))

                    st.markdown("---")
                    st.markdown("### 📊 Data Source Verification")

                    data_source_info = {
                        'Data Source': [
                            'Probability Column',
                            'Outcome Column',
                            'Gift Amount Column',
                            'Top Donors Selection',
                        ],
                        'Value': [
                            prob_col,
                            'actual_gave',
                            _last_gift_col_used if _last_gift_col_used else ('avg_gift (fallback)' if 'avg_gift' in df.columns else 'N/A'),
                            f'Top {num_to_contact:,} by {prob_col}',
                        ],
                        'Sample Values': [
                            f"Range: {df[prob_col].min():.3f} - {df[prob_col].max():.3f}",
                            f"Mean: {df['actual_gave'].mean():.4f}, Sum: {df['actual_gave'].sum():,}",
                            f"Median: ${pd.to_numeric(df[_last_gift_col_used if _last_gift_col_used else ('avg_gift' if 'avg_gift' in df.columns else None)], errors='coerce').median():,.2f}" if (_last_gift_col_used or 'avg_gift' in df.columns) else "N/A",
                            f"Min prob in top: {top_donors[prob_col].min():.3f}, Max: {top_donors[prob_col].max():.3f}",
                        ]
                    }

                    source_df = pd.DataFrame(data_source_info)
                    st.dataframe(source_df, width='stretch', hide_index=True)

                    # Export button for verification data
                    if st.button("📥 Download Verification Data (CSV)"):
                        export_data = {
                            'metric': verification_data['Input/Calculation'],
                            'value': verification_data['Value'],
                            'formula': verification_data['Formula']
                        }
                        export_df = pd.DataFrame(export_data)
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"business_impact_verification_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                # Before/After Comparison
                st.markdown("### 📊 Before & After: Targeted Outreach Impact")

                # Format ROI percentages with commas if 1000 or higher
                baseline_roi_display = f"{baseline_roi:,.0f}%" if abs(baseline_roi) >= 1000 else f"{baseline_roi:.0f}%"
                fusion_roi_display = f"{fusion_roi:,.0f}%" if abs(fusion_roi) >= 1000 else f"{fusion_roi:.0f}%"
                roi_improvement_display = f"+{roi_improvement:,.0f}%" if abs(roi_improvement) >= 1000 else f"+{roi_improvement:.0f}%"

                comparison_data = {
                    'Metric': [
                        'Contacts Made',
                        'Response Rate',
                        'Expected Responses',
                        'Total Revenue',
                        'Cost of Outreach',
                        'Net Revenue',
                        'ROI'
                    ],
                    'Baseline (Old Way)': [
                        f"{baseline_contacts:,}",
                        f"{baseline_rate:.1%}",
                        f"{baseline_responses:,}",
                        f"${baseline_revenue:,.0f}",
                        f"${baseline_cost:,.0f}",
                        f"${baseline_revenue - baseline_cost:,.0f}",
                        baseline_roi_display
                    ],
                    'Fusion Model (New Way)': [
                        f"{fusion_contacts:,}",
                        f"{fusion_response_rate:.1%}",
                        f"{fusion_responses:,}",
                        f"${fusion_revenue:,.0f}",
                        f"${fusion_cost:,.0f}",
                        f"${fusion_revenue - fusion_cost:,.0f}",
                        fusion_roi_display
                    ],
                    'Improvement': [
                        "Same effort",
                        f"+{(fusion_response_rate/baseline_rate - 1)*100:.1f}%",
                        f"+{fusion_responses - baseline_responses:,}",
                        f"+${revenue_gain:,.0f}",
                        "Same cost",
                        f"+${(fusion_revenue - fusion_cost) - (baseline_revenue - baseline_cost):,.0f}",
                        roi_improvement_display
                    ]
                }

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, width='stretch', hide_index=True, use_container_width=True)

                # Visualization
                st.markdown("### 📈 Revenue Comparison")
                col1, col2 = st.columns(2)

                with col1:
                    fig_revenue = go.Figure()
                    fig_revenue.add_trace(go.Bar(
                        name='Baseline',
                        x=['Baseline', 'Fusion'],
                        y=[baseline_revenue, fusion_revenue],
                        marker_color=['#e74c3c', '#2ecc71'],
                        text=[f"${baseline_revenue:,.0f}", f"${fusion_revenue:,.0f}"],
                        textposition='outside'
                    ))
                    # Adjust y-axis to leave room for labels above bars
                    max_revenue = max(baseline_revenue, fusion_revenue)
                    fig_revenue.update_layout(
                        title='Total Revenue: Baseline vs Fusion',
                        yaxis_title='Revenue ($)',
                        yaxis=dict(range=[0, max_revenue * 1.2]),  # Add 20% padding at top
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_revenue)

                with col2:
                    # Format ROI percentages with commas if 1000 or higher
                    baseline_roi_text = f"{baseline_roi:,.0f}%" if abs(baseline_roi) >= 1000 else f"{baseline_roi:.0f}%"
                    fusion_roi_text = f"{fusion_roi:,.0f}%" if abs(fusion_roi) >= 1000 else f"{fusion_roi:.0f}%"

                    fig_roi = go.Figure()
                    fig_roi.add_trace(go.Bar(
                        name='ROI',
                        x=['Baseline', 'Fusion'],
                        y=[baseline_roi, fusion_roi],
                        marker_color=['#e74c3c', '#2ecc71'],
                        text=[baseline_roi_text, fusion_roi_text],
                        textposition='outside'
                    ))
                    # Adjust y-axis to leave room for labels above bars
                    max_roi = max(baseline_roi, fusion_roi)
                    # Ensure minimum range even if ROI is negative
                    yaxis_min = min(0, min(baseline_roi, fusion_roi) * 1.1) if min(baseline_roi, fusion_roi) < 0 else 0
                    fig_roi.update_layout(
                        title='ROI Comparison: Baseline vs Fusion',
                        yaxis_title='ROI (%)',
                        yaxis=dict(range=[yaxis_min, max_roi * 1.2]),  # Add 20% padding at top
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_roi)

                # Chart explanation below the charts
                st.markdown(f"""
                <div style="background-color: #000000; padding: 15px; border-radius: 8px; margin-top: 20px; border-left: 4px solid #667eea;">
                    <p style="margin: 0; font-size: 14px; color: #ffffff;">
                        <strong>📊 Chart Explanation:</strong> These charts compare the Baseline and Fusion model scenarios based on contacting 
                        the top <strong>{contact_percentage}%</strong> of donors (as selected in the slider above). The revenue and ROI calculations 
                        reflect the expected outcomes when using the multi-modal fusion model versus random outreach for the same number of contacts.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Segmented Opportunities - Categorized
                st.markdown("### 🎯 Targeted Opportunities by Category")

                if 'segment' in df.columns:
                    # Fix duplicate index issue by resetting index first
                    # Create a working copy with reset index to avoid duplicate label issues
                    df_work = df.reset_index(drop=True).copy()

                    # Ensure all columns are accessible and convert to arrays for safe boolean operations
                    # This avoids any index alignment issues during boolean mask creation
                    # Handle duplicate column names by taking first column if DataFrame is returned
                    try:
                        if prob_col in df_work.columns:
                            prob_col_data = df_work[prob_col]
                            # If DataFrame (duplicate columns), take first column; otherwise it's a Series
                            if isinstance(prob_col_data, pd.DataFrame):
                                prob_series = prob_col_data.iloc[:, 0]
                            else:
                                prob_series = prob_col_data
                            prob_values = pd.to_numeric(prob_series, errors='coerce').fillna(0).values
                        else:
                            prob_values = np.zeros(len(df_work))
                    except Exception:
                        prob_values = np.zeros(len(df_work))

                    try:
                        if 'segment' in df_work.columns:
                            segment_data = df_work['segment']
                            if isinstance(segment_data, pd.DataFrame):
                                segment_series = segment_data.iloc[:, 0]
                            else:
                                segment_series = segment_data
                            segment_values = segment_series.values if isinstance(segment_series, pd.Series) else np.array(segment_series).flatten()
                        else:
                            segment_values = None
                    except Exception:
                        segment_values = None

                    try:
                        if 'total_giving' in df_work.columns:
                            total_giving_data = df_work['total_giving']
                            # If DataFrame (duplicate columns), take first column; otherwise it's a Series
                            if isinstance(total_giving_data, pd.DataFrame):
                                total_giving_series = total_giving_data.iloc[:, 0]
                            else:
                                total_giving_series = total_giving_data
                            total_giving_values = pd.to_numeric(total_giving_series, errors='coerce').fillna(0).values
                        else:
                            total_giving_values = None
                    except Exception:
                        total_giving_values = None

                    # Compute quantile separately to avoid alignment issues
                    try:
                        if 'total_giving' in df_work.columns:
                            total_giving_col = df_work['total_giving']
                            # Handle duplicate column names
                            if isinstance(total_giving_col, pd.DataFrame):
                                total_giving_series = total_giving_col.iloc[:, 0]
                            else:
                                total_giving_series = total_giving_col
                            total_giving_75th = total_giving_series.quantile(0.75)
                        else:
                            total_giving_75th = 0
                    except Exception:
                        total_giving_75th = 0

                    # Quick Wins: High prob recent donors
                    if segment_values is not None:
                        quick_wins_mask = (prob_values >= 0.7) & (segment_values == 'Recent (0-6mo)')
                        quick_wins_df = df_work.loc[quick_wins_mask].copy()
                    else:
                        quick_wins_df = pd.DataFrame()

                    # Cultivation: Medium prob, high value
                    if total_giving_values is not None:
                        cultivation_mask = (
                            (prob_values >= 0.4) & 
                            (prob_values < 0.7) &
                            (total_giving_values >= total_giving_75th)
                        )
                        cultivation_df = df_work.loc[cultivation_mask].copy()
                    else:
                        cultivation_df = pd.DataFrame()

                    # Re-engagement: Lapsed but high prob
                    if segment_values is not None:
                        reeng_mask = (
                            (prob_values >= 0.6) & 
                            (np.isin(segment_values, ['Lapsed (1-2yr)', 'Very Lapsed (2yr+)']))
                        )
                        reeng_df = df_work.loc[reeng_mask].copy()
                    else:
                        reeng_df = pd.DataFrame()

                    categories_data = []

                    # Helper function to get gift amounts (use Last_Gift like in hero metrics section)
                    def get_gift_amounts(df_subset):
                        """Get gift amounts from Last_Gift column, fallback to avg_gift, then to default"""
                        last_gift_col = None
                        for col in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount']:
                            if col in df_subset.columns:
                                last_gift_col = col
                                break

                        if last_gift_col:
                            gift_amounts = pd.to_numeric(df_subset[last_gift_col], errors='coerce').fillna(0).clip(lower=0)
                            return gift_amounts
                        elif 'avg_gift' in df_subset.columns:
                            gift_amounts = pd.to_numeric(df_subset['avg_gift'], errors='coerce').fillna(0).clip(lower=0)
                            return gift_amounts
                        else:
                            return pd.Series([500] * len(df_subset))  # Default fallback

                    if len(quick_wins_df) > 0:
                        qw_gift_amounts = get_gift_amounts(quick_wins_df)
                        qw_avg_gift = qw_gift_amounts.median() if len(qw_gift_amounts) > 0 and qw_gift_amounts.median() > 0 else (qw_gift_amounts.mean() if len(qw_gift_amounts) > 0 and qw_gift_amounts.mean() > 0 else 500)
                        # Revenue = Count × Avg Gift × Response Rate (matches hero metrics formula)
                        qw_revenue = len(quick_wins_df) * qw_avg_gift * fusion_response_rate
                        categories_data.append({
                            'Category': '🎯 Quick Wins',
                            'Count': len(quick_wins_df),
                            'Avg Probability': quick_wins_df[prob_col].mean(),
                            'Avg Gift': qw_avg_gift,
                            'Estimated Revenue': qw_revenue,
                            'Priority': 1,
                            'Description': 'High prob (>70%) recent donors (0-6mo)'
                        })

                    if len(cultivation_df) > 0:
                        cult_gift_amounts = get_gift_amounts(cultivation_df)
                        cult_avg_gift = cult_gift_amounts.median() if len(cult_gift_amounts) > 0 and cult_gift_amounts.median() > 0 else (cult_gift_amounts.mean() if len(cult_gift_amounts) > 0 and cult_gift_amounts.mean() > 0 else 500)
                        # Revenue = Count × Avg Gift × Response Rate (matches hero metrics formula)
                        cult_revenue = len(cultivation_df) * cult_avg_gift * fusion_response_rate
                        categories_data.append({
                            'Category': '🌱 Cultivation Targets',
                            'Count': len(cultivation_df),
                            'Avg Probability': cultivation_df[prob_col].mean(),
                            'Avg Gift': cult_avg_gift,
                            'Estimated Revenue': cult_revenue,
                            'Priority': 2,
                            'Description': 'Medium prob (40-70%), high lifetime value'
                        })

                    if len(reeng_df) > 0:
                        reeng_gift_amounts = get_gift_amounts(reeng_df)
                        reeng_avg_gift = reeng_gift_amounts.median() if len(reeng_gift_amounts) > 0 and reeng_gift_amounts.median() > 0 else (reeng_gift_amounts.mean() if len(reeng_gift_amounts) > 0 and reeng_gift_amounts.mean() > 0 else 500)
                        # Revenue = Count × Avg Gift × Response Rate (matches hero metrics formula)
                        reeng_revenue = len(reeng_df) * reeng_avg_gift * fusion_response_rate
                        categories_data.append({
                            'Category': '🔄 Re-engagement',
                            'Count': len(reeng_df),
                            'Avg Probability': reeng_df[prob_col].mean(),
                            'Avg Gift': reeng_avg_gift,
                            'Estimated Revenue': reeng_revenue,
                            'Priority': 3,
                            'Description': 'Lapsed (>1yr) but high prob (>60%)'
                        })

                    if categories_data:
                        categories_df = pd.DataFrame(categories_data)
                        categories_df = categories_df.sort_values('Priority')

                        # Display formatted
                        display_df = categories_df[['Category', 'Count', 'Avg Probability', 'Avg Gift', 'Estimated Revenue', 'Description']].copy()
                        display_df = display_df.rename(columns={'Avg Gift': 'Median Last Gift'})
                        display_df['Avg Probability'] = display_df['Avg Probability'].apply(lambda x: f"{x:.1%}")
                        display_df['Median Last Gift'] = display_df['Median Last Gift'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "N/A")
                        display_df['Estimated Revenue'] = display_df['Estimated Revenue'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "N/A")

                        st.dataframe(display_df, width='stretch', hide_index=True)

                        # Visual comparison
                        fig_cat = go.Figure()
                        max_revenue = categories_df['Estimated Revenue'].max()
                        fig_cat.add_trace(go.Bar(
                            x=categories_df['Category'],
                            y=categories_df['Estimated Revenue'],
                            marker_color=['#4caf50', '#2196f3', '#ff9800'][:len(categories_df)],
                            text=categories_df['Estimated Revenue'].apply(lambda x: f"${x:,.0f}"),
                            textposition='outside'
                        ))
                        fig_cat.update_layout(
                            title='Revenue Potential by Category',
                            yaxis_title='Estimated Revenue ($)',
                            yaxis=dict(range=[0, max_revenue * 1.2]),  # Add 20% padding above max value for labels
                            height=400
                        )
                        st.plotly_chart(fig_cat)
    else:
        st.info("Business impact calculations require prediction and financial data.")
