"""
Take Action page for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

import pandas as pd
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
    from dashboard.models.metrics import try_load_saved_metrics
except ImportError:
    # Fallbacks for testing
    def try_load_saved_metrics():
        return None


def render(df: pd.DataFrame, prob_threshold: float = 0.5):
    """
    Render the take action page.
    
    Args:
        df: Dataframe with donor data and predictions
        prob_threshold: Probability threshold for high-probability donors
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    st.markdown('<p class="page-title">⚡ Take Action</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Prioritized outreach recommendations and next steps</p>', unsafe_allow_html=True)
    
    # CRITICAL: Use 2025 prediction column - prioritize Will_Give_Again_Probability
    prob_col = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df.columns else 'predicted_prob'
    
    if prob_col not in df.columns:
        st.error("Prediction data not available. Please ensure the model has been trained.")
        return
    
    saved_meta = try_load_saved_metrics() or {}
    threshold = saved_meta.get('optimal_threshold', prob_threshold)
    
    # Categorize opportunities
    high_prob = df[df[prob_col] >= 0.7].copy()
    medium_prob = df[(df[prob_col] >= 0.4) & (df[prob_col] < 0.7)].copy()
    
    # Helper to ensure unique column names (avoid Arrow duplicate-name crash)
    def _ensure_unique_columns(df_in: pd.DataFrame) -> pd.DataFrame:
        cols = []
        seen = set()
        for c in df_in.columns:
            name = c
            i = 2
            while name in seen:
                name = f"{c}_{i}"
                i += 1
            cols.append(name)
            seen.add(name)
        df_in.columns = cols
        return df_in

    # Quick Wins: High probability recent donors
    if 'segment' in df.columns:
        quick_wins = high_prob[high_prob['segment'] == 'Recent (0-6mo)'].nlargest(50, prob_col) if len(high_prob) > 0 else pd.DataFrame()
        cultivation = medium_prob.nlargest(100, prob_col) if len(medium_prob) > 0 else pd.DataFrame()
        
        # Re-engagement: Lapsed but high predicted probability
        re_engagement = df[
            (df[prob_col] >= 0.6) & 
            (df['segment'].isin(['Lapsed (1-2yr)', 'Very Lapsed (2yr+)']))
        ].nlargest(50, prob_col) if 'segment' in df.columns else pd.DataFrame()
        
        # Display Quick Wins
        st.markdown("### 🎯 Quick Wins - High Probability Recent Donors")
        st.info("**Priority:** Contact immediately. These donors have high likelihood (>70%) and gave recently (0-6 months).")
        
        if len(quick_wins) > 0:
            base_cols_qw = [c for c in ['donor_id', prob_col, 'avg_gift', 'total_giving', 'segment'] if c in quick_wins.columns]
            quick_wins_display = quick_wins[base_cols_qw].copy()
            quick_wins_display = quick_wins_display.rename(columns={
                prob_col: 'Probability',
                'avg_gift': 'Avg Gift',
                'total_giving': 'Lifetime Value'
            })
            quick_wins_display['Probability'] = quick_wins_display['Probability'].apply(lambda x: f"{x:.1%}")
            quick_wins_display['Recommended Ask'] = quick_wins_display['Avg Gift'].apply(lambda x: f"${x*1.2:,.0f}" if pd.notna(x) else "N/A")
            quick_wins_display['Contact Priority'] = 'HIGH'
            quick_wins_display = _ensure_unique_columns(quick_wins_display)
            st.dataframe(quick_wins_display.head(20), width='stretch', hide_index=True)
            
            csv_quick = quick_wins[['donor_id', prob_col, 'avg_gift']].to_csv(index=False)
            st.download_button(
                "📥 Download Quick Wins List (50 donors)",
                csv_quick,
                file_name=f"quick_wins_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No quick wins identified. Consider lowering the probability threshold.")
        
        st.markdown("---")
        
        # Display Cultivation Targets
        st.markdown("### 🌱 Cultivation Targets - Medium Probability, High Value")
        st.info("**Priority:** Build relationships. These donors have moderate likelihood (40-70%) but high lifetime value.")
        
        if len(cultivation) > 0:
            base_cols_cult = [c for c in ['donor_id', prob_col, 'avg_gift', 'total_giving', 'segment'] if c in cultivation.columns]
            cultivation_display = cultivation[base_cols_cult].copy()
            cultivation_display = cultivation_display.rename(columns={
                prob_col: 'Probability',
                'avg_gift': 'Avg Gift',
                'total_giving': 'Lifetime Value'
            })
            cultivation_display['Probability'] = cultivation_display['Probability'].apply(lambda x: f"{x:.1%}")
            cultivation_display['Recommended Ask'] = cultivation_display['Avg Gift'].apply(lambda x: f"${x*1.15:,.0f}" if pd.notna(x) else "N/A")
            cultivation_display['Contact Priority'] = 'MEDIUM'
            cultivation_display = _ensure_unique_columns(cultivation_display)
            st.dataframe(cultivation_display.head(20), width='stretch', hide_index=True)
            
            csv_cult = cultivation[['donor_id', prob_col, 'avg_gift']].to_csv(index=False)
            st.download_button(
                "📥 Download Cultivation List (100 donors)",
                csv_cult,
                file_name=f"cultivation_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        
        # Display Re-engagement
        st.markdown("### 🔄 Re-engagement Opportunities - Lapsed but High Predicted Probability")
        st.info("**Priority:** Reconnect strategically. These donors haven't given recently but show strong likelihood (>60%).")
        
        if len(re_engagement) > 0:
            base_cols_re = [c for c in ['donor_id', prob_col, 'avg_gift', 'total_giving', 'segment'] if c in re_engagement.columns]
            reeng_display = re_engagement[base_cols_re].copy()
            reeng_display = reeng_display.rename(columns={
                prob_col: 'Probability',
                'avg_gift': 'Avg Gift',
                'total_giving': 'Lifetime Value'
            })
            reeng_display['Probability'] = reeng_display['Probability'].apply(lambda x: f"{x:.1%}")
            reeng_display['Recommended Ask'] = reeng_display['Avg Gift'].apply(lambda x: f"${x*1.1:,.0f}" if pd.notna(x) else "N/A")
            reeng_display['Contact Priority'] = 'STRATEGIC'
            reeng_display = _ensure_unique_columns(reeng_display)
            st.dataframe(reeng_display.head(20), width='stretch', hide_index=True)
            
            csv_reeng = re_engagement[['donor_id', prob_col, 'avg_gift']].to_csv(index=False)
            st.download_button(
                "📥 Download Re-engagement List (50 donors)",
                csv_reeng,
                file_name=f"re_engagement_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Action Plan Summary
    st.markdown("### 📋 Recommended Action Plan")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Compute expected response rate display safely
        if len(quick_wins) > 0 and prob_col in quick_wins.columns:
            _qw_mean = pd.to_numeric(quick_wins[prob_col], errors='coerce').mean()
            expected_rate_display = f"{_qw_mean:.1%}" if pd.notna(_qw_mean) else "High"
        else:
            expected_rate_display = "High"

        st.markdown(f"""
        <div style="background: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;">
            <h4 style="color: #2e7d32; margin-top: 0;">Week 1: Quick Wins</h4>
            <ul style="color: #424242;">
                <li>Contact {min(len(quick_wins), 20)} highest priority donors</li>
                <li>Personalized asks based on avg gift × 1.2</li>
                <li>Expected response rate: {expected_rate_display}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196f3;">
            <h4 style="color: #1565c0; margin-top: 0;">Week 2-4: Cultivation</h4>
            <ul style="color: #424242;">
                <li>Build relationships with {min(len(cultivation), 50)} high-value prospects</li>
                <li>Longer timeline, relationship-focused outreach</li>
                <li>Suggested ask: avg gift × 1.15</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: #fff3e0; padding: 20px; border-radius: 10px; border-left: 5px solid #ff9800;">
            <h4 style="color: #e65100; margin-top: 0;">Month 2+: Re-engagement</h4>
            <ul style="color: #424242;">
                <li>Strategic outreach to {min(len(re_engagement), 30)} lapsed high-probability donors</li>
                <li>Win-back campaigns, special offers</li>
                <li>Lower initial ask: avg gift × 1.1</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

