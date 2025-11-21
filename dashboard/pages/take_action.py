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
            # Use Last_Gift for median gift calculation (more accurate than avg_gift)
            last_gift_col = None
            for col in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount']:
                if col in quick_wins.columns:
                    last_gift_col = col
                    break
            
            # Calculate median gift - use Last_Gift if available, otherwise fallback to avg_gift
            if last_gift_col:
                quick_wins['median_gift'] = pd.to_numeric(quick_wins[last_gift_col], errors='coerce').fillna(0).clip(lower=0)
            elif 'avg_gift' in quick_wins.columns:
                quick_wins['median_gift'] = pd.to_numeric(quick_wins['avg_gift'], errors='coerce').fillna(0).clip(lower=0)
            else:
                quick_wins['median_gift'] = 0
            
            # Get name columns if available
            name_cols = []
            for col in ['First_Name', 'first_name', 'First Name']:
                if col in quick_wins.columns:
                    name_cols.append(col)
                    break
            for col in ['Last_Name', 'last_name', 'Last Name']:
                if col in quick_wins.columns:
                    name_cols.append(col)
                    break
            
            # Select columns, excluding any other lifetime value columns to avoid duplicates
            # Note: capacity and gift officer columns are excluded from display
            base_cols_qw = ['donor_id'] + name_cols + [c for c in [prob_col, 'median_gift', 'total_giving', 'segment'] if c in quick_wins.columns]
            # Remove any duplicate lifetime-related columns (keep only total_giving)
            lifetime_cols_to_exclude = ['Lifetime_Value', 'Lifetime_Giving', 'lifetime_giving', 'Lifetime Value', 'LifetimeGiving']
            base_cols_qw = [c for c in base_cols_qw if c not in lifetime_cols_to_exclude]
            # Also exclude any columns ending with _2, _3, etc. that might be duplicates
            base_cols_qw = [c for c in base_cols_qw if not c.endswith('_2') and not c.endswith('_3')]
            
            quick_wins_display = quick_wins[base_cols_qw].copy()
            
            # Format values BEFORE renaming to avoid duplicate column issues
            # Format probability
            if prob_col in quick_wins_display.columns:
                prob_data = quick_wins_display[prob_col]
                if isinstance(prob_data, pd.DataFrame):
                    prob_data = prob_data.iloc[:, 0]
                quick_wins_display[prob_col] = prob_data.apply(lambda x: f"{x:.1%}")
            # Format median gift - handle DataFrame case
            if 'median_gift' in quick_wins_display.columns:
                median_gift_data = quick_wins_display['median_gift']
                if isinstance(median_gift_data, pd.DataFrame):
                    median_gift_data = median_gift_data.iloc[:, 0]
                median_gift_series = pd.to_numeric(median_gift_data, errors='coerce').fillna(0)
                quick_wins_display['median_gift'] = median_gift_series.apply(lambda x: f"${x:,.2f}" if x > 0 else "$0.00")
            # Format total giving - handle DataFrame case
            if 'total_giving' in quick_wins_display.columns:
                total_giving_data = quick_wins_display['total_giving']
                if isinstance(total_giving_data, pd.DataFrame):
                    total_giving_data = total_giving_data.iloc[:, 0]
                total_giving_series = pd.to_numeric(total_giving_data, errors='coerce').fillna(0)
                quick_wins_display['total_giving'] = total_giving_series.apply(lambda x: f"${x:,.2f}" if x > 0 else "$0.00")
            # Calculate Recommended Ask (use numeric value from original dataframe)
            quick_wins_display['Recommended Ask'] = quick_wins['median_gift'].apply(lambda x: f"${float(x)*1.2:,.0f}" if pd.notna(x) and float(x) > 0 else "N/A")
            
            # Now rename columns
            rename_dict = {
                prob_col: 'Probability',
                'median_gift': 'Median Gift',
                'total_giving': 'Lifetime Value',
                'First_Name': 'First Name',
                'first_name': 'First Name',
                'First Name': 'First Name',
                'Last_Name': 'Last Name',
                'last_name': 'Last Name',
                'Last Name': 'Last Name'
            }
            quick_wins_display = quick_wins_display.rename(columns=rename_dict)
            quick_wins_display['Contact Priority'] = 'HIGH'
            quick_wins_display = _ensure_unique_columns(quick_wins_display)
            # Remove any columns ending with _2, _3, etc. (duplicate columns created by _ensure_unique_columns)
            # Also remove any columns with "Lifetime" and "_2" in the name
            quick_wins_display = quick_wins_display[[c for c in quick_wins_display.columns 
                                                     if not c.endswith('_2') and not c.endswith('_3') 
                                                     and not ('Lifetime' in c and '_2' in c)
                                                     and not ('lifetime' in c.lower() and '_2' in c)]]
            st.dataframe(quick_wins_display.head(20), width='stretch', hide_index=True)
            
            # Download CSV with all displayed columns
            csv_quick = quick_wins_display.to_csv(index=False)
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
            # Use Last_Gift for median gift calculation (more accurate than avg_gift)
            last_gift_col = None
            for col in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount']:
                if col in cultivation.columns:
                    last_gift_col = col
                    break
            
            # Calculate median gift - use Last_Gift if available, otherwise fallback to avg_gift
            if last_gift_col:
                cultivation['median_gift'] = pd.to_numeric(cultivation[last_gift_col], errors='coerce').fillna(0).clip(lower=0)
            elif 'avg_gift' in cultivation.columns:
                cultivation['median_gift'] = pd.to_numeric(cultivation['avg_gift'], errors='coerce').fillna(0).clip(lower=0)
            else:
                cultivation['median_gift'] = 0
            
            # Get name columns if available
            name_cols = []
            for col in ['First_Name', 'first_name', 'First Name']:
                if col in cultivation.columns:
                    name_cols.append(col)
                    break
            for col in ['Last_Name', 'last_name', 'Last Name']:
                if col in cultivation.columns:
                    name_cols.append(col)
                    break
            
            # Select columns, excluding any other lifetime value columns to avoid duplicates
            # Note: capacity and gift officer columns are excluded from display
            base_cols_cult = ['donor_id'] + name_cols + [c for c in [prob_col, 'median_gift', 'total_giving', 'segment'] if c in cultivation.columns]
            # Remove any duplicate lifetime-related columns (keep only total_giving)
            lifetime_cols_to_exclude = ['Lifetime_Value', 'Lifetime_Giving', 'lifetime_giving', 'Lifetime Value', 'LifetimeGiving']
            base_cols_cult = [c for c in base_cols_cult if c not in lifetime_cols_to_exclude]
            # Also exclude any columns ending with _2, _3, etc. that might be duplicates
            base_cols_cult = [c for c in base_cols_cult if not c.endswith('_2') and not c.endswith('_3')]
            
            cultivation_display = cultivation[base_cols_cult].copy()
            
            # Format values BEFORE renaming to avoid duplicate column issues
            # Format probability
            if prob_col in cultivation_display.columns:
                prob_data = cultivation_display[prob_col]
                if isinstance(prob_data, pd.DataFrame):
                    prob_data = prob_data.iloc[:, 0]
                cultivation_display[prob_col] = prob_data.apply(lambda x: f"{x:.1%}")
            # Format median gift - handle DataFrame case
            if 'median_gift' in cultivation_display.columns:
                median_gift_data = cultivation_display['median_gift']
                if isinstance(median_gift_data, pd.DataFrame):
                    median_gift_data = median_gift_data.iloc[:, 0]
                median_gift_series = pd.to_numeric(median_gift_data, errors='coerce').fillna(0)
                cultivation_display['median_gift'] = median_gift_series.apply(lambda x: f"${x:,.2f}" if x > 0 else "$0.00")
            # Format total giving - handle DataFrame case
            if 'total_giving' in cultivation_display.columns:
                total_giving_data = cultivation_display['total_giving']
                if isinstance(total_giving_data, pd.DataFrame):
                    total_giving_data = total_giving_data.iloc[:, 0]
                total_giving_series = pd.to_numeric(total_giving_data, errors='coerce').fillna(0)
                cultivation_display['total_giving'] = total_giving_series.apply(lambda x: f"${x:,.2f}" if x > 0 else "$0.00")
            # Calculate Recommended Ask (use numeric value from original dataframe)
            cultivation_display['Recommended Ask'] = cultivation['median_gift'].apply(lambda x: f"${float(x)*1.15:,.0f}" if pd.notna(x) and float(x) > 0 else "N/A")
            
            # Now rename columns
            rename_dict = {
                prob_col: 'Probability',
                'median_gift': 'Median Gift',
                'total_giving': 'Lifetime Value',
                'First_Name': 'First Name',
                'first_name': 'First Name',
                'First Name': 'First Name',
                'Last_Name': 'Last Name',
                'last_name': 'Last Name',
                'Last Name': 'Last Name',
            }
            cultivation_display = cultivation_display.rename(columns=rename_dict)
            cultivation_display['Contact Priority'] = 'MEDIUM'
            cultivation_display = _ensure_unique_columns(cultivation_display)
            # Remove any columns ending with _2, _3, etc. (duplicate columns created by _ensure_unique_columns)
            # Also remove any columns with "Lifetime" and "_2" in the name
            cultivation_display = cultivation_display[[c for c in cultivation_display.columns 
                                                       if not c.endswith('_2') and not c.endswith('_3') 
                                                       and not ('Lifetime' in c and '_2' in c)
                                                       and not ('lifetime' in c.lower() and '_2' in c)]]
            st.dataframe(cultivation_display.head(20), width='stretch', hide_index=True)
            
            # Download CSV with all displayed columns
            csv_cult = cultivation_display.to_csv(index=False)
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
            # Use Last_Gift for median gift calculation (more accurate than avg_gift)
            last_gift_col = None
            for col in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount']:
                if col in re_engagement.columns:
                    last_gift_col = col
                    break
            
            # Calculate median gift - use Last_Gift if available, otherwise fallback to avg_gift
            if last_gift_col:
                re_engagement['median_gift'] = pd.to_numeric(re_engagement[last_gift_col], errors='coerce').fillna(0).clip(lower=0)
            elif 'avg_gift' in re_engagement.columns:
                re_engagement['median_gift'] = pd.to_numeric(re_engagement['avg_gift'], errors='coerce').fillna(0).clip(lower=0)
            else:
                re_engagement['median_gift'] = 0
            
            # Get name columns if available
            name_cols = []
            for col in ['First_Name', 'first_name', 'First Name']:
                if col in re_engagement.columns:
                    name_cols.append(col)
                    break
            for col in ['Last_Name', 'last_name', 'Last Name']:
                if col in re_engagement.columns:
                    name_cols.append(col)
                    break
            
            # Select columns, excluding any other lifetime value columns to avoid duplicates
            # Note: capacity and gift officer columns are excluded from display
            base_cols_re = ['donor_id'] + name_cols + [c for c in [prob_col, 'median_gift', 'total_giving', 'segment'] if c in re_engagement.columns]
            # Remove any duplicate lifetime-related columns (keep only total_giving)
            lifetime_cols_to_exclude = ['Lifetime_Value', 'Lifetime_Giving', 'lifetime_giving', 'Lifetime Value', 'LifetimeGiving']
            base_cols_re = [c for c in base_cols_re if c not in lifetime_cols_to_exclude]
            # Also exclude any columns ending with _2, _3, etc. that might be duplicates
            base_cols_re = [c for c in base_cols_re if not c.endswith('_2') and not c.endswith('_3')]
            
            reeng_display = re_engagement[base_cols_re].copy()
            
            # Format values BEFORE renaming to avoid duplicate column issues
            # Format probability
            if prob_col in reeng_display.columns:
                prob_data = reeng_display[prob_col]
                if isinstance(prob_data, pd.DataFrame):
                    prob_data = prob_data.iloc[:, 0]
                reeng_display[prob_col] = prob_data.apply(lambda x: f"{x:.1%}")
            # Format median gift - handle DataFrame case
            if 'median_gift' in reeng_display.columns:
                median_gift_data = reeng_display['median_gift']
                if isinstance(median_gift_data, pd.DataFrame):
                    median_gift_data = median_gift_data.iloc[:, 0]
                median_gift_series = pd.to_numeric(median_gift_data, errors='coerce').fillna(0)
                reeng_display['median_gift'] = median_gift_series.apply(lambda x: f"${x:,.2f}" if x > 0 else "$0.00")
            # Format total giving - handle DataFrame case
            if 'total_giving' in reeng_display.columns:
                total_giving_data = reeng_display['total_giving']
                if isinstance(total_giving_data, pd.DataFrame):
                    total_giving_data = total_giving_data.iloc[:, 0]
                total_giving_series = pd.to_numeric(total_giving_data, errors='coerce').fillna(0)
                reeng_display['total_giving'] = total_giving_series.apply(lambda x: f"${x:,.2f}" if x > 0 else "$0.00")
            # Calculate Recommended Ask (use numeric value from original dataframe)
            reeng_display['Recommended Ask'] = re_engagement['median_gift'].apply(lambda x: f"${float(x)*1.1:,.0f}" if pd.notna(x) and float(x) > 0 else "N/A")
            
            # Now rename columns
            rename_dict = {
                prob_col: 'Probability',
                'median_gift': 'Median Gift',
                'total_giving': 'Lifetime Value',
                'First_Name': 'First Name',
                'first_name': 'First Name',
                'First Name': 'First Name',
                'Last_Name': 'Last Name',
                'last_name': 'Last Name',
                'Last Name': 'Last Name',
            }
            reeng_display = reeng_display.rename(columns=rename_dict)
            reeng_display['Contact Priority'] = 'STRATEGIC'
            reeng_display = _ensure_unique_columns(reeng_display)
            # Remove any columns ending with _2, _3, etc. (duplicate columns created by _ensure_unique_columns)
            # Also remove any columns with "Lifetime" and "_2" in the name
            reeng_display = reeng_display[[c for c in reeng_display.columns 
                                          if not c.endswith('_2') and not c.endswith('_3') 
                                          and not ('Lifetime' in c and '_2' in c)
                                          and not ('lifetime' in c.lower() and '_2' in c)]]
            st.dataframe(reeng_display.head(20), width='stretch', hide_index=True)
            
            # Download CSV with all displayed columns
            csv_reeng = reeng_display.to_csv(index=False)
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
        <div style="background: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50; height: 200px; display: flex; flex-direction: column; overflow: hidden;">
            <h4 style="color: #2e7d32; margin-top: 0; margin-bottom: 10px; font-size: 16px;">Week 1: Quick Wins</h4>
            <ul style="color: #424242; margin: 0; padding-left: 20px; flex-grow: 1; font-size: 13px; line-height: 1.4;">
                <li style="margin-bottom: 6px;">Contact {min(len(quick_wins), 20)} highest priority donors</li>
                <li style="margin-bottom: 6px;">Personalized asks based on median gift × 1.2</li>
                <li style="margin-bottom: 6px;">Expected response rate: {expected_rate_display}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196f3; height: 200px; display: flex; flex-direction: column; overflow: hidden;">
            <h4 style="color: #1565c0; margin-top: 0; margin-bottom: 10px; font-size: 16px;">Week 2-4: Cultivation</h4>
            <ul style="color: #424242; margin: 0; padding-left: 20px; flex-grow: 1; font-size: 13px; line-height: 1.4;">
                <li style="margin-bottom: 6px;">Build relationships with {min(len(cultivation), 50)} high-value prospects</li>
                <li style="margin-bottom: 6px;">Longer timeline, relationship-focused outreach</li>
                <li style="margin-bottom: 6px;">Suggested ask: median gift × 1.15</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: #fff3e0; padding: 20px; border-radius: 10px; border-left: 5px solid #ff9800; height: 200px; display: flex; flex-direction: column; overflow: hidden;">
            <h4 style="color: #e65100; margin-top: 0; margin-bottom: 10px; font-size: 16px;">Month 2+: Re-engagement</h4>
            <ul style="color: #424242; margin: 0; padding-left: 20px; flex-grow: 1; font-size: 13px; line-height: 1.4;">
                <li style="margin-bottom: 6px;">Strategic outreach to {min(len(re_engagement), 30)} lapsed high-probability donors</li>
                <li style="margin-bottom: 6px;">Win-back campaigns, special offers</li>
                <li style="margin-bottom: 6px;">Lower initial ask: median gift × 1.1</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

