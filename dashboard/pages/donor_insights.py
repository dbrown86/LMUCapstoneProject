"""
Donor Insights page for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

import pandas as pd
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
    from dashboard.pages.utils import get_segment_stats
except ImportError:
    # Fallbacks for testing
    def get_segment_stats(df, use_cache=True):
        return pd.DataFrame()

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
    Render the donor insights page.
    
    Args:
        df: Dataframe with donor data and segments
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    st.markdown('<p class="page-title">💎 Donor Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Segment analysis and revenue opportunities</p>', unsafe_allow_html=True)
    
    # Segment analysis
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### 💰 Revenue Opportunity by Segment")
    
    segment_stats = get_segment_stats(df)
    
    colors_map = {
        'Recent (0-6mo)': '#4caf50',
        'Recent (6-12mo)': '#8bc34a',
        'Lapsed (1-2yr)': '#ffc107',
        'Very Lapsed (2yr+)': '#ff5722'
    }
    
    fig = go.Figure()
    for segment in segment_stats.index:
        fig.add_trace(go.Bar(
            name=str(segment),
            x=[str(segment)],
            y=[segment_stats.loc[segment, 'estimated_revenue']],
            marker_color=colors_map.get(segment, '#2196f3'),
            text=[f"${segment_stats.loc[segment, 'estimated_revenue']:,.0f}"],
            textposition='outside'
        ))
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis_title='Estimated Revenue ($)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#e0e0e0')
    )
    
    plotly_chart_silent(fig, config={'displayModeBar': True, 'displaylogo': False})
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tactical Recommendations by Segment
    st.markdown("### 📋 Tactical Recommendations by Segment")
    
    if 'segment' in df.columns:
        segments_list = df['segment'].unique()
        
        for segment in segments_list:
            segment_data = df[df['segment'] == segment]
            if len(segment_data) == 0:
                continue
            
            avg_prob = segment_data['predicted_prob'].mean() if 'predicted_prob' in segment_data.columns else 0
            avg_gift = segment_data['avg_gift'].mean() if 'avg_gift' in segment_data.columns else 0
            segment_count = len(segment_data)
            
            with st.expander(f"📌 {segment} ({segment_count:,} donors, {avg_prob:.1%} avg probability)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div style="background: #f5f5f5; padding: 15px; border-radius: 8px;">
                        <h5>Recommended Strategy</h5>
                        <ul>
                            <li><strong>Contact Frequency:</strong> {"Weekly" if segment == "Recent (0-6mo)" else "Monthly" if segment in ["Recent (6-12mo)", "Lapsed (1-2yr)"] else "Quarterly"}</li>
                            <li><strong>Suggested Ask:</strong> ${avg_gift * 1.2:,.0f} (20% above average)</li>
                            <li><strong>Best Channel:</strong> {"Email + Phone" if segment == "Recent (0-6mo)" else "Email + Mail" if segment == "Recent (6-12mo)" else "Mail" if segment == "Lapsed (1-2yr)" else "Mail + Special Offer"}</li>
                            <li><strong>Optimal Timing:</strong> {"Anytime" if segment == "Recent (0-6mo)" else "Q4 or Q1" if segment == "Recent (6-12mo)" else "Year-end" if segment == "Lapsed (1-2yr)" else "Special campaigns"}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Sample message template
                    templates = {
                        "Recent (0-6mo)": f"Thank you for your recent support! Your continued partnership means the world. We'd love to share how your ${avg_gift:,.0f} gift is making an impact. Would you consider supporting us again?",
                        "Recent (6-12mo)": f"It's been wonderful working with you! Since your last gift, we've achieved significant milestones. Your ${avg_gift:,.0f} contribution helped make this possible. Would you like to continue this impact?",
                        "Lapsed (1-2yr)": f"We miss you! It's been a while since we connected. So much has happened since your last gift. We'd love to reconnect and share how your support could make a difference again.",
                        "Very Lapsed (2yr+)": f"Reconnecting with valued supporters like you is important to us. We'd love to share our vision for the future and explore how you might want to be involved again."
                    }
                    
                    template = templates.get(segment, "We'd love to reconnect and share how your support makes a difference.")
                    
                    st.markdown(f"""
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 8px;">
                        <h5>Sample Message Template</h5>
                        <p style="font-style: italic; color: #424242;">"{template}"</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Cohort Analysis
    st.markdown("### 📊 Cohort Analysis: Segment Performance Comparison")
    
    if 'segment' in df.columns and 'predicted_prob' in df.columns and 'actual_gave' in df.columns:
        cohort_analysis = df.groupby('segment').agg({
            'predicted_prob': 'mean',
            'actual_gave': 'mean',
            'donor_id': 'count',
            'avg_gift': 'mean'
        }).round(4)
        
        # Calculate likelihood ratio vs overall average
        overall_prob = df['predicted_prob'].mean()
        cohort_analysis['Likelihood_Ratio'] = cohort_analysis['predicted_prob'] / overall_prob if overall_prob > 0 else 1.0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 Segment Likelihood Comparison")
            fig_cohort = go.Figure()
            fig_cohort.add_trace(go.Bar(
                x=cohort_analysis.index,
                y=cohort_analysis['Likelihood_Ratio'],
                marker_color=['#4caf50', '#8bc34a', '#ffc107', '#ff5722'][:len(cohort_analysis)],
                text=[f"{x:.2f}x" for x in cohort_analysis['Likelihood_Ratio']],
                textposition='outside'
            ))
            fig_cohort.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Average")
            fig_cohort.update_layout(
                title='Likelihood Ratio vs. Average',
                yaxis_title='Multiplier',
                height=350
            )
            plotly_chart_silent(fig_cohort, config={'displayModeBar': True, 'displaylogo': False})
            
            # Show key insight
            max_segment = cohort_analysis['Likelihood_Ratio'].idxmax()
            max_ratio = cohort_analysis.loc[max_segment, 'Likelihood_Ratio']
            min_segment = cohort_analysis['Likelihood_Ratio'].idxmin()
            min_ratio = cohort_analysis.loc[min_segment, 'Likelihood_Ratio']
            
            st.success(f"💡 **Key Insight**: {max_segment} donors are **{max_ratio:.2f}x more likely** to give than average, while {min_segment} donors are **{min_ratio:.2f}x** the average.")
        
        with col2:
            st.markdown("#### 📈 Movement Between Segments (Conceptual)")
            # Simulate segment movement over time
            movement_data = {
                'From Recent (0-6mo)': {'Recent (6-12mo)': 0.85, 'Lapsed (1-2yr)': 0.10, 'Very Lapsed (2yr+)': 0.05},
                'From Recent (6-12mo)': {'Recent (0-6mo)': 0.15, 'Lapsed (1-2yr)': 0.70, 'Very Lapsed (2yr+)': 0.15},
                'From Lapsed (1-2yr)': {'Recent (0-6mo)': 0.10, 'Recent (6-12mo)': 0.20, 'Very Lapsed (2yr+)': 0.70},
            }
            
            st.info("""
            **Segment Progression Pattern:**
            
            - **Recent (0-6mo)**: 85% stay recent, 10% move to lapsed, 5% become very lapsed
            - **Recent (6-12mo)**: 15% re-engage, 70% move to lapsed, 15% become very lapsed  
            - **Lapsed (1-2yr)**: 10% re-engage, 20% stay lapsed, 70% become very lapsed
            
            **Recommendation**: Focus retention efforts on Recent segments before they lapse.
            """)
    
    # Graduation Paths
    st.markdown("### 🎓 Graduation Paths: Moving Donors Up Segments")
    
    if 'segment' in df.columns:
        st.markdown("""
        **Strategic Approach to Donor Advancement:**
        
        1. **Recent (6-12mo) → Recent (0-6mo)**: Regular engagement campaigns
        2. **Lapsed (1-2yr) → Recent**: Win-back campaigns with special offers
        3. **Very Lapsed (2yr+) → Active**: Major re-engagement initiatives
        
        **Key Metrics to Track:**
        - Conversion rate by segment
        - Average time to graduate to higher segment
        - Revenue impact of successful graduations
        """)

