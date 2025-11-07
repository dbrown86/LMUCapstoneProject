"""
Features page for the dashboard.
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
    from dashboard.models.metrics import get_feature_importance
except ImportError:
    # Fallbacks for testing
    def get_feature_importance(df):
        return pd.DataFrame({'feature': [], 'importance': []})


def render(df: pd.DataFrame):
    """
    Render the features page.
    
    Args:
        df: Dataframe with features and predictions
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    st.markdown('<p class="page-title">🔬 Feature Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Top predictive features and importance scores</p>', unsafe_allow_html=True)
    
    feature_importance = get_feature_importance(df)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### 📊 Feature Importance (Correlation with Target)")
    
    # Show which outcome column is being used
    outcome_col = 'Gave_Again_In_2024' if 'Gave_Again_In_2024' in df.columns else ('actual_gave' if 'actual_gave' in df.columns else None)
    if outcome_col:
        outcome_name = 'Gave_Again_In_2024' if outcome_col == 'Gave_Again_In_2024' else 'actual_gave'
        st.caption(f"💡 **Note**: Feature importance is calculated as correlation with '{outcome_name}' outcome from the multi-modal fusion model dataset. "
                   f"Higher correlation indicates stronger predictive power for identifying donors who gave again in 2024.")
    
    fig = go.Figure(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h',
        marker=dict(
            color=feature_importance['importance'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=feature_importance['importance'].round(4),
        textposition='outside'
    ))
    
    fig.update_layout(
        xaxis_title='Importance Score',
        yaxis_title='',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#e0e0e0'),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature Distributions
    st.markdown("### 📈 Feature Distribution Comparison")
    
    # Get feature columns for the distribution comparison
    feature_cols = []
    if 'actual_gave' in df.columns or 'Gave_Again_In_2024' in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ['actual_gave', 'Gave_Again_In_2024', 'donor_id', 'predicted_prob', 'Will_Give_Again_Probability', 'Legacy_Intent_Probability']]
        feature_cols = [c for c in feature_cols if c in df.columns][:10]
    
    if 'predicted_prob' in df.columns and len(feature_cols) > 0:
        selected_feature = st.selectbox("Select Feature to Analyze", feature_cols[:5])
        
        if selected_feature:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution for donors who gave vs didn't
                if 'actual_gave' in df.columns and selected_feature in df.columns:
                    # Handle duplicate indices by resetting index
                    df_work = df.reset_index(drop=True).copy()
                    
                    # Convert actual_gave to numeric, handling various formats
                    actual_gave_work = pd.to_numeric(df_work['actual_gave'], errors='coerce')
                    
                    # More inclusive filtering: treat any non-zero, non-NaN value as "gave"
                    gave_mask = (actual_gave_work > 0) & (actual_gave_work.notna())
                    not_gave_mask = (actual_gave_work == 0) | (actual_gave_work.isna())
                    
                    # Extract feature values for each group - handle potential DataFrame return
                    if gave_mask.any():
                        gave_feature_data = df_work.loc[gave_mask, selected_feature]
                        if isinstance(gave_feature_data, pd.DataFrame):
                            gave_feature = gave_feature_data.iloc[:, 0]
                        else:
                            gave_feature = gave_feature_data
                    else:
                        gave_feature = pd.Series(dtype=float)
                    
                    if not_gave_mask.any():
                        not_gave_feature_data = df_work.loc[not_gave_mask, selected_feature]
                        if isinstance(not_gave_feature_data, pd.DataFrame):
                            not_gave_feature = not_gave_feature_data.iloc[:, 0]
                        else:
                            not_gave_feature = not_gave_feature_data
                    else:
                        not_gave_feature = pd.Series(dtype=float)
                    
                    # Convert to numeric and remove NaN
                    gave_dist = pd.to_numeric(gave_feature, errors='coerce').dropna()
                    not_gave_dist = pd.to_numeric(not_gave_feature, errors='coerce').dropna()
                else:
                    gave_dist = pd.Series(dtype=float)
                    not_gave_dist = pd.Series(dtype=float)
                
                fig_dist = go.Figure()
                
                # Combine both distributions to calculate shared bin edges
                if len(gave_dist) > 0 or len(not_gave_dist) > 0:
                    all_values = pd.concat([gave_dist, not_gave_dist]) if len(gave_dist) > 0 and len(not_gave_dist) > 0 else (gave_dist if len(gave_dist) > 0 else not_gave_dist)
                    if len(all_values) > 0:
                        min_val = all_values.min()
                        max_val = all_values.max()
                        # Use shared bin edges for both histograms to ensure perfect alignment
                        num_bins = 30
                        
                        if len(gave_dist) > 0:
                            fig_dist.add_trace(go.Histogram(
                                x=gave_dist.values,
                                name='Gave',
                                opacity=0.5,
                                marker_color='#2ecc71',
                                xbins=dict(start=min_val, end=max_val, size=(max_val - min_val) / num_bins),
                                histnorm=''
                            ))
                        if len(not_gave_dist) > 0:
                            fig_dist.add_trace(go.Histogram(
                                x=not_gave_dist.values,
                                name='Did Not Give',
                                opacity=0.5,
                                marker_color='#e74c3c',
                                xbins=dict(start=min_val, end=max_val, size=(max_val - min_val) / num_bins),
                                histnorm=''
                            ))
                
                fig_dist.update_layout(
                    title=f'Distribution: {selected_feature}',
                    xaxis_title=selected_feature,
                    yaxis_title='Count',
                    barmode='overlay',
                    height=350,
                    showlegend=True
                )
                st.plotly_chart(fig_dist)
                st.caption("💡 **What this means**: Compare the distribution of this feature between donors who gave vs. didn't. Overlap suggests the feature alone isn't highly predictive.")
            
            with col2:
                # Scatter plot: feature vs prediction probability
                sample_df = df.sample(min(5000, len(df))) if len(df) > 5000 else df
                
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=sample_df[selected_feature],
                    y=sample_df['predicted_prob'],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=sample_df['predicted_prob'],
                        colorscale='RdYlGn',
                        showscale=True
                    ),
                    opacity=0.6
                ))
                fig_scatter.update_layout(
                    title=f'Feature vs. Prediction Probability',
                    xaxis_title=selected_feature,
                    yaxis_title='Predicted Probability',
                    height=350
                )
                st.plotly_chart(fig_scatter)
                st.caption("💡 **What this means**: This shows how the model uses this feature. Patterns suggest the feature's influence on predictions.")

