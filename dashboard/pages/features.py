"""
Features page for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

import pandas as pd
import numpy as np
import warnings
import logging
import plotly.graph_objects as go

# Suppress ALL warnings at module level using simplefilter (recommended approach)
warnings.simplefilter("ignore")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')

# Suppress Streamlit logger warnings
logging.getLogger('streamlit').setLevel(logging.CRITICAL)
logging.getLogger('streamlit.runtime.caching').setLevel(logging.CRITICAL)

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


def plotly_chart_no_warnings(fig, width='stretch', config=None, **kwargs):
    """
    Display Plotly chart with all warnings suppressed (Solution 3).
    Implements proper config parameter usage (Solution 1).
    
    Args:
        fig: Plotly figure object
        width: Chart width ('stretch' or 'content')
        config: Plotly config dict. If None, uses empty dict.
        **kwargs: Additional recognized parameters
    """
    if not STREAMLIT_AVAILABLE:
        return None
    
    # CRITICAL: Always provide config, even if empty (Solution 1)
    if config is None:
        config = {}  # Empty dict prevents warnings
    
    # Filter kwargs to only recognized parameters
    recognized = {'theme', 'key', 'on_select', 'selection_mode'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in recognized}
    
    # Suppress all warnings using simplefilter (Solution 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*keyword arguments.*deprecated.*")
        
        try:
            result = st.plotly_chart(fig, width=width, config=config, **filtered_kwargs)
            return result
        except Exception as e:
            # If there's an actual error (not a warning), re-raise it
            raise e


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
    
    # Validate feature importance data before plotting
    if feature_importance is None or len(feature_importance) == 0:
        st.warning("⚠️ No feature importance data available. This may occur if the outcome column is missing or if there are insufficient valid data points.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Ensure importance scores are valid (finite and non-negative)
    feature_importance = feature_importance.copy()
    feature_importance = feature_importance[
        feature_importance['importance'].notna() & 
        np.isfinite(feature_importance['importance']) &
        (feature_importance['importance'] >= 0)
    ]
    
    if len(feature_importance) == 0:
        st.warning("⚠️ No valid feature importance scores found. All correlations were invalid (NaN or infinite).")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Sort by importance (descending) to show most important features first
    feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    
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
        xaxis_title='Importance Score (Absolute Correlation)',
        yaxis_title='',
        height=max(400, len(feature_importance) * 30),  # Dynamic height based on number of features
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#e0e0e0'),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    plotly_chart_no_warnings(fig, width='stretch')
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature Impact Analysis
    st.markdown("### 📊 Feature Impact Analysis")
    st.markdown("Select a feature to see how it relates to donor outcomes")
    
    # Get feature columns for analysis (use same logic as feature importance)
    outcome_col = 'Gave_Again_In_2024' if 'Gave_Again_In_2024' in df.columns else ('actual_gave' if 'actual_gave' in df.columns else None)
    
    if outcome_col and len(feature_importance) > 0 and 'feature' in feature_importance.columns:
        # Use top features from importance chart for selection
        # Remove duplicates to prevent showing same feature twice
        top_features = feature_importance['feature'].head(10).drop_duplicates().tolist()
        if len(top_features) > 0:
            selected_feature = st.selectbox("Select Feature to Analyze", top_features, key="feature_analysis")
        else:
            selected_feature = None
    else:
        selected_feature = None
    
    if selected_feature is not None and selected_feature in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot comparison: Gave vs Didn't Give
                df_work = df.reset_index(drop=True).copy()
                    
                # Handle duplicate columns - if accessing returns DataFrame, take first column
                outcome_data = df_work[outcome_col]
                if isinstance(outcome_data, pd.DataFrame):
                    outcome_series = pd.to_numeric(outcome_data.iloc[:, 0], errors='coerce')
                else:
                    outcome_series = pd.to_numeric(outcome_data, errors='coerce')
                
                # Handle duplicate columns for feature
                feature_data = df_work[selected_feature]
                duplicate_columns_info = None
                if isinstance(feature_data, pd.DataFrame):
                    # Multiple columns with same name - take first and warn
                    feature_series = pd.to_numeric(feature_data.iloc[:, 0], errors='coerce')
                    if len(feature_data.columns) > 1:
                        # Get actual column names if they have suffixes
                        actual_cols = list(feature_data.columns)
                        duplicate_columns_info = {
                            'count': len(actual_cols),
                            'columns': actual_cols,
                            'used_column': actual_cols[0] if actual_cols else selected_feature
                        }
                        st.warning(f"⚠️ **Duplicate columns detected**: '{selected_feature}' appears {len(actual_cols)} times ({', '.join(actual_cols[:3])}{'...' if len(actual_cols) > 3 else ''}). Using '{actual_cols[0]}' for analysis.")
                else:
                    feature_series = pd.to_numeric(feature_data, errors='coerce')
                
                # Validate feature data
                if selected_feature.lower() in ['total_giving', 'lifetime_giving', 'lifetime giving']:
                    # For giving-related fields, check for negative values (shouldn't exist)
                    negative_count = (feature_series < 0).sum()
                    if negative_count > 0:
                        st.warning(f"⚠️ **Data quality issue**: Found {negative_count:,} negative values in {selected_feature}. These will be excluded from analysis.")
                        feature_series = feature_series.clip(lower=0)
                
                # Validate network size features
                is_network_size_feature = 'network_size' in selected_feature.lower() or 'network' in selected_feature.lower()
                if is_network_size_feature:
                    # Network size should be non-negative integers (count of relationships)
                    negative_count = (feature_series < 0).sum()
                    if negative_count > 0:
                        st.warning(f"⚠️ **Data quality issue**: Found {negative_count:,} negative values in {selected_feature}. Network sizes should be non-negative. These will be excluded from analysis.")
                        feature_series = feature_series.clip(lower=0)
                    
                    # Check for non-integer values (network size should be counts)
                    non_integer_count = ((feature_series % 1) != 0).sum()
                    if non_integer_count > 0:
                        st.info(f"ℹ️ **Note**: {non_integer_count:,} non-integer values found in {selected_feature}. Network sizes are typically integer counts, but analysis will proceed with current values.")
                
                # Create aligned dataframe
                analysis_df = pd.DataFrame({
                    'feature': feature_series,
                    'outcome': outcome_series
                }).dropna()
                
                if len(analysis_df) > 0:
                    # Separate into groups
                    gave_values = analysis_df[analysis_df['outcome'] > 0]['feature'].values
                    not_gave_values = analysis_df[analysis_df['outcome'] == 0]['feature'].values
                    
                    fig_box = go.Figure()
                    
                    if len(gave_values) > 0:
                        fig_box.add_trace(go.Box(
                            y=gave_values,
                            name='Gave Again',
                            marker_color='#2ecc71',
                            boxmean='sd'
                        ))
                    
                    if len(not_gave_values) > 0:
                        fig_box.add_trace(go.Box(
                            y=not_gave_values,
                            name='Did Not Give',
                            marker_color='#e74c3c',
                            boxmean='sd'
                        ))
                    
                    fig_box.update_layout(
                        title=f'Feature Distribution by Outcome',
                        yaxis_title=selected_feature,
                        xaxis_title='Outcome',
                        height=400,
                        showlegend=True,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    plotly_chart_no_warnings(fig_box, width='stretch')
                    
                    # Summary statistics
                    if len(gave_values) > 0 and len(not_gave_values) > 0:
                        stats_df = pd.DataFrame({
                            'Group': ['Gave Again', 'Did Not Give'],
                            'Mean': [np.mean(gave_values), np.mean(not_gave_values)],
                            'Median': [np.median(gave_values), np.median(not_gave_values)],
                            'Std Dev': [np.std(gave_values), np.std(not_gave_values)],
                            'Count': [len(gave_values), len(not_gave_values)]
                        })
                        st.dataframe(stats_df, width='stretch', hide_index=True)
            
            with col2:
                # Binned analysis: Outcome rate by feature value ranges
                if len(analysis_df) > 0:
                    try:
                        # Create bins for the feature
                        feature_values = analysis_df['feature'].values
                        unique_count = len(np.unique(feature_values))
                        
                        # Check if feature has variance
                        if unique_count <= 1:
                            st.info("Feature has no variance (constant values). Cannot create binned analysis.")
                        else:
                            # Improved binning strategy
                            # For giving-related fields, use more bins and better distribution
                            is_giving_field = selected_feature.lower() in ['total_giving', 'lifetime_giving', 'lifetime giving', 'avg_gift', 'last_gift']
                            # Network size features are integer counts, need special binning
                            is_network_size_feature = 'network_size' in selected_feature.lower() or 'network' in selected_feature.lower()
                            
                            if is_network_size_feature:
                                # For network size features: use integer-based binning
                                # Network sizes are typically small integers (0, 1, 2, 3, ...)
                                # Use quantile-based bins but ensure they work well with integer values
                                try:
                                    # Calculate adaptive bin count for network features
                                    max_network_size = int(feature_values.max())
                                    unique_network_sizes = len(np.unique(feature_values))
                                    
                                    # If there are few unique values, use them directly
                                    if unique_network_sizes <= 15:
                                        # Use unique values as bins
                                        unique_sizes = np.sort(np.unique(feature_values))
                                        # Create bins from unique values
                                        bin_edges = np.concatenate([[unique_sizes[0] - 0.5], unique_sizes + 0.5])
                                        analysis_df['bin'] = pd.cut(analysis_df['feature'], bins=bin_edges, include_lowest=True, duplicates='drop')
                                    else:
                                        # Too many unique values, use quantile-based binning
                                        n_bins = min(15, max(8, len(analysis_df) // 200))
                                        bin_edges = np.quantile(feature_values, np.linspace(0, 1, n_bins + 1))
                                        bin_edges = np.unique(bin_edges)
                                        # Round to nearest integer for cleaner bins
                                        bin_edges = np.round(bin_edges).astype(int)
                                        bin_edges = np.unique(bin_edges)
                                        if len(bin_edges) < 2:
                                            bin_edges = np.array([int(feature_values.min()), int(feature_values.max())])
                                        # Add small offsets for proper binning
                                        bin_edges = np.concatenate([[bin_edges[0] - 0.5], bin_edges[1:] + 0.5])
                                        analysis_df['bin'] = pd.cut(analysis_df['feature'], bins=bin_edges, include_lowest=True, duplicates='drop')
                                except (ValueError, IndexError) as e:
                                    # Fallback: use standard quantile binning
                                    n_bins = min(12, max(6, len(analysis_df) // 200))
                                    try:
                                        bin_edges = np.quantile(feature_values, np.linspace(0, 1, n_bins + 1))
                                        bin_edges = np.unique(bin_edges)
                                        if len(bin_edges) < 2:
                                            bin_edges = np.array([feature_values.min(), feature_values.max()])
                                        analysis_df['bin'] = pd.cut(analysis_df['feature'], bins=bin_edges, include_lowest=True, duplicates='drop')
                                    except:
                                        analysis_df['bin'] = pd.cut(analysis_df['feature'], bins=n_bins, include_lowest=True, duplicates='drop')
                            elif is_giving_field:
                                # For giving amounts: use more bins (15-20) and handle skewed distribution
                                # Calculate adaptive bin count based on data size, but allow more bins
                                base_bins = max(10, min(20, len(analysis_df) // 200))
                                n_bins = base_bins
                                
                                # For giving amounts, use percentile-based binning to handle skewness
                                # Use more bins in lower ranges where most data is
                                try:
                                    # Create bins using percentiles - more granular in lower ranges
                                    # Use deciles (10%) for lower 50%, then larger bins for upper 50%
                                    lower_percentiles = np.linspace(0, 50, 6)  # 0, 10, 20, 30, 40, 50
                                    upper_percentiles = np.linspace(50, 100, max(5, n_bins - 5))  # Remaining bins for upper half
                                    all_percentiles = np.concatenate([lower_percentiles, upper_percentiles[1:]])  # Remove duplicate 50%
                                    all_percentiles = np.unique(all_percentiles)  # Remove any duplicates
                                    
                                    bin_edges = np.percentile(feature_values, all_percentiles)
                                    bin_edges = np.unique(bin_edges)  # Remove duplicate edges
                                    
                                    # Ensure we have at least 2 unique bin edges
                                    if len(bin_edges) < 2:
                                        # Fallback: use min/max
                                        bin_edges = np.array([feature_values.min(), feature_values.max()])
                                    elif len(bin_edges) < 5:
                                        # If too few edges, create more using quantiles
                                        bin_edges = np.quantile(feature_values, np.linspace(0, 1, max(6, len(bin_edges) + 3)))
                                        bin_edges = np.unique(bin_edges)
                                    
                                    analysis_df['bin'] = pd.cut(analysis_df['feature'], bins=bin_edges, include_lowest=True, duplicates='drop')
                                except (ValueError, IndexError) as e:
                                    # Fallback: use standard quantile binning
                                    try:
                                        bin_edges = np.quantile(feature_values, np.linspace(0, 1, n_bins + 1))
                                        bin_edges = np.unique(bin_edges)
                                        if len(bin_edges) < 2:
                                            bin_edges = np.array([feature_values.min(), feature_values.max()])
                                        analysis_df['bin'] = pd.cut(analysis_df['feature'], bins=bin_edges, include_lowest=True, duplicates='drop')
                                    except:
                                        # Final fallback: use fixed number of bins
                                        analysis_df['bin'] = pd.cut(analysis_df['feature'], bins=n_bins, include_lowest=True, duplicates='drop')
                            else:
                                # For other features: standard adaptive binning
                                n_bins = min(15, max(8, len(analysis_df) // 150))  # Increased from 10 to 15 max
                                
                                if unique_count > n_bins:
                                    # Create quantile-based bins for better distribution
                                    try:
                                        bin_edges = np.quantile(feature_values, np.linspace(0, 1, n_bins + 1))
                                        bin_edges = np.unique(bin_edges)  # Remove duplicates
                                        
                                        # Ensure we have at least 2 unique bin edges
                                        if len(bin_edges) < 2:
                                            # Fallback: use min/max
                                            bin_edges = np.array([feature_values.min(), feature_values.max()])
                                        
                                        analysis_df['bin'] = pd.cut(analysis_df['feature'], bins=bin_edges, include_lowest=True, duplicates='drop')
                                    except (ValueError, IndexError):
                                        # Fallback: use unique values
                                        analysis_df['bin'] = pd.cut(analysis_df['feature'], bins=n_bins, include_lowest=True, duplicates='drop')
                                else:
                                    # Use unique values if too few
                                    analysis_df['bin'] = analysis_df['feature']
                            
                            # Calculate outcome rate per bin
                            bin_stats = analysis_df.groupby('bin', observed=False).agg(
                                outcome_rate=('outcome', 'mean'),
                                count=('outcome', 'size'),
                                feature_mean=('feature', 'mean')
                            ).reset_index()
                            
                            # Diagnostic: Check why bins might be few
                            total_bins_created = len(bin_stats)
                            if is_giving_field and total_bins_created < 8:
                                # Show diagnostic info for giving fields with few bins
                                duplicate_ratio = 1 - (unique_count / len(analysis_df))
                                if duplicate_ratio > 0.3:  # More than 30% duplicate values
                                    st.info(f"ℹ️ **Binning Note**: {total_bins_created} bins created. Many duplicate values ({duplicate_ratio:.1%}) in {selected_feature} may have collapsed bins. Consider that many donors may have similar giving amounts.")
                            
                            if len(bin_stats) > 0:
                                # Sort by feature mean
                                bin_stats = bin_stats.sort_values('feature_mean').reset_index(drop=True)
                                
                                # Create bar chart
                                fig_bins = go.Figure()
                                fig_bins.add_trace(go.Bar(
                                    x=[f"{row['feature_mean']:.2f}" for _, row in bin_stats.iterrows()],
                                    y=bin_stats['outcome_rate'],
                                    marker=dict(
                                        color=bin_stats['outcome_rate'],
                                        colorscale='RdYlGn',
                                        showscale=True,
                                        colorbar=dict(title="Outcome Rate")
                                    ),
                                    text=[f"{row['outcome_rate']:.1%}" for _, row in bin_stats.iterrows()],
                                    textposition='outside',
                                    hovertemplate=(
                                        "Feature Value: %{x}<br>" +
                                        "Outcome Rate: %{y:.1%}<br>" +
                                        "Sample Size: %{customdata}<br>" +
                                        "<extra></extra>"
                                    ),
                                    customdata=bin_stats['count']
                                ))
                                
                                fig_bins.update_layout(
                                    title=f'Outcome Rate by Feature Value',
                                    xaxis_title=f'{selected_feature} (binned)',
                                    yaxis_title='Outcome Rate (Gave Again %)',
                                    height=400,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    yaxis=dict(range=[0, 1], tickformat='.0%')
                                )
                                plotly_chart_no_warnings(fig_bins, width='stretch')
                                st.caption(f"💡 **Insight**: Shows how the likelihood of giving again changes across different values of {selected_feature}. Higher bars indicate feature values associated with better outcomes.")
                            else:
                                st.info("Could not create bins for this feature.")
                    except Exception as e:
                        st.warning(f"Error creating binned analysis: {str(e)}")
                else:
                    st.info("Insufficient data for binned analysis.")
    else:
        st.info("Feature impact analysis requires outcome data and feature importance results.")

