"""
Helper utilities for page modules.
Extracted from alternate_dashboard.py for shared use across pages.
"""

import pandas as pd
import numpy as np

# Optional Streamlit import
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


def filter_dataframe(df: pd.DataFrame, regions: list, donor_types: list, segments: list, use_cache: bool = True) -> pd.DataFrame:
    """
    Filter dataframe by regions, donor_types, and segments (cached for performance).
    
    Args:
        df: Dataframe to filter
        regions: List of regions to include
        donor_types: List of donor types to include
        segments: List of segments to include
        use_cache: If True and Streamlit is available, use Streamlit caching
    
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if use_cache and STREAMLIT_AVAILABLE:
        @st.cache_data(ttl=3600, show_spinner=False, max_entries=50)  # Cache for 1 hour, up to 50 filter combinations
        def _filter_cached(df_hash, regions_tuple, donor_types_tuple, segments_tuple):
            # Use hash of dataframe shape/columns as key instead of full dataframe
            return _filter_dataframe_internal(df, list(regions_tuple), list(donor_types_tuple), list(segments_tuple))
        # Create hash from dataframe metadata for efficient caching
        df_hash = hash((len(df), tuple(df.columns), df.shape))
        return _filter_cached(df_hash, tuple(regions) if regions else (), tuple(donor_types) if donor_types else (), tuple(segments) if segments else ())
    else:
        return _filter_dataframe_internal(df, regions, donor_types, segments)


def _filter_dataframe_internal(df: pd.DataFrame, regions: list, donor_types: list, segments: list) -> pd.DataFrame:
    """Internal function to filter dataframe (without caching decorator)."""
    df_filtered = df.copy()
    if regions:
        df_filtered = df_filtered[df_filtered['region'].isin(regions)]
    if donor_types:
        df_filtered = df_filtered[df_filtered['donor_type'].isin(donor_types)]
    if segments:
        df_filtered = df_filtered[df_filtered['segment'].isin(segments)]
    return df_filtered


def get_segment_stats(df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """
    Get statistics by segment (cached for performance).
    
    Args:
        df: Dataframe with segment column
        use_cache: If True and Streamlit is available, use Streamlit caching
    
    Returns:
        pd.DataFrame: Statistics by segment
    """
    if use_cache and STREAMLIT_AVAILABLE:
        @st.cache_data(ttl=3600, show_spinner=False)
        def _get_cached(df_hash):
            return _get_segment_stats_internal(df)
        df_hash = hash((len(df), tuple(df.columns), df.shape))
        return _get_cached(df_hash)
    else:
        return _get_segment_stats_internal(df)


def _get_segment_stats_internal(df: pd.DataFrame) -> pd.DataFrame:
    """Internal function to get segment stats (without caching decorator)."""
    # Drop duplicate columns to avoid pandas aggregation errors
    try:
        df = df.loc[:, ~df.columns.duplicated()].copy()
    except Exception:
        df = df.copy()

    if 'segment' not in df.columns:
        return pd.DataFrame()
    
    # Ensure numeric types where needed
    for col in ['predicted_prob', 'total_giving', 'avg_gift']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Aggregate safely
    group_obj = df.groupby('segment', observed=False)
    segment_stats = group_obj['donor_id'].count().to_frame('donor_id')
    if 'predicted_prob' in df.columns:
        segment_stats['predicted_prob'] = group_obj['predicted_prob'].mean()
    if 'total_giving' in df.columns:
        segment_stats['total_giving'] = group_obj['total_giving'].sum()
    if 'avg_gift' in df.columns:
        segment_stats['avg_gift'] = group_obj['avg_gift'].mean()
    segment_stats = segment_stats.round(2)
    
    # Calculate estimated revenue
    if set(['predicted_prob', 'avg_gift']).issubset(segment_stats.columns):
        segment_stats['estimated_revenue'] = (
            segment_stats['donor_id'].astype(float)
            * segment_stats['predicted_prob'].astype(float)
            * segment_stats['avg_gift'].astype(float)
        )
    else:
        segment_stats['estimated_revenue'] = 0.0
    
    return segment_stats


def get_value_counts(series: pd.Series, use_cache: bool = True) -> pd.Series:
    """
    Get value counts (cached for performance).
    
    Args:
        series: Series to count
        use_cache: If True and Streamlit is available, use Streamlit caching
    
    Returns:
        pd.Series: Value counts
    """
    if use_cache and STREAMLIT_AVAILABLE:
        @st.cache_data
        def _get_cached():
            return series.value_counts()
        return _get_cached()
    else:
        return series.value_counts()

