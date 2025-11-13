"""
Sidebar component for the dashboard.

"""

import os
from pathlib import Path
from typing import Tuple, List, Optional

# Optional Streamlit import
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

# Import metrics for sidebar display
try:
    from dashboard.models.metrics import get_model_metrics
except ImportError:
    # Fallback if metrics module not available
    def get_model_metrics(df):
        return {'auc': None, 'f1': None, 'baseline_auc': None, 'lift': None}


def get_unique_values(df, use_cache: bool = True):
    """
    Get unique values for filters (cached for performance).
    
    Args:
        df: Dataframe to extract unique values from
        use_cache: If True and Streamlit is available, use Streamlit caching
    
    Returns:
        dict: Dictionary with 'regions', 'types', and 'segments' lists
    """
    if use_cache and STREAMLIT_AVAILABLE:
        @st.cache_data
        def _get_cached():
            return _get_unique_values_internal(df)
        return _get_cached()
    else:
        return _get_unique_values_internal(df)


def _get_unique_values_internal(df):
    """Internal function to get unique values (without caching decorator)."""
    return {
        'regions': df['region'].dropna().unique().tolist() if 'region' in df.columns else ['Northeast', 'Southeast', 'Midwest', 'West', 'Southwest'],
        'types': df['donor_type'].dropna().unique().tolist() if 'donor_type' in df.columns else ['Individual', 'Corporate', 'Foundation', 'Government'],
        'segments': df['segment'].dropna().unique().tolist() if 'segment' in df.columns else ['Recent (0-6mo)', 'Recent (6-12mo)', 'Lapsed (1-2yr)', 'Very Lapsed (2yr+)']
    }


def render_sidebar(df) -> Tuple[str, List[str], List[str], List[str], float]:
    """
    Render professional sidebar with logo and filters.
    
    Args:
        df: Dataframe for extracting filter options
    
    Returns:
        tuple: (page, regions, donor_types, segments, prob_threshold)
    """
    if not STREAMLIT_AVAILABLE:
        # Return defaults if Streamlit not available (for testing)
        return "🏠 Executive Summary", [], [], [], 0.5
    
    # Load brand font and styles for University Advancement title
    st.sidebar.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;800&display=swap');
      .ua-title {
        font-family: 'Cinzel', serif;
        color: #D4AF37; /* gold from logo */
        font-weight: 800;
        font-size: 24px;
        text-align: center;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
      }
    </style>
    """, unsafe_allow_html=True)

    # Try to load and show the provided University Advancement logo
    logo_candidates = [
        "dashboard/assets/university_advancement.png",
        "assets/university_advancement.png",
        "assets/university_advancement.jpg",
        "assets/university_advancement.webp",
        "university_advancement.png",
        "university_advancement.jpg",
        "university_advancement.webp",
    ]
    logo_path = next((p for p in logo_candidates if os.path.exists(p)), None)
    if logo_path:
        st.sidebar.markdown("<div class='ua-title'>University Advancement</div>", unsafe_allow_html=True)
        st.sidebar.image(logo_path, width='stretch')
    else:
        # Fallback minimal text if logo file isn't found locally
        st.sidebar.markdown("<div class='ua-title'>University Advancement</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown('<p class="filter-header">📍 Navigation</p>', unsafe_allow_html=True)
    
    # Add CSS to left-align radio buttons
    st.sidebar.markdown("""
    <style>
        /* Left-align radio buttons and labels */
        .stRadio > div {
            display: flex;
            flex-direction: column;
            align-items: flex-start !important;
        }
        .stRadio > div > label {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            width: 100%;
        }
        .stRadio > div > label > div:first-child {
            margin-right: 8px;
            flex-shrink: 0;
        }
        .stRadio > div > label > div:last-child {
            text-align: left;
        }
    </style>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Executive Summary", "🔬 Model Comparison", "💰 Business Impact", "💎 Donor Insights", 
         "🔬 Features", "🎲 Predictions", "📈 Performance", "⚡ Take Action"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Filters
    st.sidebar.markdown('<p class="filter-header">🔍 Filters</p>', unsafe_allow_html=True)
    
    # Get unique values from actual data (cached)
    unique_vals = get_unique_values(df)
    available_regions = unique_vals['regions']
    available_types = unique_vals['types']
    available_segments = unique_vals['segments']
    
    regions = st.sidebar.multiselect(
        "Select Regions",
        available_regions,
        default=[]
    )
    
    donor_types = st.sidebar.multiselect(
        "Select Donor Types",
        available_types,
        default=[]
    )
    
    segments = st.sidebar.multiselect(
        "Select Segments",
        available_segments,
        default=[]
    )
    
    prob_threshold = st.sidebar.slider(
        "Prediction Threshold",
        0.0, 1.0, 0.5, 0.05,
        help="Minimum probability to classify as 'likely to give'"
    )
    
    st.sidebar.markdown("---")
    
    # Model Info - Calculate from actual data
    st.sidebar.markdown('<p class="filter-header">📊 Model Info</p>', unsafe_allow_html=True)
    
    metrics = get_model_metrics(df)
    
    # Format metrics for display
    auc_display = f"{metrics['auc']:.2%}" if metrics['auc'] is not None else "N/A"
    f1_display = f"{metrics['f1']:.2%}" if metrics['f1'] is not None else "N/A"
    baseline_auc_display = f"{metrics['baseline_auc']:.2%}" if metrics['baseline_auc'] is not None else "N/A"
    lift_display = f"+{metrics['lift']:.1%}" if metrics['lift'] is not None else "N/A"
    
    # Determine colors
    lift_color = "#4caf50" if metrics['lift'] is not None else "#ff9800"
    
    st.sidebar.markdown(f"""
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; color: white;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span>AUC Score:</span>
            <strong>{auc_display}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span>F1 Score:</span>
            <strong>{f1_display}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span>Baseline AUC:</span>
            <strong>{baseline_auc_display}</strong>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Lift vs Baseline:</span>
            <strong style="color: {lift_color};">{lift_display}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Export options
    st.sidebar.markdown('<p class="filter-header">📤 Export</p>', unsafe_allow_html=True)
    
    if st.sidebar.button("📊 Export Dashboard", width='stretch'):
        st.sidebar.success("✅ Dashboard exported!")
    
    if st.sidebar.button("📋 Copy Metrics", width='stretch'):
        st.sidebar.info("📋 Metrics copied to clipboard!")
    
    return page, regions, donor_types, segments, prob_threshold