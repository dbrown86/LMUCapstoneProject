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
        @st.cache_data(ttl=7200, show_spinner=False)  # 2 hour cache, no spinner
        def _get_cached(df_hash):
            return _get_unique_values_internal(df)
        # Use hash of dataframe metadata for efficient caching
        df_hash = hash((len(df), tuple(df.columns), df.shape))
        return _get_cached(df_hash)
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
    
    # CRITICAL: Remove default Streamlit sidebar navigation IMMEDIATELY
    # This must run before any other sidebar content to prevent flash
    st.sidebar.markdown("""
    <style>
    /* Hide default Streamlit page navigation immediately - highest priority */
    [data-testid="stSidebarNav"] {display: none !important; visibility: hidden !important; opacity: 0 !important; height: 0 !important; overflow: hidden !important; position: absolute !important; left: -9999px !important; width: 0 !important;}
    section[data-testid="stSidebarNav"] {display: none !important; visibility: hidden !important; opacity: 0 !important; height: 0 !important; overflow: hidden !important; position: absolute !important; left: -9999px !important; width: 0 !important;}
    nav[data-testid="stSidebarNav"] {display: none !important; visibility: hidden !important; opacity: 0 !important; height: 0 !important; overflow: hidden !important; position: absolute !important; left: -9999px !important; width: 0 !important;}
    .css-1544g2n {display: none !important; visibility: hidden !important; opacity: 0 !important; height: 0 !important; overflow: hidden !important; position: absolute !important; left: -9999px !important; width: 0 !important;}
    </style>
    <script>
    (function() {
        function removeNav() {
            ['[data-testid="stSidebarNav"]', 'section[data-testid="stSidebarNav"]', 'nav[data-testid="stSidebarNav"]', '.css-1544g2n'].forEach(s => {
                document.querySelectorAll(s).forEach(el => {
                    el.style.cssText = 'display: none !important; visibility: hidden !important; opacity: 0 !important; height: 0 !important; overflow: hidden !important; position: absolute !important; left: -9999px !important; width: 0 !important;';
                    try { el.remove(); } catch(e) {}
                });
            });
        }
        removeNav();
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', removeNav);
        }
        setInterval(removeNav, 5);
        const obs = new MutationObserver(removeNav);
        if (document.body) obs.observe(document.body, {childList: true, subtree: true});
    })();
    </script>
    """, unsafe_allow_html=True)
    
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
        ["🏠 Executive Summary", "🔬 Model Comparison", "💰 Business Impact", 
         "🔬 Features", "📈 Performance", "⚡ Take Action", "📚 About"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Model Info - Calculate from actual data
    st.sidebar.markdown('<p class="filter-header">📊 Multimodal Fusion Model</p>', unsafe_allow_html=True)
    
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
    
    # Filters removed from UI; use default (no filters, 0.5 threshold)
    regions: List[str] = []
    donor_types: List[str] = []
    segments: List[str] = []
    prob_threshold: float = 0.5

    return page, regions, donor_types, segments, prob_threshold