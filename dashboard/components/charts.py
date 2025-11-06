"""
Chart utilities for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

import warnings

# Optional Streamlit import
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


def plotly_chart_silent(fig, width='stretch'):
    """
    Display Plotly chart with all warnings suppressed.
    
    Args:
        fig: Plotly figure object
        width: Chart width ('stretch' or specific width)
    
    Returns:
        Streamlit chart component (if Streamlit available)
    """
    if not STREAMLIT_AVAILABLE:
        # Return None if Streamlit not available (for testing)
        return None
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.plotly_chart(fig, width=width)

