#!/usr/bin/env python3
"""
Interactive Donor Prediction Dashboard (modular)
Runs entirely from modular pages and components.
"""

import sys
import warnings
from pathlib import Path

# ==========================================
# STEP 1: Suppress ALL warnings at Python level FIRST
# ==========================================
warnings.simplefilter("ignore")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

# ==========================================
# STEP 2: Install stderr filter BEFORE any other imports!
# This catches Streamlit's warnings that are printed directly to stderr
# ==========================================
class StreamlitWarningFilter:
    """Filter to suppress Streamlit deprecation warnings from stderr"""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.filter_keywords = [
            'keyword arguments', 'deprecated', 'use config instead',
            'will be removed in a future release', 'plotly configuration',
            'use_container_width', 'replace `use_container_width` with `width`',
            'please replace'
        ]
    
    def write(self, text):
        """Filter out deprecation warnings before writing to stderr"""
        if not isinstance(text, str):
            text = str(text)
        text_lower = text.lower()
        # Check if this is a deprecation warning we want to suppress
        if any(keyword in text_lower for keyword in self.filter_keywords):
            if any(word in text_lower for word in ['plotly', 'streamlit', 'config', 'width', 'container']):
                return  # Suppress this warning
        # Write everything else
        try:
            self.original_stderr.write(text)
        except:
            pass
    
    def flush(self):
        """Flush the original stderr"""
        try:
            self.original_stderr.flush()
        except:
            pass
    
    def __getattr__(self, name):
        """Delegate all other attributes to original stderr"""
        return getattr(self.original_stderr, name)

# Install stderr filter BEFORE importing anything that might trigger warnings
_original_stderr = sys.stderr
if not isinstance(sys.stderr, StreamlitWarningFilter):
    sys.stderr = StreamlitWarningFilter(_original_stderr)

# ==========================================
# STEP 3: Import logging and suppress all Streamlit loggers
# ==========================================
import logging

# Suppress ALL logging from Streamlit before it initializes
for logger_name in ['streamlit', 'streamlit.runtime', 'streamlit.runtime.caching', 
                    'streamlit.elements', 'streamlit.logger', 'streamlit.warning',
                    'streamlit.deprecation', 'streamlit.runtime.scriptrunner']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    # Disable all handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

# Also set root logger to ERROR
logging.getLogger().setLevel(logging.ERROR)

# ==========================================
# STEP 4: Set environment variables BEFORE importing Streamlit
# ==========================================
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'error'

# ==========================================
# STEP 5: NOW import streamlit (after all suppression is active)
# ==========================================
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import streamlit as st

# ==========================================
# STEP 6: Apply Streamlit-specific warning suppression (Solution 2)
# ==========================================
# Method 1: Set the deprecation warning flag if it exists
try:
    import streamlit.elements.plotly_chart as plotly_chart_module
    if hasattr(plotly_chart_module, '_DEPRECATION_WARNING_SHOWN'):
        plotly_chart_module._DEPRECATION_WARNING_SHOWN = True
except:
    pass

# Method 2: Monkey-patch st.warning to filter out deprecation warnings
try:
    _original_st_warning = st.warning
    def filtered_warning(message, *args, **kwargs):
        """Filter out Plotly deprecation warnings"""
        if isinstance(message, str):
            msg_lower = message.lower()
            # Filter out specific deprecation warnings
            if any(phrase in msg_lower for phrase in [
                'keyword arguments have been deprecated',
                'use config instead',
                'use_container_width',
                'plotly configuration'
            ]):
                return  # Suppress this warning
        return _original_st_warning(message, *args, **kwargs)
    st.warning = filtered_warning
except:
    pass

# Method 3: Monkey-patch st.logger if it exists
try:
    import streamlit.logger as st_logger
    if hasattr(st_logger, 'get_logger'):
        _original_get_logger = st_logger.get_logger
        def filtered_get_logger(name):
            logger = _original_get_logger(name)
            logger.setLevel(logging.CRITICAL)
            return logger
        st_logger.get_logger = filtered_get_logger
except:
    pass

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Config, styles, data, sidebar, and pages
from dashboard.config import settings
from dashboard.components.styles import get_css_styles
from dashboard.components.sidebar import render_sidebar
from dashboard.data.loader import load_full_dataset
from dashboard.pages import (
    render_dashboard,
    render_model_comparison,
    render_business_impact,
    render_donor_insights,
    render_features,
    render_predictions,
    render_performance,
    render_take_action,
)


def main() -> None:
    # ==========================================
    # Wrap entire main function in warning suppression context
    # ==========================================
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("ignore", category=DeprecationWarning)
        
        # Page config - disable automatic page discovery
        page_config = settings.PAGE_CONFIG.copy()
        page_config['menu_items'] = None  # Remove default menu items
        st.set_page_config(**page_config)

        # Global styles
        st.markdown(get_css_styles(), unsafe_allow_html=True)

        # Load dataset (cached inside loader)
        df = load_full_dataset(use_cache=True)

        # Sidebar: navigation + filters
        page, regions, donor_types, segments, prob_threshold = render_sidebar(df)

        # Route to pages (all within warning suppression context)
        if page == "🏠 Executive Summary":
            render_dashboard(df, regions, donor_types, segments, prob_threshold)
        elif page == "🔬 Model Comparison":
            render_model_comparison(df)
        elif page == "💰 Business Impact":
            render_business_impact(df, prob_threshold)
        elif page == "💎 Donor Insights":
            render_donor_insights(df)
        elif page == "🔬 Features":
            render_features(df)
        elif page == "🎲 Predictions":
            render_predictions(df)
        elif page == "📈 Performance":
            render_performance(df)
        elif page == "⚡ Take Action":
            render_take_action(df, prob_threshold)
        else:
            st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
