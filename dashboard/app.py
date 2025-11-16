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

# Config, styles, data, sidebar - import immediately (lightweight)
from dashboard.config import settings
from dashboard.components.styles import get_css_styles
from dashboard.components.sidebar import render_sidebar
from dashboard.data.loader import load_full_dataset

# Lazy import pages - only load when needed (performance optimization)
# This reduces initial load time by deferring heavy imports
def _lazy_import_page(page_name):
    """Lazy import page modules to reduce initial load time"""
    if page_name == "dashboard":
        from dashboard.pages import render_dashboard
        return render_dashboard
    elif page_name == "model_comparison":
        from dashboard.pages import render_model_comparison
        return render_model_comparison
    elif page_name == "business_impact":
        from dashboard.pages import render_business_impact
        return render_business_impact
    elif page_name == "features":
        from dashboard.pages import render_features
        return render_features
    elif page_name == "performance":
        from dashboard.pages import render_performance
        return render_performance
    elif page_name == "take_action":
        from dashboard.pages import render_take_action
        return render_take_action
    elif page_name == "about":
        from dashboard.pages import render_about
        return render_about
    return None


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

        # Load dataset (cached inside loader) - show progress for first load
        with st.spinner("Loading dataset..."):
            df = load_full_dataset(use_cache=True)

        # Sidebar: navigation + filters
        page, regions, donor_types, segments, prob_threshold = render_sidebar(df)

        # Route to pages with lazy loading (all within warning suppression context)
        # Only import the specific page module when needed
        try:
            if page == "🏠 Executive Summary":
                render_func = _lazy_import_page("dashboard")
                render_func(df, regions, donor_types, segments, prob_threshold)
            elif page == "🔬 Model Comparison":
                render_func = _lazy_import_page("model_comparison")
                render_func(df)
            elif page == "💰 Business Impact":
                render_func = _lazy_import_page("business_impact")
                render_func(df, prob_threshold)
            elif page == "🔬 Features":
                render_func = _lazy_import_page("features")
                render_func(df)
            elif page == "📈 Performance":
                render_func = _lazy_import_page("performance")
                render_func(df)
            elif page == "⚡ Take Action":
                render_func = _lazy_import_page("take_action")
                render_func(df, prob_threshold)
            elif page == "📚 About":
                render_func = _lazy_import_page("about")
                render_func(df)
            else:
                st.error(f"Unknown page: {page}")
        except Exception as e:
            st.error(f"Error loading page: {e}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
