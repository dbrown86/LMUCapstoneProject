"""
Chart utilities for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

import warnings
import logging
import sys
import io
import contextlib

# Suppress ALL warnings at module level using simplefilter (recommended approach)
warnings.simplefilter("ignore")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')

# Suppress Streamlit logging at module level
for logger_name in ['streamlit', 'streamlit.runtime', 'streamlit.elements', 'streamlit.logger']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)

# Optional Streamlit import
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


# Global stderr filter to catch Streamlit warnings
class StreamlitWarningFilter:
    """Filter to suppress Streamlit deprecation warnings from stderr"""
    
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.filter_keywords = [
            'keyword arguments',
            'deprecated',
            'use config instead',
            'will be removed in a future release',
            'plotly configuration options'
        ]
    
    def write(self, text):
        """Filter out deprecation warnings before writing to stderr"""
        if not isinstance(text, str):
            text = str(text)
        
        text_lower = text.lower()
        # Check if this is a deprecation warning we want to suppress
        if any(keyword in text_lower for keyword in self.filter_keywords):
            # Check if it's specifically about plotly or streamlit
            if 'plotly' in text_lower or 'streamlit' in text_lower or 'config' in text_lower:
                # Suppress this warning
                return
        
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


# Install the global stderr filter on module import
_original_stderr = sys.stderr
if not isinstance(sys.stderr, StreamlitWarningFilter):
    sys.stderr = StreamlitWarningFilter(_original_stderr)


def plotly_chart_silent(fig, width='stretch', use_container_width=None, config=None, **kwargs):
    """
    Display Plotly chart with all warnings suppressed and proper config.
    Implements Solution 1 and 3 from recommendations.
    Optimized for performance with caching.
    
    Args:
        fig: Plotly figure object
        width: Chart width ('stretch' or 'content'). Defaults to 'stretch'.
        use_container_width: Deprecated. Use width='stretch' instead.
        config: Plotly config dict. If None, uses empty dict (Solution 1).
        **kwargs: Additional arguments (will be filtered to avoid warnings)
    
    Returns:
        Streamlit chart component (if Streamlit available)
    """
    if not STREAMLIT_AVAILABLE:
        # Return None if Streamlit not available (for testing)
        return None
    
    # Handle deprecated use_container_width parameter
    if use_container_width is not None:
        width = 'stretch' if use_container_width else 'content'
    
    # CRITICAL: Always provide config parameter, even if empty (Solution 1)
    # This is what Streamlit expects in the new API
    if config is None:
        config = {}  # Empty dict instead of None prevents warnings
    
    # Filter kwargs to only include recognized st.plotly_chart parameters
    # Recognized parameters: width, config, theme, key, on_select, selection_mode
    recognized_params = {'theme', 'key', 'on_select', 'selection_mode'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in recognized_params}
    
    # Suppress warnings using context manager (Solution 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*keyword arguments.*deprecated.*")
        
        # Call st.plotly_chart with proper parameters (Solution 1)
        # ONLY pass: fig, width, config, and filtered recognized params
        # Use use_container_width=False to prevent re-rendering on every interaction
        try:
            result = st.plotly_chart(fig, width=width, config=config, **filtered_kwargs)
            return result
        except Exception as e:
            # If there's an actual error (not a warning), re-raise it
            raise e


# Alternative function name for compatibility
def plotly_chart_no_warnings(fig, width='stretch', config=None):
    """
    Alias for plotly_chart_silent for backward compatibility.
    
    Args:
        fig: Plotly figure object
        width: Chart width ('stretch' or 'content')
        config: Plotly config dict (optional)
    
    Returns:
        Streamlit chart component
    """
    return plotly_chart_silent(fig, width=width, config=config)
