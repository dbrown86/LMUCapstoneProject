"""
Performance optimization utilities for the dashboard.
Includes caching, lazy loading, and performance monitoring.
"""

import functools
import time
from typing import Callable, Any, Optional
import hashlib
import pickle

# Optional Streamlit import
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


def cache_chart_figure(func: Callable) -> Callable:
    """
    Cache Plotly chart figures to avoid regeneration on every render.
    Uses function arguments as cache key.
    """
    if not STREAMLIT_AVAILABLE:
        return func
    
    @functools.lru_cache(maxsize=50)
    def _cached_func(*args, **kwargs):
        # Create a hash of arguments for caching
        try:
            # Convert args and kwargs to a hashable format
            cache_key = (args, tuple(sorted(kwargs.items())))
            return func(*args, **kwargs)
        except TypeError:
            # If arguments aren't hashable, use pickle
            cache_key = pickle.dumps((args, kwargs))
            return func(*args, **kwargs)
    
    # Use Streamlit's cache_data for better integration
    @st.cache_data(ttl=3600, show_spinner=False)
    def _streamlit_cached(*args, **kwargs):
        return func(*args, **kwargs)
    
    return _streamlit_cached


def cache_expensive_computation(ttl: int = 3600, show_spinner: bool = True):
    """
    Decorator for caching expensive computations with configurable TTL.
    
    Args:
        ttl: Time to live in seconds (default: 1 hour)
        show_spinner: Whether to show loading spinner
    """
    def decorator(func: Callable) -> Callable:
        if not STREAMLIT_AVAILABLE:
            return func
        
        @st.cache_data(ttl=ttl, show_spinner=show_spinner)
        def _cached(*args, **kwargs):
            return func(*args, **kwargs)
        
        return _cached
    return decorator


def lazy_import(module_name: str):
    """
    Lazy import utility - only import module when accessed.
    
    Usage:
        plotly = lazy_import('plotly.graph_objects')
        fig = plotly.Figure()
    """
    class LazyModule:
        def __init__(self, name):
            self._name = name
            self._module = None
        
        def _load(self):
            if self._module is None:
                import importlib
                self._module = importlib.import_module(self._name)
            return self._module
        
        def __getattr__(self, item):
            return getattr(self._load(), item)
    
    return LazyModule(module_name)


def with_progress(message: str = "Processing..."):
    """
    Context manager for showing progress during long operations.
    
    Usage:
        with with_progress("Loading data..."):
            result = expensive_operation()
    """
    class ProgressContext:
        def __init__(self, msg):
            self.msg = msg
            self.container = None
        
        def __enter__(self):
            if STREAMLIT_AVAILABLE:
                self.container = st.empty()
                self.container.info(f"⏳ {self.msg}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.container:
                self.container.empty()
            return False
    
    return ProgressContext(message)


def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure and log execution time of functions.
    Useful for performance profiling.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        if STREAMLIT_AVAILABLE and elapsed > 0.5:  # Only log if > 500ms
            st.sidebar.caption(f"⏱️ {func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper


def optimize_dataframe(df, columns_to_keep: Optional[list] = None):
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        columns_to_keep: Optional list of columns to keep (drops others)
    
    Returns:
        Optimized DataFrame
    """
    import pandas as pd
    import numpy as np
    
    if columns_to_keep:
        df = df[columns_to_keep].copy()
    
    # Downcast numeric types
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert object columns to category if they have few unique values
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If < 50% unique values
            df[col] = df[col].astype('category')
    
    return df

