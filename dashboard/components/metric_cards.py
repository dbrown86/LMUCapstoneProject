"""
Metric card components for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

# Optional Streamlit import
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


def render_metric_card(
    label: str,
    value: str,
    icon: str = "",
    gradient: str = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    border_color: str = "#667eea",
    height: str = "auto",
    subtitle: str = None
) -> None:
    """
    Render a metric card with custom styling.
    
    Args:
        label: Metric label (e.g., "AUC Score")
        value: Metric value (e.g., "0.85" or "85%")
        icon: Optional emoji icon
        gradient: CSS gradient string for background
        border_color: Color for left border
        height: Card height (e.g., "170px", "auto")
        subtitle: Optional subtitle text below value
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    style_attrs = f"background: {gradient}; color: white; border: none; border-left: none;"
    if height != "auto":
        style_attrs += f" height: {height};"
    style_attrs += " display: flex; flex-direction: column; justify-content: space-between;"
    
    subtitle_html = f'<div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">{subtitle}</div>' if subtitle else ""
    
    card_html = f"""
    <div class="metric-card" style="{style_attrs}">
        <div class="metric-label" style="color: white; white-space: nowrap; font-size: 12px;">{icon} {label}</div>
        <div class="metric-value" style="color: white;">{value}</div>
        {subtitle_html}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def render_simple_metric_card(
    label: str,
    value: str,
    border_color: str = "#667eea"
) -> None:
    """
    Render a simple metric card with default styling.
    
    Args:
        label: Metric label
        value: Metric value
        border_color: Color for left border
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    card_html = f"""
    <div class="metric-card" style="border-left-color: {border_color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)

