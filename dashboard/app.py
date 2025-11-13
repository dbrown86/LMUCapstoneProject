#!/usr/bin/env python3
"""
Interactive Donor Prediction Dashboard (modular)
Runs entirely from modular pages and components.
"""

import sys
from pathlib import Path

import streamlit as st

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

    # Route to pages
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
