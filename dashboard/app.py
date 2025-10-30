#!/usr/bin/env python3
"""
Interactive Donor Prediction Dashboard
Main Streamlit application for the LMU Capstone Project
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import dashboard pages
from pages import overview, search, predictions, explanations, analytics

# Configure Streamlit page
st.set_page_config(
    page_title="🎯 Donor Prediction Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .donor-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Donor Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Overview", "🔍 Search & Filter", "📊 Predictions", "🧠 Model Explanations", "📈 Analytics"]
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Stats")
    
    # Load basic stats (cached)
    @st.cache_data
    def load_basic_stats():
        try:
            import pandas as pd
            donors_df = pd.read_csv('data/synthetic_donor_dataset/donors.csv')
            return {
                'total_donors': len(donors_df),
                'legacy_intent': len(donors_df[donors_df['Legacy_Intent_Binary'] == True]),
                'legacy_rate': f"{(len(donors_df[donors_df['Legacy_Intent_Binary'] == True]) / len(donors_df) * 100):.1f}%"
            }
        except Exception as e:
            return {
                'total_donors': 0,
                'legacy_intent': 0,
                'legacy_rate': "0.0%"
            }
    
    stats = load_basic_stats()
    st.sidebar.metric("Total Donors", f"{stats['total_donors']:,}")
    st.sidebar.metric("Legacy Intent", f"{stats['legacy_intent']:,}")
    st.sidebar.metric("Legacy Rate", stats['legacy_rate'])
    
    # Model status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Status")
    
    @st.cache_data
    def check_model_status():
        try:
            # Check if model files exist
            model_files = [
                'scripts/advanced_multimodal_ensemble.py',
                'data/bert_embeddings_real.npy',
                'data/gnn_embeddings_real.npy'
            ]
            return all(os.path.exists(f) for f in model_files)
        except:
            return False
    
    model_ready = check_model_status()
    if model_ready:
        st.sidebar.success("✅ Model Ready")
    else:
        st.sidebar.error("❌ Model Not Ready")
        st.sidebar.markdown("Run the model pipeline first:")
        st.sidebar.code("python scripts/advanced_multimodal_ensemble.py")
    
    # Page routing
    try:
        if page == "🏠 Overview":
            overview.show()
        elif page == "🔍 Search & Filter":
            search.show()
        elif page == "📊 Predictions":
            predictions.show()
        elif page == "🧠 Model Explanations":
            explanations.show()
        elif page == "📈 Analytics":
            analytics.show()
        else:
            st.error(f"Unknown page: {page}")
    except Exception as e:
        st.error(f"Error loading page '{page}': {str(e)}")
        st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            LMU Capstone Project - Donor Legacy Intent Prediction Dashboard<br>
            Built with Streamlit | Powered by Advanced Multimodal ML
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
