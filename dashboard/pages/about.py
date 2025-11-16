"""
About/Documentation page for the dashboard.
Includes architecture diagram, dataset overview, and model methodology.
"""

import pandas as pd
from pathlib import Path

# Optional Streamlit import
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

# Import chart wrapper (with fallback)
try:
    from dashboard.components.charts import plotly_chart_silent
except ImportError:
    # Fallback: use st.plotly_chart directly with config (filter kwargs)
    def plotly_chart_silent(fig, width='stretch', config=None, **kwargs):
        if config is None:
            config = {'displayModeBar': True, 'displaylogo': False}
        if STREAMLIT_AVAILABLE:
            # Filter to only recognized parameters to avoid deprecation warnings
            recognized = {'theme', 'key', 'on_select', 'selection_mode'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in recognized}
            return st.plotly_chart(fig, width=width, config=config, **filtered_kwargs)
        return None


def render(df: pd.DataFrame = None):
    """
    Render the About/Documentation page.
    
    Args:
        df: Optional dataframe (for dataset statistics if available)
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    st.markdown('<p class="page-title">📚 About the Model</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Architecture, methodology, and dataset overview</p>', unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🏗️ Architecture", "📊 Dataset", "🤖 Model Methodology"])
    
    # Tab 1: Architecture Diagram
    with tab1:
        st.markdown("### 🏗️ Multimodal Fusion Architecture")
        st.markdown("""
        This diagram illustrates the end-to-end deep learning pipeline for donor prediction, 
        from synthetic data generation through interactive dashboard deployment.
        """)
        
        # Load and display the architecture diagram HTML with animations preserved
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).parent.parent / 'architecture_diagram.html',
            Path('dashboard/architecture_diagram.html'),
            Path('docs/architecture_diagram.html'),
        ]
        
        diagram_html = None
        diagram_path = None
        
        for path in possible_paths:
            if path.exists():
                diagram_path = path
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        diagram_html = f.read()
                    break
                except Exception as e:
                    st.warning(f"Error reading diagram: {e}")
                    continue
        
        if diagram_html:
            # Use components.v1.html to preserve animations, CSS, and JavaScript
            # This preserves all animations defined in the HTML/CSS (fadeInLeft, bounce, etc.)
            st.components.v1.html(
                diagram_html, 
                height=1200, 
                scrolling=True
            )
            st.caption("💡 **Note**: The diagram includes animated transitions. Scroll to see all layers of the architecture.")
        else:
            st.warning("⚠️ Architecture diagram file not found. Tried paths: " + ", ".join([str(p) for p in possible_paths]))
        
        st.markdown("---")
        st.markdown("""
        **Key Components:**
        - **6-Layer Pipeline**: Data generation → Preprocessing → Feature engineering → Model training → Evaluation → Deployment
        - **Multimodal Fusion**: Combines GNN, RNN, LSTM, CNN, and MLP architectures
        - **End-to-End Learning**: All components trained jointly for optimal performance
        - **Interactive Dashboard**: Real-time predictions and insights via Streamlit
        """)
    
    # Tab 2: Dataset Overview
    with tab2:
        st.markdown("### 📊 Synthetic Donor Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Dataset Characteristics
            
            **Total Records**: 500,000
            
            **Time Period**: 1980 - 2025
            
            **Data Sources**:
            - Donor demographics and giving history
            - Contact reports and engagement data
            - Family and relationship networks
            - Campaign interactions
            """)
        
        with col2:
            st.markdown("""
            #### Constituent Types
            
            - **Alum**: 54.9% (27,440 records)
            - **Parent**: 20.4% (10,215 records)
            - **Friend**: 14.7% (7,372 records)
            - **Foundation**: 4.0% (1,977 records)
            - **Corporation**: 3.0% (1,518 records)
            - **Trustee**: 1.4% (698 records)
            - **Trust**: 1.1% (532 records)
            - **Regent**: 0.5% (248 records)
            """)
        
        st.markdown("---")
        
        st.markdown("#### 🎯 Rating Tiers (16 Tiers: A-P)")
        st.markdown("""
        The dataset uses a 16-tier rating system where **A = highest capacity** and **P = lowest capacity**.
        Distribution follows a realistic bell curve, with most donors in the middle tiers (L-M).
        """)
        
        # Rating capacity amounts
        rating_capacities = {
            'A': '$100M+',
            'B': '$50M - $99.9M',
            'C': '$25M - $49.9M',
            'D': '$10M - $24.9M',
            'E': '$5M - $9.9M',
            'F': '$1M - $4.9M',
            'G': '$500K - $999.9K',
            'H': '$250K - $499.9K',
            'I': '$100K - $249.9K',
            'J': '$50K - $99.9K',
            'K': '$25K - $49.9K',
            'L': '$10K - $24.9K',
            'M': '$5K - $9.9K',
            'N': '$2.5K - $4.9K',
            'O': '$1K - $2.4K',
            'P': 'Less than $1K'
        }
        
        # Display rating tiers with capacity amounts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**High Capacity Tiers:**")
            for rating in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                capacity = rating_capacities[rating]
                st.markdown(f"- **Rating {rating}**: {capacity}")
        
        with col2:
            st.markdown("**Standard Capacity Tiers:**")
            for rating in ['I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
                capacity = rating_capacities[rating]
                st.markdown(f"- **Rating {rating}**: {capacity}")
        
        st.markdown("""
        **Distribution Summary:**
        - **Top Tier (A)**: 43 donors (0.09%) - $100M+ capacity
        - **Middle Tiers (I-P)**: Majority of donors (87.5%)
        - **Most Common**: Rating L (7,381 donors, 14.76%) - $10K-$24.9K capacity
        """)
        
        st.markdown("---")
        
        st.markdown("#### 🔗 Network & Relationship Features")
        st.markdown("""
        **Family Networks**:
        - 30% of donors (15,000) are part of family networks
        - 5,445 unique family units
        - Average family size: 2.75 donors
        - Largest family: 5 donors
        
        **Relationship Types**:
        - **Head**: 5,445 (36.3%) - Primary family member
        - **Spouse**: 5,445 (36.3%) - Spousal relationships
        - **Sibling**: 1,398 (9.3%) - Sibling connections
        - **Parent**: 1,394 (9.3%) - Parent-child relationships
        - **Child**: 1,318 (8.8%) - Child relationships
        
        **Network Structure**: Multi-generational families with interconnected relationships
        enabling analysis of family giving patterns and network effects.
        """)
        
        st.markdown("---")
        
        st.markdown("#### 📅 Giving History")
        st.markdown("""
        - **Earliest Gift**: January 1, 1980
        - **Latest Gift**: October 4, 2025
        - **Time Span**: 45.8 years of giving history
        - **Most Active Year**: 2025 (2,701 gifts)
        - **Donors with Valid Dates**: 29,907 (59.8%)
        
        **Giving Statistics**:
        - Total lifetime giving: $3.26 billion
        - Average lifetime giving: $65,167.88
        - Median lifetime giving: $419.31
        - Maximum lifetime giving: $100,000,000.00
        """)
        
        st.markdown("---")
        
        st.markdown("#### 🌍 Geographic Distribution")
        st.markdown("""
        Evenly distributed across 7 regions:
        - Mountain West: 14.4%
        - Midwest: 14.4%
        - Southwest: 14.3%
        - West Coast: 14.3%
        - International: 14.2%
        - Southeast: 14.2%
        - Northeast: 14.2%
        """)
    
    # Tab 3: Model Methodology
    with tab3:
        st.markdown("### 🤖 Model Methodology")
        
        st.markdown("""
        #### Multimodal Fusion Architecture
        
        The model combines multiple neural network architectures to capture different aspects of donor behavior:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔗 Graph Neural Network (GNN)**
            - Purpose: Captures donor relationships and network effects
            - Input: Family connections, relationship types
            - Output: 64-dimensional graph embeddings
            
            **🔄 Recurrent Neural Network (RNN)**
            - Purpose: Models sequential giving patterns
            - Input: Temporal giving history
            - Output: Sequential pattern representations
            
            **📝 Long Short-Term Memory (LSTM)**
            - Purpose: Captures long-term temporal dependencies
            - Input: Historical giving sequences
            - Output: Long-term memory representations
            """)
        
        with col2:
            st.markdown("""
            **🖼️ Convolutional Neural Network (CNN)**
            - Purpose: Feature extraction from structured data
            - Input: Tabular feature matrices
            - Output: Learned feature representations
            
            **📊 Multi-Layer Perceptron (MLP)**
            - Purpose: Processes tabular donor features
            - Input: 50+ engineered features
            - Output: 128-dimensional representations
            
            **🔗 Fusion Layer**
            - Purpose: Combines all modality outputs
            - Method: Cross-modal attention + weighted fusion
            - Output: Unified 256-dimensional representation
            """)
        
        st.markdown("---")
        
        st.markdown("#### Training Configuration")
        st.markdown("""
        **Data Splits**:
        - **Training**: 2021-2022 data
        - **Validation**: 2023 data
        - **Test**: 2025 data
        
        **Training Parameters**:
        - Optimizer: Adam
        - Loss Function: Cross-Entropy
        - Regularization: Dropout (0.3), Early Stopping
        - Batch Size: Optimized for 500K dataset
        
        **Model Features**:
        - End-to-end differentiable training
        - Cross-modal attention mechanisms (6 attention layers)
        - Modality importance gates (learned weights)
        - Unified prediction head
        """)
        
        st.markdown("---")
        
        st.markdown("#### Feature Engineering")
        st.markdown("""
        **50+ Engineered Features**:
        - **RFM Analysis**: Recency, Frequency, Monetary scores
        - **Network Features**: PageRank, centrality measures
        - **Temporal Patterns**: Giving trends, seasonality
        - **Engagement Metrics**: Event attendance, campaign interactions
        - **Statistical Aggregations**: Mean, median, std dev of giving patterns
        - **Text Embeddings**: BERT-encoded contact reports (768-dim)
        - **Graph Structure**: Relationship networks and family connections
        """)

