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
            st.markdown("""
            <div style="font-family: inherit; font-size: inherit;">
            <strong>High Capacity Tiers:</strong><br>
            • <strong>Rating A</strong>: $100M+<br>
            • <strong>Rating B</strong>: $50M - $99.9M<br>
            • <strong>Rating C</strong>: $25M - $49.9M<br>
            • <strong>Rating D</strong>: $10M - $24.9M<br>
            • <strong>Rating E</strong>: $5M - $9.9M<br>
            • <strong>Rating F</strong>: $1M - $4.9M<br>
            • <strong>Rating G</strong>: $500K - $999.9K<br>
            • <strong>Rating H</strong>: $250K - $499.9K
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="font-family: inherit; font-size: inherit;">
            <strong>Standard Capacity Tiers:</strong><br>
            • <strong>Rating I</strong>: $100K - $249.9K<br>
            • <strong>Rating J</strong>: $50K - $99.9K<br>
            • <strong>Rating K</strong>: $25K - $49.9K<br>
            • <strong>Rating L</strong>: $10K - $24.9K<br>
            • <strong>Rating M</strong>: $5K - $9.9K<br>
            • <strong>Rating N</strong>: $2.5K - $4.9K<br>
            • <strong>Rating O</strong>: $1K - $2.4K<br>
            • <strong>Rating P</strong>: Less than $1K
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")  # Line space before Distribution Summary
        
        st.markdown("""
        <div style="font-family: inherit; font-size: inherit;">
        <strong>Distribution Summary:</strong><br>
        • <strong>Top Tier (A)</strong>: 43 donors (0.09%) - $100M+ capacity<br>
        • <strong>Middle Tiers (I-P)</strong>: Majority of donors (87.5%)<br>
        • <strong>Most Common</strong>: Rating L (7,381 donors, 14.76%) - $10K - $24.9K capacity
        </div>
        """, unsafe_allow_html=True)
        
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
            **📊 Tabular Encoder (MLP)**
            - Purpose: Processes engineered donor features
            - Input: 60 selected features (RFM, recency, engagement, capacity, temporal, network)
            - Architecture: Residual connections, BatchNorm, GELU activation
            - Output: 256-dimensional representations
            
            **📝 Sequence Encoder (LSTM + Attention + CNN)**
            - Purpose: Captures temporal giving patterns and long-term dependencies
            - Input: Last 12 gift amounts per donor
            - Architecture: Bidirectional LSTM → Self-attention → 1D Convolution
            - Output: 256-dimensional sequential representations
            
            **🔗 Network Encoder (MLP)**
            - Purpose: Processes pre-computed network centrality features
            - Input: Network size, PageRank, degree centrality, influence scores (5 features)
            - Output: 64-dimensional network representations
            """)
        
        with col2:
            st.markdown("""
            **📝 Text Encoder (MLP with SVD)**
            - Purpose: Encodes contact report engagement metrics
            - Input: Aggregate text features (number of reports, avg text length, recent contacts) compressed via SVD to 32 dimensions
            - Architecture: Linear projection (32→32) → LayerNorm → GELU → Linear (32→64)
            - Output: 64-dimensional text representations
            - **Current Limitation**: Only uses metadata (count, length, recency) - actual text content and semantic meaning are not extracted
            
            **🔗 Cross-Modal Attention**
            - Purpose: Fuses tabular and sequence modalities
            - Architecture: Multi-head attention (4 heads)
            - Method: Attends tabular features to sequence features
            - Output: 256-dimensional attended representations
            
            **🔗 Fusion Layer**
            - Purpose: Combines all modality outputs
            - Architecture: Residual blocks, LayerNorm, GELU activation
            - Dimensions: 384 → 512 → 256 → 128 → 1 (final prediction)
            - Output: Single probability score for "will give again in 2025"
            """)
        
        st.markdown("---")
        
        st.markdown("#### Training Configuration")
        st.markdown("""
        **Data Splits**:
        - **Training**: 1980-2023 historical data
        - **Validation**: 2024 data
        - **Test**: 2025 data (prediction target)
        
        **Training Parameters**:
        - Optimizer: AdamW
        - Loss Function: BCE with Logits (with class weights for imbalance)
        - Learning Rate: 0.0001 (with ReduceLROnPlateau scheduler)
        - Regularization: Dropout (0.3), Weight Decay (1e-4), Early Stopping
        - Batch Size: 2048 (optimized for 500K dataset)
        - Hidden Dimension: 256 (increased capacity for better learning)
        
        **Model Features**:
        - End-to-end differentiable training
        - Cross-modal attention (Multi-head attention with 4 heads)
        - Residual connections for stable training
        - Batch normalization and layer normalization
        - Feature selection (top 60 features from ~102 candidates)
        - Unified prediction head with residual blocks
        - Target: Predicting "will give again in 2025"
        """)
        
        st.markdown("---")
        
        st.markdown("#### Feature Engineering")
        st.markdown("""
        **60 Selected Features** (from ~102 engineered features):
        - **RFM Analysis**: Recency, Frequency, Monetary scores (15 features)
        - **Recency & Engagement**: Days since last gift, consecutive years, gift frequency (10 features)
        - **Temporal Patterns**: Giving trends, seasonality, time-weighted giving
        - **Network Features**: Network size, PageRank, degree centrality, influence scores (5 features)
        - **Capacity Features**: Rating tier, industry, region, age
        - **Strategic Features**: High-value donor indicators, engagement metrics
        - **Influence Features**: Professional indicators, giving influence scores
        - **Text Features**: Contact report aggregates (number of reports, average text length, recent contacts in 90 days) compressed via SVD to 32 dimensions
        - **Sequence Features**: Last 12 gift amounts per donor (processed by LSTM)
        
        **Feature Selection**: Top 60 features selected via mutual information to optimize training speed while maintaining 95%+ performance.
        
        **⚠️ Text Processing Limitation**: 
        The model currently uses only aggregate statistics from contact reports (count, length, recency). The actual text content, sentiment, topics, and outcome categories (positive/negative/unresponsive) are not extracted. This represents a significant opportunity for improvement, as contact reports contain valuable semantic information that could enhance prediction accuracy.
        """)
        
        st.markdown("---")
        
        st.markdown("#### 🚀 Future Enhancement Opportunities")
        st.markdown("""
        **Text Processing Improvements**:
        
        The current model uses contact reports minimally (only metadata). The following enhancements could significantly improve prediction accuracy:
        
        **1. Sentiment Analysis**:
        - Extract positive/negative/unresponsive outcomes from contact report text
        - Use keyword-based sentiment (regex patterns for "expressed interest", "declined", "unresponsive")
        - Create binary features: `has_positive_outcome`, `has_negative_outcome`, `has_unresponsive_outcome`
        - Calculate sentiment scores per donor (ratio of positive to negative contacts)
        
        **2. Topic Extraction**:
        - Identify discussion topics from contact reports (scholarship support, capital campaign, planned giving, etc.)
        - Create topic frequency features per donor
        - Use topic modeling (LDA, NMF) or keyword matching
        - Track which topics correlate with future giving
        
        **3. Text Embeddings** (Lightweight Approaches):
        - **TF-IDF + SVD**: Extract term frequency features, compress via SVD (faster than BERT)
        - **Sentence Transformers**: Use lightweight models (all-MiniLM-L6-v2, ~384-dim) instead of full BERT
        - **Word2Vec/FastText**: Pre-trained embeddings for domain-specific terms
        - **Character-level embeddings**: For handling typos and variations
        
        **4. Structured Information Extraction**:
        - Parse outcome categories already present in synthetic data
        - Extract contact types (Meeting, Phone Call, Email, Event)
        - Track contact frequency patterns (bursts vs. steady engagement)
        - Identify relationship quality indicators from text patterns
        
        **5. Temporal Text Features**:
        - Sentiment trends over time (improving vs. declining)
        - Topic evolution (changing interests)
        - Engagement velocity (increasing/decreasing contact frequency)
        - Time-weighted sentiment (recent sentiment weighted more heavily)
        """)
    
    # Credits/Attribution section at the bottom
    st.markdown("---")
    st.markdown("""
    <div style="
        max-width: 460px;
        margin: 40px auto;
        padding: 30px;
        border-radius: 16px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 45%, #394b65 100%);
        color: #f0f4ff;
        text-align: center;
        box-shadow: 0 12px 28px rgba(18, 24, 47, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.08);
    ">
        <p style="font-size: 13px; letter-spacing: 2px; text-transform: uppercase; margin: 0 0 12px; color: #b8c7ff;">
            CREATED BY
        </p>
        <p style="font-size: 24px; font-weight: 600; margin: 0 0 6px;">Danielle Brown</p>
        <p style="font-size: 15px; margin: 0 0 2px;">Loyola Marymount University</p>
        <p style="font-size: 15px; margin: 0 0 18px;">M.S. in Computer Science Senior Capstone Project</p>
        <p style="font-size: 13px; letter-spacing: 3px; margin: 0; color: #d7deff;">December 2025</p>
    </div>
    """, unsafe_allow_html=True)

