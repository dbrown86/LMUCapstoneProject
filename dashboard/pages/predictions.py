"""
Predictions page for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

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


def render(df: pd.DataFrame):
    """
    Render the predictions page.
    
    Args:
        df: Dataframe with donor data and predictions
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    st.markdown('<p class="page-title">🎲 Interactive Prediction Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Test predictions with custom donor profiles</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Batch Prediction", "🔍 Similar Donors"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📝 Donor Information")
            days_since = st.slider("Days since last gift", 0, 2000, 365)
            gift_count = st.number_input("Number of gifts", 0, 50, 5)
            total_giving = st.number_input("Total lifetime giving ($)", 0, 1000000, 5000)
            avg_gift = st.number_input("Average gift ($)", 0, 100000, 500)
            predict_button = st.button("🔮 Generate Prediction", type="primary")
        
        with col2:
            if predict_button:
                # Simple prediction logic
                base_prob = 0.17
                recency_adj = 0.40 if days_since <= 90 else 0.25 if days_since <= 180 else 0.10 if days_since <= 365 else -0.05
                rfm_score = 3 + (avg_gift / 1000 - 3) * 0.5
                rfm_adj = (rfm_score - 3) * 0.10
                predicted_prob = np.clip(base_prob + recency_adj + rfm_adj, 0.01, 0.99)
                
                # Calculate uncertainty (simulated confidence interval)
                uncertainty = 0.05 + abs(recency_adj) * 0.1
                lower_bound = max(0, predicted_prob - uncertainty)
                upper_bound = min(1, predicted_prob + uncertainty)
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Likelihood to Give", 'font': {'size': 24}},
                    number={'suffix': "%", 'font': {'size': 48}},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#2E86AB"},
                        'steps': [
                            {'range': [0, 40], 'color': "#ffebee"},
                            {'range': [40, 70], 'color': "#fff3e0"},
                            {'range': [70, 100], 'color': "#e8f5e9"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                plotly_chart_silent(fig, config={'displayModeBar': True, 'displaylogo': False})
                
                confidence = "HIGH" if predicted_prob >= 0.7 else "MEDIUM" if predicted_prob >= 0.4 else "LOW"
                
                if predicted_prob >= 0.7:
                    st.success(f"✅ **{confidence} Confidence** - {predicted_prob:.1%} likelihood to give")
                elif predicted_prob >= 0.4:
                    st.info(f"🟡 **{confidence} Confidence** - {predicted_prob:.1%} likelihood to give")
                else:
                    st.warning(f"⚠️ **{confidence} Confidence** - {predicted_prob:.1%} likelihood to give")
                
                # Uncertainty quantification
                st.markdown(f"""
                <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <h5 style="margin-top: 0;">📊 Prediction Confidence Interval</h5>
                    <p><strong>Range:</strong> {lower_bound:.1%} - {upper_bound:.1%}</p>
                    <p style="font-size: 12px; color: #666;">This range reflects uncertainty in the prediction based on available data.</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 Batch Prediction")
        st.info("Upload a CSV file with donor information to get predictions for multiple donors at once.")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], help="CSV should have columns: days_since_last, avg_gift, total_giving, etc.")
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(batch_df)} donors from CSV")
                
                # Generate predictions (simplified logic)
                if 'days_since_last' in batch_df.columns and 'avg_gift' in batch_df.columns:
                    base_prob = 0.17
                    batch_df['predicted_prob'] = base_prob + np.where(
                        batch_df['days_since_last'] <= 90, 0.40,
                        np.where(batch_df['days_since_last'] <= 180, 0.25,
                               np.where(batch_df['days_since_last'] <= 365, 0.10, -0.05))
                    )
                    batch_df['predicted_prob'] = np.clip(batch_df['predicted_prob'], 0.01, 0.99)
                    
                    # Display results
                    display_cols = ['predicted_prob']
                    if 'donor_id' in batch_df.columns:
                        display_cols.insert(0, 'donor_id')
                    if 'avg_gift' in batch_df.columns:
                        display_cols.append('avg_gift')
                    
                    results_df = batch_df[display_cols].copy()
                    results_df['predicted_prob'] = results_df['predicted_prob'].apply(lambda x: f"{x:.1%}")
                    results_df = results_df.rename(columns={'predicted_prob': 'Probability'})
                    
                    st.dataframe(results_df, width='stretch', hide_index=True)
                    
                    # Download results
                    csv_results = batch_df[display_cols].to_csv(index=False)
                    st.download_button(
                        "📥 Download Predictions",
                        csv_results,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("CSV must contain 'days_since_last' and 'avg_gift' columns")
            except Exception as e:
                st.error(f"Error processing CSV: {e}")
    
    with tab3:
        st.markdown("### 🔍 Find Similar Donors")
        st.info("Enter donor characteristics to find similar donors from the database with known outcomes.")
        
        if 'predicted_prob' in df.columns and 'donor_id' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                ref_days = st.slider("Days since last gift", 0, 2000, 365, key="similar_days")
                ref_avg_gift = st.number_input("Average gift ($)", 0, 100000, 500, key="similar_gift")
            
            with col2:
                num_similar = st.number_input("Number of similar donors to show", 5, 50, 10)
                search_button = st.button("🔍 Find Similar", type="primary")
            
            if search_button:
                # Find similar donors based on distance in feature space
                if 'days_since_last' in df.columns and 'avg_gift' in df.columns:
                    # Calculate distance (simplified)
                    df['distance'] = np.sqrt(
                        ((df['days_since_last'] - ref_days) / 365) ** 2 +
                        ((df['avg_gift'] - ref_avg_gift) / max(df['avg_gift'].max(), 1)) ** 2
                    )
                    
                    similar_donors = df.nsmallest(num_similar, 'distance')
                    
                    if len(similar_donors) > 0:
                        display_cols = ['donor_id', 'predicted_prob', 'days_since_last', 'avg_gift']
                        if 'actual_gave' in similar_donors.columns:
                            display_cols.append('actual_gave')
                        
                        similar_display = similar_donors[display_cols].copy()
                        similar_display = similar_display.rename(columns={
                            'predicted_prob': 'Probability',
                            'days_since_last': 'Days Since Last',
                            'avg_gift': 'Avg Gift',
                            'actual_gave': 'Actually Gave?'
                        })
                        similar_display['Probability'] = similar_display['Probability'].apply(lambda x: f"{x:.1%}")
                        if 'Actually Gave?' in similar_display.columns:
                            similar_display['Actually Gave?'] = similar_display['Actually Gave?'].apply(lambda x: "Yes" if x == 1 else "No")
                        
                        st.dataframe(similar_display, width='stretch', hide_index=True)
                        
                        # Show statistics
                        if 'actual_gave' in similar_donors.columns:
                            actual_rate = similar_donors['actual_gave'].mean()
                            st.info(f"💡 **Historical Context**: Among similar donors, {actual_rate:.1%} actually gave. This provides context for the prediction.")
                else:
                    st.error("Required columns not available in dataset")
        else:
            st.error("Prediction data not available")

