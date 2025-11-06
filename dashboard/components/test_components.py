"""
Test script for dashboard components.
Run this to verify component extraction works correctly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

def test_components():
    """Test that components module works correctly."""
    print("=" * 60)
    print("Testing Dashboard Components Module")
    print("=" * 60)
    print()
    
    # Test 1: CSS Styles
    print("🎨 Test 1: CSS Styles...")
    try:
        from dashboard.components.styles import get_css_styles
        css = get_css_styles()
        assert len(css) > 0
        assert '.metric-card' in css
        assert '.chart-container' in css
        print(f"   ✅ CSS styles loaded ({len(css)} characters)")
    except Exception as e:
        print(f"   ❌ Failed to load CSS: {e}")
        return False
    print()
    
    # Test 2: Chart utilities
    print("📊 Test 2: Chart utilities...")
    try:
        from dashboard.components.charts import plotly_chart_silent
        import plotly.graph_objects as go
        
        # Create a test figure
        fig = go.Figure(go.Bar(x=[1, 2, 3], y=[4, 5, 6]))
        
        # Test function (will return None if Streamlit not available, which is fine)
        result = plotly_chart_silent(fig)
        print(f"   ✅ Chart utility function works")
    except Exception as e:
        print(f"   ❌ Failed to test chart utility: {e}")
        return False
    print()
    
    # Test 3: Metric cards
    print("📈 Test 3: Metric cards...")
    try:
        from dashboard.components.metric_cards import render_metric_card, render_simple_metric_card
        
        # Test function (will return None if Streamlit not available, which is fine)
        render_metric_card("Test Label", "Test Value", icon="📊")
        render_simple_metric_card("Test Label", "Test Value")
        print(f"   ✅ Metric card functions work")
    except Exception as e:
        print(f"   ❌ Failed to test metric cards: {e}")
        return False
    print()
    
    # Test 4: Sidebar utilities
    print("🔧 Test 4: Sidebar utilities...")
    try:
        from dashboard.components.sidebar import get_unique_values
        
        # Create test dataframe (all arrays must be same length)
        test_df = pd.DataFrame({
            'region': ['Northeast', 'Southeast', 'Midwest'],
            'donor_type': ['Individual', 'Corporate', 'Foundation'],
            'segment': ['Recent (0-6mo)', 'Lapsed (1-2yr)', 'Recent (0-6mo)']
        })
        
        unique_vals = get_unique_values(test_df, use_cache=False)
        assert 'regions' in unique_vals
        assert 'types' in unique_vals
        assert 'segments' in unique_vals
        print(f"   ✅ Unique values extracted: {len(unique_vals['regions'])} regions, {len(unique_vals['types'])} types")
    except Exception as e:
        print(f"   ❌ Failed to test sidebar utilities: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Test 5: Sidebar rendering (without Streamlit)
    print("📱 Test 5: Sidebar rendering...")
    try:
        from dashboard.components.sidebar import render_sidebar
        
        # Create test dataframe (all arrays must be same length)
        test_df = pd.DataFrame({
            'region': ['Northeast', 'Southeast'],
            'donor_type': ['Individual', 'Corporate'],
            'segment': ['Recent (0-6mo)', 'Lapsed (1-2yr)']
        })
        
        # This will return defaults if Streamlit not available (for testing)
        result = render_sidebar(test_df)
        assert len(result) == 5
        page, regions, donor_types, segments, threshold = result
        print(f"   ✅ Sidebar function works (returns: page={page}, threshold={threshold})")
    except Exception as e:
        print(f"   ❌ Failed to test sidebar rendering: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    print("=" * 60)
    print("✅ All component tests passed!")
    print("=" * 60)
    print()
    print("Note: Some functions require Streamlit to fully test UI rendering.")
    print("      Core functionality has been verified.")
    print()
    print("Next step: Extract pages (Phase 6)")
    print()
    return True

if __name__ == "__main__":
    success = test_components()
    sys.exit(0 if success else 1)

