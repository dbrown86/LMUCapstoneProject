"""
Test script for page modules.
Run this to verify all pages can be imported and have correct structure.
"""

import pandas as pd
import numpy as np

def test_pages():
    print("=" * 60)
    print("Testing Dashboard Page Modules")
    print("=" * 60)
    print()
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'donor_id': range(100),
        'region': ['Northeast'] * 50 + ['Southeast'] * 50,
        'donor_type': ['Individual'] * 100,
        'segment': ['Recent (0-6mo)'] * 25 + ['Recent (6-12mo)'] * 25 + ['Lapsed (1-2yr)'] * 25 + ['Very Lapsed (2yr+)'] * 25,
        'predicted_prob': np.random.uniform(0, 1, 100),
        'actual_gave': np.random.choice([0, 1], 100),
        'avg_gift': np.random.uniform(100, 1000, 100),
        'total_giving': np.random.uniform(1000, 10000, 100),
        'days_since_last': np.random.uniform(0, 1000, 100),
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
    })
    
    print("📄 Test 1: Import page modules...")
    try:
        from dashboard.pages import performance, features, donor_insights, predictions
        from dashboard.pages.utils import filter_dataframe, get_segment_stats
        print("   ✅ All page modules imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import page modules: {e}")
        return False
    print()
    
    print("📊 Test 2: Test page utilities...")
    try:
        # Test filter_dataframe
        filtered = filter_dataframe(test_df, ['Northeast'], ['Individual'], ['Recent (0-6mo)'], use_cache=False)
        assert len(filtered) > 0
        print("   ✅ filter_dataframe works")
        
        # Test get_segment_stats
        stats = get_segment_stats(test_df, use_cache=False)
        assert len(stats) > 0
        assert 'estimated_revenue' in stats.columns
        print("   ✅ get_segment_stats works")
    except Exception as e:
        print(f"   ❌ Failed to test utilities: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    print("📈 Test 3: Test page render functions (structure only)...")
    try:
        # Just check that functions exist and are callable
        assert callable(performance.render)
        assert callable(features.render)
        assert callable(donor_insights.render)
        assert callable(predictions.render)
        print("   ✅ All page render functions are callable")
    except Exception as e:
        print(f"   ❌ Failed to test render functions: {e}")
        return False
    print()
    
    print("=" * 60)
    print("✅ All page module tests passed!")
    print("=" * 60)
    print()
    print("Note: Full rendering tests require Streamlit to be running.")
    print("Next step: Update alternate_dashboard.py to use new page modules")
    print()

if __name__ == "__main__":
    test_pages()

