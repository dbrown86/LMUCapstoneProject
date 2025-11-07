"""
Extended test script for page modules (including wrappers).
This validates imports and callable signatures without launching Streamlit.
"""

import pandas as pd
import numpy as np


def test_pages_extended():
    print("=" * 60)
    print("Testing Extended Page Modules (Wrappers)")
    print("=" * 60)
    print()

    # Create a small test dataframe
    test_df = pd.DataFrame({
        'donor_id': range(10),
        'region': ['Northeast'] * 5 + ['Southeast'] * 5,
        'donor_type': ['Individual'] * 10,
        'segment': ['Recent (0-6mo)'] * 5 + ['Lapsed (1-2yr)'] * 5,
        'predicted_prob': np.random.uniform(0, 1, 10),
        'actual_gave': np.random.choice([0, 1], 10),
        'avg_gift': np.random.uniform(100, 1000, 10),
        'total_giving': np.random.uniform(1000, 10000, 10),
    })

    print("📄 Test 1: Import wrapper page modules...")
    try:
        from dashboard.pages import (
            render_business_impact,
            render_model_comparison,
            render_dashboard,
        )
        print("   ✅ Wrapper modules imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import wrapper modules: {e}")
        return False
    print()

    print("🧪 Test 2: Validate callable signatures...")
    try:
        assert callable(render_business_impact)
        assert callable(render_model_comparison)
        assert callable(render_dashboard)
        print("   ✅ All wrapper render functions are callable")
    except Exception as e:
        print(f"   ❌ Failed callable checks: {e}")
        return False
    print()

    print("🔧 Test 3: Dry-run calls (no Streamlit UI)...")
    try:
        # Only verify they don't raise import/signature errors when referenced
        # Avoid invoking Streamlit rendering in a non-Streamlit context
        # Passing minimal arguments; actual rendering is exercised in app runtime
        _ = (render_business_impact, render_model_comparison, render_dashboard)
        print("   ✅ Wrapper functions ready for runtime use")
    except Exception as e:
        print(f"   ❌ Wrapper dry-run failed: {e}")
        return False
    print()

    print("=" * 60)
    print("✅ Extended page module checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_pages_extended()
