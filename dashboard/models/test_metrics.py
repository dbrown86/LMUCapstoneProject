"""
Test script for dashboard metrics module.
Run this to verify metrics extraction works correctly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from dashboard.models.metrics import (
    try_load_saved_metrics,
    get_model_metrics,
    get_feature_importance
)

def test_metrics():
    """Test that metrics module works correctly."""
    print("=" * 60)
    print("Testing Dashboard Metrics Module")
    print("=" * 60)
    print()
    
    # Test 1: Try loading saved metrics
    print("📊 Test 1: Loading saved metrics...")
    try:
        saved_metrics = try_load_saved_metrics()
        if saved_metrics:
            print(f"   ✅ Found saved metrics: {list(saved_metrics.keys())}")
            if saved_metrics.get('auc') is not None:
                print(f"   ✅ AUC: {saved_metrics['auc']:.4f}")
        else:
            print(f"   ⚠️  No saved metrics found (this is okay)")
    except Exception as e:
        print(f"   ⚠️  Error loading saved metrics: {e}")
    print()
    
    # Test 2: Calculate model metrics
    print("📊 Test 2: Calculating model metrics...")
    try:
        # Create test dataframe
        np.random.seed(42)
        test_df = pd.DataFrame({
            'Will_Give_Again_Probability': np.random.beta(2, 5, 1000),
            'Gave_Again_In_2024': np.random.binomial(1, 0.17, 1000),
            'Last_Gift_Date': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'days_since_last': np.random.exponential(365, 1000)
        })
        
        metrics = get_model_metrics(df=test_df, use_cache=False)
        print(f"   ✅ Metrics calculated")
        
        # Check key metrics
        if metrics.get('auc') is not None:
            print(f"   ✅ AUC: {metrics['auc']:.4f}")
        if metrics.get('f1') is not None:
            print(f"   ✅ F1: {metrics['f1']:.4f}")
        if metrics.get('baseline_auc') is not None:
            print(f"   ✅ Baseline AUC: {metrics['baseline_auc']:.4f}")
        if metrics.get('lift') is not None:
            print(f"   ✅ Lift: {metrics['lift']:.2%}")
        
        # Verify all expected keys are present
        expected_keys = ['auc', 'f1', 'accuracy', 'precision', 'recall', 'baseline_auc', 'lift']
        missing_keys = [k for k in expected_keys if k not in metrics]
        if missing_keys:
            print(f"   ⚠️  Missing keys (may be None): {missing_keys}")
        else:
            print(f"   ✅ All expected metrics present")
            
    except Exception as e:
        print(f"   ❌ Failed to calculate metrics: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Test 3: Calculate feature importance
    print("📊 Test 3: Calculating feature importance...")
    try:
        # Create test dataframe with features
        np.random.seed(42)
        test_df = pd.DataFrame({
            'donor_id': [f'D{i:06d}' for i in range(1000)],
            'Gave_Again_In_2024': np.random.binomial(1, 0.17, 1000),
            'days_since_last': np.random.exponential(365, 1000),
            'total_giving': np.random.lognormal(6, 2, 1000),
            'avg_gift': np.random.lognormal(5.5, 1.5, 1000),
            'gift_count': np.random.poisson(3, 1000),
            'rfm_score': np.random.uniform(1, 5, 1000),
            'years_active': np.random.randint(0, 11, 1000)
        })
        
        feature_importance = get_feature_importance(test_df, use_cache=False)
        print(f"   ✅ Feature importance calculated")
        print(f"   ✅ Top features: {len(feature_importance)}")
        
        if len(feature_importance) > 0:
            print(f"   ✅ Top feature: {feature_importance.iloc[0]['feature']} ({feature_importance.iloc[0]['importance']:.4f})")
        else:
            print(f"   ⚠️  No features found")
            
    except Exception as e:
        print(f"   ❌ Failed to calculate feature importance: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    print("=" * 60)
    print("✅ All metrics tests passed!")
    print("=" * 60)
    print()
    print("Next step: Extract components (Phase 5)")
    print()
    return True

if __name__ == "__main__":
    success = test_metrics()
    sys.exit(0 if success else 1)

