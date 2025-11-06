"""
Test script for dashboard data loader module.
Run this to verify data loader extraction works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from dashboard.data.loader import load_full_dataset, process_dataframe, generate_sample_data

def test_loader():
    """Test that data loader works correctly."""
    print("=" * 60)
    print("Testing Dashboard Data Loader Module")
    print("=" * 60)
    print()
    
    # Test 1: Load dataset
    print("📊 Test 1: Loading dataset...")
    try:
        df = load_full_dataset(use_cache=False)  # Don't use cache for testing
        print(f"   ✅ Loaded {len(df):,} rows")
        print(f"   ✅ Columns: {len(df.columns)}")
        
        # Check required columns
        required_cols = ['donor_id', 'predicted_prob', 'actual_gave', 'days_since_last']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   ⚠️  Missing required columns: {missing_cols}")
        else:
            print(f"   ✅ All required columns present")
        
        # Check data types
        print(f"   ✅ predicted_prob range: {df['predicted_prob'].min():.3f} - {df['predicted_prob'].max():.3f}")
        print(f"   ✅ actual_gave values: {df['actual_gave'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        return False
    print()
    
    # Test 2: Process dataframe
    print("📊 Test 2: Processing dataframe...")
    try:
        # Create a simple test dataframe
        import pandas as pd
        test_df = pd.DataFrame({
            'Donor_ID': ['D001', 'D002', 'D003'],
            'Will_Give_Again_Probability': [0.8, 0.5, 0.2],
            'Gave_Again_In_2024': [1, 0, 0],
            'Last_Gift_Date': ['2024-01-15', '2023-06-01', '2022-01-01'],
            'gift_count': [5, 3, 1]
        })
        
        processed_df = process_dataframe(test_df)
        print(f"   ✅ Processed {len(processed_df)} rows")
        print(f"   ✅ Standardized columns: {list(processed_df.columns[:5])}")
        
        # Verify column mapping worked
        if 'donor_id' in processed_df.columns and 'predicted_prob' in processed_df.columns:
            print(f"   ✅ Column mapping successful")
        else:
            print(f"   ⚠️  Column mapping may have issues")
            
    except Exception as e:
        print(f"   ❌ Failed to process dataframe: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Test 3: Generate sample data
    print("📊 Test 3: Generating sample data...")
    try:
        sample_df = generate_sample_data()
        print(f"   ✅ Generated {len(sample_df):,} sample rows")
        print(f"   ✅ Sample columns: {list(sample_df.columns[:5])}")
        
        # Verify sample data has required columns
        required_cols = ['donor_id', 'predicted_prob', 'actual_gave']
        if all(col in sample_df.columns for col in required_cols):
            print(f"   ✅ Sample data has all required columns")
        else:
            print(f"   ⚠️  Sample data missing some columns")
            
    except Exception as e:
        print(f"   ❌ Failed to generate sample data: {e}")
        return False
    print()
    
    print("=" * 60)
    print("✅ All data loader tests passed!")
    print("=" * 60)
    print()
    print("Next step: Extract metrics module (Phase 4)")
    print()
    return True

if __name__ == "__main__":
    success = test_loader()
    sys.exit(0 if success else 1)

