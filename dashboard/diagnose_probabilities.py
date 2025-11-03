#!/usr/bin/env python3
"""
Diagnostic script to trace the source of probability values
Run this to understand where predicted_prob values come from and why they might be 1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def diagnose_probability_source():
    """Trace where predicted_prob values come from"""
    
    print("=" * 80)
    print("PROBABILITY SOURCE DIAGNOSTIC")
    print("=" * 80)
    
    root = Path(__file__).resolve().parent.parent
    data_dir_env = os.getenv("LMU_DATA_DIR")
    env_dir = Path(data_dir_env).resolve() if data_dir_env else None
    
    # Priority 1: Check Parquet files
    parquet_paths = [
        str(root / "data/parquet_export/donors_with_network_features.parquet"),
        str(root / "donors_with_network_features.parquet"),
        str(root / "data/donors.parquet"),
        "data/parquet_export/donors_with_network_features.parquet",
        "donors_with_network_features.parquet",
    ]
    if env_dir:
        parquet_paths.extend([
            str(env_dir / "donors_with_network_features.parquet"),
            str(env_dir / "data/donors.parquet"),
        ])
    
    df = None
    source_file = None
    
    for path in parquet_paths:
        if os.path.exists(path):
            try:
                print(f"\n📁 Found data file: {path}")
                df = pd.read_parquet(path, engine='pyarrow')
                source_file = path
                break
            except Exception as e:
                print(f"   ⚠️ Could not read: {e}")
    
    if df is None:
        print("\n❌ No data file found. Cannot diagnose.")
        return
    
    print(f"\n✅ Loaded {len(df):,} rows from {source_file}")
    print(f"   Columns: {list(df.columns)[:10]}...")
    
    # Check for probability columns
    prob_cols = [col for col in df.columns if 'prob' in col.lower() or 'probability' in col.lower()]
    print(f"\n🔍 Probability-related columns found: {prob_cols}")
    
    if not prob_cols:
        print("\n❌ No probability columns found in source data!")
        return
    
    # Check the main probability column (legacy_intent_probability)
    main_prob_col = None
    for col in ['legacy_intent_probability', 'Legacy_Intent_Probability', 'predicted_prob']:
        if col in df.columns:
            main_prob_col = col
            break
    
    if main_prob_col:
        print(f"\n📊 Analyzing column: '{main_prob_col}'")
        prob_values = pd.to_numeric(df[main_prob_col], errors='coerce')
        
        print(f"\n   Statistics:")
        print(f"   - Total values: {len(prob_values):,}")
        print(f"   - Non-null: {prob_values.notna().sum():,}")
        print(f"   - Min: {prob_values.min():.6f}")
        print(f"   - Max: {prob_values.max():.6f}")
        print(f"   - Mean: {prob_values.mean():.6f}")
        print(f"   - Median: {prob_values.median():.6f}")
        print(f"   - Std: {prob_values.std():.6f}")
        
        # Count exact 1.0s
        exact_ones = (prob_values == 1.0).sum()
        near_ones = ((prob_values >= 0.99) & (prob_values < 1.0)).sum()
        print(f"\n   🔴 Critical Values:")
        print(f"   - Exactly 1.0: {exact_ones:,} ({exact_ones/len(prob_values)*100:.2f}%)")
        print(f"   - ≥0.99 but <1.0: {near_ones:,} ({near_ones/len(prob_values)*100:.2f}%)")
        
        # Check unique values
        unique_vals = prob_values.unique()
        print(f"\n   📈 Unique values: {len(unique_vals):,}")
        if len(unique_vals) <= 20:
            print(f"   - Unique values: {sorted(unique_vals)}")
        else:
            print(f"   - First 10 unique values: {sorted(unique_vals)[:10]}")
            print(f"   - Last 10 unique values: {sorted(unique_vals)[-10:]}")
        
        # Distribution analysis
        print(f"\n   📊 Distribution (percentiles):")
        for p in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
            val = prob_values.quantile(p/100)
            print(f"   - {p:3d}th percentile: {val:.6f}")
        
        # Check if values look like binary (only 0 and 1)
        unique_binary = set(prob_values.dropna().unique())
        if unique_binary.issubset({0.0, 1.0}):
            print(f"\n   ⚠️ WARNING: Values appear to be BINARY (0 or 1) instead of probabilities!")
            print(f"      This suggests the model is outputting predictions, not probabilities.")
        
        # Check if values are clipped
        if exact_ones > 0:
            # Check distribution of values above 0.95
            high_vals = prob_values[prob_values >= 0.95]
            print(f"\n   🔍 High-value analysis (≥0.95):")
            print(f"   - Count: {len(high_vals):,}")
            if len(high_vals) > 0:
                print(f"   - Distribution: {high_vals.value_counts().head(10).to_dict()}")
        
        # Sample rows with 1.0
        if exact_ones > 0:
            print(f"\n   🔴 Sample rows with exactly 1.0:")
            ones_df = df[prob_values == 1.0].head(5)
            print(f"   {ones_df[['ID', 'Donor_ID'] + [main_prob_col]].to_string(index=False)}")
            
            # Check if 1.0 values correlate with actual_gave
            if 'Legacy_Intent_Binary' in df.columns or 'legacy_intent_binary' in df.columns:
                binary_col = 'Legacy_Intent_Binary' if 'Legacy_Intent_Binary' in df.columns else 'legacy_intent_binary'
                ones_with_binary = df[prob_values == 1.0][binary_col].value_counts()
                print(f"\n   📊 For rows with prob=1.0, binary label distribution:")
                print(f"   {ones_with_binary.to_dict()}")
                
                # Check correlation
                if len(ones_with_binary) > 0:
                    correlation = (prob_values == 1.0).astype(int).corr(df[binary_col].astype(int))
                    print(f"   - Correlation between prob=1.0 and binary label: {correlation:.4f}")
        
        print(f"\n" + "=" * 80)
        print("DIAGNOSIS:")
        print("=" * 80)
        
        if exact_ones > len(prob_values) * 0.01:  # More than 1% are 1.0
            print("🚨 SUSPICIOUS: More than 1% of values are exactly 1.0")
            print("   - Likely causes:")
            print("     1. Model outputting binary predictions (0/1) instead of probabilities")
            print("     2. Probabilities being clipped/capped at 1.0")
            print("     3. Data preprocessing error (wrong column used)")
            print("     4. Model miscalibration or overfitting")
        
        if unique_binary.issubset({0.0, 1.0}):
            print("🚨 CRITICAL: All values are binary (0 or 1)")
            print("   - The model is NOT outputting probabilities")
            print("   - You need to use predict_proba() instead of predict()")
            print("   - OR the data was saved incorrectly")
        
        if prob_values.max() > 1.0:
            print("⚠️ WARNING: Some values are > 1.0")
            print("   - These need to be normalized (divide by max)")
        
        if prob_values.min() < 0.0:
            print("⚠️ WARNING: Some values are < 0.0")
            print("   - Probabilities should be in [0, 1] range")
    
    # Check for other related columns
    print(f"\n📋 Other relevant columns:")
    relevant_cols = [col for col in df.columns if any(x in col.lower() for x in ['legacy', 'intent', 'binary', 'gave', 'target', 'label'])]
    for col in relevant_cols[:10]:
        print(f"   - {col}")
        if df[col].dtype in ['int64', 'float64']:
            unique = df[col].unique()
            print(f"     Unique values: {sorted(unique) if len(unique) <= 10 else f'{len(unique)} values'}")
    
    print(f"\n" + "=" * 80)
    print("TRACING DATA FLOW:")
    print("=" * 80)
    
    # Simulate what the dashboard does
    print("\n1. Original column in parquet: 'Legacy_Intent_Probability'")
    if main_prob_col:
        print(f"   - Original values: {prob_values.describe()}")
        print(f"   - Sample values: {list(prob_values.head(10))}")
    
    # Check what happens during column mapping
    print("\n2. Column mapping (dashboard process_dataframe):")
    print("   - 'Legacy_Intent_Probability' → 'predicted_prob'")
    if main_prob_col == 'Legacy_Intent_Probability':
        mapped_probs = prob_values.copy()  # This is what becomes predicted_prob
        print(f"   - After mapping: Same values (max={mapped_probs.max():.4f})")
        
        # Check normalization
        if mapped_probs.max() > 1:
            print(f"   - Normalization triggered: dividing by {mapped_probs.max()}")
            mapped_probs = mapped_probs / mapped_probs.max()
        else:
            print(f"   - No normalization needed (max ≤ 1)")
        
        # Check if actual_gave is being used incorrectly
        if 'Legacy_Intent_Binary' in df.columns:
            binary_col = df['Legacy_Intent_Binary']
            print(f"\n3. Checking for confusion with binary column:")
            print(f"   - 'Legacy_Intent_Binary' exists with values: {sorted(binary_col.unique())}")
            print(f"   - Binary column should map to 'actual_gave', NOT 'predicted_prob'")
            
            # Check if they match
            prob_binary_match = (mapped_probs == binary_col).sum()
            print(f"   - Values where prob matches binary: {prob_binary_match} ({prob_binary_match/len(df)*100:.2f}%)")
            if prob_binary_match > len(df) * 0.9:
                print(f"   🚨 CRITICAL: 'predicted_prob' appears to match 'Legacy_Intent_Binary'!")
                print(f"      This means probabilities are actually binary predictions!")
    
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("1. Check your model's prediction code - ensure it uses predict_proba(), not predict()")
    print("2. Look for np.clip(..., 0, 1) or similar clipping operations")
    print("3. Verify the parquet file contains actual probabilities, not binary predictions")
    print("4. Check model training code for calibration issues")
    print("5. Consider applying probability calibration (IsotonicRegression) if needed")
    print("\n6. **SPECIFIC CHECK**: Verify 'Legacy_Intent_Probability' column is NOT the same as")
    print("   'Legacy_Intent_Binary' - these should be different columns!")
    print("\n7. If you trained a model, check where predictions were saved and ensure:")
    print("   - predict_proba() was used (not predict())")
    print("   - Probabilities were saved (not binary 0/1)")
    print("   - No clipping/capping at 1.0 occurred")
    
    return df, main_prob_col

if __name__ == "__main__":
    diagnose_probability_source()

