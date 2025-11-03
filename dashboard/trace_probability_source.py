#!/usr/bin/env python3
"""
Comprehensive script to trace the source of probability values and identify issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob
from datetime import datetime

def clear_streamlit_cache():
    """Instructions for clearing Streamlit cache"""
    print("=" * 80)
    print("STEP 1: CLEAR STREAMLIT CACHE")
    print("=" * 80)
    print("\nTo clear Streamlit cache, run one of these commands:")
    print("\nOption 1 (PowerShell):")
    print("  streamlit cache clear")
    print("\nOption 2: Restart Streamlit")
    print("  - Stop the current Streamlit process (Ctrl+C)")
    print("  - Restart with: streamlit run dashboard/alternate_dashboard.py")
    print("\nOption 3: Manually delete cache")
    print("  - Delete the .streamlit/cache directory if it exists")
    print("\n" + "-" * 80 + "\n")

def find_prediction_files():
    """Find any prediction or model output files"""
    print("=" * 80)
    print("STEP 2: CHECK FOR MODEL PREDICTION FILES")
    print("=" * 80)
    
    root = Path(__file__).resolve().parent.parent
    patterns = [
        "**/*predictions*.parquet",
        "**/*model_output*.parquet",
        "**/*prediction*.parquet",
        "**/*predict*.parquet",
        "**/*inference*.parquet",
        "results/**/*.parquet",
        "models/**/*.parquet",
        "data/**/*predictions*.parquet",
        "data/**/*model_output*.parquet"
    ]
    
    found_files = []
    for pattern in patterns:
        for file_path in root.glob(pattern):
            if file_path.is_file():
                found_files.append(file_path)
    
    if found_files:
        print(f"\n✅ Found {len(found_files)} potential prediction file(s):\n")
        for f in found_files:
            print(f"  📁 {f}")
            try:
                df_check = pd.read_parquet(f, engine='pyarrow')
                print(f"     - Rows: {len(df_check):,}")
                print(f"     - Columns: {list(df_check.columns)[:5]}...")
                
                # Check for probability columns
                prob_cols = [c for c in df_check.columns if 'prob' in c.lower() or 'probability' in c.lower()]
                if prob_cols:
                    print(f"     - Probability columns: {prob_cols}")
                    for col in prob_cols:
                        vals = pd.to_numeric(df_check[col], errors='coerce')
                        ones = (vals == 1.0).sum()
                        max_val = vals.max()
                        print(f"       * {col}: max={max_val:.4f}, exactly 1.0: {ones:,} ({ones/len(vals)*100:.2f}%)")
            except Exception as e:
                print(f"     - ⚠️ Could not read: {e}")
    else:
        print("\n✅ No separate prediction files found.")
        print("   This is normal - predictions may be in the main data file.")
    
    print("\n" + "-" * 80 + "\n")
    return found_files

def verify_loaded_file():
    """Verify which file is actually being loaded"""
    print("=" * 80)
    print("STEP 3: VERIFY LOADED FILE")
    print("=" * 80)
    
    root = Path(__file__).resolve().parent.parent
    data_dir_env = os.getenv("LMU_DATA_DIR")
    env_dir = Path(data_dir_env).resolve() if data_dir_env else None
    
    # Check all candidate paths (same as dashboard)
    parquet_paths = [
        str(root / "data/parquet_export/donors_with_network_features.parquet"),
        str(root / "donors_with_network_features.parquet"),
        str(root / "data/donors.parquet"),
        "data/parquet_export/donors_with_network_features.parquet",
        "donors_with_network_features.parquet",
        "data/donors.parquet",
    ]
    if env_dir:
        parquet_paths.extend([
            str(env_dir / "donors_with_network_features.parquet"),
            str(env_dir / "data/donors.parquet"),
        ])
    
    print("\nChecking candidate files (in priority order):\n")
    loaded_file = None
    
    for i, path in enumerate(parquet_paths, 1):
        if os.path.exists(path):
            print(f"{i}. ✅ EXISTS: {path}")
            if loaded_file is None:  # First one found is what would be loaded
                loaded_file = path
                print(f"   ⭐ THIS FILE WILL BE LOADED")
                try:
                    df = pd.read_parquet(path, engine='pyarrow')
                    prob_col = None
                    for col in ['Legacy_Intent_Probability', 'legacy_intent_probability', 'predicted_prob']:
                        if col in df.columns:
                            prob_col = col
                            break
                    
                    if prob_col:
                        vals = pd.to_numeric(df[prob_col], errors='coerce')
                        ones = (vals == 1.0).sum()
                        max_val = vals.max()
                        print(f"   - Probability column: {prob_col}")
                        print(f"   - Max value: {max_val:.6f}")
                        print(f"   - Exactly 1.0: {ones:,} ({ones/len(vals)*100:.2f}%)")
                        print(f"   - Sample values: {sorted(vals.unique())[:10]}")
                    else:
                        print(f"   - ⚠️ No probability column found!")
                except Exception as e:
                    print(f"   - ❌ Error reading: {e}")
            else:
                print(f"   (Not loaded - first file takes priority)")
        else:
            print(f"{i}. ❌ NOT FOUND: {path}")
    
    if loaded_file:
        print(f"\n✅ Confirmed: Dashboard will load: {loaded_file}")
        expected = "data/parquet_export/donors_with_network_features.parquet"
        if expected in loaded_file:
            print(f"✅ This matches the expected file!")
        else:
            print(f"⚠️ WARNING: This does NOT match expected: {expected}")
    else:
        print("\n❌ No valid data file found!")
    
    print("\n" + "-" * 80 + "\n")
    return loaded_file

def check_model_inference_code():
    """Search for model inference code and check for issues"""
    print("=" * 80)
    print("STEP 4: CHECK MODEL INFERENCE CODE")
    print("=" * 80)
    
    root = Path(__file__).resolve().parent.parent
    
    # Search for common inference patterns
    print("\nSearching for model inference code...\n")
    
    patterns_to_check = [
        ("**/*train*.py", "Training scripts"),
        ("**/*predict*.py", "Prediction scripts"),
        ("**/*inference*.py", "Inference scripts"),
        ("final_model/**/*.py", "Final model scripts"),
        ("src/**/*.py", "Source scripts"),
        ("scripts/**/*.py", "Scripts directory")
    ]
    
    found_issues = []
    checked_files = []
    
    for pattern, description in patterns_to_check:
        print(f"\n🔍 Checking {description} ({pattern}):")
        files = list(root.glob(pattern))
        if files:
            print(f"   Found {len(files)} file(s)")
            for file_path in files[:10]:  # Limit to first 10
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        checked_files.append(file_path)
                        
                        # Check for problematic patterns
                        issues = []
                        
                        # Check 1: Using predict() instead of predict_proba()
                        if '.predict(' in content and '.predict_proba(' not in content:
                            if 'predict_proba' not in content:
                                issues.append("⚠️ Uses .predict() - should use .predict_proba() for probabilities")
                        
                        # Check 2: Clipping at 1.0
                        if 'np.clip' in content or '.clip(' in content:
                            if '1.0' in content or '1,' in content or ', 1)' in content:
                                issues.append("⚠️ Contains clipping - may cap values at 1.0")
                        
                        # Check 3: Saving binary instead of probabilities
                        if 'Legacy_Intent_Binary' in content or 'binary' in content.lower():
                            if 'Legacy_Intent_Probability' in content or 'probability' in content.lower():
                                issues.append("ℹ️ References both binary and probability - ensure correct one is saved")
                        
                        # Check 4: Writing to parquet with probability column
                        if 'to_parquet' in content and 'Legacy_Intent_Probability' in content:
                            issues.append("ℹ️ Saves to parquet with Legacy_Intent_Probability column")
                        
                        if issues:
                            found_issues.append((file_path, issues))
                            print(f"\n   📄 {file_path.name}:")
                            for issue in issues:
                                print(f"      {issue}")
                except Exception as e:
                    pass
        else:
            print(f"   No files found")
    
    if found_issues:
        print("\n" + "=" * 80)
        print("SUMMARY OF POTENTIAL ISSUES FOUND:")
        print("=" * 80)
        for file_path, issues in found_issues:
            print(f"\n📄 {file_path}")
            for issue in issues:
                print(f"   {issue}")
        print("\n⚠️ ACTION REQUIRED:")
        print("   1. Review these files to ensure predict_proba() is used")
        print("   2. Check that probabilities (not binary) are saved")
        print("   3. Verify no clipping occurs at 1.0")
    else:
        print("\n✅ No obvious issues found in inference code patterns.")
        print("   This doesn't guarantee correctness - manual review still recommended.")
    
    print("\n" + "-" * 80 + "\n")
    return found_issues

def generate_report():
    """Generate a comprehensive diagnostic report"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DIAGNOSTIC REPORT")
    print("=" * 80)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    clear_streamlit_cache()
    prediction_files = find_prediction_files()
    loaded_file = verify_loaded_file()
    inference_issues = check_model_inference_code()
    
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    recommendations.append("\n1. ✅ Clear Streamlit cache (see Step 1)")
    
    if prediction_files:
        recommendations.append(f"\n2. ⚠️ Found {len(prediction_files)} prediction file(s)")
        recommendations.append("   - Check if these contain the 1.0 values")
        recommendations.append("   - Verify they're not overwriting the main data file")
    else:
        recommendations.append("\n2. ✅ No separate prediction files found")
    
    if loaded_file:
        recommendations.append(f"\n3. ✅ Dashboard loads from: {loaded_file}")
        recommendations.append("   - Verify this file has correct probabilities (max < 1.0)")
        recommendations.append("   - Check file modification date - if recent, model may have overwritten it")
    else:
        recommendations.append("\n3. ❌ No data file found - dashboard will use sample data")
    
    if inference_issues:
        recommendations.append(f"\n4. ⚠️ Found {len(inference_issues)} file(s) with potential issues")
        recommendations.append("   - Review these files for:")
        recommendations.append("     * Use of predict() instead of predict_proba()")
        recommendations.append("     * Clipping operations that might cap at 1.0")
        recommendations.append("     * Saving binary values instead of probabilities")
    else:
        recommendations.append("\n4. ✅ No obvious code issues found")
        recommendations.append("   - Still recommend manual code review")
    
    recommendations.append("\n5. 🔍 Next Steps:")
    recommendations.append("   a. Clear Streamlit cache")
    recommendations.append("   b. Reload dashboard and check if 1.0 values persist")
    recommendations.append("   c. If they persist, check if model inference recently ran")
    recommendations.append("   d. If model ran, verify predict_proba() was used")
    recommendations.append("   e. Check the actual loaded file with: python dashboard/diagnose_probabilities.py")
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "=" * 80)
    print("\nFor more detailed analysis, run:")
    print("  python dashboard/diagnose_probabilities.py")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    generate_report()

