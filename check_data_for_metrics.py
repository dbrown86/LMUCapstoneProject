"""
Diagnostic script to check if data is ready for dashboard metrics calculation
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("DIAGNOSTIC: Checking Data for Dashboard Metrics")
print("=" * 80)

# Load parquet file
parquet_path = Path("data/parquet_export/donors_with_network_features.parquet")
if not parquet_path.exists():
    print(f"❌ Parquet file not found: {parquet_path}")
    exit(1)

print(f"\n✅ Loading: {parquet_path}")
df = pd.read_parquet(parquet_path, engine='pyarrow')
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns):,}")

# Check days_since_last
print("\n" + "=" * 80)
print("CHECKING: days_since_last (for baseline AUC)")
print("=" * 80)

if 'days_since_last' in df.columns:
    valid = df['days_since_last'].notna().sum()
    print(f"✅ Found 'days_since_last' column")
    print(f"   Valid values: {valid:,} / {len(df):,}")
    if valid > 0:
        print(f"   Range: {df['days_since_last'].min():.0f} - {df['days_since_last'].max():.0f} days")
    else:
        print("   ⚠️  Column exists but all values are NaN")
elif 'Days_Since_Last_Gift' in df.columns:
    valid = df['Days_Since_Last_Gift'].notna().sum()
    print(f"✅ Found 'Days_Since_Last_Gift' column (can be mapped)")
    print(f"   Valid values: {valid:,} / {len(df):,}")
    if valid > 0:
        print(f"   Range: {df['Days_Since_Last_Gift'].min():.0f} - {df['Days_Since_Last_Gift'].max():.0f} days")
elif 'Last_Gift_Date' in df.columns:
    print(f"✅ Found 'Last_Gift_Date' column (can calculate days_since_last)")
    date_valid = pd.to_datetime(df['Last_Gift_Date'], errors='coerce').notna().sum()
    print(f"   Valid dates: {date_valid:,} / {len(df):,}")
    if date_valid > 0:
        dates = pd.to_datetime(df['Last_Gift_Date'], errors='coerce')
        days = (pd.Timestamp.now() - dates).dt.days.clip(lower=0)
        print(f"   Would calculate range: {days.min():.0f} - {days.max():.0f} days")
else:
    print("❌ No column found for days_since_last")
    print("   Checked: days_since_last, Days_Since_Last_Gift, Last_Gift_Date")

# Check actual_gave / Gave_Again_In_2024
print("\n" + "=" * 80)
print("CHECKING: actual_gave / Gave_Again_In_2024 (for metrics)")
print("=" * 80)

if 'Gave_Again_In_2024' in df.columns:
    pos = df['Gave_Again_In_2024'].sum()
    total = len(df)
    rate = df['Gave_Again_In_2024'].mean()
    unique = sorted(df['Gave_Again_In_2024'].dropna().unique())
    print(f"✅ Found 'Gave_Again_In_2024' column")
    print(f"   Positive (gave again): {pos:,} ({rate:.1%})")
    print(f"   Negative (didn't give): {total - pos:,} ({(1-rate):.1%})")
    print(f"   Unique values: {unique}")
    
    if len(unique) >= 2:
        print(f"   ✅ Both classes present - CAN calculate metrics")
    else:
        print(f"   ❌ Only one class - CANNOT calculate metrics")
        print(f"   ⚠️  All donors have the same outcome value")
else:
    print("❌ 'Gave_Again_In_2024' column NOT found")
    print("   This needs to be created from giving history")

# Check Will_Give_Again_Probability
print("\n" + "=" * 80)
print("CHECKING: Will_Give_Again_Probability (predictions)")
print("=" * 80)

if 'Will_Give_Again_Probability' in df.columns:
    valid = df['Will_Give_Again_Probability'].notna().sum()
    print(f"✅ Found 'Will_Give_Again_Probability' column")
    print(f"   Valid values: {valid:,} / {len(df):,}")
    if valid > 0:
        print(f"   Range: {df['Will_Give_Again_Probability'].min():.3f} - {df['Will_Give_Again_Probability'].max():.3f}")
        print(f"   Mean: {df['Will_Give_Again_Probability'].mean():.3f}")
else:
    print("❌ 'Will_Give_Again_Probability' column NOT found")

# Summary and recommendations
print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

has_days = 'days_since_last' in df.columns or 'Days_Since_Last_Gift' in df.columns or 'Last_Gift_Date' in df.columns
has_outcome = 'Gave_Again_In_2024' in df.columns
has_predictions = 'Will_Give_Again_Probability' in df.columns

if has_days and has_outcome and has_predictions:
    if 'Gave_Again_In_2024' in df.columns:
        unique_outcomes = sorted(df['Gave_Again_In_2024'].dropna().unique())
        if len(unique_outcomes) >= 2:
            print("✅ ALL REQUIREMENTS MET!")
            print("   - days_since_last: Available or can be calculated")
            print("   - Gave_Again_In_2024: Exists with both classes")
            print("   - Will_Give_Again_Probability: Exists")
            print("\n   The dashboard should be able to calculate:")
            print("   - Baseline AUC ✓")
            print("   - Lift vs Baseline ✓")
            print("   - All other metrics ✓")
        else:
            print("⚠️  DATA ISSUE: Gave_Again_In_2024 has only one class")
            print("   All donors have the same outcome value")
            print("\n   ACTION REQUIRED:")
            print("   1. Check giving history for 2024")
            print("   2. Verify that some donors gave and some didn't")
            print("   3. Recompute Gave_Again_In_2024 if needed")
    else:
        print("✅ Most requirements met, but missing Gave_Again_In_2024")
        print("\n   ACTION REQUIRED:")
        print("   Run: python final_model/src/generate_will_give_again_predictions.py")
        print("   This will create Gave_Again_In_2024 from giving history")
elif not has_predictions:
    print("❌ MISSING: Will_Give_Again_Probability")
    print("\n   ACTION REQUIRED:")
    print("   Run: python final_model/src/generate_will_give_again_predictions.py")
    print("   This will generate predictions for all donors")
elif not has_outcome:
    print("❌ MISSING: Gave_Again_In_2024")
    print("\n   ACTION REQUIRED:")
    print("   The dashboard will compute this from giving history automatically")
    print("   OR run inference script to pre-compute it:")
    print("   python final_model/src/generate_will_give_again_predictions.py")
elif not has_days:
    print("❌ MISSING: days_since_last or Last_Gift_Date")
    print("\n   This is unusual - check your parquet file")
    print("   The dashboard will generate sample values as fallback")

print("\n" + "=" * 80)

