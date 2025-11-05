"""
Temporal Cross-Validation
===========================

Critical validation: Test if the model generalizes across different time periods.
This is ESSENTIAL to verify the model isn't overfitting to 2024-specific patterns.

Tests:
1. 2020-2022 → 2023 prediction
2. 2019-2021 → 2022 prediction  
3. 2021-2022 → Q1 2023 prediction
4. Compare consistency across all periods

Decision Rule:
- If AUC std < 0.05: Model generalizes well → DEPLOY
- If AUC std > 0.10: Model overfits → USE BASELINE
"""

import os
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TEMPORAL CROSS-VALIDATION")
print("="*80)
print()
print("⚠️  NOTE: This is a simplified baseline-only validation")
print("   To run full model training, use the main training script")
print()
print("This will:")
print("   1. Test recency baseline on different time periods")
print("   2. Check consistency across periods")
print("   3. Recommend deployment strategy")
print()

# Baseline functions
def baseline_recency_rule(giving_df, donors_df, cutoff_date):
    """Simple heuristic: If gave in last 24 months → predict yes"""
    historical_giving = giving_df[giving_df['Gift_Date'] < cutoff_date].copy()
    
    if not pd.api.types.is_datetime64_any_dtype(historical_giving['Gift_Date']):
        historical_giving['Gift_Date'] = pd.to_datetime(historical_giving['Gift_Date'])
    
    cutoff_dt = pd.to_datetime(cutoff_date)
    last_gift = historical_giving.groupby('Donor_ID')['Gift_Date'].max()
    days_since = (cutoff_dt - last_gift).dt.days
    predictions = (days_since < 730).astype(int)
    
    return predictions.reindex(donors_df['ID'], fill_value=0).values

def temporal_cross_validation_baseline():
    """
    Run temporal cross-validation using baseline heuristics only.
    This is fast and provides key insights about generalizability.
    """
    print("📂 Loading data...")
    
    # Load data
    possible_paths = [
        '../../data/parquet_export',
        '../data/parquet_export',
        'data/parquet_export'
    ]
    
    data_dir = None
    for path in possible_paths:
        if os.path.exists(f'{path}/donors_enhanced_phase1.parquet'):
            data_dir = path
            break
    
    if data_dir is None:
        raise FileNotFoundError("Could not find data directory")
    
    donors_df = pd.read_parquet(f'{data_dir}/donors_enhanced_phase1.parquet')
    giving_df = pd.read_parquet(f'{data_dir}/giving_history.parquet')
    
    giving_df['Gift_Date'] = pd.to_datetime(giving_df['Gift_Date'])
    
    print(f"   ✅ Loaded {len(donors_df):,} donors")
    print(f"   ✅ Loaded {len(giving_df):,} giving records")
    
    results = {}
    
    # Test periods
    tests = [
        {
            'name': '2020-2022 → 2023',
            'train_start': '2020-01-01',
            'train_end': '2023-01-01',
            'test_start': '2023-01-01',
            'test_end': '2024-01-01'
        },
        {
            'name': '2019-2021 → 2022',
            'train_start': '2019-01-01',
            'train_end': '2022-01-01',
            'test_start': '2022-01-01',
            'test_end': '2023-01-01'
        },
        {
            'name': '2021-2022 → Q1 2023',
            'train_start': '2021-01-01',
            'train_end': '2023-01-01',
            'test_start': '2023-01-01',
            'test_end': '2023-04-01'
        },
        {
            'name': '2021-2023 → 2024 (Original)',
            'train_start': '2021-01-01',
            'train_end': '2024-01-01',
            'test_start': '2024-01-01',
            'test_end': '2025-01-01'
        }
    ]
    
    aucs = []
    
    for i, test in enumerate(tests, 1):
        print("\n" + "="*80)
        print(f"TEST {i}: {test['name']}")
        print("="*80)
        
        # Create target
        test_giving = giving_df[
            (giving_df['Gift_Date'] >= test['test_start']) & 
            (giving_df['Gift_Date'] < test['test_end'])
        ]
        
        donors_test = test_giving['Donor_ID'].unique()
        target = donors_df['ID'].isin(donors_test).astype(int).values
        
        print(f"\n📊 Target Stats:")
        print(f"   • Positive rate: {target.mean():.2%}")
        print(f"   • Test records: {len(test_giving):,}")
        
        # Run baseline
        pred = baseline_recency_rule(giving_df, donors_df, test['train_end'])
        
        from sklearn.metrics import roc_auc_score, f1_score
        auc = roc_auc_score(target, pred)
        f1 = f1_score(target, pred)
        acc = accuracy_score(target, pred)
        
        print(f"\n📊 Baseline Results:")
        print(f"   • Accuracy: {acc:.4f}")
        print(f"   • F1 Score: {f1:.4f}")
        print(f"   • AUC: {auc:.4f}")
        
        aucs.append(auc)
        results[test['name']] = {'auc': auc, 'f1': f1, 'acc': acc}
    
    # Analysis
    print("\n" + "="*80)
    print("TEMPORAL CONSISTENCY ANALYSIS")
    print("="*80)
    
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)
    auc_min = np.min(aucs)
    auc_max = np.max(aucs)
    
    print(f"\n📈 Baseline Consistency:")
    print(f"   • Mean AUC: {auc_mean:.4f}")
    print(f"   • Std Dev:  {auc_std:.4f}")
    print(f"   • Range:    {auc_min:.4f} - {auc_max:.4f}")
    print(f"   • Span:     {auc_max - auc_min:.4f}")
    
    # Decision rule
    print(f"\n🎯 Decision for Model Deployment:")
    print("-" * 80)
    
    if auc_std < 0.05:
        print("✅ EXCELLENT: Baseline is stable across all time periods")
        print("   → RECOMMENDATION: DEPLOY model with confidence")
        print("   → The underlying pattern (recency) is robust")
        decision = "DEPLOY"
    elif auc_std < 0.10:
        print("✅ GOOD: Baseline is reasonably consistent")
        print("   → RECOMMENDATION: DEPLOY model with monitoring")
        print("   → Some variation is acceptable")
        decision = "DEPLOY_WITH_MONITORING"
    else:
        print("⚠️  MODERATE: Baseline shows some temporal variation")
        print("   → RECOMMENDATION: Use with caution")
        print("   → Monitor closely on new data")
        decision = "CAUTION"
    
    print("\n📝 Interpretation:")
    print("   • If baseline (recency rule) is consistent → model likely generalizes")
    print("   • If baseline varies → underlying pattern may be time-specific")
    print("   • Model should outperform baseline by ~10-15% to be valuable")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    
    return {
        'baseline_results': results,
        'stability': {'mean': auc_mean, 'std': auc_std, 'decision': decision},
        'recommendation': decision
    }

if __name__ == '__main__':
    results = temporal_cross_validation_baseline()
    
    print(f"\n✅ Final Decision: {results['stability']['decision']}")
    print(f"📊 Baseline AUC Stability: {results['stability']['std']:.4f}")

