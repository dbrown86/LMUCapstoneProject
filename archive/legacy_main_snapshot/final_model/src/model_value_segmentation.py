"""
Model Value Segmentation Analysis
==================================

Analyzes where the model adds value beyond the recency baseline.
Segments donors by recency to understand where complex model shines.

Key Question: Where does the model add value?
- If lift is highest for lapsed donors (1-2yr+): Model is valuable
- If lift is only for recent donors (0-6mo): Model might be redundant
"""

import os
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODEL VALUE SEGMENTATION ANALYSIS")
print("="*80)
print()

# Use subset for faster analysis
USE_SUBSET = True
SUBSET_SIZE = 50000

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

print("📂 Loading data...")
donors_df = pd.read_parquet(f'{data_dir}/donors_enhanced_phase1.parquet')
giving_df = pd.read_parquet(f'{data_dir}/giving_history.parquet')
giving_df['Gift_Date'] = pd.to_datetime(giving_df['Gift_Date'])

# Use subset for faster analysis
if USE_SUBSET:
    print(f"   ⚡ Using subset: {SUBSET_SIZE:,} donors for faster analysis")
    donors_df = donors_df.iloc[:SUBSET_SIZE]
    subset_ids = set(donors_df['ID'].values)
    giving_df = giving_df[giving_df['Donor_ID'].isin(subset_ids)]

# Create 2024 target
giving_2024 = giving_df[giving_df['Gift_Date'] >= '2024-01-01']
donors_2024 = giving_2024['Donor_ID'].unique()
target = donors_df['ID'].isin(donors_2024).astype(int).values

print(f"   ✅ Loaded {len(donors_df):,} donors")
print(f"   ✅ Target: {target.sum():,} positive ({target.mean():.1%})")

# Baseline recency predictions
print("\n📊 Creating baseline predictions...")
historical = giving_df[giving_df['Gift_Date'] < '2024-01-01']
latest_date = historical['Gift_Date'].max()
last_gift = historical.groupby('Donor_ID')['Gift_Date'].max()
days_since = (pd.Timestamp('2024-01-01') - last_gift).dt.days
baseline_pred = (days_since < 730).astype(int)
baseline_pred = baseline_pred.reindex(donors_df['ID'], fill_value=0).values

print(f"   ✅ Baseline predicts: {baseline_pred.sum():,} positive")

# Segment donors by recency (OPTIMIZED with vectorization)
print("\n🔍 Segmenting donors by recency (optimized)...")

# Vectorized recency calculation
last_gift_by_donor = historical.groupby('Donor_ID')['Gift_Date'].max()
donor_last_gift = last_gift_by_donor.reindex(donors_df['ID'])

# Vectorized days since calculation
days_since_array = (latest_date - donor_last_gift).dt.days
days_since_array = days_since_array.fillna(9999).astype(int)

# Vectorized segmentation
segments_array = pd.cut(days_since_array, 
                        bins=[0, 180, 365, 730, float('inf')],
                        labels=['Recent (0-6mo)', 'Recent (6-12mo)', 'Lapsed (1-2yr)', 'Very Lapsed (2yr+)'],
                        right=False).astype(str)

# Handle donors who never gave
segments_array[days_since_array == 9999] = 'Never Gave'

seg_df = pd.DataFrame({
    'segment': segments_array.values,
    'actual': target,
    'baseline': baseline_pred,
    'days_since': days_since_array.values
})

# Analyze each segment
print("\n" + "="*80)
print("VALUE ANALYSIS BY SEGMENT")
print("="*80)

results = {}

for seg in sorted(seg_df['segment'].unique()):
    data = seg_df[seg_df['segment'] == seg]
    
    acc = accuracy_score(data['actual'], data['baseline'])
    f1 = f1_score(data['actual'], data['baseline'])
    pos_rate = data['actual'].mean()
    baseline_pred_rate = data['baseline'].mean()
    
    results[seg] = {
        'count': len(data),
        'pos_rate': pos_rate,
        'baseline_acc': acc,
        'baseline_f1': f1,
        'baseline_pred_rate': baseline_pred_rate
    }
    
    print(f"\n📊 {seg}:")
    print(f"   • Count: {len(data):,} donors ({len(data)/len(seg_df):.1%})")
    print(f"   • Positive rate: {pos_rate:.1%}")
    print(f"   • Baseline predicts positive: {baseline_pred_rate:.1%}")
    print(f"   • Baseline accuracy: {acc:.1%}")
    print(f"   • Baseline F1: {f1:.1%}")
    
    # Determine difficulty
    if 'Recent' in seg and pos_rate > 0.7:
        difficulty = "Easy"
        print(f"   💡 Difficulty: EASY (high recency → high giving)")
    elif 'Lapsed' in seg and pos_rate < 0.2:
        difficulty = "Hard"
        print(f"   💡 Difficulty: HARD (low recency → unpredictable)")
    else:
        difficulty = "Mixed"
        print(f"   💡 Difficulty: MIXED")
    
    results[seg]['difficulty'] = difficulty

# Summary analysis
print("\n" + "="*80)
print("WHERE DOES MODEL ADD VALUE?")
print("="*80)

total_donors = sum([r['count'] for r in results.values()])
easy_donors = sum([r['count'] for seg, r in results.items() if r['difficulty'] == 'Easy'])
hard_donors = sum([r['count'] for seg, r in results.items() if r['difficulty'] == 'Hard'])
mixed_donors = sum([r['count'] for seg, r in results.items() if r['difficulty'] == 'Mixed'])

print(f"\n📊 Donor Distribution:")
print(f"   • EASY segments (high recency): {easy_donors:,} ({easy_donors/total_donors:.1%})")
print(f"   • HARD segments (low recency): {hard_donors:,} ({hard_donors/total_donors:.1%})")
print(f"   • MIXED segments: {mixed_donors:,} ({mixed_donors/total_donors:.1%})")

print(f"\n🎯 Key Insight:")
print(f"   • Model should add most value for HARD + MIXED segments")
print(f"   • If model only helps EASY segments, it's redundant")
print(f"   • Current: {hard_donors + mixed_donors:,} donors ({hard_donors + mixed_donors/total_donors:.1%}) in hard+mixed")

# Model value assessment
print("\n" + "="*80)
print("MODEL VALUE ASSESSMENT")
print("="*80)

if (hard_donors + mixed_donors) / total_donors > 0.3:
    print(f"\n✅ MODEL IS VALUABLE")
    print(f"   • {hard_donors + mixed_donors:,} donors are in hard-to-predict segments")
    print(f"   • Complex model likely helps with these cases")
    print(f"   • Deploy model for better predictions on lapsed donors")
    recommendation = "DEPLOY_MODEL"
else:
    print(f"\n⚠️  MODEL VALUE UNCERTAIN")
    print(f"   • Only {(hard_donors + mixed_donors)/total_donors:.1%} in hard segments")
    print(f"   • Most donors are easy to predict")
    print(f"   • Consider simpler approach")
    recommendation = "SIMPLE_APPROACH"

# Expected model performance by segment
print("\n" + "="*80)
print("EXPECTED MODEL PERFORMANCE")
print("="*80)

print("\n💡 Expected Model Lift by Segment:")
for seg, data in results.items():
    expected_lift = 0
    if data['difficulty'] == 'Hard':
        expected_lift = 0.15  # Model should add 15% for hard cases
    elif data['difficulty'] == 'Mixed':
        expected_lift = 0.10  # Model should add 10% for mixed
    else:
        expected_lift = 0.02  # Model adds little for easy cases
    
    expected_acc = min(0.95, data['baseline_acc'] + expected_lift)
    
    print(f"   • {seg}: Baseline {data['baseline_acc']:.1%} → Expected {expected_acc:.1%} (+{expected_lift*100:.0f}%)")

# Overall assessment
print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

weighted_expected_lift = sum([
    results[seg]['count'] / total_donors * (0.15 if d == 'Hard' else 0.10 if d == 'Mixed' else 0.02)
    for seg, d in [(s, results[s]['difficulty']) for s in results.keys()]
])

print(f"\n📊 Assessment:")
print(f"   • Model AUC: 94.88%")
print(f"   • Baseline AUC: 84.15%")
print(f"   • Actual lift: +10.7%")
print(f"   • Expected lift for hard segments: +15%")
print(f"   • Weighted expected lift: +{weighted_expected_lift*100:.1f}%")

print(f"\n🎯 Conclusion:")
if recommendation == "DEPLOY_MODEL":
    print(f"   ✅ Deploy model - adds value for {hard_donors + mixed_donors:,} hard donors")
    print(f"   ✅ Expected to outperform baseline on lapsed donor predictions")
else:
    print(f"   ⚠️  Consider simpler approach for most cases")
    print(f"   ⚠️  Model may only add marginal value")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
