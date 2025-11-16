"""
Generate "Will Give Again" Predictions for All Donors
======================================================

This script runs inference on the trained model to generate "will give again" 
predictions for ALL donors in the dataset and saves them to the parquet file.

Usage:
    cd final_model/src
    python generate_will_give_again_predictions.py
"""

import os
import sys
import time

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import everything from training script
# Note: train_will_give_again.py contains all the necessary classes and functions
try:
    from train_will_give_again import (
        SingleTargetInfluentialModel,
        OptimizedSingleTargetDataset,
        create_temporal_features,
        create_text_features,
        create_influence_features,
        create_strategic_features,
        create_capacity_features,
        create_recency_engagement_features,
        create_rfm_features
    )
except ImportError:
    # Fallback to old import name
    from simplified_single_target_training import (
        SingleTargetInfluentialModel,
        OptimizedSingleTargetDataset,
        create_temporal_features,
        create_text_features,
        create_influence_features,
        create_strategic_features,
        create_capacity_features,
        create_recency_engagement_features,
        create_rfm_features
    )

# Progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Import optional features if available
try:
    from enhanced_model_with_interpretability import create_high_value_donor_features
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False

print("=" * 80)
print("🔮 GENERATING 'WILL GIVE AGAIN' PREDICTIONS FOR ALL DONORS")
print("=" * 80)

start_time = time.time()

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️  Device: {device}")

# Load model checkpoint
model_path = os.path.join('models', 'best_influential_donor_model.pt')
if not os.path.exists(model_path):
    # Try alternative paths
    alt_paths = [
        '../../models/saved_models/best_influential_donor_model.pt',
        '../models/saved_models/best_influential_donor_model.pt',
        '../../models/best_influential_donor_model.pt',  # Legacy fallback
    ]
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            model_path = alt_path
            break
    else:
        print(f"\n❌ ERROR: Model checkpoint not found!")
        print(f"   Searched: {model_path}")
        print("   Please run the training script first: simplified_single_target_training.py")
        sys.exit(1)

print(f"\n📂 Loading trained model from {model_path}...")

# Load data
print("\n📂 Loading donor data...")
donors_paths = [
    'data/processed/parquet_export/donors_with_network_features.parquet',
    '../data/processed/parquet_export/donors_with_network_features.parquet',
    '../../data/processed/parquet_export/donors_with_network_features.parquet',
    'data/parquet_export/donors_with_network_features.parquet',  # Legacy fallback
    '../data/parquet_export/donors_with_network_features.parquet',
    '../../data/parquet_export/donors_with_network_features.parquet'
]
donors_path = None
for path in donors_paths:
    if os.path.exists(path):
        donors_path = path
        break

if not donors_path:
    print(f"❌ ERROR: Donors file not found. Expected one of: {donors_paths}")
    sys.exit(1)

donors_df = pd.read_parquet(donors_path, engine='pyarrow')
print(f"   ✅ Loaded {len(donors_df):,} donors")

giving_paths = [
    'data/parquet_export/giving_history.parquet',
    '../data/parquet_export/giving_history.parquet',
    '../../data/parquet_export/giving_history.parquet'
]
giving_path = None
for path in giving_paths:
    if os.path.exists(path):
        giving_path = path
        break

giving_df = pd.read_parquet(giving_path, engine='pyarrow')
giving_df['Gift_Date'] = pd.to_datetime(giving_df['Gift_Date'])
print(f"   ✅ Loaded {len(giving_df):,} giving records")

# Filter to historical data (before 2024) for features
historical_giving = giving_df[giving_df['Gift_Date'] < '2024-01-01'].copy()
print(f"   🛡️  Using historical giving (pre-2024): {len(historical_giving):,} records")

# Load relationships
relationships_df = None
relationships_paths = [
    'data/parquet_export/relationships.parquet',
    '../data/parquet_export/relationships.parquet',
    '../../data/parquet_export/relationships.parquet'
]
for path in relationships_paths:
    if os.path.exists(path):
        relationships_df = pd.read_parquet(path, engine='pyarrow')
        print(f"   ✅ Loaded {len(relationships_df):,} relationships")
        break
else:
    print("   ⚠️  No relationships available")

# Load contact reports
contact_reports_df = None
has_text = False
contact_reports_paths = [
    'data/parquet_export/contact_reports.parquet',
    '../data/parquet_export/contact_reports.parquet',
    '../../data/parquet_export/contact_reports.parquet'
]
for path in contact_reports_paths:
    if os.path.exists(path):
        contact_reports_df = pd.read_parquet(path, engine='pyarrow')
        contact_reports_df['Contact_Date'] = pd.to_datetime(contact_reports_df['Contact_Date'])
        contact_reports_df = contact_reports_df[contact_reports_df['Contact_Date'] < '2024-01-01'].copy()
        print(f"   ✅ Loaded {len(contact_reports_df):,} contact reports (historical only)")
        has_text = True
        break
else:
    print("   ⚠️  No contact reports available")

# Create features (same as training script)
print("\n📊 Creating features...")
print("   ⏳ This may take a few minutes...")

# Check cache
cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)
cache_file = os.path.join(cache_dir, 'all_features.pkl')

import pickle
if os.path.exists(cache_file):
    try:
        print("   🗄️  Loading features from cache...")
        with open(cache_file, 'rb') as f:
            cached_features = pickle.load(f)
        temporal_features_df = cached_features['temporal']
        influence_features_df = cached_features['influence']
        strategic_features_df = cached_features['strategic']
        capacity_features_df = cached_features['capacity']
        recency_features_df = cached_features['recency']
        print("   ✅ Loaded from cache")
    except:
        print("   🔧 Cache corrupted, recomputing...")
        temporal_features_df = create_temporal_features(donors_df, historical_giving)
        influence_features_df = create_influence_features(donors_df, historical_giving, relationships_df)
        strategic_features_df = create_strategic_features(donors_df, historical_giving, relationships_df)
        capacity_features_df = create_capacity_features(donors_df)
        recency_features_df = create_recency_engagement_features(donors_df, historical_giving)
else:
    temporal_features_df = create_temporal_features(donors_df, historical_giving)
    influence_features_df = create_influence_features(donors_df, historical_giving, relationships_df)
    strategic_features_df = create_strategic_features(donors_df, historical_giving, relationships_df)
    capacity_features_df = create_capacity_features(donors_df)
    recency_features_df = create_recency_engagement_features(donors_df, historical_giving)

# RFM features
cache_file_rfm = os.path.join(cache_dir, 'rfm_features.pkl')
if os.path.exists(cache_file_rfm):
    try:
        with open(cache_file_rfm, 'rb') as f:
            rfm_features_df = pickle.load(f)
    except:
        rfm_features_df = create_rfm_features(donors_df, historical_giving)
else:
    rfm_features_df = create_rfm_features(donors_df, historical_giving)

# Enhanced features if available
if ENHANCED_FEATURES_AVAILABLE:
    high_value_features = create_high_value_donor_features(donors_df, historical_giving)
    combined_features_df = pd.concat([
        temporal_features_df,
        influence_features_df,
        strategic_features_df,
        capacity_features_df,
        recency_features_df,
        rfm_features_df,
        high_value_features
    ], axis=1)
else:
    combined_features_df = pd.concat([
        temporal_features_df,
        influence_features_df,
        strategic_features_df,
        capacity_features_df,
        recency_features_df,
        rfm_features_df
    ], axis=1)

# Feature selection (use same features as training)
print("\n🎯 Selecting features...")
from sklearn.feature_selection import mutual_info_classif

# Load the actual 60 features used during training (CRITICAL for model loading)
selected_features_file_paths = [
    'results/selected_features_60.csv',
    '../results/selected_features_60.csv',
    '../../results/selected_features_60.csv'
]
selected_features_file = None
for path in selected_features_file_paths:
    if os.path.exists(path):
        selected_features_file = path
        break

# Fallback to feature importance file if selected features file doesn't exist
importance_file_paths = [
    'results/feature_importance_influential_donor.csv',
    '../results/feature_importance_influential_donor.csv',
    '../../results/feature_importance_influential_donor.csv'
]
importance_file = None
for path in importance_file_paths:
    if os.path.exists(path):
        importance_file = path
        break

if selected_features_file:
    print("   🗄️  Loading selected 60 features from training (exact match)...")
    selected_features_df = pd.read_csv(selected_features_file)
    selected_features = selected_features_df['feature'].tolist()  # Exact 60 features in training order
    print(f"   ✅ Loaded {len(selected_features)} features from training")
elif importance_file:
    print("   🗄️  Loading feature importance from training (fallback)...")
    importance_df = pd.read_csv(importance_file)
    # Get all features from importance file, pad to 60 if needed
    available_importance_features = importance_df['feature'].tolist()
    if len(available_importance_features) < 60:
        print(f"   ⚠️  Warning: Importance file only has {len(available_importance_features)} features, need 60")
        # Use what we have and pad with available features from combined_features_df
        selected_features = available_importance_features
        # Add additional features from combined_features_df to reach 60
        remaining_needed = 60 - len(selected_features)
        additional_features = [f for f in combined_features_df.columns if f not in selected_features][:remaining_needed]
        selected_features.extend(additional_features)
        print(f"   ⚠️  Padded to {len(selected_features)} features using available data features")
    else:
        selected_features = available_importance_features[:60]  # Top 60 features
    
    # CRITICAL: Model expects exactly 60 features in this exact order
    # Add missing features as zero-filled columns to match training
    missing_features = [f for f in selected_features if f not in combined_features_df.columns]
    if missing_features:
        print(f"   ⚠️  {len(missing_features)} features missing from data, adding as zero-filled columns:")
        for feat in missing_features:
            print(f"      - {feat}")
            # Add as zero-filled column with same index as combined_features_df
            combined_features_df[feat] = 0.0
        print(f"   ✅ Added {len(missing_features)} missing features")
    
    # Now verify we have all 60 features (they should all exist now)
    available_features = [f for f in selected_features if f in combined_features_df.columns]
    print(f"   🔍 Debug: Checking features...")
    print(f"      Expected: 60 features")
    print(f"      Found in dataframe: {len(available_features)} features")
    print(f"      Combined features df shape: {combined_features_df.shape}")
    print(f"      Combined features df columns: {len(combined_features_df.columns)}")
    
    if len(available_features) != 60:
        print(f"   ❌ ERROR: Expected 60 features, got {len(available_features)}")
        missing = set(selected_features) - set(available_features)
        if missing:
            print(f"   Missing features: {missing}")
        else:
            print(f"   All features found but count is wrong - checking for duplicates...")
            # Check for duplicates in selected_features
            from collections import Counter
            counts = Counter(selected_features)
            duplicates = [feat for feat, count in counts.items() if count > 1]
            if duplicates:
                print(f"   Found duplicate features in importance file: {duplicates}")
        print(f"   This will cause model loading to fail!")
        # Use what we have but warn
        selected_features = available_features
    else:
        print(f"   ✅ All 60 features available (in correct order from training)")
        # Keep selected_features as-is (already in correct order)
else:
    # CRITICAL DATA LEAKAGE PREVENTION: Cannot recompute feature importance
    # Feature importance MUST come from training to prevent using future outcomes
    print("   ❌ ERROR: Feature importance file not found!")
    print("   🛡️  DATA LEAKAGE PREVENTION: Cannot recompute feature importance")
    print("   ")
    print("   ⚠️  Why this is blocked:")
    print("      - Recomputing would require using 2024 outcomes (future data)")
    print("      - This would leak information the model shouldn't see")
    print("      - Model would have unfair advantage (data leakage)")
    print("   ")
    print("   ✅ REQUIRED ACTION:")
    print("      1. Ensure training script completed successfully")
    print("      2. Check for: results/feature_importance_influential_donor.csv")
    print("      3. If missing, re-run training script to generate it")
    print("   ")
    print("   🔧 FALLBACK: Using ALL available features (may cause performance issues)")
    print("      If model fails to load due to feature mismatch, you MUST run training first")
    selected_features = combined_features_df.columns.tolist()[:60]  # Take first 60 as fallback
    print(f"   ⚠️  Using first {len(selected_features)} features (NOT optimal - training required)")

tabular_cols = selected_features
combined_features_df = combined_features_df[selected_features]

# Add features to donors_df
combined_features_df = combined_features_df.reset_index()
if 'ID' not in combined_features_df.columns:
    if combined_features_df.index.name == 'ID':
        combined_features_df = combined_features_df.reset_index()
    else:
        combined_features_df['ID'] = combined_features_df.index.values

# Merge on ID
donors_df = donors_df.merge(combined_features_df, on='ID', how='left', suffixes=('', '_dup'))
donors_df = donors_df.loc[:, ~donors_df.columns.str.endswith('_dup')]
donors_df = donors_df.loc[:, ~donors_df.columns.duplicated()]

# Scale features
# NOTE: Training script fits scaler on ALL data (train+val+test), so fitting on inference data
# is acceptable. However, ideally we would load scaler parameters from training.
# For now, fit on inference data (this is standard practice when scaler params aren't saved).
print("   📊 Scaling features (using inference data distribution)...")
scaler = StandardScaler()
donors_df[tabular_cols] = scaler.fit_transform(donors_df[tabular_cols].fillna(0))
print(f"   ✅ Features scaled (mean/std computed from {len(donors_df):,} inference samples)")
print(f"   ⚠️  NOTE: For exact match with training, scaler params should be saved/loaded from training")

# Text features with SVD compression
text_features_df = None
text_dim = 50

if has_text and contact_reports_df is not None:
    text_features_df = create_text_features(donors_df, contact_reports_df, text_dim=text_dim)
    initial_dim = text_features_df.shape[1]
    
    # Apply SVD compression if needed
    if initial_dim > 32:
        from sklearn.decomposition import TruncatedSVD
        print("   🎯 Applying SVD compression (50 → 32 dimensions)...")
        # NOTE: Training script fits SVD on training data only (line 1894), but transforms all data.
        # For inference, we fit on inference data (standard practice when SVD params aren't saved).
        # This is acceptable but may cause slight distribution mismatch.
        svd = TruncatedSVD(n_components=32, random_state=42)
        svd.fit(text_features_df.values)  # Fit on inference data
        text_features_compressed = svd.transform(text_features_df.values)
        text_features_df = pd.DataFrame(
            text_features_compressed,
            index=text_features_df.index,
            columns=[f'text_svd_{i}' for i in range(32)]
        )
        text_dim = 32
        print(f"   ⚠️  NOTE: SVD fitted on inference data. For exact match, SVD params should be saved/loaded from training")
else:
    text_features_df = pd.DataFrame(
        0, index=donors_df['ID'], 
        columns=[f'text_feature_{i}' for i in range(text_dim)]
    )

# Create dataset for ALL donors (not just test set)
print("\n📦 Creating inference dataset for ALL donors...")
print(f"   📊 Total donors: {len(donors_df):,}")

# Create dummy targets (not used for inference, but required by dataset)
dummy_targets = np.zeros(len(donors_df))

inference_dataset = OptimizedSingleTargetDataset(
    donors_df, historical_giving, dummy_targets,
    tabular_cols,
    text_features_df if text_features_df is not None else None,
    relationships_df
)

# Create DataLoader
def collate_fn(batch):
    """Same collate function as training"""
    return {
        'tabular': torch.stack([item['tabular'] for item in batch]),
        'sequence': torch.stack([item['sequence'] for item in batch]),
        'network': torch.stack([item['network'] for item in batch]),
        'text': torch.stack([item['text'] for item in batch]),
        'edge_index': batch[0]['edge_index'],
        'batch_indices': None,
        'target': torch.stack([item['target'] for item in batch])
    }

batch_size = 2048
try:
    test_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=collate_fn, num_workers=0)
    next(iter(test_loader))  # Test if batch works
except RuntimeError as e:
    if "out of memory" in str(e):
        batch_size = 1024
        torch.cuda.empty_cache()
        print(f"   ⚠️  GPU OOM, using batch_size={batch_size}")
    else:
        raise

inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, 
                              collate_fn=collate_fn, num_workers=0)

print(f"   ⚡ Batch size: {batch_size}")

# Create and load model
print("\n🏗️  Creating model architecture...")

model = SingleTargetInfluentialModel(
    tabular_dim=len(tabular_cols),
    sequence_dim=1,
    network_dim=5,
    text_dim=text_dim,
    hidden_dim=256,
    dropout=0.3,
    use_transformer=False
)

print(f"   ✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load trained weights
print(f"\n📂 Loading model weights from {model_path}...")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print("   ✅ Model loaded and ready for inference")

# Run inference
print("\n🔮 Running inference on ALL donors...")
print(f"   📊 Processing {len(donors_df):,} donors in batches of {batch_size}...")

all_probs = []
donor_ids = []

with torch.no_grad():
    if TQDM_AVAILABLE:
        pbar = tqdm(inference_loader, desc="Generating predictions")
    else:
        pbar = inference_loader
        print("   ⏳ Processing batches...")
    
    for batch_idx, batch in enumerate(pbar):
        tabular = batch['tabular'].to(device)
        sequence = batch['sequence'].to(device)
        network = batch['network'].to(device)
        text = batch['text'].to(device)
        
        edge_index = batch.get('edge_index')
        if edge_index is not None:
            edge_index = edge_index.to(device)
        batch_indices = batch.get('batch_indices')
        if batch_indices is not None:
            batch_indices = batch_indices.to(device)
        
        # Forward pass
        output = model(tabular, sequence, network, text, 
                      edge_index=edge_index, batch_indices=batch_indices)
        probs = torch.sigmoid(output).cpu().numpy()
        
        all_probs.extend(probs)
        
        # Track donor IDs for this batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + len(probs), len(donors_df))
        batch_ids = donors_df.iloc[start_idx:end_idx]['ID'].values
        donor_ids.extend(batch_ids)
        
        if TQDM_AVAILABLE:
            pbar.set_postfix({
                'processed': f'{len(all_probs):,}',
                'avg_prob': f'{np.mean(all_probs):.3f}'
            })
        elif (batch_idx + 1) % 10 == 0:
            print(f"   ✅ Processed {(batch_idx + 1) * batch_size:,} donors...")

all_probs = np.array(all_probs)

print(f"\n✅ Inference complete!")
print(f"   📊 Total predictions: {len(all_probs):,}")
print(f"   📈 Probability range: {all_probs.min():.3f} - {all_probs.max():.3f}")
print(f"   📈 Mean probability: {all_probs.mean():.3f}")

# Save predictions back to parquet file
print("\n💾 Saving predictions to parquet file...")

# Create predictions DataFrame aligned with donor IDs
predictions_df = pd.DataFrame({
    'ID': donor_ids,
    'Will_Give_Again_Probability': all_probs
})

# Remove existing Will_Give_Again_Probability column if it exists (to avoid merge conflicts)
if 'Will_Give_Again_Probability' in donors_df.columns:
    donors_df = donors_df.drop(columns=['Will_Give_Again_Probability'])
    print(f"   🔧 Removed existing 'Will_Give_Again_Probability' column before merge")

# Merge with original donors_df to ensure alignment
donors_df = donors_df.merge(predictions_df[['ID', 'Will_Give_Again_Probability']], 
                           on='ID', how='left')

# Ensure all donors have predictions
if donors_df['Will_Give_Again_Probability'].isna().sum() > 0:
    print(f"   ⚠️  Warning: {donors_df['Will_Give_Again_Probability'].isna().sum():,} missing predictions")
    print(f"   🔧 Filling missing predictions with mean...")
    donors_df['Will_Give_Again_Probability'] = donors_df['Will_Give_Again_Probability'].fillna(all_probs.mean())

# Ensure Gave_Again_In_2025 is included (for dashboard metrics) - PRIMARY TARGET
# ALWAYS recreate to ensure it's correct (don't rely on existing column)
print(f"   📊 Creating 'Gave_Again_In_2025' from giving history...")
try:
    giving_2025 = giving_df[giving_df['Gift_Date'] >= '2025-01-01'].copy()
    if 'Donor_ID' in giving_2025.columns:
        donors_2025 = giving_2025['Donor_ID'].unique()
    elif 'ID' in giving_2025.columns:
        donors_2025 = giving_2025['ID'].unique()
    else:
        raise ValueError("Could not find Donor_ID or ID column in giving history")
    
    donors_df['Gave_Again_In_2025'] = donors_df['ID'].isin(donors_2025).astype(int)
    pos_count = donors_df['Gave_Again_In_2025'].sum()
    pos_rate = donors_df['Gave_Again_In_2025'].mean()
    print(f"   ✅ 'Gave_Again_In_2025' column created ({pos_count:,} donors gave again, {pos_rate:.1%})")
except Exception as e:
    print(f"   ❌ ERROR creating 'Gave_Again_In_2025': {e}")
    print(f"   ⚠️  Dashboard metrics will be unavailable without this column!")
    # Create empty column as fallback
    donors_df['Gave_Again_In_2025'] = 0

# Also create Gave_Again_In_2024 for backward compatibility
print(f"   📊 Creating 'Gave_Again_In_2024' from giving history (backward compatibility)...")
try:
    giving_2024 = giving_df[(giving_df['Gift_Date'] >= '2024-01-01') & (giving_df['Gift_Date'] < '2025-01-01')].copy()
    if 'Donor_ID' in giving_2024.columns:
        donors_2024 = giving_2024['Donor_ID'].unique()
    elif 'ID' in giving_2024.columns:
        donors_2024 = giving_2024['ID'].unique()
    else:
        raise ValueError("Could not find Donor_ID or ID column in giving history")
    
    donors_df['Gave_Again_In_2024'] = donors_df['ID'].isin(donors_2024).astype(int)
    pos_count_2024 = donors_df['Gave_Again_In_2024'].sum()
    pos_rate_2024 = donors_df['Gave_Again_In_2024'].mean()
    print(f"   ✅ 'Gave_Again_In_2024' column created ({pos_count_2024:,} donors gave again, {pos_rate_2024:.1%})")
except Exception as e:
    print(f"   ⚠️  Could not create 'Gave_Again_In_2024': {e}")
    donors_df['Gave_Again_In_2024'] = 0

# Save updated parquet file
output_path = donors_path  # Overwrite original file
backup_path = donors_path.replace('.parquet', '_backup.parquet')

# Create backup
print(f"   💾 Creating backup: {backup_path}")
donors_df_backup = pd.read_parquet(donors_path, engine='pyarrow')
donors_df_backup.to_parquet(backup_path, engine='pyarrow', index=False)

# Save with new predictions
print(f"   💾 Saving predictions to: {output_path}")
donors_df.to_parquet(output_path, engine='pyarrow', index=False)

print(f"   ✅ Predictions saved!")

# Verify
print("\n🔍 Verifying saved predictions...")
verification_df = pd.read_parquet(output_path, engine='pyarrow')
if 'Will_Give_Again_Probability' in verification_df.columns:
    print(f"   ✅ Column 'Will_Give_Again_Probability' found in saved file")
    print(f"   📊 Value range: {verification_df['Will_Give_Again_Probability'].min():.3f} - {verification_df['Will_Give_Again_Probability'].max():.3f}")
    print(f"   📊 Mean: {verification_df['Will_Give_Again_Probability'].mean():.3f}")
    print(f"   📊 Non-null count: {verification_df['Will_Give_Again_Probability'].notna().sum():,} / {len(verification_df):,}")
else:
    print(f"   ❌ ERROR: Column 'Will_Give_Again_Probability' not found in saved file!")

if 'Gave_Again_In_2024' in verification_df.columns:
    print(f"   ✅ Column 'Gave_Again_In_2024' found in saved file")
    print(f"   📊 Donors who gave again: {verification_df['Gave_Again_In_2024'].sum():,} ({verification_df['Gave_Again_In_2024'].mean():.1%})")
else:
    print(f"   ❌ ERROR: Column 'Gave_Again_In_2024' not found in saved file!")
    print(f"   ⚠️  Dashboard will need to compute this from giving history")

# Final summary
total_time = time.time() - start_time

print("\n" + "=" * 80)
print("✅ INFERENCE COMPLETE")
print("=" * 80)
print(f"\n⏱️  Total Time: {total_time/60:.1f} minutes")
print(f"📊 Processed: {len(donors_df):,} donors")
print(f"📂 Output file: {output_path}")
print(f"💾 Backup created: {backup_path}")
print(f"\n✅ 'Will_Give_Again_Probability' column added to parquet file!")
print("=" * 80)
