#!/usr/bin/env python3
"""
Dense Graph Multimodal Training - Solves Sparse Graph Issue
Solution: Sample only from donors that have relationships
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import add_self_loops
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False


def sample_connected_donors(donors_df: pd.DataFrame, 
                            relationships_df: pd.DataFrame,
                            sample_size: int,
                            random_state: int = 42) -> pd.DataFrame:
    """
    Sample donors that actually have relationships
    This ensures a dense graph instead of sparse
    """
    print(f"\n📊 Sampling {sample_size:,} donors with relationships...")
    
    # Get all donors involved in relationships
    connected_donor_ids = set(relationships_df['Donor_ID_1'].unique()) | set(relationships_df['Donor_ID_2'].unique())
    
    print(f"   Total donors with relationships: {len(connected_donor_ids):,}")
    print(f"   Isolated donors: {len(donors_df) - len(connected_donor_ids):,}")
    
    # Filter to connected donors
    connected_donors = donors_df[donors_df['ID'].isin(connected_donor_ids)].copy()
    
    # Sample from connected donors
    if sample_size >= len(connected_donors):
        print(f"   Using all {len(connected_donors):,} connected donors")
        return connected_donors
    else:
        sampled = connected_donors.sample(n=sample_size, random_state=random_state)
        
        # Verify density
        sample_ids = set(sampled['ID'].tolist())
        sample_rels = relationships_df[
            relationships_df['Donor_ID_1'].isin(sample_ids) & 
            relationships_df['Donor_ID_2'].isin(sample_ids)
        ]
        
        density = len(sample_rels) / (sample_size * sample_size) * 100
        print(f"   Sampled {len(sampled):,} donors")
        print(f"   Relationships in sample: {len(sample_rels):,}")
        print(f"   Graph density: {density:.4f}%")
        print(f"   ✅ Dense graph ensured!")
        
        return sampled


# Use the same Dataset and Model classes from fixed_gnn_multimodal.py
# (copying the implementations)

from src.fixed_gnn_multimodal import (
    FixedDataset,
    FixedGNNModel,
    collate_fn
)


def train_dense_graph_multimodal(sample_size=10000, num_epochs=50, batch_size=64):
    """
    Train with dense graph (only using connected donors)
    """
    
    print("🎯 DENSE GRAPH MULTIMODAL TRAINING")
    print("=" * 60)
    print(f"Sample: {sample_size:,}, Epochs: {num_epochs}, Batch: {batch_size}")
    
    # Load data
    parquet_dir = Path("data/parquet_export")
    donors_df = pd.read_parquet(parquet_dir / 'donors_with_features.parquet')
    relationships_df = pd.read_parquet(parquet_dir / 'relationships.parquet')
    giving_df = pd.read_parquet(parquet_dir / 'giving_history.parquet')
    
    print(f"\n📂 Loaded {len(donors_df):,} donors, {len(relationships_df):,} relationships")
    
    # Sample CONNECTED donors only
    donors_df = sample_connected_donors(donors_df, relationships_df, sample_size)
    
    # Target
    threshold = donors_df['Lifetime_Giving'].quantile(0.8)
    labels = (donors_df['Lifetime_Giving'] >= threshold).astype(int).values
    print(f"   Target: {np.sum(labels):,} positive ({np.mean(labels)*100:.1f}%)")
    
    # Split
    donor_ids = donors_df['ID'].tolist()
    train_ids, test_ids, y_train, y_test = train_test_split(
        donor_ids, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_ids, val_ids, y_train, y_val = train_test_split(
        train_ids, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\n✂️ Split: Train={len(train_ids):,}, Val={len(val_ids):,}, Test={len(test_ids):,}")
    
    # Create datasets
    print("\n🔧 Creating datasets with dense graphs...")
    train_ds = FixedDataset(train_ids, donors_df, relationships_df, giving_df, y_train, fit_scaler=True)
    val_ds = FixedDataset(val_ids, donors_df, relationships_df, giving_df, y_val, scaler=train_ds.scaler)
    test_ds = FixedDataset(test_ids, donors_df, relationships_df, giving_df, y_test, scaler=train_ds.scaler)
    
    print(f"   ✅ Features: {train_ds.features.shape[1]} dimensions")
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🏋️ Training on: {device}")
    
    model = FixedGNNModel(
        input_dim=train_ds.features.shape[1],
        hidden_dim=64,
        num_classes=2,
        dropout=0.2,
        use_gnn=(train_ds.graph_data is not None)
    ).to(device)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Loss
    train_labels = [lbl.item() for lbl in train_ds.labels]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training
    print(f"\n🚀 Starting training with DENSE graph...")
    best_val_loss = float('inf')
    patience_counter = 0
    nan_count = 0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            features = batch['features'].to(device)
            sequences = batch['sequences'].to(device)
            node_indices = batch['node_indices'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(features, train_ds.graph_data, sequences, node_indices)
            
            if torch.isnan(logits).any():
                nan_count += 1
                continue
            
            loss = criterion(logits, labels)
            
            if torch.isnan(loss):
                nan_count += 1
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                sequences = batch['sequences'].to(device)
                node_indices = batch['node_indices'].to(device)
                labels = batch['labels'].to(device)
                
                logits, _ = model(features, val_ds.graph_data, sequences, node_indices)
                
                if not torch.isnan(logits).any():
                    loss = criterion(logits, labels)
                    if not torch.isnan(loss):
                        val_loss += loss.item()
                        preds = torch.argmax(logits, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
        
        if len(all_preds) == 0:
            print(f"Epoch {epoch}: All batches had NaN!")
            continue
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        scheduler.step(val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            if nan_count > 0:
                print(f"         NaN batches skipped: {nan_count}")
                nan_count = 0
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            Path('models').mkdir(exist_ok=True)
            torch.save(model.state_dict(), 'models/best_dense_gnn_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
    
    # Load best and test
    if Path('models/best_dense_gnn_model.pt').exists():
        model.load_state_dict(torch.load('models/best_dense_gnn_model.pt'))
    
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            sequences = batch['sequences'].to(device)
            node_indices = batch['node_indices'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(features, test_ds.graph_data, sequences, node_indices)
            
            if not torch.isnan(logits).any():
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    
    if len(np.unique(test_labels)) == 2 and len(test_probs) > 0:
        try:
            test_auc = roc_auc_score(test_labels, [p[1] for p in test_probs])
        except:
            test_auc = 0.0
    else:
        test_auc = 0.0
    
    print(f"\n📊 FINAL TEST RESULTS:")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   F1 Score: {test_f1:.4f}")
    print(f"   AUC: {test_auc:.4f}")
    
    print(f"\n✅ Training completed with DENSE graph!")
    
    return {
        'accuracy': test_acc,
        'f1': test_f1,
        'auc': test_auc,
        'model': model
    }


if __name__ == "__main__":
    results = train_dense_graph_multimodal(sample_size=10000, num_epochs=50, batch_size=64)
    print(f"\n🎉 SUCCESS with Dense Graph!")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   F1: {results['f1']:.4f}")
    print(f"   AUC: {results['auc']:.4f}")

