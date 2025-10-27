#!/usr/bin/env python3
"""
STABLE Multimodal Training - Addresses Gradient Explosion
Key fixes:
1. Log transform + RobustScaler + clipping
2. Proper weight initialization
3. Batch normalization everywhere
4. Lower learning rate (0.0001)
5. Gradient clipping (max_norm=1.0)
6. Simplified LSTM (1 layer instead of 2)
7. Check for NaN and skip batch if detected
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
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False


class StableDataset(Dataset):
    """Dataset with robust feature handling"""
    
    def __init__(self, donor_ids, donors_df, relationships_df, giving_df,
                 labels, scaler=None, fit_scaler=False, max_seq_len=20):
        
        self.donor_ids = donor_ids
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.scaler = scaler
        self.max_seq_len = max_seq_len
        
        # Donor ID to index mapping
        self.donor_id_to_idx = {did: idx for idx, did in enumerate(donor_ids)}
        
        # Filter donors
        self.donors_df = donors_df[donors_df['ID'].isin(donor_ids)].copy()
        self.donors_df = self.donors_df.set_index('ID')
        
        # Prepare features
        self.features = self._prepare_features(fit_scaler)
        
        # Store dataframes
        self.relationships_df = relationships_df
        self.giving_df = giving_df
        
        # Build graph
        self.graph_data = self._build_graph()
        
        # Index giving history
        self.giving_by_donor = {}
        if giving_df is not None and len(giving_df) > 0:
            for did in donor_ids:
                donor_giving = giving_df[giving_df['Donor_ID'] == did]
                if len(donor_giving) > 0:
                    self.giving_by_donor[did] = donor_giving.copy()
    
    def _prepare_features(self, fit_scaler):
        """Prepare features with extreme value handling"""
        
        # Define features
        feature_cols = {
            'Lifetime_Giving': 'log',
            'Last_Gift': 'log',
            'Total_Yr_Giving_Count': 'standard',
            'Consecutive_Yr_Giving_Count': 'standard',
            'Engagement_Score': 'standard',
            'Legacy_Intent_Probability': 'standard',
            'Legacy_Intent_Binary': 'standard',
            'Estimated_Age': 'standard',
            'Class_Year': 'standard',
            'Parent_Year': 'standard'
        }
        
        feature_list = []
        
        for col, transform in feature_cols.items():
            if col not in self.donors_df.columns:
                continue
            
            data = self.donors_df[col].copy()
            
            # Fill NaN with median
            data = data.fillna(data.median() if not data.isna().all() else 0)
            
            # Transform
            if transform == 'log':
                data = np.log1p(np.clip(data, 0, None))
            
            feature_list.append(data.values.reshape(-1, 1))
        
        if len(feature_list) == 0:
            raise ValueError("No features available")
        
        features = np.hstack(feature_list)
        
        # Robust scaling
        if fit_scaler:
            self.scaler = RobustScaler()
            features = self.scaler.fit_transform(features)
        elif self.scaler:
            features = self.scaler.transform(features)
        
        # Clip extreme values
        features = np.clip(features, -5, 5)
        
        # Check for NaN/inf
        if np.isnan(features).any() or np.isinf(features).any():
            print("⚠️ Warning: NaN/inf detected in features, replacing with 0")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _build_graph(self):
        """Build graph structure"""
        if self.relationships_df is None or not PYTORCH_GEOMETRIC_AVAILABLE:
            return None
        
        rel_df = self.relationships_df[
            self.relationships_df['Donor_ID_1'].isin(self.donor_ids) & 
            self.relationships_df['Donor_ID_2'].isin(self.donor_ids)
        ].copy()
        
        if len(rel_df) == 0:
            return None
        
        rel_df['idx_1'] = rel_df['Donor_ID_1'].map(self.donor_id_to_idx)
        rel_df['idx_2'] = rel_df['Donor_ID_2'].map(self.donor_id_to_idx)
        rel_df = rel_df.dropna(subset=['idx_1', 'idx_2'])
        
        if len(rel_df) == 0:
            return None
        
        edge_index = torch.tensor([
            rel_df['idx_1'].tolist() + rel_df['idx_2'].tolist(),
            rel_df['idx_2'].tolist() + rel_df['idx_1'].tolist()
        ], dtype=torch.long)
        
        return Data(x=self.features, edge_index=edge_index, num_nodes=len(self.donor_ids))
    
    def _get_sequence(self, donor_id):
        """Get giving sequence"""
        if donor_id not in self.giving_by_donor:
            return torch.zeros(self.max_seq_len, 3), 0
        
        donor_giving = self.giving_by_donor[donor_id]
        if 'Gift_Date' in donor_giving.columns:
            donor_giving = donor_giving.sort_values('Gift_Date')
        
        features = []
        for _, row in donor_giving.iterrows():
            amount = np.log1p(max(0, row.get('Gift_Amount', 0)))
            year = row.get('Campaign_Year', 2020) / 2020.0
            anon = 1 if row.get('Anonymous', False) else 0
            features.append([amount, year, anon])
        
        seq = torch.tensor(features, dtype=torch.float32)
        actual_len = min(len(seq), self.max_seq_len)
        
        if len(seq) < self.max_seq_len:
            padding = torch.zeros(self.max_seq_len - len(seq), 3)
            seq = torch.cat([seq, padding], dim=0)
        else:
            seq = seq[-self.max_seq_len:]
        
        return seq, actual_len
    
    def __len__(self):
        return len(self.donor_ids)
    
    def __getitem__(self, idx):
        donor_id = self.donor_ids[idx]
        features = self.features[idx]
        label = self.labels[idx]
        sequence, seq_len = self._get_sequence(donor_id)
        
        return {
            'features': features,
            'sequence': sequence,
            'seq_len': seq_len,
            'node_idx': idx,
            'label': label
        }


class StableMultimodalModel(nn.Module):
    """Stable multimodal model with proper initialization"""
    
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout=0.3, use_gnn=True):
        super().__init__()
        
        self.use_gnn = use_gnn and PYTORCH_GEOMETRIC_AVAILABLE
        
        # Tabular encoder
        self.tab_enc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # GNN
        if self.use_gnn:
            self.gnn1 = GCNConv(input_dim, hidden_dim)
            self.gnn2 = GCNConv(hidden_dim, hidden_dim)
            self.gnn_bn1 = nn.BatchNorm1d(hidden_dim)
            self.gnn_bn2 = nn.BatchNorm1d(hidden_dim)
        else:
            self.graph_enc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        
        # LSTM (simplified to 1 layer)
        self.lstm = nn.LSTM(3, hidden_dim, num_layers=1, batch_first=True, dropout=0)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier
        self.classifier = nn.Linear(64, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)  # Lower gain
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=0.5)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, features, graph_data, sequences, seq_lens, node_indices):
        device = features.device
        
        # Tabular
        tab_out = self.tab_enc(features)
        
        # Graph
        if self.use_gnn and graph_data is not None:
            x = graph_data.x.to(device)
            edge_index = graph_data.edge_index.to(device)
            
            x = self.gnn1(x, edge_index)
            x = self.gnn_bn1(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
            
            x = self.gnn2(x, edge_index)
            x = self.gnn_bn2(x)
            x = F.relu(x)
            
            graph_out = x[node_indices]
        else:
            graph_out = self.graph_enc(features) if hasattr(self, 'graph_enc') else tab_out
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(sequences)
        seq_out = hidden[-1]
        
        # Check for NaN
        if torch.isnan(tab_out).any() or torch.isnan(graph_out).any() or torch.isnan(seq_out).any():
            print("⚠️ NaN detected in forward pass!")
            # Return zeros to prevent crash
            return torch.zeros(features.size(0), 2, device=device), torch.zeros(features.size(0), 3, device=device)
        
        # Fusion
        combined = torch.cat([tab_out, graph_out, seq_out], dim=1)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        # Dummy attention for compatibility
        attention = torch.ones(features.size(0), 3, device=device) / 3.0
        
        return logits, attention


def collate_fn(batch):
    return {
        'features': torch.stack([b['features'] for b in batch]),
        'sequences': torch.stack([b['sequence'] for b in batch]),
        'seq_lens': torch.tensor([b['seq_len'] for b in batch]),
        'node_indices': torch.tensor([b['node_idx'] for b in batch]),
        'labels': torch.stack([b['label'] for b in batch])
    }


def train_stable_multimodal(sample_size=5000, num_epochs=50, batch_size=64):
    """Train stable multimodal model"""
    
    print("🎯 STABLE MULTIMODAL FUSION TRAINING")
    print("=" * 60)
    print(f"Sample: {sample_size:,}, Epochs: {num_epochs}, Batch: {batch_size}")
    
    # Load data
    parquet_dir = Path("data/parquet_export")
    donors_df = pd.read_parquet(parquet_dir / 'donors_with_features.parquet')
    relationships_df = pd.read_parquet(parquet_dir / 'relationships.parquet')
    giving_df = pd.read_parquet(parquet_dir / 'giving_history.parquet')
    
    print(f"\n📂 Loaded {len(donors_df):,} donors")
    
    # Sample
    if sample_size and sample_size < len(donors_df):
        donors_df = donors_df.sample(n=sample_size, random_state=42)
    
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
    
    print(f"✂️ Split: Train={len(train_ids):,}, Val={len(val_ids):,}, Test={len(test_ids):,}")
    
    # Create datasets
    print("\n🔧 Creating datasets...")
    train_ds = StableDataset(train_ids, donors_df, relationships_df, giving_df, y_train, fit_scaler=True)
    val_ds = StableDataset(val_ids, donors_df, relationships_df, giving_df, y_val, scaler=train_ds.scaler)
    test_ds = StableDataset(test_ids, donors_df, relationships_df, giving_df, y_test, scaler=train_ds.scaler)
    
    print(f"   ✅ Features: {train_ds.features.shape[1]} dimensions")
    if train_ds.graph_data:
        print(f"   ✅ Graph: {train_ds.graph_data.num_nodes} nodes, {train_ds.graph_data.num_edges} edges")
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🏋️ Training on: {device}")
    
    model = StableMultimodalModel(
        input_dim=train_ds.features.shape[1],
        hidden_dim=64,
        num_classes=2,
        dropout=0.3,
        use_gnn=(train_ds.graph_data is not None)
    ).to(device)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Loss
    train_labels = [lbl.item() for lbl in train_ds.labels]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training
    print(f"\n🚀 Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        nan_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            features = batch['features'].to(device)
            sequences = batch['sequences'].to(device)
            seq_lens = batch['seq_lens'].to(device)
            node_indices = batch['node_indices'].to(device)
            labels = batch['labels'].to(device)
            
            # Check for NaN in inputs
            if torch.isnan(features).any() or torch.isnan(sequences).any():
                nan_batches += 1
                continue
            
            logits, _ = model(features, train_ds.graph_data, sequences, seq_lens, node_indices)
            
            # Check for NaN in outputs
            if torch.isnan(logits).any():
                nan_batches += 1
                continue
            
            loss = criterion(logits, labels)
            
            if torch.isnan(loss):
                nan_batches += 1
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        if nan_batches > 0:
            print(f"   ⚠️ Skipped {nan_batches} batches due to NaN")
        
        train_loss /= max(len(train_loader) - nan_batches, 1)
        
        # Validate
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                sequences = batch['sequences'].to(device)
                seq_lens = batch['seq_lens'].to(device)
                node_indices = batch['node_indices'].to(device)
                labels = batch['labels'].to(device)
                
                logits, _ = model(features, val_ds.graph_data, sequences, seq_lens, node_indices)
                
                if not torch.isnan(logits).any():
                    loss = criterion(logits, labels)
                    if not torch.isnan(loss):
                        val_loss += loss.item()
                        preds = torch.argmax(logits, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds) if len(all_preds) > 0 else 0
        val_f1 = f1_score(all_labels, all_preds, average='weighted') if len(all_preds) > 0 else 0
        
        scheduler.step(val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_stable_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
    
    # Test
    if Path('models/best_stable_model.pt').exists():
        model.load_state_dict(torch.load('models/best_stable_model.pt'))
    
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            sequences = batch['sequences'].to(device)
            seq_lens = batch['seq_lens'].to(device)
            node_indices = batch['node_indices'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(features, test_ds.graph_data, sequences, seq_lens, node_indices)
            
            if not torch.isnan(logits).any():
                preds = torch.argmax(logits, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    
    print(f"\n📊 FINAL TEST RESULTS:")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   F1 Score: {test_f1:.4f}")
    
    return {'accuracy': test_acc, 'f1': test_f1, 'model': model}


if __name__ == "__main__":
    results = train_stable_multimodal(sample_size=5000, num_epochs=50, batch_size=64)
    print(f"\n🎉 Training completed successfully!")
    print(f"   Final Accuracy: {results['accuracy']:.4f}")
    print(f"   Final F1: {results['f1']:.4f}")

