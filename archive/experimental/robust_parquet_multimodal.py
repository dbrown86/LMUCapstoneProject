#!/usr/bin/env python3
"""
ROBUST Multimodal Training Pipeline - Addresses NaN Issues
Fixes:
1. Extreme value handling (log transform for skewed features)
2. Proper NaN handling
3. Gradient clipping
4. Batch normalization
5. Lower learning rate
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import PyTorch Geometric
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False


class RobustMultimodalDataset(Dataset):
    """
    Robust dataset that handles extreme values and NaNs properly
    """
    
    def __init__(self,
                 donor_ids: List[int],
                 donors_df: pd.DataFrame,
                 relationships_df: Optional[pd.DataFrame],
                 giving_df: Optional[pd.DataFrame],
                 events_df: Optional[pd.DataFrame],
                 labels: np.ndarray,
                 scaler: Optional[RobustScaler] = None,
                 fit_scaler: bool = False,
                 max_sequence_length: int = 30):
        
        self.donor_ids = donor_ids
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.scaler = scaler
        self.max_sequence_length = max_sequence_length
        
        # Pre-compute donor ID to index mapping
        self.donor_id_to_idx = {donor_id: idx for idx, donor_id in enumerate(donor_ids)}
        
        # Filter donors
        self.donors_df = donors_df[donors_df['ID'].isin(donor_ids)].copy()
        self.donors_df = self.donors_df.set_index('ID')
        
        # Prepare tabular features with robust handling
        self.tabular_features = self._prepare_robust_features(fit_scaler, scaler)
        
        # Store dataframes
        self.relationships_df = relationships_df
        self.giving_df = giving_df
        self.events_df = events_df
        
        # Build graph structure
        print("   🕸️ Building graph structure...")
        self.graph_data = self._build_graph_structure()
        
        # Pre-group giving history for fast lookup
        if self.giving_df is not None and len(self.giving_df) > 0:
            print("   📊 Indexing giving history...")
            self.giving_by_donor = {}
            for donor_id in donor_ids:
                donor_giving = self.giving_df[self.giving_df['Donor_ID'] == donor_id]
                if len(donor_giving) > 0:
                    self.giving_by_donor[donor_id] = donor_giving.copy()
        else:
            self.giving_by_donor = {}
    
    def _prepare_robust_features(self, fit_scaler: bool, scaler: Optional[RobustScaler]) -> torch.Tensor:
        """Prepare features with robust handling of extreme values"""
        
        # Define features with their transformations
        feature_config = {
            'Lifetime_Giving': 'log',  # Log transform for extreme values
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
        
        # Collect features
        feature_list = []
        feature_names = []
        
        for col, transform in feature_config.items():
            if col in self.donors_df.columns:
                data = self.donors_df[col].copy()
                
                # Handle NaN
                data = data.fillna(data.median() if not data.isna().all() else 0)
                
                # Apply transformation
                if transform == 'log':
                    # Log transform (add 1 to handle zeros)
                    data = np.log1p(data.clip(lower=0))
                
                feature_list.append(data.values.reshape(-1, 1))
                feature_names.append(col)
        
        if len(feature_list) == 0:
            raise ValueError("No features available")
        
        # Stack features
        features = np.hstack(feature_list)
        
        print(f"      Using {len(feature_names)} features: {feature_names[:5]}...")
        
        # Robust scaling (less sensitive to outliers than StandardScaler)
        if fit_scaler:
            if self.scaler is None:
                self.scaler = RobustScaler()
            features = self.scaler.fit_transform(features)
        elif scaler is not None:
            self.scaler = scaler
            features = scaler.transform(features)
        
        # Clip extreme values after scaling
        features = np.clip(features, -10, 10)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _build_graph_structure(self) -> Optional[Data]:
        """Build graph structure"""
        if self.relationships_df is None or len(self.relationships_df) == 0:
            return None
        
        # Filter relationships
        rel_df = self.relationships_df[
            self.relationships_df['Donor_ID_1'].isin(self.donor_ids) & 
            self.relationships_df['Donor_ID_2'].isin(self.donor_ids)
        ].copy()
        
        if len(rel_df) == 0 or not PYTORCH_GEOMETRIC_AVAILABLE:
            return None
        
        # Map to indices
        rel_df['idx_1'] = rel_df['Donor_ID_1'].map(self.donor_id_to_idx)
        rel_df['idx_2'] = rel_df['Donor_ID_2'].map(self.donor_id_to_idx)
        rel_df = rel_df.dropna(subset=['idx_1', 'idx_2'])
        
        if len(rel_df) == 0:
            return None
        
        # Create edge index (bidirectional)
        edge_index_1 = torch.tensor([rel_df['idx_1'].values, rel_df['idx_2'].values], dtype=torch.long)
        edge_index_2 = torch.tensor([rel_df['idx_2'].values, rel_df['idx_1'].values], dtype=torch.long)
        edge_index = torch.cat([edge_index_1, edge_index_2], dim=1)
        
        # Create graph data
        graph_data = Data(
            x=self.tabular_features,
            edge_index=edge_index,
            num_nodes=len(self.donor_ids)
        )
        
        return graph_data
    
    def _get_giving_sequence(self, donor_id: int) -> Tuple[torch.Tensor, int]:
        """Get giving sequence for a donor"""
        if donor_id not in self.giving_by_donor:
            return torch.zeros(self.max_sequence_length, 3), 0
        
        donor_giving = self.giving_by_donor[donor_id]
        
        if len(donor_giving) == 0:
            return torch.zeros(self.max_sequence_length, 3), 0
        
        # Sort by date if available
        if 'Gift_Date' in donor_giving.columns:
            donor_giving = donor_giving.sort_values('Gift_Date')
        
        # Extract features with robust handling
        features = []
        for _, row in donor_giving.iterrows():
            amount = row.get('Gift_Amount', 0)
            # Log transform amount
            amount = np.log1p(max(0, amount))
            
            features.append([
                amount,
                row.get('Campaign_Year', 2020) / 2020.0,  # Normalize year
                1 if row.get('Anonymous', False) else 0
            ])
        
        sequence = torch.tensor(features, dtype=torch.float32)
        actual_length = min(len(sequence), self.max_sequence_length)
        
        # Pad or truncate
        if len(sequence) < self.max_sequence_length:
            padding = torch.zeros(self.max_sequence_length - len(sequence), 3)
            sequence = torch.cat([sequence, padding], dim=0)
        else:
            sequence = sequence[-self.max_sequence_length:]
        
        return sequence, actual_length
    
    def __len__(self) -> int:
        return len(self.donor_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        donor_id = self.donor_ids[idx]
        
        # Tabular features
        tabular = self.tabular_features[idx]
        
        # Label
        label = self.labels[idx]
        
        # Giving sequence
        giving_sequence, giving_length = self._get_giving_sequence(donor_id)
        
        # Graph node index
        node_idx = idx
        
        return {
            'tabular_features': tabular,
            'giving_sequence': giving_sequence,
            'giving_length': giving_length,
            'node_idx': node_idx,
            'label': label,
            'donor_id': donor_id
        }


class RobustMultimodalModel(nn.Module):
    """
    Robust multimodal model with batch normalization and dropout
    """
    
    def __init__(self,
                 tabular_dim: int,
                 graph_hidden_dim: int = 64,
                 sequence_hidden_dim: int = 64,
                 fusion_hidden_dim: int = 128,
                 num_classes: int = 2,
                 dropout: float = 0.4,
                 use_gnn: bool = True):
        super().__init__()
        
        self.use_gnn = use_gnn and PYTORCH_GEOMETRIC_AVAILABLE
        
        # 1. Tabular Encoder with BatchNorm
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Graph Neural Network
        if self.use_gnn:
            self.graph_conv1 = GCNConv(tabular_dim, graph_hidden_dim)
            self.graph_conv2 = GCNConv(graph_hidden_dim, graph_hidden_dim)
            self.graph_bn1 = nn.BatchNorm1d(graph_hidden_dim)
            self.graph_bn2 = nn.BatchNorm1d(graph_hidden_dim)
        else:
            self.graph_encoder = nn.Sequential(
                nn.Linear(tabular_dim, graph_hidden_dim),
                nn.BatchNorm1d(graph_hidden_dim),
                nn.ReLU()
            )
        
        # 3. Sequence Encoder (LSTM)
        self.sequence_lstm = nn.LSTM(
            input_size=3,
            hidden_size=sequence_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if sequence_hidden_dim > 1 else 0
        )
        
        # 4. Attention Fusion
        fusion_input_dim = 64 + graph_hidden_dim + sequence_hidden_dim
        self.attention_weights = nn.Linear(fusion_input_dim, 3)
        
        # 5. Fusion Layer with BatchNorm
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.BatchNorm1d(fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 6. Classifier
        self.classifier = nn.Linear(fusion_hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with Xavier/Kaiming"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self,
                tabular_features: torch.Tensor,
                graph_data: Optional[Data],
                giving_sequences: torch.Tensor,
                giving_lengths: torch.Tensor,
                node_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = tabular_features.size(0)
        device = tabular_features.device
        
        # 1. Encode tabular
        tabular_encoded = self.tabular_encoder(tabular_features)
        
        # 2. Encode graph
        if self.use_gnn and graph_data is not None:
            x = graph_data.x.to(device)
            edge_index = graph_data.edge_index.to(device)
            
            x = self.graph_conv1(x, edge_index)
            x = self.graph_bn1(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            x = self.graph_conv2(x, edge_index)
            x = self.graph_bn2(x)
            x = F.relu(x)
            
            graph_encoded = x[node_indices]
        else:
            graph_encoded = self.graph_encoder(tabular_features) if hasattr(self, 'graph_encoder') else tabular_encoded
        
        # 3. Encode sequences
        lstm_out, (hidden, cell) = self.sequence_lstm(giving_sequences)
        sequence_encoded = hidden[-1]
        
        # 4. Concatenate all features
        all_features = torch.cat([
            tabular_encoded,
            graph_encoded,
            sequence_encoded
        ], dim=1)
        
        # 5. Attention
        attention_logits = self.attention_weights(all_features)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # 6. Fusion
        fused = self.fusion_layer(all_features)
        
        # 7. Classification
        logits = self.classifier(fused)
        
        return logits, attention_weights


def collate_fn(batch):
    """Custom collate function"""
    return {
        'tabular_features': torch.stack([item['tabular_features'] for item in batch]),
        'giving_sequences': torch.stack([item['giving_sequence'] for item in batch]),
        'giving_lengths': torch.tensor([item['giving_length'] for item in batch]),
        'node_indices': torch.tensor([item['node_idx'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch]),
        'donor_ids': [item['donor_id'] for item in batch]
    }


def train_robust_multimodal(sample_size: Optional[int] = 10000,
                            target_type: str = 'high_value',
                            num_epochs: int = 50,
                            batch_size: int = 64,
                            learning_rate: float = 0.0001,  # Lower LR
                            max_sequence_length: int = 30):
    """
    Train robust multimodal model
    """
    print("🎯 ROBUST MULTIMODAL FUSION TRAINING")
    print("=" * 60)
    print(f"   Sample size: {sample_size if sample_size else 'ALL 500K'}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    # Load data
    parquet_dir = Path("data/parquet_export")
    
    print("\n📂 Loading Parquet files...")
    donors_df = pd.read_parquet(parquet_dir / 'donors_with_features.parquet')
    relationships_df = pd.read_parquet(parquet_dir / 'relationships.parquet')
    giving_df = pd.read_parquet(parquet_dir / 'giving_history.parquet')
    events_df = pd.read_parquet(parquet_dir / 'event_attendance.parquet')
    
    print(f"   ✅ Loaded {len(donors_df):,} donors")
    
    # Sample if needed
    if sample_size and sample_size < len(donors_df):
        donors_df = donors_df.sample(n=sample_size, random_state=42)
        print(f"   📊 Sampled {sample_size:,} donors")
    
    # Create target
    if target_type == 'high_value':
        threshold = donors_df['Lifetime_Giving'].quantile(0.8)
        labels = (donors_df['Lifetime_Giving'] >= threshold).astype(int).values
    else:
        labels = (donors_df['Legacy_Intent_Binary'] == 1).astype(int).values
    
    print(f"   Target: {np.sum(labels):,} positive ({np.mean(labels)*100:.1f}%)")
    
    # Split data
    donor_ids = donors_df['ID'].tolist()
    train_ids, test_ids, y_train, y_test = train_test_split(
        donor_ids, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_ids, val_ids, y_train, y_val = train_test_split(
        train_ids, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\n✂️ Split: Train={len(train_ids):,}, Val={len(val_ids):,}, Test={len(test_ids):,}")
    
    # Create datasets
    print("\n🔧 Creating robust datasets...")
    train_ds = RobustMultimodalDataset(
        train_ids, donors_df, relationships_df, giving_df, events_df,
        y_train, fit_scaler=True, max_sequence_length=max_sequence_length
    )
    
    val_ds = RobustMultimodalDataset(
        val_ids, donors_df, relationships_df, giving_df, events_df,
        y_val, scaler=train_ds.scaler, max_sequence_length=max_sequence_length
    )
    
    test_ds = RobustMultimodalDataset(
        test_ids, donors_df, relationships_df, giving_df, events_df,
        y_test, scaler=train_ds.scaler, max_sequence_length=max_sequence_length
    )
    
    print("   ✅ Datasets created")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🏋️ Training on: {device}")
    
    sample = train_ds[0]
    model = RobustMultimodalModel(
        tabular_dim=sample['tabular_features'].shape[0],
        graph_hidden_dim=64,
        sequence_hidden_dim=64,
        fusion_hidden_dim=128,
        num_classes=2,
        dropout=0.4,
        use_gnn=(train_ds.graph_data is not None)
    ).to(device)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Loss with class weights
    train_labels = [lbl.item() for lbl in train_ds.labels]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    print(f"\n🚀 Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            tabular = batch['tabular_features'].to(device)
            sequences = batch['giving_sequences'].to(device)
            lengths = batch['giving_lengths'].to(device)
            node_indices = batch['node_indices'].to(device)
            labels = batch['labels'].to(device)
            
            logits, attention = model(tabular, train_ds.graph_data, sequences, lengths, node_indices)
            loss = criterion(logits, labels)
            
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
                tabular = batch['tabular_features'].to(device)
                sequences = batch['giving_sequences'].to(device)
                lengths = batch['giving_lengths'].to(device)
                node_indices = batch['node_indices'].to(device)
                labels = batch['labels'].to(device)
                
                logits, attention = model(tabular, val_ds.graph_data, sequences, lengths, node_indices)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Update scheduler
        scheduler.step(val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_robust_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
    
    # Load best model and test
    model.load_state_dict(torch.load('models/best_robust_model.pt'))
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            tabular = batch['tabular_features'].to(device)
            sequences = batch['giving_sequences'].to(device)
            lengths = batch['giving_lengths'].to(device)
            node_indices = batch['node_indices'].to(device)
            labels = batch['labels'].to(device)
            
            logits, attention = model(tabular, test_ds.graph_data, sequences, lengths, node_indices)
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    
    print(f"\n📊 FINAL TEST RESULTS:")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   F1 Score: {test_f1:.4f}")
    
    return {
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'model': model
    }


if __name__ == "__main__":
    results = train_robust_multimodal(
        sample_size=5000,
        num_epochs=50,
        batch_size=64
    )

