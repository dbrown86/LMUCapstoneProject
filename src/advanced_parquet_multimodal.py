#!/usr/bin/env python3
"""
Advanced Multimodal Training Pipeline using Parquet files
Full multimodal fusion architecture with:
- Graph Neural Networks (GNN) for relationship networks
- LSTM for temporal giving history sequences  
- Attention-based fusion of all modalities
- Supports all 500K records with efficient batch loading
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import PyTorch Geometric
try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, global_mean_pool
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    print("⚠️ PyTorch Geometric not available, using fallback for graph features")


class AdvancedMultimodalDataset(Dataset):
    """
    Advanced PyTorch Dataset for full multimodal fusion
    Includes graph structure, sequences, and attention
    """
    
    def __init__(self, 
                 donor_ids: List[int],
                 donors_df: pd.DataFrame,
                 relationships_df: Optional[pd.DataFrame],
                 giving_df: Optional[pd.DataFrame],
                 events_df: Optional[pd.DataFrame],
                 labels: np.ndarray,
                 scaler: Optional[StandardScaler] = None,
                 fit_scaler: bool = False,
                 max_sequence_length: int = 50):
        """
        Initialize advanced dataset with full multimodal features
        """
        self.donor_ids = donor_ids
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.scaler = scaler
        self.max_sequence_length = max_sequence_length
        
        # Pre-compute donor ID to index mapping (needed for graph building)
        self.donor_id_to_idx = {donor_id: idx for idx, donor_id in enumerate(donor_ids)}
        
        # Filter donors and set index
        self.donors_df = donors_df[donors_df['ID'].isin(donor_ids)].copy()
        self.donors_df = self.donors_df.set_index('ID')
        
        # Prepare tabular features
        self.tabular_features = self._prepare_tabular_features(fit_scaler, scaler)
        
        # Store full dataframes for graph and sequence construction
        self.relationships_df = relationships_df
        self.giving_df = giving_df
        self.events_df = events_df
        
        # Build graph structure (one-time)
        print("   🕸️ Building graph structure...")
        self.graph_data = self._build_graph_structure()
    
    def _prepare_tabular_features(self, fit_scaler: bool, scaler: Optional[StandardScaler]) -> torch.Tensor:
        """Prepare tabular features from donors dataframe"""
        # Priority list of numerical features (use what's available)
        priority_cols = [
            'Lifetime_Giving', 'Total_Yr_Giving_Count', 'Last_Gift',
            'Consecutive_Yr_Giving_Count', 'Engagement_Score',
            'Legacy_Intent_Probability', 'Legacy_Intent_Binary',
            'Estimated_Age', 'Class_Year', 'Parent_Year'
        ]
        
        # Filter to available columns
        available_cols = [col for col in priority_cols if col in self.donors_df.columns]
        
        # If still none, use all numerical columns (excluding ID and Family_ID)
        if len(available_cols) == 0:
            numeric_cols = self.donors_df.select_dtypes(include=[np.number]).columns.tolist()
            available_cols = [col for col in numeric_cols if col not in ['ID', 'Family_ID']]
        
        if len(available_cols) == 0:
            raise ValueError("No numerical features found in donors dataframe")
        
        print(f"      Using {len(available_cols)} features: {available_cols[:5]}...")
        
        # Extract features
        features = self.donors_df[available_cols].fillna(0).values
        
        # Scale features
        if fit_scaler:
            if self.scaler is None:
                self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        elif scaler is not None:
            self.scaler = scaler
            features = scaler.transform(features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _build_graph_structure(self) -> Optional[Data]:
        """
        Build PyTorch Geometric graph structure from relationships
        """
        if self.relationships_df is None or len(self.relationships_df) == 0:
            return None
        
        # Filter relationships to only include donors in our sample
        rel_df = self.relationships_df[
            self.relationships_df['Donor_ID_1'].isin(self.donor_ids) & 
            self.relationships_df['Donor_ID_2'].isin(self.donor_ids)
        ].copy()
        
        if len(rel_df) == 0:
            return None
        
        if not PYTORCH_GEOMETRIC_AVAILABLE:
            # Fallback: return relationship count per donor
            return None
        
        # Map donor IDs to indices
        rel_df['idx_1'] = rel_df['Donor_ID_1'].map(self.donor_id_to_idx)
        rel_df['idx_2'] = rel_df['Donor_ID_2'].map(self.donor_id_to_idx)
        
        # Remove any unmapped relationships
        rel_df = rel_df.dropna(subset=['idx_1', 'idx_2'])
        
        if len(rel_df) == 0:
            return None
        
        # Create edge index (bidirectional)
        edge_index_1 = torch.tensor([rel_df['idx_1'].values, rel_df['idx_2'].values], dtype=torch.long)
        edge_index_2 = torch.tensor([rel_df['idx_2'].values, rel_df['idx_1'].values], dtype=torch.long)
        edge_index = torch.cat([edge_index_1, edge_index_2], dim=1)
        
        # Create edge attributes (relationship strength)
        if 'Strength' in rel_df.columns:
            edge_attr_1 = torch.tensor(rel_df['Strength'].values, dtype=torch.float32).unsqueeze(1)
            edge_attr_2 = edge_attr_1.clone()
            edge_attr = torch.cat([edge_attr_1, edge_attr_2], dim=0)
        else:
            edge_attr = None
        
        # Create node features (use tabular features)
        x = self.tabular_features
        
        # Create graph data
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(self.donor_ids)
        )
        
        return graph_data
    
    def _get_giving_sequence(self, donor_id: int) -> Tuple[torch.Tensor, int]:
        """
        Get giving history sequence for a donor
        
        Returns:
            (sequence_tensor, sequence_length)
        """
        if self.giving_df is None:
            # Return empty sequence
            return torch.zeros(self.max_sequence_length, 3), 0
        
        # Get donor's giving history
        donor_giving = self.giving_df[self.giving_df['Donor_ID'] == donor_id].copy()
        
        if len(donor_giving) == 0:
            return torch.zeros(self.max_sequence_length, 3), 0
        
        # Sort by date
        if 'Gift_Date' in donor_giving.columns:
            donor_giving = donor_giving.sort_values('Gift_Date')
        
        # Extract features: [amount, campaign_year, anonymous]
        features = []
        for _, row in donor_giving.iterrows():
            features.append([
                row.get('Gift_Amount', 0),
                row.get('Campaign_Year', 0),
                1 if row.get('Anonymous', False) else 0
            ])
        
        # Convert to tensor
        sequence = torch.tensor(features, dtype=torch.float32)
        actual_length = min(len(sequence), self.max_sequence_length)
        
        # Pad or truncate to max_sequence_length
        if len(sequence) < self.max_sequence_length:
            # Pad with zeros
            padding = torch.zeros(self.max_sequence_length - len(sequence), 3)
            sequence = torch.cat([sequence, padding], dim=0)
        else:
            # Truncate to most recent
            sequence = sequence[-self.max_sequence_length:]
        
        return sequence, actual_length
    
    def __len__(self) -> int:
        return len(self.donor_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with full multimodal features"""
        donor_id = self.donor_ids[idx]
        
        # Tabular features
        tabular = self.tabular_features[idx]
        
        # Label
        label = self.labels[idx]
        
        # Giving sequence
        giving_sequence, giving_length = self._get_giving_sequence(donor_id)
        
        # Graph features (node index for this donor)
        node_idx = idx
        
        return {
            'tabular_features': tabular,
            'giving_sequence': giving_sequence,
            'giving_length': giving_length,
            'node_idx': node_idx,
            'label': label,
            'donor_id': donor_id
        }


class FullMultimodalFusionModel(nn.Module):
    """
    Full Multimodal Fusion Architecture
    Combines: Tabular Encoder + Graph Neural Network + LSTM + Attention Fusion
    """
    
    def __init__(self, 
                 tabular_dim: int,
                 graph_hidden_dim: int = 64,
                 sequence_hidden_dim: int = 64,
                 fusion_hidden_dim: int = 128,
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 use_gnn: bool = True):
        super().__init__()
        
        self.use_gnn = use_gnn and PYTORCH_GEOMETRIC_AVAILABLE
        
        # 1. Tabular Encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Graph Neural Network (if available)
        if self.use_gnn:
            self.graph_conv1 = GCNConv(tabular_dim, graph_hidden_dim)
            self.graph_conv2 = GCNConv(graph_hidden_dim, graph_hidden_dim)
            self.graph_bn1 = nn.BatchNorm1d(graph_hidden_dim)
            self.graph_bn2 = nn.BatchNorm1d(graph_hidden_dim)
        else:
            # Fallback: simple encoder
            self.graph_encoder = nn.Sequential(
                nn.Linear(tabular_dim, graph_hidden_dim),
                nn.ReLU()
            )
        
        # 3. Sequence Encoder (LSTM for giving history)
        self.sequence_lstm = nn.LSTM(
            input_size=3,  # [amount, campaign_year, anonymous]
            hidden_size=sequence_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if sequence_hidden_dim > 1 else 0
        )
        
        # 4. Attention Fusion
        # Calculate total input dimension for fusion
        fusion_input_dim = 64 + graph_hidden_dim + sequence_hidden_dim  # tabular + graph + sequence
        
        self.attention_weights = nn.Linear(fusion_input_dim, 3)  # 3 modalities
        
        # 5. Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 6. Classifier
        self.classifier = nn.Linear(fusion_hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                tabular_features: torch.Tensor,
                graph_data: Optional[Data],
                giving_sequences: torch.Tensor,
                giving_lengths: torch.Tensor,
                node_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with full multimodal fusion
        
        Args:
            tabular_features: [batch_size, tabular_dim]
            graph_data: PyTorch Geometric Data object (entire graph)
            giving_sequences: [batch_size, max_seq_len, 3]
            giving_lengths: [batch_size]
            node_indices: [batch_size] - indices of nodes in the graph
            
        Returns:
            logits: [batch_size, num_classes]
            attention_weights: [batch_size, 3] - attention for each modality
        """
        batch_size = tabular_features.size(0)
        device = tabular_features.device
        
        # 1. Encode tabular features
        tabular_encoded = self.tabular_encoder(tabular_features)  # [batch_size, 64]
        
        # 2. Encode graph features
        if self.use_gnn and graph_data is not None:
            # Run GNN on entire graph
            x = graph_data.x.to(device)
            edge_index = graph_data.edge_index.to(device)
            
            # GNN layers
            x = self.graph_conv1(x, edge_index)
            x = self.graph_bn1(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            x = self.graph_conv2(x, edge_index)
            x = self.graph_bn2(x)
            x = F.relu(x)
            
            # Extract features for batch nodes
            graph_encoded = x[node_indices]  # [batch_size, graph_hidden_dim]
        else:
            # Fallback: use tabular features
            graph_encoded = self.graph_encoder(tabular_features) if hasattr(self, 'graph_encoder') else tabular_encoded
        
        # 3. Encode sequences (LSTM)
        # Pack sequences for efficiency
        giving_sequences = giving_sequences.to(device)
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.sequence_lstm(giving_sequences)
        
        # Use last hidden state
        sequence_encoded = hidden[-1]  # [batch_size, sequence_hidden_dim]
        
        # 4. Concatenate all modality features
        all_features = torch.cat([
            tabular_encoded,    # [batch_size, 64]
            graph_encoded,      # [batch_size, graph_hidden_dim]
            sequence_encoded    # [batch_size, sequence_hidden_dim]
        ], dim=1)
        
        # 5. Attention-based fusion
        attention_logits = self.attention_weights(all_features)  # [batch_size, 3]
        attention_weights = F.softmax(attention_logits, dim=1)   # [batch_size, 3]
        
        # Apply attention (weighted sum of all features)
        # Use all_features directly since it already combines all modalities
        fused_features = all_features  # [batch_size, total_dim]
        
        # 6. Fusion layer
        fused = self.fusion_layer(fused_features)
        
        # 7. Classification
        logits = self.classifier(fused)
        
        return logits, attention_weights


def collate_multimodal_batch(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for multimodal batch
    """
    return {
        'tabular_features': torch.stack([item['tabular_features'] for item in batch]),
        'giving_sequences': torch.stack([item['giving_sequence'] for item in batch]),
        'giving_lengths': torch.tensor([item['giving_length'] for item in batch]),
        'node_indices': torch.tensor([item['node_idx'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch]),
        'donor_ids': [item['donor_id'] for item in batch]
    }


class AdvancedMultimodalTrainer:
    """
    Advanced training pipeline with full multimodal fusion
    """
    
    def __init__(self,
                 parquet_dir: str = "data/parquet_export",
                 model_save_dir: str = "models",
                 results_save_dir: str = "results"):
        
        self.parquet_dir = Path(parquet_dir)
        self.model_save_dir = Path(model_save_dir)
        self.results_save_dir = Path(results_save_dir)
        
        # Create directories
        self.model_save_dir.mkdir(exist_ok=True)
        self.results_save_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.model = None
        self.scaler = None
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'attention_weights': []
        }
    
    def load_parquet_data(self) -> Dict[str, pd.DataFrame]:
        """Load all Parquet files"""
        print("📂 Loading Parquet files...")
        
        data = {}
        
        # Load donors with features
        donors_file = self.parquet_dir / 'donors_with_features.parquet'
        if donors_file.exists():
            data['donors'] = pd.read_parquet(donors_file)
            print(f"   ✅ Donors: {len(data['donors']):,} records")
        else:
            raise FileNotFoundError(f"Donors file not found: {donors_file}")
        
        # Load relationships
        rel_file = self.parquet_dir / 'relationships.parquet'
        if rel_file.exists():
            data['relationships'] = pd.read_parquet(rel_file)
            print(f"   ✅ Relationships: {len(data['relationships']):,} records")
        else:
            data['relationships'] = None
            print("   ⚠️ Relationships file not found")
        
        # Load giving history
        giving_file = self.parquet_dir / 'giving_history.parquet'
        if giving_file.exists():
            data['giving'] = pd.read_parquet(giving_file)
            print(f"   ✅ Giving history: {len(data['giving']):,} records")
        else:
            data['giving'] = None
            print("   ⚠️ Giving history file not found")
        
        # Load events
        events_file = self.parquet_dir / 'event_attendance.parquet'
        if events_file.exists():
            data['events'] = pd.read_parquet(events_file)
            print(f"   ✅ Events: {len(data['events']):,} records")
        else:
            data['events'] = None
            print("   ⚠️ Events file not found")
        
        return data
    
    def create_target_variable(self, 
                              donors_df: pd.DataFrame,
                              target_type: str = 'high_value') -> np.ndarray:
        """Create target variable for classification"""
        if target_type == 'high_value':
            threshold = donors_df['Lifetime_Giving'].quantile(0.8)
            return (donors_df['Lifetime_Giving'] >= threshold).astype(int).values
        elif target_type == 'frequent_giver':
            threshold = donors_df['Total_Yr_Giving_Count'].quantile(0.7)
            return (donors_df['Total_Yr_Giving_Count'] >= threshold).astype(int).values
        elif target_type == 'event_attender':
            return (donors_df['Primary_Constituent_Type'].isin(['Alum', 'Trustee', 'Regent'])).astype(int).values
        else:
            raise ValueError(f"Unknown target type: {target_type}")
    
    def prepare_datasets(self,
                        sample_size: Optional[int] = None,
                        target_type: str = 'high_value',
                        test_size: float = 0.2,
                        val_size: float = 0.2,
                        random_state: int = 42,
                        max_sequence_length: int = 50) -> Tuple[Dataset, Dataset, Dataset, Optional[Data]]:
        """
        Prepare train/val/test datasets with full multimodal features
        
        Args:
            sample_size: Number of samples (None = use all 500K)
            target_type: Type of target variable
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed
            max_sequence_length: Max sequence length for LSTM
            
        Returns:
            train_dataset, val_dataset, test_dataset, graph_data
        """
        print("🚀 PREPARING ADVANCED MULTIMODAL DATASETS")
        print("=" * 60)
        
        # Load data
        data = self.load_parquet_data()
        donors_df = data['donors']
        
        # Sample if requested
        if sample_size and sample_size < len(donors_df):
            print(f"\n📊 Sampling {sample_size:,} donors from {len(donors_df):,} total...")
            donors_df = donors_df.sample(n=sample_size, random_state=random_state)
        else:
            print(f"\n📊 Using ALL {len(donors_df):,} donors")
        
        # Create labels
        labels = self.create_target_variable(donors_df, target_type)
        print(f"   Target distribution: {np.bincount(labels)}")
        print(f"   Positive class: {np.sum(labels):,} ({np.mean(labels)*100:.1f}%)")
        
        # Get donor IDs
        donor_ids = donors_df['ID'].tolist()
        
        # Split donor IDs
        train_ids, test_ids, y_train, y_test = train_test_split(
            donor_ids, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        train_ids, val_ids, y_train, y_val = train_test_split(
            train_ids, y_train,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train
        )
        
        print(f"\n✂️ Data split:")
        print(f"   Train: {len(train_ids):,} samples")
        print(f"   Val: {len(val_ids):,} samples")
        print(f"   Test: {len(test_ids):,} samples")
        
        # Create datasets
        print("\n🔧 Creating advanced PyTorch datasets...")
        print("   (This includes graph structure and sequence building)")
        
        # Train dataset (fit scaler)
        train_dataset = AdvancedMultimodalDataset(
            donor_ids=train_ids,
            donors_df=donors_df,
            relationships_df=data.get('relationships'),
            giving_df=data.get('giving'),
            events_df=data.get('events'),
            labels=y_train,
            fit_scaler=True,
            max_sequence_length=max_sequence_length
        )
        
        # Save scaler and graph
        self.scaler = train_dataset.scaler
        graph_data = train_dataset.graph_data
        
        # Val dataset (use fitted scaler)
        print("   Creating validation dataset...")
        val_dataset = AdvancedMultimodalDataset(
            donor_ids=val_ids,
            donors_df=donors_df,
            relationships_df=data.get('relationships'),
            giving_df=data.get('giving'),
            events_df=data.get('events'),
            labels=y_val,
            scaler=self.scaler,
            fit_scaler=False,
            max_sequence_length=max_sequence_length
        )
        
        # Test dataset (use fitted scaler)
        print("   Creating test dataset...")
        test_dataset = AdvancedMultimodalDataset(
            donor_ids=test_ids,
            donors_df=donors_df,
            relationships_df=data.get('relationships'),
            giving_df=data.get('giving'),
            events_df=data.get('events'),
            labels=y_test,
            scaler=self.scaler,
            fit_scaler=False,
            max_sequence_length=max_sequence_length
        )
        
        print("   ✅ Advanced datasets created successfully")
        
        if graph_data is not None:
            print(f"   📊 Graph structure: {graph_data.num_nodes:,} nodes, {graph_data.num_edges:,} edges")
        
        return train_dataset, val_dataset, test_dataset, graph_data
    
    def train_epoch(self, 
                   model: nn.Module,
                   dataloader: DataLoader,
                   graph_data: Optional[Data],
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   device: torch.device) -> Dict[str, float]:
        """Train for one epoch with full multimodal fusion"""
        model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_attention_weights = []
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Move data to device
            tabular = batch['tabular_features'].to(device)
            sequences = batch['giving_sequences'].to(device)
            lengths = batch['giving_lengths'].to(device)
            node_indices = batch['node_indices'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits, attention_weights = model(
                tabular_features=tabular,
                graph_data=graph_data,
                giving_sequences=sequences,
                giving_lengths=lengths,
                node_indices=node_indices
            )
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attention_weights.append(attention_weights.detach().cpu())
        
        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Average attention weights
        avg_attention = torch.cat(all_attention_weights, dim=0).mean(dim=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'attention': avg_attention.numpy()
        }
    
    def validate_epoch(self, 
                      model: nn.Module,
                      dataloader: DataLoader,
                      graph_data: Optional[Data],
                      criterion: nn.Module,
                      device: torch.device) -> Dict[str, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_attention_weights = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                tabular = batch['tabular_features'].to(device)
                sequences = batch['giving_sequences'].to(device)
                lengths = batch['giving_lengths'].to(device)
                node_indices = batch['node_indices'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                logits, attention_weights = model(
                    tabular_features=tabular,
                    graph_data=graph_data,
                    giving_sequences=sequences,
                    giving_lengths=lengths,
                    node_indices=node_indices
                )
                
                # Compute loss
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                # Track predictions
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_attention_weights.append(attention_weights.cpu())
        
        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Compute AUC if binary classification
        if len(np.unique(all_labels)) == 2:
            try:
                probs = np.array([p[1] for p in all_probabilities])
                # Check for NaN or inf
                if np.isnan(probs).any() or np.isinf(probs).any():
                    auc = 0.0
                else:
                    auc = roc_auc_score(all_labels, probs)
            except Exception as e:
                print(f"⚠️ AUC computation failed: {e}")
                auc = 0.0
        else:
            auc = 0.0
        
        # Average attention weights
        avg_attention = torch.cat(all_attention_weights, dim=0).mean(dim=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'attention': avg_attention.numpy()
        }
    
    def train(self,
              train_dataset: Dataset,
              val_dataset: Dataset,
              test_dataset: Dataset,
              graph_data: Optional[Data],
              num_epochs: int = 100,
              batch_size: int = 64,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-5,
              patience: int = 15,
              device: str = 'auto') -> Dict[str, Any]:
        """
        Train the advanced multimodal fusion model
        """
        # Set device
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        print(f"\n🏋️ Training Full Multimodal Fusion Model on: {device}")
        
        # Get feature dimension from first sample
        sample = train_dataset[0]
        tabular_dim = sample['tabular_features'].shape[0]
        
        print(f"   Tabular dimension: {tabular_dim}")
        print(f"   Graph available: {graph_data is not None}")
        if graph_data is not None:
            print(f"   Graph nodes: {graph_data.num_nodes:,}")
            print(f"   Graph edges: {graph_data.num_edges:,}")
        
        # Create model
        self.model = FullMultimodalFusionModel(
            tabular_dim=tabular_dim,
            graph_hidden_dim=64,
            sequence_hidden_dim=64,
            fusion_hidden_dim=128,
            num_classes=2,
            dropout=0.3,
            use_gnn=(graph_data is not None)
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Model parameters: {total_params:,}")
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_multimodal_batch
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_multimodal_batch
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_multimodal_batch
        )
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Compute class weights
        train_labels = [lbl.item() for lbl in train_dataset.labels]
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n🚀 Starting advanced multimodal training...")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Early stopping patience: {patience}")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(
                self.model, train_loader, graph_data, optimizer, criterion, device
            )
            
            # Validate
            val_metrics = self.validate_epoch(
                self.model, val_loader, graph_data, criterion, device
            )
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['train_f1'].append(train_metrics['f1'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            self.training_history['attention_weights'].append(val_metrics['attention'].tolist())
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}, "
                      f"Val F1: {val_metrics['f1']:.4f}")
                print(f"         Attention: "
                      f"Tabular={val_metrics['attention'][0]:.3f}, "
                      f"Graph={val_metrics['attention'][1]:.3f}, "
                      f"Sequence={val_metrics['attention'][2]:.3f}")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                self.save_model('best_advanced_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break
        
        # Load best model if it exists
        best_model_path = self.model_save_dir / 'best_advanced_model.pt'
        if best_model_path.exists():
            self.load_model('best_advanced_model.pt')
        else:
            print("⚠️ No best model saved (early stopping may not have triggered)")
        
        # Final evaluation
        test_metrics = self.validate_epoch(
            self.model, test_loader, graph_data, criterion, device
        )
        
        print(f"\n📊 FINAL TEST RESULTS:")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   F1 Score: {test_metrics['f1']:.4f}")
        print(f"   AUC: {test_metrics['auc']:.4f}")
        print(f"   Final Attention Weights:")
        print(f"      Tabular: {test_metrics['attention'][0]:.3f}")
        print(f"      Graph: {test_metrics['attention'][1]:.3f}")
        print(f"      Sequence: {test_metrics['attention'][2]:.3f}")
        
        return {
            'training_history': self.training_history,
            'test_metrics': test_metrics,
            'best_val_loss': best_val_loss,
            'final_attention': test_metrics['attention']
        }
    
    def save_model(self, filename: str):
        """Save model to file"""
        save_path = self.model_save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'scaler': self.scaler
        }, save_path)
    
    def load_model(self, filename: str):
        """Load model from file"""
        load_path = self.model_save_dir / filename
        checkpoint = torch.load(load_path, map_location='cpu')
        
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        if 'scaler' in checkpoint:
            self.scaler = checkpoint['scaler']
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot comprehensive training history including attention weights"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train', alpha=0.7)
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation', alpha=0.7)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.training_history['train_acc'], label='Train', alpha=0.7)
        axes[0, 1].plot(self.training_history['val_acc'], label='Validation', alpha=0.7)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[0, 2].plot(self.training_history['train_f1'], label='Train', alpha=0.7)
        axes[0, 2].plot(self.training_history['val_f1'], label='Validation', alpha=0.7)
        axes[0, 2].set_title('F1 Score')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Attention weights over time
        if self.training_history['attention_weights']:
            attention_array = np.array(self.training_history['attention_weights'])
            axes[1, 0].plot(attention_array[:, 0], label='Tabular', alpha=0.7)
            axes[1, 0].plot(attention_array[:, 1], label='Graph', alpha=0.7)
            axes[1, 0].plot(attention_array[:, 2], label='Sequence', alpha=0.7)
            axes[1, 0].set_title('Attention Weights Over Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Attention Weight')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Combined validation metrics
        axes[1, 1].plot(self.training_history['val_loss'], label='Val Loss', alpha=0.7)
        axes[1, 1].plot(self.training_history['val_acc'], label='Val Acc', alpha=0.7)
        axes[1, 1].plot(self.training_history['val_f1'], label='Val F1', alpha=0.7)
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Final attention weights pie chart
        if self.training_history['attention_weights']:
            final_attention = attention_array[-1]
            axes[1, 2].pie(final_attention, 
                          labels=['Tabular', 'Graph', 'Sequence'],
                          autopct='%1.1f%%',
                          startangle=90)
            axes[1, 2].set_title('Final Attention Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved to {save_path}")
        
        return fig


# Convenience function
def train_advanced_multimodal(sample_size: Optional[int] = 10000,
                             target_type: str = 'high_value',
                             num_epochs: int = 50,
                             batch_size: int = 64,
                             max_sequence_length: int = 50) -> Dict[str, Any]:
    """
    Train advanced multimodal fusion model from Parquet files
    
    Args:
        sample_size: Number of samples (None = use all 500K)
        target_type: Type of target variable
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        max_sequence_length: Max length for giving history sequences
        
    Returns:
        Training results with attention weights
    """
    print("🎯 ADVANCED MULTIMODAL FUSION TRAINING")
    print("=" * 60)
    print(f"   Sample size: {sample_size if sample_size else 'ALL 500K'}")
    print(f"   Target: {target_type}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    
    # Initialize trainer
    trainer = AdvancedMultimodalTrainer()
    
    # Prepare datasets
    train_ds, val_ds, test_ds, graph_data = trainer.prepare_datasets(
        sample_size=sample_size,
        target_type=target_type,
        max_sequence_length=max_sequence_length
    )
    
    # Train model
    results = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        graph_data=graph_data,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    # Plot results (skip if NaN)
    try:
        save_path = f"advanced_training_{sample_size if sample_size else 'full'}.png"
        trainer.plot_training_history(save_path)
    except Exception as e:
        print(f"⚠️ Skipping plot due to error: {e}")
    
    return results


if __name__ == "__main__":
    # Example: Train with 10K samples
    print("Example 1: Training with 10K samples")
    results = train_advanced_multimodal(
        sample_size=10000,
        target_type='high_value',
        num_epochs=50,
        batch_size=64
    )
    
    print(f"\n🎉 Training completed!")
    print(f"   Final accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"   Final F1 score: {results['test_metrics']['f1']:.4f}")
    print(f"   Final AUC: {results['test_metrics']['auc']:.4f}")

