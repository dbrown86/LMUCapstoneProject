#!/usr/bin/env python3
"""
Multimodal Fusion Architecture with SQL Database Integration
Updated to use the 500K donor SQL database with efficient data loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our SQL data loader
from .sql_data_loader import SQLDataLoader, load_multimodal_data, build_donor_graph

# Try to import PyTorch Geometric
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. Graph components will be disabled.")


class TabularEncoder(nn.Module):
    """
    Encoder for tabular donor data
    Handles both categorical and numerical features
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.3,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        return self.encoder(x)


class GraphEncoder(nn.Module):
    """
    Graph neural network encoder for relationship data
    Uses multiple GNN layers with different architectures
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 gnn_type: str = 'gcn'):
        super().__init__()
        
        if not PYTORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for graph encoding")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type == 'gcn':
                layer = GCNConv(hidden_dim, hidden_dim)
            elif gnn_type == 'gat':
                layer = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            elif gnn_type == 'sage':
                layer = SAGEConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.gnn_layers.append(layer)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class SequenceEncoder(nn.Module):
    """
    Encoder for sequential data (giving history, event attendance)
    Uses LSTM or Transformer architecture
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 use_transformer: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_transformer = use_transformer
        
        if use_transformer:
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            # LSTM encoder
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
            self.hidden_dim *= 2  # Bidirectional
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        if self.use_transformer:
            # Input projection
            x = self.input_proj(x)
            
            # Transformer encoding
            x = self.encoder(x)
            
            # Global pooling (mean over sequence length)
            if lengths is not None:
                # Mask padding
                mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
                x = x * mask.unsqueeze(-1).float()
                x = x.sum(dim=1) / lengths.unsqueeze(1).float()
            else:
                x = x.mean(dim=1)
        else:
            # LSTM encoding
            if lengths is not None:
                # Pack sequence
                x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            
            x, (hidden, cell) = self.encoder(x)
            
            if lengths is not None:
                x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            
            # Use last hidden state
            x = hidden[-1]  # Last layer hidden state
        
        return self.dropout_layer(x)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of multiple modalities
    Learns to weight different modalities dynamically
    """
    
    def __init__(self, 
                 input_dims: Dict[str, int],
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Project all modalities to same dimension
        self.projections = nn.ModuleDict({
            modality: nn.Linear(dim, hidden_dim)
            for modality, dim in input_dims.items()
        })
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, modality_features: Dict[str, torch.Tensor]):
        # Project all modalities to same dimension
        projected_features = {}
        for modality, features in modality_features.items():
            if modality in self.projections:
                projected_features[modality] = self.projections[modality](features)
        
        # Stack features for attention
        modalities = list(projected_features.keys())
        stacked_features = torch.stack([projected_features[mod] for mod in modalities], dim=1)
        
        # Self-attention
        attended_features, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Global pooling
        fused_features = attended_features.mean(dim=1)
        
        # Output projection
        output = self.output_proj(fused_features)
        
        return output, attention_weights


class MultimodalDonorPredictor(nn.Module):
    """
    Complete multimodal architecture for donor prediction
    Integrates tabular, graph, and sequential data
    """
    
    def __init__(self, 
                 tabular_input_dim: int,
                 graph_input_dim: int,
                 sequence_input_dim: int,
                 hidden_dim: int = 128,
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 use_graph: bool = True,
                 use_sequence: bool = True,
                 gnn_type: str = 'gcn',
                 use_transformer: bool = False):
        super().__init__()
        
        self.use_graph = use_graph and PYTORCH_GEOMETRIC_AVAILABLE
        self.use_sequence = use_sequence
        
        # Tabular encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            hidden_dims=[hidden_dim, hidden_dim // 2],
            dropout=dropout
        )
        
        # Graph encoder (if available)
        if self.use_graph:
            self.graph_encoder = GraphEncoder(
                input_dim=graph_input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                gnn_type=gnn_type
            )
        
        # Sequence encoder (if enabled)
        if self.use_sequence:
            self.sequence_encoder = SequenceEncoder(
                input_dim=sequence_input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_transformer=use_transformer
            )
        
        # Fusion layer
        input_dims = {'tabular': self.tabular_encoder.output_dim}
        if self.use_graph:
            input_dims['graph'] = hidden_dim
        if self.use_sequence:
            input_dims['sequence'] = hidden_dim
        
        self.fusion = AttentionFusion(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, 
                tabular_features: torch.Tensor,
                graph_data: Optional[Data] = None,
                sequence_features: Optional[torch.Tensor] = None,
                sequence_lengths: Optional[torch.Tensor] = None):
        
        # Encode tabular features
        tabular_encoded = self.tabular_encoder(tabular_features)
        
        # Prepare modality features
        modality_features = {'tabular': tabular_encoded}
        
        # Encode graph features (if available)
        if self.use_graph and graph_data is not None:
            graph_encoded = self.graph_encoder(
                graph_data.x, 
                graph_data.edge_index, 
                graph_data.batch
            )
            # Global pooling for graph features
            if hasattr(graph_data, 'batch') and graph_data.batch is not None:
                graph_encoded = global_mean_pool(graph_encoded, graph_data.batch)
            else:
                graph_encoded = graph_encoded.mean(dim=0, keepdim=True).expand(tabular_encoded.size(0), -1)
            modality_features['graph'] = graph_encoded
        
        # Encode sequence features (if available)
        if self.use_sequence and sequence_features is not None:
            sequence_encoded = self.sequence_encoder(sequence_features, sequence_lengths)
            modality_features['sequence'] = sequence_encoded
        
        # Fuse modalities
        fused_features, attention_weights = self.fusion(modality_features)
        
        # Classify
        logits = self.classifier(fused_features)
        
        return logits, attention_weights


class MultimodalDataProcessor:
    """
    Data processor for multimodal donor data from SQL database
    Handles feature engineering and data preparation
    """
    
    def __init__(self, db_path: str = "data/synthetic_donor_dataset_500k_dense/donor_database.db"):
        self.db_path = db_path
        self.scaler = None
        self.feature_columns = None
        
    def prepare_tabular_features(self, 
                                donors_df: pd.DataFrame,
                                enhanced_df: pd.DataFrame,
                                fit_scaler: bool = True) -> torch.Tensor:
        """
        Prepare tabular features for the model
        
        Args:
            donors_df: Donor data
            enhanced_df: Enhanced features
            fit_scaler: Whether to fit the scaler
            
        Returns:
            Tensor of tabular features
        """
        # Merge donor and enhanced data
        merged_df = donors_df.merge(enhanced_df, left_on='ID', right_on='Donor_ID', how='left')
        
        # Select numeric features
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'ID']
        
        # Handle missing values
        features_df = merged_df[numeric_cols].fillna(0)
        
        # Store feature columns for later use
        if self.feature_columns is None:
            self.feature_columns = numeric_cols
        
        # Scale features
        if fit_scaler:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features_df)
        else:
            features = self.scaler.transform(features_df)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def prepare_sequence_features(self, 
                                 giving_df: pd.DataFrame,
                                 events_df: pd.DataFrame,
                                 max_sequence_length: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequence features from giving history and events
        
        Args:
            giving_df: Giving history data
            events_df: Event attendance data
            max_sequence_length: Maximum sequence length
            
        Returns:
            Tuple of (sequence_features, sequence_lengths)
        """
        # This is a simplified implementation
        # In practice, you'd want more sophisticated sequence preparation
        
        # For now, return dummy data
        batch_size = len(giving_df['Donor_ID'].unique()) if not giving_df.empty else 1
        sequence_lengths = torch.randint(1, max_sequence_length, (batch_size,))
        
        # Create dummy sequence features
        sequence_features = torch.randn(batch_size, max_sequence_length, 10)
        
        return sequence_features, sequence_lengths
    
    def load_and_prepare_data(self, 
                             donor_ids: Optional[List[int]] = None,
                             limit: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Load and prepare all data modalities
        
        Args:
            donor_ids: Specific donor IDs to load
            limit: Maximum number of donors to load
            
        Returns:
            Dictionary with prepared data tensors
        """
        with SQLDataLoader(self.db_path) as loader:
            # Load all data with limit applied directly
            data = loader.get_multimodal_data(donor_ids=donor_ids, limit=limit)
            
            # Prepare tabular features
            tabular_features = self.prepare_tabular_features(
                data['donors'], 
                data['enhanced_fields']
            )
            
            # Prepare graph data
            if PYTORCH_GEOMETRIC_AVAILABLE:
                graph_data = loader.build_graph_data(
                    donor_ids=data['donors']['ID'].tolist()
                )
            else:
                graph_data = None
            
            # Prepare sequence features
            sequence_features, sequence_lengths = self.prepare_sequence_features(
                data['giving_history'],
                data['event_attendance']
            )
            
            return {
                'tabular_features': tabular_features,
                'graph_data': graph_data,
                'sequence_features': sequence_features,
                'sequence_lengths': sequence_lengths,
                'donor_ids': data['donors']['ID'].tolist()
            }


def create_multimodal_model(db_path: str = "data/synthetic_donor_dataset_500k_dense/donor_database.db",
                           sample_size: int = 1000) -> Tuple[MultimodalDonorPredictor, MultimodalDataProcessor]:
    """
    Create and initialize a multimodal model with data from SQL database
    
    Args:
        db_path: Path to SQL database
        sample_size: Number of samples to use for determining input dimensions
        
    Returns:
        Tuple of (model, data_processor)
    """
    # Initialize data processor
    processor = MultimodalDataProcessor(db_path)
    
    # Load sample data to determine input dimensions
    sample_data = processor.load_and_prepare_data(limit=sample_size)
    
    # Determine input dimensions
    tabular_input_dim = sample_data['tabular_features'].shape[1]
    graph_input_dim = sample_data['graph_data'].x.shape[1] if sample_data['graph_data'] is not None else 10
    sequence_input_dim = sample_data['sequence_features'].shape[2]
    
    # Create model
    model = MultimodalDonorPredictor(
        tabular_input_dim=tabular_input_dim,
        graph_input_dim=graph_input_dim,
        sequence_input_dim=sequence_input_dim,
        use_graph=PYTORCH_GEOMETRIC_AVAILABLE,
        use_sequence=True
    )
    
    return model, processor


# Example usage
if __name__ == "__main__":
    # Create model
    model, processor = create_multimodal_model()
    
    print(f"Model created with:")
    print(f"  Tabular input dim: {model.tabular_encoder.input_dim}")
    print(f"  Graph input dim: {model.graph_encoder.input_dim if hasattr(model, 'graph_encoder') else 'N/A'}")
    print(f"  Sequence input dim: {model.sequence_encoder.input_dim if hasattr(model, 'sequence_encoder') else 'N/A'}")
    
    # Load some data
    data = processor.load_and_prepare_data(limit=100)
    
    print(f"Loaded data with {len(data['donor_ids'])} donors")
    print(f"Tabular features shape: {data['tabular_features'].shape}")
    if data['graph_data'] is not None:
        print(f"Graph data: {data['graph_data'].num_nodes} nodes, {data['graph_data'].num_edges} edges")
    print(f"Sequence features shape: {data['sequence_features'].shape}")
