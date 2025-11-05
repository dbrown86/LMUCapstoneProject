#!/usr/bin/env python3
"""
Improved Multimodal Fusion Architecture for Donor Prediction
Enhanced version with better performance and stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class ImprovedCrossModalAttention(nn.Module):
    """Improved cross-modal attention with better stability"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(ImprovedCrossModalAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        
        # Apply layer normalization
        x_norm = self.layer_norm(x)
        
        # Project to Q, K, V
        q = self.q_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        # Output projection and residual connection
        output = self.out_proj(attn_output)
        return output + x  # Residual connection

class ModalityImportanceGate(nn.Module):
    """Learnable modality importance weighting"""
    
    def __init__(self, dim):
        super(ModalityImportanceGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        importance = self.gate(x)
        return x * importance

class ImprovedTabularEncoder(nn.Module):
    """Improved tabular encoder with residual connections"""
    
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super(ImprovedTabularEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder layers with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Skip connection
        self.skip_connection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
    def forward(self, x):
        # Main path
        encoded = self.encoder(x)
        
        # Skip connection
        skip = self.skip_connection(x)
        
        return encoded + skip  # Residual connection

class ImprovedTextEncoder(nn.Module):
    """Improved text encoder with attention pooling"""
    
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.1):
        super(ImprovedTextEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Projection layer
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softmax(dim=1)
        )
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Project to hidden dimension
        projected = self.projection(x)
        
        # For BERT embeddings, we treat each sample as a single sequence
        # x shape: (batch_size, 768) -> projected shape: (batch_size, hidden_dim)
        # No attention pooling needed for single vector per sample
        pooled = projected
        
        # Encode
        encoded = self.encoder(pooled)
        return encoded

class ImprovedGraphEncoder(nn.Module):
    """Improved graph encoder with residual connections"""
    
    def __init__(self, input_dim=64, hidden_dim=256, dropout=0.1):
        super(ImprovedGraphEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Skip connection
        self.skip_connection = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        # Main path
        encoded = self.encoder(x)
        
        # Skip connection
        skip = self.skip_connection(x)
        
        return encoded + skip  # Residual connection

class ImprovedMultimodalFusionModel(nn.Module):
    """Improved multimodal fusion model with better architecture"""
    
    def __init__(self, tabular_dim, text_dim, graph_dim, 
                 hidden_dim=256, fusion_dim=512, num_classes=2, 
                 dropout=0.1, num_attention_heads=8):
        super(ImprovedMultimodalFusionModel, self).__init__()
        
        self.tabular_dim = tabular_dim
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        
        # Modality-specific encoders
        self.tabular_encoder = ImprovedTabularEncoder(tabular_dim, hidden_dim, dropout)
        self.text_encoder = ImprovedTextEncoder(text_dim, hidden_dim, dropout)
        self.graph_encoder = ImprovedGraphEncoder(graph_dim, hidden_dim, dropout)
        
        # Modality importance gates
        self.tabular_gate = ModalityImportanceGate(hidden_dim)
        self.text_gate = ModalityImportanceGate(hidden_dim)
        self.graph_gate = ModalityImportanceGate(hidden_dim)
        
        # Cross-modal attention
        self.cross_modal_attention = ImprovedCrossModalAttention(
            hidden_dim, num_attention_heads, dropout
        )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim * 3, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.BatchNorm1d(fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 4, num_classes),
            nn.Softmax(dim=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, tabular, text, graph, modality_mask):
        batch_size = tabular.size(0)
        
        # Encode each modality
        tabular_encoded = self.tabular_encoder(tabular)
        text_encoded = self.text_encoder(text)
        graph_encoded = self.graph_encoder(graph)
        
        # Apply modality importance gates
        tabular_weighted = self.tabular_gate(tabular_encoded) * modality_mask[:, 0:1]
        text_weighted = self.text_gate(text_encoded) * modality_mask[:, 1:2]
        graph_weighted = self.graph_gate(graph_encoded) * modality_mask[:, 2:3]
        
        # Stack for cross-modal attention
        modalities = torch.stack([tabular_weighted, text_weighted, graph_weighted], dim=1)
        
        # Apply cross-modal attention
        attended_modalities = self.cross_modal_attention(modalities)
        
        # Flatten for fusion
        fused_features = attended_modalities.view(batch_size, -1)
        
        # Fusion layers
        fused = self.fusion_layers(fused_features)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits, fused_features

class ImprovedMultimodalDataset(Dataset):
    """Improved dataset with better data handling"""
    
    def __init__(self, tabular_features, text_embeddings, graph_embeddings, labels, 
                 modality_mask=None):
        self.tabular = torch.FloatTensor(tabular_features) if tabular_features is not None else None
        self.text = torch.FloatTensor(text_embeddings) if text_embeddings is not None else None
        self.graph = torch.FloatTensor(graph_embeddings) if graph_embeddings is not None else None
        self.labels = torch.LongTensor(labels)
        
        # Create modality availability mask
        if modality_mask is None:
            self.modality_mask = torch.ones(len(labels), 3)
            if tabular_features is None:
                self.modality_mask[:, 0] = 0
            if text_embeddings is None:
                self.modality_mask[:, 1] = 0
            if graph_embeddings is None:
                self.modality_mask[:, 2] = 0
        else:
            self.modality_mask = torch.FloatTensor(modality_mask)
        
        self.length = len(labels)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Get the actual dimensions for each modality
        tabular_dim = self.tabular.shape[1] if self.tabular is not None else 15
        text_dim = self.text.shape[1] if self.text is not None else 768
        graph_dim = self.graph.shape[1] if self.graph is not None else 64
        
        item = {
            'tabular': self.tabular[idx] if self.tabular is not None else torch.zeros(tabular_dim),
            'text': self.text[idx] if self.text is not None else torch.zeros(text_dim),
            'graph': self.graph[idx] if self.graph is not None else torch.zeros(graph_dim),
            'modality_mask': self.modality_mask[idx],
            'label': self.labels[idx]
        }
        return item

def create_improved_multimodal_model(tabular_dim, text_dim, graph_dim, 
                                   hidden_dim=256, fusion_dim=512, 
                                   num_classes=2, dropout=0.1, 
                                   num_attention_heads=8, device='cuda'):
    """Create improved multimodal model with better parameters"""
    
    model = ImprovedMultimodalFusionModel(
        tabular_dim=tabular_dim,
        text_dim=text_dim,
        graph_dim=graph_dim,
        hidden_dim=hidden_dim,
        fusion_dim=fusion_dim,
        num_classes=num_classes,
        dropout=dropout,
        num_attention_heads=num_attention_heads
    ).to(device)
    
    return model

def get_improved_optimizer(model, lr=0.0001, weight_decay=1e-5):
    """Get improved optimizer with better parameters"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

def get_improved_scheduler(optimizer, epochs=100, warmup_epochs=10):
    """Get improved learning rate scheduler"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print("Improved multimodal architecture loaded!")
