#!/usr/bin/env python3
"""
Advanced Multimodal Fusion Architecture
Improved version with better fusion mechanisms and training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadCrossModalAttention(nn.Module):
    """Advanced multi-head cross-modal attention"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadCrossModalAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Multi-head projections
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
        return output + x

class ModalitySpecificEncoder(nn.Module):
    """Modality-specific encoder with better architecture"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, dropout=0.1):
        super(ModalitySpecificEncoder, self).__init__()
        
        # Dimensionality reduction for high-dim inputs
        if input_dim > 200:
            self.dim_reduction = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.dim_reduction = nn.Identity()
        
        # Main encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim if input_dim > 200 else input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Skip connection
        self.skip_connection = nn.Linear(
            hidden_dim if input_dim > 200 else input_dim, 
            output_dim
        ) if (hidden_dim if input_dim > 200 else input_dim) != output_dim else nn.Identity()
        
    def forward(self, x):
        # Dimensionality reduction
        reduced = self.dim_reduction(x)
        
        # Main encoding
        encoded = self.encoder(reduced)
        
        # Skip connection
        skip = self.skip_connection(reduced)
        
        return encoded + skip

class TransformerFusion(nn.Module):
    """Transformer-based fusion mechanism"""
    
    def __init__(self, dim, num_heads=8, num_layers=2, dropout=0.1):
        super(TransformerFusion, self).__init__()
        
        self.layers = nn.ModuleList([
            MultiHeadCrossModalAttention(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, num_modalities, dim)
        for layer, norm in zip(self.layers, self.layer_norms):
            # Apply layer normalization before attention
            x_norm = norm(x)
            # Apply attention
            x = layer(x_norm)
            # Apply dropout
            x = self.dropout(x)
        
        return x

class ModalityImportanceGate(nn.Module):
    """Learnable modality importance weighting"""
    
    def __init__(self, dim, num_modalities=3):
        super(ModalityImportanceGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        self.num_modalities = num_modalities
        
    def forward(self, x):
        # x shape: (batch_size, num_modalities, dim)
        batch_size, num_modalities, dim = x.size()
        
        # Calculate importance for each modality
        importance = self.gate(x.view(-1, dim)).view(batch_size, num_modalities, 1)
        
        # Apply importance weighting
        weighted_x = x * importance
        
        return weighted_x, importance

class AdvancedMultimodalFusionModel(nn.Module):
    """Advanced multimodal fusion model with better architecture"""
    
    def __init__(self, tabular_dim, text_dim, graph_dim, 
                 hidden_dim=256, fusion_dim=512, num_classes=2, 
                 dropout=0.1, num_attention_heads=8):
        super(AdvancedMultimodalFusionModel, self).__init__()
        
        self.tabular_dim = tabular_dim
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        
        # Modality-specific encoders with dimensionality reduction
        self.tabular_encoder = ModalitySpecificEncoder(tabular_dim, hidden_dim, 128, dropout)
        self.text_encoder = ModalitySpecificEncoder(text_dim, hidden_dim, 128, dropout)
        self.graph_encoder = ModalitySpecificEncoder(graph_dim, hidden_dim, 128, dropout)
        
        # Modality importance gates
        self.importance_gate = ModalityImportanceGate(128, 3)
        
        # Transformer-based fusion
        self.fusion_transformer = TransformerFusion(128, num_attention_heads, 2, dropout)
        
        # Cross-modal attention
        self.cross_modal_attention = MultiHeadCrossModalAttention(128, num_attention_heads, dropout)
        
        # Fusion layers with residual connections
        self.fusion_layers = nn.Sequential(
            nn.Linear(128 * 3, fusion_dim),
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
        
        # Encode each modality with dimensionality reduction
        tabular_encoded = self.tabular_encoder(tabular)
        text_encoded = self.text_encoder(text)
        graph_encoded = self.graph_encoder(graph)
        
        # Stack modalities for fusion
        modalities = torch.stack([tabular_encoded, text_encoded, graph_encoded], dim=1)
        
        # Apply modality importance gates
        weighted_modalities, importance = self.importance_gate(modalities)
        
        # Apply transformer fusion
        fused_modalities = self.fusion_transformer(weighted_modalities)
        
        # Apply cross-modal attention
        attended_modalities = self.cross_modal_attention(fused_modalities)
        
        # Flatten for final fusion
        fused_features = attended_modalities.view(batch_size, -1)
        
        # Fusion layers
        fused = self.fusion_layers(fused_features)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits, fused_features, importance

class CyclicalLearningRateScheduler:
    """Cyclical learning rate scheduler"""
    
    def __init__(self, optimizer, base_lr=0.001, max_lr=0.01, step_size=100):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        cycle = np.floor(1 + self.step_count / (2 * self.step_size))
        x = np.abs(self.step_count / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def create_advanced_multimodal_model(tabular_dim, text_dim, graph_dim, 
                                   hidden_dim=256, fusion_dim=512, 
                                   num_classes=2, dropout=0.1, 
                                   num_attention_heads=8, device='cuda'):
    """Create advanced multimodal model"""
    
    model = AdvancedMultimodalFusionModel(
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

def get_advanced_optimizer(model, lr=0.001, weight_decay=1e-4):
    """Get advanced optimizer with better parameters"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

def get_cyclical_scheduler(optimizer, base_lr=0.0001, max_lr=0.001, step_size=50):
    """Get cyclical learning rate scheduler"""
    return CyclicalLearningRateScheduler(optimizer, base_lr, max_lr, step_size)

print("Advanced multimodal architecture loaded!")


