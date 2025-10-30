#!/usr/bin/env python3
"""
Advanced Multimodal Fusion Architecture V3
Improved version with UMAP, feature selection, and better training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced data"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LearnableFusionWeights(nn.Module):
    """Learnable fusion weights for different modalities"""
    
    def __init__(self, num_modalities=3, hidden_dim=128):
        super(LearnableFusionWeights, self).__init__()
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        
        # Learnable weights for each modality
        self.weight_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        
        # Global fusion weight
        self.global_weight = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=1)
        )
        
    def forward(self, modalities):
        # modalities shape: (batch_size, num_modalities, hidden_dim)
        batch_size, num_modalities, hidden_dim = modalities.size()
        
        # Calculate individual modality weights
        modality_weights = []
        for i, weight_net in enumerate(self.weight_networks):
            weight = weight_net(modalities[:, i, :])
            modality_weights.append(weight)
        
        modality_weights = torch.cat(modality_weights, dim=1)  # (batch_size, num_modalities)
        
        # Calculate global fusion weights
        global_weights = self.global_weight(modalities.view(batch_size, -1))
        
        # Combine individual and global weights
        final_weights = modality_weights * global_weights
        
        # Apply weights
        weighted_modalities = modalities * final_weights.unsqueeze(-1)
        
        return weighted_modalities, final_weights

class HierarchicalFusion(nn.Module):
    """Hierarchical fusion with local and global attention"""
    
    def __init__(self, dim, num_heads=16, num_layers=4, dropout=0.1):
        super(HierarchicalFusion, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Local attention (within modalities)
        self.local_attention = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Global attention (across modalities)
        self.global_attention = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers * 2)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, num_modalities, dim)
        for i in range(self.num_layers):
            # Local attention within each modality
            x_norm = self.layer_norms[i * 2](x)
            local_out, _ = self.local_attention(x_norm, x_norm, x_norm)
            x = x + self.dropout(local_out)
            
            # Global attention across modalities
            x_norm = self.layer_norms[i * 2 + 1](x)
            global_out, _ = self.global_attention(x_norm, x_norm, x_norm)
            x = x + self.dropout(global_out)
            
            # Feed-forward network
            x = x + self.ffns[i](x)
        
        return x

class AdvancedMultimodalFusionModelV3(nn.Module):
    """Advanced multimodal fusion model V3 with major improvements"""
    
    def __init__(self, tabular_dim, text_dim, graph_dim, 
                 hidden_dim=256, fusion_dim=512, num_classes=2, 
                 dropout=0.1, num_attention_heads=16):
        super(AdvancedMultimodalFusionModelV3, self).__init__()
        
        self.tabular_dim = tabular_dim
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        
        # Enhanced modality-specific encoders
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Learnable fusion weights
        self.fusion_weights = LearnableFusionWeights(3, 128)
        
        # Hierarchical fusion
        self.hierarchical_fusion = HierarchicalFusion(128, num_attention_heads, 4, dropout)
        
        # Enhanced fusion layers
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
        
        # Encode each modality
        tabular_encoded = self.tabular_encoder(tabular)
        text_encoded = self.text_encoder(text)
        graph_encoded = self.graph_encoder(graph)
        
        # Stack modalities for fusion
        modalities = torch.stack([tabular_encoded, text_encoded, graph_encoded], dim=1)
        
        # Apply learnable fusion weights
        weighted_modalities, fusion_weights = self.fusion_weights(modalities)
        
        # Apply hierarchical fusion
        fused_modalities = self.hierarchical_fusion(weighted_modalities)
        
        # Flatten for final fusion
        fused_features = fused_modalities.view(batch_size, -1)
        
        # Fusion layers
        fused = self.fusion_layers(fused_features)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits, fused_features, fusion_weights

class OneCycleLR:
    """OneCycleLR scheduler implementation"""
    
    def __init__(self, optimizer, max_lr=0.01, total_steps=None, pct_start=0.3, 
                 anneal_strategy='cos', div_factor=25, final_div_factor=1e4):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.base_lr = max_lr / div_factor
        self.final_lr = self.base_lr / final_div_factor
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        if self.step_count > self.total_steps:
            return
            
        # Calculate current phase
        if self.step_count <= self.total_steps * self.pct_start:
            # Warmup phase
            pct = self.step_count / (self.total_steps * self.pct_start)
            lr = self.base_lr + (self.max_lr - self.base_lr) * pct
        else:
            # Annealing phase
            pct = (self.step_count - self.total_steps * self.pct_start) / (self.total_steps * (1 - self.pct_start))
            if self.anneal_strategy == 'cos':
                lr = self.final_lr + (self.max_lr - self.final_lr) * (1 + math.cos(math.pi * pct)) / 2
            else:
                lr = self.max_lr - (self.max_lr - self.final_lr) * pct
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def create_advanced_multimodal_model_v3(tabular_dim, text_dim, graph_dim, 
                                       hidden_dim=256, fusion_dim=512, 
                                       num_classes=2, dropout=0.1, 
                                       num_attention_heads=16, device='cuda'):
    """Create advanced multimodal model V3"""
    
    model = AdvancedMultimodalFusionModelV3(
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

def get_advanced_optimizer_v3(model, lr=0.001, weight_decay=1e-4):
    """Get advanced optimizer V3"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

def get_onecycle_scheduler(optimizer, max_lr=0.01, total_steps=500, pct_start=0.3):
    """Get OneCycleLR scheduler"""
    return OneCycleLR(optimizer, max_lr, total_steps, pct_start)

def reduce_embeddings_umap(embeddings, n_components=128, random_state=42):
    """Reduce embeddings using UMAP"""
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine'
    )
    return reducer.fit_transform(embeddings)

print("Advanced multimodal architecture V3 loaded!")


