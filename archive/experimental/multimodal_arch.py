# Multimodal Fusion Architecture for Donor Prediction
# Combines Tabular + Text (BERT) + Graph (GNN) Modalities

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("Multimodal fusion libraries loaded!")

# =============================================================================
# STEP 1: MULTIMODAL DATASET WITH MISSING DATA HANDLING
# =============================================================================

class MultimodalDonorDataset(Dataset):
    """
    Dataset that combines tabular, text, and graph features
    Handles missing modalities gracefully
    """
    
    def __init__(self, tabular_features, text_embeddings, graph_embeddings, labels, 
                 modality_mask=None):
        """
        Args:
            tabular_features: (N, D_tab) numpy array or None
            text_embeddings: (N, D_text) numpy array or None
            graph_embeddings: (N, D_graph) numpy array or None
            labels: (N,) numpy array
            modality_mask: (N, 3) binary mask for [tabular, text, graph] availability
        """
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
        tabular_dim = self.tabular.shape[1] if self.tabular is not None else 9
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

# =============================================================================
# STEP 2: CROSS-MODAL ATTENTION MECHANISM
# =============================================================================

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention to learn interactions between modalities
    """
    
    def __init__(self, dim, num_heads=4):
        super(CrossModalAttention, self).__init__()
        self.dim = dim
        
        # Adjust num_heads if dim is not divisible by num_heads
        if dim % num_heads != 0:
            # Find the largest divisor of dim that is <= num_heads
            for i in range(min(num_heads, dim), 0, -1):
                if dim % i == 0:
                    num_heads = i
                    break
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query_modality, key_value_modality, mask=None):
        """
        Args:
            query_modality: (B, dim) - modality asking questions
            key_value_modality: (B, dim) - modality providing context
            mask: (B,) - availability mask for key_value_modality
        """
        B = query_modality.shape[0]
        
        # Linear projections
        Q = self.query(query_modality).view(B, self.num_heads, self.head_dim)
        K = self.key(key_value_modality).view(B, self.num_heads, self.head_dim)
        V = self.value(key_value_modality).view(B, self.num_heads, self.head_dim)
        
        # Attention scores
        scores = torch.einsum('bhd,bhd->bh', Q, K) * self.scale  # (B, num_heads)
        
        # Apply mask if provided
        if mask is not None:
            # Set masked positions to very large negative value (but not -inf to avoid NaN)
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Attention weights with NaN handling
        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads)
        
        # Replace any NaN values with uniform attention (fallback for completely masked inputs)
        if torch.isnan(attn_weights).any():
            attn_weights = torch.where(
                torch.isnan(attn_weights),
                torch.ones_like(attn_weights) / self.num_heads,
                attn_weights
            )
        
        # Apply attention to values
        # Standard multi-head attention: apply attention weights to values and concatenate
        # attn_weights: (B, num_heads), V: (B, num_heads, head_dim)
        
        # Apply attention to each head
        attended_values = attn_weights.unsqueeze(-1) * V  # (B, num_heads, head_dim)
        
        # Concatenate all heads
        out = attended_values.contiguous().view(B, -1)  # (B, num_heads * head_dim) = (B, dim)
        
        # Output projection
        out = self.out(out)
        
        return out, attn_weights

# =============================================================================
# STEP 3: MODALITY-SPECIFIC ENCODERS
# =============================================================================

class TabularEncoder(nn.Module):
    """Encode tabular donor features"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(TabularEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

class TextEncoder(nn.Module):
    """Encode BERT text embeddings"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(TextEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

class GraphEncoder(nn.Module):
    """Encode GNN graph embeddings"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(GraphEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

# =============================================================================
# STEP 4: COMPLETE MULTIMODAL FUSION ARCHITECTURE
# =============================================================================

class MultimodalFusionModel(nn.Module):
    """
    Complete multimodal fusion architecture with:
    - Modality-specific encoders
    - Cross-modal attention
    - Missing data handling
    - Modality importance scoring
    - Unified prediction head
    """
    
    def __init__(self, tabular_dim, text_dim, graph_dim, 
                 hidden_dim=128, fusion_dim=256, num_classes=2,
                 dropout=0.3, num_attention_heads=4):
        super(MultimodalFusionModel, self).__init__()
        
        # Modality-specific encoders (project all to same dimension)
        self.tabular_encoder = TabularEncoder(tabular_dim, hidden_dim, fusion_dim, dropout)
        self.text_encoder = TextEncoder(text_dim, hidden_dim, fusion_dim, dropout)
        self.graph_encoder = GraphEncoder(graph_dim, hidden_dim, fusion_dim, dropout)
        
        # Cross-modal attention mechanisms
        self.text_to_tabular_attn = CrossModalAttention(fusion_dim, num_attention_heads)
        self.graph_to_tabular_attn = CrossModalAttention(fusion_dim, num_attention_heads)
        self.tabular_to_text_attn = CrossModalAttention(fusion_dim, num_attention_heads)
        self.graph_to_text_attn = CrossModalAttention(fusion_dim, num_attention_heads)
        self.tabular_to_graph_attn = CrossModalAttention(fusion_dim, num_attention_heads)
        self.text_to_graph_attn = CrossModalAttention(fusion_dim, num_attention_heads)
        
        # Modality importance gates (learned weights for each modality)
        # Use dynamic input size based on actual fusion_dim
        self.modality_gates = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 3),
            nn.Softmax(dim=1)
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Unified prediction head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, max(fusion_dim // 2, 16)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(fusion_dim // 2, 16), num_classes)
        )
        
        # Store attention weights for interpretability
        self.last_attention_weights = {}
        self.last_modality_importance = None
    
    def forward(self, tabular, text, graph, modality_mask):
        """
        Args:
            tabular: (B, tabular_dim)
            text: (B, text_dim)
            graph: (B, graph_dim)
            modality_mask: (B, 3) - binary mask [has_tabular, has_text, has_graph]
        
        Returns:
            logits: (B, num_classes)
            modality_importance: (B, 3) - learned importance weights
        """
        # batch_size = tabular.shape[0]  # Not used in current implementation
        
        # Encode each modality
        tabular_encoded = self.tabular_encoder(tabular)
        text_encoded = self.text_encoder(text)
        graph_encoded = self.graph_encoder(graph)
        
        # Apply modality masks (zero out missing modalities)
        tabular_encoded = tabular_encoded * modality_mask[:, 0:1]
        text_encoded = text_encoded * modality_mask[:, 1:2]
        graph_encoded = graph_encoded * modality_mask[:, 2:3]
        
        # Cross-modal attention (bidirectional)
        # Tabular attends to text and graph
        tabular_from_text, attn_t2tab = self.text_to_tabular_attn(
            tabular_encoded, text_encoded, modality_mask[:, 1]
        )
        tabular_from_graph, attn_g2tab = self.graph_to_tabular_attn(
            tabular_encoded, graph_encoded, modality_mask[:, 2]
        )
        tabular_enhanced = tabular_encoded + tabular_from_text + tabular_from_graph
        
        # Text attends to tabular and graph
        text_from_tabular, attn_tab2t = self.tabular_to_text_attn(
            text_encoded, tabular_encoded, modality_mask[:, 0]
        )
        text_from_graph, attn_g2t = self.graph_to_text_attn(
            text_encoded, graph_encoded, modality_mask[:, 2]
        )
        text_enhanced = text_encoded + text_from_tabular + text_from_graph
        
        # Graph attends to tabular and text
        graph_from_tabular, attn_tab2g = self.tabular_to_graph_attn(
            graph_encoded, tabular_encoded, modality_mask[:, 0]
        )
        graph_from_text, attn_t2g = self.text_to_graph_attn(
            graph_encoded, text_encoded, modality_mask[:, 1]
        )
        graph_enhanced = graph_encoded + graph_from_tabular + graph_from_text
        
        # Store attention weights for interpretability
        self.last_attention_weights = {
            'text_to_tabular': attn_t2tab,
            'graph_to_tabular': attn_g2tab,
            'tabular_to_text': attn_tab2t,
            'graph_to_text': attn_g2t,
            'tabular_to_graph': attn_tab2g,
            'text_to_graph': attn_t2g
        }
        
        # Learn modality importance (which modalities are most informative)
        combined_modalities = torch.cat([tabular_enhanced, text_enhanced, graph_enhanced], dim=1)
        
        # Check for NaN in combined modalities before gates
        if torch.isnan(combined_modalities).any():
            print("Warning: NaN detected in combined_modalities, replacing with zeros")
            combined_modalities = torch.where(
                torch.isnan(combined_modalities),
                torch.zeros_like(combined_modalities),
                combined_modalities
            )
        
        modality_importance = self.modality_gates(combined_modalities)  # (B, 3)
        
        # Check for NaN in modality importance
        if torch.isnan(modality_importance).any():
            print("Warning: NaN detected in modality_importance, using uniform weights")
            modality_importance = torch.ones_like(modality_importance) / 3.0
        
        self.last_modality_importance = modality_importance
        
        # Apply modality importance weights
        weighted_tabular = tabular_enhanced * modality_importance[:, 0:1]
        weighted_text = text_enhanced * modality_importance[:, 1:2]
        weighted_graph = graph_enhanced * modality_importance[:, 2:3]
        
        # Concatenate weighted modalities
        fused = torch.cat([weighted_tabular, weighted_text, weighted_graph], dim=1)
        
        # Fusion layers
        fused = self.fusion(fused)
        
        # Check for NaN before classification
        if torch.isnan(fused).any():
            print("Warning: NaN detected in fused features, replacing with zeros")
            fused = torch.where(
                torch.isnan(fused),
                torch.zeros_like(fused),
                fused
            )
        
        # Classification
        logits = self.classifier(fused)
        
        return logits, modality_importance

# =============================================================================
# STEP 5: MISSING DATA HANDLING STRATEGY
# =============================================================================

class MissingDataHandler:
    """Handle missing modalities in the dataset"""
    
    def __init__(self):
        self.modality_stats = {}
    
    def create_modality_mask(self, donors_df, contact_reports_df, has_graph_embeddings=True):
        """
        Create binary mask indicating which modalities are available for each donor
        
        Returns:
            modality_mask: (N, 3) array with [has_tabular, has_text, has_graph]
        """
        num_donors = len(donors_df)
        modality_mask = np.ones((num_donors, 3), dtype=np.float32)
        
        # Tabular data (always available for all donors)
        modality_mask[:, 0] = 1
        
        # Text data (only available if donor has contact reports)
        donor_ids_with_reports = set(contact_reports_df['Donor_ID'].unique())
        for idx, donor_id in enumerate(donors_df['ID']):
            if donor_id not in donor_ids_with_reports:
                modality_mask[idx, 1] = 0
        
        # Graph data (only available if donor is in family network)
        if has_graph_embeddings:
            for idx, has_family in enumerate(donors_df['Family_ID'].notna()):
                if not has_family:
                    modality_mask[idx, 2] = 0
        else:
            modality_mask[:, 2] = 0
        
        # Statistics
        self.modality_stats = {
            'total_donors': num_donors,
            'has_all_modalities': (modality_mask.sum(axis=1) == 3).sum(),
            'has_tabular_only': ((modality_mask[:, 0] == 1) & (modality_mask[:, 1:].sum(axis=1) == 0)).sum(),
            'has_tabular_text': ((modality_mask[:, :2].sum(axis=1) == 2) & (modality_mask[:, 2] == 0)).sum(),
            'has_tabular_graph': ((modality_mask[:, 0] == 1) & (modality_mask[:, 2] == 1) & (modality_mask[:, 1] == 0)).sum(),
            'missing_text': (modality_mask[:, 1] == 0).sum(),
            'missing_graph': (modality_mask[:, 2] == 0).sum()
        }
        
        print("Modality Availability Statistics:")
        print("-" * 60)
        for key, value in self.modality_stats.items():
            percentage = (value / num_donors * 100) if 'total' not in key else value
            print(f"  {key}: {value:,} ({percentage:.1f}%)")
        
        return modality_mask
    
    def handle_missing_embeddings(self, embeddings, modality_mask, modality_idx):
        """
        Replace missing embeddings with zeros or learned representations
        
        Args:
            embeddings: (N, D) array of embeddings (may contain NaNs)
            modality_mask: (N, 3) availability mask
            modality_idx: 0=tabular, 1=text, 2=graph
        """
        # Replace NaNs with zeros
        embeddings = np.nan_to_num(embeddings, 0)
        
        # Zero out embeddings for donors without this modality
        embeddings[modality_mask[:, modality_idx] == 0] = 0
        
        return embeddings

# =============================================================================
# STEP 6: TRAINING PIPELINE
# =============================================================================

class MultimodalTrainer:
    """Complete training pipeline for multimodal model"""
    
    def __init__(self, model, device, learning_rate=1e-3, weight_decay=1e-4, class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Use class weights if provided to handle imbalanced data
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.modality_importance_history = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move to device
            tabular = batch['tabular'].to(self.device)
            text = batch['text'].to(self.device)
            graph = batch['graph'].to(self.device)
            modality_mask = batch['modality_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, modality_importance = self.model(tabular, text, graph, modality_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        all_modality_importance = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                tabular = batch['tabular'].to(self.device)
                text = batch['text'].to(self.device)
                graph = batch['graph'].to(self.device)
                modality_mask = batch['modality_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, modality_importance = self.model(tabular, text, graph, modality_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_modality_importance.append(modality_importance.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        # Average modality importance across batch
        avg_modality_importance = np.concatenate(all_modality_importance, axis=0).mean(axis=0)
        
        return avg_loss, accuracy, all_predictions, all_labels, all_probs, avg_modality_importance
    
    def train(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        """Complete training loop"""
        print(f"Training multimodal model for {epochs} epochs...")
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, _, _, _, mod_importance = self.evaluate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.modality_importance_history.append(mod_importance)
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Modality Importance - Tabular: {mod_importance[0]:.3f}, Text: {mod_importance[1]:.3f}, Graph: {mod_importance[2]:.3f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_multimodal_model.pt')
                print("Saved best model!")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_multimodal_model.pt'))
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
        
        return best_val_acc
    
    def plot_training_curves(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(self.val_losses, label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.train_accs, label='Train Acc', marker='o')
        axes[1].plot(self.val_accs, label='Val Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Modality importance over time
        if self.modality_importance_history:
            importance_array = np.array(self.modality_importance_history)
            axes[2].plot(importance_array[:, 0], label='Tabular', marker='o')
            axes[2].plot(importance_array[:, 1], label='Text', marker='s')
            axes[2].plot(importance_array[:, 2], label='Graph', marker='^')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Importance Weight')
            axes[2].set_title('Modality Importance Over Time')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ============================================================================= 
# STEP 7: DATA PREPARATION FOR MULTIMODAL MODEL
# =============================================================================

def prepare_multimodal_data(donors_df, contact_reports_df, 
                           bert_embeddings=None, gnn_embeddings=None,
                           target_column='Legacy_Intent_Binary'):
    """
    Prepare all modalities for multimodal model
    
    Returns:
        tabular_features, text_embeddings, graph_embeddings, labels, modality_mask
    """
    print("Preparing multimodal data...")
    
    # 1. Tabular features (using available columns)
    tabular_cols = [
        'Lifetime_Giving', 'Engagement_Score', 'Estimated_Age'
    ]
    
    # Check which columns are available and add them
    available_cols = []
    for col in tabular_cols:
        if col in donors_df.columns:
            available_cols.append(col)
        else:
            print(f"Warning: Column '{col}' not found in dataset")
    
    # Add any additional numeric columns that might be useful
    numeric_cols = donors_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col not in available_cols and col != 'ID' and col != 'Legacy_Intent_Binary':
            available_cols.append(col)
    
    print(f"Using tabular columns: {available_cols}")
    tabular_cols = available_cols
    
    tabular_features = donors_df[tabular_cols].fillna(0).values
    
    # Scale tabular features
    scaler = StandardScaler()
    tabular_features = scaler.fit_transform(tabular_features)
    
    # Save scaler for inference
    import joblib
    joblib.dump(scaler, 'multimodal_tabular_scaler.pkl')
    print("✅ Saved tabular feature scaler")
    
    print(f"Tabular features shape: {tabular_features.shape}")
    
    # 2. Text embeddings (from BERT)
    if bert_embeddings is not None:
        # Normalize text embeddings for consistent scale
        text_scaler = StandardScaler()
        text_embeddings = text_scaler.fit_transform(bert_embeddings)
        joblib.dump(text_scaler, 'multimodal_text_scaler.pkl')
        print(f"Text embeddings shape: {text_embeddings.shape} (normalized)")
    else:
        # Create dummy embeddings if not available
        text_embeddings = np.zeros((len(donors_df), 768))
        print("No BERT embeddings provided, using zeros")
    
    # 3. Graph embeddings (from GNN)
    if gnn_embeddings is not None:
        # Normalize graph embeddings for consistent scale
        graph_scaler = StandardScaler()
        graph_embeddings = graph_scaler.fit_transform(gnn_embeddings)
        joblib.dump(graph_scaler, 'multimodal_graph_scaler.pkl')
        print(f"Graph embeddings shape: {graph_embeddings.shape} (normalized)")
    else:
        # Create dummy embeddings if not available
        graph_embeddings = np.zeros((len(donors_df), 64))
        print("No GNN embeddings provided, using zeros")
    
    # 4. Labels
    labels = donors_df[target_column].fillna(0).astype(int).values
    print(f"Labels shape: {labels.shape}, distribution: {np.bincount(labels)}")
    
    # 5. Modality mask
    handler = MissingDataHandler()
    modality_mask = handler.create_modality_mask(
        donors_df, 
        contact_reports_df,
        has_graph_embeddings=(gnn_embeddings is not None)
    )
    
    return tabular_features, text_embeddings, graph_embeddings, labels, modality_mask

# =============================================================================
# STEP 8: MAIN EXECUTION PIPELINE
# =============================================================================

def run_multimodal_fusion_pipeline(
    donors_df,
    contact_reports_df,
    bert_embeddings=None,
    gnn_embeddings=None,
    target_column='Legacy_Intent_Binary',
    batch_size=32,
    epochs=50,
    learning_rate=1e-3
):
    """
    Complete multimodal fusion pipeline
    
    Args:
        donors_df: DataFrame with donor information
        contact_reports_df: DataFrame with contact reports
        bert_embeddings: (N, 768) BERT embeddings or None
        gnn_embeddings: (N, D) GNN embeddings or None
        target_column: Column name for target variable
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        Dictionary with model, trainer, and results
    """
    
    print("=" * 80)
    print("MULTIMODAL FUSION PIPELINE")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Prepare data
    print("\n" + "=" * 60)
    print("PREPARING MULTIMODAL DATA")
    print("=" * 60)
    
    tabular, text, graph, labels, modality_mask = prepare_multimodal_data(
        donors_df, contact_reports_df, bert_embeddings, gnn_embeddings, target_column
    )
    
    # Split data
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])
    
    # Create datasets
    train_dataset = MultimodalDonorDataset(
        tabular[train_idx], text[train_idx], graph[train_idx], 
        labels[train_idx], modality_mask[train_idx]
    )
    val_dataset = MultimodalDonorDataset(
        tabular[val_idx], text[val_idx], graph[val_idx],
        labels[val_idx], modality_mask[val_idx]
    )
    test_dataset = MultimodalDonorDataset(
        tabular[test_idx], text[test_idx], graph[test_idx],
        labels[test_idx], modality_mask[test_idx]
    )
    
    # Create data loaders (num_workers=0 for Colab compatibility)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    
    # Initialize model
    print("\n" + "=" * 60)
    print("INITIALIZING MULTIMODAL MODEL")
    print("=" * 60)
    
    # Dynamically determine dimensions
    tabular_dim = tabular.shape[1]
    text_dim = text.shape[1]
    graph_dim = graph.shape[1]
    
    print(f"Model dimensions: Tabular={tabular_dim}, Text={text_dim}, Graph={graph_dim}")
    
    # Use a smaller fusion_dim to avoid dimension mismatches
    fusion_dim = 32  # Use smaller dimension to avoid memory issues
    print(f"Using fusion_dim: {fusion_dim}")
    
    # Reduce attention heads to ensure compatibility
    num_attention_heads = 2  # Must divide evenly into fusion_dim
    
    model = MultimodalFusionModel(
        tabular_dim=tabular_dim,
        text_dim=text_dim,
        graph_dim=graph_dim,
        hidden_dim=32,  # Reduce hidden_dim to match fusion_dim
        fusion_dim=fusion_dim,
        num_classes=len(np.unique(labels)),
        dropout=0.3,
        num_attention_heads=num_attention_heads
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test model with sample data to catch dimension issues early
    print("Testing model with sample data...")
    try:
        sample_tabular = torch.randn(2, tabular_dim)
        sample_text = torch.randn(2, text_dim)
        sample_graph = torch.randn(2, graph_dim)
        sample_mask = torch.ones(2, 3)
        
        with torch.no_grad():
            _ = model(sample_tabular, sample_text, sample_graph, sample_mask)
        print("✅ Model test passed!")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        raise e
    
    # Train model
    print("\n" + "=" * 60)
    print("TRAINING MULTIMODAL MODEL")
    print("=" * 60)
    
    # Calculate class weights to handle imbalanced data
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    print(f"Class weights (to handle imbalance): {class_weights}")
    
    trainer = MultimodalTrainer(model, device, learning_rate=learning_rate, class_weights=class_weights)
    best_val_acc = trainer.train(train_loader, val_loader, epochs=epochs, early_stopping_patience=10)
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    test_loss, test_acc, predictions, true_labels, probs, test_mod_importance = trainer.evaluate(test_loader)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Modality Importance - Tabular: {test_mod_importance[0]:.3f}, Text: {test_mod_importance[1]:.3f}, Graph: {test_mod_importance[2]:.3f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['No Legacy Intent', 'Legacy Intent']))
    
    # AUC score
    try:
        probs_array = np.array(probs)
        auc = roc_auc_score(true_labels, probs_array[:, 1])
        print(f"\nTest AUC: {auc:.4f}")
    except Exception as e:
        print(f"Could not calculate AUC: {e}")
        auc = None
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Legacy Intent', 'Legacy Intent'],
                yticklabels=['No Legacy Intent', 'Legacy Intent'])
    plt.title('Confusion Matrix - Multimodal Prediction')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Analyze modality importance
    print(f"Average Modality Importance on Test Set:")
    print(f"  Tabular: {test_mod_importance[0]:.3f}")
    print(f"  Text: {test_mod_importance[1]:.3f}")
    print(f"  Graph: {test_mod_importance[2]:.3f}")
    
    # Save results
    results = {
        'test_accuracy': test_acc,
        'test_auc': auc,
        'best_val_accuracy': best_val_acc,
        'test_modality_importance': test_mod_importance,
        'predictions': predictions,
        'true_labels': true_labels
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results
    }, 'multimodal_fusion_model.pt')
    
    print("\nModel saved!")
    
    return {
        'model': model,
        'trainer': trainer,
        'results': results,
        'test_loader': test_loader
    }