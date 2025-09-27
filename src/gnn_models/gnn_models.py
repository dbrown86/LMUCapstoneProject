# Graph Neural Network models for donor analysis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.utils import to_networkx, from_networkx
import torch_geometric.transforms as T
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

class DonorGraphPreprocessor:
    """Comprehensive preprocessing for donor graph data"""
    
    def __init__(self, donors_df, relationships_df, contact_reports_df=None, giving_history_df=None):
        self.donors_df = donors_df.copy()
        self.relationships_df = relationships_df.copy()
        self.contact_reports_df = contact_reports_df.copy() if contact_reports_df is not None else pd.DataFrame()
        self.giving_history_df = giving_history_df.copy() if giving_history_df is not None else pd.DataFrame()
        
        # Initialize encoders
        self.label_encoders = {}
        self.feature_scaler = StandardScaler()
        
        print("DonorGraphPreprocessor initialized")
    
    def create_node_features(self):
        """Create comprehensive node features for donors"""
        print("Creating node features...")
        
        # Select and prepare numeric features
        numeric_features = [
            'Lifetime_Giving', 'Last_Gift', 'Consecutive_Yr_Giving_Count', 
            'Total_Yr_Giving_Count', 'Engagement_Score', 'Legacy_Intent_Probability',
            'Estimated_Age'
        ]
        
        # Handle missing values
        for col in numeric_features:
            if col in self.donors_df.columns:
                self.donors_df[col] = self.donors_df[col].fillna(0)
        
        # Create feature matrix
        X_numeric = self.donors_df[numeric_features].values
        
        # Encode categorical features
        categorical_features = ['Rating', 'Primary_Constituent_Type', 'Prospect_Stage', 'Geographic_Region']
        X_categorical = []
        
        for feature in categorical_features:
            if feature in self.donors_df.columns:
                le = LabelEncoder()
                encoded = le.fit_transform(self.donors_df[feature].astype(str))
                self.label_encoders[feature] = le
                X_categorical.append(encoded.reshape(-1, 1))
        
        if X_categorical:
            X_categorical = np.hstack(X_categorical)
            X_features = np.hstack([X_numeric, X_categorical])
        else:
            X_features = X_numeric
        
        # Scale features
        X_features_scaled = self.feature_scaler.fit_transform(X_features)
        
        # Add family-specific features
        family_features = self._create_family_features()
        X_features_final = np.hstack([X_features_scaled, family_features])
        
        print(f"Created node features: {X_features_final.shape}")
        return torch.tensor(X_features_final, dtype=torch.float)
    
    def _create_family_features(self):
        """Create family-specific node features"""
        family_features = []
        
        for _, donor in self.donors_df.iterrows():
            features = []
            
            # Has family indicator
            features.append(1.0 if pd.notna(donor.get('Family_ID')) else 0.0)
            
            # Family giving potential encoding
            fgp = donor.get('Family_Giving_Potential', 'Individual')
            fgp_encoding = {'Individual': 0, 'Low': 1, 'Medium': 2, 'High': 3}
            features.append(fgp_encoding.get(fgp, 0))
            
            # Relationship type encoding
            rel_type = donor.get('Relationship_Type', 'None')
            rel_encoding = {'None': 0, 'Head': 1, 'Spouse': 2, 'Child': 3, 'Parent': 4, 'Sibling': 5}
            features.append(rel_encoding.get(rel_type, 0))
            
            family_features.append(features)
        
        return np.array(family_features)
    
    def create_edge_index(self):
        """Create edge index from family relationships"""
        print("Creating edge index from family relationships...")
        
        if self.relationships_df.empty:
            print("No family relationships found. Creating empty graph.")
            num_nodes = len(self.donors_df)
            return torch.tensor([[], []], dtype=torch.long), torch.tensor([], dtype=torch.float)
        
        # Create donor ID to index mapping
        donor_ids = self.donors_df['ID'].values
        id_to_idx = {donor_id: idx for idx, donor_id in enumerate(donor_ids)}
        
        edge_list = []
        edge_weights = []
        
        # Group by family and create edges
        family_groups = self.relationships_df.groupby('Family_ID')
        
        for family_id, family_members in family_groups:
            member_ids = family_members['Donor_ID'].values
            
            # Create edges between all family members (complete subgraph)
            for i, donor_i in enumerate(member_ids):
                for j, donor_j in enumerate(member_ids):
                    if i < j and donor_i in id_to_idx and donor_j in id_to_idx:
                        idx_i = id_to_idx[donor_i]
                        idx_j = id_to_idx[donor_j]
                        
                        # Add both directions for undirected graph
                        edge_list.extend([[idx_i, idx_j], [idx_j, idx_i]])
                        
                        # Calculate edge weight based on giving correlation
                        weight = self._calculate_edge_weight(donor_i, donor_j)
                        edge_weights.extend([weight, weight])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
            edge_attr = torch.tensor([], dtype=torch.float)
        
        print(f"Created edge index: {edge_index.shape}")
        return edge_index, edge_attr
    
    def _calculate_edge_weight(self, donor_i, donor_j):
        """Calculate edge weight between two family members"""
        # Get donor information
        donor_i_info = self.donors_df[self.donors_df['ID'] == donor_i].iloc[0]
        donor_j_info = self.donors_df[self.donors_df['ID'] == donor_j].iloc[0]
        
        # Base weight
        weight = 0.5
        
        # Increase weight if both are donors
        if donor_i_info['Lifetime_Giving'] > 0 and donor_j_info['Lifetime_Giving'] > 0:
            weight += 0.3
        
        # Increase weight based on giving similarity
        giving_i = donor_i_info['Lifetime_Giving']
        giving_j = donor_j_info['Lifetime_Giving']
        
        if giving_i > 0 and giving_j > 0:
            # Normalized giving similarity
            max_giving = max(giving_i, giving_j)
            min_giving = min(giving_i, giving_j)
            similarity = min_giving / max_giving if max_giving > 0 else 0
            weight += similarity * 0.2
        
        return weight
    
    def create_target_labels(self, target_column='Legacy_Intent_Binary'):
        """Create target labels for node classification"""
        print(f"Creating target labels from {target_column}...")
        
        if target_column not in self.donors_df.columns:
            print(f"Warning: {target_column} not found. Creating random labels.")
            labels = torch.randint(0, 2, (len(self.donors_df),))
        else:
            labels = torch.tensor(self.donors_df[target_column].values, dtype=torch.long)
        
        print(f"Created labels: {labels.shape}, distribution: {torch.bincount(labels)}")
        return labels
    
    def create_graph_data(self, target_column='Legacy_Intent_Binary'):
        """Create complete PyTorch Geometric Data object"""
        print("Creating complete graph data object...")
        
        # Create all components
        x = self.create_node_features()
        edge_index, edge_attr = self.create_edge_index()
        y = self.create_target_labels(target_column)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(self.donors_df)
        )
        
        print(f"Graph created successfully!")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Node features: {data.x.shape}")
        print(f"  Has edge attributes: {data.edge_attr is not None}")
        
        return data

class FeatureEncoder:
    """Advanced feature encoding system for donor graphs"""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
    
    def encode_contact_reports(self, contact_reports_df, donors_df):
        """Encode contact report text features using TF-IDF"""
        print("Encoding contact report features...")
        
        if contact_reports_df.empty:
            return np.zeros((len(donors_df), 5))  # Return zero features
        
        # Aggregate contact reports per donor
        donor_reports = contact_reports_df.groupby('Donor_ID')['Report_Text'].apply(' '.join).reset_index()
        
        # Create TF-IDF features
        tfidf = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
        tfidf_features = tfidf.fit_transform(donor_reports['Report_Text'].fillna(''))
        
        # Reduce dimensionality using SVD
        svd = TruncatedSVD(n_components=5, random_state=42)
        text_features = svd.fit_transform(tfidf_features.toarray())
        
        # Map back to all donors
        text_feature_dict = dict(zip(donor_reports['Donor_ID'], text_features))
        
        donor_text_features = []
        for donor_id in donors_df['ID']:
            if donor_id in text_feature_dict:
                donor_text_features.append(text_feature_dict[donor_id])
            else:
                donor_text_features.append(np.zeros(5))
        
        self.encoders['contact_tfidf'] = tfidf
        self.encoders['contact_svd'] = svd
        
        print(f"Contact report features encoded: {np.array(donor_text_features).shape}")
        return np.array(donor_text_features)
    
    def encode_giving_patterns(self, giving_history_df, donors_df):
        """Encode giving history patterns"""
        print("Encoding giving pattern features...")
        
        if giving_history_df.empty:
            return np.zeros((len(donors_df), 6))
        
        # Calculate giving patterns per donor
        giving_patterns = []
        
        for donor_id in donors_df['ID']:
            donor_gifts = giving_history_df[giving_history_df['Donor_ID'] == donor_id]
            
            if len(donor_gifts) == 0:
                patterns = [0, 0, 0, 0, 0, 0]  # No giving history
            else:
                # Extract temporal patterns
                donor_gifts['Gift_Date'] = pd.to_datetime(donor_gifts['Gift_Date'])
                donor_gifts = donor_gifts.sort_values('Gift_Date')
                
                # Calculate features
                total_gifts = len(donor_gifts)
                avg_gift = donor_gifts['Gift_Amount'].mean()
                std_gift = donor_gifts['Gift_Amount'].std() if len(donor_gifts) > 1 else 0
                trend = self._calculate_giving_trend(donor_gifts)
                recency = (pd.Timestamp.now() - donor_gifts['Gift_Date'].max()).days
                frequency = total_gifts / max(1, (donor_gifts['Gift_Date'].max() - donor_gifts['Gift_Date'].min()).days / 365.25)
                
                patterns = [total_gifts, avg_gift, std_gift, trend, recency, frequency]
            
            giving_patterns.append(patterns)
        
        # Scale the features
        scaler = StandardScaler()
        giving_patterns_scaled = scaler.fit_transform(giving_patterns)
        self.scalers['giving_patterns'] = scaler
        
        print(f"Giving pattern features encoded: {giving_patterns_scaled.shape}")
        return giving_patterns_scaled
    
    def _calculate_giving_trend(self, donor_gifts):
        """Calculate giving trend (increasing, stable, decreasing)"""
        if len(donor_gifts) < 2:
            return 0
        
        amounts = donor_gifts['Gift_Amount'].values
        x = np.arange(len(amounts))
        
        # Simple linear regression to get trend
        slope = np.polyfit(x, amounts, 1)[0]
        return slope

class GraphSAGE(nn.Module):
    """GraphSAGE implementation for donor classification"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        # Apply GraphSAGE convolutions
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling for graph-level predictions (if batch is provided)
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        
        return x

class GCNModel(nn.Module):
    """Graph Convolutional Network implementation"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(GCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x, edge_index, batch=None):
        # Apply GCN convolutions
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling for graph-level predictions (if batch is provided)
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        
        return x

class DonorGNNTrainer:
    """Complete training pipeline for donor GNN models"""
    
    def __init__(self, model, device, lr=0.01, weight_decay=5e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def create_masks(self, data, train_ratio=0.6, val_ratio=0.2):
        """Create train/validation/test masks"""
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        return train_mask, val_mask, test_mask
    
    def train_epoch(self, data, train_mask):
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(data.x, data.edge_index)
        loss = self.criterion(out[train_mask], data.y[train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            pred = out[train_mask].argmax(dim=1)
            acc = (pred == data.y[train_mask]).float().mean()
        
        return loss.item(), acc.item()
    
    def evaluate(self, data, mask):
        """Evaluate model"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            loss = self.criterion(out[mask], data.y[mask])
            pred = out[mask].argmax(dim=1)
            acc = (pred == data.y[mask]).float().mean()
            
            # Calculate AUC if binary classification
            if out.shape[1] == 2:
                probs = F.softmax(out[mask], dim=1)[:, 1]
                try:
                    auc = roc_auc_score(data.y[mask].cpu(), probs.cpu())
                except:
                    auc = 0.5
            else:
                auc = 0.0
        
        return loss.item(), acc.item(), auc
    
    def train(self, data, epochs=200, early_stopping_patience=20):
        """Complete training loop"""
        print("Starting training...")
        
        # Move data to device
        data = data.to(self.device)
        
        # Create masks
        train_mask, val_mask, test_mask = self.create_masks(data)
        print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Train
            train_loss, train_acc = self.train_epoch(data, train_mask)
            
            # Validate
            val_loss, val_acc, val_auc = self.evaluate(data, val_mask)
            
            # Update scheduler
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_donor_gnn_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if epoch % 20 == 0:
                print(f"Epoch {epoch:03d}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Load best model and evaluate on test set
        self.model.load_state_dict(torch.load('best_donor_gnn_model.pt'))
        test_loss, test_acc, test_auc = self.evaluate(data, test_mask)
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        return {
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        axes[0].plot(self.train_losses, label='Train Loss', alpha=0.7)
        axes[0].plot(self.val_losses, label='Val Loss', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(self.train_accs, label='Train Acc', alpha=0.7)
        axes[1].plot(self.val_accs, label='Val Acc', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
