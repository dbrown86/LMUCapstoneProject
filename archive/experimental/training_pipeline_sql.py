#!/usr/bin/env python3
"""
SQL-Integrated Training Pipeline for Multimodal Donor Prediction
Updated to use the 500K donor SQL database with efficient data loading
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from .sql_data_loader import SQLDataLoader, load_multimodal_data, build_donor_graph
from .multimodal_arch_sql import MultimodalDonorPredictor, MultimodalDataProcessor, create_multimodal_model

# Try to import PyTorch Geometric
try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False


class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal donor data
    Handles tabular, graph, and sequential data
    """
    
    def __init__(self, 
                 tabular_features: torch.Tensor,
                 labels: torch.Tensor,
                 graph_data: Optional[Data] = None,
                 sequence_features: Optional[torch.Tensor] = None,
                 sequence_lengths: Optional[torch.Tensor] = None,
                 donor_ids: Optional[List[int]] = None):
        
        self.tabular_features = tabular_features
        self.labels = labels
        self.graph_data = graph_data
        self.sequence_features = sequence_features
        self.sequence_lengths = sequence_lengths
        self.donor_ids = donor_ids or list(range(len(tabular_features)))
        
        assert len(tabular_features) == len(labels), "Features and labels must have same length"
    
    def __len__(self):
        return len(self.tabular_features)
    
    def __getitem__(self, idx):
        item = {
            'tabular_features': self.tabular_features[idx],
            'labels': self.labels[idx],
            'donor_id': self.donor_ids[idx]
        }
        
        if self.graph_data is not None:
            # For graph data, we need to handle batching differently
            # This is a simplified approach - in practice, you'd want more sophisticated graph batching
            item['graph_data'] = self.graph_data
        
        if self.sequence_features is not None:
            item['sequence_features'] = self.sequence_features[idx]
            if self.sequence_lengths is not None:
                item['sequence_lengths'] = self.sequence_lengths[idx]
        
        return item


class SQLTrainingPipeline:
    """
    Complete training pipeline for multimodal donor prediction using SQL database
    """
    
    def __init__(self, 
                 db_path: str = "data/synthetic_donor_dataset_500k_dense/donor_database.db",
                 model_save_dir: str = "models",
                 results_save_dir: str = "results"):
        
        self.db_path = db_path
        self.model_save_dir = Path(model_save_dir)
        self.results_save_dir = Path(results_save_dir)
        
        # Create directories
        self.model_save_dir.mkdir(exist_ok=True)
        self.results_save_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_processor = MultimodalDataProcessor(db_path)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
    
    def create_target_variable(self, 
                              donors_df: pd.DataFrame,
                              target_type: str = 'high_value') -> np.ndarray:
        """
        Create target variable for classification
        
        Args:
            donors_df: Donor data
            target_type: Type of target ('high_value', 'frequent_giver', 'event_attender')
            
        Returns:
            Target labels
        """
        if target_type == 'high_value':
            # High-value donors (top 20% by lifetime giving)
            threshold = donors_df['Lifetime_Giving'].quantile(0.8)
            return (donors_df['Lifetime_Giving'] >= threshold).astype(int).values
        
        elif target_type == 'frequent_giver':
            # Frequent givers (multiple years of giving)
            threshold = donors_df['Total_Yr_Giving_Count'].quantile(0.7)
            return (donors_df['Total_Yr_Giving_Count'] >= threshold).astype(int).values
        
        elif target_type == 'event_attender':
            # Event attenders (have attended events)
            return (donors_df['Primary_Constituent_Type'].isin(['Alum', 'Trustee', 'Regent'])).astype(int).values
        
        else:
            raise ValueError(f"Unknown target type: {target_type}")
    
    def load_training_data(self, 
                          target_type: str = 'high_value',
                          sample_size: Optional[int] = None,
                          test_size: float = 0.2,
                          val_size: float = 0.2,
                          random_state: int = 42) -> Dict[str, Any]:
        """
        Load and split training data from SQL database
        
        Args:
            target_type: Type of target variable to create
            sample_size: Number of samples to use (None for all)
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            random_state: Random seed
            
        Returns:
            Dictionary with train/val/test data
        """
        print("Loading data from SQL database...")
        
        with SQLDataLoader(self.db_path) as loader:
            # Load donor data
            donors_df = loader.get_donors(limit=sample_size)
            print(f"Loaded {len(donors_df):,} donors")
            
            # Create target variable
            labels = self.create_target_variable(donors_df, target_type)
            print(f"Target distribution: {np.bincount(labels)}")
            
            # Load all multimodal data with limit
            data = loader.get_multimodal_data(donor_ids=donors_df['ID'].tolist(), limit=sample_size)
            
            # Prepare tabular features
            tabular_features = self.data_processor.prepare_tabular_features(
                data['donors'], 
                data['enhanced_fields'],
                fit_scaler=True
            )
            
            # Prepare graph data
            if PYTORCH_GEOMETRIC_AVAILABLE:
                graph_data = loader.build_graph_data(
                    donor_ids=donors_df['ID'].tolist()
                )
                print(f"Graph data: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
            else:
                graph_data = None
            
            # Prepare sequence features
            sequence_features, sequence_lengths = self.data_processor.prepare_sequence_features(
                data['giving_history'],
                data['event_attendance']
            )
            
            # Convert to tensors
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            # Split data
            X_temp, X_test, y_temp, y_test = train_test_split(
                tabular_features, labels_tensor, 
                test_size=test_size, random_state=random_state, stratify=labels
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, 
                test_size=val_size, random_state=random_state, stratify=y_temp.numpy()
            )
            
            # Create datasets
            train_dataset = MultimodalDataset(
                X_train, y_train, graph_data, sequence_features, sequence_lengths
            )
            val_dataset = MultimodalDataset(
                X_val, y_val, graph_data, sequence_features, sequence_lengths
            )
            test_dataset = MultimodalDataset(
                X_test, y_test, graph_data, sequence_features, sequence_lengths
            )
            
            return {
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'test_dataset': test_dataset,
                'graph_data': graph_data,
                'donor_ids': donors_df['ID'].tolist()
            }
    
    def create_model(self, 
                    tabular_input_dim: int,
                    graph_input_dim: int = 10,
                    sequence_input_dim: int = 10,
                    num_classes: int = 2,
                    hidden_dim: int = 128,
                    dropout: float = 0.3) -> MultimodalDonorPredictor:
        """
        Create multimodal model
        
        Args:
            tabular_input_dim: Dimension of tabular features
            graph_input_dim: Dimension of graph node features
            sequence_input_dim: Dimension of sequence features
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
            
        Returns:
            Multimodal model
        """
        self.model = MultimodalDonorPredictor(
            tabular_input_dim=tabular_input_dim,
            graph_input_dim=graph_input_dim,
            sequence_input_dim=sequence_input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_graph=PYTORCH_GEOMETRIC_AVAILABLE,
            use_sequence=True
        )
        
        return self.model
    
    def train_epoch(self, 
                   model: nn.Module,
                   dataloader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   device: torch.device) -> Dict[str, float]:
        """
        Train model for one epoch
        
        Args:
            model: Model to train
            dataloader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to use
            
        Returns:
            Dictionary with training metrics
        """
        model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Move data to device
            tabular_features = batch['tabular_features'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits, attention_weights = model(
                tabular_features=tabular_features,
                graph_data=batch.get('graph_data'),
                sequence_features=batch.get('sequence_features'),
                sequence_lengths=batch.get('sequence_lengths')
            )
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1
        }
    
    def validate_epoch(self, 
                      model: nn.Module,
                      dataloader: DataLoader,
                      criterion: nn.Module,
                      device: torch.device) -> Dict[str, float]:
        """
        Validate model for one epoch
        
        Args:
            model: Model to validate
            dataloader: Validation data loader
            criterion: Loss function
            device: Device to use
            
        Returns:
            Dictionary with validation metrics
        """
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                tabular_features = batch['tabular_features'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                logits, attention_weights = model(
                    tabular_features=tabular_features,
                    graph_data=batch.get('graph_data'),
                    sequence_features=batch.get('sequence_features'),
                    sequence_lengths=batch.get('sequence_lengths')
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
        
        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Compute AUC if binary classification
        if len(np.unique(all_labels)) == 2:
            auc = roc_auc_score(all_labels, [p[1] for p in all_probabilities])
        else:
            auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc
        }
    
    def train(self, 
              data: Dict[str, Any],
              num_epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-5,
              patience: int = 10,
              device: str = 'auto') -> Dict[str, Any]:
        """
        Train the multimodal model
        
        Args:
            data: Training data dictionary
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            device: Device to use ('auto', 'cpu', 'cuda')
            
        Returns:
            Training results
        """
        # Set device
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        print(f"Training on device: {device}")
        
        # Create model if not exists
        if self.model is None:
            tabular_dim = data['train_dataset'].tabular_features.shape[1]
            graph_dim = data['graph_data'].x.shape[1] if data['graph_data'] is not None else 10
            sequence_dim = data['train_dataset'].sequence_features.shape[2] if data['train_dataset'].sequence_features is not None else 10
            
            self.model = self.create_model(
                tabular_input_dim=tabular_dim,
                graph_input_dim=graph_dim,
                sequence_input_dim=sequence_dim
            )
        
        self.model = self.model.to(device)
        
        # Create data loaders
        train_loader = DataLoader(
            data['train_dataset'], 
            batch_size=batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            data['val_dataset'], 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # Setup training
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Compute class weights for imbalanced data
        train_labels = data['train_dataset'].labels.numpy()
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(self.model, train_loader, optimizer, criterion, device)
            
            # Validate
            val_metrics = self.validate_epoch(self.model, val_loader, criterion, device)
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['train_f1'].append(train_metrics['f1'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}, "
                      f"Val F1: {val_metrics['f1']:.4f}")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                self.save_model('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.load_model('best_model.pt')
        
        # Final evaluation
        test_loader = DataLoader(data['test_dataset'], batch_size=batch_size, shuffle=False)
        test_metrics = self.validate_epoch(self.model, test_loader, criterion, device)
        
        print(f"\nFinal Test Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {test_metrics['f1']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        
        return {
            'training_history': self.training_history,
            'test_metrics': test_metrics,
            'best_val_loss': best_val_loss
        }
    
    def save_model(self, filename: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = self.model_save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, filename: str):
        """Load model from file"""
        load_path = self.model_save_dir / filename
        checkpoint = torch.load(load_path, map_location='cpu')
        
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        print(f"Model loaded from {load_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.training_history['train_acc'], label='Train')
        axes[0, 1].plot(self.training_history['val_acc'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(self.training_history['train_f1'], label='Train')
        axes[1, 0].plot(self.training_history['val_f1'], label='Validation')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined metrics
        axes[1, 1].plot(self.training_history['val_loss'], label='Val Loss', alpha=0.7)
        axes[1, 1].plot(self.training_history['val_acc'], label='Val Acc', alpha=0.7)
        axes[1, 1].plot(self.training_history['val_f1'], label='Val F1', alpha=0.7)
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SQLTrainingPipeline()
    
    # Load training data
    data = pipeline.load_training_data(
        target_type='high_value',
        sample_size=10000,  # Use 10K samples for quick testing
        test_size=0.2,
        val_size=0.2
    )
    
    print(f"Training data: {len(data['train_dataset'])} samples")
    print(f"Validation data: {len(data['val_dataset'])} samples")
    print(f"Test data: {len(data['test_dataset'])} samples")
    
    # Train model
    results = pipeline.train(
        data=data,
        num_epochs=50,
        batch_size=64,
        learning_rate=0.001,
        patience=10
    )
    
    # Plot training history
    pipeline.plot_training_history('training_history.png')
    
    print("Training completed!")
