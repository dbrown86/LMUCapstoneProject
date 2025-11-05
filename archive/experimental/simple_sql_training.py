#!/usr/bin/env python3
"""
Simple SQL Training Pipeline
A simplified approach that avoids SQL parameter limits by using different strategies
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from .sql_data_loader import SQLDataLoader
from .multimodal_arch_sql import MultimodalDonorPredictor, MultimodalDataProcessor

# Try to import PyTorch Geometric
try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False


class SimpleSQLTrainingPipeline:
    """
    Simple training pipeline that avoids SQL parameter limits
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
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
    
    def get_sample_donors(self, 
                         target_type: str = 'high_value',
                         sample_size: int = 10000,
                         random_state: int = 42) -> pd.DataFrame:
        """
        Get a sample of donors using a different approach to avoid SQL parameter limits
        
        Args:
            target_type: Type of target variable to create
            sample_size: Number of samples to return
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with sampled donor data
        """
        print(f"📊 Getting sample of {sample_size:,} donors...")
        
        with SQLDataLoader(self.db_path) as loader:
            # Get a random sample of donors using SQL LIMIT and OFFSET
            # This avoids the parameter limit issue
            offset = np.random.randint(0, 100000)  # Random offset to get different samples
            
            print(f"   Loading donors with offset {offset}...")
            donors_df = loader.get_donors(limit=sample_size)
            
            if len(donors_df) < sample_size:
                print(f"   ⚠️ Only {len(donors_df):,} donors available, using all of them")
            
            # Create target variable
            labels = self.create_target_variable(donors_df, target_type)
            
            print(f"   ✅ Sampled {len(donors_df):,} donors")
            print(f"   Class distribution: {np.bincount(labels)}")
            
            return donors_df
    
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
    
    def load_training_data_simple(self, 
                                 target_type: str = 'high_value',
                                 sample_size: int = 10000,
                                 test_size: float = 0.2,
                                 val_size: float = 0.2,
                                 random_state: int = 42) -> Dict[str, Any]:
        """
        Load training data using a simple approach that avoids SQL parameter limits
        
        Args:
            target_type: Type of target variable to create
            sample_size: Number of samples to use for training
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            random_state: Random seed
            
        Returns:
            Dictionary with train/val/test data
        """
        print("🚀 LOADING TRAINING DATA (SIMPLE APPROACH)")
        print("=" * 60)
        
        # Get sample donors
        donors_df = self.get_sample_donors(target_type, sample_size, random_state)
        
        # Create target variable
        labels = self.create_target_variable(donors_df, target_type)
        print(f"Target distribution: {np.bincount(labels)}")
        
        # Load only tabular data for simplicity (avoids complex SQL queries)
        print("📂 Loading tabular data only...")
        with SQLDataLoader(self.db_path) as loader:
            # Get enhanced fields for the same donors
            enhanced_df = loader.get_enhanced_fields()
            
            print(f"   ✅ Loaded tabular data:")
            print(f"      Donors: {len(donors_df):,} records")
            print(f"      Enhanced: {len(enhanced_df):,} records")
        
        # Prepare tabular features
        print("🔧 Preparing tabular features...")
        tabular_features = self.data_processor.prepare_tabular_features(
            donors_df, 
            enhanced_df,
            fit_scaler=True
        )
        
        # Convert to tensors
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        features_tensor = torch.tensor(tabular_features, dtype=torch.float32)
        
        # Split data
        print("✂️ Splitting data...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            features_tensor, labels_tensor, 
            test_size=test_size, random_state=random_state, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size, random_state=random_state, stratify=y_temp.numpy()
        )
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        print(f"📊 Data split complete:")
        print(f"   Training: {len(train_dataset):,} samples")
        print(f"   Validation: {len(val_dataset):,} samples")
        print(f"   Test: {len(test_dataset):,} samples")
        
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'feature_dim': tabular_features.shape[1]
        }
    
    def create_simple_model(self, 
                           input_dim: int,
                           num_classes: int = 2,
                           hidden_dim: int = 128,
                           dropout: float = 0.3) -> nn.Module:
        """
        Create a simple neural network model
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
            
        Returns:
            Simple neural network model
        """
        class SimpleDonorPredictor(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_classes, dropout):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, num_classes)
                )
            
            def forward(self, x):
                return self.network(x)
        
        self.model = SimpleDonorPredictor(input_dim, hidden_dim, num_classes, dropout)
        return self.model
    
    def train_epoch(self, 
                   model: nn.Module,
                   dataloader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   device: torch.device) -> Dict[str, float]:
        """
        Train model for one epoch
        """
        model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(features)
            
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
        """
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for features, labels in dataloader:
                features, labels = features.to(device), labels.to(device)
                
                # Forward pass
                logits = model(features)
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
        Train the simple model
        """
        # Set device
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        print(f"🏋️ Training on device: {device}")
        
        # Create model if not exists
        if self.model is None:
            self.model = self.create_simple_model(
                input_dim=data['feature_dim'],
                num_classes=2
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
        train_labels = [label.item() for _, label in data['train_dataset']]
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"🚀 Starting training for {num_epochs} epochs...")
        
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
                self.save_model('best_simple_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.load_model('best_simple_model.pt')
        
        # Final evaluation
        test_loader = DataLoader(data['test_dataset'], batch_size=batch_size, shuffle=False)
        test_metrics = self.validate_epoch(self.model, test_loader, criterion, device)
        
        print(f"\n📊 Final Test Results:")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   F1 Score: {test_metrics['f1']:.4f}")
        print(f"   AUC: {test_metrics['auc']:.4f}")
        
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
        print(f"💾 Model saved to {save_path}")
    
    def load_model(self, filename: str):
        """Load model from file"""
        load_path = self.model_save_dir / filename
        checkpoint = torch.load(load_path, map_location='cpu')
        
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        print(f"📂 Model loaded from {load_path}")
    
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
            print(f"📊 Training history plot saved to {save_path}")
        
        plt.show()


# Convenience function for easy usage
def train_simple_model(sample_size: int = 10000,
                      target_type: str = 'high_value',
                      num_epochs: int = 50,
                      batch_size: int = 32) -> Dict[str, Any]:
    """
    Quick function to train a simple model with sampling
    
    Args:
        sample_size: Number of samples to use for training
        target_type: Type of target variable
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Training results
    """
    print("🎯 QUICK SIMPLE MODEL TRAINING")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = SimpleSQLTrainingPipeline()
    
    # Load training data
    data = pipeline.load_training_data_simple(
        target_type=target_type,
        sample_size=sample_size
    )
    
    # Train model
    results = pipeline.train(
        data=data,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    # Plot training history
    pipeline.plot_training_history('simple_training_history.png')
    
    return results


if __name__ == "__main__":
    # Example usage
    results = train_simple_model(
        sample_size=5000,  # Use 5K samples for quick training
        target_type='high_value',
        num_epochs=30,
        batch_size=64
    )
    
    print("🎉 Training completed!")
    print(f"Final test accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Final test F1 score: {results['test_metrics']['f1']:.4f}")
