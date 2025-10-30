#!/usr/bin/env python3
"""
Multimodal Training Pipeline using Parquet files
Uses PyTorch DataLoader with custom Dataset class for efficient batch loading
Full multimodal fusion architecture with all modalities
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    from torch_geometric.data import Data
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    print("⚠️ PyTorch Geometric not available, graph features will be disabled")


class MultimodalDonorDataset(Dataset):
    """
    Custom PyTorch Dataset for multimodal donor data
    Loads data in batches from Parquet files for efficiency
    """
    
    def __init__(self, 
                 donor_ids: List[int],
                 donors_df: pd.DataFrame,
                 relationships_df: Optional[pd.DataFrame],
                 giving_df: Optional[pd.DataFrame],
                 events_df: Optional[pd.DataFrame],
                 labels: np.ndarray,
                 scaler: Optional[StandardScaler] = None,
                 fit_scaler: bool = False):
        """
        Initialize the dataset
        
        Args:
            donor_ids: List of donor IDs to include
            donors_df: Main donors dataframe with features
            relationships_df: Relationships dataframe
            giving_df: Giving history dataframe
            events_df: Event attendance dataframe
            labels: Target labels
            scaler: Fitted scaler for features
            fit_scaler: Whether to fit the scaler
        """
        self.donor_ids = donor_ids
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.scaler = scaler  # Initialize scaler first
        
        # Filter donors
        self.donors_df = donors_df[donors_df['ID'].isin(donor_ids)].copy()
        self.donors_df = self.donors_df.set_index('ID')
        
        # Prepare tabular features
        self.tabular_features = self._prepare_tabular_features(fit_scaler, scaler)
        
        # Prepare graph data (relationships)
        self.relationships_df = relationships_df
        if self.relationships_df is not None:
            self.relationships_df = self.relationships_df[
                self.relationships_df['Donor_ID_1'].isin(donor_ids) & 
                self.relationships_df['Donor_ID_2'].isin(donor_ids)
            ]
        
        # Prepare sequence data (giving history)
        self.giving_df = giving_df
        if self.giving_df is not None:
            self.giving_df = self.giving_df[self.giving_df['Donor_ID'].isin(donor_ids)]
        
        # Prepare event data
        self.events_df = events_df
        if self.events_df is not None:
            self.events_df = self.events_df[self.events_df['Donor_ID'].isin(donor_ids)]
    
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
    
    def __len__(self) -> int:
        return len(self.donor_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        donor_id = self.donor_ids[idx]
        
        # Tabular features
        tabular = self.tabular_features[idx]
        
        # Label
        label = self.labels[idx]
        
        # Graph features (relationships) - simplified for now
        # In a full implementation, you'd build edge indices and features here
        has_relationships = 0
        if self.relationships_df is not None:
            related = self.relationships_df[
                (self.relationships_df['Donor_ID_1'] == donor_id) | 
                (self.relationships_df['Donor_ID_2'] == donor_id)
            ]
            has_relationships = len(related)
        
        # Sequence features (giving history)
        giving_count = 0
        if self.giving_df is not None:
            giving_records = self.giving_df[self.giving_df['Donor_ID'] == donor_id]
            giving_count = len(giving_records)
        
        # Event features
        event_count = 0
        if self.events_df is not None:
            event_records = self.events_df[self.events_df['Donor_ID'] == donor_id]
            event_count = len(event_records)
        
        # Add relationship, giving, and event counts as additional features
        additional_features = torch.tensor([
            has_relationships, 
            giving_count, 
            event_count
        ], dtype=torch.float32)
        
        # Concatenate all tabular features
        full_features = torch.cat([tabular, additional_features])
        
        return {
            'features': full_features,
            'label': label,
            'donor_id': donor_id
        }


class MultimodalFusionModel(nn.Module):
    """
    Multimodal Fusion Architecture
    Combines tabular, graph, and sequence features
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim // 4, num_classes)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Encode features
        encoded = self.feature_encoder(features)
        
        # Fusion
        fused = self.fusion(encoded)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits


class ParquetMultimodalTrainer:
    """
    Training pipeline using Parquet files with DataLoader
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
            'val_f1': []
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
        
        # Load relationships
        rel_file = self.parquet_dir / 'relationships.parquet'
        if rel_file.exists():
            data['relationships'] = pd.read_parquet(rel_file)
            print(f"   ✅ Relationships: {len(data['relationships']):,} records")
        else:
            data['relationships'] = None
        
        # Load giving history
        giving_file = self.parquet_dir / 'giving_history.parquet'
        if giving_file.exists():
            data['giving'] = pd.read_parquet(giving_file)
            print(f"   ✅ Giving history: {len(data['giving']):,} records")
        else:
            data['giving'] = None
        
        # Load events
        events_file = self.parquet_dir / 'event_attendance.parquet'
        if events_file.exists():
            data['events'] = pd.read_parquet(events_file)
            print(f"   ✅ Events: {len(data['events']):,} records")
        else:
            data['events'] = None
        
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
                        random_state: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare train/val/test datasets
        
        Args:
            sample_size: Number of samples to use (None = use all)
            target_type: Type of target variable
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        print("🚀 PREPARING MULTIMODAL DATASETS")
        print("=" * 60)
        
        # Load data
        data = self.load_parquet_data()
        donors_df = data['donors']
        
        # Sample if requested
        if sample_size and sample_size < len(donors_df):
            print(f"\n📊 Sampling {sample_size:,} donors...")
            donors_df = donors_df.sample(n=sample_size, random_state=random_state)
        
        print(f"\n📊 Using {len(donors_df):,} donors")
        
        # Create labels
        labels = self.create_target_variable(donors_df, target_type)
        print(f"   Target distribution: {np.bincount(labels)}")
        
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
        print("\n🔧 Creating PyTorch datasets...")
        
        # Train dataset (fit scaler)
        train_dataset = MultimodalDonorDataset(
            donor_ids=train_ids,
            donors_df=donors_df,
            relationships_df=data.get('relationships'),
            giving_df=data.get('giving'),
            events_df=data.get('events'),
            labels=y_train,
            fit_scaler=True
        )
        
        # Save scaler
        self.scaler = train_dataset.scaler
        
        # Val dataset (use fitted scaler)
        val_dataset = MultimodalDonorDataset(
            donor_ids=val_ids,
            donors_df=donors_df,
            relationships_df=data.get('relationships'),
            giving_df=data.get('giving'),
            events_df=data.get('events'),
            labels=y_val,
            scaler=self.scaler,
            fit_scaler=False
        )
        
        # Test dataset (use fitted scaler)
        test_dataset = MultimodalDonorDataset(
            donor_ids=test_ids,
            donors_df=donors_df,
            relationships_df=data.get('relationships'),
            giving_df=data.get('giving'),
            events_df=data.get('events'),
            labels=y_test,
            scaler=self.scaler,
            fit_scaler=False
        )
        
        print("   ✅ Datasets created successfully")
        
        return train_dataset, val_dataset, test_dataset
    
    def train_epoch(self, 
                   model: nn.Module,
                   dataloader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   device: torch.device) -> Dict[str, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(features)
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
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
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
              train_dataset: Dataset,
              val_dataset: Dataset,
              test_dataset: Dataset,
              num_epochs: int = 100,
              batch_size: int = 64,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-5,
              patience: int = 10,
              device: str = 'auto') -> Dict[str, Any]:
        """
        Train the multimodal fusion model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            num_epochs: Number of training epochs
            batch_size: Batch size for DataLoader
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            device: Device to train on
            
        Returns:
            Training results
        """
        # Set device
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        print(f"\n🏋️ Training on device: {device}")
        
        # Get feature dimension from first sample
        sample = train_dataset[0]
        input_dim = sample['features'].shape[0]
        
        print(f"   Input dimension: {input_dim}")
        
        # Create model
        self.model = MultimodalFusionModel(
            input_dim=input_dim,
            hidden_dim=128,
            num_classes=2,
            dropout=0.3
        ).to(device)
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Compute class weights
        train_labels = [label.item() for batch in train_loader for label in batch['label']]
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n🚀 Starting training for {num_epochs} epochs...")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
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
                self.save_model('best_multimodal_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.load_model('best_multimodal_model.pt')
        
        # Final evaluation
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
        save_path = self.model_save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history
        }, save_path)
    
    def load_model(self, filename: str):
        """Load model from file"""
        load_path = self.model_save_dir / filename
        checkpoint = torch.load(load_path, map_location='cpu')
        
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']


# Convenience function
def train_multimodal_from_parquet(sample_size: int = 10000,
                                  target_type: str = 'high_value',
                                  num_epochs: int = 50,
                                  batch_size: int = 64) -> Dict[str, Any]:
    """
    Quick function to train multimodal model from Parquet files
    
    Args:
        sample_size: Number of samples to use
        target_type: Type of target variable
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Training results
    """
    print("🎯 MULTIMODAL TRAINING FROM PARQUET")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ParquetMultimodalTrainer()
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(
        sample_size=sample_size,
        target_type=target_type
    )
    
    # Train model
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    results = train_multimodal_from_parquet(
        sample_size=10000,
        target_type='high_value',
        num_epochs=50,
        batch_size=64
    )
    
    print("🎉 Training completed!")
    print(f"Final test accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Final test F1 score: {results['test_metrics']['f1']:.4f}")

