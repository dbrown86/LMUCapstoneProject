#!/usr/bin/env python3
"""
Simple Donor Training Example - No Prompts
Trains a donor legacy intent prediction model using the new training pipeline
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from src.integrated_trainer import IntegratedTrainer


class DonorLegacyClassifier(nn.Module):
    """Neural network for donor legacy intent prediction"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def main():
    print("=" * 80)
    print("DONOR LEGACY INTENT PREDICTION - SIMPLE TRAINING")
    print("=" * 80)
    
    # Load donor data
    print("\n1. Loading donor data...")
    data_path = Path('synthetic_donor_dataset')
    
    donors_df = pd.read_csv(data_path / 'donors.csv')
    print(f"   Loaded {len(donors_df):,} donors")
    
    # Prepare features
    print("\n2. Preparing features...")
    numeric_features = [
        'Lifetime_Giving',
        'Last_Gift',
        'Consecutive_Yr_Giving_Count',
        'Total_Yr_Giving_Count',
        'Engagement_Score',
        'Legacy_Intent_Probability',
        'Estimated_Age'
    ]
    
    # Convert Rating to numeric
    rating_map = {
        'A': 10, 'B': 9, 'C': 8, 'D': 7, 'E': 6, 'F': 5,
        'G': 4, 'H': 3, 'I': 2, 'J': 1, 'K': 0.5, 'L': 0.1,
        'M': 0.05, 'N': 0.01, 'O': 0.005, 'P': 0.001
    }
    donors_df['Rating_Numeric'] = donors_df['Rating'].map(rating_map).fillna(1.0)
    numeric_features.append('Rating_Numeric')
    
    # Additional features
    donors_df['Has_Family'] = donors_df['Family_ID'].notna().astype(int)
    donors_df['Giving_Consistency'] = donors_df['Consecutive_Yr_Giving_Count'] / (donors_df['Total_Yr_Giving_Count'] + 1)
    donors_df['Avg_Gift_Size'] = donors_df['Lifetime_Giving'] / (donors_df['Total_Yr_Giving_Count'] + 1)
    
    numeric_features.extend(['Has_Family', 'Giving_Consistency', 'Avg_Gift_Size'])
    
    X = donors_df[numeric_features].fillna(0).values
    y = donors_df['Legacy_Intent_Binary'].values
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]:,}")
    print(f"   Class distribution: {np.bincount(y)}")
    print(f"   Imbalance ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.2f}:1")
    
    # Create model
    print("\n3. Creating model...")
    model = DonorLegacyClassifier(input_dim=X.shape[1], hidden_dims=[128, 64, 32], dropout=0.3)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    print("\n4. Initializing trainer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    trainer = IntegratedTrainer(
        model=model,
        model_type='pytorch',
        device=device,
        random_state=42
    )
    
    # Train model
    print("\n5. Training model...")
    print("   This may take a few minutes...")
    
    results = trainer.fit(
        X, y,
        # Data splitting
        test_size=0.2,
        val_size=0.2,
        stratify=True,
        # Class balancing
        balance_strategy='smote',
        use_class_weights=True,
        # Training parameters
        epochs=100,
        batch_size=64,
        learning_rate=0.001,
        optimizer_name='adam',
        # Early stopping
        patience=15,
        min_delta=0.0001,
        # Checkpointing
        checkpoint_dir='donor_model_checkpoints',
        save_best_only=True,
        # Verbosity
        verbose=1
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    
    test_results = results['test_results']
    
    print("\nTest Set Performance:")
    print(f"  Accuracy:  {test_results['test_accuracy']:.4f} ({test_results['test_accuracy']*100:.2f}%)")
    print(f"  Precision: {test_results['test_precision']:.4f}")
    print(f"  Recall:    {test_results['test_recall']:.4f}")
    print(f"  F1 Score:  {test_results['test_f1']:.4f}")
    print(f"  AUC-ROC:   {test_results['test_roc_auc']:.4f}")
    print(f"  MCC:       {test_results['test_mcc']:.4f}")
    
    print("\nTraining Info:")
    print(f"  Best epoch: {results['best_epoch']}")
    print(f"  Total epochs: {results['training_results']['total_epochs']}")
    print(f"  Best val AUC: {results['training_results']['best_val_auc']:.4f}")
    
    print("\nGenerated Files:")
    print("  [OK] donor_model_checkpoints/best_model.pt - Best model weights")
    print("  [OK] donor_model_checkpoints/training_summary.json - Training statistics")
    
    # Plot training curves
    print("\n6. Generating training curves...")
    try:
        trainer.plot_training_curves(save_path='donor_training_curves.png')
        print("  [OK] donor_training_curves.png - Training visualization")
    except Exception as e:
        print(f"  [WARN] Could not generate plot: {e}")
    
    print("\n" + "=" * 80)
    print("SUCCESS! Model trained and ready for use.")
    print("=" * 80)
    
    print("\nNext steps:")
    print("  1. Review the training curves in donor_training_curves.png")
    print("  2. Load the best model: torch.load('models/donor_model_checkpoints/best_model.pt')")
    print("  3. Use trainer.predict(X_new) to make predictions on new donors")
    
    return trainer, results


if __name__ == "__main__":
    main()

