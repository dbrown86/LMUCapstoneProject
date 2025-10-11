#!/usr/bin/env python3
"""
Basic Training Example
Demonstrates how to use the training pipeline for a simple classification task
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.integrated_trainer import IntegratedTrainer
from src.training_pipeline import DataSplitter, CrossValidationFramework, ClassBalancer


def example_1_simple_pytorch_model():
    """Example 1: Train a simple PyTorch model"""
    print("=" * 80)
    print("EXAMPLE 1: SIMPLE PYTORCH MODEL")
    print("=" * 80)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        weights=[0.75, 0.25],  # Imbalanced
        random_state=42
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Define a simple model
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 2)
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Initialize model and trainer
    model = SimpleNN(input_dim=X.shape[1])
    trainer = IntegratedTrainer(model, model_type='pytorch', device='cpu', random_state=42)
    
    # Train model
    results = trainer.fit(
        X, y,
        test_size=0.2,
        val_size=0.2,
        stratify=True,
        balance_strategy='smote',
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        patience=10,
        verbose=1
    )
    
    # Plot training curves
    trainer.plot_training_curves()
    
    print(f"\nBest Validation AUC: {results['training_results']['best_val_auc']:.4f}")
    print(f"Test Accuracy: {results['test_results']['test_accuracy']:.4f}")
    print(f"Test AUC: {results['test_results']['test_roc_auc']:.4f}")
    
    return trainer, results


def example_2_sklearn_model():
    """Example 2: Train a sklearn model with the pipeline"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: SKLEARN MODEL")
    print("=" * 80)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=2000,
        n_features=15,
        n_informative=10,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize sklearn model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    trainer = IntegratedTrainer(model, model_type='sklearn', random_state=42)
    
    # Train model
    results = trainer.fit(
        X, y,
        test_size=0.2,
        val_size=0.2,
        stratify=True,
        balance_strategy='smote',
        use_cross_validation=True,
        cv_folds=5,
        verbose=1
    )
    
    print(f"\nTest Accuracy: {results['test_results']['test_accuracy']:.4f}")
    print(f"Test AUC: {results['test_results']['test_roc_auc']:.4f}")
    
    return trainer, results


def example_3_custom_data_splitting():
    """Example 3: Use custom data splitting and class balancing"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: CUSTOM DATA SPLITTING AND BALANCING")
    print("=" * 80)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=5000,
        n_features=25,
        n_informative=20,
        n_classes=2,
        weights=[0.85, 0.15],  # Highly imbalanced
        random_state=42
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Step 1: Custom data splitting
    splitter = DataSplitter(test_size=0.15, val_size=0.15, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data(X, y, stratify=True)
    
    # Step 2: Try multiple balancing strategies
    print("\nComparing balancing strategies...")
    
    strategies = ['smote', 'adasyn', 'borderline_smote']
    best_strategy = None
    best_f1 = 0
    
    for strategy in strategies:
        print(f"\n  Testing {strategy}...")
        balancer = ClassBalancer(strategy=strategy, random_state=42)
        X_train_balanced, y_train_balanced = balancer.fit_resample(X_train, y_train)
        
        # Train a quick model to evaluate
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate on validation set
        from sklearn.metrics import f1_score
        y_val_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_val_pred)
        print(f"    Validation F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_strategy = strategy
    
    print(f"\nBest balancing strategy: {best_strategy} (F1: {best_f1:.4f})")
    
    # Step 3: Train final model with best strategy
    class BestNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(64, 2)
            )
        
        def forward(self, x):
            return self.network(x)
    
    model = BestNN(input_dim=X.shape[1])
    trainer = IntegratedTrainer(model, model_type='pytorch', device='cpu', random_state=42)
    
    results = trainer.fit(
        X, y,
        balance_strategy=best_strategy,
        epochs=100,
        batch_size=64,
        patience=20,
        verbose=1
    )
    
    return trainer, results


def example_4_cross_validation():
    """Example 4: Comprehensive cross-validation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: CROSS-VALIDATION")
    print("=" * 80)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        weights=[0.75, 0.25],
        random_state=42
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize CV framework
    cv_framework = CrossValidationFramework(n_splits=5, random_state=42)
    
    # Test multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'RF Balanced': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n  Evaluating {name}...")
        cv_results = cv_framework.run_cross_validation(
            model, X, y,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            stratified=True
        )
        results[name] = cv_results
    
    # Plot CV results
    cv_framework.plot_cv_results(save_path='cv_comparison.png')
    
    return results


def example_5_model_checkpointing():
    """Example 5: Save and load model checkpoints"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: MODEL CHECKPOINTING")
    print("=" * 80)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=2000,
        n_features=15,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    
    # Define model
    class CheckpointModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, 2)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    # Train model
    model = CheckpointModel(input_dim=X.shape[1])
    trainer = IntegratedTrainer(model, model_type='pytorch', device='cpu', random_state=42)
    
    print("Training model with checkpointing...")
    results = trainer.fit(
        X, y,
        epochs=30,
        batch_size=32,
        checkpoint_dir='example_checkpoints',
        save_best_only=True,
        verbose=1
    )
    
    print(f"\nBest checkpoint saved at epoch {results['best_epoch']}")
    
    # Load checkpoint and make predictions
    print("\nLoading best checkpoint...")
    from src.training_pipeline import ModelCheckpointer
    
    checkpointer = ModelCheckpointer(checkpoint_dir='example_checkpoints')
    new_model = CheckpointModel(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(new_model.parameters())
    
    checkpoint = checkpointer.load_checkpoint(
        'example_checkpoints/best_model.pt',
        new_model,
        optimizer
    )
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint AUC: {checkpoint['metric_value']:.4f}")
    
    return trainer, checkpoint


def main():
    """Run all examples"""
    print("=" * 80)
    print("TRAINING PIPELINE EXAMPLES")
    print("=" * 80)
    print("\nThis script demonstrates various features of the training pipeline.")
    print("Each example showcases different aspects of the system.\n")
    
    # Run examples
    example_1_simple_pytorch_model()
    example_2_sklearn_model()
    example_3_custom_data_splitting()
    example_4_cross_validation()
    example_5_model_checkpointing()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()



