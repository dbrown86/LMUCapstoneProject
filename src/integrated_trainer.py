#!/usr/bin/env python3
"""
Integrated Model Trainer
Combines all training pipeline components into a unified training system
Supports both PyTorch and scikit-learn models
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our pipeline components
from src.training_pipeline import (
    DataSplitter, CrossValidationFramework, EarlyStoppingCallback,
    ModelCheckpointer, PerformanceMetrics, ClassBalancer
)


class IntegratedTrainer:
    """
    Unified training system that integrates all pipeline components
    Supports end-to-end model training with best practices
    """
    
    def __init__(
        self,
        model: Union[nn.Module, Any],
        model_type: str = 'pytorch',
        device: str = 'cpu',
        random_state: int = 42
    ):
        """
        Initialize IntegratedTrainer
        
        Args:
            model: Model to train (PyTorch nn.Module or sklearn model)
            model_type: 'pytorch' or 'sklearn'
            device: Device for PyTorch models ('cpu' or 'cuda')
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.model_type = model_type
        self.device = device
        self.random_state = random_state
        
        # Training components
        self.data_splitter = None
        self.class_balancer = None
        self.early_stopping = None
        self.checkpointer = None
        self.metrics_calculator = PerformanceMetrics()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Move PyTorch model to device
        if model_type == 'pytorch' and isinstance(model, nn.Module):
            self.model = self.model.to(device)
        
        print(f"IntegratedTrainer initialized: model_type={model_type}, device={device}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        # Data splitting parameters
        test_size: float = 0.2,
        val_size: float = 0.2,
        stratify: bool = True,
        # Class balancing parameters
        balance_strategy: Optional[str] = 'smote',
        use_class_weights: bool = True,
        # Training parameters
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        optimizer_name: str = 'adam',
        loss_fn: Optional[nn.Module] = None,
        # Early stopping parameters
        patience: int = 15,
        min_delta: float = 0.0001,
        # Checkpointing parameters
        checkpoint_dir: str = 'checkpoints',
        save_best_only: bool = True,
        # Cross-validation parameters
        use_cross_validation: bool = False,
        cv_folds: int = 5,
        # Other parameters
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train model with comprehensive pipeline
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion for test set
            val_size: Proportion for validation set
            stratify: Use stratified splitting
            balance_strategy: Strategy for class balancing (None to disable)
            use_class_weights: Use class weights in loss function
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            optimizer_name: Optimizer ('adam', 'sgd', 'rmsprop')
            loss_fn: Loss function for PyTorch models
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            checkpoint_dir: Directory for checkpoints
            save_best_only: Save only best model
            use_cross_validation: Run cross-validation before training
            cv_folds: Number of CV folds
            verbose: Verbosity level (0, 1, 2)
        
        Returns:
            Dictionary containing training results
        """
        print("=" * 80)
        print("INTEGRATED MODEL TRAINING PIPELINE")
        print("=" * 80)
        
        # Step 1: Data Splitting
        if verbose > 0:
            print("\n" + "=" * 70)
            print("STEP 1: DATA SPLITTING")
            print("=" * 70)
        
        self.data_splitter = DataSplitter(test_size=test_size, val_size=val_size, random_state=self.random_state)
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_splitter.split_data(
            X, y, stratify=stratify
        )
        
        # Step 2: Class Balancing
        if balance_strategy is not None:
            if verbose > 0:
                print("\n" + "=" * 70)
                print("STEP 2: CLASS BALANCING")
                print("=" * 70)
            
            self.class_balancer = ClassBalancer(strategy=balance_strategy, random_state=self.random_state)
            X_train, y_train = self.class_balancer.fit_resample(X_train, y_train)
        
        # Compute class weights if requested
        class_weights = None
        if use_class_weights and self.model_type == 'pytorch':
            if self.class_balancer is None:
                self.class_balancer = ClassBalancer(strategy='class_weight', random_state=self.random_state)
            class_weights_dict = self.class_balancer.compute_class_weights(y_train)
            class_weights = torch.tensor(list(class_weights_dict.values()), dtype=torch.float).to(self.device)
        
        # Step 3: Cross-Validation (optional)
        if use_cross_validation:
            if verbose > 0:
                print("\n" + "=" * 70)
                print("STEP 3: CROSS-VALIDATION")
                print("=" * 70)
            
            cv_framework = CrossValidationFramework(n_splits=cv_folds, random_state=self.random_state)
            
            if self.model_type == 'sklearn':
                cv_results = cv_framework.run_cross_validation(
                    self.model, X_train, y_train, stratified=stratify
                )
            else:
                # For PyTorch, we need custom CV
                print("Cross-validation for PyTorch models requires manual implementation.")
                print("Skipping CV and proceeding to training.")
        
        # Step 4: Setup Training Components
        if verbose > 0:
            print("\n" + "=" * 70)
            print("STEP 4: TRAINING SETUP")
            print("=" * 70)
        
        # Early stopping
        self.early_stopping = EarlyStoppingCallback(
            patience=patience, min_delta=min_delta, mode='max', restore_best=True
        )
        
        # Checkpointing
        self.checkpointer = ModelCheckpointer(
            checkpoint_dir=checkpoint_dir, mode='max', save_best_only=save_best_only
        )
        
        # Step 5: Model Training
        if verbose > 0:
            print("\n" + "=" * 70)
            print("STEP 5: MODEL TRAINING")
            print("=" * 70)
        
        if self.model_type == 'pytorch':
            training_results = self._train_pytorch_model(
                X_train, y_train, X_val, y_val,
                epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                optimizer_name=optimizer_name, loss_fn=loss_fn, class_weights=class_weights,
                verbose=verbose
            )
        else:
            training_results = self._train_sklearn_model(X_train, y_train, verbose=verbose)
        
        # Step 6: Final Evaluation
        if verbose > 0:
            print("\n" + "=" * 70)
            print("STEP 6: FINAL EVALUATION")
            print("=" * 70)
        
        # Evaluate on test set
        test_results = self._evaluate(X_test, y_test, prefix='test_', verbose=verbose)
        
        # Combine results
        final_results = {
            'training_results': training_results,
            'test_results': test_results,
            'history': self.history,
            'best_epoch': self.early_stopping.best_epoch if hasattr(self.early_stopping, 'best_epoch') else None,
            'splits': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            }
        }
        
        # Save results summary
        self._save_results_summary(final_results, checkpoint_dir)
        
        if verbose > 0:
            print("\n" + "=" * 80)
            print("TRAINING PIPELINE COMPLETED")
            print("=" * 80)
            self._print_final_summary(final_results)
        
        return final_results
    
    def _train_pytorch_model(
        self,
        X_train, y_train, X_val, y_val,
        epochs, batch_size, learning_rate, optimizer_name, loss_fn, class_weights,
        verbose
    ):
        """Train PyTorch model"""
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float),
            torch.tensor(y_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Setup loss function
        if loss_fn is None:
            if class_weights is not None:
                loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()
        
        # Training loop
        print(f"\nTraining for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader, optimizer, loss_fn)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader, loss_fn)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Print progress
            if verbose > 0 and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val AUC: {val_metrics.get('val_roc_auc', 0):.4f}")
            
            # Check early stopping
            val_auc = val_metrics.get('val_roc_auc', val_metrics.get('val_f1', 0))
            if self.early_stopping(val_auc, self.model, epoch):
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save checkpoint
            self.checkpointer.save_checkpoint(
                self.model, optimizer, epoch, val_auc, val_metrics, model_name='model'
            )
        
        return {
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'best_val_auc': self.early_stopping.best_value,
            'total_epochs': len(self.history['train_loss'])
        }
    
    def _train_epoch(self, train_loader, optimizer, loss_fn):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            
            # Calculate loss
            loss = loss_fn(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs), prefix='train_'
        )
        
        return avg_loss, metrics
    
    def _validate_epoch(self, val_loader, loss_fn):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = loss_fn(outputs, batch_y)
                
                # Track metrics
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs), prefix='val_'
        )
        
        return avg_loss, metrics
    
    def _train_sklearn_model(self, X_train, y_train, verbose):
        """Train sklearn model"""
        print("Training sklearn model...")
        
        # Check if model supports class_weight
        if hasattr(self.model, 'class_weight'):
            self.model.set_params(class_weight='balanced')
        
        # Train model
        self.model.fit(X_train, y_train)
        
        print("Training completed.")
        
        return {'model_type': 'sklearn', 'trained': True}
    
    def _evaluate(self, X, y, prefix='', verbose=1):
        """Evaluate model"""
        if self.model_type == 'pytorch':
            self.model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
            
            y_pred = preds.cpu().numpy()
            y_pred_proba = probs[:, 1].cpu().numpy()
        else:
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y, y_pred, y_pred_proba, prefix=prefix)
        
        if verbose > 0:
            self.metrics_calculator.print_metrics(metrics)
        
        return metrics
    
    def predict(self, X):
        """Make predictions"""
        if self.model_type == 'pytorch':
            self.model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
            
            return preds.cpu().numpy(), probs[:, 1].cpu().numpy()
        else:
            preds = self.model.predict(X)
            probs = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
            return preds, probs
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves"""
        if not self.history['train_loss']:
            print("No training history to plot.")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        axes[0].plot(epochs_range, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs_range, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # AUC curves
        train_aucs = [m.get('train_roc_auc', 0) for m in self.history['train_metrics']]
        val_aucs = [m.get('val_roc_auc', 0) for m in self.history['val_metrics']]
        
        axes[1].plot(epochs_range, train_aucs, 'b-', label='Training AUC', linewidth=2)
        axes[1].plot(epochs_range, val_aucs, 'r-', label='Validation AUC', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('AUC-ROC', fontsize=12)
        axes[1].set_title('Training and Validation AUC', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def _save_results_summary(self, results, checkpoint_dir):
        """Save results summary to JSON"""
        summary = {
            'best_epoch': results.get('best_epoch'),
            'test_metrics': {k: float(v) for k, v in results['test_results'].items()},
            'splits': results['splits']
        }
        
        summary_path = Path(checkpoint_dir) / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults summary saved to {summary_path}")
    
    def _print_final_summary(self, results):
        """Print final results summary"""
        print("\nFINAL RESULTS SUMMARY")
        print("=" * 80)
        
        if results['best_epoch'] is not None:
            print(f"Best Epoch: {results['best_epoch']}")
        
        print(f"\nDataset Splits:")
        print(f"  Training:   {results['splits']['train_size']:,} samples")
        print(f"  Validation: {results['splits']['val_size']:,} samples")
        print(f"  Test:       {results['splits']['test_size']:,} samples")
        
        print(f"\nTest Set Performance:")
        test_metrics = results['test_results']
        
        # Extract key metrics (remove prefix)
        key_metrics = {
            'Accuracy': test_metrics.get('test_accuracy', 0),
            'Precision': test_metrics.get('test_precision', 0),
            'Recall': test_metrics.get('test_recall', 0),
            'F1': test_metrics.get('test_f1', 0),
            'AUC-ROC': test_metrics.get('test_roc_auc', 0)
        }
        
        for metric, value in key_metrics.items():
            print(f"  {metric:12s}: {value:.4f}")
        
        print("=" * 80)


def demo_integrated_trainer():
    """Demonstrate the integrated trainer"""
    print("=" * 80)
    print("INTEGRATED TRAINER DEMONSTRATION")
    print("=" * 80)
    
    # Generate sample data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        weights=[0.8, 0.2],
        random_state=42
    )
    
    print(f"\nGenerated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Define a simple PyTorch model
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 2)
            self.dropout = nn.Dropout(0.3)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # Initialize model
    model = SimpleClassifier(input_dim=X.shape[1])
    
    # Initialize trainer
    trainer = IntegratedTrainer(model, model_type='pytorch', device='cpu', random_state=42)
    
    # Train model
    results = trainer.fit(
        X, y,
        test_size=0.2,
        val_size=0.2,
        stratify=True,
        balance_strategy='smote',
        use_class_weights=True,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        optimizer_name='adam',
        patience=10,
        checkpoint_dir='demo_checkpoints',
        save_best_only=True,
        use_cross_validation=False,
        verbose=1
    )
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Plot additional metrics
    X_test = results['test_results']
    # Note: We'd need to store test data to plot confusion matrix
    # trainer.metrics_calculator.plot_confusion_matrix(y_test, y_pred)
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED")
    print("=" * 80)
    
    return trainer, results


if __name__ == "__main__":
    demo_integrated_trainer()

