#!/usr/bin/env python3
"""
Comprehensive Model Training and Validation Pipeline
Implements stratified splits, cross-validation, early stopping, checkpointing, and comprehensive metrics
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, StratifiedShuffleSplit, 
    cross_val_score, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, matthews_corrcoef, cohen_kappa_score
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
warnings.filterwarnings('ignore')

# Try to import imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTEENN, SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Class balancing will be limited.")


class DataSplitter:
    """
    Comprehensive data splitting strategy handler
    Implements stratified train/validation/test splits with various options
    """
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
        """
        Initialize DataSplitter
        
        Args:
            test_size: Proportion of data for test set (0.0 to 1.0)
            val_size: Proportion of training data for validation set (0.0 to 1.0)
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        print(f"DataSplitter initialized: test={test_size:.1%}, val={val_size:.1%}")
    
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        stratify: bool = True,
        return_indices: bool = False
    ) -> Tuple:
        """
        Split data into train/validation/test sets with stratification
        
        Args:
            X: Feature matrix
            y: Target labels
            stratify: Whether to use stratified splitting
            return_indices: If True, also return the indices of splits
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
            If return_indices=True, also returns (train_idx, val_idx, test_idx)
        """
        print("=" * 70)
        print("DATA SPLITTING")
        print("=" * 70)
        
        # First split: separate test set
        stratify_param = y if stratify else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, 
            stratify=stratify_param
        )
        
        # Second split: separate validation set from training
        val_size_adjusted = self.val_size / (1 - self.test_size)
        stratify_param = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=stratify_param
        )
        
        # Print split statistics
        self._print_split_stats(y_train, y_val, y_test)
        
        if return_indices:
            # Reconstruct indices (this is approximate if using actual indices)
            n_total = len(X)
            n_train = len(X_train)
            n_val = len(X_val)
            n_test = len(X_test)
            train_idx = np.arange(n_train)
            val_idx = np.arange(n_train, n_train + n_val)
            test_idx = np.arange(n_train + n_val, n_total)
            
            return X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _print_split_stats(self, y_train, y_val, y_test):
        """Print statistics about the data splits"""
        total_samples = len(y_train) + len(y_val) + len(y_test)
        
        print(f"\nTotal samples: {total_samples:,}")
        print(f"Training samples: {len(y_train):,} ({len(y_train)/total_samples*100:.1f}%)")
        print(f"Validation samples: {len(y_val):,} ({len(y_val)/total_samples*100:.1f}%)")
        print(f"Test samples: {len(y_test):,} ({len(y_test)/total_samples*100:.1f}%)")
        
        # Class distribution in each split
        print("\nClass distribution:")
        for split_name, split_y in [("Training", y_train), ("Validation", y_val), ("Test", y_test)]:
            unique, counts = np.unique(split_y, return_counts=True)
            class_dist = dict(zip(unique, counts))
            class_percentages = {k: v/len(split_y)*100 for k, v in class_dist.items()}
            print(f"  {split_name}: {class_dist} -> {class_percentages}")


class CrossValidationFramework:
    """
    Comprehensive cross-validation framework
    Supports stratified K-fold and custom CV strategies
    """
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize CrossValidationFramework
        
        Args:
            n_splits: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_results = []
        
        print(f"CrossValidationFramework initialized: {n_splits}-fold CV")
    
    def run_cross_validation(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray,
        scoring: List[str] = None,
        stratified: bool = True
    ) -> Dict:
        """
        Run stratified cross-validation
        
        Args:
            model: Model to evaluate (must have fit and predict methods)
            X: Feature matrix
            y: Target labels
            scoring: List of scoring metrics
            stratified: Whether to use stratified folds
        
        Returns:
            Dictionary containing CV results
        """
        print("=" * 70)
        print(f"RUNNING {self.n_splits}-FOLD CROSS-VALIDATION")
        print("=" * 70)
        
        if scoring is None:
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Create cross-validation splitter
        if stratified:
            cv_splitter = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
        else:
            from sklearn.model_selection import KFold
            cv_splitter = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
        
        # Run cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv_splitter, scoring=scoring,
            return_train_score=True, n_jobs=-1
        )
        
        # Store results
        self.cv_results.append(cv_results)
        
        # Print results
        self._print_cv_results(cv_results, scoring)
        
        return cv_results
    
    def run_manual_cross_validation(
        self,
        train_fn: Callable,
        predict_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        stratified: bool = True
    ) -> Dict:
        """
        Run manual cross-validation for custom models (e.g., PyTorch)
        
        Args:
            train_fn: Function that trains the model, signature: train_fn(X_train, y_train) -> model
            predict_fn: Function that makes predictions, signature: predict_fn(model, X_val) -> predictions
            X: Feature matrix
            y: Target labels
            stratified: Whether to use stratified folds
        
        Returns:
            Dictionary containing CV results
        """
        print("=" * 70)
        print(f"RUNNING MANUAL {self.n_splits}-FOLD CROSS-VALIDATION")
        print("=" * 70)
        
        # Create cross-validation splitter
        if stratified:
            cv_splitter = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
        else:
            from sklearn.model_selection import KFold
            cv_splitter = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
        
        fold_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        # Run cross-validation manually
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y), 1):
            print(f"\nFold {fold_idx}/{self.n_splits}...")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model
            model = train_fn(X_train_fold, y_train_fold)
            
            # Make predictions
            y_pred, y_pred_proba = predict_fn(model, X_val_fold)
            
            # Calculate metrics
            fold_results['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            fold_results['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
            fold_results['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
            fold_results['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
            fold_results['roc_auc'].append(roc_auc_score(y_val_fold, y_pred_proba))
        
        # Store results
        self.cv_results.append(fold_results)
        
        # Print results
        self._print_cv_results_manual(fold_results)
        
        return fold_results
    
    def _print_cv_results(self, cv_results, scoring):
        """Print cross-validation results"""
        print("\nCross-Validation Results:")
        print("-" * 70)
        
        for metric in scoring:
            train_key = f'train_{metric}'
            test_key = f'test_{metric}'
            
            if train_key in cv_results and test_key in cv_results:
                train_scores = cv_results[train_key]
                test_scores = cv_results[test_key]
                
                print(f"{metric.upper()}:")
                print(f"  Training:   {np.mean(train_scores):.4f} (+/- {np.std(train_scores):.4f})")
                print(f"  Validation: {np.mean(test_scores):.4f} (+/- {np.std(test_scores):.4f})")
    
    def _print_cv_results_manual(self, fold_results):
        """Print manual cross-validation results"""
        print("\nCross-Validation Results:")
        print("-" * 70)
        
        for metric, scores in fold_results.items():
            print(f"{metric.upper()}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    def plot_cv_results(self, save_path: Optional[str] = None):
        """Plot cross-validation results"""
        if not self.cv_results:
            print("No CV results to plot.")
            return
        
        latest_results = self.cv_results[-1]
        
        # Extract metrics
        metrics = []
        mean_scores = []
        std_scores = []
        
        for key in latest_results.keys():
            if key.startswith('test_'):
                metric_name = key.replace('test_', '')
                metrics.append(metric_name)
                mean_scores.append(np.mean(latest_results[key]))
                std_scores.append(np.std(latest_results[key]))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(metrics))
        ax.bar(x_pos, mean_scores, yerr=std_scores, align='center', alpha=0.7, 
               capsize=10, color='steelblue')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_title(f'{self.n_splits}-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CV plot saved to {save_path}")
        
        plt.show()


class EarlyStoppingCallback:
    """
    Early stopping callback to prevent overfitting
    Monitors validation metric and stops training when it stops improving
    """
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.0001, 
        mode: str = 'min',
        restore_best: bool = True
    ):
        """
        Initialize EarlyStopping
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/AUC
            restore_best: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        
        print(f"EarlyStopping initialized: patience={patience}, mode={mode}")
    
    def __call__(self, current_value: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop
        
        Args:
            current_value: Current epoch's monitored metric value
            model: Model being trained
            epoch: Current epoch number
        
        Returns:
            True if training should stop, False otherwise
        """
        # Check if there's improvement
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:  # mode == 'max'
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
            
            # Save best model state
            if self.restore_best and hasattr(model, 'state_dict'):
                self.best_model_state = model.state_dict().copy() if hasattr(model, 'state_dict') else None
            
            return False
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best value: {self.best_value:.6f} at epoch {self.best_epoch}")
                
                # Restore best model weights
                if self.restore_best and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                    print("Restored best model weights")
                
                return True
            
            return False


class ModelCheckpointer:
    """
    Model checkpointing system
    Saves model checkpoints based on performance metrics
    """
    
    def __init__(
        self, 
        checkpoint_dir: str = 'checkpoints',
        mode: str = 'max',
        save_best_only: bool = True,
        save_frequency: int = 10
    ):
        """
        Initialize ModelCheckpointer
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            mode: 'min' for loss, 'max' for accuracy/AUC
            save_best_only: Whether to save only the best model
            save_frequency: Save checkpoint every N epochs (if not save_best_only)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_checkpoint_path = None
        
        print(f"ModelCheckpointer initialized: {checkpoint_dir}")
    
    def save_checkpoint(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric_value: float,
        metrics_dict: Dict[str, float],
        model_name: str = 'model'
    ) -> Optional[str]:
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metric_value: Value of monitored metric
            metrics_dict: Dictionary of all metrics
            model_name: Name for the checkpoint file
        
        Returns:
            Path to saved checkpoint or None
        """
        # Check if we should save
        should_save = False
        
        if self.save_best_only:
            if self.mode == 'min':
                is_best = metric_value < self.best_value
            else:
                is_best = metric_value > self.best_value
            
            if is_best:
                self.best_value = metric_value
                should_save = True
        else:
            # Save every N epochs
            should_save = (epoch % self.save_frequency == 0)
        
        if not should_save:
            return None
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric_value': metric_value,
            'metrics': metrics_dict,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate filename
        if self.save_best_only:
            filename = f'best_{model_name}.pt'
        else:
            filename = f'{model_name}_epoch_{epoch}.pt'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        if self.save_best_only:
            self.best_checkpoint_path = checkpoint_path
            print(f"Saved best model checkpoint: {checkpoint_path}")
        else:
            print(f"Saved checkpoint: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self, 
        checkpoint_path: str, 
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
        
        Returns:
            Dictionary containing checkpoint information
        """
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Metric value: {checkpoint['metric_value']:.4f}")
        
        return checkpoint


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator
    Computes and tracks various classification metrics
    """
    
    def __init__(self):
        """Initialize PerformanceMetrics"""
        self.metrics_history = []
        
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            prefix: Prefix for metric names (e.g., 'train_', 'val_')
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred)
        metrics[f'{prefix}precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f'{prefix}recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f'{prefix}f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            metrics[f'{prefix}roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics[f'{prefix}avg_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Advanced metrics
        metrics[f'{prefix}mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics[f'{prefix}cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics[f'{prefix}specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics[f'{prefix}npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in a formatted way"""
        print("\nPerformance Metrics:")
        print("-" * 70)
        
        # Group metrics by type
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1']
        prob_metrics = ['roc_auc', 'avg_precision']
        advanced_metrics = ['mcc', 'cohen_kappa', 'specificity', 'npv']
        
        # Remove prefix for printing
        clean_metrics = {k.split('_', 1)[-1] if '_' in k else k: v for k, v in metrics.items()}
        
        print("Basic Metrics:")
        for metric in basic_metrics:
            if metric in clean_metrics:
                print(f"  {metric.upper():15s}: {clean_metrics[metric]:.4f}")
        
        print("\nProbability-Based Metrics:")
        for metric in prob_metrics:
            if metric in clean_metrics:
                print(f"  {metric.upper():15s}: {clean_metrics[metric]:.4f}")
        
        print("\nAdvanced Metrics:")
        for metric in advanced_metrics:
            if metric in clean_metrics:
                print(f"  {metric.upper():15s}: {clean_metrics[metric]:.4f}")
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names if class_names else ['Class 0', 'Class 1'],
                   yticklabels=class_names if class_names else ['Class 0', 'Class 1'])
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2, 
                label=f'PR Curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()


class ClassBalancer:
    """
    Comprehensive class balancing handler
    Implements various techniques for handling class imbalance
    """
    
    def __init__(self, strategy: str = 'smote', random_state: int = 42):
        """
        Initialize ClassBalancer
        
        Args:
            strategy: Balancing strategy ('smote', 'adasyn', 'borderline_smote', 
                     'smote_tomek', 'smote_enn', 'undersample', 'class_weight')
            random_state: Random seed
        """
        self.strategy = strategy
        self.random_state = random_state
        self.sampler = None
        
        if not IMBLEARN_AVAILABLE and strategy != 'class_weight':
            print(f"Warning: {strategy} not available. Falling back to 'class_weight'")
            self.strategy = 'class_weight'
        
        print(f"ClassBalancer initialized: strategy={self.strategy}")
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply class balancing to the dataset
        
        Args:
            X: Feature matrix
            y: Target labels
        
        Returns:
            Resampled (X, y)
        """
        print(f"\nApplying {self.strategy} for class balancing...")
        
        # Print original distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Original distribution: {dict(zip(unique, counts))}")
        
        if self.strategy == 'smote' and IMBLEARN_AVAILABLE:
            self.sampler = SMOTE(random_state=self.random_state)
        elif self.strategy == 'adasyn' and IMBLEARN_AVAILABLE:
            self.sampler = ADASYN(random_state=self.random_state)
        elif self.strategy == 'borderline_smote' and IMBLEARN_AVAILABLE:
            self.sampler = BorderlineSMOTE(random_state=self.random_state)
        elif self.strategy == 'smote_tomek' and IMBLEARN_AVAILABLE:
            self.sampler = SMOTETomek(random_state=self.random_state)
        elif self.strategy == 'smote_enn' and IMBLEARN_AVAILABLE:
            self.sampler = SMOTEENN(random_state=self.random_state)
        elif self.strategy == 'undersample' and IMBLEARN_AVAILABLE:
            self.sampler = RandomUnderSampler(random_state=self.random_state)
        elif self.strategy == 'class_weight':
            # For class weighting, we don't resample
            print("Using class weights - no resampling needed")
            return X, y
        else:
            print(f"Unknown strategy: {self.strategy}. Returning original data.")
            return X, y
        
        # Resample
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        
        # Print new distribution
        unique, counts = np.unique(y_resampled, return_counts=True)
        print(f"Resampled distribution: {dict(zip(unique, counts))}")
        
        return X_resampled, y_resampled
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for cost-sensitive learning
        
        Args:
            y: Target labels
        
        Returns:
            Dictionary mapping class labels to weights
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        print(f"Computed class weights: {class_weights}")
        
        return class_weights


def demo_training_pipeline():
    """Demonstrate the training pipeline components"""
    print("=" * 80)
    print("TRAINING PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    # Generate sample imbalanced data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.8, 0.2],
        random_state=42
    )
    
    print(f"\nGenerated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # 1. Data Splitting
    splitter = DataSplitter(test_size=0.2, val_size=0.2, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data(X, y, stratify=True)
    
    # 2. Class Balancing
    balancer = ClassBalancer(strategy='smote', random_state=42)
    X_train_balanced, y_train_balanced = balancer.fit_resample(X_train, y_train)
    
    # 3. Train a simple model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_balanced, y_train_balanced)
    
    # 4. Calculate metrics
    metrics_calc = PerformanceMetrics()
    
    # Validation metrics
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_metrics = metrics_calc.calculate_metrics(y_val, y_val_pred, y_val_proba, prefix='val_')
    metrics_calc.print_metrics(val_metrics)
    
    # 5. Cross-validation
    cv_framework = CrossValidationFramework(n_splits=5, random_state=42)
    cv_results = cv_framework.run_cross_validation(
        model, X_train_balanced, y_train_balanced, stratified=True
    )
    
    # 6. Plot results
    metrics_calc.plot_confusion_matrix(y_val, y_val_pred, class_names=['No Intent', 'Intent'])
    metrics_calc.plot_roc_curve(y_val, y_val_proba)
    metrics_calc.plot_precision_recall_curve(y_val, y_val_proba)
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    demo_training_pipeline()



















