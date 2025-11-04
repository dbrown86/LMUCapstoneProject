# Model Training and Validation Pipeline Guide

## Overview

This comprehensive training and validation pipeline provides a production-ready system for training machine learning models with best practices including:

- **Stratified train/validation/test splits** for maintaining class distribution
- **Cross-validation framework** for robust model evaluation
- **Early stopping** to prevent overfitting
- **Model checkpointing** to save best models
- **Comprehensive performance metrics** for thorough evaluation
- **Class imbalance handling** with multiple strategies

## Architecture

The pipeline consists of three main modules:

### 1. `training_pipeline.py`
Core components for the training pipeline:
- `DataSplitter`: Stratified data splitting
- `CrossValidationFramework`: K-fold cross-validation
- `EarlyStoppingCallback`: Early stopping mechanism
- `ModelCheckpointer`: Model saving and loading
- `PerformanceMetrics`: Comprehensive metrics calculation
- `ClassBalancer`: Class imbalance handling

### 2. `integrated_trainer.py`
Unified training system that combines all components:
- `IntegratedTrainer`: End-to-end training orchestrator

### 3. Integration with existing models
- Compatible with PyTorch models (GNN, BERT-based)
- Compatible with scikit-learn models
- Easy integration with existing codebase

## Quick Start

### Basic Usage

```python
from src.integrated_trainer import IntegratedTrainer
import torch.nn as nn

# Define your model
class MyModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Prepare your data
X = ... # Your feature matrix
y = ... # Your target labels

# Initialize model
model = MyModel(input_dim=X.shape[1])

# Initialize trainer
trainer = IntegratedTrainer(
    model=model,
    model_type='pytorch',
    device='cuda',  # or 'cpu'
    random_state=42
)

# Train model
results = trainer.fit(
    X, y,
    test_size=0.2,
    val_size=0.2,
    stratify=True,
    balance_strategy='smote',
    epochs=100,
    batch_size=32,
    patience=15
)

# Make predictions
predictions, probabilities = trainer.predict(X_new)
```

## Component Details

### 1. Data Splitting

The `DataSplitter` class implements stratified train/validation/test splits:

```python
from src.training_pipeline import DataSplitter

splitter = DataSplitter(test_size=0.2, val_size=0.2, random_state=42)
X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data(X, y, stratify=True)
```

**Features:**
- Maintains class distribution across splits
- Configurable split ratios
- Reproducible with random_state
- Automatic statistics reporting

### 2. Cross-Validation Framework

The `CrossValidationFramework` supports both sklearn models and custom training functions:

```python
from src.training_pipeline import CrossValidationFramework

cv_framework = CrossValidationFramework(n_splits=5, random_state=42)

# For sklearn models
cv_results = cv_framework.run_cross_validation(
    model, X, y, 
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    stratified=True
)

# For custom models (e.g., PyTorch)
def train_fn(X_train, y_train):
    # Your training logic
    return trained_model

def predict_fn(model, X_val):
    # Your prediction logic
    return predictions, probabilities

cv_results = cv_framework.run_manual_cross_validation(
    train_fn, predict_fn, X, y, stratified=True
)
```

**Features:**
- Stratified K-fold cross-validation
- Multiple scoring metrics
- Support for custom models
- Automatic result plotting

### 3. Early Stopping

The `EarlyStoppingCallback` monitors validation metrics and stops training when performance plateaus:

```python
from src.training_pipeline import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    patience=15,          # Wait 15 epochs for improvement
    min_delta=0.0001,     # Minimum improvement threshold
    mode='max',           # 'max' for accuracy/AUC, 'min' for loss
    restore_best=True     # Restore best weights when stopping
)

# In training loop
for epoch in range(epochs):
    # ... training code ...
    val_metric = evaluate(model, val_data)
    
    if early_stopping(val_metric, model, epoch):
        break  # Stop training
```

**Features:**
- Configurable patience
- Minimum delta threshold
- Automatic best model restoration
- Supports both minimization and maximization metrics

### 4. Model Checkpointing

The `ModelCheckpointer` saves model checkpoints during training:

```python
from src.training_pipeline import ModelCheckpointer

checkpointer = ModelCheckpointer(
    checkpoint_dir='checkpoints',
    mode='max',              # 'max' for accuracy/AUC, 'min' for loss
    save_best_only=True,     # Only save when model improves
    save_frequency=10        # Save every N epochs (if not save_best_only)
)

# In training loop
for epoch in range(epochs):
    # ... training code ...
    val_metric = evaluate(model, val_data)
    
    checkpointer.save_checkpoint(
        model, optimizer, epoch, val_metric, 
        metrics_dict={'accuracy': acc, 'f1': f1},
        model_name='my_model'
    )

# Load checkpoint
checkpoint = checkpointer.load_checkpoint('checkpoints/best_my_model.pt', model, optimizer)
```

**Features:**
- Save best models only or periodic saves
- Full checkpoint with optimizer state
- Automatic checkpoint management
- Easy loading and resuming

### 5. Performance Metrics

The `PerformanceMetrics` class calculates comprehensive classification metrics:

```python
from src.training_pipeline import PerformanceMetrics

metrics_calc = PerformanceMetrics()

# Calculate metrics
metrics = metrics_calc.calculate_metrics(
    y_true, y_pred, y_pred_proba,
    prefix='test_'
)

# Print metrics
metrics_calc.print_metrics(metrics)

# Plot visualizations
metrics_calc.plot_confusion_matrix(y_true, y_pred, class_names=['No Intent', 'Intent'])
metrics_calc.plot_roc_curve(y_true, y_pred_proba)
metrics_calc.plot_precision_recall_curve(y_true, y_pred_proba)
```

**Metrics Calculated:**
- Basic: Accuracy, Precision, Recall, F1
- Probability-based: AUC-ROC, Average Precision
- Advanced: Matthews Correlation Coefficient, Cohen's Kappa
- Confusion Matrix: Specificity, Negative Predictive Value

### 6. Class Imbalance Handling

The `ClassBalancer` implements multiple strategies for handling imbalanced datasets:

```python
from src.training_pipeline import ClassBalancer

# Available strategies
strategies = [
    'smote',              # Synthetic Minority Over-sampling
    'adasyn',             # Adaptive Synthetic Sampling
    'borderline_smote',   # Borderline SMOTE
    'smote_tomek',        # SMOTE + Tomek Links
    'smote_enn',          # SMOTE + Edited Nearest Neighbors
    'undersample',        # Random Under-sampling
    'class_weight'        # Cost-sensitive learning (no resampling)
]

balancer = ClassBalancer(strategy='smote', random_state=42)

# Resample data
X_resampled, y_resampled = balancer.fit_resample(X_train, y_train)

# Or compute class weights for loss function
class_weights = balancer.compute_class_weights(y_train)
```

**Features:**
- Multiple balancing strategies
- Automatic distribution reporting
- Compatible with PyTorch and sklearn
- Class weight computation for cost-sensitive learning

## Complete Example: Donor Legacy Intent Prediction

Here's a complete example integrating with your donor dataset:

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.integrated_trainer import IntegratedTrainer

# 1. Load your donor data
donors_df = pd.read_csv('synthetic_donor_dataset/donors.csv')
contact_reports_df = pd.read_csv('synthetic_donor_dataset/contact_reports.csv')
relationships_df = pd.read_csv('synthetic_donor_dataset/relationships.csv')

# 2. Prepare features
# Extract numeric features
numeric_features = [
    'Lifetime_Giving', 'Last_Gift', 'Engagement_Score',
    'Consecutive_Yr_Giving_Count', 'Total_Yr_Giving_Count',
    'Estimated_Age', 'Legacy_Intent_Probability'
]

X = donors_df[numeric_features].fillna(0).values
y = donors_df['Legacy_Intent_Binary'].values

# 3. Define model
class DonorClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 4. Initialize and train
model = DonorClassifier(input_dim=len(numeric_features))

trainer = IntegratedTrainer(
    model=model,
    model_type='pytorch',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    random_state=42
)

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
    # Cross-validation
    use_cross_validation=False,
    cv_folds=5,
    # Verbosity
    verbose=1
)

# 5. Plot training curves
trainer.plot_training_curves(save_path='donor_training_curves.png')

# 6. Make predictions on new donors
new_donors = pd.read_csv('new_donors.csv')
X_new = new_donors[numeric_features].fillna(0).values
predictions, probabilities = trainer.predict(X_new)

# 7. Export results
new_donors['Predicted_Legacy_Intent'] = predictions
new_donors['Legacy_Probability'] = probabilities
new_donors.to_csv('donor_predictions.csv', index=False)

print(f"Training completed!")
print(f"Best AUC: {results['training_results']['best_val_auc']:.4f}")
print(f"Test Accuracy: {results['test_results']['test_accuracy']:.4f}")
```

## Integration with Existing GNN Models

To integrate with your existing GNN models:

```python
from src.gnn_models.gnn_models import GraphSAGE, DonorGNNTrainer
from src.integrated_trainer import IntegratedTrainer

# Load graph data
graph_data = ...  # Your PyTorch Geometric Data object

# Instead of using DonorGNNTrainer, use IntegratedTrainer
model = GraphSAGE(input_dim=graph_data.x.shape[1], hidden_dim=64, output_dim=2)

trainer = IntegratedTrainer(model, model_type='pytorch', device='cuda')

# Extract features and labels for training
X = graph_data.x.cpu().numpy()
y = graph_data.y.cpu().numpy()

# Train with full pipeline
results = trainer.fit(
    X, y,
    balance_strategy='smote',
    epochs=200,
    patience=20,
    ...
)
```

## Best Practices

### 1. Data Preparation
- Always check for missing values before training
- Normalize/standardize features for neural networks
- Use stratified splitting for imbalanced datasets

### 2. Class Imbalance
- Start with SMOTE for moderate imbalance (2:1 to 10:1)
- Use combination methods (SMOTE+ENN) for severe imbalance (>10:1)
- Always use class weights with neural networks

### 3. Training
- Start with a learning rate of 0.001 and adjust based on performance
- Use early stopping with patience=15-20 for medium datasets
- Enable checkpointing to save best models
- Monitor both training and validation metrics

### 4. Evaluation
- Always evaluate on a held-out test set
- Use cross-validation for small datasets (<5000 samples)
- Focus on AUC-ROC and F1 for imbalanced datasets
- Plot confusion matrix and ROC curve

### 5. Hyperparameter Tuning
- Start with default parameters
- Tune learning rate first
- Then tune model architecture
- Finally tune regularization (dropout, weight decay)

## Troubleshooting

### Model overfitting
- Increase dropout rate
- Add more regularization (weight_decay)
- Reduce model complexity
- Get more training data or use data augmentation

### Model underfitting
- Increase model capacity (more layers/neurons)
- Train for more epochs
- Decrease regularization
- Check if features are informative

### Class imbalance not helping
- Try different balancing strategies
- Adjust class weights manually
- Use threshold optimization
- Consider ensemble methods

### Training too slow
- Increase batch size (if memory allows)
- Use GPU if available
- Reduce model complexity
- Use learning rate scheduling

## API Reference

See inline documentation in the source code for detailed API reference:
- `src/training_pipeline.py`
- `src/integrated_trainer.py`

## Examples

Additional examples are available in:
- `examples/basic_training.py` - Basic training example
- `examples/gnn_integration.py` - GNN model integration
- `examples/multimodal_training.py` - Multimodal ensemble training
- `examples/cross_validation.py` - Cross-validation example

## Support

For issues or questions:
1. Check this documentation
2. Review example scripts
3. Check inline code documentation
4. Review test cases

## Version History

### v1.0.0 (Current)
- Initial release
- Complete training pipeline
- Cross-validation framework
- Early stopping and checkpointing
- Comprehensive metrics
- Class imbalance handling













