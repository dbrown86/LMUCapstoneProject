#!/usr/bin/env python3
"""
Donor Legacy Intent Prediction with Training Pipeline
Complete example integrating the new training pipeline with your existing donor dataset
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
from pathlib import Path

# Import training pipeline
from src.integrated_trainer import IntegratedTrainer
from src.training_pipeline import DataSplitter, CrossValidationFramework, PerformanceMetrics

# Import existing project modules (optional - will skip if not available)
try:
    from src.enhanced_feature_engineering import AdvancedFeatureEngineering
except ImportError:
    AdvancedFeatureEngineering = None
    print("Note: Advanced feature engineering not available, using basic features")


class DonorLegacyClassifier(nn.Module):
    """
    Neural network for donor legacy intent prediction
    Uses tabular features from donor data
    """
    
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
        
        # Output layer (2 classes: no intent, intent)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_donor_data(data_dir='synthetic_donor_dataset'):
    """Load donor datasets"""
    print("=" * 80)
    print("LOADING DONOR DATA")
    print("=" * 80)
    
    data_path = Path(data_dir)
    
    # Load datasets
    donors_df = pd.read_csv(data_path / 'donors.csv')
    relationships_df = pd.read_csv(data_path / 'relationships.csv')
    contact_reports_df = pd.read_csv(data_path / 'contact_reports.csv')
    giving_history_df = pd.read_csv(data_path / 'giving_history.csv')
    
    print(f"\nLoaded datasets:")
    print(f"  Donors: {len(donors_df):,} records")
    print(f"  Relationships: {len(relationships_df):,} records")
    print(f"  Contact Reports: {len(contact_reports_df):,} records")
    print(f"  Giving History: {len(giving_history_df):,} records")
    
    return donors_df, relationships_df, contact_reports_df, giving_history_df


def prepare_features(donors_df, relationships_df=None, giving_history_df=None):
    """Prepare features for training"""
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    
    # Select numeric features
    numeric_features = [
        'Lifetime_Giving',
        'Last_Gift',
        'Consecutive_Yr_Giving_Count',
        'Total_Yr_Giving_Count',
        'Engagement_Score',
        'Legacy_Intent_Probability',
        'Estimated_Age'
    ]
    
    # Handle Rating (categorical -> numeric)
    rating_map = {
        'A': 10, 'B': 9, 'C': 8, 'D': 7, 'E': 6, 'F': 5, 
        'G': 4, 'H': 3, 'I': 2, 'J': 1, 'K': 0.5, 'L': 0.1, 
        'M': 0.05, 'N': 0.01, 'O': 0.005, 'P': 0.001
    }
    
    donors_df['Rating_Numeric'] = donors_df['Rating'].map(rating_map).fillna(1.0)
    numeric_features.append('Rating_Numeric')
    
    # Create additional features
    # Family features
    donors_df['Has_Family'] = donors_df['Family_ID'].notna().astype(int)
    numeric_features.append('Has_Family')
    
    # Giving consistency
    donors_df['Giving_Consistency'] = donors_df['Consecutive_Yr_Giving_Count'] / (donors_df['Total_Yr_Giving_Count'] + 1)
    numeric_features.append('Giving_Consistency')
    
    # Average gift size
    donors_df['Avg_Gift_Size'] = donors_df['Lifetime_Giving'] / (donors_df['Total_Yr_Giving_Count'] + 1)
    numeric_features.append('Avg_Gift_Size')
    
    # Engagement ratio
    donors_df['Engagement_Ratio'] = donors_df['Engagement_Score'] * donors_df['Rating_Numeric']
    numeric_features.append('Engagement_Ratio')
    
    # Extract features and target
    X = donors_df[numeric_features].fillna(0).values
    y = donors_df['Legacy_Intent_Binary'].values
    
    print(f"\nPrepared features:")
    print(f"  Feature count: {X.shape[1]}")
    print(f"  Sample count: {X.shape[0]:,}")
    print(f"  Target distribution: {np.bincount(y)}")
    print(f"  Class imbalance ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.2f}:1")
    
    return X, y, numeric_features


def train_donor_model(X, y, feature_names, model_config=None):
    """Train donor legacy intent prediction model"""
    print("\n" + "=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)
    
    if model_config is None:
        model_config = {
            'hidden_dims': [128, 64, 32],
            'dropout': 0.3
        }
    
    # Initialize model
    model = DonorLegacyClassifier(
        input_dim=X.shape[1],
        hidden_dims=model_config['hidden_dims'],
        dropout=model_config['dropout']
    )
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    trainer = IntegratedTrainer(
        model=model,
        model_type='pytorch',
        device=device,
        random_state=42
    )
    
    # Train model with comprehensive pipeline
    results = trainer.fit(
        X, y,
        # Data splitting
        test_size=0.2,
        val_size=0.2,
        stratify=True,
        # Class balancing
        balance_strategy='smote',  # Try: 'smote', 'adasyn', 'borderline_smote'
        use_class_weights=True,
        # Training parameters
        epochs=150,
        batch_size=64,
        learning_rate=0.001,
        optimizer_name='adam',
        # Early stopping
        patience=20,
        min_delta=0.0001,
        # Checkpointing
        checkpoint_dir='donor_model_checkpoints',
        save_best_only=True,
        # Cross-validation
        use_cross_validation=False,  # Set to True for smaller datasets
        cv_folds=5,
        # Verbosity
        verbose=1
    )
    
    return trainer, results


def evaluate_and_visualize(trainer, results):
    """Evaluate model and create visualizations"""
    print("\n" + "=" * 80)
    print("MODEL EVALUATION AND VISUALIZATION")
    print("=" * 80)
    
    # Plot training curves
    print("\nGenerating training curves...")
    trainer.plot_training_curves(save_path='donor_training_curves.png')
    
    # Extract test set performance
    test_results = results['test_results']
    
    print("\nTest Set Performance:")
    print(f"  Accuracy:  {test_results['test_accuracy']:.4f} ({test_results['test_accuracy']*100:.2f}%)")
    print(f"  Precision: {test_results['test_precision']:.4f}")
    print(f"  Recall:    {test_results['test_recall']:.4f}")
    print(f"  F1 Score:  {test_results['test_f1']:.4f}")
    print(f"  AUC-ROC:   {test_results['test_roc_auc']:.4f}")
    print(f"  MCC:       {test_results['test_mcc']:.4f}")
    
    # Business metrics
    print("\nBusiness Impact Estimation:")
    
    # Assuming:
    # - Average legacy gift value: $50,000
    # - Cost per false positive (wasted outreach): $200
    # - Cost per false negative (missed opportunity): $5,000
    
    # We'd need the actual predictions to calculate this properly
    # This is a simplified estimation based on precision/recall
    
    precision = test_results['test_precision']
    recall = test_results['test_recall']
    
    # Estimate true positives per 100 actual legacy donors
    estimated_tp = recall * 100
    estimated_fn = (1 - recall) * 100
    
    # Estimate false positives (depends on negative samples, assume 400 non-legacy per 100 legacy)
    estimated_fp = (1 - precision) / precision * estimated_tp if precision > 0 else 0
    
    print(f"  Per 100 actual legacy donors:")
    print(f"    Identified: {estimated_tp:.0f} donors")
    print(f"    Missed: {estimated_fn:.0f} donors")
    print(f"    False alarms: {estimated_fp:.0f} donors")
    
    potential_value = estimated_tp * 50000
    missed_value = estimated_fn * 50000
    wasted_cost = estimated_fp * 200
    
    print(f"  Estimated impact:")
    print(f"    Potential legacy value captured: ${potential_value:,.0f}")
    print(f"    Missed legacy value: ${missed_value:,.0f}")
    print(f"    Wasted outreach cost: ${wasted_cost:,.0f}")
    print(f"    Net value: ${potential_value - wasted_cost:,.0f}")
    
    return test_results


def make_predictions_on_new_donors(trainer, feature_names, new_donors_path=None):
    """Make predictions on new donors"""
    print("\n" + "=" * 80)
    print("MAKING PREDICTIONS ON NEW DONORS")
    print("=" * 80)
    
    if new_donors_path is None:
        print("\nNo new donors file provided. Skipping predictions.")
        return None
    
    # Load new donors
    new_donors_df = pd.read_csv(new_donors_path)
    print(f"\nLoaded {len(new_donors_df):,} new donors")
    
    # Prepare features (use same feature engineering as training)
    X_new = new_donors_df[feature_names].fillna(0).values
    
    # Make predictions
    predictions, probabilities = trainer.predict(X_new)
    
    # Add predictions to dataframe
    new_donors_df['Predicted_Legacy_Intent'] = predictions
    new_donors_df['Legacy_Probability'] = probabilities
    new_donors_df['Confidence'] = np.maximum(probabilities, 1 - probabilities)
    
    # Sort by probability
    new_donors_df = new_donors_df.sort_values('Legacy_Probability', ascending=False)
    
    # Save results
    output_path = 'donor_predictions.csv'
    new_donors_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    # Show top prospects
    print("\nTop 10 Legacy Prospects:")
    print("-" * 80)
    top_10 = new_donors_df.head(10)[['ID', 'First_Name', 'Last_Name', 'Legacy_Probability', 'Confidence']]
    print(top_10.to_string(index=False))
    
    # Statistics
    print(f"\nPrediction Statistics:")
    print(f"  Total donors evaluated: {len(new_donors_df):,}")
    print(f"  Predicted legacy intent: {predictions.sum():,} ({predictions.sum()/len(predictions)*100:.1f}%)")
    print(f"  High confidence (>0.8): {(new_donors_df['Confidence'] > 0.8).sum():,}")
    print(f"  Medium confidence (0.6-0.8): {((new_donors_df['Confidence'] >= 0.6) & (new_donors_df['Confidence'] <= 0.8)).sum():,}")
    print(f"  Low confidence (<0.6): {(new_donors_df['Confidence'] < 0.6).sum():,}")
    
    return new_donors_df


def hyperparameter_search(X, y):
    """Perform hyperparameter search (optional advanced feature)"""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH")
    print("=" * 80)
    
    print("\nTesting different configurations...")
    
    configurations = [
        {'hidden_dims': [64, 32], 'dropout': 0.2, 'lr': 0.001, 'balance': 'smote'},
        {'hidden_dims': [128, 64, 32], 'dropout': 0.3, 'lr': 0.001, 'balance': 'smote'},
        {'hidden_dims': [256, 128, 64], 'dropout': 0.4, 'lr': 0.0005, 'balance': 'adasyn'},
        {'hidden_dims': [128, 64, 32], 'dropout': 0.3, 'lr': 0.001, 'balance': 'borderline_smote'}
    ]
    
    best_config = None
    best_auc = 0
    results_summary = []
    
    for i, config in enumerate(configurations, 1):
        print(f"\n  Configuration {i}/{len(configurations)}:")
        print(f"    Hidden dims: {config['hidden_dims']}")
        print(f"    Dropout: {config['dropout']}")
        print(f"    Learning rate: {config['lr']}")
        print(f"    Balance strategy: {config['balance']}")
        
        # Initialize model
        model = DonorLegacyClassifier(
            input_dim=X.shape[1],
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout']
        )
        
        trainer = IntegratedTrainer(model, model_type='pytorch', device='cpu', random_state=42)
        
        # Quick training (fewer epochs for search)
        results = trainer.fit(
            X, y,
            balance_strategy=config['balance'],
            epochs=50,  # Reduced for faster search
            batch_size=64,
            learning_rate=config['lr'],
            patience=10,
            verbose=0  # Suppress detailed output
        )
        
        auc = results['test_results']['test_roc_auc']
        f1 = results['test_results']['test_f1']
        
        print(f"    Results: AUC={auc:.4f}, F1={f1:.4f}")
        
        results_summary.append({
            'config': i,
            'auc': auc,
            'f1': f1,
            **config
        })
        
        if auc > best_auc:
            best_auc = auc
            best_config = config
    
    print(f"\nBest configuration (AUC={best_auc:.4f}):")
    print(f"  Hidden dims: {best_config['hidden_dims']}")
    print(f"  Dropout: {best_config['dropout']}")
    print(f"  Learning rate: {best_config['lr']}")
    print(f"  Balance strategy: {best_config['balance']}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv('hyperparameter_search_results.csv', index=False)
    print("\nFull results saved to hyperparameter_search_results.csv")
    
    return best_config, results_summary


def main():
    """Main execution function"""
    print("=" * 80)
    print("DONOR LEGACY INTENT PREDICTION WITH TRAINING PIPELINE")
    print("=" * 80)
    print("\nThis script demonstrates how to use the new training pipeline")
    print("with your existing donor dataset for legacy intent prediction.\n")
    
    # Step 1: Load data
    donors_df, relationships_df, contact_reports_df, giving_history_df = load_donor_data()
    
    # Step 2: Prepare features
    X, y, feature_names = prepare_features(donors_df, relationships_df, giving_history_df)
    
    # Step 3: Optional - Hyperparameter search
    print("\n" + "=" * 80)
    
    # Check if running interactively
    try:
        import sys
        if sys.stdin.isatty():
            user_input = input("Run hyperparameter search? (y/n, default=n): ").strip().lower()
        else:
            user_input = 'n'
            print("Non-interactive mode: Skipping hyperparameter search")
    except:
        user_input = 'n'
        print("Skipping hyperparameter search. Using default configuration.")
    
    if user_input == 'y':
        best_config, search_results = hyperparameter_search(X, y)
        model_config = {
            'hidden_dims': best_config['hidden_dims'],
            'dropout': best_config['dropout']
        }
    else:
        print("Using default configuration.")
        model_config = None
    
    # Step 4: Train final model
    trainer, results = train_donor_model(X, y, feature_names, model_config)
    
    # Step 5: Evaluate and visualize
    test_results = evaluate_and_visualize(trainer, results)
    
    # Step 6: Make predictions on new donors (if available)
    # Uncomment the following line if you have new donors to predict on:
    # predictions_df = make_predictions_on_new_donors(trainer, feature_names, 'new_donors.csv')
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    print("\nGenerated files:")
    print("  - donor_model_checkpoints/: Saved model checkpoints")
    print("  - donor_training_curves.png: Training visualization")
    print("  - donor_model_checkpoints/training_summary.json: Results summary")
    
    print("\nNext steps:")
    print("  1. Review the training curves and test set performance")
    print("  2. Load the best checkpoint for production use")
    print("  3. Use the model to make predictions on new donors")
    print("  4. Integrate with your existing multimodal pipeline")
    
    return trainer, results


if __name__ == "__main__":
    main()



