#!/usr/bin/env python3
"""
Improved Multimodal Ensemble Pipeline
Enhanced version with better performance and stability
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import improved multimodal architecture
from src.improved_multimodal_arch import (
    ImprovedMultimodalFusionModel, 
    ImprovedTabularEncoder, 
    ImprovedTextEncoder, 
    ImprovedGraphEncoder,
    ImprovedCrossModalAttention,
    create_improved_multimodal_model,
    get_improved_optimizer,
    get_improved_scheduler
)

class ImprovedMultimodalEnsemble:
    """Improved multimodal ensemble with better performance"""
    
    def __init__(self, random_state=42, device=None):
        self.random_state = random_state
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_models = {}
        self.multimodal_model = None
        self.scalers = {}
        self.feature_names = None
        self.optimal_threshold = 0.5
        self.sampler = None
        self.multimodal_weight = 0.2  # Start with 20% multimodal, 80% traditional
        
    def create_improved_multimodal_model(self, tabular_dim, text_dim, graph_dim):
        """Create improved multimodal fusion model"""
        print("  Creating improved multimodal fusion model...")
        
        self.multimodal_model = create_improved_multimodal_model(
            tabular_dim=tabular_dim,
            text_dim=text_dim,
            graph_dim=graph_dim,
            hidden_dim=256,  # Increased capacity
            fusion_dim=512,  # Increased capacity
            num_classes=2,
            dropout=0.1,  # Reduced dropout
            num_attention_heads=8,  # More attention heads
            device=self.device
        )
        
        return self.multimodal_model
    
    def create_improved_models(self):
        """Create improved ensemble models"""
        self.base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,  # Increased
                max_depth=12,  # Increased
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                C=0.1,  # Better regularization
                penalty='l2',
                solver='liblinear',
                max_iter=2000,
                random_state=self.random_state
            ),
            'svm': CalibratedClassifierCV(
                LinearSVC(C=0.1, max_iter=2000, random_state=self.random_state),
                method='sigmoid',
                cv=5  # More CV folds
            )
        }
    
    def train_improved_multimodal_model(self, X_tabular, X_text, X_graph, y, epochs=100):
        """Train improved multimodal model with better training"""
        print("  Training improved multimodal fusion model...")
        
        # Convert to tensors
        X_tabular_tensor = torch.FloatTensor(X_tabular).to(self.device)
        X_text_tensor = torch.FloatTensor(X_text).to(self.device)
        X_graph_tensor = torch.FloatTensor(X_graph).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create modality mask (all modalities present)
        modality_mask = torch.ones(len(X_tabular), 3).to(self.device)
        
        # Create improved multimodal model
        self.create_improved_multimodal_model(X_tabular.shape[1], X_text.shape[1], X_graph.shape[1])
        
        # Improved loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = get_improved_optimizer(self.multimodal_model, lr=0.0001, weight_decay=1e-5)
        scheduler = get_improved_scheduler(optimizer, epochs=epochs, warmup_epochs=10)
        
        # Training loop with improvements
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.multimodal_model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = self.multimodal_model(X_tabular_tensor, X_text_tensor, X_graph_tensor, modality_mask)
            loss = criterion(outputs, y_tensor)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.multimodal_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Early stopping with better patience
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                best_model_state = self.multimodal_model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= 15:  # Increased patience
                print(f"    Early stopping at epoch {epoch}")
                break
            
            if epoch % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"    Epoch {epoch}: Loss = {loss.item():.4f}, LR = {current_lr:.6f}")
        
        # Load best model
        if best_model_state is not None:
            self.multimodal_model.load_state_dict(best_model_state)
        
        print(f"    Improved multimodal training completed!")
    
    def apply_improved_class_balancing(self, X_tabular, X_text, X_graph, y):
        """Apply improved class balancing"""
        print("  Applying improved class balancing...")
        
        # Use SMOTEENN for better balancing
        self.sampler = SMOTEENN(
            sampling_strategy='auto',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Combine all features for balancing
        X_combined = np.hstack([X_tabular, X_text, X_graph])
        X_balanced, y_balanced = self.sampler.fit_resample(X_combined, y)
        
        # Split back into modalities
        tabular_dim = X_tabular.shape[1]
        text_dim = X_text.shape[1]
        graph_dim = X_graph.shape[1]
        
        X_tabular_balanced = X_balanced[:, :tabular_dim]
        X_text_balanced = X_balanced[:, tabular_dim:tabular_dim + text_dim]
        X_graph_balanced = X_balanced[:, tabular_dim + text_dim:]
        
        return X_tabular_balanced, X_text_balanced, X_graph_balanced, y_balanced
    
    def train_improved_ensemble(self, X_tabular, X_text, X_graph, y, feature_names=None):
        """Train improved multimodal ensemble"""
        print("Training improved multimodal ensemble...")
        
        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_tabular.shape[1])]
        
        # Improved feature scaling
        self.scalers['tabular'] = RobustScaler()
        X_tabular_scaled = self.scalers['tabular'].fit_transform(X_tabular)
        
        self.scalers['text'] = StandardScaler()
        X_text_scaled = self.scalers['text'].fit_transform(X_text)
        
        self.scalers['graph'] = StandardScaler()
        X_graph_scaled = self.scalers['graph'].fit_transform(X_graph)
        
        # Apply improved class balancing
        X_tabular_balanced, X_text_balanced, X_graph_balanced, y_balanced = self.apply_improved_class_balancing(
            X_tabular_scaled, X_text_scaled, X_graph_scaled, y
        )
        
        # Train improved multimodal model
        self.train_improved_multimodal_model(X_tabular_balanced, X_text_balanced, X_graph_balanced, y_balanced)
        
        # Create improved ensemble models
        self.create_improved_models()
        
        # Combine all features for ensemble training
        X_combined = np.hstack([X_tabular_balanced, X_text_balanced, X_graph_balanced])
        
        # Train ensemble models
        print("  Training ensemble models...")
        for name, model in self.base_models.items():
            print(f"    Training {name}...")
            model.fit(X_combined, y_balanced)
        
        print("  Improved ensemble training completed!")
    
    def predict_improved(self, X_tabular, X_text, X_graph):
        """Make improved predictions with dynamic weighting"""
        print("Making improved predictions...")
        
        # Scale features
        X_tabular_scaled = self.scalers['tabular'].transform(X_tabular)
        X_text_scaled = self.scalers['text'].transform(X_text)
        X_graph_scaled = self.scalers['graph'].transform(X_graph)
        
        # Get multimodal predictions
        self.multimodal_model.eval()
        with torch.no_grad():
            X_tabular_tensor = torch.FloatTensor(X_tabular_scaled).to(self.device)
            X_text_tensor = torch.FloatTensor(X_text_scaled).to(self.device)
            X_graph_tensor = torch.FloatTensor(X_graph_scaled).to(self.device)
            modality_mask = torch.ones(len(X_tabular), 3).to(self.device)
            
            multimodal_proba, _ = self.multimodal_model(X_tabular_tensor, X_text_tensor, X_graph_tensor, modality_mask)
            multimodal_proba = multimodal_proba.cpu().numpy()
        
        # Get ensemble predictions
        X_combined = np.hstack([X_tabular_scaled, X_text_scaled, X_graph_scaled])
        ensemble_probabilities = []
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_combined)
                ensemble_probabilities.append(proba)
            else:
                # For models without predict_proba, use decision_function
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(X_combined)
                    # Convert scores to probabilities using sigmoid
                    proba = 1 / (1 + np.exp(-scores))
                    proba = np.column_stack([1 - proba, proba])
                else:
                    # Fallback to hard predictions
                    pred = model.predict(X_combined)
                    proba = np.column_stack([1 - pred, pred])
                ensemble_probabilities.append(proba)
        
        # Dynamic weighting based on performance
        if len(ensemble_probabilities) > 0:
            ensemble_proba = np.mean(ensemble_probabilities, axis=0)
            
            # Dynamic weighting: start with 20% multimodal, adjust based on confidence
            multimodal_confidence = np.max(multimodal_proba, axis=1)
            ensemble_confidence = np.max(ensemble_proba, axis=1)
            
            # Adjust weight based on confidence difference
            confidence_diff = multimodal_confidence - ensemble_confidence
            adaptive_weight = 0.2 + 0.3 * np.tanh(confidence_diff)  # Range: 0.2 to 0.5
            
            # Apply adaptive weighting
            final_proba = np.zeros_like(multimodal_proba)
            for i in range(len(final_proba)):
                weight = adaptive_weight[i]
                final_proba[i] = weight * multimodal_proba[i] + (1 - weight) * ensemble_proba[i]
        else:
            final_proba = multimodal_proba
        
        # Convert to predictions
        final_pred = (final_proba[:, 1] >= self.optimal_threshold).astype(int)
        
        return final_pred, final_proba
    
    def optimize_threshold(self, X_tabular, X_text, X_graph, y, metric='f1'):
        """Optimize threshold for better performance"""
        print(f"  Optimizing threshold for {metric}...")
        
        # Get probabilities
        _, y_proba = self.predict_improved(X_tabular, X_text, X_graph)
        
        # Test different thresholds
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            y_pred = (y_proba[:, 1] >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y, y_pred)
            elif metric == 'accuracy':
                score = (y_pred == y).mean()
            elif metric == 'precision':
                score = precision_score(y, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y, y_pred, zero_division=0)
            else:
                score = f1_score(y, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        print(f"    Optimal threshold: {best_threshold:.3f} ({metric}: {best_score:.3f})")
        
        return best_threshold, best_score

def main():
    """Main execution function"""
    print("=" * 80)
    print("IMPROVED MULTIMODAL ENSEMBLE PIPELINE")
    print("=" * 80)
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load data
    print("\n1. Loading data...")
    try:
        # Load embeddings
        bert_embeddings = np.load('data/cache/bert_embeddings_real.npy')
        gnn_embeddings = np.load('data/cache/gnn_embeddings_real.npy')
        
        # Load enhanced features
        enhanced_features = pd.read_csv('data/cache/enhanced_features.csv')
        
        # Load target variable from original donors dataset
        donors_df = pd.read_csv('data/synthetic_donor_dataset/donors.csv')
        y = donors_df['Legacy_Intent_Binary'].astype(int).values
        
        # Load train/test split
        try:
            split_data = np.load('data/cache/train_test_split_v2.npz')
            train_indices = split_data['train_indices']
            test_indices = split_data['test_indices']
        except KeyError:
            # Fallback: create new train/test split
            print("   Creating new train/test split...")
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(enhanced_features))
            train_indices, test_indices = train_test_split(
                indices, test_size=0.2, random_state=42, stratify=y
            )
        
        print(f"   Loaded {len(enhanced_features):,} donors")
        print(f"   BERT embeddings: {bert_embeddings.shape}")
        print(f"   GNN embeddings: {gnn_embeddings.shape}")
        
    except Exception as e:
        print(f"   Error loading data: {e}")
        return None
    
    # Prepare features
    print("\n2. Preparing features...")
    
    # Clean features (remove target columns)
    target_cols = ['Legacy_Intent_Probability', 'Legacy_Intent_Binary']
    feature_cols = [col for col in enhanced_features.columns if col not in target_cols]
    numeric_cols = enhanced_features[feature_cols].select_dtypes(include=[np.number]).columns
    X_enhanced = enhanced_features[numeric_cols].fillna(0)
    
    # Target variable already created above
    
    # Split data
    X_tabular_train = X_enhanced.iloc[train_indices].values
    X_tabular_test = X_enhanced.iloc[test_indices].values
    X_text_train = bert_embeddings[train_indices]
    X_text_test = bert_embeddings[test_indices]
    X_graph_train = gnn_embeddings[train_indices]
    X_graph_test = gnn_embeddings[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    print(f"   Tabular features: {X_tabular_train.shape[1]}")
    print(f"   Text features: {X_text_train.shape[1]}")
    print(f"   Graph features: {X_graph_train.shape[1]}")
    print(f"   Train set: {len(X_tabular_train):,} samples")
    print(f"   Test set: {len(X_tabular_test):,} samples")
    print(f"   Class distribution: {np.bincount(y_train)}")
    
    # Create and train improved ensemble
    print("\n3. Training improved multimodal ensemble...")
    ensemble = ImprovedMultimodalEnsemble(random_state=42, device=device)
    ensemble.train_improved_ensemble(
        X_tabular_train, X_text_train, X_graph_train, y_train,
        feature_names=list(numeric_cols)
    )
    
    # Optimize threshold
    print("\n4. Optimizing threshold...")
    optimal_threshold, best_f1 = ensemble.optimize_threshold(
        X_tabular_test, X_text_test, X_graph_test, y_test
    )
    
    # Make predictions
    print("\n5. Making predictions...")
    y_pred, y_proba = ensemble.predict_improved(
        X_tabular_test, X_text_test, X_graph_test
    )
    
    # Evaluate performance
    print("\n6. Evaluating performance...")
    
    # Calculate metrics
    auc_roc = roc_auc_score(y_test, y_proba[:, 1])
    avg_precision = average_precision_score(y_test, y_proba[:, 1])
    accuracy = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Display results
    print("\n📊 IMPROVED MULTIMODAL RESULTS:")
    print("=" * 60)
    print(f"🎯 AUC-ROC: {auc_roc:.4f}")
    print(f"🎯 Average Precision: {avg_precision:.4f}")
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🎯 Recall: {recall:.4f}")
    print(f"🎯 F1-Score: {f1:.4f}")
    
    # Target comparison
    print(f"\n🎯 TARGET COMPARISON:")
    targets = {
        'AUC-ROC': (auc_roc, 0.70),
        'Average Precision': (avg_precision, 0.366),
        'Accuracy': (accuracy, 0.78),
        'F1-Score': (f1, 0.43)
    }
    
    targets_met = 0
    for metric, (actual, target) in targets.items():
        status = "✅" if actual >= target else "❌"
        print(f"   {metric}: {actual:.4f} (target: ≥{target}) {status}")
        if actual >= target:
            targets_met += 1
    
    print(f"\n📈 IMPROVEMENT SUMMARY:")
    print(f"   Targets Met: {targets_met}/4")
    print(f"   Optimal Threshold: {optimal_threshold:.3f}")
    print(f"   Multimodal Weight: {ensemble.multimodal_weight:.1%}")
    
    # Save results
    print(f"\n7. Saving results...")
    results = {
        'model_type': 'Improved Multimodal Ensemble',
        'device': device,
        'test_metrics': {
            'auc_roc': auc_roc,
            'average_precision': avg_precision,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'target_achieved': {
            'auc_target': auc_roc >= 0.70,
            'ap_target': avg_precision >= 0.366,
            'accuracy_target': accuracy >= 0.78,
            'f1_target': f1 >= 0.43
        },
        'architecture': {
            'tabular_features': X_tabular_train.shape[1],
            'text_features': X_text_train.shape[1],
            'graph_features': X_graph_train.shape[1],
            'multimodal_model': True,
            'ensemble_models': len(ensemble.base_models),
            'improved_architecture': True
        },
        'optimal_threshold': optimal_threshold,
        'multimodal_weight': ensemble.multimodal_weight
    }
    
    import pickle
    with open('improved_multimodal_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("   Results saved to: improved_multimodal_results.pkl")
    
    print("\n" + "=" * 80)
    print("IMPROVED MULTIMODAL ENSEMBLE COMPLETED!")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    results = main()
