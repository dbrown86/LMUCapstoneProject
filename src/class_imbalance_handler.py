#!/usr/bin/env python3
"""
Enhanced Class Imbalance Handling System
Implements SMOTE, cost-sensitive learning, and threshold optimization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("Warning: imbalanced-learn not available. Using basic class balancing.")
    IMBLEARN_AVAILABLE = False
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class AdvancedClassImbalanceHandler:
    """
    Comprehensive class imbalance handling system
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.sampling_methods = {}
        self.threshold_optimizer = None
        
    def analyze_class_distribution(self, X, y, verbose=True):
        """Analyze class distribution and imbalance severity"""
        if verbose:
            print("=" * 60)
            print("CLASS DISTRIBUTION ANALYSIS")
            print("=" * 60)
        
        # Count classes
        class_counts = Counter(y)
        total_samples = len(y)
        
        # Calculate ratios
        majority_class = max(class_counts.keys(), key=lambda x: class_counts[x])
        minority_class = min(class_counts.keys(), key=lambda x: class_counts[x])
        
        majority_count = class_counts[majority_class]
        minority_count = class_counts[minority_class]
        
        imbalance_ratio = majority_count / minority_count
        
        if verbose:
            print(f"Total samples: {total_samples:,}")
            print(f"Majority class ({majority_class}): {majority_count:,} ({majority_count/total_samples*100:.1f}%)")
            print(f"Minority class ({minority_class}): {minority_count:,} ({minority_count/total_samples*100:.1f}%)")
            print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
            
            # Categorize imbalance severity
            if imbalance_ratio <= 2:
                severity = "Mild"
            elif imbalance_ratio <= 10:
                severity = "Moderate"
            elif imbalance_ratio <= 20:
                severity = "Severe"
            else:
                severity = "Extreme"
            
            print(f"Imbalance severity: {severity}")
        
        return {
            'class_counts': class_counts,
            'imbalance_ratio': imbalance_ratio,
            'severity': severity,
            'majority_class': majority_class,
            'minority_class': minority_class
        }
    
    def apply_smote_variants(self, X_train, y_train, X_val, y_val):
        """Apply different SMOTE variants and compare performance"""
        print("\n" + "=" * 60)
        print("APPLYING SMOTE VARIANTS")
        print("=" * 60)
        
        if not IMBLEARN_AVAILABLE:
            print("imbalanced-learn not available. Using basic class balancing with sklearn.")
            return self._apply_basic_balancing(X_train, y_train, X_val, y_val)
        
        # Define SMOTE variants
        smote_variants = {
            'Original': None,
            'SMOTE': SMOTE(random_state=self.random_state),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=self.random_state),
            'ADASYN': ADASYN(random_state=self.random_state),
            'SMOTEENN': SMOTEENN(random_state=self.random_state),
            'SMOTETomek': SMOTETomek(random_state=self.random_state)
        }
        
        results = {}
        
        for name, sampler in smote_variants.items():
            print(f"\nTesting {name}...")
            
            if sampler is None:
                # Original data
                X_resampled, y_resampled = X_train, y_train
            else:
                try:
                    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                except Exception as e:
                    print(f"Error with {name}: {e}")
                    continue
            
            # Train a simple classifier for comparison
            clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            clf.fit(X_resampled, y_resampled)
            
            # Predict on validation set
            y_pred = clf.predict(X_val)
            y_pred_proba = clf.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            
            # Store results
            results[name] = {
                'X_resampled': X_resampled,
                'y_resampled': y_resampled,
                'classifier': clf,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'y_pred_proba': y_pred_proba
            }
            
            # Print class distribution after sampling
            class_counts = Counter(y_resampled)
            print(f"  After sampling: {dict(class_counts)}")
            print(f"  F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Find best method
        best_method = max(results.keys(), key=lambda x: results[x]['f1_score'])
        print(f"\nBest method: {best_method} (F1: {results[best_method]['f1_score']:.4f})")
        
        self.sampling_methods = results
        return results, best_method
    
    def _apply_basic_balancing(self, X_train, y_train, X_val, y_val):
        """Apply basic class balancing when imbalanced-learn is not available"""
        print("Using basic sklearn class balancing...")
        
        # Use class_weight='balanced' with RandomForest
        basic_variants = {
            'Original': RandomForestClassifier(random_state=self.random_state),
            'Balanced': RandomForestClassifier(class_weight='balanced', random_state=self.random_state)
        }
        
        results = {}
        
        for name, model in basic_variants.items():
            print(f"\nTesting {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            
            # Store results
            results[name] = {
                'X_resampled': X_train,  # No resampling, use original
                'y_resampled': y_train,
                'classifier': model,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Find best method
        best_method = max(results.keys(), key=lambda x: results[x]['f1_score'])
        print(f"\nBest method: {best_method} (F1: {results[best_method]['f1_score']:.4f})")
        
        self.sampling_methods = results
        return results, best_method
    
    def optimize_threshold(self, y_true, y_pred_proba, method='f1'):
        """Optimize decision threshold for different business objectives"""
        print("\n" + "=" * 60)
        print("THRESHOLD OPTIMIZATION")
        print("=" * 60)
        
        # Get precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Get ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        
        # Calculate F1 scores for different thresholds
        f1_scores = []
        for threshold in pr_thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_thresh)
            f1_scores.append(f1)
        
        # Find optimal thresholds
        optimal_thresholds = {}
        
        if method == 'f1' or method == 'all':
            best_f1_idx = np.argmax(f1_scores)
            optimal_thresholds['f1'] = pr_thresholds[best_f1_idx]
        
        if method == 'precision' or method == 'all':
            # Find threshold that maximizes precision while maintaining reasonable recall
            precision_thresh_idx = np.where(recall >= 0.5)[0]
            if len(precision_thresh_idx) > 0:
                best_precision_idx = precision_thresh_idx[np.argmax(precision[precision_thresh_idx])]
                optimal_thresholds['precision'] = pr_thresholds[best_precision_idx]
        
        if method == 'recall' or method == 'all':
            # Find threshold that maximizes recall while maintaining reasonable precision
            recall_thresh_idx = np.where(precision >= 0.3)[0]
            if len(recall_thresh_idx) > 0:
                best_recall_idx = recall_thresh_idx[np.argmax(recall[recall_thresh_idx])]
                optimal_thresholds['recall'] = pr_thresholds[best_recall_idx]
        
        # Print results
        print("Optimal thresholds:")
        for metric, threshold in optimal_thresholds.items():
            y_pred_opt = (y_pred_proba >= threshold).astype(int)
            f1_opt = f1_score(y_true, y_pred_opt)
            precision_opt = precision_score(y_true, y_pred_opt)
            recall_opt = recall_score(y_true, y_pred_opt)
            
            print(f"  {metric.capitalize()}: {threshold:.4f}")
            print(f"    F1: {f1_opt:.4f}, Precision: {precision_opt:.4f}, Recall: {recall_opt:.4f}")
        
        # Plot threshold optimization
        self._plot_threshold_optimization(y_true, y_pred_proba, pr_thresholds, f1_scores, optimal_thresholds)
        
        self.threshold_optimizer = optimal_thresholds
        return optimal_thresholds
    
    def _plot_threshold_optimization(self, y_true, y_pred_proba, thresholds, f1_scores, optimal_thresholds):
        """Plot threshold optimization results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        axes[0, 0].plot(recall, precision, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Recall')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision-Recall Curve')
        axes[0, 0].grid(True, alpha=0.3)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, 'r-', linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score vs Threshold
        axes[1, 0].plot(thresholds, f1_scores, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mark optimal thresholds
        for metric, threshold in optimal_thresholds.items():
            axes[1, 0].axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label=f'Optimal {metric}')
        axes[1, 0].legend()
        
        # Threshold comparison table
        axes[1, 1].axis('off')
        table_data = []
        for metric, threshold in optimal_thresholds.items():
            y_pred_opt = (y_pred_proba >= threshold).astype(int)
            f1_opt = f1_score(y_true, y_pred_opt)
            precision_opt = precision_score(y_true, y_pred_opt)
            recall_opt = recall_score(y_true, y_pred_opt)
            
            table_data.append([metric.capitalize(), f"{threshold:.4f}", 
                             f"{f1_opt:.4f}", f"{precision_opt:.4f}", f"{recall_opt:.4f}"])
        
        table = axes[1, 1].table(cellText=table_data,
                               colLabels=['Metric', 'Threshold', 'F1', 'Precision', 'Recall'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Optimal Thresholds Performance')
        
        plt.tight_layout()
        plt.show()
    
    def create_cost_sensitive_weights(self, y, cost_ratio=5.0):
        """Create cost-sensitive class weights"""
        print(f"\nCreating cost-sensitive weights with ratio {cost_ratio}:1...")
        
        class_counts = Counter(y)
        total_samples = len(y)
        
        # Calculate weights (inverse frequency with cost adjustment)
        weights = {}
        for class_label, count in class_counts.items():
            if class_label == 1:  # Minority class (legacy intent)
                weights[class_label] = total_samples / (len(class_counts) * count * cost_ratio)
            else:  # Majority class
                weights[class_label] = total_samples / (len(class_counts) * count)
        
        print(f"Class weights: {weights}")
        return weights
    
    def evaluate_business_impact(self, y_true, y_pred, y_pred_proba, cost_fp=100, cost_fn=500):
        """Evaluate business impact with custom cost functions"""
        print("\n" + "=" * 60)
        print("BUSINESS IMPACT EVALUATION")
        print("=" * 60)
        
        # Calculate confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Calculate costs
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        cost_per_prediction = total_cost / len(y_true)
        
        # Calculate business metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Expected value calculation
        # Assuming each true positive generates $10,000 in legacy value
        legacy_value_per_tp = 10000
        total_legacy_value = tp * legacy_value_per_tp
        net_value = total_legacy_value - total_cost
        
        print(f"Confusion Matrix:")
        print(f"  True Positives: {tp:,}")
        print(f"  False Positives: {fp:,}")
        print(f"  True Negatives: {tn:,}")
        print(f"  False Negatives: {fn:,}")
        print()
        print(f"Performance Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print()
        print(f"Business Impact:")
        print(f"  False Positive Cost: ${cost_fp:,} per case")
        print(f"  False Negative Cost: ${cost_fn:,} per case")
        print(f"  Total Cost: ${total_cost:,}")
        print(f"  Cost per Prediction: ${cost_per_prediction:.2f}")
        print()
        print(f"Legacy Value:")
        print(f"  Legacy Value per TP: ${legacy_value_per_tp:,}")
        print(f"  Total Legacy Value: ${total_legacy_value:,}")
        print(f"  Net Value: ${net_value:,}")
        
        return {
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'performance': {'precision': precision, 'recall': recall, 'f1': f1},
            'costs': {'total_cost': total_cost, 'cost_per_prediction': cost_per_prediction},
            'value': {'total_legacy_value': total_legacy_value, 'net_value': net_value}
        }
    
    def create_ensemble_sampling(self, X_train, y_train, n_estimators=5):
        """Create ensemble of models with different sampling strategies"""
        print("\n" + "=" * 60)
        print("CREATING ENSEMBLE WITH DIFFERENT SAMPLING STRATEGIES")
        print("=" * 60)
        
        # Define different sampling strategies
        sampling_strategies = {
            'original': None,
            'smote': SMOTE(random_state=self.random_state),
            'borderline_smote': BorderlineSMOTE(random_state=self.random_state),
            'adasyn': ADASYN(random_state=self.random_state),
            'smote_enn': SMOTEENN(random_state=self.random_state)
        }
        
        ensemble_models = []
        
        for strategy_name, sampler in sampling_strategies.items():
            print(f"Training model with {strategy_name} sampling...")
            
            # Apply sampling
            if sampler is None:
                X_resampled, y_resampled = X_train, y_train
            else:
                try:
                    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                except Exception as e:
                    print(f"Error with {strategy_name}: {e}")
                    continue
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=self.random_state,
                class_weight='balanced'
            )
            model.fit(X_resampled, y_resampled)
            
            ensemble_models.append({
                'strategy': strategy_name,
                'model': model,
                'sampler': sampler
            })
            
            print(f"  Model trained with {len(X_resampled)} samples")
        
        print(f"Created ensemble with {len(ensemble_models)} models")
        return ensemble_models
    
    def predict_ensemble(self, ensemble_models, X_test, method='voting'):
        """Make predictions using ensemble of models"""
        if method == 'voting':
            # Simple majority voting
            predictions = []
            for model_info in ensemble_models:
                pred = model_info['model'].predict(X_test)
                predictions.append(pred)
            
            # Majority vote
            ensemble_pred = np.round(np.mean(predictions, axis=0))
            ensemble_proba = np.mean([model_info['model'].predict_proba(X_test)[:, 1] 
                                    for model_info in ensemble_models], axis=0)
        
        elif method == 'weighted':
            # Weighted voting based on individual model performance
            # This would require validation scores from training
            ensemble_pred = None
            ensemble_proba = None
        
        return ensemble_pred, ensemble_proba

def demo_class_imbalance_handling():
    """Demonstrate the class imbalance handling system"""
    print("=" * 80)
    print("CLASS IMBALANCE HANDLING DEMONSTRATION")
    print("=" * 80)
    
    # Create sample imbalanced dataset
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    
    # Generate imbalanced dataset
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.8, 0.2],  # 80/20 split
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Initialize handler
    handler = AdvancedClassImbalanceHandler()
    
    # Analyze class distribution
    handler.analyze_class_distribution(X_train, y_train)
    
    # Apply SMOTE variants
    smote_results, best_method = handler.apply_smote_variants(X_train, y_train, X_val, y_val)
    
    # Use best method for threshold optimization
    best_y_pred_proba = smote_results[best_method]['y_pred_proba']
    optimal_thresholds = handler.optimize_threshold(y_val, best_y_pred_proba)
    
    # Evaluate business impact
    best_y_pred = (best_y_pred_proba >= optimal_thresholds['f1']).astype(int)
    business_impact = handler.evaluate_business_impact(y_val, best_y_pred, best_y_pred_proba)
    
    # Create ensemble
    ensemble_models = handler.create_ensemble_sampling(X_train, y_train)
    
    # Make ensemble predictions
    ensemble_pred, ensemble_proba = handler.predict_ensemble(ensemble_models, X_val)
    
    # Evaluate ensemble performance
    ensemble_f1 = f1_score(y_val, ensemble_pred)
    print(f"\nEnsemble F1 Score: {ensemble_f1:.4f}")
    
    return handler, smote_results, optimal_thresholds, business_impact

if __name__ == "__main__":
    demo_class_imbalance_handling()
