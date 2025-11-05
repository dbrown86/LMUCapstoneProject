#!/usr/bin/env python3
"""
Enhanced Ensemble Model V2 for Donor Legacy Intent Prediction
Advanced ensemble methods with hyperparameter optimization and cost-sensitive learning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsembleModelV2:
    """
    Advanced ensemble model V2 with hyperparameter optimization and cost-sensitive learning
    """
    
    def __init__(self, random_state=42, device=None):
        self.random_state = random_state
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_models = {}
        self.meta_model = None
        self.calibrators = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.samplers = {}
        self.optimal_params = {}
        
    def create_optimized_base_models(self):
        """Create base models with optimized hyperparameters"""
        print("Creating optimized base models...")
        
        # Random Forest with optimized parameters
        self.base_models['random_forest'] = RandomForestClassifier(
            n_estimators=120,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Gradient Boosting with optimized parameters
        self.base_models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=self.random_state
        )
        
        # Logistic Regression with regularization
        self.base_models['logistic_regression'] = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            max_iter=1000,
            random_state=self.random_state
        )
        
        # Calibrated Linear SVM (faster than RBF SVC)
        self.base_models['svm'] = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000, dual=False, random_state=self.random_state),
            method='sigmoid',
            cv=3
        )
        
        # Neural Network with optimized architecture
        self.base_models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=300,
            random_state=self.random_state
        )
        
        # Balanced Random Forest for imbalanced data
        self.base_models['balanced_rf'] = BalancedRandomForestClassifier(
            n_estimators=80,
            max_depth=12,
            min_samples_split=15,
            min_samples_leaf=8,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Balanced Bagging Classifier
        self.base_models['balanced_bagging'] = BalancedBaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=8),
            n_estimators=80,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        print(f"Created {len(self.base_models)} optimized base models")
        
    def optimize_hyperparameters(self, X, y, n_trials=50):
        """Optimize hyperparameters using Optuna"""
        print(f"Optimizing hyperparameters with {n_trials} trials...")
        
        def objective(trial):
            # Sample hyperparameters
            model_name = trial.suggest_categorical('model', ['random_forest', 'gradient_boosting', 'logistic_regression'])
            
            if model_name == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 5, 20),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            elif model_name == 'gradient_boosting':
                model = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    random_state=self.random_state
                )
            else:  # logistic_regression
                model = LogisticRegression(
                    C=trial.suggest_float('C', 0.01, 10.0),
                    penalty=trial.suggest_categorical('penalty', ['l1', 'l2']),
                    solver='liblinear',
                    max_iter=1000,
                    random_state=self.random_state
                )
            
            # Cross-validation score
            scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
            return scores.mean()
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Update best parameters
        best_params = study.best_params
        self.optimal_params = best_params
        
        print(f"Best parameters: {best_params}")
        print(f"Best AUC: {study.best_value:.4f}")
        
        return best_params
    
    def apply_advanced_sampling(self, X, y, method='smoteenn'):
        """Apply advanced sampling techniques for imbalanced data"""
        print(f"Applying {method} sampling...")
        
        if method == 'smote':
            sampler = SMOTE(random_state=self.random_state)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=self.random_state)
        elif method == 'borderline_smote':
            sampler = BorderlineSMOTE(random_state=self.random_state)
        elif method == 'smoteenn':
            sampler = SMOTEENN(random_state=self.random_state)
        elif method == 'smotetomek':
            sampler = SMOTETomek(random_state=self.random_state)
        elif method == 'tomek_links':
            sampler = TomekLinks()
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        self.samplers[method] = sampler
        
        print(f"Resampled from {len(y)} to {len(y_resampled)} samples")
        print(f"New class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train_ensemble_with_optimization(self, X, y, feature_names=None, 
                                       use_sampling=True, sampling_method='smoteenn',
                                       use_feature_selection=True, n_features=200,
                                       use_calibration=True, calibration_method='isotonic'):
        """Train ensemble with full optimization pipeline"""
        print("Training optimized ensemble...")
        
        # Apply sampling if requested
        if use_sampling:
            X_processed, y_processed = self.apply_advanced_sampling(X, y, sampling_method)
        else:
            X_processed, y_processed = X, y
        
        # Feature selection
        if use_feature_selection:
            print(f"Selecting top {n_features} features...")
            selector = SelectKBest(mutual_info_classif, k=min(n_features, X_processed.shape[1]))
            X_selected = selector.fit_transform(X_processed, y_processed)
            self.feature_selectors['mutual_info'] = selector
        else:
            X_selected = X_processed
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_selected)
        self.scalers['robust'] = scaler
        
        # Create optimized base models
        self.create_optimized_base_models()
        
        # Train base models
        print("Training base models...")
        base_predictions = {}
        base_probabilities = {}
        
        for name, model in self.base_models.items():
            print(f"  Training {name}...")
            
            # Train model
            model.fit(X_scaled, y_processed)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            base_predictions[name] = y_pred
            base_probabilities[name] = y_proba
            
            # Calculate performance
            auc = roc_auc_score(y_processed, y_proba)
            print(f"    {name} AUC: {auc:.4f}")
        
        # Create meta-features for stacking
        print("Creating meta-features...")
        meta_features = np.column_stack(list(base_probabilities.values()))
        
        # Train meta-model
        print("Training meta-model...")
        self.meta_model = LogisticRegression(random_state=self.random_state)
        self.meta_model.fit(meta_features, y_processed)
        
        # Calibrate models if requested
        if use_calibration:
            print("Calibrating models...")
            for name, model in self.base_models.items():
                # Skip if already a calibrated wrapper
                if isinstance(model, CalibratedClassifierCV):
                    self.calibrators[name] = model
                    continue
                if hasattr(model, 'predict_proba'):
                    calibrated_model = CalibratedClassifierCV(model, method=calibration_method, cv=3)
                    calibrated_model.fit(X_scaled, y_processed)
                    self.calibrators[name] = calibrated_model
        
        print("Ensemble training completed!")
        
        return {
            'base_predictions': base_predictions,
            'base_probabilities': base_probabilities,
            'meta_features': meta_features
        }
    
    def predict_ensemble(self, X, return_individual=False):
        """Make ensemble predictions"""
        # Apply same preprocessing
        if 'mutual_info' in self.feature_selectors:
            X_selected = self.feature_selectors['mutual_info'].transform(X)
        else:
            X_selected = X
            
        X_scaled = self.scalers['robust'].transform(X_selected)
        
        # Get base model predictions
        base_predictions = {}
        base_probabilities = {}
        
        for name, model in self.base_models.items():
            if name in self.calibrators:
                # Use calibrated model
                y_proba = self.calibrators[name].predict_proba(X_scaled)[:, 1]
            else:
                # Use original model
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_scaled)[:, 1]
                else:
                    y_proba = model.predict(X_scaled)
            
            y_pred = (y_proba >= 0.5).astype(int)
            
            base_predictions[name] = y_pred
            base_probabilities[name] = y_proba
        
        # Create meta-features
        meta_features = np.column_stack(list(base_probabilities.values()))
        
        # Meta-model prediction
        ensemble_proba = self.meta_model.predict_proba(meta_features)[:, 1]
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        if return_individual:
            return ensemble_pred, ensemble_proba, base_predictions, base_probabilities
        else:
            return ensemble_pred, ensemble_proba
    
    def optimize_threshold(self, y_true, y_proba, metric='f1', cost_matrix=None):
        """Optimize threshold for different metrics"""
        print(f"Optimizing threshold for {metric}...")
        
        thresholds = np.linspace(0.01, 0.99, 200)
        best_score = -np.inf
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred)
            elif metric == 'cost' and cost_matrix is not None:
                # Calculate cost-based score
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                cost = fp * cost_matrix['fp'] + fn * cost_matrix['fn']
                score = -cost  # Negative because we want to minimize cost
            else:
                continue
                
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print(f"Best threshold: {best_threshold:.3f}, Score: {best_score:.4f}")
        return best_threshold, best_score
    
    def compute_business_metrics(self, y_true, y_pred, y_proba, 
                                contact_cost=50, avg_gift_value=1000, 
                                conversion_rate=0.1):
        """Compute business metrics"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Only charge cost for contacted (predicted positive) cases
        total_contacts = int((y_pred == 1).sum())
        total_cost = total_contacts * contact_cost
        total_revenue = int(tp) * avg_gift_value * conversion_rate
        
        # Business metrics
        roi = ((total_revenue - total_cost) / total_cost) * 100 if total_cost > 0 else 0
        net_value = total_revenue - total_cost
        cost_per_contact = total_cost / total_contacts
        revenue_per_contact = total_revenue / total_contacts
        
        return {
            'roi': roi,
            'net_value': net_value,
            'cost_per_contact': cost_per_contact,
            'revenue_per_contact': revenue_per_contact,
            'total_contacts': total_contacts,
            'total_cost': total_cost,
            'total_revenue': total_revenue
        }
