#!/usr/bin/env python3
"""
Enhanced Ensemble Model for Donor Legacy Intent Prediction
Combines BERT, GNN, and traditional ML models with advanced feature engineering
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureEngineering:
    """
    Advanced feature engineering for donor data
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
        self.pca_transformers = {}
        
    def create_donor_features(self, donors_df):
        """Create advanced donor features"""
        print("Creating enhanced donor features...")
        
        features_df = donors_df.copy()
        
        # 1. Giving pattern features
        features_df['giving_consistency'] = features_df['Consecutive_Yr_Giving_Count'] / features_df['Total_Yr_Giving_Count'].replace(0, 1)
        features_df['avg_gift_size'] = features_df['Lifetime_Giving'] / features_df['Total_Yr_Giving_Count'].replace(0, 1)
        features_df['gift_frequency'] = features_df['Total_Yr_Giving_Count'] / features_df['Estimated_Age'].replace(0, 1)
        
        # 2. Engagement features
        features_df['engagement_intensity'] = features_df['Engagement_Score'] / features_df['Lifetime_Giving'].replace(0, 1)
        features_df['rating_engagement_alignment'] = features_df['Rating'] * features_df['Engagement_Score']
        
        # 3. Demographic features
        features_df['age_group'] = pd.cut(features_df['Estimated_Age'], 
                                        bins=[0, 30, 50, 70, 100], 
                                        labels=[0, 1, 2, 3])
        features_df['age_group'] = features_df['age_group'].astype(int)
        
        # 4. Wealth indicators
        features_df['wealth_proxy'] = features_df['Rating'] * features_df['Lifetime_Giving']
        features_df['family_giving_potential_norm'] = features_df['Family_Giving_Potential'] / features_df['Family_Giving_Potential'].max()
        
        # 5. Interaction features
        features_df['rating_giving_interaction'] = features_df['Rating'] * features_df['Lifetime_Giving']
        features_df['engagement_age_interaction'] = features_df['Engagement_Score'] * features_df['Estimated_Age']
        
        # 6. Categorical encodings
        features_df = pd.get_dummies(features_df, columns=['Primary_Constituent_Type', 'Geographic_Region', 'Professional_Background'])
        
        # 7. Text-based features from interest keywords
        if 'Interest_Keywords' in features_df.columns:
            features_df['num_interests'] = features_df['Interest_Keywords'].str.split(',').str.len().fillna(0)
            features_df['has_planned_giving_interest'] = features_df['Interest_Keywords'].str.contains('planned|legacy|estate', case=False, na=False).astype(int)
        
        print(f"Created {features_df.shape[1]} features from {donors_df.shape[1]} original features")
        return features_df
    
    def create_relationship_features(self, relationships_df, donors_df):
        """Create features from relationship network"""
        print("Creating relationship network features...")
        
        # Network features
        donor_network_features = {}
        
        for donor_id in donors_df['ID']:
            # Family connections
            family_connections = relationships_df[relationships_df['Donor_ID'] == donor_id]
            
            features = {
                'num_family_connections': len(family_connections),
                'has_family_connections': 1 if len(family_connections) > 0 else 0
            }
            
            # Relationship type distribution
            if len(family_connections) > 0:
                relationship_types = family_connections['Relationship_Type'].value_counts()
                for rel_type in ['Spouse', 'Child', 'Parent', 'Sibling']:
                    features[f'num_{rel_type.lower()}_connections'] = relationship_types.get(rel_type, 0)
            
            donor_network_features[donor_id] = features
        
        # Convert to DataFrame
        network_df = pd.DataFrame.from_dict(donor_network_features, orient='index').reset_index()
        network_df.columns = ['ID'] + list(network_df.columns[1:])
        
        print(f"Created {network_df.shape[1]-1} network features")
        return network_df
    
    def create_temporal_features(self, giving_history_df=None):
        """Create temporal features from giving history"""
        if giving_history_df is None or giving_history_df.empty:
            print("No giving history data available")
            return pd.DataFrame()
        
        print("Creating temporal giving features...")
        
        temporal_features = {}
        
        for donor_id in giving_history_df['Donor_ID'].unique():
            donor_history = giving_history_df[giving_history_df['Donor_ID'] == donor_id]
            
            features = {
                'num_gifts': len(donor_history),
                'total_giving_amount': donor_history['Amount'].sum(),
                'avg_gift_amount': donor_history['Amount'].mean(),
                'max_gift_amount': donor_history['Amount'].max(),
                'gift_frequency_months': donor_history['Date'].dt.to_period('M').nunique(),
                'has_recent_giving': 1 if donor_history['Date'].max() > pd.Timestamp.now() - pd.DateOffset(years=1) else 0
            }
            
            # Giving trend analysis
            if len(donor_history) > 1:
                donor_history_sorted = donor_history.sort_values('Date')
                amounts = donor_history_sorted['Amount'].values
                # Simple trend: increasing, decreasing, or stable
                if len(amounts) >= 3:
                    recent_avg = amounts[-3:].mean()
                    earlier_avg = amounts[:3].mean()
                    if recent_avg > earlier_avg * 1.1:
                        features['giving_trend'] = 1  # Increasing
                    elif recent_avg < earlier_avg * 0.9:
                        features['giving_trend'] = -1  # Decreasing
                    else:
                        features['giving_trend'] = 0  # Stable
                else:
                    features['giving_trend'] = 0
            else:
                features['giving_trend'] = 0
            
            temporal_features[donor_id] = features
        
        temporal_df = pd.DataFrame.from_dict(temporal_features, orient='index').reset_index()
        temporal_df.columns = ['ID'] + list(temporal_df.columns[1:])
        
        print(f"Created {temporal_df.shape[1]-1} temporal features")
        return temporal_df
    
    def select_features(self, X, y, method='mutual_info', k=50):
        """Select most informative features"""
        print(f"Selecting top {k} features using {method}...")
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            raise ValueError("Method must be 'mutual_info' or 'f_classif'")
        
        X_selected = selector.fit_transform(X, y)
        
        # Store selector for later use
        self.feature_selectors[method] = selector
        
        print(f"Selected {X_selected.shape[1]} features from {X.shape[1]}")
        return X_selected, selector
    
    def apply_dimensionality_reduction(self, X, method='pca', n_components=20):
        """Apply dimensionality reduction"""
        print(f"Applying {method} dimensionality reduction...")
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError("Only PCA supported currently")
        
        X_reduced = reducer.fit_transform(X)
        
        # Store reducer for later use
        self.pca_transformers[method] = reducer
        
        print(f"Reduced from {X.shape[1]} to {X_reduced.shape[1]} dimensions")
        print(f"Explained variance ratio: {reducer.explained_variance_ratio_.sum():.4f}")
        
        return X_reduced, reducer

class AdvancedEnsembleModel:
    """
    Advanced ensemble model combining multiple approaches
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.meta_model = None
        self.feature_engineering = EnhancedFeatureEngineering()
        self.scalers = {}
        
    def create_base_models(self):
        """Create diverse base models"""
        print("Creating base models...")
        
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=self.random_state
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.random_state
            )
        }
        
        print(f"Created {len(self.models)} base models")
    
    def create_meta_model(self):
        """Create meta-model for stacking"""
        self.meta_model = LogisticRegression(
            C=0.1,
            class_weight='balanced',
            random_state=self.random_state
        )
        print("Created meta-model for stacking")
    
    def prepare_multimodal_features(self, donors_df, bert_embeddings, gnn_embeddings, 
                                  relationships_df=None, giving_history_df=None):
        """Prepare all feature types for ensemble"""
        print("Preparing multimodal features...")
        
        # 1. Enhanced donor features
        enhanced_donors = self.feature_engineering.create_donor_features(donors_df)
        
        # 2. Relationship features
        if relationships_df is not None and not relationships_df.empty:
            network_features = self.feature_engineering.create_relationship_features(relationships_df, donors_df)
            enhanced_donors = enhanced_donors.merge(network_features, on='ID', how='left')
        
        # 3. Temporal features
        if giving_history_df is not None and not giving_history_df.empty:
            temporal_features = self.feature_engineering.create_temporal_features(giving_history_df)
            enhanced_donors = enhanced_donors.merge(temporal_features, on='ID', how='left')
        
        # 4. Remove non-numeric columns for modeling
        numeric_columns = enhanced_donors.select_dtypes(include=[np.number]).columns
        tabular_features = enhanced_donors[numeric_columns].fillna(0)
        
        # 5. Combine with embeddings
        features_dict = {
            'tabular': tabular_features.values,
            'bert': bert_embeddings,
            'gnn': gnn_embeddings
        }
        
        print(f"Feature shapes:")
        print(f"  Tabular: {tabular_features.shape}")
        print(f"  BERT: {bert_embeddings.shape}")
        print(f"  GNN: {gnn_embeddings.shape}")
        
        return features_dict, tabular_features.columns.tolist()
    
    def train_ensemble(self, features_dict, y, feature_names=None, 
                      use_feature_selection=True, use_dimensionality_reduction=True):
        """Train the ensemble model"""
        print("\n" + "=" * 60)
        print("TRAINING ENHANCED ENSEMBLE MODEL")
        print("=" * 60)
        
        # Prepare features
        X_tabular = features_dict['tabular']
        X_bert = features_dict['bert']
        X_gnn = features_dict['gnn']
        
        # Combine all features
        X_combined = np.hstack([X_tabular, X_bert, X_gnn])
        print(f"Combined features shape: {X_combined.shape}")
        
        # Apply feature selection if requested
        if use_feature_selection and X_combined.shape[1] > 100:
            X_selected, selector = self.feature_engineering.select_features(
                X_combined, y, method='mutual_info', k=100
            )
            X_combined = X_selected
        
        # Apply dimensionality reduction if requested
        if use_dimensionality_reduction and X_combined.shape[1] > 50:
            X_reduced, reducer = self.feature_engineering.apply_dimensionality_reduction(
                X_combined, method='pca', n_components=50
            )
            X_combined = X_reduced
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_combined)
        self.scalers['main'] = scaler
        
        # Create base models
        self.create_base_models()
        
        # Train base models
        print("\nTraining base models...")
        base_predictions = np.zeros((len(X_scaled), len(self.models)))
        base_probabilities = np.zeros((len(X_scaled), len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            print(f"Training {name}...")
            model.fit(X_scaled, y)
            
            # Get predictions and probabilities
            base_predictions[:, i] = model.predict(X_scaled)
            if hasattr(model, 'predict_proba'):
                base_probabilities[:, i] = model.predict_proba(X_scaled)[:, 1]
            else:
                base_probabilities[:, i] = base_predictions[:, i]
            
            print(f"  {name} trained successfully")
        
        # Create and train meta-model
        print("\nTraining meta-model...")
        self.create_meta_model()
        
        # Use base model probabilities as features for meta-model
        meta_features = base_probabilities
        self.meta_model.fit(meta_features, y)
        
        print("Ensemble training completed!")
        
        return {
            'X_processed': X_scaled,
            'base_predictions': base_predictions,
            'base_probabilities': base_probabilities,
            'meta_features': meta_features
        }
    
    def predict_ensemble(self, features_dict, return_individual=False):
        """Make predictions using the ensemble"""
        print("Making ensemble predictions...")
        
        # Prepare features (same preprocessing as training)
        X_tabular = features_dict['tabular']
        X_bert = features_dict['bert']
        X_gnn = features_dict['gnn']
        
        # Combine and preprocess
        X_combined = np.hstack([X_tabular, X_bert, X_gnn])
        
        # Apply same preprocessing as training
        if hasattr(self.feature_engineering, 'feature_selectors') and 'mutual_info' in self.feature_engineering.feature_selectors:
            X_combined = self.feature_engineering.feature_selectors['mutual_info'].transform(X_combined)
        
        if hasattr(self.feature_engineering, 'pca_transformers') and 'pca' in self.feature_engineering.pca_transformers:
            X_combined = self.feature_engineering.pca_transformers['pca'].transform(X_combined)
        
        X_scaled = self.scalers['main'].transform(X_combined)
        
        # Get base model predictions
        base_predictions = np.zeros((len(X_scaled), len(self.models)))
        base_probabilities = np.zeros((len(X_scaled), len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            base_predictions[:, i] = model.predict(X_scaled)
            if hasattr(model, 'predict_proba'):
                base_probabilities[:, i] = model.predict_proba(X_scaled)[:, 1]
            else:
                base_probabilities[:, i] = base_predictions[:, i]
        
        # Meta-model prediction
        meta_features = base_probabilities
        ensemble_probabilities = self.meta_model.predict_proba(meta_features)[:, 1]
        ensemble_predictions = (ensemble_probabilities > 0.5).astype(int)
        
        if return_individual:
            return {
                'ensemble_predictions': ensemble_predictions,
                'ensemble_probabilities': ensemble_probabilities,
                'base_predictions': base_predictions,
                'base_probabilities': base_probabilities,
                'model_names': list(self.models.keys())
            }
        else:
            return ensemble_predictions, ensemble_probabilities
    
    def evaluate_ensemble(self, y_true, ensemble_predictions, ensemble_probabilities):
        """Evaluate ensemble performance"""
        print("\n" + "=" * 60)
        print("ENSEMBLE EVALUATION")
        print("=" * 60)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_true, ensemble_predictions)
        precision = precision_score(y_true, ensemble_predictions)
        recall = recall_score(y_true, ensemble_predictions)
        f1 = f1_score(y_true, ensemble_predictions)
        auc = roc_auc_score(y_true, ensemble_probabilities)
        
        print(f"Ensemble Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, ensemble_predictions))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, ensemble_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Ensemble Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def analyze_feature_importance(self, feature_names=None):
        """Analyze feature importance across models"""
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        importance_df = pd.DataFrame()
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
                
                model_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances,
                    'model': name
                })
                
                importance_df = pd.concat([importance_df, model_importance])
        
        if not importance_df.empty:
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            
            # Average importance across models
            avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=True)
            
            avg_importance.tail(20).plot(kind='barh')
            plt.title('Top 20 Features by Average Importance')
            plt.xlabel('Average Importance')
            plt.tight_layout()
            plt.show()
            
            return importance_df
        else:
            print("No feature importance available for current models")
            return None

def create_business_metrics_evaluator():
    """Create a business metrics evaluator for donor legacy intent"""
    
    class BusinessMetricsEvaluator:
        def __init__(self):
            self.cost_false_positive = 500  # Cost of pursuing false positive
            self.cost_false_negative = 2000  # Cost of missing true positive
            self.value_true_positive = 50000  # Value of successful legacy gift
            
        def calculate_roi(self, y_true, y_pred, y_pred_proba):
            """Calculate ROI based on business metrics"""
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            total_cost = (fp * self.cost_false_positive) + (fn * self.cost_false_negative)
            total_value = tp * self.value_true_positive
            
            roi = (total_value - total_cost) / total_cost if total_cost > 0 else float('inf')
            
            return {
                'roi': roi,
                'total_cost': total_cost,
                'total_value': total_value,
                'net_value': total_value - total_cost,
                'tp': tp, 'fp': fp, 'fn': fn
            }
        
        def evaluate_thresholds(self, y_true, y_pred_proba, thresholds=None):
            """Evaluate business metrics across different thresholds"""
            if thresholds is None:
                thresholds = np.arange(0.1, 0.9, 0.05)
            
            results = []
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                metrics = self.calculate_roi(y_true, y_pred, y_pred_proba)
                metrics['threshold'] = threshold
                results.append(metrics)
            
            return pd.DataFrame(results)
    
    return BusinessMetricsEvaluator()

# Example usage
def demo_enhanced_ensemble():
    """Demonstrate the enhanced ensemble model"""
    print("=" * 80)
    print("ENHANCED ENSEMBLE MODEL DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X_tabular, y = make_classification(n_samples=5000, n_features=50, n_classes=2, 
                                     weights=[0.8, 0.2], random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tabular, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create dummy embeddings
    X_bert_train = np.random.randn(len(X_train), 768)
    X_gnn_train = np.random.randn(len(X_train), 64)
    X_bert_test = np.random.randn(len(X_test), 768)
    X_gnn_test = np.random.randn(len(X_test), 64)
    
    # Prepare features
    features_dict_train = {
        'tabular': X_train,
        'bert': X_bert_train,
        'gnn': X_gnn_train
    }
    
    features_dict_test = {
        'tabular': X_test,
        'bert': X_bert_test,
        'gnn': X_gnn_test
    }
    
    # Create and train ensemble
    ensemble = AdvancedEnsembleModel()
    ensemble.train_ensemble(features_dict_train, y_train)
    
    # Make predictions
    predictions, probabilities = ensemble.predict_ensemble(features_dict_test)
    
    # Evaluate
    ensemble.evaluate_ensemble(y_test, predictions, probabilities)
    
    # Business metrics
    business_evaluator = create_business_metrics_evaluator()
    roi_metrics = business_evaluator.calculate_roi(y_test, predictions, probabilities)
    
    print(f"\nBusiness ROI Metrics:")
    print(f"  ROI: {roi_metrics['roi']:.2f}")
    print(f"  Net Value: ${roi_metrics['net_value']:,.2f}")
    
    return ensemble

if __name__ == "__main__":
    demo_enhanced_ensemble()
