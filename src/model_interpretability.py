#!/usr/bin/env python3
"""
Model Interpretability Module for Capstone Project
Implements SHAP values, feature importance, and attention visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class ModelInterpreter:
    """
    Comprehensive model interpretability for multimodal donor prediction
    Addresses user story: "As a researcher, I want model explanations"
    """
    
    def __init__(self, ensemble_model, feature_names):
        self.ensemble_model = ensemble_model
        self.feature_names = feature_names
        self.shap_values = None
        
    def compute_shap_values(self, X_test, sample_size=100):
        """
        Compute SHAP values for model interpretability
        
        User Story: As a researcher, I want to understand which features drive predictions
        """
        print("Computing SHAP values...")
        
        # Sample for faster computation
        if len(X_test) > sample_size:
            indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test[indices]
        else:
            X_sample = X_test
        
        # Use TreeExplainer for Random Forest
        rf_model = self.ensemble_model.models.get('random_forest')
        if rf_model:
            explainer = shap.TreeExplainer(rf_model)
            self.shap_values = explainer.shap_values(X_sample)
            
            print(f"SHAP values computed for {len(X_sample)} samples")
            return self.shap_values
        else:
            print("Random Forest model not found in ensemble")
            return None
    
    def plot_shap_summary(self, X_test, max_display=20):
        """
        Create SHAP summary plot showing feature importance
        """
        if self.shap_values is None:
            self.compute_shap_values(X_test)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values[1] if isinstance(self.shap_values, list) else self.shap_values,
            X_test,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Feature Importance - Legacy Intent Prediction')
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("SHAP summary plot saved as shap_summary.png")
    
    def get_top_features_for_donor(self, donor_idx, X_test, top_n=10):
        """
        Get top features influencing prediction for a specific donor
        
        User Story: As a VP of Advancement, I want to understand why a donor was flagged
        """
        if self.shap_values is None:
            self.compute_shap_values(X_test)
        
        # Get SHAP values for this donor
        donor_shap = self.shap_values[1][donor_idx] if isinstance(self.shap_values, list) else self.shap_values[donor_idx]
        
        # Get feature importances
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': np.abs(donor_shap),
            'actual_shap': donor_shap
        }).sort_values('shap_value', ascending=False)
        
        return feature_importance.head(top_n)
    
    def analyze_feature_importance(self, X_test, y_test):
        """
        Comprehensive feature importance analysis across all models
        """
        print("Analyzing feature importance across ensemble...")
        
        importance_data = []
        
        # Get importance from each model
        for model_name, model in self.ensemble_model.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                for i, importance in enumerate(importances):
                    if i < len(self.feature_names):
                        importance_data.append({
                            'model': model_name,
                            'feature': self.feature_names[i],
                            'importance': importance
                        })
        
        # Create DataFrame
        importance_df = pd.DataFrame(importance_data)
        
        # Aggregate across models
        avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        top_features = avg_importance.head(20)
        plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Average Importance')
        plt.title('Top 20 Features Across Ensemble Models')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(top_features.head(10).items(), 1):
            print(f"{i}. {feature}: {importance:.4f}")
        
        return avg_importance
    
    def explain_prediction(self, donor_id, donor_features, prediction_proba):
        """
        Provide human-readable explanation for a single prediction
        
        User Story: As a planned giving officer, I want to understand WHY a donor was identified
        """
        print(f"\n" + "="*60)
        print(f"PREDICTION EXPLANATION FOR DONOR {donor_id}")
        print("="*60)
        
        print(f"\nPrediction Probability: {prediction_proba:.2%}")
        print(f"Confidence Level: {'High' if prediction_proba > 0.7 or prediction_proba < 0.3 else 'Moderate' if prediction_proba > 0.6 or prediction_proba < 0.4 else 'Low'}")
        
        # Get top influencing features
        top_features = self.get_top_features_for_donor(0, donor_features.reshape(1, -1), top_n=5)
        
        print(f"\nTop 5 Influencing Factors:")
        for i, row in top_features.iterrows():
            direction = "increases" if row['actual_shap'] > 0 else "decreases"
            print(f"  {i+1}. {row['feature']}: {direction} legacy intent likelihood")
        
        return top_features
    
    def create_confidence_scores(self, y_pred_proba):
        """
        Generate confidence scores for VP of Advancement
        
        User Story: As a VP of Advancement, I want confidence scores to prioritize outreach
        """
        # Confidence is distance from 0.5 (uncertain)
        confidence_scores = np.abs(y_pred_proba - 0.5) * 2  # Scale to 0-1
        
        # Categorize
        confidence_levels = pd.cut(
            confidence_scores,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        return confidence_scores, confidence_levels
    
    def generate_prospect_report(self, donors_df, predictions, probabilities, top_n=50):
        """
        Generate actionable prospect report for advancement team
        
        User Story: As a VP of Advancement, I want prioritized prospect lists
        """
        # Create report DataFrame
        report = donors_df.copy()
        report['Legacy_Intent_Probability'] = probabilities
        report['Legacy_Intent_Prediction'] = predictions
        
        # Add confidence scores
        confidence_scores, confidence_levels = self.create_confidence_scores(probabilities)
        report['Confidence_Score'] = confidence_scores
        report['Confidence_Level'] = confidence_levels
        
        # Prioritize: High probability + High confidence
        report['Priority_Score'] = probabilities * confidence_scores
        
        # Sort by priority
        report = report.sort_values('Priority_Score', ascending=False)
        
        # Get top prospects
        top_prospects = report.head(top_n)
        
        print(f"\n" + "="*80)
        print(f"TOP {top_n} LEGACY GIFT PROSPECTS")
        print("="*80)
        print(f"\nTotal Prospects Identified: {(predictions == 1).sum():,}")
        print(f"High Confidence Prospects: {(confidence_levels == 'High').sum():,}")
        print(f"Medium Confidence Prospects: {(confidence_levels == 'Medium').sum():,}")
        
        # Display top 10
        display_cols = ['ID', 'Full_Name', 'Legacy_Intent_Probability', 
                       'Confidence_Level', 'Priority_Score', 'Lifetime_Giving', 'Rating']
        display_cols = [col for col in display_cols if col in top_prospects.columns]
        
        print(f"\nTop 10 Prospects:")
        print(top_prospects[display_cols].head(10).to_string(index=False))
        
        # Save full report
        report.to_csv('legacy_prospect_report.csv', index=False)
        print(f"\nFull report saved to legacy_prospect_report.csv")
        
        return report

def demo_interpretability():
    """Demonstrate interpretability features"""
    print("="*80)
    print("MODEL INTERPRETABILITY DEMONSTRATION")
    print("="*80)
    
    # Create sample data
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                              n_classes=2, weights=[0.8, 0.2], random_state=42)
    
    feature_names = [f'feature_{i}' for i in range(20)]
    
    # Train simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[:800], y[:800])
    
    # Create mock ensemble
    class MockEnsemble:
        def __init__(self, model):
            self.models = {'random_forest': model}
    
    ensemble = MockEnsemble(model)
    
    # Initialize interpreter
    interpreter = ModelInterpreter(ensemble, feature_names)
    
    # Demonstrate features
    print("\n1. Computing SHAP values...")
    interpreter.compute_shap_values(X[800:900])
    
    print("\n2. Analyzing feature importance...")
    interpreter.analyze_feature_importance(X[800:], y[800:])
    
    print("\n3. Generating confidence scores...")
    y_pred_proba = model.predict_proba(X[800:])[:, 1]
    confidence_scores, confidence_levels = interpreter.create_confidence_scores(y_pred_proba)
    
    print(f"   High Confidence: {(confidence_levels == 'High').sum()}")
    print(f"   Medium Confidence: {(confidence_levels == 'Medium').sum()}")
    print(f"   Low Confidence: {(confidence_levels == 'Low').sum()}")
    
    print("\n✅ Interpretability module demonstration complete!")

if __name__ == "__main__":
    demo_interpretability()


