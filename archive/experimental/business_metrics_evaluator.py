#!/usr/bin/env python3
"""
Business Metrics Evaluator for Donor Legacy Intent Prediction
Provides comprehensive business-focused evaluation metrics and ROI analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class DonorLegacyBusinessEvaluator:
    """
    Comprehensive business metrics evaluator for donor legacy intent prediction
    """
    
    def __init__(self, 
                 cost_false_positive=500,      # Cost of pursuing false positive
                 cost_false_negative=2000,     # Cost of missing true positive  
                 value_true_positive=50000,    # Value of successful legacy gift
                 operational_cost_per_contact=50):  # Cost per donor contact
        """
        Initialize business evaluator with cost parameters
        
        Args:
            cost_false_positive: Cost of pursuing a donor who won't give (staff time, resources)
            cost_false_negative: Opportunity cost of missing a donor who would give
            value_true_positive: Expected value of a successful legacy gift
            operational_cost_per_contact: Cost per donor contact attempt
        """
        self.cost_false_positive = cost_false_positive
        self.cost_false_negative = cost_false_negative
        self.value_true_positive = value_true_positive
        self.operational_cost_per_contact = operational_cost_per_contact
        
        print(f"Business Evaluator Initialized:")
        print(f"  False Positive Cost: ${cost_false_positive:,}")
        print(f"  False Negative Cost: ${cost_false_negative:,}")
        print(f"  True Positive Value: ${value_true_positive:,}")
        print(f"  Contact Cost: ${operational_cost_per_contact}")
    
    def calculate_business_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive business metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            dict: Comprehensive business metrics
        """
        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Standard metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Business-specific calculations
        total_cost = (fp * self.cost_false_positive) + (fn * self.cost_false_negative)
        total_value = tp * self.value_true_positive
        net_value = total_value - total_cost
        
        # ROI calculations
        roi = (net_value / total_cost) if total_cost > 0 else float('inf')
        cost_per_prediction = total_cost / len(y_true)
        value_per_prediction = total_value / len(y_true)
        
        # Efficiency metrics
        precision_efficiency = precision * 100  # % of pursued donors who actually give
        recall_efficiency = recall * 100       # % of actual donors we successfully identify
        
        # Contact efficiency
        total_contacts = tp + fp
        contact_success_rate = (tp / total_contacts) if total_contacts > 0 else 0
        contact_efficiency = contact_success_rate * 100
        
        # Cost-benefit ratio
        cost_benefit_ratio = total_value / total_cost if total_cost > 0 else float('inf')
        
        metrics = {
            # Confusion matrix
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            
            # Standard ML metrics
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            
            # Business metrics
            'total_cost': total_cost,
            'total_value': total_value,
            'net_value': net_value,
            'roi': roi,
            'cost_per_prediction': cost_per_prediction,
            'value_per_prediction': value_per_prediction,
            'cost_benefit_ratio': cost_benefit_ratio,
            
            # Efficiency metrics
            'precision_efficiency': precision_efficiency,
            'recall_efficiency': recall_efficiency,
            'contact_success_rate': contact_success_rate,
            'contact_efficiency': contact_efficiency,
            'total_contacts': total_contacts,
            
            # Additional metrics
            'total_samples': len(y_true),
            'positive_samples': np.sum(y_true),
            'negative_samples': np.sum(y_true == 0),
            'predicted_positive': np.sum(y_pred),
            'predicted_negative': np.sum(y_pred == 0)
        }
        
        # Add AUC if probabilities provided
        if y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def evaluate_threshold_impact(self, y_true, y_pred_proba, thresholds=None):
        """
        Evaluate business impact across different decision thresholds
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            thresholds: List of thresholds to evaluate (default: 0.1 to 0.9)
            
        Returns:
            pd.DataFrame: Business metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            metrics = self.calculate_business_metrics(y_true, y_pred, y_pred_proba)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def find_optimal_threshold(self, y_true, y_pred_proba, optimization_metric='net_value'):
        """
        Find optimal threshold based on business metrics
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            optimization_metric: Metric to optimize ('net_value', 'roi', 'f1_score', 'precision')
            
        Returns:
            dict: Optimal threshold and corresponding metrics
        """
        threshold_results = self.evaluate_threshold_impact(y_true, y_pred_proba)
        
        # Find optimal threshold
        if optimization_metric == 'net_value':
            optimal_idx = threshold_results['net_value'].idxmax()
        elif optimization_metric == 'roi':
            # Handle infinite ROI values
            roi_series = threshold_results['roi'].replace([np.inf, -np.inf], np.nan)
            optimal_idx = roi_series.idxmax()
        elif optimization_metric == 'f1_score':
            optimal_idx = threshold_results['f1_score'].idxmax()
        elif optimization_metric == 'precision':
            optimal_idx = threshold_results['precision'].idxmax()
        else:
            raise ValueError(f"Unknown optimization metric: {optimization_metric}")
        
        optimal_threshold = threshold_results.loc[optimal_idx, 'threshold']
        optimal_metrics = threshold_results.loc[optimal_idx].to_dict()
        
        print(f"Optimal threshold for {optimization_metric}: {optimal_threshold:.4f}")
        print(f"Net Value: ${optimal_metrics['net_value']:,.2f}")
        print(f"ROI: {optimal_metrics['roi']:.2f}")
        print(f"F1 Score: {optimal_metrics['f1_score']:.4f}")
        
        return {
            'threshold': optimal_threshold,
            'metrics': optimal_metrics,
            'all_results': threshold_results
        }
    
    def plot_threshold_analysis(self, threshold_results):
        """Plot comprehensive threshold analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Net Value vs Threshold
        axes[0, 0].plot(threshold_results['threshold'], threshold_results['net_value'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Net Value ($)')
        axes[0, 0].set_title('Net Value vs Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # ROI vs Threshold
        roi_clean = threshold_results['roi'].replace([np.inf, -np.inf], np.nan)
        axes[0, 1].plot(threshold_results['threshold'], roi_clean, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('ROI')
        axes[0, 1].set_title('ROI vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision vs Recall
        axes[0, 2].plot(threshold_results['recall'], threshold_results['precision'], 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision vs Recall')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Contact Efficiency vs Threshold
        axes[1, 0].plot(threshold_results['threshold'], threshold_results['contact_efficiency'], 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Contact Efficiency (%)')
        axes[1, 0].set_title('Contact Efficiency vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Total Contacts vs Threshold
        axes[1, 1].plot(threshold_results['threshold'], threshold_results['total_contacts'], 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Total Contacts')
        axes[1, 1].set_title('Total Contacts vs Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Cost-Benefit Ratio vs Threshold
        cb_clean = threshold_results['cost_benefit_ratio'].replace([np.inf, -np.inf], np.nan)
        axes[1, 2].plot(threshold_results['threshold'], cb_clean, 'brown', linewidth=2)
        axes[1, 2].set_xlabel('Threshold')
        axes[1, 2].set_ylabel('Cost-Benefit Ratio')
        axes[1, 2].set_title('Cost-Benefit Ratio vs Threshold')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_business_report(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """
        Generate comprehensive business report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model for reporting
            
        Returns:
            dict: Comprehensive business report
        """
        print("=" * 80)
        print(f"BUSINESS EVALUATION REPORT - {model_name}")
        print("=" * 80)
        
        # Calculate metrics
        metrics = self.calculate_business_metrics(y_true, y_pred, y_pred_proba)
        
        # Print executive summary
        print("\n" + "=" * 60)
        print("EXECUTIVE SUMMARY")
        print("=" * 60)
        print(f"Model Performance:")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision_efficiency']:.1f}% efficiency)")
        print(f"  Recall: {metrics['recall']:.4f} ({metrics['recall_efficiency']:.1f}% efficiency)")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        print(f"\nBusiness Impact:")
        print(f"  Net Value: ${metrics['net_value']:,.2f}")
        print(f"  ROI: {metrics['roi']:.2f}")
        print(f"  Cost-Benefit Ratio: {metrics['cost_benefit_ratio']:.2f}")
        
        print(f"\nOperational Metrics:")
        print(f"  Total Contacts: {metrics['total_contacts']:,}")
        print(f"  Contact Success Rate: {metrics['contact_efficiency']:.1f}%")
        print(f"  Cost per Prediction: ${metrics['cost_per_prediction']:.2f}")
        print(f"  Value per Prediction: ${metrics['value_per_prediction']:.2f}")
        
        # Detailed breakdown
        print("\n" + "=" * 60)
        print("DETAILED BREAKDOWN")
        print("=" * 60)
        
        cm = metrics['confusion_matrix']
        print(f"Confusion Matrix:")
        print(f"  True Positives: {cm['tp']:,} (${cm['tp'] * self.value_true_positive:,} value)")
        print(f"  False Positives: {cm['fp']:,} (${cm['fp'] * self.cost_false_positive:,} cost)")
        print(f"  True Negatives: {cm['tn']:,}")
        print(f"  False Negatives: {cm['fn']:,} (${cm['fn'] * self.cost_false_negative:,} opportunity cost)")
        
        print(f"\nCost Breakdown:")
        print(f"  False Positive Cost: ${cm['fp'] * self.cost_false_positive:,}")
        print(f"  False Negative Cost: ${cm['fn'] * self.cost_false_negative:,}")
        print(f"  Total Cost: ${metrics['total_cost']:,}")
        
        print(f"\nValue Breakdown:")
        print(f"  True Positive Value: ${metrics['total_value']:,}")
        print(f"  Net Value: ${metrics['net_value']:,}")
        
        # Threshold analysis if probabilities provided
        if y_pred_proba is not None:
            print("\n" + "=" * 60)
            print("THRESHOLD OPTIMIZATION")
            print("=" * 60)
            
            # Find optimal threshold for net value
            optimal_net = self.find_optimal_threshold(y_true, y_pred_proba, 'net_value')
            
            # Find optimal threshold for ROI
            optimal_roi = self.find_optimal_threshold(y_true, y_pred_proba, 'roi')
            
            print(f"\nOptimal Thresholds:")
            print(f"  For Net Value: {optimal_net['threshold']:.4f}")
            print(f"  For ROI: {optimal_roi['threshold']:.4f}")
            
            # Plot threshold analysis
            self.plot_threshold_analysis(optimal_net['all_results'])
        
        # Recommendations
        print("\n" + "=" * 60)
        print("BUSINESS RECOMMENDATIONS")
        print("=" * 60)
        
        if metrics['precision'] < 0.5:
            print("WARNING: LOW PRECISION: Consider increasing decision threshold to reduce false positives")
        elif metrics['recall'] < 0.5:
            print("WARNING: LOW RECALL: Consider decreasing decision threshold to capture more true positives")
        
        if metrics['roi'] < 1.0:
            print("WARNING: NEGATIVE ROI: Model is not profitable. Consider:")
            print("   - Improving model accuracy")
            print("   - Reducing operational costs")
            print("   - Adjusting cost assumptions")
        else:
            print(f"SUCCESS: POSITIVE ROI: Model generates {metrics['roi']:.1f}x return on investment")
        
        if metrics['contact_efficiency'] < 20:
            print("WARNING: LOW CONTACT EFFICIENCY: Consider more targeted outreach strategies")
        else:
            print(f"SUCCESS: GOOD CONTACT EFFICIENCY: {metrics['contact_efficiency']:.1f}% of contacts are successful")
        
        return {
            'metrics': metrics,
            'optimal_thresholds': {
                'net_value': optimal_net if y_pred_proba is not None else None,
                'roi': optimal_roi if y_pred_proba is not None else None
            }
        }
    
    def compare_models(self, model_results, model_names=None):
        """
        Compare multiple models using business metrics
        
        Args:
            model_results: List of (y_true, y_pred, y_pred_proba) tuples
            model_names: List of model names
            
        Returns:
            pd.DataFrame: Comparison of models
        """
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(model_results))]
        
        print("=" * 80)
        print("MODEL COMPARISON - BUSINESS METRICS")
        print("=" * 80)
        
        comparison_data = []
        
        for i, (y_true, y_pred, y_pred_proba) in enumerate(model_results):
            metrics = self.calculate_business_metrics(y_true, y_pred, y_pred_proba)
            
            comparison_data.append({
                'Model': model_names[i],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'Net_Value': metrics['net_value'],
                'ROI': metrics['roi'],
                'Contact_Efficiency': metrics['contact_efficiency'],
                'Total_Contacts': metrics['total_contacts'],
                'Cost_Per_Prediction': metrics['cost_per_prediction']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print comparison table
        print("\nModel Comparison:")
        print(comparison_df.round(4).to_string(index=False))
        
        # Find best models
        best_net_value = comparison_df.loc[comparison_df['Net_Value'].idxmax(), 'Model']
        best_roi = comparison_df.loc[comparison_df['ROI'].idxmax(), 'Model']
        best_f1 = comparison_df.loc[comparison_df['F1_Score'].idxmax(), 'Model']
        
        print(f"\nBest Models:")
        print(f"  Highest Net Value: {best_net_value}")
        print(f"  Highest ROI: {best_roi}")
        print(f"  Highest F1 Score: {best_f1}")
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Net Value comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['Net_Value'])
        axes[0, 0].set_title('Net Value Comparison')
        axes[0, 0].set_ylabel('Net Value ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # ROI comparison
        roi_clean = comparison_df['ROI'].replace([np.inf, -np.inf], np.nan)
        axes[0, 1].bar(comparison_df['Model'], roi_clean)
        axes[0, 1].set_title('ROI Comparison')
        axes[0, 1].set_ylabel('ROI')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['F1_Score'])
        axes[1, 0].set_title('F1 Score Comparison')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Contact Efficiency comparison
        axes[1, 1].bar(comparison_df['Model'], comparison_df['Contact_Efficiency'])
        axes[1, 1].set_title('Contact Efficiency Comparison')
        axes[1, 1].set_ylabel('Contact Efficiency (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df

def create_scenario_analyzer():
    """Create a scenario analyzer for different business situations"""
    
    class ScenarioAnalyzer:
        def __init__(self):
            self.scenarios = {
                'conservative': {
                    'cost_false_positive': 1000,
                    'cost_false_negative': 3000,
                    'value_true_positive': 40000,
                    'description': 'Conservative: High costs, moderate value'
                },
                'moderate': {
                    'cost_false_positive': 500,
                    'cost_false_negative': 2000,
                    'value_true_positive': 50000,
                    'description': 'Moderate: Balanced costs and value'
                },
                'aggressive': {
                    'cost_false_positive': 200,
                    'cost_false_negative': 1000,
                    'value_true_positive': 60000,
                    'description': 'Aggressive: Low costs, high value'
                },
                'high_value': {
                    'cost_false_positive': 800,
                    'cost_false_negative': 5000,
                    'value_true_positive': 100000,
                    'description': 'High Value: Major gift scenario'
                }
            }
        
        def analyze_scenarios(self, y_true, y_pred, y_pred_proba):
            """Analyze model performance across different business scenarios"""
            print("=" * 80)
            print("SCENARIO ANALYSIS")
            print("=" * 80)
            
            scenario_results = {}
            
            for scenario_name, params in self.scenarios.items():
                # Extract description and remove it from params for evaluator
                description = params.get('description', scenario_name)
                print(f"\n{description}:")
                print("-" * 60)
                
                # Create evaluator params without description
                evaluator_params = {k: v for k, v in params.items() if k != 'description'}
                evaluator = DonorLegacyBusinessEvaluator(**evaluator_params)
                metrics = evaluator.calculate_business_metrics(y_true, y_pred, y_pred_proba)
                
                scenario_results[scenario_name] = metrics
                
                print(f"Net Value: ${metrics['net_value']:,.2f}")
                print(f"ROI: {metrics['roi']:.2f}")
                print(f"Contact Efficiency: {metrics['contact_efficiency']:.1f}%")
            
            return scenario_results
    
    return ScenarioAnalyzer()

# Example usage
def demo_business_evaluator():
    """Demonstrate the business metrics evaluator"""
    print("=" * 80)
    print("BUSINESS METRICS EVALUATOR DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample predictions with some imbalance
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    y_pred_proba = np.random.beta(2, 5, size=n_samples)  # Skewed towards 0
    y_pred_proba[y_true == 1] = np.random.beta(5, 2, size=np.sum(y_true))  # Higher for true positives
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Create evaluator
    evaluator = DonorLegacyBusinessEvaluator()
    
    # Generate business report
    report = evaluator.generate_business_report(y_true, y_pred, y_pred_proba, "Demo Model")
    
    # Scenario analysis
    scenario_analyzer = create_scenario_analyzer()
    scenario_results = scenario_analyzer.analyze_scenarios(y_true, y_pred, y_pred_proba)
    
    return evaluator, report, scenario_results

if __name__ == "__main__":
    demo_business_evaluator()
