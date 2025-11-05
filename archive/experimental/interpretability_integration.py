#!/usr/bin/env python3
"""
Integration script for multimodal interpretability features
Demonstrates how to use interpretability with existing pipeline
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing modules
from model_interpretability import MultimodalModelInterpreter
from enhanced_ensemble_model import AdvancedEnsembleModel
from multimodal_arch import MultimodalFusionModel, prepare_multimodal_data
from gnn_models import DonorGraphPreprocessor, GraphSAGE
from bert_pipeline import ContactReportPreprocessor, EmbeddingExtractor

class InterpretabilityPipeline:
    """
    Complete interpretability pipeline for multimodal donor prediction
    Integrates with existing training and prediction pipelines
    """
    
    def __init__(self, ensemble_model=None, multimodal_model=None):
        self.ensemble_model = ensemble_model
        self.multimodal_model = multimodal_model
        self.interpreter = MultimodalModelInterpreter(
            ensemble_model=ensemble_model,
            multimodal_model=multimodal_model
        )
        
        # Modality-specific components
        self.bert_tokenizer = None
        self.bert_model = None
        self.gnn_model = None
        self.graph_data = None
        
    def setup_bert_components(self, model_name='bert-base-uncased'):
        """Setup BERT components for text interpretability"""
        print("Setting up BERT components for interpretability...")
        
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.bert_model.eval()
        
        print("✅ BERT components ready for attention visualization")
    
    def setup_gnn_components(self, graph_data, gnn_model):
        """Setup GNN components for graph interpretability"""
        print("Setting up GNN components for interpretability...")
        
        self.gnn_model = gnn_model
        self.graph_data = graph_data
        
        print("✅ GNN components ready for graph importance analysis")
    
    def run_comprehensive_interpretability(self, donors_df, contact_reports_df, 
                                         bert_embeddings=None, gnn_embeddings=None,
                                         sample_donor_ids=None, save_reports=True):
        """
        Run comprehensive interpretability analysis on the multimodal model
        
        Args:
            donors_df: Donor dataframe
            contact_reports_df: Contact reports dataframe
            bert_embeddings: Pre-computed BERT embeddings
            gnn_embeddings: Pre-computed GNN embeddings
            sample_donor_ids: List of donor IDs to analyze in detail
            save_reports: Whether to save HTML reports
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE MULTIMODAL INTERPRETABILITY ANALYSIS")
        print("="*80)
        
        # 1. Prepare multimodal data
        print("\n1. Preparing multimodal data...")
        tabular_features, text_embeddings, graph_embeddings, labels, modality_mask = prepare_multimodal_data(
            donors_df, contact_reports_df, bert_embeddings, gnn_embeddings
        )
        
        # 2. Tabular SHAP Analysis
        print("\n2. Computing tabular SHAP values...")
        shap_values, shap_indices = self.interpreter.compute_tabular_shap_values(
            tabular_features, sample_size=200
        )
        
        # 3. Text Attention Analysis (if BERT components available)
        if self.bert_tokenizer is not None and self.bert_model is not None:
            print("\n3. Analyzing text attention patterns...")
            self._analyze_text_attention(contact_reports_df, sample_donor_ids)
        
        # 4. Graph Importance Analysis (if GNN components available)
        if self.gnn_model is not None and self.graph_data is not None:
            print("\n4. Computing graph importance scores...")
            graph_importance = self.interpreter.compute_graph_importance_scores(
                self.graph_data, self.gnn_model
            )
            
            # Visualize graph importance
            self.interpreter.visualize_graph_importance(
                self.graph_data, 
                graph_importance['node_importance'],
                top_k=30
            )
        
        # 5. Confidence Intervals
        print("\n5. Computing prediction confidence intervals...")
        if self.ensemble_model:
            # Get predictions from ensemble
            features_dict = {
                'tabular': tabular_features,
                'bert': text_embeddings,
                'gnn': graph_embeddings
            }
            predictions, probabilities = self.ensemble_model.predict_ensemble(features_dict)
            
            confidence_intervals = self.interpreter.compute_confidence_intervals(
                probabilities, method='bootstrap', n_bootstrap=500
            )
        
        # 6. Individual Donor Analysis
        if sample_donor_ids:
            print(f"\n6. Analyzing {len(sample_donor_ids)} individual donors...")
            for donor_id in sample_donor_ids[:5]:  # Limit to 5 for demo
                self._analyze_individual_donor(
                    donor_id, donors_df, contact_reports_df,
                    tabular_features, text_embeddings, graph_embeddings,
                    save_reports=save_reports
                )
        
        # 7. Generate Summary Report
        print("\n7. Generating summary interpretability report...")
        self._generate_summary_report(
            donors_df, tabular_features, shap_values, 
            confidence_intervals if 'confidence_intervals' in locals() else None
        )
        
        print("\n✅ Comprehensive interpretability analysis completed!")
        
        return {
            'shap_values': shap_values,
            'confidence_intervals': confidence_intervals if 'confidence_intervals' in locals() else None,
            'graph_importance': graph_importance if 'graph_importance' in locals() else None
        }
    
    def _analyze_text_attention(self, contact_reports_df, sample_donor_ids):
        """Analyze text attention for sample donors"""
        if sample_donor_ids is None:
            sample_donor_ids = contact_reports_df['Donor_ID'].unique()[:3]
        
        for donor_id in sample_donor_ids:
            # Get contact reports for this donor
            donor_reports = contact_reports_df[contact_reports_df['Donor_ID'] == donor_id]
            
            if len(donor_reports) > 0:
                # Combine reports
                combined_text = ' [SEP] '.join(donor_reports['Report_Text'].fillna('').astype(str))
                
                # Create attention heatmap
                try:
                    attention_weights, tokens = self.interpreter.create_attention_heatmap(
                        combined_text, self.bert_tokenizer, self.bert_model, layer_idx=-1
                    )
                    print(f"  ✅ Attention analysis completed for donor {donor_id}")
                except Exception as e:
                    print(f"  ⚠️  Attention analysis failed for donor {donor_id}: {e}")
    
    def _analyze_individual_donor(self, donor_id, donors_df, contact_reports_df,
                                tabular_features, text_embeddings, graph_embeddings,
                                save_reports=True):
        """Analyze individual donor with comprehensive breakdown"""
        print(f"\n  Analyzing donor {donor_id}...")
        
        # Find donor index
        donor_idx = donors_df[donors_df['ID'] == donor_id].index
        if len(donor_idx) == 0:
            print(f"    ⚠️  Donor {donor_id} not found in dataset")
            return
        
        donor_idx = donor_idx[0]
        
        # Get donor features
        donor_tabular = tabular_features[donor_idx:donor_idx+1]
        donor_text = text_embeddings[donor_idx:donor_idx+1] if text_embeddings is not None else None
        donor_graph = graph_embeddings[donor_idx:donor_idx+1] if graph_embeddings is not None else None
        
        # Get prediction
        if self.ensemble_model:
            features_dict = {
                'tabular': donor_tabular,
                'bert': donor_text if donor_text is not None else np.zeros((1, 768)),
                'gnn': donor_graph if donor_graph is not None else np.zeros((1, 64))
            }
            prediction, probability = self.ensemble_model.predict_ensemble(features_dict)
            probability = probability[0]
        else:
            probability = 0.5  # Default if no model available
        
        # Create feature breakdown
        breakdown = self.interpreter.create_feature_contribution_breakdown(
            donor_tabular[0], probability,
            tabular_features=self.interpreter.feature_names,
            text_features=None,  # Would need token-level features
            graph_features=None  # Would need node-level features
        )
        
        # Generate report
        if save_reports:
            report_path = f'interpretability_report_donor_{donor_id}.html'
            self.interpreter.generate_interpretability_report(
                donor_id, breakdown, save_path=report_path
            )
            print(f"    ✅ Report saved: {report_path}")
        
        # Print summary
        print(f"    Prediction: {probability:.3f} ({breakdown['confidence_level']} confidence)")
        print(f"    Top features: {[f['feature'] for f in breakdown['top_features'][:3]]}")
    
    def _generate_summary_report(self, donors_df, tabular_features, shap_values, confidence_intervals):
        """Generate summary interpretability report"""
        print("Generating summary interpretability report...")
        
        # Create summary visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. SHAP Summary Plot
        if shap_values is not None:
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]  # Positive class
            else:
                shap_vals = shap_values
            
            # Plot mean absolute SHAP values
            mean_shap = np.abs(shap_vals).mean(axis=0)
            top_features = np.argsort(mean_shap)[-15:]
            
            axes[0, 0].barh(range(len(top_features)), mean_shap[top_features])
            axes[0, 0].set_yticks(range(len(top_features)))
            axes[0, 0].set_yticklabels([f'Feature {i}' for i in top_features])
            axes[0, 0].set_title('Top 15 Features by SHAP Importance')
            axes[0, 0].set_xlabel('Mean |SHAP Value|')
        
        # 2. Prediction Distribution
        if confidence_intervals is not None:
            mean_prob = confidence_intervals['mean_probability']
            ci_lower = confidence_intervals['ci_lower']
            ci_upper = confidence_intervals['ci_upper']
            
            axes[0, 1].bar(['Mean Probability'], [mean_prob], 
                          yerr=[[mean_prob - ci_lower], [ci_upper - mean_prob]],
                          capsize=10, color='skyblue', alpha=0.7)
            axes[0, 1].set_title('Prediction Confidence Interval')
            axes[0, 1].set_ylabel('Probability')
            axes[0, 1].set_ylim(0, 1)
        
        # 3. Dataset Overview
        axes[1, 0].pie([len(donors_df), len(donors_df[donors_df['Legacy_Intent_Binary'] == 1])], 
                      labels=['Non-Legacy', 'Legacy Intent'], 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Dataset Class Distribution')
        
        # 4. Feature Importance Heatmap
        if shap_values is not None:
            # Sample of SHAP values for heatmap
            sample_size = min(50, len(shap_vals))
            sample_indices = np.random.choice(len(shap_vals), sample_size, replace=False)
            sample_shap = shap_vals[sample_indices]
            
            im = axes[1, 1].imshow(sample_shap.T, cmap='RdBu_r', aspect='auto')
            axes[1, 1].set_title('SHAP Values Heatmap (Sample)')
            axes[1, 1].set_xlabel('Sample Index')
            axes[1, 1].set_ylabel('Feature Index')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('interpretability_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Summary report saved as interpretability_summary.png")

def demo_interpretability_integration():
    """Demonstrate interpretability integration with existing pipeline"""
    print("="*80)
    print("INTERPRETABILITY INTEGRATION DEMONSTRATION")
    print("="*80)
    
    # Load sample data
    try:
        donors_df = pd.read_csv('synthetic_donor_dataset/donors.csv')
        contact_reports_df = pd.read_csv('synthetic_donor_dataset/contact_reports.csv')
        print(f"✅ Loaded data: {len(donors_df)} donors, {len(contact_reports_df)} reports")
    except FileNotFoundError:
        print("❌ Sample data not found. Please run data generation first.")
        return
    
    # Create sample ensemble model
    print("\nCreating sample ensemble model...")
    ensemble = AdvancedEnsembleModel()
    ensemble.create_base_models()
    
    # Create interpretability pipeline
    interpretability_pipeline = InterpretabilityPipeline(ensemble_model=ensemble)
    
    # Setup BERT components (optional)
    try:
        interpretability_pipeline.setup_bert_components()
        print("✅ BERT components ready")
    except Exception as e:
        print(f"⚠️  BERT setup failed: {e}")
    
    # Run interpretability analysis
    print("\nRunning interpretability analysis...")
    results = interpretability_pipeline.run_comprehensive_interpretability(
        donors_df, contact_reports_df,
        sample_donor_ids=donors_df['ID'].head(3).tolist(),
        save_reports=True
    )
    
    print("\n✅ Interpretability integration demonstration completed!")
    return results

if __name__ == "__main__":
    demo_interpretability_integration()



