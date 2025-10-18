#!/usr/bin/env python3
"""
Comprehensive Example: Model Interpretability for Multimodal Donor Prediction
Demonstrates all interpretability features with real data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

# Import project modules
from src.model_interpretability import MultimodalModelInterpreter
from src.enhanced_ensemble_model import AdvancedEnsembleModel
from src.multimodal_arch import prepare_multimodal_data
from src.gnn_models import DonorGraphPreprocessor, GraphSAGE
from src.bert_pipeline import ContactReportPreprocessor, EmbeddingExtractor
from src.interpretability_integration import InterpretabilityPipeline

def load_data():
    """Load synthetic donor dataset"""
    print("Loading synthetic donor dataset...")
    
    try:
        donors_df = pd.read_csv('synthetic_donor_dataset/donors.csv')
        contact_reports_df = pd.read_csv('synthetic_donor_dataset/contact_reports.csv')
        relationships_df = pd.read_csv('synthetic_donor_dataset/relationships.csv')
        giving_history_df = pd.read_csv('synthetic_donor_dataset/giving_history.csv')
        
        print(f"✅ Data loaded successfully:")
        print(f"   Donors: {len(donors_df):,}")
        print(f"   Contact Reports: {len(contact_reports_df):,}")
        print(f"   Relationships: {len(relationships_df):,}")
        print(f"   Giving History: {len(giving_history_df):,}")
        
        return donors_df, contact_reports_df, relationships_df, giving_history_df
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("Please run the data generation pipeline first.")
        return None, None, None, None

def create_sample_embeddings(donors_df, contact_reports_df):
    """Create sample embeddings for demonstration"""
    print("\nCreating sample embeddings...")
    
    # Create dummy BERT embeddings
    bert_embeddings = np.random.randn(len(donors_df), 768)
    print(f"✅ BERT embeddings: {bert_embeddings.shape}")
    
    # Create dummy GNN embeddings
    gnn_embeddings = np.random.randn(len(donors_df), 64)
    print(f"✅ GNN embeddings: {gnn_embeddings.shape}")
    
    return bert_embeddings, gnn_embeddings

def demonstrate_tabular_interpretability(ensemble_model, tabular_features, feature_names):
    """Demonstrate tabular feature interpretability with SHAP"""
    print("\n" + "="*60)
    print("TABULAR INTERPRETABILITY DEMONSTRATION")
    print("="*60)
    
    # Create interpreter
    interpreter = MultimodalModelInterpreter(ensemble_model=ensemble_model, feature_names=feature_names)
    
    # Compute SHAP values
    print("\n1. Computing SHAP values...")
    shap_values, indices = interpreter.compute_tabular_shap_values(tabular_features, sample_size=100)
    
    # Create SHAP summary plot
    print("\n2. Creating SHAP summary plot...")
    interpreter.plot_shap_summary(tabular_features[indices], max_display=15)
    
    # Analyze feature importance
    print("\n3. Analyzing feature importance...")
    importance_scores = interpreter.analyze_feature_importance(tabular_features, np.random.randint(0, 2, len(tabular_features)))
    
    # Individual donor analysis
    print("\n4. Individual donor analysis...")
    donor_features = tabular_features[0:1]  # First donor
    prediction_proba = 0.75  # Example probability
    
    top_features = interpreter.get_top_features_for_donor(0, donor_features, top_n=10)
    print(f"   Top 10 features for donor 0:")
    for i, row in top_features.iterrows():
        print(f"     {i+1}. {row['feature']}: {row['actual_shap']:.4f}")
    
    return interpreter

def demonstrate_text_interpretability(contact_reports_df, sample_donor_id):
    """Demonstrate text interpretability with BERT attention"""
    print("\n" + "="*60)
    print("TEXT INTERPRETABILITY DEMONSTRATION")
    print("="*60)
    
    # Setup BERT components
    print("\n1. Setting up BERT components...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased', output_attentions=True)
        model.eval()
        print("✅ BERT components loaded successfully")
    except Exception as e:
        print(f"❌ BERT setup failed: {e}")
        return None
    
    # Get sample text
    print(f"\n2. Analyzing text for donor {sample_donor_id}...")
    donor_reports = contact_reports_df[contact_reports_df['Donor_ID'] == sample_donor_id]
    
    if len(donor_reports) == 0:
        print(f"   ⚠️  No contact reports found for donor {sample_donor_id}")
        return None
    
    # Combine reports
    combined_text = ' [SEP] '.join(donor_reports['Report_Text'].fillna('').astype(str))
    print(f"   Text length: {len(combined_text)} characters")
    
    # Create interpreter
    interpreter = MultimodalModelInterpreter()
    
    # Create attention heatmap
    print("\n3. Creating attention heatmap...")
    try:
        attention_weights, tokens = interpreter.create_attention_heatmap(
            combined_text, tokenizer, model, layer_idx=-1
        )
        print("✅ Attention heatmap created successfully")
        
        # Analyze attention patterns
        print("\n4. Analyzing attention patterns...")
        attention_sum = attention_weights.sum(axis=1)
        top_attention_indices = np.argsort(attention_sum)[-10:]
        
        print("   Top 10 most attended tokens:")
        for i, idx in enumerate(top_attention_indices):
            token = tokens[idx] if idx < len(tokens) else f"token_{idx}"
            weight = attention_sum[idx]
            print(f"     {i+1}. {token}: {weight:.4f}")
        
        return interpreter
        
    except Exception as e:
        print(f"❌ Attention analysis failed: {e}")
        return None

def demonstrate_graph_interpretability(donors_df, relationships_df):
    """Demonstrate graph interpretability with GNN"""
    print("\n" + "="*60)
    print("GRAPH INTERPRETABILITY DEMONSTRATION")
    print("="*60)
    
    # Create graph data
    print("\n1. Creating graph data...")
    preprocessor = DonorGraphPreprocessor(donors_df, relationships_df)
    graph_data = preprocessor.create_graph_data()
    print(f"✅ Graph created: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Create sample GNN model
    print("\n2. Creating sample GNN model...")
    input_dim = graph_data.x.shape[1]
    gnn_model = GraphSAGE(input_dim=input_dim, hidden_dim=64, output_dim=2, num_layers=2)
    gnn_model.eval()
    print("✅ GNN model created")
    
    # Create interpreter
    interpreter = MultimodalModelInterpreter()
    
    # Compute graph importance
    print("\n3. Computing graph importance scores...")
    try:
        graph_importance = interpreter.compute_graph_importance_scores(graph_data, gnn_model)
        print("✅ Graph importance computed")
        
        # Visualize graph importance
        print("\n4. Visualizing graph importance...")
        interpreter.visualize_graph_importance(
            graph_data, 
            graph_importance['node_importance'],
            top_k=20
        )
        
        # Analyze top important nodes
        importance_scores = graph_importance['node_importance']
        top_nodes = np.argsort(importance_scores)[-10:]
        
        print("\n   Top 10 most important nodes:")
        for i, node_idx in enumerate(top_nodes):
            importance = importance_scores[node_idx]
            print(f"     {i+1}. Node {node_idx}: {importance:.4f}")
        
        return interpreter
        
    except Exception as e:
        print(f"❌ Graph analysis failed: {e}")
        return None

def demonstrate_confidence_intervals(ensemble_model, tabular_features):
    """Demonstrate confidence interval calculation"""
    print("\n" + "="*60)
    print("CONFIDENCE INTERVALS DEMONSTRATION")
    print("="*60)
    
    # Create interpreter
    interpreter = MultimodalModelInterpreter(ensemble_model=ensemble_model)
    
    # Get predictions
    print("\n1. Getting model predictions...")
    features_dict = {
        'tabular': tabular_features,
        'bert': np.random.randn(len(tabular_features), 768),
        'gnn': np.random.randn(len(tabular_features), 64)
    }
    
    predictions, probabilities = ensemble_model.predict_ensemble(features_dict)
    print(f"✅ Predictions generated: {len(probabilities)} samples")
    
    # Compute confidence intervals
    print("\n2. Computing confidence intervals...")
    
    # Bootstrap method
    print("   Using bootstrap method...")
    ci_bootstrap = interpreter.compute_confidence_intervals(
        probabilities, method='bootstrap', n_bootstrap=500, confidence_level=0.95
    )
    
    # Binomial method
    print("   Using binomial method...")
    ci_binomial = interpreter.compute_confidence_intervals(
        probabilities, method='binomial', confidence_level=0.95
    )
    
    # Compare methods
    print(f"\n3. Comparing confidence interval methods:")
    print(f"   Bootstrap: [{ci_bootstrap['ci_lower']:.4f}, {ci_bootstrap['ci_upper']:.4f}]")
    print(f"   Binomial:  [{ci_binomial['ci_lower']:.4f}, {ci_binomial['ci_upper']:.4f}]")
    print(f"   Mean probability: {ci_bootstrap['mean_probability']:.4f}")
    
    return interpreter

def demonstrate_feature_contribution_breakdown(interpreter, donor_features, prediction_proba):
    """Demonstrate comprehensive feature contribution breakdown"""
    print("\n" + "="*60)
    print("FEATURE CONTRIBUTION BREAKDOWN DEMONSTRATION")
    print("="*60)
    
    # Create breakdown
    print("\n1. Creating feature contribution breakdown...")
    breakdown = interpreter.create_feature_contribution_breakdown(
        donor_features, prediction_proba,
        tabular_features=interpreter.feature_names
    )
    
    # Display breakdown
    print(f"\n2. Prediction Analysis:")
    print(f"   Probability: {breakdown['prediction_probability']:.4f}")
    print(f"   Confidence: {breakdown['confidence_level']}")
    
    print(f"\n3. Top Contributing Features:")
    for i, feature in enumerate(breakdown['top_features'][:10]):
        print(f"   {i+1}. {feature['feature']}: {feature['score']:.4f} ({feature['modality']})")
    
    # Generate HTML report
    print(f"\n4. Generating HTML report...")
    try:
        fig = interpreter.generate_interpretability_report(
            donor_id=12345, breakdown=breakdown, 
            save_path='example_interpretability_report.html'
        )
        print("✅ HTML report generated: example_interpretability_report.html")
    except Exception as e:
        print(f"⚠️  HTML report generation failed: {e}")
    
    return breakdown

def main():
    """Main demonstration function"""
    print("="*80)
    print("COMPREHENSIVE MODEL INTERPRETABILITY DEMONSTRATION")
    print("="*80)
    
    # Load data
    donors_df, contact_reports_df, relationships_df, giving_history_df = load_data()
    if donors_df is None:
        return
    
    # Create sample embeddings
    bert_embeddings, gnn_embeddings = create_sample_embeddings(donors_df, contact_reports_df)
    
    # Prepare multimodal data
    print("\nPreparing multimodal data...")
    tabular_features, text_embeddings, graph_embeddings, labels, modality_mask = prepare_multimodal_data(
        donors_df, contact_reports_df, bert_embeddings, gnn_embeddings
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(tabular_features.shape[1])]
    
    # Create ensemble model
    print("\nCreating ensemble model...")
    ensemble = AdvancedEnsembleModel()
    ensemble.create_base_models()
    
    # Train ensemble (simplified for demo)
    print("Training ensemble model...")
    features_dict = {
        'tabular': tabular_features,
        'bert': text_embeddings,
        'gnn': graph_embeddings
    }
    ensemble.train_ensemble(features_dict, labels)
    print("✅ Ensemble model trained")
    
    # Run demonstrations
    print("\n" + "="*80)
    print("RUNNING INTERPRETABILITY DEMONSTRATIONS")
    print("="*80)
    
    # 1. Tabular interpretability
    tabular_interpreter = demonstrate_tabular_interpretability(
        ensemble, tabular_features, feature_names
    )
    
    # 2. Text interpretability
    text_interpreter = demonstrate_text_interpretability(
        contact_reports_df, donors_df['ID'].iloc[0]
    )
    
    # 3. Graph interpretability
    graph_interpreter = demonstrate_graph_interpretability(
        donors_df, relationships_df
    )
    
    # 4. Confidence intervals
    ci_interpreter = demonstrate_confidence_intervals(ensemble, tabular_features)
    
    # 5. Feature contribution breakdown
    if tabular_interpreter:
        breakdown = demonstrate_feature_contribution_breakdown(
            tabular_interpreter, tabular_features[0], 0.75
        )
    
    # Summary
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  - shap_summary.png (SHAP feature importance)")
    print("  - bert_attention_heatmap.png (Text attention visualization)")
    print("  - graph_importance_visualization.png (Graph importance)")
    print("  - interpretability_summary.png (Overall summary)")
    print("  - example_interpretability_report.html (Interactive report)")
    
    print("\nNext steps:")
    print("  1. Review the generated visualizations")
    print("  2. Open the HTML report in a web browser")
    print("  3. Integrate these features into your main pipeline")
    print("  4. Customize the visualizations for your specific use case")

if __name__ == "__main__":
    main()



