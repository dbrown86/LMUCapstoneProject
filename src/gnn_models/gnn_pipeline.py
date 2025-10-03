# Main GNN pipeline for donor analysis
import torch
import pandas as pd
import numpy as np
from .gnn_models import DonorGraphPreprocessor, FeatureEncoder, GraphSAGE, GCNModel, DonorGNNTrainer
from .gnn_analysis import get_node_embeddings, visualize_embeddings, analyze_feature_importance, analyze_family_networks

def setup_environment():
    """Set up PyTorch Geometric environment and check GPU availability"""
    print("=" * 60)
    print("STEP 1: PYTORCH GEOMETRIC ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Check PyTorch version and GPU availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Device set to: {device}")
    return device

def main_gnn_pipeline(donors_df, relationships_df, contact_reports_df=None, giving_history_df=None):
    """Complete GNN pipeline execution"""
    
    print("=" * 80)
    print("DONOR GRAPH NEURAL NETWORK PIPELINE")
    print("=" * 80)
    
    # Step 1: Setup environment
    device = setup_environment()
    
    # Step 2: Preprocess data and create graph
    print("\n" + "=" * 60)
    print("STEP 2: GRAPH DATA PREPROCESSING")
    print("=" * 60)
    
    preprocessor = DonorGraphPreprocessor(
        donors_df, relationships_df, contact_reports_df, giving_history_df
    )
    
    # Create enhanced features
    feature_encoder = FeatureEncoder()
    
    # Encode additional features if available
    if contact_reports_df is not None and not contact_reports_df.empty:
        contact_features = feature_encoder.encode_contact_reports(contact_reports_df, donors_df)
        # Add to donors_df for preprocessing
        for i, features in enumerate(contact_features):
            for j, feature in enumerate(features):
                donors_df.loc[i, f'contact_feature_{j}'] = feature
    
    if giving_history_df is not None and not giving_history_df.empty:
        giving_features = feature_encoder.encode_giving_patterns(giving_history_df, donors_df)
        # Add to donors_df for preprocessing
        for i, features in enumerate(giving_features):
            for j, feature in enumerate(features):
                donors_df.loc[i, f'giving_pattern_{j}'] = feature
    
    # Update preprocessor with enhanced data
    preprocessor.donors_df = donors_df
    
    # Create graph data
    graph_data = preprocessor.create_graph_data(target_column='Legacy_Intent_Binary')
    
    # Step 3: Initialize models
    print("\n" + "=" * 60)
    print("STEP 3: MODEL INITIALIZATION")
    print("=" * 60)
    
    input_dim = graph_data.x.shape[1]
    hidden_dim = 64
    output_dim = 2  # Binary classification
    
    # GraphSAGE model
    sage_model = GraphSAGE(input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.3)
    
    # GCN model
    gcn_model = GCNModel(input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.3)
    
    print(f"Models initialized with input_dim={input_dim}, hidden_dim={hidden_dim}")
    
    # Step 4: Train GraphSAGE with class imbalance handling
    print("\n" + "=" * 60)
    print("STEP 4: GRAPHSAGE TRAINING (WITH CLASS IMBALANCE HANDLING)")
    print("=" * 60)
    
    sage_trainer = DonorGNNTrainer(sage_model, device, lr=0.01, weight_decay=5e-4, auto_weights=True)
    sage_results = sage_trainer.train(graph_data, epochs=200, early_stopping_patience=20, stratify=True)
    
    print("\nGraphSAGE Training Curves:")
    sage_trainer.plot_training_curves()
    
    # Step 5: Train GCN with class imbalance handling
    print("\n" + "=" * 60)
    print("STEP 5: GCN TRAINING (WITH CLASS IMBALANCE HANDLING)")
    print("=" * 60)
    
    gcn_trainer = DonorGNNTrainer(gcn_model, device, lr=0.01, weight_decay=5e-4, auto_weights=True)
    gcn_results = gcn_trainer.train(graph_data, epochs=200, early_stopping_patience=20, stratify=True)
    
    print("\nGCN Training Curves:")
    gcn_trainer.plot_training_curves()
    
    # Step 6: Model comparison
    print("\n" + "=" * 60)
    print("STEP 6: MODEL COMPARISON")
    print("=" * 60)
    
    comparison_df = pd.DataFrame({
        'Model': ['GraphSAGE', 'GCN'],
        'Test_Accuracy': [sage_results['test_acc'], gcn_results['test_acc']],
        'Test_AUC': [sage_results['test_auc'], gcn_results['test_auc']],
        'Test_F1': [sage_results['test_f1'], gcn_results['test_f1']],
        'Best_Val_AUC': [sage_results['best_val_auc'], gcn_results['best_val_auc']]
    })
    
    print("Model Performance Comparison:")
    print(comparison_df.round(4))
    
    # Step 7: Advanced analysis and interpretability
    print("\n" + "=" * 60)
    print("STEP 7: ADVANCED ANALYSIS")
    print("=" * 60)
    
    # Node embedding analysis - use AUC instead of accuracy for model selection
    best_model = sage_trainer.model if sage_results['test_auc'] > gcn_results['test_auc'] else gcn_trainer.model
    best_trainer = sage_trainer if sage_results['test_auc'] > gcn_results['test_auc'] else gcn_trainer
    
    embeddings = get_node_embeddings(best_model, graph_data, device)
    visualize_embeddings(embeddings, graph_data.y.cpu().numpy(), donors_df)
    
    # Feature importance analysis
    analyze_feature_importance(best_model, graph_data, preprocessor, device)
    
    # Family network analysis
    analyze_family_networks(graph_data, embeddings, donors_df, relationships_df)
    
    return {
        'sage_results': sage_results,
        'gcn_results': gcn_results,
        'graph_data': graph_data,
        'preprocessor': preprocessor,
        'best_model': best_model,
        'embeddings': embeddings
    }

def example_usage():
    """Example of how to use the GNN implementation"""
    
    print("Loading synthetic donor dataset...")
    
    # Load your dataset (replace with actual file paths)
    donors_df = pd.read_csv('synthetic_donor_dataset/donors.csv')
    relationships_df = pd.read_csv('synthetic_donor_dataset/relationships.csv')
    
    # Optional: load additional data
    try:
        contact_reports_df = pd.read_csv('synthetic_donor_dataset/contact_reports.csv')
        giving_history_df = pd.read_csv('synthetic_donor_dataset/giving_history.csv')
    except:
        contact_reports_df = pd.DataFrame()
        giving_history_df = pd.DataFrame()
    
    print(f"Loaded {len(donors_df):,} donors with {len(relationships_df):,} family relationships")
    
    # Run complete GNN pipeline
    results = main_gnn_pipeline(
        donors_df, 
        relationships_df, 
        contact_reports_df, 
        giving_history_df
    )
    
    # Optional: Run hyperparameter optimization
    # best_params = hyperparameter_optimization(results['graph_data'], device, n_trials=20)
    
    print("\nGNN Pipeline completed successfully!")
    return results

# Run the example if this script is executed directly
if __name__ == "__main__":
    # Install required packages first
    print("Make sure you have installed the required packages:")
    print("pip install torch torch-geometric scikit-learn matplotlib seaborn networkx")
    print("pip install optuna  # Optional for hyperparameter optimization")
    print()
    
    # Uncomment to run the example
    # results = example_usage()
