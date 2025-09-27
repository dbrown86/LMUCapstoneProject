# GNN analysis and visualization functions
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import k_hop_subgraph
from .gnn_models import DonorGraphPreprocessor, FeatureEncoder

def get_node_embeddings(model, data, device):
    """Extract node embeddings from trained model"""
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        # Get embeddings from second-to-last layer
        x = data.x
        edge_index = data.edge_index
        
        # Forward pass through convolution layers only
        for i, (conv, bn) in enumerate(zip(model.convs[:-1], model.batch_norms[:-1])):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        
        # Final convolution without classifier
        x = model.convs[-1](x, edge_index)
        x = model.batch_norms[-1](x)
        
    return x.cpu().numpy()

def visualize_embeddings(embeddings, labels, donors_df):
    """Visualize node embeddings using t-SNE"""
    print("Creating embedding visualization...")
    
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot by legacy intent
    colors = ['red', 'blue']
    for i, label in enumerate(['No Legacy Intent', 'Legacy Intent']):
        mask = labels == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=colors[i], label=label, alpha=0.6, s=20)
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Donor Node Embeddings Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot by family status
    plt.figure(figsize=(12, 8))
    
    family_mask = donors_df['Family_ID'].notna()
    plt.scatter(embeddings_2d[~family_mask, 0], embeddings_2d[~family_mask, 1], 
               c='gray', label='Individual Donors', alpha=0.6, s=20)
    plt.scatter(embeddings_2d[family_mask, 0], embeddings_2d[family_mask, 1], 
               c='green', label='Family Members', alpha=0.6, s=20)
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Donor Embeddings by Family Status')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_feature_importance(model, data, preprocessor, device):
    """Analyze feature importance using gradient-based methods"""
    print("Analyzing feature importance...")
    
    model.eval()
    data = data.to(device)
    
    # Enable gradients for input features
    data.x.requires_grad_(True)
    
    # Forward pass
    output = model(data.x, data.edge_index)
    
    # Calculate gradients for positive class
    positive_class_output = output[:, 1].sum()
    positive_class_output.backward()
    
    # Get feature importance (average absolute gradient)
    feature_importance = torch.abs(data.x.grad).mean(dim=0).cpu().numpy()
    
    # Get feature names (simplified)
    feature_names = ['Lifetime_Giving', 'Last_Gift', 'Consecutive_Years', 'Total_Years',
                    'Engagement_Score', 'Legacy_Probability', 'Age', 'Rating', 'Constituent_Type',
                    'Stage', 'Region'] + [f'Feature_{i}' for i in range(len(feature_importance) - 11)]
    
    if len(feature_names) != len(feature_importance):
        feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(feature_importance)],
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Plot top features
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance (Average Absolute Gradient)')
    plt.title('Top 15 Most Important Features for Legacy Intent Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print("Top 10 Most Important Features:")
    print(importance_df.head(10))

def analyze_family_networks(graph_data, embeddings, donors_df, relationships_df):
    """Analyze family network patterns"""
    print("Analyzing family network patterns...")
    
    if relationships_df.empty:
        print("No family relationships to analyze")
        return
    
    # Calculate family-level statistics
    family_stats = []
    
    for family_id in relationships_df['Family_ID'].unique():
        family_members = relationships_df[relationships_df['Family_ID'] == family_id]['Donor_ID'].values
        
        # Get family member indices
        member_indices = []
        for donor_id in family_members:
            try:
                idx = donors_df[donors_df['ID'] == donor_id].index[0]
                member_indices.append(idx)
            except:
                continue
        
        if len(member_indices) < 2:
            continue
        
        # Calculate family statistics
        family_donors_df = donors_df.iloc[member_indices]
        
        stats = {
            'Family_ID': family_id,
            'Size': len(member_indices),
            'Total_Giving': family_donors_df['Lifetime_Giving'].sum(),
            'Avg_Giving': family_donors_df['Lifetime_Giving'].mean(),
            'Legacy_Intent_Rate': family_donors_df['Legacy_Intent_Binary'].mean(),
            'Avg_Engagement': family_donors_df['Engagement_Score'].mean(),
            'Embedding_Similarity': calculate_family_embedding_similarity(embeddings, member_indices)
        }
        
        family_stats.append(stats)
    
    family_df = pd.DataFrame(family_stats)
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Family size distribution
    axes[0,0].hist(family_df['Size'], bins=10, alpha=0.7, color='skyblue')
    axes[0,0].set_xlabel('Family Size')
    axes[0,0].set_ylabel('Number of Families')
    axes[0,0].set_title('Family Size Distribution')
    
    # Total giving vs family size
    axes[0,1].scatter(family_df['Size'], family_df['Total_Giving'], alpha=0.6)
    axes[0,1].set_xlabel('Family Size')
    axes[0,1].set_ylabel('Total Family Giving ($)')
    axes[0,1].set_title('Family Giving vs Size')
    axes[0,1].set_yscale('log')
    
    # Legacy intent rate distribution
    axes[1,0].hist(family_df['Legacy_Intent_Rate'], bins=10, alpha=0.7, color='lightgreen')
    axes[1,0].set_xlabel('Legacy Intent Rate')
    axes[1,0].set_ylabel('Number of Families')
    axes[1,0].set_title('Family Legacy Intent Rate Distribution')
    
    # Embedding similarity vs legacy intent rate
    axes[1,1].scatter(family_df['Embedding_Similarity'], family_df['Legacy_Intent_Rate'], alpha=0.6)
    axes[1,1].set_xlabel('Family Embedding Similarity')
    axes[1,1].set_ylabel('Legacy Intent Rate')
    axes[1,1].set_title('Embedding Similarity vs Legacy Intent')
    
    plt.tight_layout()
    plt.show()
    
    print("Family Network Analysis Summary:")
    print(f"Number of families analyzed: {len(family_df)}")
    print(f"Average family size: {family_df['Size'].mean():.2f}")
    print(f"Average family giving: ${family_df['Total_Giving'].mean():,.2f}")
    print(f"Average legacy intent rate: {family_df['Legacy_Intent_Rate'].mean():.3f}")
    print(f"Average embedding similarity: {family_df['Embedding_Similarity'].mean():.3f}")

def calculate_family_embedding_similarity(embeddings, member_indices):
    """Calculate average pairwise cosine similarity within family"""
    if len(member_indices) < 2:
        return 0.0
    
    family_embeddings = embeddings[member_indices]
    similarity_matrix = cosine_similarity(family_embeddings)
    
    # Get upper triangular part (excluding diagonal)
    upper_triangular = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    return upper_triangular.mean()

class GNNExplainer:
    """Explainability tools for GNN models"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def explain_node_prediction(self, data, node_idx, num_hops=2):
        """Explain prediction for a specific node using attention weights"""
        self.model.eval()
        data = data.to(self.device)
        
        # Get k-hop neighborhood
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, num_hops, data.edge_index, relabel_nodes=True
        )
        
        # Extract subgraph
        x_sub = data.x[subset]
        y_sub = data.y[subset] if data.y is not None else None
        
        # Get prediction
        with torch.no_grad():
            output = self.model(x_sub, edge_index)
            prediction = output[mapping].argmax().item()
            confidence = F.softmax(output[mapping], dim=0).max().item()
        
        return {
            'node_idx': node_idx,
            'prediction': prediction,
            'confidence': confidence,
            'neighbors': subset.tolist(),
            'subgraph_size': len(subset)
        }
    
    def explain_prediction_patterns(self, data, predictions, donors_df):
        """Analyze prediction patterns across different donor groups"""
        data_cpu = data.to('cpu')
        results = {}
        
        # By family status
        family_mask = torch.tensor([pd.notna(val) for val in donors_df['Family_ID'].values])
        
        results['family_members'] = {
            'accuracy': (predictions[family_mask] == data_cpu.y[family_mask]).float().mean().item(),
            'positive_rate': (predictions[family_mask] == 1).float().mean().item(),
            'count': family_mask.sum().item()
        }
        
        results['individual_donors'] = {
            'accuracy': (predictions[~family_mask] == data_cpu.y[~family_mask]).float().mean().item(),
            'positive_rate': (predictions[~family_mask] == 1).float().mean().item(),
            'count': (~family_mask).sum().item()
        }
        
        return results

def hyperparameter_optimization(graph_data, device, n_trials=50):
    """Optimize hyperparameters using Optuna"""
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Run: pip install optuna")
        return None
    
    from .gnn_models import GraphSAGE, DonorGNNTrainer
    
    def objective(trial):
        # Hyperparameters to optimize
        hidden_dim = trial.suggest_int('hidden_dim', 32, 128, step=32)
        num_layers = trial.suggest_int('num_layers', 2, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        # Create model
        input_dim = graph_data.x.shape[1]
        model = GraphSAGE(input_dim, hidden_dim, 2, num_layers, dropout)
        
        # Train model
        trainer = DonorGNNTrainer(model, device, lr, weight_decay)
        results = trainer.train(graph_data, epochs=100, early_stopping_patience=10)
        
        return results['best_val_acc']
    
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Best hyperparameters:")
    print(study.best_params)
    print(f"Best validation accuracy: {study.best_value:.4f}")
    
    return study.best_params
