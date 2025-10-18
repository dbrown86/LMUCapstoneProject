#!/usr/bin/env python3
"""
Create All Visualizations for Capstone Presentation
Generates publication-quality figures for your presentation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def create_model_comparison():
    """Visualization 1: Model Performance Comparison"""
    print("Creating Model Comparison visualization...")
    
    # Data (UPDATE these with your actual results from run_enhanced_with_real_embeddings.py)
    models = ['GraphSAGE', 'GCN', 'Enhanced\nEnsemble']
    
    # Original GNN results
    accuracy = [0.719, 0.703, 0.XXX]  # UPDATE: Fill with your enhanced result
    auc = [0.726, 0.716, 0.XXX]       # UPDATE: Fill with your enhanced result
    recall = [0.67, 0.64, 0.XXX]      # UPDATE: Fill with your enhanced result  
    precision = [0.39, 0.37, 0.XXX]   # UPDATE: Fill with your enhanced result
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    bars1 = axes[0, 0].bar(models, accuracy, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].axhline(y=0.70, color='gray', linestyle='--', alpha=0.3, label='Target: 70%')
    axes[0, 0].legend()
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1%}', ha='center', va='bottom', fontsize=10)
    
    # AUC-ROC
    bars2 = axes[0, 1].bar(models, auc, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
    axes[0, 1].set_ylabel('AUC-ROC', fontsize=12)
    axes[0, 1].set_title('Model AUC-ROC Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].axhline(y=0.70, color='gray', linestyle='--', alpha=0.3, label='Target: 0.70')
    axes[0, 1].legend()
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Recall
    bars3 = axes[1, 0].bar(models, recall, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
    axes[1, 0].set_ylabel('Recall', fontsize=12)
    axes[1, 0].set_title('Model Recall Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].axhline(y=0.60, color='gray', linestyle='--', alpha=0.3, label='Target: 60%')
    axes[1, 0].legend()
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1%}', ha='center', va='bottom', fontsize=10)
    
    # Precision
    bars4 = axes[1, 1].bar(models, precision, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
    axes[1, 1].set_ylabel('Precision', fontsize=12)
    axes[1, 1].set_title('Model Precision Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].axhline(y=0.40, color='gray', linestyle='--', alpha=0.3, label='Target: 40%')
    axes[1, 1].legend()
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1%}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Multimodal Deep Learning Model Performance', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: figures/model_comparison.png")

def create_smote_comparison():
    """Visualization 2: SMOTE Method Comparison"""
    print("Creating SMOTE Comparison visualization...")
    
    methods = ['Original', 'SMOTE', 'Borderline\nSMOTE', 'ADASYN', 'SMOTEENN', 'SMOTE\nTomek']
    f1_scores = [0.2851, 0.3595, 0.3577, 0.3611, 0.4565, 0.3696]
    precision = [0.5108, 0.4414, 0.4401, 0.4453, 0.4166, 0.4512]
    recall = [0.1977, 0.3032, 0.3013, 0.3037, 0.5049, 0.3131]
    
    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Class Imbalance Handling: SMOTE Method Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 0.6])
    ax.axhline(y=0.4, color='gray', linestyle='--', alpha=0.3)
    
    # Annotate best method
    best_idx = f1_scores.index(max(f1_scores))
    ax.annotate('BEST', xy=(best_idx, f1_scores[best_idx] + 0.02),
                ha='center', fontsize=12, fontweight='bold', color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig('figures/smote_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: figures/smote_comparison.png")

def create_confusion_matrices():
    """Visualization 3: Confusion Matrix"""
    print("Creating Confusion Matrix visualization...")
    
    # Your actual confusion matrix (UPDATE with results)
    cm = np.array([[4670, 3292],
                   [695, 1343]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=['No Legacy Intent', 'Legacy Intent'],
                yticklabels=['No Legacy Intent', 'Legacy Intent'],
                ax=ax, annot_kws={'size': 14})
    
    ax.set_title('Confusion Matrix - Enhanced Multimodal Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    # Add annotations
    total = cm.sum()
    accuracy = (cm[0,0] + cm[1,1]) / total
    ax.text(0.5, -0.15, f'Accuracy: {accuracy:.1%}', 
            ha='center', transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: figures/confusion_matrix.png")

def create_architecture_diagram():
    """Visualization 4: Model Architecture Overview"""
    print("Creating Architecture Diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Multimodal Deep Learning Architecture', 
            ha='center', fontsize=18, fontweight='bold')
    
    # Layer 1: Data Sources
    ax.text(0.15, 0.85, 'Demographics\n(Tabular Data)', ha='center', 
            bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.6), fontsize=10)
    ax.text(0.5, 0.85, 'Contact Reports\n(Text Data)', ha='center',
            bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.6), fontsize=10)
    ax.text(0.85, 0.85, 'Family Networks\n(Graph Data)', ha='center',
            bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.6), fontsize=10)
    
    # Layer 2: Models
    ax.text(0.15, 0.65, 'Feature\nEngineering\n(50+ features)', ha='center',
            bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.4), fontsize=9)
    ax.text(0.5, 0.65, 'BERT-base\n(110M params)\n768-dim', ha='center',
            bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.4), fontsize=9)
    ax.text(0.85, 0.65, 'GraphSAGE/GCN\n(3 layers)\n64-dim', ha='center',
            bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.4), fontsize=9)
    
    # Layer 3: Integration
    ax.text(0.5, 0.45, 'Multimodal Feature Fusion\n(844 dimensions)', ha='center',
            bbox=dict(boxstyle='round', facecolor='#9b59b6', alpha=0.6), fontsize=11, fontweight='bold')
    
    # Layer 4: Class Imbalance
    ax.text(0.5, 0.30, 'SMOTEENN Class Balancing', ha='center',
            bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.6), fontsize=10)
    
    # Layer 5: Ensemble
    ax.text(0.5, 0.15, 'Ensemble Model (5 base models + meta-learner)', ha='center',
            bbox=dict(boxstyle='round', facecolor='#34495e', alpha=0.6), fontsize=11, fontweight='bold', color='white')
    
    # Layer 6: Output
    ax.text(0.5, 0.02, 'Legacy Intent Prediction\n(with confidence scores)', ha='center',
            bbox=dict(boxstyle='round', facecolor='#16a085', alpha=0.6), fontsize=10, color='white')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='gray', alpha=0.5)
    
    for x_pos in [0.15, 0.5, 0.85]:
        ax.annotate('', xy=(x_pos, 0.70), xytext=(x_pos, 0.80),
                   arrowprops=arrow_props)
    
    ax.annotate('', xy=(0.5, 0.50), xytext=(0.15, 0.62), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.50), xytext=(0.5, 0.62), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.50), xytext=(0.85, 0.62), arrowprops=arrow_props)
    
    ax.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.42), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.20), xytext=(0.5, 0.27), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.07), xytext=(0.5, 0.12), arrowprops=arrow_props)
    
    plt.tight_layout()
    plt.savefig('figures/architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: figures/architecture_diagram.png")

def create_confusion_matrices():
    """Visualization 3: Confusion Matrix"""
    print("Creating Confusion Matrix visualization...")
    
    # Your actual confusion matrix (UPDATE with results from run_enhanced_with_real_embeddings.py)
    cm = np.array([[4670, 3292],
                   [695, 1343]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=['No Legacy Intent', 'Legacy Intent'],
                yticklabels=['No Legacy Intent', 'Legacy Intent'],
                ax=ax, annot_kws={'size': 14, 'fontweight': 'bold'})
    
    ax.set_title('Enhanced Model Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    # Add metrics annotation
    total = cm.sum()
    accuracy = (cm[0,0] + cm[1,1]) / total
    precision = cm[1,1] / (cm[0,1] + cm[1,1])
    recall = cm[1,1] / (cm[1,0] + cm[1,1])
    
    metrics_text = f'Accuracy: {accuracy:.1%}\nPrecision: {precision:.1%}\nRecall: {recall:.1%}'
    ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: figures/confusion_matrix.png")

def create_class_distribution():
    """Visualization 5: Class Distribution"""
    print("Creating Class Distribution visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before SMOTE
    classes = ['No Legacy Intent\n(79.6%)', 'Legacy Intent\n(20.4%)']
    counts = [39810, 10190]
    colors = ['#3498db', '#e74c3c']
    
    axes[0].bar(classes, counts, color=colors, alpha=0.7)
    axes[0].set_ylabel('Number of Donors', fontsize=12)
    axes[0].set_title('Original Class Distribution\n(Imbalanced: 3.91:1 ratio)', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 45000])
    
    for i, (cls, count) in enumerate(zip(classes, counts)):
        axes[0].text(i, count + 1000, f'{count:,}', ha='center', fontsize=11, fontweight='bold')
    
    # After SMOTEENN
    classes_after = ['No Legacy Intent\n(44%)','Legacy Intent\n(56%)']
    counts_after = [10498, 13416]
    
    axes[1].bar(classes_after, counts_after, color=colors, alpha=0.7)
    axes[1].set_ylabel('Number of Samples', fontsize=12)
    axes[1].set_title('After SMOTEENN Balancing\n(Training Set Only)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 15000])
    
    for i, (cls, count) in enumerate(zip(classes_after, counts_after)):
        axes[1].text(i, count + 300, f'{count:,}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: figures/class_distribution.png")

def create_feature_engineering_summary():
    """Visualization 6: Feature Engineering Pipeline"""
    print("Creating Feature Engineering Summary...")
    
    stages = ['Original\nFeatures', 'Enhanced\nFeatures', 'With\nEmbeddings', 
              'After\nSelection', 'After\nPCA']
    dimensions = [29, 50, 844, 100, 50]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#e74c3c']
    bars = ax.bar(stages, dimensions, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title('Feature Engineering Pipeline', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 900])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 20,
               f'{int(height)}', ha='center', fontsize=12, fontweight='bold')
    
    # Add descriptions
    descriptions = [
        'Donor data',
        '+Demographics\n+Giving patterns\n+Engagement',
        '+BERT (768)\n+GNN (64)',
        'Mutual info\nselection',
        'PCA reduction\n(100% variance)'
    ]
    
    for i, (stage, desc) in enumerate(zip(stages, descriptions)):
        ax.text(i, -100, desc, ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig('figures/feature_engineering.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: figures/feature_engineering.png")

def main():
    """Create all visualizations"""
    print("="*80)
    print("CREATING CAPSTONE VISUALIZATIONS")
    print("="*80)
    
    # Create figures directory
    import os
    os.makedirs('figures', exist_ok=True)
    print("Created figures/ directory\n")
    
    # Create all visualizations
    create_model_comparison()
    create_smote_comparison()
    create_confusion_matrices()
    create_class_distribution()
    create_feature_engineering_summary()
    create_architecture_diagram()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS CREATED!")
    print("="*80)
    print("\nGenerated files in figures/:")
    print("  1. model_comparison.png - Performance across models")
    print("  2. smote_comparison.png - Class imbalance methods")
    print("  3. confusion_matrix.png - Prediction results")
    print("  4. class_distribution.png - Before/after balancing")
    print("  5. feature_engineering.png - Feature pipeline")
    print("  6. architecture_diagram.png - System architecture")
    
    print("\n📝 NOTE: Update the model_comparison.png with your actual results")
    print("   after running run_enhanced_with_real_embeddings.py")
    
    print("\n✅ Ready for capstone presentation!")

if __name__ == "__main__":
    main()

