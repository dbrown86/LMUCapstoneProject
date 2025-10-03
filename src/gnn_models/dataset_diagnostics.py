# Dataset diagnostics for class imbalance analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DatasetDiagnostics:
    """Comprehensive dataset diagnostic tools for class imbalance detection"""
    
    def __init__(self, donors_df, relationships_df=None):
        self.donors_df = donors_df.copy()
        self.relationships_df = relationships_df
        self.target_column = 'Legacy_Intent_Binary'
        
    def analyze_class_distribution(self):
        """Analyze the distribution of target classes"""
        print("=" * 60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        if self.target_column not in self.donors_df.columns:
            print(f"Error: Target column '{self.target_column}' not found in dataset")
            return None
            
        # Basic statistics
        class_counts = self.donors_df[self.target_column].value_counts()
        total_samples = len(self.donors_df)
        
        print(f"Total samples: {total_samples:,}")
        print(f"Target column: {self.target_column}")
        print("\nClass Distribution:")
        print("-" * 40)
        
        for class_label, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"Class {class_label}: {count:,} samples ({percentage:.2f}%)")
        
        # Calculate imbalance ratio
        minority_class = class_counts.min()
        majority_class = class_counts.max()
        imbalance_ratio = majority_class / minority_class
        
        print(f"\nClass Imbalance Metrics:")
        print(f"Imbalance Ratio: {imbalance_ratio:.2f}:1")
        print(f"Minority Class Size: {minority_class:,}")
        print(f"Majority Class Size: {majority_class:,}")
        
        # Severity assessment
        if imbalance_ratio < 2:
            severity = "Balanced"
        elif imbalance_ratio < 10:
            severity = "Mild Imbalance"
        elif imbalance_ratio < 100:
            severity = "Moderate Imbalance"
        else:
            severity = "Severe Imbalance"
            
        print(f"Imbalance Severity: {severity}")
        
        return {
            'class_counts': class_counts,
            'total_samples': total_samples,
            'imbalance_ratio': imbalance_ratio,
            'severity': severity
        }
    
    def visualize_class_distribution(self):
        """Create visualizations for class distribution"""
        if self.target_column not in self.donors_df.columns:
            print(f"Error: Target column '{self.target_column}' not found")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Class Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Bar plot
        class_counts = self.donors_df[self.target_column].value_counts()
        axes[0, 0].bar(class_counts.index, class_counts.values, 
                      color=['#ff9999', '#66b3ff'], alpha=0.7)
        axes[0, 0].set_title('Class Counts')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Count')
        
        # Add value labels on bars
        for i, v in enumerate(class_counts.values):
            axes[0, 0].text(class_counts.index[i], v + max(class_counts.values)*0.01, 
                           f'{v:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Pie chart
        axes[0, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                      colors=['#ff9999', '#66b3ff'], startangle=90)
        axes[0, 1].set_title('Class Proportions')
        
        # 3. Log scale bar plot (if severe imbalance)
        axes[1, 0].bar(class_counts.index, class_counts.values, 
                      color=['#ff9999', '#66b3ff'], alpha=0.7)
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_title('Class Counts (Log Scale)')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Count (log scale)')
        
        # 4. Cumulative distribution
        sorted_counts = class_counts.sort_values(ascending=False)
        cumulative_pct = (sorted_counts.cumsum() / sorted_counts.sum() * 100)
        
        axes[1, 1].bar(range(len(cumulative_pct)), cumulative_pct, 
                      color=['#ff9999', '#66b3ff'], alpha=0.7)
        axes[1, 1].set_title('Cumulative Class Distribution')
        axes[1, 1].set_xlabel('Class Rank')
        axes[1, 1].set_ylabel('Cumulative Percentage')
        axes[1, 1].set_xticks(range(len(cumulative_pct)))
        axes[1, 1].set_xticklabels(sorted_counts.index)
        
        plt.tight_layout()
        plt.show()
        
    def analyze_train_test_split_balance(self, test_size=0.2, random_state=42):
        """Analyze class distribution in train/test splits"""
        print("\n" + "=" * 60)
        print("TRAIN/TEST SPLIT BALANCE ANALYSIS")
        print("=" * 60)
        
        if self.target_column not in self.donors_df.columns:
            print(f"Error: Target column '{self.target_column}' not found")
            return
            
        # Create splits
        X = self.donors_df.drop(columns=[self.target_column])
        y = self.donors_df[self.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Analyze distributions
        train_counts = y_train.value_counts()
        test_counts = y_test.value_counts()
        
        print(f"Train set: {len(y_train):,} samples")
        print(f"Test set: {len(y_test):,} samples")
        
        print(f"\nTrain Set Distribution:")
        for class_label, count in train_counts.items():
            percentage = (count / len(y_train)) * 100
            print(f"  Class {class_label}: {count:,} ({percentage:.2f}%)")
            
        print(f"\nTest Set Distribution:")
        for class_label, count in test_counts.items():
            percentage = (count / len(y_test)) * 100
            print(f"  Class {class_label}: {count:,} ({percentage:.2f}%)")
        
        # Check if stratified split maintained proportions
        original_props = self.donors_df[self.target_column].value_counts(normalize=True)
        train_props = y_train.value_counts(normalize=True)
        test_props = y_test.value_counts(normalize=True)
        
        print(f"\nProportion Consistency Check:")
        print(f"{'Class':<10} {'Original':<10} {'Train':<10} {'Test':<10} {'Train Diff':<12} {'Test Diff':<12}")
        print("-" * 70)
        
        for class_label in original_props.index:
            orig_pct = original_props[class_label] * 100
            train_pct = train_props[class_label] * 100
            test_pct = test_props[class_label] * 100
            train_diff = abs(train_pct - orig_pct)
            test_diff = abs(test_pct - orig_pct)
            
            print(f"{class_label:<10} {orig_pct:>8.2f}% {train_pct:>8.2f}% {test_pct:>8.2f}% "
                  f"{train_diff:>10.2f}% {test_diff:>10.2f}%")
        
        return {
            'train_counts': train_counts,
            'test_counts': test_counts,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def analyze_feature_distributions_by_class(self, features_to_analyze=None):
        """Analyze feature distributions by class to identify potential bias"""
        print("\n" + "=" * 60)
        print("FEATURE DISTRIBUTION BY CLASS ANALYSIS")
        print("=" * 60)
        
        if self.target_column not in self.donors_df.columns:
            print(f"Error: Target column '{self.target_column}' not found")
            return
            
        # Select features to analyze
        if features_to_analyze is None:
            # Select numeric features
            numeric_features = self.donors_df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column in numeric_features:
                numeric_features.remove(self.target_column)
            features_to_analyze = numeric_features[:10]  # Limit to 10 features for readability
        
        print(f"Analyzing {len(features_to_analyze)} features by class...")
        
        # Create subplots for feature distributions
        n_features = len(features_to_analyze)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Feature Distributions by Class', fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(features_to_analyze):
            row = idx // n_cols
            col = idx % n_cols
            
            if feature not in self.donors_df.columns:
                continue
                
            # Create box plots for each class
            class_data = []
            class_labels = []
            
            for class_label in sorted(self.donors_df[self.target_column].unique()):
                class_values = self.donors_df[
                    self.donors_df[self.target_column] == class_label
                ][feature].dropna()
                
                if len(class_values) > 0:
                    class_data.append(class_values)
                    class_labels.append(f'Class {class_label} (n={len(class_values)})')
            
            if class_data:
                axes[row, col].boxplot(class_data, labels=class_labels)
                axes[row, col].set_title(f'{feature}')
                axes[row, col].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for idx in range(len(features_to_analyze), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Statistical summary by class
        print(f"\nStatistical Summary by Class:")
        print("-" * 80)
        
        summary_stats = []
        for feature in features_to_analyze:
            if feature not in self.donors_df.columns:
                continue
                
            for class_label in sorted(self.donors_df[self.target_column].unique()):
                class_data = self.donors_df[
                    self.donors_df[self.target_column] == class_label
                ][feature].dropna()
                
                if len(class_data) > 0:
                    stats = {
                        'Feature': feature,
                        'Class': class_label,
                        'Count': len(class_data),
                        'Mean': class_data.mean(),
                        'Std': class_data.std(),
                        'Min': class_data.min(),
                        'Max': class_data.max(),
                        'Median': class_data.median()
                    }
                    summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        if not summary_df.empty:
            print(summary_df.round(3).to_string(index=False))
        
        return summary_df
    
    def analyze_family_relationships_balance(self):
        """Analyze class distribution within family relationships"""
        if self.relationships_df is None or self.relationships_df.empty:
            print("\nNo relationship data available for family analysis")
            return
            
        print("\n" + "=" * 60)
        print("FAMILY RELATIONSHIP CLASS BALANCE ANALYSIS")
        print("=" * 60)
        
        # Get family information
        family_groups = self.relationships_df.groupby('Family_ID')
        
        family_stats = []
        for family_id, family_members in family_groups:
            donor_ids = family_members['Donor_ID'].values
            
            # Get class labels for family members
            family_donors = self.donors_df[self.donors_df['ID'].isin(donor_ids)]
            
            if len(family_donors) > 1 and self.target_column in family_donors.columns:
                class_counts = family_donors[self.target_column].value_counts()
                total_members = len(family_donors)
                
                # Calculate family homogeneity (how similar class labels are)
                max_class_count = class_counts.max()
                homogeneity = max_class_count / total_members
                
                family_stats.append({
                    'Family_ID': family_id,
                    'Total_Members': total_members,
                    'Class_0_Count': class_counts.get(0, 0),
                    'Class_1_Count': class_counts.get(1, 0),
                    'Homogeneity': homogeneity,
                    'Has_Mixed_Classes': len(class_counts) > 1
                })
        
        if family_stats:
            family_df = pd.DataFrame(family_stats)
            
            print(f"Families with multiple members: {len(family_df)}")
            print(f"Average family size: {family_df['Total_Members'].mean():.2f}")
            print(f"Families with mixed classes: {family_df['Has_Mixed_Classes'].sum()}")
            print(f"Average homogeneity: {family_df['Homogeneity'].mean():.3f}")
            
            # Visualize family class distributions
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Family size distribution
            axes[0].hist(family_df['Total_Members'], bins=20, alpha=0.7, color='skyblue')
            axes[0].set_title('Family Size Distribution')
            axes[0].set_xlabel('Family Size')
            axes[0].set_ylabel('Number of Families')
            
            # Homogeneity distribution
            axes[1].hist(family_df['Homogeneity'], bins=20, alpha=0.7, color='lightgreen')
            axes[1].set_title('Family Class Homogeneity')
            axes[1].set_xlabel('Homogeneity Score')
            axes[1].set_ylabel('Number of Families')
            
            # Mixed vs homogeneous families
            mixed_counts = family_df['Has_Mixed_Classes'].value_counts()
            axes[2].pie(mixed_counts.values, labels=['Homogeneous', 'Mixed Classes'], 
                       autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
            axes[2].set_title('Family Class Distribution')
            
            plt.tight_layout()
            plt.show()
            
            return family_df
        
        return None
    
    def generate_imbalance_recommendations(self, analysis_results=None):
        """Generate recommendations for handling class imbalance"""
        print("\n" + "=" * 60)
        print("CLASS IMBALANCE HANDLING RECOMMENDATIONS")
        print("=" * 60)
        
        if analysis_results is None:
            analysis_results = self.analyze_class_distribution()
        
        if analysis_results is None:
            return
            
        imbalance_ratio = analysis_results['imbalance_ratio']
        severity = analysis_results['severity']
        
        print(f"Based on your imbalance ratio of {imbalance_ratio:.2f}:1 ({severity}):")
        print()
        
        # General recommendations
        print("🔧 GENERAL RECOMMENDATIONS:")
        print("1. Use stratified sampling for train/test splits (already implemented)")
        print("2. Consider using class weights in your loss function")
        print("3. Evaluate using metrics beyond accuracy (precision, recall, F1, AUC)")
        print("4. Consider the business cost of false positives vs false negatives")
        print()
        
        # Specific recommendations based on severity
        if severity == "Balanced":
            print("✅ Your dataset is well-balanced! Continue with current approach.")
            
        elif severity == "Mild Imbalance":
            print("⚠️ MILD IMBALANCE RECOMMENDATIONS:")
            print("- Add class weights to CrossEntropyLoss: weight=torch.tensor([1.0, 2-5x])")
            print("- Monitor precision/recall for both classes")
            print("- Consider focal loss if misclassification costs vary")
            
        elif severity == "Moderate Imbalance":
            print("⚠️ MODERATE IMBALANCE RECOMMENDATIONS:")
            print("- Use class weights: weight=torch.tensor([1.0, 5-10x])")
            print("- Consider SMOTE or other oversampling techniques")
            print("- Use stratified k-fold cross-validation")
            print("- Focus on minority class performance metrics")
            print("- Consider ensemble methods (bagging/boosting)")
            
        elif severity == "Severe Imbalance":
            print("🚨 SEVERE IMBALANCE RECOMMENDATIONS:")
            print("- Use class weights: weight=torch.tensor([1.0, 10-50x])")
            print("- Implement cost-sensitive learning")
            print("- Use focal loss or dice loss")
            print("- Consider anomaly detection approaches")
            print("- Use stratified sampling with high minority class ratios")
            print("- Consider data augmentation for minority class")
            print("- Use ensemble methods with balanced sampling")
        
        # GNN-specific recommendations
        print("\n🧠 GNN-SPECIFIC RECOMMENDATIONS:")
        print("- Use weighted random sampling for node classification")
        print("- Consider graph augmentation techniques")
        print("- Implement attention mechanisms to focus on minority class")
        print("- Use graph-level loss functions that account for class imbalance")
        print("- Consider using different aggregation strategies (mean vs max pooling)")
        
        # Code examples
        print("\n💻 CODE IMPLEMENTATION EXAMPLES:")
        print("""
# 1. Add class weights to your trainer:
class_counts = torch.bincount(data.y)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * len(class_weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 2. Use focal loss for severe imbalance:
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# 3. Stratified sampling for graph data:
def create_stratified_masks(data, train_ratio=0.6, val_ratio=0.2):
    from sklearn.model_selection import train_test_split
    
    indices = torch.arange(data.num_nodes)
    train_idx, temp_idx = train_test_split(
        indices, test_size=1-train_ratio, 
        stratify=data.y, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, 
        stratify=data.y[temp_idx], random_state=42
    )
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, val_mask, test_mask
        """)
    
    def run_full_diagnostics(self):
        """Run complete diagnostic analysis"""
        print("🔍 RUNNING COMPLETE DATASET DIAGNOSTICS")
        print("=" * 80)
        
        # Run all analyses
        class_analysis = self.analyze_class_distribution()
        self.visualize_class_distribution()
        
        split_analysis = self.analyze_train_test_split_balance()
        feature_analysis = self.analyze_feature_distributions_by_class()
        family_analysis = self.analyze_family_relationships_balance()
        
        # Generate recommendations
        self.generate_imbalance_recommendations(class_analysis)
        
        return {
            'class_analysis': class_analysis,
            'split_analysis': split_analysis,
            'feature_analysis': feature_analysis,
            'family_analysis': family_analysis
        }

def run_dataset_diagnostics(donors_df, relationships_df=None):
    """Convenience function to run full diagnostics"""
    diagnostics = DatasetDiagnostics(donors_df, relationships_df)
    return diagnostics.run_full_diagnostics()

# Example usage
if __name__ == "__main__":
    # Load your dataset
    donors_df = pd.read_csv('synthetic_donor_dataset/donors.csv')
    relationships_df = pd.read_csv('synthetic_donor_dataset/relationships.csv')
    
    # Run diagnostics
    results = run_dataset_diagnostics(donors_df, relationships_df)
