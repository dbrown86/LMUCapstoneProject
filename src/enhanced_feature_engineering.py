#!/usr/bin/env python3
"""
Enhanced Feature Engineering for Donor Legacy Intent Prediction
Advanced feature creation, selection, and transformation techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import re
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineering:
    """
    Advanced feature engineering for donor data
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.transformers = {}
        
    def create_donor_demographic_features(self, donors_df):
        """Create advanced demographic features"""
        print("Creating demographic features...")
        
        features_df = donors_df.copy()
        
        # Age-related features
        if 'Estimated_Age' in features_df.columns:
            features_df['age_group'] = pd.cut(features_df['Estimated_Age'], 
                                            bins=[0, 30, 45, 60, 75, 100], 
                                            labels=['Young', 'Adult', 'Middle', 'Senior', 'Elderly'])
            
        # Age interaction with other features
        if 'Lifetime_Giving' in features_df.columns:
            features_df['giving_per_age'] = features_df['Lifetime_Giving'] / features_df['Estimated_Age'].replace(0, 1)
        
        if 'Engagement_Score' in features_df.columns:
            features_df['engagement_age_ratio'] = features_df['Engagement_Score'] / features_df['Estimated_Age'].replace(0, 1)
        
        # Convert Rating to numeric for all subsequent operations
        if 'Rating' in features_df.columns and features_df['Rating'].dtype == 'object':
            rating_map = {
                'A': 10, 'B': 9, 'C': 8, 'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1,
                'K': 0.5, 'L': 0.1, 'M': 0.05, 'N': 0.01, 'O': 0.005, 'P': 0.001
            }
            features_df['Rating'] = features_df['Rating'].map(rating_map).fillna(1.0)
        
        # Gender-based features
        if 'Gender' in features_df.columns:
            # Create gender interaction features
            features_df['is_female'] = (features_df['Gender'] == 'Female').astype(int)
            features_df['is_male'] = (features_df['Gender'] == 'Male').astype(int)
        
        # Geographic features
        if 'Geographic_Region' in features_df.columns:
            # Create region-based features
            region_wealth_map = {
                'Northeast': 1.2,
                'West': 1.1,
                'Midwest': 0.9,
                'South': 0.8
            }
            features_df['region_wealth_factor'] = features_df['Geographic_Region'].map(region_wealth_map).fillna(1.0)
            
            # Region interaction with giving
            if 'Lifetime_Giving' in features_df.columns:
                features_df['regional_giving_adjusted'] = features_df['Lifetime_Giving'] * features_df['region_wealth_factor']
        
        # Professional background features
        if 'Professional_Background' in features_df.columns:
            # Create professional categories
            high_income_professions = ['Business Executive', 'Doctor', 'Lawyer', 'Engineer', 'Finance Professional']
            features_df['high_income_profession'] = features_df['Professional_Background'].isin(high_income_professions).astype(int)
            
            # Professional giving potential
            profession_potential_map = {
                'Business Executive': 1.5,
                'Doctor': 1.3,
                'Lawyer': 1.2,
                'Engineer': 1.1,
                'Finance Professional': 1.4,
                'Teacher': 0.7,
                'Retired': 0.9,
                'Student': 0.3
            }
            features_df['profession_giving_potential'] = features_df['Professional_Background'].map(profession_potential_map).fillna(1.0)
        
        print(f"Created {features_df.shape[1] - donors_df.shape[1]} demographic features")
        return features_df
    
    def create_giving_pattern_features(self, donors_df):
        """Create advanced giving pattern features"""
        print("Creating giving pattern features...")
        
        features_df = donors_df.copy()
        
        # Basic giving metrics
        if all(col in features_df.columns for col in ['Lifetime_Giving', 'Total_Yr_Giving_Count']):
            features_df['avg_gift_size'] = features_df['Lifetime_Giving'] / features_df['Total_Yr_Giving_Count'].replace(0, 1)
            features_df['gift_size_consistency'] = features_df['avg_gift_size'] / (features_df['Lifetime_Giving'].std() + 1e-8)
        
        # Consecutive giving features
        if 'Consecutive_Yr_Giving_Count' in features_df.columns:
            features_df['giving_consistency'] = features_df['Consecutive_Yr_Giving_Count'] / features_df['Total_Yr_Giving_Count'].replace(0, 1)
            features_df['is_consistent_donor'] = (features_df['giving_consistency'] > 0.7).astype(int)
            features_df['is_irregular_donor'] = (features_df['giving_consistency'] < 0.3).astype(int)
        
        # Giving velocity features
        if all(col in features_df.columns for col in ['Total_Yr_Giving_Count', 'Estimated_Age']):
            features_df['giving_velocity'] = features_df['Total_Yr_Giving_Count'] / features_df['Estimated_Age'].replace(0, 1)
            features_df['high_velocity_donor'] = (features_df['giving_velocity'] > features_df['giving_velocity'].quantile(0.8)).astype(int)
        
        # Rating alignment features
        if all(col in features_df.columns for col in ['Rating', 'Lifetime_Giving']):
            # Rating should already be converted to numeric in demographic features
            # Normalize giving to 0-10 scale for comparison with rating
            giving_normalized = (features_df['Lifetime_Giving'] - features_df['Lifetime_Giving'].min()) / \
                               (features_df['Lifetime_Giving'].max() - features_df['Lifetime_Giving'].min()) * 10
            features_df['rating_giving_alignment'] = abs(features_df['Rating'] - giving_normalized)
            features_df['well_aligned_donor'] = (features_df['rating_giving_alignment'] < 2).astype(int)
        
        # Prospect stage features
        if 'Prospect_Stage' in features_df.columns:
            stage_numeric_map = {
                'Cold': 1,
                'Warm': 2,
                'Hot': 3,
                'Qualified': 4,
                'Cultivated': 5
            }
            features_df['prospect_stage_numeric'] = features_df['Prospect_Stage'].map(stage_numeric_map).fillna(1)
            features_df['high_stage_prospect'] = (features_df['prospect_stage_numeric'] >= 4).astype(int)
        
        # Board affiliation features
        if 'Board_Affiliations' in features_df.columns:
            features_df['num_board_affiliations'] = features_df['Board_Affiliations'].str.split(',').str.len().fillna(0)
            features_df['has_board_affiliation'] = (features_df['num_board_affiliations'] > 0).astype(int)
            features_df['multiple_board_affiliations'] = (features_df['num_board_affiliations'] > 1).astype(int)
        
        print(f"Created {features_df.shape[1] - donors_df.shape[1]} giving pattern features")
        return features_df
    
    def create_engagement_features(self, donors_df):
        """Create advanced engagement features"""
        print("Creating engagement features...")
        
        features_df = donors_df.copy()
        
        if 'Engagement_Score' in features_df.columns:
            # Engagement level categorization
            features_df['engagement_level'] = pd.cut(features_df['Engagement_Score'],
                                                   bins=[0, 2, 4, 6, 8, 10],
                                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            # Engagement intensity relative to giving
            if 'Lifetime_Giving' in features_df.columns:
                features_df['engagement_intensity'] = features_df['Engagement_Score'] / (features_df['Lifetime_Giving'] + 1)
                features_df['high_engagement_low_giving'] = ((features_df['Engagement_Score'] > 6) & 
                                                           (features_df['Lifetime_Giving'] < features_df['Lifetime_Giving'].median())).astype(int)
            
            # Engagement consistency
            features_df['engagement_tier'] = pd.cut(features_df['Engagement_Score'],
                                                  bins=[0, 3, 6, 10],
                                                  labels=['Low', 'Medium', 'High'])
        
        # Interest keyword features
        if 'Interest_Keywords' in features_df.columns:
            features_df['num_interests'] = features_df['Interest_Keywords'].str.split(',').str.len().fillna(0)
            
            # Legacy-related interests
            legacy_keywords = ['planned giving', 'legacy', 'estate', 'bequest', 'endowment', 'foundation']
            features_df['has_legacy_interest'] = features_df['Interest_Keywords'].str.contains(
                '|'.join(legacy_keywords), case=False, na=False).astype(int)
            
            # High-value interests
            high_value_keywords = ['major gift', 'capital campaign', 'building', 'scholarship', 'research']
            features_df['has_high_value_interest'] = features_df['Interest_Keywords'].str.contains(
                '|'.join(high_value_keywords), case=False, na=False).astype(int)
            
            # Diversity of interests
            features_df['interest_diversity'] = features_df['Interest_Keywords'].apply(
                lambda x: len(set(str(x).split(','))) if pd.notna(x) else 0)
        
        print(f"Created {features_df.shape[1] - donors_df.shape[1]} engagement features")
        return features_df
    
    def create_family_network_features(self, donors_df, relationships_df):
        """Create family network features"""
        print("Creating family network features...")
        
        features_df = donors_df.copy()
        
        if relationships_df is None or relationships_df.empty:
            print("No relationship data available")
            return features_df
        
        # Family network metrics
        family_features = {}
        
        for donor_id in donors_df['ID']:
            donor_relationships = relationships_df[relationships_df['Donor_ID'] == donor_id]
            
            features = {
                'num_family_connections': len(donor_relationships),
                'has_family_connections': 1 if len(donor_relationships) > 0 else 0,
                'family_network_size': 0
            }
            
            # Relationship type analysis
            if len(donor_relationships) > 0:
                relationship_types = donor_relationships['Relationship_Type'].value_counts()
                
                # Specific relationship counts
                for rel_type in ['Spouse', 'Child', 'Parent', 'Sibling', 'Other']:
                    features[f'num_{rel_type.lower()}_connections'] = relationship_types.get(rel_type, 0)
                
                # Family network complexity
                features['family_network_size'] = len(donor_relationships)
                features['family_network_diversity'] = len(relationship_types)
                
                # Immediate family indicator
                immediate_family = relationship_types.get('Spouse', 0) + relationship_types.get('Child', 0)
                features['has_immediate_family'] = 1 if immediate_family > 0 else 0
            
            family_features[donor_id] = features
        
        # Convert to DataFrame and merge
        family_df = pd.DataFrame.from_dict(family_features, orient='index').reset_index()
        family_df.columns = ['ID'] + list(family_df.columns[1:])
        
        features_df = features_df.merge(family_df, on='ID', how='left')
        
        # Fill NaN values with 0 for family features
        family_columns = [col for col in family_df.columns if col != 'ID']
        features_df[family_columns] = features_df[family_columns].fillna(0)
        
        print(f"Created {len(family_columns)} family network features")
        return features_df
    
    def create_temporal_features(self, donors_df, giving_history_df=None):
        """Create temporal features from giving history"""
        print("Creating temporal features...")
        
        features_df = donors_df.copy()
        
        if giving_history_df is None or giving_history_df.empty:
            print("No giving history data available")
            return features_df
        
        # Convert date column if it exists
        if 'Date' in giving_history_df.columns:
            giving_history_df['Date'] = pd.to_datetime(giving_history_df['Date'])
        
        temporal_features = {}
        
        for donor_id in donors_df['ID']:
            donor_history = giving_history_df[giving_history_df['Donor_ID'] == donor_id]
            
            features = {
                'num_gifts': len(donor_history),
                'total_giving_amount': 0,
                'avg_gift_amount': 0,
                'max_gift_amount': 0,
                'min_gift_amount': 0,
                'gift_frequency_months': 0,
                'has_recent_giving': 0,
                'giving_trend': 0,
                'gift_volatility': 0,
                'seasonal_giving_pattern': 0
            }
            
            if len(donor_history) > 0:
                # Basic giving metrics
                amounts = donor_history['Amount'].values
                features['total_giving_amount'] = amounts.sum()
                features['avg_gift_amount'] = amounts.mean()
                features['max_gift_amount'] = amounts.max()
                features['min_gift_amount'] = amounts.min()
                
                # Frequency metrics
                if 'Date' in giving_history_df.columns:
                    features['gift_frequency_months'] = donor_history['Date'].dt.to_period('M').nunique()
                    
                    # Recent giving indicator
                    recent_threshold = pd.Timestamp.now() - pd.DateOffset(years=1)
                    features['has_recent_giving'] = 1 if donor_history['Date'].max() > recent_threshold else 0
                    
                    # Giving trend analysis
                    if len(donor_history) >= 3:
                        donor_history_sorted = donor_history.sort_values('Date')
                        amounts_sorted = donor_history_sorted['Amount'].values
                        
                        # Simple trend calculation
                        recent_avg = amounts_sorted[-3:].mean()
                        earlier_avg = amounts_sorted[:3].mean()
                        
                        if recent_avg > earlier_avg * 1.1:
                            features['giving_trend'] = 1  # Increasing
                        elif recent_avg < earlier_avg * 0.9:
                            features['giving_trend'] = -1  # Decreasing
                        else:
                            features['giving_trend'] = 0  # Stable
                    
                    # Gift volatility
                    if len(amounts) > 1:
                        features['gift_volatility'] = amounts.std() / (amounts.mean() + 1e-8)
                    
                    # Seasonal giving pattern
                    if 'Date' in giving_history_df.columns and len(donor_history) >= 4:
                        monthly_giving = donor_history.groupby(donor_history['Date'].dt.month)['Amount'].sum()
                        if len(monthly_giving) >= 3:
                            # Check for seasonal patterns (e.g., year-end giving)
                            december_giving = monthly_giving.get(12, 0)
                            total_giving = monthly_giving.sum()
                            features['seasonal_giving_pattern'] = december_giving / (total_giving + 1e-8)
            
            temporal_features[donor_id] = features
        
        # Convert to DataFrame and merge
        temporal_df = pd.DataFrame.from_dict(temporal_features, orient='index').reset_index()
        temporal_df.columns = ['ID'] + list(temporal_df.columns[1:])
        
        features_df = features_df.merge(temporal_df, on='ID', how='left')
        
        # Fill NaN values with 0 for temporal features
        temporal_columns = [col for col in temporal_df.columns if col != 'ID']
        features_df[temporal_columns] = features_df[temporal_columns].fillna(0)
        
        print(f"Created {len(temporal_columns)} temporal features")
        return features_df
    
    def create_interaction_features(self, features_df):
        """Create interaction features between important variables"""
        print("Creating interaction features...")
        
        enhanced_df = features_df.copy()
        
        # Rating and engagement interaction
        if all(col in enhanced_df.columns for col in ['Rating', 'Engagement_Score']):
            enhanced_df['rating_engagement_product'] = enhanced_df['Rating'] * enhanced_df['Engagement_Score']
            enhanced_df['rating_engagement_ratio'] = enhanced_df['Rating'] / (enhanced_df['Engagement_Score'] + 1)
        
        # Age and giving interaction
        if all(col in enhanced_df.columns for col in ['Estimated_Age', 'Lifetime_Giving']):
            enhanced_df['age_giving_product'] = enhanced_df['Estimated_Age'] * enhanced_df['Lifetime_Giving']
            enhanced_df['age_giving_ratio'] = enhanced_df['Lifetime_Giving'] / (enhanced_df['Estimated_Age'] + 1)
        
        # Family and engagement interaction
        if all(col in enhanced_df.columns for col in ['num_family_connections', 'Engagement_Score']):
            enhanced_df['family_engagement_interaction'] = enhanced_df['num_family_connections'] * enhanced_df['Engagement_Score']
        
        # Professional background and giving interaction
        if all(col in enhanced_df.columns for col in ['profession_giving_potential', 'Lifetime_Giving']):
            enhanced_df['profession_giving_alignment'] = enhanced_df['Lifetime_Giving'] * enhanced_df['profession_giving_potential']
        
        # Board affiliation and engagement interaction
        if all(col in enhanced_df.columns for col in ['num_board_affiliations', 'Engagement_Score']):
            enhanced_df['board_engagement_interaction'] = enhanced_df['num_board_affiliations'] * enhanced_df['Engagement_Score']
        
        print(f"Created {enhanced_df.shape[1] - features_df.shape[1]} interaction features")
        return enhanced_df
    
    def create_polynomial_features(self, features_df, degree=2, include_interactions=True):
        """Create polynomial features for important variables"""
        print(f"Creating polynomial features (degree {degree})...")
        
        enhanced_df = features_df.copy()
        
        # Select important numeric features for polynomial expansion
        important_features = []
        for col in ['Rating', 'Engagement_Score', 'Lifetime_Giving', 'Estimated_Age']:
            if col in enhanced_df.columns:
                important_features.append(col)
        
        if len(important_features) < 2:
            print("Not enough important features for polynomial expansion")
            return enhanced_df
        
        # Create polynomial features
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=not include_interactions)
        
        # Get numeric features
        numeric_data = enhanced_df[important_features].fillna(0)
        
        # Create polynomial features
        poly_features = poly.fit_transform(numeric_data)
        
        # Create feature names
        feature_names = poly.get_feature_names_out(important_features)
        
        # Add polynomial features to dataframe
        for i, feature_name in enumerate(feature_names):
            if feature_name not in enhanced_df.columns:  # Avoid duplicates
                enhanced_df[f'poly_{feature_name}'] = poly_features[:, i]
        
        print(f"Created {poly_features.shape[1]} polynomial features")
        return enhanced_df
    
    def create_clustering_features(self, features_df, n_clusters=5):
        """Create clustering-based features"""
        print(f"Creating clustering features ({n_clusters} clusters)...")
        
        enhanced_df = features_df.copy()
        
        # Select numeric features for clustering
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 3:
            print("Not enough numeric features for clustering")
            return enhanced_df
        
        # Prepare data for clustering
        clustering_data = features_df[numeric_cols].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        clustering_data_scaled = scaler.fit_transform(clustering_data)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(clustering_data_scaled)
        
        # Add cluster features
        enhanced_df['donor_cluster'] = cluster_labels
        
        # Create cluster-based features
        for i in range(n_clusters):
            enhanced_df[f'is_cluster_{i}'] = (cluster_labels == i).astype(int)
        
        # Calculate distance to cluster centers
        distances = kmeans.transform(clustering_data_scaled)
        for i in range(n_clusters):
            enhanced_df[f'distance_to_cluster_{i}'] = distances[:, i]
        
        # Silhouette score
        silhouette = silhouette_score(clustering_data_scaled, cluster_labels)
        print(f"Clustering silhouette score: {silhouette:.4f}")
        
        print(f"Created {n_clusters + 1} clustering features")
        return enhanced_df
    
    def select_features(self, X, y, method='mutual_info', k=50, cv_folds=5):
        """Advanced feature selection"""
        print(f"Selecting top {k} features using {method}...")
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'rfe':
            # Recursive Feature Elimination
            base_estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            selector = RFE(estimator=base_estimator, n_features_to_select=k)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        
        # Store selector for later use
        self.feature_selectors[method] = selector
        
        print(f"Selected {X_selected.shape[1]} features from {X.shape[1]}")
        return X_selected, selector
    
    def apply_dimensionality_reduction(self, X, method='pca', n_components=20):
        """Apply dimensionality reduction techniques"""
        print(f"Applying {method} dimensionality reduction...")
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=self.random_state)
        elif method == 'ica':
            reducer = FastICA(n_components=n_components, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        X_reduced = reducer.fit_transform(X)
        
        # Store reducer for later use
        self.transformers[method] = reducer
        
        if method == 'pca':
            explained_variance = reducer.explained_variance_ratio_.sum()
            print(f"Explained variance ratio: {explained_variance:.4f}")
        
        print(f"Reduced from {X.shape[1]} to {X_reduced.shape[1]} dimensions")
        return X_reduced, reducer
    
    def visualize_feature_importance(self, feature_names, importances, top_n=20):
        """Visualize feature importance"""
        print(f"Visualizing top {top_n} feature importances...")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.tail(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def create_comprehensive_features(self, donors_df, relationships_df=None, giving_history_df=None):
        """Create comprehensive feature set using all methods"""
        print("=" * 80)
        print("CREATING COMPREHENSIVE FEATURE SET")
        print("=" * 80)
        
        # Start with original data
        features_df = donors_df.copy()
        original_features = features_df.shape[1]
        
        # Ensure numeric columns are properly typed
        print("Converting data types...")
        numeric_columns = ['Lifetime_Giving', 'Last_Gift', 'Consecutive_Yr_Giving_Count', 
                          'Total_Yr_Giving_Count', 'Family_Giving_Potential', 'Engagement_Score', 
                          'Legacy_Intent_Probability', 'Estimated_Age']
        
        for col in numeric_columns:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
        
        # Handle boolean columns
        if 'Legacy_Intent_Binary' in features_df.columns:
            features_df['Legacy_Intent_Binary'] = features_df['Legacy_Intent_Binary'].astype(int)
        
        # Apply all feature engineering methods
        features_df = self.create_donor_demographic_features(features_df)
        features_df = self.create_giving_pattern_features(features_df)
        features_df = self.create_engagement_features(features_df)
        features_df = self.create_family_network_features(features_df, relationships_df)
        features_df = self.create_temporal_features(features_df, giving_history_df)
        features_df = self.create_interaction_features(features_df)
        features_df = self.create_polynomial_features(features_df, degree=2)
        features_df = self.create_clustering_features(features_df, n_clusters=5)
        
        total_features = features_df.shape[1]
        new_features = total_features - original_features
        
        print(f"\nFeature Engineering Summary:")
        print(f"  Original features: {original_features}")
        print(f"  New features created: {new_features}")
        print(f"  Total features: {total_features}")
        
        return features_df
    
    def get_feature_summary(self, features_df):
        """Get summary of all features"""
        print("\n" + "=" * 60)
        print("FEATURE SUMMARY")
        print("=" * 60)
        
        # Numeric features
        numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Numeric features: {len(numeric_features)}")
        
        # Categorical features
        categorical_features = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Categorical features: {len(categorical_features)}")
        
        # Binary features
        binary_features = [col for col in numeric_features if features_df[col].nunique() == 2]
        print(f"Binary features: {len(binary_features)}")
        
        # Missing values
        missing_values = features_df.isnull().sum()
        features_with_missing = missing_values[missing_values > 0]
        
        if len(features_with_missing) > 0:
            print(f"\nFeatures with missing values:")
            for feature, missing_count in features_with_missing.items():
                missing_pct = missing_count / len(features_df) * 100
                print(f"  {feature}: {missing_count} ({missing_pct:.1f}%)")
        
        # Feature correlation analysis
        if len(numeric_features) > 1:
            correlation_matrix = features_df[numeric_features].corr()
            high_corr_pairs = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:  # High correlation threshold
                        high_corr_pairs.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            corr_value
                        ))
            
            if high_corr_pairs:
                print(f"\nHigh correlation pairs (|r| > 0.8):")
                for feat1, feat2, corr in high_corr_pairs[:10]:  # Show top 10
                    print(f"  {feat1} - {feat2}: {corr:.3f}")
        
        return {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'binary_features': binary_features,
            'missing_values': features_with_missing,
            'high_correlation_pairs': high_corr_pairs
        }

# Example usage
def demo_enhanced_feature_engineering():
    """Demonstrate the enhanced feature engineering"""
    print("=" * 80)
    print("ENHANCED FEATURE ENGINEERING DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample donor data
    donors_df = pd.DataFrame({
        'ID': range(n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Estimated_Age': np.random.normal(55, 15, n_samples).astype(int),
        'Rating': np.random.uniform(1, 10, n_samples),
        'Lifetime_Giving': np.random.exponential(5000, n_samples),
        'Engagement_Score': np.random.uniform(1, 10, n_samples),
        'Geographic_Region': np.random.choice(['Northeast', 'West', 'Midwest', 'South'], n_samples),
        'Professional_Background': np.random.choice(['Business Executive', 'Doctor', 'Teacher', 'Engineer', 'Retired'], n_samples),
        'Total_Yr_Giving_Count': np.random.poisson(3, n_samples),
        'Consecutive_Yr_Giving_Count': np.random.poisson(2, n_samples),
        'Interest_Keywords': [','.join(np.random.choice(['education', 'healthcare', 'planned giving', 'legacy'], 
                                                       size=np.random.randint(1, 4))) for _ in range(n_samples)],
        'Board_Affiliations': np.random.choice(['Board Member', 'Advisory Board', ''], n_samples, p=[0.1, 0.2, 0.7])
    })
    
    # Create sample relationship data
    relationships_df = pd.DataFrame({
        'Donor_ID': np.random.choice(donors_df['ID'], size=200),
        'Relationship_Type': np.random.choice(['Spouse', 'Child', 'Parent', 'Sibling'], 200)
    })
    
    # Create sample giving history
    giving_history_df = pd.DataFrame({
        'Donor_ID': np.random.choice(donors_df['ID'], size=500),
        'Amount': np.random.exponential(1000, 500),
        'Date': pd.date_range('2020-01-01', '2023-12-31', periods=500)
    })
    
    # Initialize feature engineering
    fe = AdvancedFeatureEngineering()
    
    # Create comprehensive features
    enhanced_df = fe.create_comprehensive_features(donors_df, relationships_df, giving_history_df)
    
    # Get feature summary
    feature_summary = fe.get_feature_summary(enhanced_df)
    
    print(f"\nFinal dataset shape: {enhanced_df.shape}")
    
    return fe, enhanced_df, feature_summary

if __name__ == "__main__":
    demo_enhanced_feature_engineering()
