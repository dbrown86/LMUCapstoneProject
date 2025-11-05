#!/usr/bin/env python3
"""
Advanced Feature Engineering V2 for Donor Legacy Intent Prediction
Enhanced with temporal features, RFM analysis, and advanced transformations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import re
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineeringV2:
    """
    Advanced feature engineering V2 with temporal, RFM, and sophisticated transformations
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.transformers = {}
        self.clusterers = {}
        
    def create_temporal_features(self, donors_df, giving_history_df=None):
        """Create advanced temporal features including RFM analysis"""
        print("Creating temporal features...")
        
        features_df = donors_df.copy()
        # Drop any label/target columns to prevent leakage during feature selection
        leakage_cols = [
            'Legacy_Intent_Probability',
            'Legacy_Intent_Binary'
        ]
        features_df = features_df.drop(columns=[c for c in leakage_cols if c in features_df.columns], errors='ignore')
        
        # Basic temporal features
        if 'Estimated_Age' in features_df.columns:
            features_df['age_squared'] = features_df['Estimated_Age'] ** 2
            features_df['age_log'] = np.log1p(features_df['Estimated_Age'])
            features_df['age_sqrt'] = np.sqrt(features_df['Estimated_Age'])
            
        # RFM Analysis (Recency, Frequency, Monetary)
        if giving_history_df is not None:
            print("  Creating RFM features...")
            rfm_features = self._create_rfm_features(donors_df, giving_history_df)
            features_df = pd.concat([features_df, rfm_features], axis=1)
        
        # Temporal decay features
        if 'Last_Gift' in features_df.columns:
            features_df['last_gift_days'] = (pd.Timestamp.now() - pd.to_datetime(features_df['Last_Gift'], errors='coerce')).dt.days
            features_df['last_gift_days'] = features_df['last_gift_days'].fillna(365)
            
            # Temporal decay score
            features_df['temporal_decay_score'] = np.exp(-features_df['last_gift_days'] / 365)
            
        # Seasonality features
        if 'Last_Gift' in features_df.columns:
            last_gift_date = pd.to_datetime(features_df['Last_Gift'], errors='coerce')
            features_df['last_gift_month'] = last_gift_date.dt.month
            features_df['last_gift_quarter'] = last_gift_date.dt.quarter
            features_df['last_gift_is_december'] = (last_gift_date.dt.month == 12).astype(int)
            
        return features_df
    
    def _create_rfm_features(self, donors_df, giving_history_df):
        """Create RFM (Recency, Frequency, Monetary) features"""
        rfm_data = []
        
        for donor_id in donors_df['ID']:
            donor_giving = giving_history_df[giving_history_df['Donor_ID'] == donor_id]
            
            if len(donor_giving) == 0:
                rfm_data.append({
                    'rfm_recency': 0,
                    'rfm_frequency': 0,
                    'rfm_monetary': 0,
                    'rfm_score': 0,
                    'avg_gift_amount': 0,
                    'gift_consistency': 0,
                    'giving_trend': 0
                })
                continue
                
            # Recency (days since last gift)
            last_gift_date = pd.to_datetime(donor_giving['Gift_Date'].max())
            recency = (pd.Timestamp.now() - last_gift_date).days
            
            # Frequency (number of gifts)
            frequency = len(donor_giving)
            
            # Monetary (total amount)
            monetary = donor_giving['Gift_Amount'].sum()
            
            # Additional features
            avg_gift = donor_giving['Gift_Amount'].mean()
            
            # Gift consistency (coefficient of variation)
            gift_consistency = 1 - (donor_giving['Gift_Amount'].std() / (donor_giving['Gift_Amount'].mean() + 1e-8))
            
            # Giving trend (slope of gift amounts over time)
            if len(donor_giving) > 1:
                donor_giving_sorted = donor_giving.sort_values('Gift_Date')
                x = np.arange(len(donor_giving_sorted))
                y = donor_giving_sorted['Gift_Amount'].values
                trend = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
            else:
                trend = 0
                
            rfm_data.append({
                'rfm_recency': recency,
                'rfm_frequency': frequency,
                'rfm_monetary': monetary,
                'rfm_score': recency * frequency * monetary,
                'avg_gift_amount': avg_gift,
                'gift_consistency': gift_consistency,
                'giving_trend': trend
            })
            
        return pd.DataFrame(rfm_data, index=donors_df.index)
    
    def create_interaction_features(self, features_df):
        """Create advanced interaction features"""
        print("Creating interaction features...")
        
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        interaction_features = []
        
        # Polynomial interactions for top features
        important_features = ['Lifetime_Giving', 'Last_Gift', 'Engagement_Score', 'Estimated_Age']
        available_features = [col for col in important_features if col in features_df.columns]
        
        for i, feat1 in enumerate(available_features):
            for feat2 in available_features[i+1:]:
                if feat1 in features_df.columns and feat2 in features_df.columns:
                    # Multiplicative interaction
                    interaction = features_df[feat1] * features_df[feat2]
                    features_df[f'{feat1}_x_{feat2}'] = interaction
                    
                    # Ratio interaction
                    ratio = features_df[feat1] / (features_df[feat2] + 1e-8)
                    features_df[f'{feat1}_div_{feat2}'] = ratio
                    
                    # Difference interaction
                    diff = features_df[feat1] - features_df[feat2]
                    features_df[f'{feat1}_minus_{feat2}'] = diff
                    
        return features_df
    
    def create_statistical_features(self, features_df):
        """Create statistical features"""
        print("Creating statistical features...")
        
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        # Rolling statistics for time series features
        for col in ['Lifetime_Giving', 'Last_Gift', 'Engagement_Score']:
            if col in features_df.columns:
                # Z-score normalization
                features_df[f'{col}_zscore'] = stats.zscore(features_df[col])
                
                # Percentile ranking
                features_df[f'{col}_percentile'] = features_df[col].rank(pct=True)
                
                # Log transformation
                features_df[f'{col}_log'] = np.log1p(features_df[col])
                
                # Square root transformation
                features_df[f'{col}_sqrt'] = np.sqrt(features_df[col])
                
        return features_df
    
    def create_clustering_features(self, features_df, n_clusters=5):
        """Create clustering-based features"""
        print(f"Creating clustering features with {n_clusters} clusters...")
        
        # Select numeric features for clustering
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        clustering_features = ['Lifetime_Giving', 'Last_Gift', 'Engagement_Score', 'Estimated_Age']
        available_features = [col for col in clustering_features if col in features_df.columns]
        
        if len(available_features) < 2:
            return features_df
            
        # Prepare data for clustering
        X_cluster = features_df[available_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_cluster_scaled = scaler.fit_transform(X_cluster)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(X_cluster_scaled)
        
        features_df['donor_cluster'] = cluster_labels
        
        # Distance to cluster centers
        for i in range(n_clusters):
            cluster_center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(X_cluster_scaled - cluster_center, axis=1)
            features_df[f'distance_to_cluster_{i}'] = distances
            
        # Cluster size features
        cluster_sizes = pd.Series(cluster_labels).value_counts()
        features_df['cluster_size'] = features_df['donor_cluster'].map(cluster_sizes)
        
        return features_df
    
    def create_advanced_text_features(self, contact_reports_df):
        """Create advanced text features from contact reports"""
        print("Creating advanced text features...")
        
        if contact_reports_df is None or len(contact_reports_df) == 0:
            return pd.DataFrame()
            
        text_features = []
        
        for donor_id in contact_reports_df['Donor_ID'].unique():
            donor_reports = contact_reports_df[contact_reports_df['Donor_ID'] == donor_id]
            
            # Text length features
            avg_report_length = donor_reports['Report_Text'].str.len().mean() if 'Report_Text' in donor_reports.columns else 0
            max_report_length = donor_reports['Report_Text'].str.len().max() if 'Report_Text' in donor_reports.columns else 0
            
            # Sentiment indicators (simple keyword-based)
            if 'Report_Text' in donor_reports.columns:
                positive_words = ['thank', 'appreciate', 'grateful', 'excellent', 'wonderful', 'amazing']
                negative_words = ['concern', 'issue', 'problem', 'disappointed', 'unhappy']
                
                all_text = ' '.join(donor_reports['Report_Text'].fillna('').astype(str))
                positive_count = sum(1 for word in positive_words if word in all_text.lower())
                negative_count = sum(1 for word in negative_words if word in all_text.lower())
                
                sentiment_score = positive_count - negative_count
            else:
                sentiment_score = 0
                
            text_features.append({
                'Donor_ID': donor_id,
                'avg_report_length': avg_report_length,
                'max_report_length': max_report_length,
                'sentiment_score': sentiment_score,
                'report_count': len(donor_reports)
            })
            
        return pd.DataFrame(text_features)
    
    def create_network_features(self, relationships_df):
        """Create network-based features from relationships"""
        print("Creating network features...")
        
        if relationships_df is None or len(relationships_df) == 0:
            return pd.DataFrame()
            
        network_features = []
        
        for donor_id in relationships_df['Donor_ID'].unique():
            donor_relationships = relationships_df[relationships_df['Donor_ID'] == donor_id]
            
            # Network metrics
            degree_centrality = len(donor_relationships)
            
            # Relationship type diversity
            if 'Relationship_Type' in donor_relationships.columns:
                relationship_diversity = donor_relationships['Relationship_Type'].nunique()
            else:
                relationship_diversity = 0
                
            # Family network size
            family_relationships = donor_relationships[
                donor_relationships['Relationship_Type'].str.contains('family|spouse|child|parent', case=False, na=False)
            ] if 'Relationship_Type' in donor_relationships.columns else pd.DataFrame()
            family_network_size = len(family_relationships)
            
            network_features.append({
                'Donor_ID': donor_id,
                'degree_centrality': degree_centrality,
                'relationship_diversity': relationship_diversity,
                'family_network_size': family_network_size
            })
            
        return pd.DataFrame(network_features)
    
    def create_advanced_features(self, donors_df, contact_reports_df=None, 
                               giving_history_df=None, relationships_df=None):
        """Create all advanced features"""
        print("Creating advanced features...")
        
        # Start with basic features
        features_df = donors_df.copy()
        
        # Temporal features
        features_df = self.create_temporal_features(features_df, giving_history_df)
        
        # Interaction features
        features_df = self.create_interaction_features(features_df)
        
        # Statistical features
        features_df = self.create_statistical_features(features_df)
        
        # Clustering features
        features_df = self.create_clustering_features(features_df)
        
        # Text features
        if contact_reports_df is not None:
            text_features = self.create_advanced_text_features(contact_reports_df)
            if not text_features.empty:
                features_df = features_df.merge(text_features, left_on='ID', right_on='Donor_ID', how='left')
        
        # Network features
        if relationships_df is not None:
            network_features = self.create_network_features(relationships_df)
            if not network_features.empty:
                features_df = features_df.merge(network_features, left_on='ID', right_on='Donor_ID', how='left')
        
        # Fill missing values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
        
        print(f"Created {len(features_df.columns)} features")
        return features_df
    
    def select_features(self, X, y, method='mutual_info', k=200):
        """Advanced feature selection"""
        print(f"Selecting features using {method}...")
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        elif method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            selector = SelectFromModel(rf, max_features=k)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        X_selected = selector.fit_transform(X, y)
        self.feature_selectors[method] = selector
        
        print(f"Selected {X_selected.shape[1]} features from {X.shape[1]}")
        return X_selected
    
    def transform_features(self, X, method='pca', n_components=100):
        """Advanced feature transformation"""
        print(f"Transforming features using {method}...")
        
        if method == 'pca':
            transformer = PCA(n_components=n_components, random_state=self.random_state)
        elif method == 'ica':
            transformer = FastICA(n_components=n_components, random_state=self.random_state)
        elif method == 'svd':
            transformer = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        X_transformed = transformer.fit_transform(X)
        self.transformers[method] = transformer
        
        print(f"Transformed to {X_transformed.shape[1]} components")
        return X_transformed
