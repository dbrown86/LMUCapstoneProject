#!/usr/bin/env python3
"""
Data Loading Utilities for Dashboard
Handles loading and caching of donor data and model predictions
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import streamlit as st

class DataLoader:
    """Centralized data loading with caching"""
    
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.data_dir = self.project_root / 'data'
        self.cache_dir = self.data_dir / 'cache'
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @st.cache_data
    def load_donors(self):
        """Load donor data with caching"""
        try:
            donors_path = self.data_dir / 'synthetic_donor_dataset' / 'donors.csv'
            if not donors_path.exists():
                raise FileNotFoundError(f"Donor data not found at {donors_path}")
            
            donors_df = pd.read_csv(donors_path)
            return donors_df
        except Exception as e:
            st.error(f"Error loading donor data: {e}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_contact_reports(self):
        """Load contact reports data with caching"""
        try:
            reports_path = self.data_dir / 'synthetic_donor_dataset' / 'contact_reports.csv'
            if not reports_path.exists():
                raise FileNotFoundError(f"Contact reports not found at {reports_path}")
            
            reports_df = pd.read_csv(reports_path)
            return reports_df
        except Exception as e:
            st.error(f"Error loading contact reports: {e}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_giving_history(self):
        """Load giving history data with caching"""
        try:
            history_path = self.data_dir / 'synthetic_donor_dataset' / 'giving_history.csv'
            if not history_path.exists():
                raise FileNotFoundError(f"Giving history not found at {history_path}")
            
            history_df = pd.read_csv(history_path)
            return history_df
        except Exception as e:
            st.error(f"Error loading giving history: {e}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_relationships(self):
        """Load relationships data with caching"""
        try:
            relationships_path = self.data_dir / 'synthetic_donor_dataset' / 'relationships.csv'
            if not relationships_path.exists():
                raise FileNotFoundError(f"Relationships not found at {relationships_path}")
            
            relationships_df = pd.read_csv(relationships_path)
            return relationships_df
        except Exception as e:
            st.error(f"Error loading relationships: {e}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_bert_embeddings(self):
        """Load BERT embeddings with caching"""
        try:
            bert_path = self.data_dir / 'bert_embeddings_real.npy'
            if not bert_path.exists():
                raise FileNotFoundError(f"BERT embeddings not found at {bert_path}")
            
            embeddings = np.load(bert_path)
            return embeddings
        except Exception as e:
            st.warning(f"BERT embeddings not available: {e}")
            return None
    
    @st.cache_data
    def load_gnn_embeddings(self):
        """Load GNN embeddings with caching"""
        try:
            gnn_path = self.data_dir / 'gnn_embeddings_real.npy'
            if not gnn_path.exists():
                raise FileNotFoundError(f"GNN embeddings not found at {gnn_path}")
            
            embeddings = np.load(gnn_path)
            return embeddings
        except Exception as e:
            st.warning(f"GNN embeddings not available: {e}")
            return None
    
    @st.cache_data
    def load_predictions(self, results_file='fast_multimodal_results.pkl'):
        """Load model predictions with caching"""
        try:
            results_path = self.project_root / results_file
            if not results_path.exists():
                raise FileNotFoundError(f"Results file not found at {results_path}")
            
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
            return results
        except Exception as e:
            st.warning(f"Model predictions not available: {e}")
            return None
    
    def get_donor_by_id(self, donor_id):
        """Get specific donor data by ID"""
        donors_df = self.load_donors()
        if donors_df.empty:
            return None
        
        donor_data = donors_df[donors_df['ID'] == donor_id]
        if len(donor_data) == 0:
            return None
        
        return donor_data.iloc[0].to_dict()
    
    def search_donors(self, search_term, filters=None):
        """Search donors with filters"""
        donors_df = self.load_donors()
        if donors_df.empty:
            return pd.DataFrame()
        
        # Apply text search
        if search_term:
            search_cols = ['Full_Name', 'First_Name', 'Last_Name', 'ID']
            mask = pd.Series([False] * len(donors_df))
            
            for col in search_cols:
                if col in donors_df.columns:
                    mask |= donors_df[col].astype(str).str.contains(
                        search_term, case=False, na=False
                    )
            
            donors_df = donors_df[mask]
        
        # Apply filters
        if filters:
            if 'age_min' in filters and filters['age_min'] is not None:
                donors_df = donors_df[donors_df['Estimated_Age'] >= filters['age_min']]
            
            if 'age_max' in filters and filters['age_max'] is not None:
                donors_df = donors_df[donors_df['Estimated_Age'] <= filters['age_max']]
            
            if 'giving_min' in filters and filters['giving_min'] is not None:
                donors_df = donors_df[donors_df['Lifetime_Giving'] >= filters['giving_min']]
            
            if 'giving_max' in filters and filters['giving_max'] is not None:
                donors_df = donors_df[donors_df['Lifetime_Giving'] <= filters['giving_max']]
            
            if 'legacy_intent' in filters and filters['legacy_intent'] != 'All':
                legacy_value = True if filters['legacy_intent'] == 'Yes' else False
                donors_df = donors_df[donors_df['Legacy_Intent_Binary'] == legacy_value]
            
            if 'location' in filters and filters['location'] and filters['location'] != 'All':
                donors_df = donors_df[donors_df['Geographic_Region'] == filters['location']]
        
        return donors_df
    
    def get_model_status(self):
        """Check if model files are available"""
        model_files = [
            'scripts/advanced_multimodal_ensemble.py',
            'data/bert_embeddings_real.npy',
            'data/gnn_embeddings_real.npy'
        ]
        
        status = {}
        for file_path in model_files:
            full_path = self.project_root / file_path
            status[file_path] = full_path.exists()
        
        return status
    
    def get_data_summary(self):
        """Get summary of available data"""
        summary = {}
        
        # Donor data
        donors_df = self.load_donors()
        if not donors_df.empty:
            summary['donors'] = {
                'count': len(donors_df),
                'legacy_intent': len(donors_df[donors_df['Legacy_Intent_Binary'] == True]),
                'legacy_rate': f"{(len(donors_df[donors_df['Legacy_Intent_Binary'] == True]) / len(donors_df) * 100):.1f}%"
            }
        
        # Contact reports
        reports_df = self.load_contact_reports()
        if not reports_df.empty:
            summary['contact_reports'] = {'count': len(reports_df)}
        
        # Model predictions
        predictions = self.load_predictions()
        if predictions is not None:
            summary['predictions'] = {
                'available': True,
                'test_size': len(predictions.get('y_test', [])),
                'has_metrics': 'test_metrics' in predictions
            }
        else:
            summary['predictions'] = {'available': False}
        
        return summary

# Global data loader instance
data_loader = DataLoader()
