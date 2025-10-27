#!/usr/bin/env python3
"""
SQL Database Data Loader for 500K Donor Dataset
Provides efficient data loading from SQLite database with relationship graph construction
"""

import sqlite3
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SQLDataLoader:
    """
    Efficient data loader for the 500K donor SQL database
    Supports both tabular and graph-based data loading
    """
    
    def __init__(self, db_path: str = "data/synthetic_donor_dataset_500k_dense/donor_database.db"):
        """
        Initialize SQL data loader
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {db_path}")
        
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
    
    def _disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._disconnect()
    
    def get_donors(self, 
                   limit: Optional[int] = None,
                   donor_ids: Optional[List[int]] = None,
                   filters: Optional[Dict[str, any]] = None,
                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load donors data with optional filtering
        
        Args:
            limit: Maximum number of records to return
            filters: Dictionary of column: value filters
            columns: Specific columns to select
            
        Returns:
            DataFrame with donor data
        """
        query = "SELECT * FROM donors"
        conditions = []
        params = []
        
        # Add donor_ids filter
        if donor_ids:
            placeholders = ','.join(['?' for _ in donor_ids])
            conditions.append(f"ID IN ({placeholders})")
            params.extend(donor_ids)
        
        # Add other filters
        if filters:
            for col, value in filters.items():
                if isinstance(value, (list, tuple)):
                    placeholders = ','.join(['?' for _ in value])
                    conditions.append(f"{col} IN ({placeholders})")
                    params.extend(value)
                else:
                    conditions.append(f"{col} = ?")
                    params.append(value)
        
        # Apply conditions
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Add column selection
        if columns:
            query = query.replace("SELECT *", f"SELECT {', '.join(columns)}")
        
        # Add limit
        if limit:
            query += f" LIMIT {limit}"
        
        # Execute query (params already set above)
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_relationships(self, 
                         donor_ids: Optional[List[int]] = None,
                         relationship_types: Optional[List[str]] = None,
                         min_strength: Optional[float] = None,
                         limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load relationships data with optional filtering
        
        Args:
            donor_ids: Specific donor IDs to filter by
            relationship_types: Specific relationship types to include
            min_strength: Minimum relationship strength threshold
            
        Returns:
            DataFrame with relationship data
        """
        query = "SELECT * FROM relationships WHERE 1=1"
        params = []
        
        if donor_ids:
            placeholders = ','.join(['?' for _ in donor_ids])
            query += f" AND (Donor_ID_1 IN ({placeholders}) OR Donor_ID_2 IN ({placeholders}))"
            params.extend(donor_ids + donor_ids)
        
        if relationship_types:
            placeholders = ','.join(['?' for _ in relationship_types])
            query += f" AND Relationship_Category IN ({placeholders})"
            params.extend(relationship_types)
        
        if min_strength is not None:
            query += " AND Relationship_Strength >= ?"
            params.append(min_strength)
        
        # Add limit
        if limit:
            query += f" LIMIT {limit}"
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_giving_history(self, 
                          donor_ids: Optional[List[int]] = None,
                          date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Load giving history with optional filtering
        
        Args:
            donor_ids: Specific donor IDs to filter by
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
            
        Returns:
            DataFrame with giving history
        """
        query = "SELECT * FROM giving_history WHERE 1=1"
        params = []
        
        if donor_ids:
            placeholders = ','.join(['?' for _ in donor_ids])
            query += f" AND Donor_ID IN ({placeholders})"
            params.extend(donor_ids)
        
        if date_range:
            start_date, end_date = date_range
            query += " AND Gift_Date >= ? AND Gift_Date <= ?"
            params.extend([start_date, end_date])
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_event_attendance(self, 
                            donor_ids: Optional[List[int]] = None,
                            event_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load event attendance data
        
        Args:
            donor_ids: Specific donor IDs to filter by
            event_names: Specific event names to include
            
        Returns:
            DataFrame with event attendance
        """
        query = "SELECT * FROM event_attendance WHERE 1=1"
        params = []
        
        if donor_ids:
            placeholders = ','.join(['?' for _ in donor_ids])
            query += f" AND Donor_ID IN ({placeholders})"
            params.extend(donor_ids)
        
        if event_names:
            placeholders = ','.join(['?' for _ in event_names])
            query += f" AND Event_Name IN ({placeholders})"
            params.extend(event_names)
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_contact_reports(self, 
                           donor_ids: Optional[List[int]] = None,
                           contact_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load contact reports data
        
        Args:
            donor_ids: Specific donor IDs to filter by
            contact_types: Specific contact types to include
            
        Returns:
            DataFrame with contact reports
        """
        query = "SELECT * FROM contact_reports WHERE 1=1"
        params = []
        
        if donor_ids:
            placeholders = ','.join(['?' for _ in donor_ids])
            query += f" AND Donor_ID IN ({placeholders})"
            params.extend(donor_ids)
        
        if contact_types:
            placeholders = ','.join(['?' for _ in contact_types])
            query += f" AND Contact_Type IN ({placeholders})"
            params.extend(contact_types)
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_enhanced_fields(self, 
                           donor_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load enhanced fields (ML features)
        
        Args:
            donor_ids: Specific donor IDs to filter by
            
        Returns:
            DataFrame with enhanced fields
        """
        query = "SELECT * FROM enhanced_fields WHERE 1=1"
        params = []
        
        if donor_ids:
            placeholders = ','.join(['?' for _ in donor_ids])
            query += f" AND Donor_ID IN ({placeholders})"
            params.extend(donor_ids)
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_family_relationships(self, 
                                donor_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load family relationships
        
        Args:
            donor_ids: Specific donor IDs to filter by
            
        Returns:
            DataFrame with family relationships
        """
        query = "SELECT * FROM family_relationships WHERE 1=1"
        params = []
        
        if donor_ids:
            placeholders = ','.join(['?' for _ in donor_ids])
            query += f" AND Donor_ID IN ({placeholders})"
            params.extend(donor_ids)
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_multimodal_data(self, 
                           donor_ids: Optional[List[int]] = None,
                           limit: Optional[int] = None,
                           include_graph: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load all data modalities for multimodal fusion
        
        Args:
            donor_ids: Specific donor IDs to filter by
            limit: Maximum number of records to return
            include_graph: Whether to include graph relationships
            
        Returns:
            Dictionary with all data modalities
        """
        data = {}
        
        # Core donor data
        data['donors'] = self.get_donors(donor_ids=donor_ids, limit=limit)
        
        # If we have donor_ids and they're too many, batch them
        if donor_ids and len(donor_ids) > 1000:  # SQLite parameter limit is around 1000
            # Get the actual donor IDs from the loaded donors
            actual_donor_ids = data['donors']['ID'].tolist()
            donor_ids = actual_donor_ids[:1000] if len(actual_donor_ids) > 1000 else actual_donor_ids
        
        # Enhanced features
        data['enhanced_fields'] = self.get_enhanced_fields(donor_ids=donor_ids)
        
        # Giving history
        data['giving_history'] = self.get_giving_history(donor_ids=donor_ids)
        
        # Event attendance
        data['event_attendance'] = self.get_event_attendance(donor_ids=donor_ids)
        
        # Contact reports
        data['contact_reports'] = self.get_contact_reports(donor_ids=donor_ids)
        
        # Family relationships
        data['family_relationships'] = self.get_family_relationships(donor_ids=donor_ids)
        
        # Graph relationships (if requested)
        if include_graph:
            data['relationships'] = self.get_relationships(donor_ids=donor_ids)
        
        return data
    
    def build_graph_data(self, 
                        donor_ids: Optional[List[int]] = None,
                        relationship_types: Optional[List[str]] = None,
                        min_strength: float = 0.1) -> Data:
        """
        Build PyTorch Geometric graph data from relationships
        
        Args:
            donor_ids: Specific donor IDs to include in graph
            relationship_types: Specific relationship types to include
            min_strength: Minimum relationship strength
            
        Returns:
            PyTorch Geometric Data object
        """
        # Get relationships
        relationships_df = self.get_relationships(
            donor_ids=donor_ids,
            relationship_types=relationship_types,
            min_strength=min_strength
        )
        
        if relationships_df.empty:
            # Return empty graph
            return Data(x=torch.empty(0, 1), edge_index=torch.empty(2, 0, dtype=torch.long))
        
        # Get unique donor IDs from relationships
        all_donor_ids = set(relationships_df['Donor_ID_1'].unique()) | set(relationships_df['Donor_ID_2'].unique())
        all_donor_ids = sorted(list(all_donor_ids))
        
        # Create ID mapping
        id_to_idx = {donor_id: idx for idx, donor_id in enumerate(all_donor_ids)}
        
        # Build edge index
        edge_indices = []
        edge_attrs = []
        
        for _, row in relationships_df.iterrows():
            src_idx = id_to_idx[row['Donor_ID_1']]
            dst_idx = id_to_idx[row['Donor_ID_2']]
            
            # Add both directions for undirected graph
            edge_indices.append([src_idx, dst_idx])
            edge_indices.append([dst_idx, src_idx])
            
            # Edge attributes: strength and type encoding
            strength = row['Relationship_Strength']
            rel_type = row['Relationship_Type']
            
            # Simple type encoding (can be improved)
            type_encoding = hash(rel_type) % 10  # Simple hash-based encoding
            
            edge_attrs.append([strength, type_encoding])
            edge_attrs.append([strength, type_encoding])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # Get donor features
        donors_df = self.get_donors(donor_ids=all_donor_ids)
        enhanced_df = self.get_enhanced_fields(donor_ids=all_donor_ids)
        
        # Merge donor and enhanced data
        merged_df = donors_df.merge(enhanced_df, on='ID', how='left')
        
        # Select numeric features for node features
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        node_features = merged_df[numeric_cols].fillna(0).values
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            donor_ids=torch.tensor(all_donor_ids, dtype=torch.long)
        )
    
    def get_database_stats(self) -> Dict[str, int]:
        """
        Get database statistics
        
        Returns:
            Dictionary with table record counts
        """
        tables = ['donors', 'relationships', 'family_relationships', 
                 'event_attendance', 'giving_history', 'contact_reports', 'enhanced_fields']
        
        stats = {}
        for table in tables:
            query = f"SELECT COUNT(*) as count FROM {table}"
            result = self.conn.execute(query).fetchone()
            stats[table] = result['count']
        
        return stats
    
    def close(self):
        """Close database connection"""
        self._disconnect()


# Convenience functions for easy usage
def load_donor_data(db_path: str = "data/synthetic_donor_dataset_500k_dense/donor_database.db",
                   limit: Optional[int] = None) -> pd.DataFrame:
    """
    Quick function to load donor data
    
    Args:
        db_path: Path to database
        limit: Maximum number of records
        
    Returns:
        DataFrame with donor data
    """
    with SQLDataLoader(db_path) as loader:
        return loader.get_donors(limit=limit)

def load_multimodal_data(db_path: str = "data/synthetic_donor_dataset_500k_dense/donor_database.db",
                        donor_ids: Optional[List[int]] = None,
                        limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Quick function to load all multimodal data
    
    Args:
        db_path: Path to database
        donor_ids: Specific donor IDs to load
        
    Returns:
        Dictionary with all data modalities
    """
    with SQLDataLoader(db_path) as loader:
        return loader.get_multimodal_data(donor_ids=donor_ids, limit=limit)

def build_donor_graph(db_path: str = "data/synthetic_donor_dataset_500k_dense/donor_database.db",
                     donor_ids: Optional[List[int]] = None,
                     min_strength: float = 0.1) -> Data:
    """
    Quick function to build donor relationship graph
    
    Args:
        db_path: Path to database
        donor_ids: Specific donor IDs to include
        min_strength: Minimum relationship strength
        
    Returns:
        PyTorch Geometric Data object
    """
    with SQLDataLoader(db_path) as loader:
        return loader.build_graph_data(donor_ids=donor_ids, min_strength=min_strength)
