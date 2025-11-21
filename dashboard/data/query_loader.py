"""
Query-based data loader for efficient memory usage.
Loads data from SQLite on-demand instead of loading entire dataset into memory.
"""

import pandas as pd
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import os

# Optional Streamlit import
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


class QueryBasedLoader:
    """
    A DataFrame-like interface that queries SQLite on-demand.
    This allows working with large datasets without loading everything into memory.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize the query-based loader.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        # Cache for column names and metadata
        self._columns_cache = None
        self._row_count_cache = None
        
        # Get column mapping for standardizing column names
        from dashboard.config import settings
        self.column_mapping = settings.COLUMN_MAPPING.copy()
        self._extended_mapping = {
            'Donor_ID': 'donor_id',
            'donorid': 'donor_id',
            'ID': 'donor_id',
            'prediction': 'predicted_prob',
            'probability': 'predicted_prob',
            'score': 'predicted_prob',
            'Legacy_Intent_Binary': 'actual_gave',
            'legacy_intent_binary': 'actual_gave',
            'gave': 'actual_gave',
            'label': 'actual_gave',
            'target': 'actual_gave',
            'y': 'actual_gave',
            'Days_Since_Last_Gift': 'days_since_last',
            'days_since_last_gift': 'days_since_last',
            'Last_Gift_Date': 'last_gift_date',
            'Lifetime_Giving': 'total_giving',
            'lifetime_giving': 'total_giving',
            'Lifetime Giving': 'total_giving',
            'LifetimeGiving': 'total_giving',
            'total_amount': 'total_giving',
            'Average_Gift': 'avg_gift',
            'average_gift': 'avg_gift',
            'avg_gift_amount': 'avg_gift',
            'Avg_Gift_Amount': 'avg_gift',
            'Num_Gifts': 'gift_count',
            'num_gifts': 'gift_count',
            'gifts': 'gift_count',
            'RFM_Score': 'rfm_score',
            'rfm': 'rfm_score',
            'Recency_Score': 'recency_score',
            'Frequency_Score': 'frequency_score',
            'Monetary_Score': 'monetary_score',
            'Years_Active': 'years_active',
            'Consecutive_Years': 'consecutive_years',
            'Geographic_Region': 'region',
            'Region': 'region',
            'Primary_Constituent_Type': 'donor_type',
            'Donor_Type': 'donor_type',
            'type': 'donor_type',
            'Full_Name': 'donor_name',
            'full_name': 'donor_name',
            'Name': 'donor_name',
            'name': 'donor_name'
        }
        self.column_mapping.update(self._extended_mapping)
    
    def _get_connection(self):
        """Get a SQLite connection."""
        return sqlite3.connect(str(self.db_path))
    
    @property
    def columns(self) -> List[str]:
        """Get standardized column names (after mapping)."""
        if self._columns_cache is None:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT * FROM donors LIMIT 0")
                db_columns = [description[0] for description in cursor.description]
                # Map to standardized names
                standardized = []
                for db_col in db_columns:
                    if db_col in self.column_mapping:
                        standardized.append(self.column_mapping[db_col])
                    else:
                        standardized.append(db_col)
                self._columns_cache = standardized
            finally:
                conn.close()
        return self._columns_cache.copy()
    
    def __contains__(self, key: str) -> bool:
        """Check if a column exists (using standardized names)."""
        return key in self.columns
    
    def _map_where_clause(self, where: str) -> str:
        """
        Map standardized column names in WHERE clause to database column names.
        
        Args:
            where: WHERE clause with potentially standardized column names
            
        Returns:
            WHERE clause with database column names
        """
        # Get actual database column names
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM donors LIMIT 0")
            db_columns = [description[0] for description in cursor.description]
        finally:
            conn.close()
        
        # Create reverse mapping: standardized -> database
        reverse_map = {}
        for db_col in db_columns:
            if db_col in self.column_mapping:
                std_name = self.column_mapping[db_col]
                reverse_map[std_name] = db_col
            else:
                reverse_map[db_col] = db_col
        
        # Replace standardized names with database names in WHERE clause
        mapped_where = where
        for std_name, db_name in reverse_map.items():
            if std_name != db_name:
                # Replace column references in WHERE clause
                # Match column names that are quoted or unquoted
                import re
                # Pattern: "column_name" or column_name (not part of a string)
                pattern = rf'\b"{re.escape(std_name)}"|\b{re.escape(std_name)}\b'
                mapped_where = re.sub(pattern, f'"{db_name}"', mapped_where)
        
        return mapped_where
    
    def __len__(self) -> int:
        """Get total row count."""
        if self._row_count_cache is None:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM donors")
                self._row_count_cache = cursor.fetchone()[0]
            finally:
                conn.close()
        return self._row_count_cache
    
    def query(self, 
              where: Optional[str] = None,
              columns: Optional[List[str]] = None,
              limit: Optional[int] = None,
              order_by: Optional[str] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.
        
        Args:
            where: WHERE clause (without 'WHERE' keyword) - use actual column names from database
            columns: List of columns to select (None = all columns)
            limit: Maximum number of rows to return
            order_by: ORDER BY clause (without 'ORDER BY' keyword) - use actual column names from database
            
        Returns:
            pd.DataFrame with query results (with standardized column names)
        """
        conn = self._get_connection()
        try:
            # Build SELECT clause
            if columns:
                # Map standardized names back to database column names
                db_cols = []
                for col in columns:
                    # Find the database column name (reverse mapping)
                    db_col = None
                    for db_name, std_name in self.column_mapping.items():
                        if std_name == col and db_name in self.columns:
                            db_col = db_name
                            break
                    if not db_col:
                        # Try direct match
                        db_col = col if col in self.columns else None
                    if db_col:
                        db_cols.append(db_col)
                select_clause = ", ".join([f'"{col}"' for col in db_cols]) if db_cols else "*"
            else:
                select_clause = "*"
            
            # Build query
            query = f"SELECT {select_clause} FROM donors"
            
            if where:
                # Map standardized column names to database column names
                mapped_where = self._map_where_clause(where)
                query += f" WHERE {mapped_where}"
            
            if order_by:
                # Map standardized column names to database column names
                mapped_order_by = self._map_where_clause(order_by)  # Same mapping logic works
                query += f" ORDER BY {mapped_order_by}"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn)
            
            # Apply column mapping to standardize column names
            rename_dict = {}
            for db_col in df.columns:
                if db_col in self.column_mapping:
                    rename_dict[db_col] = self.column_mapping[db_col]
            if rename_dict:
                df = df.rename(columns=rename_dict)
            
            return self._process_dataframe(df)
        finally:
            conn.close()
    
    def filter(self, 
               regions: Optional[List[str]] = None,
               donor_types: Optional[List[str]] = None,
               segments: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter data by regions, donor_types, and segments.
        
        Args:
            regions: List of regions to include
            donor_types: List of donor types to include
            segments: List of segments to include
            
        Returns:
            Filtered DataFrame
        """
        conditions = []
        
        if regions:
            region_col = 'region' if 'region' in self.columns else 'Geographic_Region'
            placeholders = ",".join(["?" for _ in regions])
            conditions.append(f'"{region_col}" IN ({placeholders})')
        
        if donor_types:
            donor_type_col = 'donor_type' if 'donor_type' in self.columns else 'Primary_Constituent_Type'
            placeholders = ",".join(["?" for _ in donor_types])
            conditions.append(f'"{donor_type_col}" IN ({placeholders})')
        
        if segments:
            segment_col = 'segment' if 'segment' in self.columns else 'Segment'
            placeholders = ",".join(["?" for _ in segments])
            conditions.append(f'"{segment_col}" IN ({placeholders})')
        
        where_clause = " AND ".join(conditions) if conditions else None
        
        # Build parameters for query
        params = []
        if regions:
            params.extend(regions)
        if donor_types:
            params.extend(donor_types)
        if segments:
            params.extend(segments)
        
        conn = self._get_connection()
        try:
            query = "SELECT * FROM donors"
            if where_clause:
                query += f" WHERE {where_clause}"
            
            df = pd.read_sql_query(query, conn, params=params if params else None)
            return self._process_dataframe(df)
        finally:
            conn.close()
    
    def get_aggregate(self,
                     group_by: Optional[str] = None,
                     aggregations: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Get aggregated data.
        
        Args:
            group_by: Column to group by
            aggregations: Dict of {column: aggregation_function}
                          e.g., {'total_giving': 'SUM', 'predicted_prob': 'AVG'}
        
        Returns:
            Aggregated DataFrame
        """
        conn = self._get_connection()
        try:
            if not aggregations:
                aggregations = {}
            
            # Build SELECT clause
            select_parts = []
            if group_by:
                select_parts.append(f'"{group_by}"')
            
            for col, func in aggregations.items():
                select_parts.append(f'{func}("{col}") AS "{col}_{func.lower()}"')
            
            query = f"SELECT {', '.join(select_parts)} FROM donors"
            
            if group_by:
                query += f" GROUP BY \"{group_by}\""
            
            df = pd.read_sql_query(query, conn)
            return self._process_dataframe(df)
        finally:
            conn.close()
    
    def get_top_n(self, 
                  n: int,
                  order_by: str,
                  where: Optional[str] = None,
                  ascending: bool = False) -> pd.DataFrame:
        """
        Get top N rows ordered by a column.
        
        Args:
            n: Number of rows to return
            order_by: Column to order by
            where: WHERE clause (optional)
            ascending: Sort order
            
        Returns:
            DataFrame with top N rows
        """
        order_direction = "ASC" if ascending else "DESC"
        query = f"SELECT * FROM donors"
        
        if where:
            query += f" WHERE {where}"
        
        query += f" ORDER BY \"{order_by}\" {order_direction} LIMIT {n}"
        
        conn = self._get_connection()
        try:
            df = pd.read_sql_query(query, conn)
            return self._process_dataframe(df)
        finally:
            conn.close()
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and standardize the dataframe columns.
        Similar to process_dataframe in loader.py but adapted for query results.
        """
        # Apply column mapping
        existing_renames = {k: v for k, v in self.column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_renames)
        
        # Ensure donor_id exists
        if 'donor_id' not in df.columns:
            df['donor_id'] = [f'D{i:06d}' for i in range(len(df))]
        
        # Handle predicted_prob
        if 'Will_Give_Again_Probability' in df.columns:
            df['predicted_prob'] = df['Will_Give_Again_Probability'].copy()
        elif 'predicted_prob' not in df.columns:
            # Look for probability columns
            prob_cols = [col for col in df.columns if 'prob' in col.lower() or 'score' in col.lower()]
            if prob_cols:
                df['predicted_prob'] = df[prob_cols[0]].copy()
        
        # Normalize predicted_prob if needed
        if 'predicted_prob' in df.columns:
            prob_series = pd.to_numeric(df['predicted_prob'], errors='coerce')
            if prob_series.max() > 1.0:
                df['predicted_prob'] = prob_series / 100.0
            else:
                df['predicted_prob'] = prob_series
        
        # Ensure numeric types for common columns
        numeric_cols = ['total_giving', 'avg_gift', 'gift_count', 'days_since_last', 
                       'rfm_score', 'recency_score', 'frequency_score', 'monetary_score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def to_dataframe(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Convert to full DataFrame (use with caution - may cause OOM).
        Only use for small datasets or when absolutely necessary.
        
        Args:
            limit: Maximum number of rows to load
            
        Returns:
            Full DataFrame
        """
        if limit:
            return self.query(limit=limit)
        else:
            conn = self._get_connection()
            try:
                df = pd.read_sql_query("SELECT * FROM donors", conn)
                return self._process_dataframe(df)
            finally:
                conn.close()


def get_query_loader(db_path: Optional[Path] = None) -> Optional[QueryBasedLoader]:
    """
    Get a QueryBasedLoader instance if SQLite database is available.
    
    Args:
        db_path: Optional path to SQLite database. If None, searches for cached database.
        
    Returns:
        QueryBasedLoader instance or None if database not found
    """
    if db_path is None:
        # Check for cached SQLite database
        kaggle_download_dir = Path(os.getenv("KAGGLE_DOWNLOAD_DIR", "/tmp/kaggle_data"))
        cached_sqlite = kaggle_download_dir / "donors_cached.db"
        
        if cached_sqlite.exists():
            db_path = cached_sqlite
        else:
            # Check other common locations
            from dashboard.config import settings
            data_paths = settings.get_data_paths()
            for sqlite_path in data_paths['sqlite_paths']:
                if os.path.exists(sqlite_path):
                    db_path = Path(sqlite_path)
                    break
    
    if db_path and Path(db_path).exists():
        try:
            return QueryBasedLoader(db_path)
        except Exception as e:
            if STREAMLIT_AVAILABLE:
                st.sidebar.warning(f"Failed to initialize query loader: {e}")
            return None
    
    return None

