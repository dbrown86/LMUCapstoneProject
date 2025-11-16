"""
Data loading module for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from typing import Optional

# Import config for paths
from dashboard.config import settings

# Optional Streamlit import - only used if available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create a mock st object for testing without Streamlit
    class MockStreamlit:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    st = MockStreamlit()


def load_full_dataset(use_cache: bool = True):
    """
    Load the complete 500K donor dataset from Parquet or SQLite - OPTIMIZED
    
    Args:
        use_cache: If True and Streamlit is available, use Streamlit caching
    
    Returns:
        pd.DataFrame: Processed donor dataset
    """
    # Use Streamlit caching if available and requested - optimized for performance
    if use_cache and STREAMLIT_AVAILABLE:
        @st.cache_data(show_spinner=False, ttl=7200, max_entries=1)  # 2 hour cache, no spinner, single entry
        def _load_cached():
            return _load_full_dataset_internal()
        return _load_cached()
    else:
        return _load_full_dataset_internal()


def _load_full_dataset_internal():
    """Internal function to load dataset (without caching decorator)."""
    root = settings.get_project_root()
    data_dir_env = os.getenv("LMU_DATA_DIR")
    env_dir = Path(data_dir_env).resolve() if data_dir_env else None
    
    # Get paths from config
    data_paths = settings.get_data_paths()
    parquet_paths = data_paths['parquet_paths']
    sqlite_paths = data_paths['sqlite_paths']
    csv_dir_candidates = data_paths['csv_dir_candidates']
    
    # Priority 1: Try Parquet file (fastest - use pyarrow engine)
    for path in parquet_paths:
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path, engine='pyarrow')
                return process_dataframe(df)
            except Exception as e:
                if STREAMLIT_AVAILABLE:
                    st.sidebar.warning(f"Failed to load {path}: {e}")
    
    # Priority 2: Try SQLite database (load full table)
    for db_path in sqlite_paths:
        if os.path.exists(db_path):
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query("SELECT * FROM donors", conn)
                conn.close()
                return process_dataframe(df)
            except Exception as e:
                if STREAMLIT_AVAILABLE:
                    st.sidebar.warning(f"Failed to load {db_path}: {e}")
    
    # Priority 3: Try CSV parts (load fewer files for speed)
    csv_dir = next((p for p in csv_dir_candidates if os.path.exists(p)), None)
    if csv_dir and os.path.exists(csv_dir):
        try:
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            if csv_files:
                dfs = []
                # Only load first 5 files for speed
                for csv_file in csv_files[:5]:
                    df_part = pd.read_csv(os.path.join(csv_dir, csv_file))
                    dfs.append(df_part)
                df = pd.concat(dfs, ignore_index=True)
                return process_dataframe(df)
        except Exception as e:
            if STREAMLIT_AVAILABLE:
                st.sidebar.warning(f"Failed to load CSV parts: {e}")
    
    # Priority 4: Glob search for Parquet across project and env dir
    parquet_patterns = []
    if env_dir:
        parquet_patterns.append(str(env_dir / "**/*.parquet"))
    parquet_patterns.extend([
        str(root / "data/**/*.parquet"),
        str(root / "**/donors*.parquet"),
    ])
    for pattern in parquet_patterns:
        try:
            for p in glob.glob(pattern, recursive=True):
                try:
                    df = pd.read_parquet(p, engine='pyarrow')
                    return process_dataframe(df)
                except Exception:
                    continue
        except Exception:
            pass
    
    # Priority 5: Glob search for SQLite DBs and load donors table if present
    db_patterns = []
    if env_dir:
        db_patterns.append(str(env_dir / "**/*.db"))
    db_patterns.extend([
        str(root / "data/**/*.db"),
        str(root / "**/*.db"),
    ])
    for pattern in db_patterns:
        try:
            for p in glob.glob(pattern, recursive=True):
                try:
                    import sqlite3
                    conn = sqlite3.connect(p)
                    # Check if donors table exists
                    try:
                        exists = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name='donors'", conn)
                        if not exists.empty:
                            df = pd.read_sql_query("SELECT * FROM donors", conn)
                            conn.close()
                            return process_dataframe(df)
                    finally:
                        conn.close()
                except Exception:
                    continue
        except Exception:
            pass
    
    # Fallback: Generate sample data
    if STREAMLIT_AVAILABLE:
        st.sidebar.error("⚠️ Could not load dataset. Using sample data.")
    return generate_sample_data()


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and standardize the dataframe columns - OPTIMIZED
    
    Args:
        df: Raw dataframe from data source
        
    Returns:
        pd.DataFrame: Processed dataframe with standardized columns
    """
    # Get column mapping from config
    column_mapping = settings.COLUMN_MAPPING.copy()
    
    # CRITICAL: Check for Gave_Again_In_2025 FIRST (primary target), then 2025 (fallback)
    gave_again_2025 = None
    gave_again_2024 = None
    if 'Gave_Again_In_2025' in df.columns:
        gave_again_2025 = df['Gave_Again_In_2025'].copy()
    if 'Gave_Again_In_2024' in df.columns:
        gave_again_2024 = df['Gave_Again_In_2024'].copy()
    
    # CRITICAL: Only map Legacy_Intent_Probability if Will_Give_Again_Probability doesn't exist
    if 'Will_Give_Again_Probability' not in df.columns:
        column_mapping['Legacy_Intent_Probability'] = 'predicted_prob'
        column_mapping['legacy_intent_probability'] = 'predicted_prob'
    
    # Extended column mapping (from original file)
    extended_mapping = {
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
    
    # Merge mappings
    column_mapping.update(extended_mapping)
    
    # Only rename columns that exist
    existing_renames = {k: v for k, v in column_mapping.items() if k in df.columns}
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
        else:
            df['predicted_prob'] = np.random.beta(2, 5, len(df))
    
    # Normalize predicted_prob if needed
    if 'predicted_prob' in df.columns:
        prob_max = df['predicted_prob'].max()
        prob_min = df['predicted_prob'].min()
        if prob_max > 1.0:
            df['predicted_prob'] = df['predicted_prob'] / prob_max
        elif prob_min < 0.0:
            df['predicted_prob'] = df['predicted_prob'] - prob_min
            if df['predicted_prob'].max() > 1.0:
                df['predicted_prob'] = df['predicted_prob'] / df['predicted_prob'].max()
    
    # Handle actual_gave - prioritize 2025, fallback to 2025
    if gave_again_2025 is not None:
        df['actual_gave'] = gave_again_2025.astype(int)
        df['Gave_Again_In_2025'] = gave_again_2025.copy()
        # Keep 2025 for backward compatibility if it exists
        if gave_again_2024 is not None:
            df['Gave_Again_In_2024'] = gave_again_2024.copy()
    elif gave_again_2024 is not None:
        df['actual_gave'] = gave_again_2024.astype(int)
        df['Gave_Again_In_2024'] = gave_again_2024.copy()
    elif 'Gave_Again_In_2025' in df.columns:
        df['actual_gave'] = df['Gave_Again_In_2025'].astype(int)
    elif 'Gave_Again_In_2024' in df.columns:
        df['actual_gave'] = df['Gave_Again_In_2024'].astype(int)
    elif 'actual_gave' not in df.columns:
        # Try to compute from giving_history (prioritize 2025, fallback to 2025)
        try:
            giving_paths = settings.get_data_paths()['giving_paths']
            giving_df = None
            for path in giving_paths:
                if os.path.exists(path):
                    giving_df = pd.read_parquet(path, engine='pyarrow')
                    if 'Gift_Date' in giving_df.columns:
                        giving_df['Gift_Date'] = pd.to_datetime(giving_df['Gift_Date'], errors='coerce')
                    break
            
            if giving_df is not None and 'Gift_Date' in giving_df.columns:
                # Try 2025 first
                giving_2025 = giving_df[giving_df['Gift_Date'] >= '2025-01-01'].copy()
                donors_2025 = giving_2025['Donor_ID'].unique() if 'Donor_ID' in giving_2025.columns else []
                donor_id_col = next((c for c in ['ID', 'Donor_ID', 'donor_id'] if c in df.columns), None)
                
                if donor_id_col and len(donors_2025) > 0:
                    df['actual_gave'] = df[donor_id_col].isin(donors_2025).astype(int)
                    df['Gave_Again_In_2025'] = df['actual_gave'].copy()
                else:
                    # Fallback to 2025
                    giving_2024 = giving_df[giving_df['Gift_Date'] >= '2025-01-01'].copy()
                    giving_2024 = giving_2024[giving_2024['Gift_Date'] < '2025-01-01']
                    donors_2024 = giving_2024['Donor_ID'].unique() if 'Donor_ID' in giving_2024.columns else []
                    if donor_id_col and len(donors_2024) > 0:
                        df['actual_gave'] = df[donor_id_col].isin(donors_2024).astype(int)
                        df['Gave_Again_In_2024'] = df['actual_gave'].copy()
                    else:
                        df['actual_gave'] = np.random.binomial(1, 0.17, len(df))
            else:
                df['actual_gave'] = np.random.binomial(1, 0.17, len(df))
        except Exception:
            df['actual_gave'] = np.random.binomial(1, 0.17, len(df))
    else:
        df['actual_gave'] = df['actual_gave'].astype(int)
    
    # Calculate days_since_last from date if needed
    if 'days_since_last' in df.columns:
        df = df.drop(columns=['days_since_last'])
    
    days_since_last_created = False
    gift_date_col = None
    for col_name in ['last_gift_date', 'Last_Gift_Date', 'LastGiftDate', 'lastGiftDate']:
        if col_name in df.columns:
            gift_date_col = col_name
            break
    
    if gift_date_col:
        try:
            date_series = pd.to_datetime(df[gift_date_col], errors='coerce')
            today = pd.Timestamp.now()
            df['days_since_last'] = (today - date_series).dt.days.clip(lower=0)
            if df['days_since_last'].notna().sum() > 0:
                days_since_last_created = True
        except Exception:
            pass
    
    # Fallback for days_since_last
    if not days_since_last_created:
        if 'Days_Since_Last_Gift' in df.columns or 'days_since_last_gift' in df.columns:
            days_col = 'Days_Since_Last_Gift' if 'Days_Since_Last_Gift' in df.columns else 'days_since_last_gift'
            df['days_since_last'] = pd.to_numeric(df[days_col], errors='coerce').clip(lower=0)
        else:
            df['days_since_last'] = 365
    
    # Fill missing numeric columns
    numeric_defaults = {
        'total_giving': 5000,
        'avg_gift': 500,
        'gift_count': 3,
        'rfm_score': 3,
        'recency_score': 3,
        'frequency_score': 3,
        'monetary_score': 3,
        'years_active': 2,
        'consecutive_years': 1
    }
    
    for col, default_val in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            try:
                series = df[col]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]

                if series.dtype == object:
                    series = series.replace({r'[^\d\.\-]': ''}, regex=True)

                df[col] = pd.to_numeric(series, errors='coerce').fillna(default_val)
            except (TypeError, AttributeError):
                df[col] = default_val
    
    # Add region and donor_type if missing
    if 'region' not in df.columns:
        df['region'] = np.random.choice(
            ['Northeast', 'Southeast', 'Midwest', 'West', 'Southwest'],
            len(df)
        )
    
    if 'donor_type' not in df.columns:
        df['donor_type'] = np.random.choice(
            ['Individual', 'Corporate', 'Foundation', 'Government'],
            len(df),
            p=[0.7, 0.15, 0.1, 0.05]
        )
    
    # Create segment
    all_segments = [
        'Recent (0-6mo)',
        'Recent (6-12mo)',
        'Lapsed (1-2yr)',
        'Very Lapsed (2yr+)',
        'Prospects/New'
    ]

    if 'days_since_last' in df.columns and 'gift_count' in df.columns:
        days = pd.to_numeric(df['days_since_last'], errors='coerce')
        gifts = pd.to_numeric(df['gift_count'], errors='coerce').fillna(0)

        segments = np.full(len(df), 'Prospects/New', dtype=object)
        valid = (gifts > 0) & days.notna()
        within_bounds = valid & (days <= 3650)

        recent_0_6_mask = within_bounds & (days <= 180)
        recent_6_12_mask = within_bounds & (days > 180) & (days <= 365)
        lapsed_mask = within_bounds & (days > 365) & (days <= 730)
        very_lapsed_mask = within_bounds & (days > 730)

        segments[recent_0_6_mask] = 'Recent (0-6mo)'
        segments[recent_6_12_mask] = 'Recent (6-12mo)'
        segments[lapsed_mask] = 'Lapsed (1-2yr)'
        segments[very_lapsed_mask] = 'Very Lapsed (2yr+)'

        df['segment'] = pd.Categorical(segments, categories=all_segments)
    else:
        df['segment'] = pd.Categorical(
            np.full(len(df), 'Prospects/New', dtype=object),
            categories=all_segments
        )
    
    return df


def generate_sample_data() -> pd.DataFrame:
    """Fallback: Generate sample data if real data cannot be loaded"""
    np.random.seed(42)
    n = 50000
    
    df = pd.DataFrame({
        'donor_id': [f'D{i:06d}' for i in range(n)],
        'predicted_prob': np.random.beta(2, 5, n),
        'actual_gave': np.random.binomial(1, 0.17, n),
        'days_since_last': np.clip(np.random.exponential(365, n), 0, 2000),
        'total_giving': np.clip(np.random.lognormal(6, 2, n), 10, 100000),
        'avg_gift': np.clip(np.random.lognormal(5.5, 1.5, n), 10, 10000),
        'gift_count': np.clip(np.random.poisson(3, n), 0, 50),
        'rfm_score': np.random.uniform(1, 5, n),
        'recency_score': np.random.randint(1, 6, n),
        'frequency_score': np.random.randint(1, 6, n),
        'monetary_score': np.random.randint(1, 6, n),
        'years_active': np.random.randint(0, 11, n),
        'consecutive_years': np.random.randint(0, 6, n),
        'region': np.random.choice(['Northeast', 'Southeast', 'Midwest', 'West', 'Southwest'], n),
        'donor_type': np.random.choice(['Individual', 'Corporate', 'Foundation', 'Government'], n, p=[0.7, 0.15, 0.1, 0.05])
    })
    
    df['segment'] = pd.cut(
        df['days_since_last'],
        bins=[0, 180, 365, 730, np.inf],
        labels=['Recent (0-6mo)', 'Recent (6-12mo)', 'Lapsed (1-2yr)', 'Very Lapsed (2yr+)']
    )
    
    return df

