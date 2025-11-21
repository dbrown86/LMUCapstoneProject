"""
Data loading module for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

import pandas as pd
import numpy as np
import os
import glob
import subprocess
import shutil
import json
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

KAGGLE_DATASET = os.getenv("KAGGLE_DATASET")
KAGGLE_DOWNLOAD_DIR = Path(os.getenv("KAGGLE_DOWNLOAD_DIR", "/tmp/kaggle_data")).resolve()
KAGGLE_KNOWN_CSVS = {"donors.csv", "contacts.csv", "events.csv", "family.csv", "giving.csv", "relationships.csv"}
KAGGLE_CACHED_PARQUET = KAGGLE_DOWNLOAD_DIR / "donors_cached.parquet"
KAGGLE_CACHED_SQLITE = KAGGLE_DOWNLOAD_DIR / "donors_cached.db"


def _download_kaggle_dataset_if_needed() -> Optional[Path]:
    """
    Download the Kaggle dataset (if configured) into a local directory so the dashboard
    can load the full 500K rows on Streamlit Cloud.

    Returns:
        Optional[Path]: Directory containing the extracted CSV/Parquet files, or None.
    """
    if not KAGGLE_DATASET:
        return None

    try:
        KAGGLE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if STREAMLIT_AVAILABLE:
            st.sidebar.warning(f"Failed to create Kaggle data directory: {exc}")
        return None

    # If files already exist, reuse them.
    if any(KAGGLE_DOWNLOAD_DIR.iterdir()):
        return KAGGLE_DOWNLOAD_DIR

    kaggle_cli = shutil.which("kaggle")
    if kaggle_cli is None:
        if STREAMLIT_AVAILABLE:
            st.sidebar.error("Kaggle CLI is not installed. Add 'kaggle' to requirements.txt.")
        return None

    # Check if Kaggle credentials are configured
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    
    if not kaggle_username or not kaggle_key:
        if STREAMLIT_AVAILABLE:
            st.sidebar.error("⚠️ Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY in Streamlit secrets.")
        return None
    
    # Ensure Kaggle credentials directory exists
    kaggle_creds_dir = Path.home() / ".kaggle"
    kaggle_creds_dir.mkdir(exist_ok=True)
    kaggle_creds_file = kaggle_creds_dir / "kaggle.json"
    
    # Write credentials if not already present or if they've changed
    creds = {"username": kaggle_username, "key": kaggle_key}
    needs_update = True
    if kaggle_creds_file.exists():
        try:
            existing_creds = json.loads(kaggle_creds_file.read_text())
            if existing_creds == creds:
                needs_update = False
        except (json.JSONDecodeError, IOError):
            pass  # File exists but is invalid, will overwrite
    
    if needs_update:
        kaggle_creds_file.write_text(json.dumps(creds))
        try:
            kaggle_creds_file.chmod(0o600)  # Restrict permissions (Unix only)
        except (OSError, AttributeError):
            pass  # chmod not available on Windows or file system doesn't support it
    
    cmd = [
        kaggle_cli,
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(KAGGLE_DOWNLOAD_DIR),
        "--unzip",
    ]

    try:
        if STREAMLIT_AVAILABLE:
            st.sidebar.info(f"📥 Downloading dataset from Kaggle: {KAGGLE_DATASET}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        if STREAMLIT_AVAILABLE:
            st.sidebar.success("✅ Kaggle dataset downloaded successfully")
        return KAGGLE_DOWNLOAD_DIR
    except subprocess.TimeoutExpired:
        if STREAMLIT_AVAILABLE:
            st.sidebar.error("⏱️ Kaggle download timed out after 10 minutes")
        return None
    except subprocess.CalledProcessError as err:
        # Capture both stdout and stderr for better error messages
        # Handle both string and bytes (depending on subprocess configuration)
        def safe_decode(value):
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="ignore")
            return str(value)
        
        stdout_msg = safe_decode(err.stdout) if err.stdout else ""
        stderr_msg = safe_decode(err.stderr) if err.stderr else ""
        error_msg = f"{stderr_msg}\n{stdout_msg}".strip() if stderr_msg or stdout_msg else str(err)
        
        if STREAMLIT_AVAILABLE:
            st.sidebar.error(f"❌ Failed to download Kaggle dataset")
            st.sidebar.error(f"Error details: {error_msg}")
            # Provide helpful troubleshooting info
            if "401" in error_msg or "Unauthorized" in error_msg:
                st.sidebar.warning("💡 Check that KAGGLE_USERNAME and KAGGLE_KEY are correct in Streamlit secrets")
            elif "404" in error_msg or "not found" in error_msg.lower():
                st.sidebar.warning(f"💡 Verify dataset name is correct: {KAGGLE_DATASET}")
        return None


def load_full_dataset(use_cache: bool = True):
    """
    Load the complete 500K donor dataset from Parquet or SQLite - OPTIMIZED
    
    Args:
        use_cache: If True and Streamlit is available, use Streamlit caching
    
    Returns:
        pd.DataFrame or QueryBasedLoader: Processed donor dataset or query-based loader
    """
    # Check if query-based loading is enabled (prevents OOM)
    use_query_loader = os.getenv("STREAMLIT_USE_QUERY_LOADER", "true").lower() == "true"
    
    if use_query_loader:
        # Try to get query-based loader (only loads data on-demand)
        from dashboard.data.query_loader import get_query_loader
        try:
            query_loader = get_query_loader()
            if query_loader is not None:
                # Test that the loader works by checking row count
                try:
                    row_count = len(query_loader)
                    if STREAMLIT_AVAILABLE:
                        st.sidebar.success(f"✅ Using query-based loader (full {row_count:,} rows available on-demand)")
                    return query_loader
                except Exception as e:
                    if STREAMLIT_AVAILABLE:
                        st.sidebar.warning(f"Query loader found but not usable: {e}. Falling back to traditional loading.")
        except Exception as e:
            if STREAMLIT_AVAILABLE:
                st.sidebar.info(f"Query-based loading not available: {e}. Using traditional loading.")
    
    # Fallback to traditional loading (loads full dataset into memory)
    if use_cache and STREAMLIT_AVAILABLE:
        @st.cache_data(show_spinner=False, ttl=7200, max_entries=1)  # 2 hour cache, no spinner, single entry
        def _load_cached():
            return _load_full_dataset_internal()
        return _load_cached()
    else:
        return _load_full_dataset_internal()


def _resolve_kaggle_csv_dir() -> Optional[Path]:
    """
    Locate the directory that actually contains the Kaggle CSV files,
    accounting for Kaggle's tendency to unzip into a subfolder.
    """
    if not KAGGLE_DOWNLOAD_DIR.exists():
        return None
    # If CSVs exist at top level, use that.
    top_level_csvs = list(KAGGLE_DOWNLOAD_DIR.glob("*.csv"))
    if top_level_csvs:
        return KAGGLE_DOWNLOAD_DIR
    # Otherwise search subdirectories for the known CSV names.
    for subdir in KAGGLE_DOWNLOAD_DIR.glob("**"):
        if not subdir.is_dir():
            continue
        csvs = {p.name for p in subdir.glob("*.csv")}
        if KAGGLE_KNOWN_CSVS.intersection(csvs):
            return subdir
    return None


def _convert_kaggle_csv_to_sqlite(csv_dir: Path) -> Optional[Path]:
    """
    Convert Kaggle donors.csv to SQLite database in chunks.
    This allows loading the FULL 500K row dataset without OOM issues.
    SQLite can handle large datasets efficiently and we can query only what's needed.
    
    Args:
        csv_dir: Directory containing the Kaggle CSV files
        
    Returns:
        Path to the created SQLite database, or None if conversion failed
    """
    # Use the global cached SQLite path
    sqlite_path = KAGGLE_CACHED_SQLITE
    
    # If SQLite already exists, use it
    if sqlite_path.exists():
        return sqlite_path
    
    # Find donors.csv
    donor_csv = None
    for name in ["donors.csv", "donors_with_network_features.csv", "donors_full.csv"]:
        candidate = csv_dir / name
        if candidate.exists():
            donor_csv = candidate
            break
    
    if not donor_csv or not donor_csv.exists():
        return None
    
    try:
        import sqlite3
        
        if STREAMLIT_AVAILABLE:
            st.sidebar.info("🔄 Converting CSV to SQLite (processing full dataset in chunks)...")
        
        # Ensure directory exists
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove existing database if it exists (incomplete)
        if sqlite_path.exists():
            sqlite_path.unlink()
        
        # Create SQLite database
        conn = sqlite3.connect(str(sqlite_path))
        
        # Read CSV in chunks and write to SQLite (handles full 500K rows without OOM)
        chunk_size = 50000
        first_chunk = True
        total_rows = 0
        
        for chunk in pd.read_csv(donor_csv, chunksize=chunk_size, low_memory=False):
            # Write chunk to SQLite (creates table on first chunk, appends on subsequent)
            chunk.to_sql('donors', conn, if_exists='append' if not first_chunk else 'replace', index=False)
            first_chunk = False
            total_rows += len(chunk)
            
            if STREAMLIT_AVAILABLE and total_rows % 100000 == 0:
                st.sidebar.info(f"   Processed {total_rows:,} rows...")
        
        # Create indexes for common query columns to speed up queries
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prob ON donors(Will_Give_Again_Probability)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_segment ON donors(segment)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_region ON donors(region)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_donor_type ON donors(donor_type)")
        except Exception:
            # Index creation is optional, continue if it fails
            pass
        
        conn.close()
        
        if STREAMLIT_AVAILABLE:
            st.sidebar.success(f"✅ SQLite database created ({total_rows:,} rows) - Full dataset available!")
        
        return sqlite_path
    except Exception as e:
        if STREAMLIT_AVAILABLE:
            st.sidebar.warning(f"Failed to convert CSV to SQLite: {e}")
        return None


def _convert_kaggle_csv_to_parquet(csv_dir: Path) -> Optional[Path]:
    """
    Convert Kaggle donors.csv to Parquet format for efficient loading.
    This reduces memory usage by:
    1. Sampling only a subset of rows (default 100K, configurable via MAX_ROWS env var)
    2. Selecting only essential columns used by the dashboard
    
    Args:
        csv_dir: Directory containing the Kaggle CSV files
        
    Returns:
        Path to the created Parquet file, or None if conversion failed
    """
    # Use the global cached Parquet path
    parquet_path = KAGGLE_CACHED_PARQUET
    
    # If Parquet already exists, use it
    if parquet_path.exists():
        return parquet_path
    
    # Find donors.csv
    donor_csv = None
    for name in ["donors.csv", "donors_with_network_features.csv"]:
        candidate = csv_dir / name
        if candidate.exists():
            donor_csv = candidate
            break
    
    if not donor_csv or not donor_csv.exists():
        return None
    
    try:
        # Load CSV with optimized dtypes to reduce memory
        if STREAMLIT_AVAILABLE:
            st.sidebar.info("🔄 Converting CSV to Parquet (sampling rows and columns for memory efficiency)...")
        
        # Get max rows from environment (default 100K to prevent OOM)
        max_rows = int(os.getenv("STREAMLIT_MAX_ROWS", "100000"))
        
        # Essential columns used by the dashboard (based on code analysis)
        essential_columns = [
            # Core identifiers
            'donor_id', 'Donor_ID', 'donorid', 'ID',
            # Names
            'First_Name', 'first_name', 'First Name',
            'Last_Name', 'last_name', 'Last Name',
            'Full_Name', 'full_name', 'Name', 'name',
            # Predictions and outcomes
            'Will_Give_Again_Probability', 'predicted_prob', 'prediction', 'probability', 'score',
            'Gave_Again_In_2025', 'Gave_Again_In_2024', 'actual_gave', 'Legacy_Intent_Binary',
            'legacy_intent_binary', 'gave', 'label', 'target', 'y',
            # Giving metrics
            'total_giving', 'Lifetime_Giving', 'lifetime_giving', 'Lifetime Giving', 
            'LifetimeGiving', 'total_amount',
            'Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount',
            'avg_gift', 'Average_Gift', 'average_gift', 'avg_gift_amount', 'Avg_Gift_Amount',
            'gift_count', 'Num_Gifts', 'num_gifts', 'gifts',
            # Segmentation
            'segment', 'Segment',
            'region', 'Geographic_Region', 'Region',
            'donor_type', 'Primary_Constituent_Type', 'Donor_Type', 'type',
            # Dates and recency
            'days_since_last', 'Days_Since_Last_Gift', 'days_since_last_gift',
            'last_gift_date', 'Last_Gift_Date',
            # RFM scores
            'rfm_score', 'RFM_Score', 'rfm',
            'recency_score', 'Recency_Score',
            'frequency_score', 'Frequency_Score',
            'monetary_score', 'Monetary_Score',
            # Other useful features
            'years_active', 'Years_Active',
            'consecutive_years', 'Consecutive_Years',
            # Capacity and gift officer (for take_action page)
            'Rating', 'rating', 'capacity', 'Capacity',
            'Gift Officer', 'gift_officer', 'Gift_Officer',
        ]
        
        # Read CSV in chunks to sample rows efficiently
        chunk_size = 50000
        chunks = []
        total_rows_read = 0
        
        for chunk in pd.read_csv(donor_csv, chunksize=chunk_size, low_memory=False):
            # Select only essential columns that exist in this chunk
            available_cols = [col for col in essential_columns if col in chunk.columns]
            # Also keep any columns that match patterns (for feature analysis page)
            feature_cols = [col for col in chunk.columns if any(
                pattern in col.lower() for pattern in ['_count', '_sum', '_mean', '_max', '_min', 'network', 'degree']
            )]
            selected_cols = list(set(available_cols + feature_cols))
            
            if selected_cols:
                chunk = chunk[selected_cols]
            
            chunks.append(chunk)
            total_rows_read += len(chunk)
            
            # Stop if we've read enough rows
            if total_rows_read >= max_rows:
                # Trim the last chunk if needed
                if total_rows_read > max_rows:
                    excess = total_rows_read - max_rows
                    chunk = chunk.iloc[:-excess]
                    chunks[-1] = chunk
                break
        
        # Combine chunks
        if not chunks:
            if STREAMLIT_AVAILABLE:
                st.sidebar.error("No data chunks loaded")
            return None
        
        df = pd.concat(chunks, ignore_index=True)
        
        # Final column selection (in case some columns weren't in all chunks)
        final_cols = [col for col in essential_columns if col in df.columns]
        feature_cols = [col for col in df.columns if any(
            pattern in col.lower() for pattern in ['_count', '_sum', '_mean', '_max', '_min', 'network', 'degree']
        )]
        final_cols = list(set(final_cols + feature_cols))
        
        if final_cols:
            df = df[final_cols]
        
        # Ensure directory exists
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet (much more memory-efficient)
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
        
        if STREAMLIT_AVAILABLE:
            st.sidebar.success(f"✅ Parquet cache created ({len(df):,} rows, {len(df.columns)} columns)")
        
        return parquet_path
    except Exception as e:
        if STREAMLIT_AVAILABLE:
            st.sidebar.warning(f"Failed to convert CSV to Parquet: {e}")
        return None


def _load_full_dataset_internal():
    """Internal function to load dataset (without caching decorator)."""
    root = settings.get_project_root()
    data_dir_env = os.getenv("LMU_DATA_DIR")
    env_dir = Path(data_dir_env).resolve() if data_dir_env else None
    
    kaggle_dir = _download_kaggle_dataset_if_needed()

    # Get paths from config
    data_paths = settings.get_data_paths()
    parquet_paths = data_paths['parquet_paths']
    sqlite_paths = data_paths['sqlite_paths']
    csv_dir_candidates = data_paths['csv_dir_candidates']

    # If Kaggle dataset was downloaded, prioritize SQLite (full dataset) or Parquet (sampled)
    if kaggle_dir:
        resolved_csv_dir = _resolve_kaggle_csv_dir()
        if resolved_csv_dir:
            # Priority 1: Try SQLite (full dataset, no OOM)
            if KAGGLE_CACHED_SQLITE.exists():
                sqlite_paths.insert(0, str(KAGGLE_CACHED_SQLITE))
            else:
                # Convert CSV to SQLite (handles full 500K rows in chunks)
                cached_sqlite = _convert_kaggle_csv_to_sqlite(resolved_csv_dir)
                if cached_sqlite:
                    sqlite_paths.insert(0, str(cached_sqlite))
            
            # Priority 2: Fallback to Parquet (sampled, if SQLite fails)
            if KAGGLE_CACHED_PARQUET.exists():
                parquet_paths.insert(0, str(KAGGLE_CACHED_PARQUET))
            else:
                # Only create Parquet if SQLite conversion failed
                cached_parquet = _convert_kaggle_csv_to_parquet(resolved_csv_dir)
                if cached_parquet:
                    parquet_paths.insert(0, str(cached_parquet))
            
            csv_dir_candidates.insert(0, str(resolved_csv_dir))
    
    # Priority 1: Try SQLite database (full dataset)
    # Note: Conversion to SQLite is done in chunks (safe), but loading still loads full table into memory.
    # If OOM persists, consider implementing query-based loading per page.
    for db_path in sqlite_paths:
        if os.path.exists(db_path):
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                # Load full dataset from SQLite
                # SQLite conversion was done in chunks (safe), but this still loads all rows into memory
                df = pd.read_sql_query("SELECT * FROM donors", conn)
                conn.close()
                if STREAMLIT_AVAILABLE:
                    st.sidebar.success(f"✅ Loaded full dataset from SQLite ({len(df):,} rows)")
                return process_dataframe(df)
            except Exception as e:
                if STREAMLIT_AVAILABLE:
                    st.sidebar.warning(f"Failed to load {db_path}: {e}")
    
    # Priority 2: Try Parquet file (sampled dataset, fallback if SQLite unavailable)
    for path in parquet_paths:
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path, engine='pyarrow')
                return process_dataframe(df)
            except Exception as e:
                if STREAMLIT_AVAILABLE:
                    st.sidebar.warning(f"Failed to load {path}: {e}")
    
    # Priority 3: Try CSV parts (load fewer files for speed)
    csv_dir = next((p for p in csv_dir_candidates if os.path.exists(p)), None)
    if csv_dir and os.path.exists(csv_dir):
        try:
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            if csv_files:
                csv_dir_path = Path(csv_dir).resolve()
                if KAGGLE_DOWNLOAD_DIR in csv_dir_path.parents or csv_dir_path == KAGGLE_DOWNLOAD_DIR:
                    donor_path = None
                    for name in ["donors.csv", "donors_with_network_features.csv", "donors_full.csv"]:
                        candidate = csv_dir_path / name
                        if candidate.exists():
                            donor_path = candidate
                            break
                    if not donor_path and csv_files:
                        donor_path = csv_dir_path / csv_files[0]
                    if donor_path and donor_path.exists():
                        df = pd.read_csv(donor_path)
                        return process_dataframe(df)
                else:
                    dfs = []
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

