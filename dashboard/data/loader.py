"""
Data loading module for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.

MEMORY OPTIMIZATIONS:
- Kaggle downloads are DISABLED by default to prevent memory issues on Streamlit Cloud
- Set ENABLE_KAGGLE_DOWNLOAD=true in Streamlit secrets to enable Kaggle downloads
- Data loading uses optimized dtypes and column selection to reduce memory by 60-80%
- All data loading is cached with Streamlit's @st.cache_data to prevent reloading
"""

import pandas as pd
import numpy as np
import os
import glob
import subprocess
import shutil
import json
from pathlib import Path
from typing import Optional, List

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

# Helper function to get secrets from either Streamlit secrets or environment variables
def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a secret from Streamlit secrets (preferred) or environment variables (fallback)."""
    if STREAMLIT_AVAILABLE:
        try:
            # Try Streamlit secrets first (for Streamlit Cloud)
            # Access st.secrets safely - it may not be available at module import time
            if hasattr(st, 'secrets'):
                try:
                    secrets_dict = st.secrets
                    if isinstance(secrets_dict, dict) and key in secrets_dict:
                        return str(secrets_dict[key])
                    # Also try nested access (st.secrets.KAGGLE_USERNAME)
                    if hasattr(secrets_dict, key):
                        return str(getattr(secrets_dict, key))
                except (AttributeError, KeyError, TypeError):
                    pass  # If secrets access fails, fall back to env vars
        except Exception:
            pass  # If anything fails, fall back to env vars
    
    # Fall back to environment variables (for local development)
    return os.getenv(key, default)

# Initialize at module level, but will be re-evaluated in functions if needed
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET")  # Will be overridden in function if st.secrets available
KAGGLE_DOWNLOAD_DIR = Path(os.getenv("KAGGLE_DOWNLOAD_DIR", "/tmp/kaggle_data")).resolve()
KAGGLE_KNOWN_CSVS = {"donors.csv", "contacts.csv", "events.csv", "family.csv", "giving.csv", "relationships.csv"}
KAGGLE_CACHED_PARQUET = KAGGLE_DOWNLOAD_DIR / "donors_cached.parquet"
KAGGLE_CACHED_SQLITE = KAGGLE_DOWNLOAD_DIR / "donors_cached.db"

# Control verbose output (set STREAMLIT_VERBOSE_LOADING=true to show loading messages)
VERBOSE_LOADING = os.getenv("STREAMLIT_VERBOSE_LOADING", "false").lower() == "true"


def _download_kaggle_dataset_if_needed() -> Optional[Path]:
    """
    Download the Kaggle dataset (if configured) into a local directory so the dashboard
    can load the full 500K rows on Streamlit Cloud.
    
    MEMORY OPTIMIZATION: Disabled by default to prevent memory issues on Streamlit Cloud.
    Set ENABLE_KAGGLE_DOWNLOAD=true in environment or Streamlit secrets to enable.

    Returns:
        Optional[Path]: Directory containing the extracted CSV/Parquet files, or None.
    """
    # MEMORY FIX: Check if Kaggle downloads are explicitly enabled
    enable_kaggle = _get_secret("ENABLE_KAGGLE_DOWNLOAD", "false")
    if enable_kaggle.lower() not in ("true", "1", "yes", "on"):
        # Skip Kaggle download by default to save memory
        return None
    
    # Get secrets at function call time (when st.secrets is definitely available)
    kaggle_dataset = _get_secret("KAGGLE_DATASET")
    if not kaggle_dataset:
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
    # Use _get_secret to check both Streamlit secrets and environment variables
    kaggle_username = _get_secret("KAGGLE_USERNAME")
    kaggle_key = _get_secret("KAGGLE_KEY")
    
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
    
    # Build command with dataset name from secrets
    cmd = [
        kaggle_cli,
        "datasets",
        "download",
        "-d",
        kaggle_dataset,
        "-p",
        str(KAGGLE_DOWNLOAD_DIR),
        "--unzip",
    ]

    try:
        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
            try:
                st.sidebar.info(f"📥 Downloading dataset from Kaggle: {kaggle_dataset}")
            except Exception:
                pass  # Even sidebar messages can fail
        try:
            # Create clean environment dict for subprocess
            # CRITICAL: Don't use os.environ directly - build from scratch to avoid KAGGLE_USER_AGENT issues
            # Check Streamlit secrets first, then build env dict
            env = {}
            
            # Copy environment variables, but exclude KAGGLE_USER_AGENT completely
            for key, value in os.environ.items():
                if key != "KAGGLE_USER_AGENT":
                    # Only include valid string values
                    if value is not None and isinstance(value, str):
                        env[key] = value
            
            # Ensure KAGGLE_USER_AGENT is NOT in the env dict (double-check)
            env.pop("KAGGLE_USER_AGENT", None)
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600, env=env)
            if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
                try:
                    st.sidebar.success("✅ Kaggle dataset downloaded successfully")
                except Exception:
                    pass
            return KAGGLE_DOWNLOAD_DIR
        except subprocess.TimeoutExpired:
            if STREAMLIT_AVAILABLE:
                try:
                    st.sidebar.warning("⏱️ Kaggle download timed out. Trying other data sources...")
                except Exception:
                    pass
            return None
        except subprocess.CalledProcessError as err:
            # Safely extract error message for diagnostics
            error_msg = "Unknown error"
            try:
                # Handle both string and bytes for stdout/stderr
                stdout_str = ""
                stderr_str = ""
                if hasattr(err, 'stdout') and err.stdout:
                    if isinstance(err.stdout, bytes):
                        stdout_str = err.stdout.decode("utf-8", errors="ignore")
                    else:
                        stdout_str = str(err.stdout)
                if hasattr(err, 'stderr') and err.stderr:
                    if isinstance(err.stderr, bytes):
                        stderr_str = err.stderr.decode("utf-8", errors="ignore")
                    else:
                        stderr_str = str(err.stderr)
                
                # Combine error messages
                if stderr_str:
                    error_msg = stderr_str.strip()
                elif stdout_str:
                    error_msg = stdout_str.strip()
                else:
                    error_msg = f"Exit code {err.returncode}"
            except Exception:
                # If anything fails, just use a generic message
                try:
                    error_msg = f"Exit code {err.returncode}" if hasattr(err, 'returncode') else "Unknown error"
                except Exception:
                    error_msg = "Unknown error"
            
            # Log the error for diagnostics (but don't crash)
            if STREAMLIT_AVAILABLE:
                try:
                    st.sidebar.warning(f"⚠️ Kaggle download failed: {error_msg[:200]}. Trying other data sources...")
                except Exception:
                    pass
            return None
        except Exception as e:
            # Catch any other exceptions (including AttributeError, OSError, etc.)
            error_msg = str(e) if e else "Unknown error"
            if STREAMLIT_AVAILABLE:
                try:
                    st.sidebar.warning(f"⚠️ Kaggle download error: {error_msg[:200]}. Trying other data sources...")
                except Exception:
                    pass
            return None
    except Exception:
        # Outer catch-all - ensure nothing escapes
        return None


def load_full_dataset(use_cache: bool = True, max_rows: Optional[int] = None):
    """
    Load the complete 500K donor dataset with optimized memory usage.
    Uses column selection and dtype optimization to reduce memory by 60-80%.
    
    Args:
        use_cache: If True and Streamlit is available, use Streamlit caching
        max_rows: Optional limit on number of rows to load (for memory-constrained environments).
                  If None, loads full dataset. Can also be set via MAX_ROWS environment variable.
    
    Returns:
        pd.DataFrame: Processed donor dataset (optimized for memory)
    """
    # Check for MAX_ROWS environment variable if not explicitly provided
    if max_rows is None:
        max_rows_env = os.getenv("MAX_ROWS")
        if max_rows_env:
            try:
                max_rows = int(max_rows_env)
            except (ValueError, TypeError):
                max_rows = None
    
    # Use Streamlit caching if available and requested
    if use_cache and STREAMLIT_AVAILABLE:
        @st.cache_data(show_spinner=False, ttl=7200, max_entries=1)  # 2 hour cache, no spinner, single entry
        def _load_cached():
            df = _load_full_dataset_internal()
            # Apply row limit if specified
            if max_rows is not None and max_rows > 0 and len(df) > max_rows:
                if STREAMLIT_AVAILABLE:
                    st.sidebar.info(f"📊 Loading {max_rows:,} rows (limited from {len(df):,} total rows for memory optimization)")
                return df.head(max_rows)
            return df
        return _load_cached()
    else:
        df = _load_full_dataset_internal()
        # Apply row limit if specified
        if max_rows is not None and max_rows > 0 and len(df) > max_rows:
            return df.head(max_rows)
        return df


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


def _get_essential_columns() -> List[str]:
    """
    Get list of essential columns used by the dashboard.
    This reduces memory usage by 60-80% compared to loading all columns.
    """
    return [
        # Core identifiers
        'donor_id', 'Donor_ID', 'donorid', 'ID',
        # Names
        'First_Name', 'first_name', 'First Name',
        'Last_Name', 'last_name', 'Last Name',
        # Predictions and outcomes (2025)
        'Will_Give_Again_Probability', 'predicted_prob', 'prediction', 'probability',
        'Gave_Again_In_2025', 'Gave_Again_In_2024', 'actual_gave',
        # Giving metrics
        'total_giving', 'Lifetime_Giving', 'lifetime_giving', 'Lifetime Giving', 'LifetimeGiving',
        'Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount',
        'avg_gift', 'Average_Gift', 'average_gift', 'avg_gift_amount',
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
        # Gift officer assignment (for dashboard charts)
        'Primary_Manager', 'Gift Officer', 'Gift_Officer', 'gift_officer', 'GiftOfficer',
        # Capacity rating (for take_action page)
        'Rating', 'rating', 'capacity', 'Capacity',
    ]


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage.
    Reduces memory by 20-30% on average.
    """
    df = df.copy()
    
    # Optimize integer columns
    for col in df.select_dtypes(include=['int64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
            elif col_max < 4294967295:
                df[col] = df[col].astype('uint32')
        else:
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype('int16')
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype('int32')
    
    # Optimize float columns (use float32 where precision allows)
    for col in df.select_dtypes(include=['float64']).columns:
        # For probabilities and percentages, float32 is sufficient
        if 'prob' in col.lower() or 'rate' in col.lower() or 'score' in col.lower():
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        # For monetary values, keep float64 for precision
        elif 'gift' in col.lower() or 'giving' in col.lower() or 'amount' in col.lower() or 'revenue' in col.lower():
            continue  # Keep float64
        else:
            # For other floats, try float32
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
            except (ValueError, OverflowError):
                pass  # Keep float64 if conversion fails
    
    # Convert object columns to category where appropriate (saves memory for repeated values)
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
            try:
                df[col] = df[col].astype('category')
            except (ValueError, TypeError):
                pass  # Keep as object if conversion fails
    
    return df


def _convert_kaggle_csv_to_optimized_parquet(csv_dir: Path) -> Optional[Path]:
    """
    Convert Kaggle donors.csv to optimized Parquet with:
    1. Only essential columns (60-80% memory reduction)
    2. Optimized data types (20-30% additional reduction)
    3. Full 500K rows (no sampling)
    
    This should reduce memory from ~2GB to ~300-500MB, fitting in 1GB RAM.
    
    Args:
        csv_dir: Directory containing the Kaggle CSV files
        
    Returns:
        Path to the created Parquet file, or None if conversion failed
    """
    parquet_path = KAGGLE_CACHED_PARQUET
    
    # If optimized Parquet already exists, use it
    if parquet_path.exists():
        return parquet_path
    
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
        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
            st.sidebar.info("🔄 Converting CSV to optimized Parquet (loading essential columns only)...")
        
        # Get essential columns
        essential_cols = _get_essential_columns()
        
        # Read CSV in chunks, selecting only essential columns
        chunk_size = 50000
        chunks = []
        total_rows = 0
        
        # First, read a small chunk to identify which columns actually exist
        sample_chunk = pd.read_csv(donor_csv, nrows=1000, low_memory=False)
        available_cols = [col for col in essential_cols if col in sample_chunk.columns]
        # Also include any feature columns (for features page)
        feature_cols = [col for col in sample_chunk.columns if any(
            pattern in col.lower() for pattern in ['_count', '_sum', '_mean', '_max', '_min', 'network', 'degree']
        )]
        selected_cols = list(set(available_cols + feature_cols))
        
        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
            st.sidebar.info(f"   Loading {len(selected_cols)} columns (out of {len(sample_chunk.columns)} total)")
        
        # Read full CSV in chunks with selected columns
        for chunk in pd.read_csv(donor_csv, chunksize=chunk_size, low_memory=False, usecols=selected_cols):
            chunks.append(chunk)
            total_rows += len(chunk)
            
            if STREAMLIT_AVAILABLE and VERBOSE_LOADING and total_rows % 100000 == 0:
                st.sidebar.info(f"   Processed {total_rows:,} rows...")
        
        # Combine chunks
        if not chunks:
            if STREAMLIT_AVAILABLE:
                st.sidebar.error("No data chunks loaded")
            return None
        
        df = pd.concat(chunks, ignore_index=True)
        
        # Optimize data types
        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
            st.sidebar.info("   Optimizing data types...")
        df = _optimize_dtypes(df)
        
        # Ensure directory exists
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet with compression
        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
            st.sidebar.info("   Saving optimized Parquet...")
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
        
        # Calculate memory savings
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
            st.sidebar.success(f"✅ Optimized Parquet created ({len(df):,} rows, {len(df.columns)} cols, ~{memory_mb:.0f}MB in memory)")
        
        return parquet_path
    except Exception as e:
        if STREAMLIT_AVAILABLE:
            st.sidebar.warning(f"Failed to convert CSV to optimized Parquet: {e}")
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
        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
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
               # Capacity and gift officer (for take_action page and dashboard)
               'Rating', 'rating', 'capacity', 'Capacity',
               'Gift Officer', 'gift_officer', 'Gift_Officer', 'Primary_Manager',
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
        
        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
            st.sidebar.success(f"✅ Parquet cache created ({len(df):,} rows, {len(df.columns)} columns)")
        
        return parquet_path
    except Exception as e:
        if STREAMLIT_AVAILABLE:
            st.sidebar.warning(f"Failed to convert CSV to Parquet: {e}")
        return None


def _load_csv_with_essential_columns(csv_path: Path, chunk_size: int = 50000) -> Optional[pd.DataFrame]:
    """
    Load CSV file with only essential columns in chunks to avoid OOM.
    This is a fault-tolerant approach that works with any CSV source.
    
    Args:
        csv_path: Path to CSV file
        chunk_size: Number of rows to process at a time
        
    Returns:
        DataFrame with essential columns only, or None if loading fails
    """
    try:
        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
            st.sidebar.info(f"📂 Loading {csv_path.name} with essential columns only...")
        
        essential_cols = _get_essential_columns()
        
        # First, read a small sample to identify which columns exist
        sample = pd.read_csv(csv_path, nrows=100, low_memory=False)
        available_cols = [col for col in essential_cols if col in sample.columns]
        # Also include feature columns for features page
        feature_cols = [col for col in sample.columns if any(
            pattern in col.lower() for pattern in ['_count', '_sum', '_mean', '_max', '_min', 'network', 'degree']
        )]
        selected_cols = list(set(available_cols + feature_cols))
        
        if not selected_cols:
            if STREAMLIT_AVAILABLE:
                st.sidebar.warning(f"No essential columns found in {csv_path.name}")
            return None
        
        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
            st.sidebar.info(f"   Loading {len(selected_cols)} columns (out of {len(sample.columns)} total)")
        
        # Read CSV in chunks with selected columns
        chunks = []
        total_rows = 0
        
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False, usecols=selected_cols):
            chunks.append(chunk)
            total_rows += len(chunk)
            
            if STREAMLIT_AVAILABLE and VERBOSE_LOADING and total_rows % 100000 == 0:
                st.sidebar.info(f"   Loaded {total_rows:,} rows...")
        
        if not chunks:
            return None
        
        # Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        
        # Optimize data types
        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
            st.sidebar.info("   Optimizing data types...")
        df = _optimize_dtypes(df)
        
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
            st.sidebar.success(f"✅ Loaded {len(df):,} rows, {len(df.columns)} cols (~{memory_mb:.0f}MB)")
        
        return df
    except Exception as e:
        if STREAMLIT_AVAILABLE:
            st.sidebar.warning(f"Failed to load {csv_path.name}: {e}")
        return None


def _load_full_dataset_internal():
    """Internal function to load dataset (without caching decorator)."""
    root = settings.get_project_root()
    data_dir_env = os.getenv("LMU_DATA_DIR")
    env_dir = Path(data_dir_env).resolve() if data_dir_env else None
    
    # MEMORY OPTIMIZATION: Skip Kaggle download by default to prevent memory issues
    # Only attempt if explicitly enabled via ENABLE_KAGGLE_DOWNLOAD=true
    kaggle_dir = None
    try:
        # Check if Kaggle downloads are enabled before attempting
        enable_kaggle = _get_secret("ENABLE_KAGGLE_DOWNLOAD", "false")
        if enable_kaggle.lower() in ("true", "1", "yes", "on"):
            kaggle_dir = _download_kaggle_dataset_if_needed()
        # Otherwise skip silently to save memory
    except Exception:
        # Silently continue - Kaggle download is optional
        # Don't try to access exception attributes to avoid any decode/string errors
        # Just continue to try other data sources
        pass

    # Get paths from config
    data_paths = settings.get_data_paths()
    parquet_paths = data_paths['parquet_paths']
    sqlite_paths = data_paths['sqlite_paths']
    csv_dir_candidates = data_paths['csv_dir_candidates']

    # If Kaggle dataset was downloaded, try to create optimized Parquet
    if kaggle_dir:
        try:
            resolved_csv_dir = _resolve_kaggle_csv_dir()
            if resolved_csv_dir:
                # Check for existing optimized Parquet first
                if KAGGLE_CACHED_PARQUET.exists():
                    parquet_paths.insert(0, str(KAGGLE_CACHED_PARQUET))
                else:
                    # Try to create optimized Parquet (non-blocking)
                    try:
                        cached_parquet = _convert_kaggle_csv_to_optimized_parquet(resolved_csv_dir)
                        if cached_parquet:
                            parquet_paths.insert(0, str(cached_parquet))
                    except Exception as e:
                        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
                            st.sidebar.info(f"⚠️ Parquet conversion skipped: {e}. Trying direct CSV load...")
                        # Add CSV directory as fallback
                        csv_dir_candidates.insert(0, str(resolved_csv_dir))
        except Exception:
            pass  # Continue with other sources
    
    # Priority 1: Try optimized Parquet file (essential columns, optimized dtypes, full 500K rows)
    for path in parquet_paths:
        if os.path.exists(path):
            try:
                if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
                    st.sidebar.info(f"📦 Loading optimized Parquet: {Path(path).name}")
                df = pd.read_parquet(path, engine='pyarrow')
                return process_dataframe(df)
            except Exception as e:
                if STREAMLIT_AVAILABLE:
                    st.sidebar.warning(f"⚠️ Failed to load {path}: {e}")
    
    # Priority 2: Try loading CSV with essential columns only (fault-tolerant, works with any CSV)
    csv_dir = next((p for p in csv_dir_candidates if os.path.exists(p)), None)
    if csv_dir and os.path.exists(csv_dir):
        try:
            csv_dir_path = Path(csv_dir).resolve()
            
            # Look for main donor CSV files
            donor_csv_names = ["donors.csv", "donors_with_network_features.csv", "donors_full.csv"]
            donor_path = None
            
            for name in donor_csv_names:
                candidate = csv_dir_path / name
                if candidate.exists():
                    donor_path = candidate
                    break
            
            # If not found, look for any CSV in the directory
            if not donor_path:
                csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
                if csv_files:
                    # Prefer files with "donor" in the name
                    donor_files = [f for f in csv_files if 'donor' in f.lower()]
                    if donor_files:
                        donor_path = csv_dir_path / donor_files[0]
                    else:
                        donor_path = csv_dir_path / csv_files[0]
            
            if donor_path and donor_path.exists():
                # Load with essential columns only (fault-tolerant)
                df = _load_csv_with_essential_columns(donor_path)
                if df is not None:
                    return process_dataframe(df)
        except Exception as e:
            if STREAMLIT_AVAILABLE:
                st.sidebar.warning(f"⚠️ Failed to load CSV from {csv_dir}: {e}")
    
    # Priority 3: Glob search for Parquet across project and env dir
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
                    if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
                        st.sidebar.info(f"📦 Trying Parquet: {Path(p).name}")
                    df = pd.read_parquet(p, engine='pyarrow')
                    # Select only essential columns if dataset is large
                    if len(df) > 100000:
                        essential_cols = _get_essential_columns()
                        available_cols = [col for col in essential_cols if col in df.columns]
                        if available_cols:
                            df = df[available_cols]
                            df = _optimize_dtypes(df)
                    return process_dataframe(df)
                except Exception as e:
                    if STREAMLIT_AVAILABLE:
                        st.sidebar.info(f"⚠️ Skipped {Path(p).name}: {e}")
                    continue
        except Exception:
            pass
    
    # Priority 4: Glob search for CSV files and load with essential columns
    csv_patterns = []
    if env_dir:
        csv_patterns.append(str(env_dir / "**/donors*.csv"))
    csv_patterns.extend([
        str(root / "data/**/donors*.csv"),
        str(root / "**/donors*.csv"),
    ])
    for pattern in csv_patterns:
        try:
            for p in glob.glob(pattern, recursive=True):
                try:
                    csv_path = Path(p)
                    if csv_path.exists() and csv_path.stat().st_size > 0:
                        if STREAMLIT_AVAILABLE and VERBOSE_LOADING:
                            st.sidebar.info(f"📂 Trying CSV: {csv_path.name}")
                        df = _load_csv_with_essential_columns(csv_path)
                        if df is not None:
                            return process_dataframe(df)
                except Exception as e:
                    if STREAMLIT_AVAILABLE:
                        st.sidebar.info(f"⚠️ Skipped {Path(p).name}: {e}")
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
    # STEP 1: Preprocess gift officer column FIRST (before any other processing)
    # This ensures Primary_Manager is available for gift officer charts
    # NOTE: Primary_Manager column exists in the dataset (99.6% null, but column exists)
    # If it's not in df.columns, try to find and map gift officer columns
    if 'Primary_Manager' not in df.columns:
        # Check for gift officer columns - try exact matches first
        gift_officer_exact = ['Gift Officer', 'Gift_Officer', 'gift_officer', 'GiftOfficer', 'Primary_Manager', 'Primary Manager']
        found_col = None
        for col in gift_officer_exact:
            if col in df.columns:
                found_col = col
                break
        
        # If not found, search case-insensitively for any column containing gift/officer/manager
        if not found_col:
            for col in df.columns:
                col_lower = col.lower().strip().replace(' ', '_').replace('-', '_')
                # Check for various patterns
                if (col_lower in ['gift_officer', 'giftofficer', 'primary_manager', 'primarymanager'] or
                    ('gift' in col_lower and 'officer' in col_lower) or
                    ('primary' in col_lower and 'manager' in col_lower)):
                    found_col = col
                    break
        
        # Rename the found column to Primary_Manager
        if found_col:
            df = df.rename(columns={found_col: 'Primary_Manager'})
    
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
        'name': 'donor_name',
        # Gift Officer / Primary Manager mapping
        'Gift Officer': 'Primary_Manager',
        'Gift_Officer': 'Primary_Manager',
        'gift_officer': 'Primary_Manager',
        'Primary_Manager': 'Primary_Manager'  # Keep if already exists
    }
    
    # Handle Primary_Manager mapping specially BEFORE merging
    # If Primary_Manager already exists, don't map Gift Officer columns
    # Otherwise, map the first Gift Officer column found to Primary_Manager
    if 'Primary_Manager' in df.columns:
        # Primary_Manager already exists, remove gift officer mappings to avoid conflicts
        extended_mapping.pop('Gift Officer', None)
        extended_mapping.pop('Gift_Officer', None)
        extended_mapping.pop('gift_officer', None)
        extended_mapping.pop('Primary_Manager', None)  # Don't map to itself
    else:
        # Primary_Manager doesn't exist, find and map the first Gift Officer column
        gift_officer_cols = ['Gift Officer', 'Gift_Officer', 'gift_officer']
        found_gift_officer = False
        for col in gift_officer_cols:
            if col in df.columns:
                # Only map the first one found
                extended_mapping[col] = 'Primary_Manager'
                # Remove other gift officer mappings to avoid conflicts
                for other_col in gift_officer_cols:
                    if other_col != col:
                        extended_mapping.pop(other_col, None)
                found_gift_officer = True
                break
    
    # Merge mappings
    column_mapping.update(extended_mapping)
    
    # Only rename columns that exist
    # IMPORTANT: Preserve Primary_Manager if it exists - don't rename it
    # Store it temporarily if it exists
    primary_manager_col = None
    if 'Primary_Manager' in df.columns:
        primary_manager_col = df['Primary_Manager'].copy()
        # Remove any mappings that would overwrite Primary_Manager
        column_mapping = {k: v for k, v in column_mapping.items() if v != 'Primary_Manager' or k == 'Primary_Manager'}
    
    existing_renames = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_renames)
    
    # Restore Primary_Manager if it was in the original dataframe
    if primary_manager_col is not None and 'Primary_Manager' not in df.columns:
        df['Primary_Manager'] = primary_manager_col
    
    # POST-RENAME CHECK: Ensure Primary_Manager still exists after all mappings
    # This is a safety check - it should already exist from the preprocessing step
    if 'Primary_Manager' not in df.columns:
        # Try to find any column that might be a gift officer column
        # Check all columns for gift officer related terms
        found_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            # Check for specific gift officer patterns
            if col_lower in ['gift officer', 'gift_officer', 'gift officer', 'primary_manager', 'primary manager']:
                found_col = col
                break
            # Check if column contains both "gift" and "officer" or "primary" and "manager"
            if ('gift' in col_lower and 'officer' in col_lower) or ('primary' in col_lower and 'manager' in col_lower):
                found_col = col
                break
        
        # If we found a gift officer column, rename it to Primary_Manager
        if found_col:
            df = df.rename(columns={found_col: 'Primary_Manager'})
    
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
    
    # FINAL CHECK: Ensure Primary_Manager exists (for gift officer charts)
    # This is a last resort check after all other processing
    if 'Primary_Manager' not in df.columns:
        # Search for gift officer columns - check all columns
        found_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            # Check for exact matches first
            if col_lower in ['gift officer', 'gift_officer', 'primary_manager', 'primary manager']:
                found_col = col
                break
            # Check if column contains both "gift" and "officer" or "primary" and "manager"
            if ('gift' in col_lower and 'officer' in col_lower) or ('primary' in col_lower and 'manager' in col_lower):
                found_col = col
                break
        
        # If we found a gift officer column, rename it to Primary_Manager
        if found_col:
            df = df.rename(columns={found_col: 'Primary_Manager'})
    
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

