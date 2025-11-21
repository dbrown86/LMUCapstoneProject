"""
Configuration settings for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

from pathlib import Path
import os

PAGE_CONFIG = {
    "page_title": "Fictitious U",
    "page_icon": "🎓",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

def get_project_root():
    """Get the project root directory."""
    root = Path(__file__).resolve().parent.parent.parent
    return root

def get_data_paths():
    """Get all candidate paths for data files."""
    root = get_project_root()
    data_dir_env = os.getenv("LMU_DATA_DIR")
    env_dir = Path(data_dir_env).resolve() if data_dir_env else None
    
    parquet_paths = [
        str(root / "data/processed/parquet_export/donors_with_network_features.parquet"),
        str(root / "data/parquet_export/donors_with_network_features.parquet"),
        str(root / "donors_with_network_features.parquet"),
        str(root / "data/donors.parquet"),
        "data/processed/parquet_export/donors_with_network_features.parquet",
        "data/parquet_export/donors_with_network_features.parquet",
        "donors_with_network_features.parquet",
        "data/donors.parquet",
    ]
    if env_dir:
        parquet_paths.extend([
            str(env_dir / "donors_with_network_features.parquet"),
            str(env_dir / "data/donors.parquet"),
        ])
    
    sqlite_paths = [
        str(root / "data/synthetic_donor_dataset_500k_dense/donor_database.db"),
        str(root / "donor_database.db"),
        "data/synthetic_donor_dataset_500k_dense/donor_database.db",
        "donor_database.db",
    ]
    if env_dir:
        sqlite_paths.extend([
            str(env_dir / "donor_database.db"),
            str(env_dir / "data/synthetic_donor_dataset_500k_dense/donor_database.db"),
        ])
    
    csv_dir_candidates = [
        str(root / "data/synthetic_donor_dataset_500k_dense/parts"),
        "data/synthetic_donor_dataset_500k_dense/parts",
    ]
    if env_dir:
        csv_dir_candidates.append(str(env_dir / "parts"))
    
    giving_paths = [
        root / "data/processed/parquet_export/giving_history.parquet",
        root / "data/parquet_export/giving_history.parquet",
        root / "giving_history.parquet",
        "data/parquet_export/giving_history.parquet"
    ]
    
    return {
        'parquet_paths': parquet_paths,
        'sqlite_paths': sqlite_paths,
        'csv_dir_candidates': csv_dir_candidates,
        'giving_paths': giving_paths,
    }

SAVED_METRICS_CANDIDATES = [
    "models/donor_model_checkpoints/training_summary.json",
    "results/training_summary.json",
    "models/checkpoints/donor_model_checkpoints/training_summary.json"
]

USE_SAVED_METRICS_ONLY = True

COLUMN_MAPPING = {
    'avg_gift_amount': 'avg_gift',
    'Avg_Gift_Amount': 'avg_gift',
    'gift_count': 'gift_count',
    'Will_Give_Again_Probability': 'Will_Give_Again_Probability',
    'predicted_prob': 'predicted_prob'
}

PROBABILITY_COLUMN_VARIANTS = [
    'Will_Give_Again_Probability',
    'predicted_prob',
    'Legacy_Intent_Probability',
    'predicted_probability'
]

OUTCOME_COLUMN_VARIANTS = [
    'Gave_Again_In_2025',
    'Gave_Again_In_2024',
    'actual_gave',
    'gave_again',
    'gave_again_in_2025',
    'gave_again_in_2024'
]

