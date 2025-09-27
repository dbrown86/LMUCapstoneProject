# Configuration file for data generation module
import os
from datetime import date

# Dataset Configuration
TOTAL_DONORS = 50000
OUTPUT_DIR = "synthetic_donor_dataset"
START_DATE = date(1940, 1, 1)
END_DATE = date(2025, 12, 31)

# Random seed for reproducibility
RANDOM_SEED = 42

# Family Configuration
FAMILY_PERCENTAGE = 0.30  # 30% of donors will be in families

# Giving Configuration
NON_DONOR_PERCENTAGE = 0.40  # 40% chance of being a non-donor
MIN_GIFT_AMOUNT = 25  # Minimum gift amount
MAX_GIFT_AMOUNT = 100000000  # Maximum gift amount ($100M)

# Rating Distribution (wealth capacity bands)
RATING_DISTRIBUTION = {
    'A': 0.001, 'B': 0.002, 'C': 0.005, 'D': 0.008,
    'E': 0.012, 'F': 0.025, 'G': 0.040, 'H': 0.060,
    'I': 0.087, 'J': 0.110, 'K': 0.130, 'L': 0.150,
    'M': 0.140, 'N': 0.120, 'O': 0.080, 'P': 0.030
}

# Constituent Type Distribution
CONSTITUENT_TYPE_DISTRIBUTION = {
    'Alum': 0.55,
    'Parent': 0.20,
    'Friend': 0.15,
    'Foundation': 0.04,
    'Corporation': 0.03,
    'Trust': 0.01,
    'Trustee': 0.015,
    'Regent': 0.005
}

# File Paths
def get_file_paths():
    """Get standardized file paths"""
    return {
        'donors': os.path.join(OUTPUT_DIR, 'donors.csv'),
        'relationships': os.path.join(OUTPUT_DIR, 'relationships.csv'),
        'contact_reports': os.path.join(OUTPUT_DIR, 'contact_reports.csv'),
        'giving_history': os.path.join(OUTPUT_DIR, 'giving_history.csv'),
        'enhanced_fields': os.path.join(OUTPUT_DIR, 'enhanced_fields.csv'),
        'dataset_analysis': os.path.join(OUTPUT_DIR, 'dataset_analysis.png'),
        'gnn_model': 'best_donor_gnn_model.pt'
    }
