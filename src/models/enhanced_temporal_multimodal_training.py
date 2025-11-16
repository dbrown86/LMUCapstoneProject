"""
Enhanced Temporal Multimodal Training Utilities
Functions for creating temporal and text features
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

def create_temporal_features(donors_df: pd.DataFrame, giving_df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from giving history.
    
    Expected columns in giving_df:
    - 'Donor_ID', 'Gift_Date', 'Gift_Amount'
    Returns a DataFrame with temporal features per donor.
    """
    donor_ids = donors_df['ID'] if 'ID' in donors_df.columns else donors_df.index
    
    if giving_df is None or len(giving_df) == 0:
        return pd.DataFrame(0, index=donor_ids, columns=[
            'gifts_last_6mo', 'gifts_last_12mo', 'gifts_last_24mo',
            'total_gifts', 'avg_gift_amount', 'gift_frequency',
            'years_active', 'gifts_per_year'
        ])
    
    g = giving_df.copy()
    if 'Gift_Date' in g.columns and not pd.api.types.is_datetime64_any_dtype(g['Gift_Date']):
        g['Gift_Date'] = pd.to_datetime(g['Gift_Date'], errors='coerce')
    
    if 'Gift_Amount' in g.columns:
        g['Gift_Amount'] = pd.to_numeric(g['Gift_Amount'], errors='coerce').fillna(0)
    else:
        g['Gift_Amount'] = 0
    
    # Time windows
    if 'Gift_Date' in g.columns and g['Gift_Date'].notna().any():
        latest = g['Gift_Date'].max()
        cutoff_6mo = latest - pd.Timedelta(days=180)
        cutoff_12mo = latest - pd.Timedelta(days=365)
        cutoff_24mo = latest - pd.Timedelta(days=730)
        
        grp = g.groupby('Donor_ID')
        gifts_6mo = grp['Gift_Date'].apply(lambda s: (s >= cutoff_6mo).sum() if s.notna().any() else 0)
        gifts_12mo = grp['Gift_Date'].apply(lambda s: (s >= cutoff_12mo).sum() if s.notna().any() else 0)
        gifts_24mo = grp['Gift_Date'].apply(lambda s: (s >= cutoff_24mo).sum() if s.notna().any() else 0)
    else:
        grp = g.groupby('Donor_ID')
        gifts_6mo = pd.Series(0, index=grp.size().index)
        gifts_12mo = pd.Series(0, index=grp.size().index)
        gifts_24mo = pd.Series(0, index=grp.size().index)
    
    total_gifts = grp.size()
    avg_gift = grp['Gift_Amount'].mean()
    
    # Years active
    if 'Gift_Date' in g.columns and g['Gift_Date'].notna().any():
        years_active = grp['Gift_Date'].apply(lambda s: (s.max() - s.min()).days / 365.25 if len(s) > 1 and s.notna().any() else 0)
        gifts_per_year = total_gifts / (years_active + 0.1)  # Avoid division by zero
    else:
        years_active = pd.Series(0, index=grp.size().index)
        gifts_per_year = pd.Series(0, index=grp.size().index)
    
    gift_frequency = total_gifts / (years_active + 0.1)
    
    features = pd.DataFrame({
        'gifts_last_6mo': gifts_6mo,
        'gifts_last_12mo': gifts_12mo,
        'gifts_last_24mo': gifts_24mo,
        'total_gifts': total_gifts,
        'avg_gift_amount': avg_gift,
        'gift_frequency': gift_frequency,
        'years_active': years_active,
        'gifts_per_year': gifts_per_year
    },
    index=grp.size().index,
    ).fillna(0.0)
    
    # Reindex to match all donors
    features = features.reindex(donor_ids, fill_value=0.0)
    
    return features


def create_text_features(donors_df: pd.DataFrame, contact_reports_df: pd.DataFrame = None, text_dim: int = 32) -> pd.DataFrame:
    """Create simple text-derived features per donor.

    Expected columns in contact_reports_df:
    - 'Donor_ID', 'Report_Text' or 'Contact_Text' (free text), 'Contact_Date' (optional)
    Returns a DataFrame with up to `text_dim` columns (first few populated).
    """
    donor_ids = donors_df['ID'] if 'ID' in donors_df.columns else donors_df.index
    dim = max(4, min(int(text_dim) if text_dim else 32, 128))

    if contact_reports_df is None or len(contact_reports_df) == 0:
        return pd.DataFrame(0, index=donor_ids, columns=[f'text_svd_{i}' for i in range(dim)])

    cr = contact_reports_df.copy()
    if 'Contact_Date' in cr.columns and not pd.api.types.is_datetime64_any_dtype(cr['Contact_Date']):
        cr['Contact_Date'] = pd.to_datetime(cr['Contact_Date'], errors='coerce')

    # Basic aggregates - check for either Contact_Text or Report_Text
    text_col = None
    if 'Report_Text' in cr.columns:
        text_col = 'Report_Text'
    elif 'Contact_Text' in cr.columns:
        text_col = 'Contact_Text'
    
    if text_col is not None:
        text_len = cr[text_col].fillna('').astype(str).str.len()
    else:
        text_len = pd.Series(0, index=cr.index)

    cr['_len'] = text_len

    # Aggregate per donor
    grp = cr.groupby('Donor_ID')
    num_reports = grp.size()
    avg_len = grp['_len'].mean()

    # Recent activity (last 90 days)
    recent_count = pd.Series(0, index=grp.size().index)
    if 'Contact_Date' in cr.columns and cr['Contact_Date'].notna().any():
        latest = cr['Contact_Date'].max()
        cutoff = latest - pd.Timedelta(days=90)
        recent_count = grp['Contact_Date'].apply(lambda s: (s >= cutoff).sum() if s.notna().any() else 0)

    # Build feature frame
    features = pd.DataFrame({
        'num_contact_reports': num_reports,
        'avg_text_length': avg_len,
        'recent_contacts_90d': recent_count
    },
    index=grp.size().index,
    ).fillna(0.0)
    
    # Reindex to match all donors
    features = features.reindex(donor_ids, fill_value=0.0)
    
    # If we have text, do simple SVD (but for now just use the basic features)
    # Pad to text_dim with zeros
    for i in range(3, dim):
        features[f'text_svd_{i}'] = 0.0
    
    # Rename columns to match expected format
    feature_cols = ['num_contact_reports', 'avg_text_length', 'recent_contacts_90d'] + [f'text_svd_{i}' for i in range(3, dim)]
    features = features[feature_cols[:dim]]
    
    return features

