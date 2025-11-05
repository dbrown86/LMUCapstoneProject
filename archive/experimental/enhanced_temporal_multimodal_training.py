"""
Lightweight temporal and text feature builders for donor analytics.
Designed to be dependency-light and fast for 500K-scale datasets.
"""

import pandas as pd
import numpy as np


def create_temporal_features(donors_df: pd.DataFrame, giving_df: pd.DataFrame) -> pd.DataFrame:
    """Create essential temporal features per donor.

    Expected columns:
    - donors_df: must contain 'ID'
    - giving_df: must contain 'Donor_ID', 'Gift_Date', 'Gift_Amount'
    """
    donor_ids = donors_df['ID'] if 'ID' in donors_df.columns else donors_df.index

    if giving_df is None or len(giving_df) == 0:
        return pd.DataFrame(
            {
                'days_since_last_gift': np.zeros(len(donor_ids), dtype=float),
                'gifts_last_6mo': np.zeros(len(donor_ids), dtype=float),
                'gifts_last_12mo': np.zeros(len(donor_ids), dtype=float),
                'gifts_last_24mo': np.zeros(len(donor_ids), dtype=float),
                'total_gifts': np.zeros(len(donor_ids), dtype=float),
                'total_amount': np.zeros(len(donor_ids), dtype=float),
                'avg_gift_amount': np.zeros(len(donor_ids), dtype=float),
            },
            index=donor_ids,
        )

    g = giving_df.copy()
    if 'Gift_Date' in g.columns and not pd.api.types.is_datetime64_any_dtype(g['Gift_Date']):
        g['Gift_Date'] = pd.to_datetime(g['Gift_Date'], errors='coerce')
    g = g.dropna(subset=['Gift_Date'])

    latest_date = g['Gift_Date'].max()
    six_mo_ago = latest_date - pd.Timedelta(days=180)
    twelve_mo_ago = latest_date - pd.Timedelta(days=365)
    twentyfour_mo_ago = latest_date - pd.Timedelta(days=730)

    group = g.groupby('Donor_ID')

    last_gift = group['Gift_Date'].max()
    days_since_last = (latest_date - last_gift).dt.days.reindex(donor_ids).astype('float')

    gifts_6 = group['Gift_Date'].apply(lambda s: (s >= six_mo_ago).sum())
    gifts_12 = group['Gift_Date'].apply(lambda s: (s >= twelve_mo_ago).sum())
    gifts_24 = group['Gift_Date'].apply(lambda s: (s >= twentyfour_mo_ago).sum())

    total_gifts = group['Gift_Amount'].count()
    total_amount = group['Gift_Amount'].sum()
    avg_amount = group['Gift_Amount'].mean()

    features = pd.DataFrame(
        {
            'days_since_last_gift': days_since_last,
            'gifts_last_6mo': gifts_6.reindex(donor_ids).astype('float'),
            'gifts_last_12mo': gifts_12.reindex(donor_ids).astype('float'),
            'gifts_last_24mo': gifts_24.reindex(donor_ids).astype('float'),
            'total_gifts': total_gifts.reindex(donor_ids).astype('float'),
            'total_amount': total_amount.reindex(donor_ids).astype('float'),
            'avg_gift_amount': avg_amount.reindex(donor_ids).astype('float'),
        },
        index=donor_ids,
    ).fillna(0.0)

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
        recent_count = grp['Contact_Date'].apply(lambda s: (s >= cutoff).sum())

    # Build feature frame
    features = pd.DataFrame(
        {
            'num_contact_reports': num_reports.reindex(donor_ids).astype('float'),
            'avg_contact_length': avg_len.reindex(donor_ids).astype('float'),
            'recent_reports_90d': recent_count.reindex(donor_ids).astype('float'),
            'text_activity_score': (num_reports.reindex(donor_ids).fillna(0).astype('float') * 0.5
                                    + recent_count.reindex(donor_ids).fillna(0).astype('float') * 1.0),
        },
        index=donor_ids,
    ).fillna(0.0)

    # Pad to requested dim with zeros
    if features.shape[1] < dim:
        for i in range(features.shape[1], dim):
            features[f'text_svd_{i}'] = 0.0

    # Rename first columns to consistent names expected downstream
    # Keep existing names as informative; downstream can accept any column names
    return features


