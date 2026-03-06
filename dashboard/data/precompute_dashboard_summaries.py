"""
Precompute dashboard-ready parquet artifacts.

Run locally to avoid loading full raw tables in Streamlit:
    python -m dashboard.data.precompute_dashboard_summaries

IMPORTANT: The source parquet may contain z-score standardized columns (for ML).
This script must use RAW columns for dashboard metrics:
- Lifetime_Giving (not total_lifetime_giving) - actual dollars
- Total_Yr_Giving_Count (not gift_count/total_gifts) - actual counts
- Last_Gift_Date - actual dates to compute days_since_last
- Geographic_Region - actual region names
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dashboard.config import settings
from dashboard.data.loader import _optimize_dtypes


# Columns to extract from source - prioritize RAW columns over standardized ML features
SLIM_COLUMNS = [
    # Identifiers
    "donor_id", "Donor_ID", "ID",
    # Names
    "First_Name", "first_name", "First Name",
    "Last_Name", "last_name", "Last Name",
    # Predictions and outcomes
    "Will_Give_Again_Probability", "predicted_prob",
    "Gave_Again_In_2025", "Gave_Again_In_2024", "actual_gave",
    # Giving - RAW columns (actual dollars)
    "Lifetime_Giving", "total_giving", "lifetime_giving", "Lifetime Giving",
    "Last_Gift", "last_gift", "LastGift", "last_gift_amount",
    # Gift counts - RAW columns (actual counts)
    "Total_Yr_Giving_Count",  # RAW: actual count 0-50
    "Consecutive_Yr_Giving_Count",  # RAW: actual consecutive years
    "gift_count", "Num_Gifts", "num_gifts",  # May be standardized, fallback
    # Segmentation - RAW columns
    "segment",
    "Geographic_Region", "region",  # RAW region names
    "Primary_Constituent_Type", "donor_type",  # RAW donor types
    # Dates - RAW for computing days_since_last
    "Last_Gift_Date", "last_gift_date",
    # Assignment
    "Primary_Manager",
    # Scores (may be standardized but still useful)
    "rfm_score", "recency_score", "frequency_score", "monetary_score",
    "years_active", "consecutive_years",
]


def _resolve_source_parquet() -> Optional[Path]:
    root = settings.get_project_root()
    for raw in settings.get_data_paths()["parquet_paths"]:
        for candidate in (root / raw, Path(raw), Path.cwd() / raw):
            try:
                resolved = candidate.resolve()
                if resolved.exists() and resolved.is_file():
                    # Skip previously generated slim file if present.
                    if resolved.name == "donors_dashboard_slim.parquet":
                        continue
                    return resolved
            except Exception:
                continue
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to standard dashboard names.
    IMPORTANT: Prioritize RAW columns over standardized ML features.
    """
    # First, handle Total_Yr_Giving_Count -> gift_count specially
    # If both exist, drop the standardized one and use the raw one
    if "Total_Yr_Giving_Count" in df.columns and "gift_count" in df.columns:
        # Check if gift_count is standardized (mean near 0)
        gc = pd.to_numeric(df["gift_count"], errors="coerce")
        if gc.mean() < 1 and gc.max() < 10:
            # Standardized - drop it and use raw
            df = df.drop(columns=["gift_count"])
    
    mapping = {
        "Donor_ID": "donor_id",
        "ID": "donor_id",
        "Lifetime_Giving": "total_giving",
        "lifetime_giving": "total_giving",
        "Lifetime Giving": "total_giving",
        "Average_Gift": "avg_gift",
        "average_gift": "avg_gift",
        "Num_Gifts": "gift_count",
        "num_gifts": "gift_count",
        "Total_Yr_Giving_Count": "gift_count",  # RAW count
        "Days_Since_Last_Gift": "days_since_last",
        "days_since_last_gift": "days_since_last",
        "Primary_Constituent_Type": "donor_type",
        "Geographic_Region": "region",  # RAW region names
        "Consecutive_Yr_Giving_Count": "consecutive_years",  # RAW consecutive years
    }
    existing = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=existing)


def _compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived columns from RAW data.
    This ensures dashboard gets actual values, not standardized ML features.
    """
    # 1. Compute days_since_last from Last_Gift_Date (actual dates)
    if "Last_Gift_Date" in df.columns:
        date_col = pd.to_datetime(df["Last_Gift_Date"], errors="coerce")
        today = pd.Timestamp.now()
        df["days_since_last"] = (today - date_col).dt.days.clip(lower=0)
        # Fill NaN with a large value (no gift recorded)
        df["days_since_last"] = df["days_since_last"].fillna(9999)
    elif "days_since_last" not in df.columns:
        df["days_since_last"] = 365  # Default fallback

    # 2. Ensure gift_count is a proper Series with actual values
    if "gift_count" in df.columns:
        # Handle potential DataFrame (duplicate columns)
        gc_col = df["gift_count"]
        if isinstance(gc_col, pd.DataFrame):
            gc_col = gc_col.iloc[:, 0]
        df["gift_count"] = pd.to_numeric(gc_col, errors="coerce").fillna(0)
    else:
        df["gift_count"] = 0

    # 3. Compute avg_gift from total_giving / gift_count
    if "avg_gift" not in df.columns:
        if "total_giving" in df.columns and "gift_count" in df.columns:
            total = pd.to_numeric(df["total_giving"], errors="coerce").fillna(0)
            count = pd.to_numeric(df["gift_count"], errors="coerce").fillna(0)
            # Avoid division by zero
            df["avg_gift"] = np.where(count > 0, total / count, 0)
        else:
            df["avg_gift"] = 0

    # 4. Create segment from days_since_last and gift_count
    days = pd.to_numeric(df["days_since_last"], errors="coerce").fillna(9999)
    gifts = pd.to_numeric(df["gift_count"], errors="coerce").fillna(0)

    all_segments = [
        "Recent (0-6mo)",
        "Recent (6-12mo)",
        "Lapsed (1-2yr)",
        "Very Lapsed (2yr+)",
        "Prospects/New"
    ]

    segments = np.full(len(df), "Prospects/New", dtype=object)
    valid = (gifts > 0) & days.notna()
    within_bounds = valid & (days <= 3650)

    recent_0_6_mask = within_bounds & (days <= 180)
    recent_6_12_mask = within_bounds & (days > 180) & (days <= 365)
    lapsed_mask = within_bounds & (days > 365) & (days <= 730)
    very_lapsed_mask = within_bounds & (days > 730)

    segments[recent_0_6_mask] = "Recent (0-6mo)"
    segments[recent_6_12_mask] = "Recent (6-12mo)"
    segments[lapsed_mask] = "Lapsed (1-2yr)"
    segments[very_lapsed_mask] = "Very Lapsed (2yr+)"

    df["segment"] = pd.Categorical(segments, categories=all_segments)

    # 5. Ensure region exists
    if "region" not in df.columns:
        if "Geographic_Region" in df.columns:
            df["region"] = df["Geographic_Region"]
        else:
            df["region"] = "Unknown"

    return df


def main() -> None:
    source = _resolve_source_parquet()
    if source is None:
        raise FileNotFoundError("No source donor parquet file found.")

    out_dir = settings.get_data_paths()["summary_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading source parquet: {source}")
    df = pd.read_parquet(source, engine="pyarrow")
    print(f"Source shape: {df.shape}")

    # Select columns that exist in source
    keep = [c for c in SLIM_COLUMNS if c in df.columns]
    print(f"Keeping {len(keep)} columns from source")
    slim = df[keep].copy()

    # Normalize column names
    slim = _normalize_columns(slim)

    # CRITICAL: Compute derived columns from RAW data BEFORE optimization
    # This ensures days_since_last, segment, avg_gift use actual values
    slim = _compute_derived_columns(slim)

    # Verify segment distribution looks correct
    if "segment" in slim.columns:
        print("Segment distribution:")
        print(slim["segment"].value_counts())

    # Remove any duplicate columns before optimization
    slim = slim.loc[:, ~slim.columns.duplicated()]
    print(f"Columns after deduplication: {list(slim.columns)}")

    slim = _optimize_dtypes(slim)
    slim_path = out_dir / "donors_dashboard_slim.parquet"
    slim.to_parquet(slim_path, engine="pyarrow", compression="snappy", index=False)

    prob_col = "Will_Give_Again_Probability" if "Will_Give_Again_Probability" in slim.columns else "predicted_prob"
    outcome_col = "Gave_Again_In_2025" if "Gave_Again_In_2025" in slim.columns else ("Gave_Again_In_2024" if "Gave_Again_In_2024" in slim.columns else "actual_gave")
    if prob_col not in slim.columns:
        slim["predicted_prob"] = 0.0
        prob_col = "predicted_prob"

    overview = {
        "rows": len(slim),
        "high_confidence_count": int((pd.to_numeric(slim[prob_col], errors="coerce").fillna(0) >= 0.7).sum()),
        "avg_prob": float(pd.to_numeric(slim[prob_col], errors="coerce").mean()),
        "median_total_giving": float(pd.to_numeric(slim["total_giving"], errors="coerce").median()) if "total_giving" in slim.columns else 0.0,
    }
    if outcome_col in slim.columns:
        overview["actual_response_rate"] = float(pd.to_numeric(slim[outcome_col], errors="coerce").mean())
    pd.DataFrame([overview]).to_parquet(out_dir / "overview_summary.parquet", engine="pyarrow", compression="snappy", index=False)

    if "segment" in slim.columns:
        segment_summary = (
            slim.groupby("segment", observed=False)
            .agg(
                donor_count=("segment", "size"),
                avg_prob=(prob_col, "mean"),
                total_giving=("total_giving", "sum") if "total_giving" in slim.columns else (prob_col, "size"),
            )
            .reset_index()
        )
        segment_summary.to_parquet(out_dir / "segment_summary.parquet", engine="pyarrow", compression="snappy", index=False)

    take_action = {
        "quick_wins": int(((pd.to_numeric(slim[prob_col], errors="coerce") >= 0.7) & (slim.get("segment") == "Recent (0-6mo)")).sum()) if "segment" in slim.columns else 0,
        "cultivation": int(((pd.to_numeric(slim[prob_col], errors="coerce") >= 0.4) & (pd.to_numeric(slim[prob_col], errors="coerce") < 0.7)).sum()),
        "re_engagement": int(((pd.to_numeric(slim[prob_col], errors="coerce") >= 0.6) & (slim.get("segment").isin(["Lapsed (1-2yr)", "Very Lapsed (2yr+)"]))).sum()) if "segment" in slim.columns else 0,
    }
    pd.DataFrame([take_action]).to_parquet(out_dir / "take_action_summary.parquet", engine="pyarrow", compression="snappy", index=False)

    print(f"Created slim parquet: {slim_path}")
    print(f"Created summaries in: {out_dir}")


if __name__ == "__main__":
    main()

