"""
Precompute dashboard-ready parquet artifacts.

Run locally to avoid loading full raw tables in Streamlit:
    python -m dashboard.data.precompute_dashboard_summaries
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from dashboard.config import settings
from dashboard.data.loader import _optimize_dtypes


SLIM_COLUMNS = [
    "donor_id", "Donor_ID", "ID",
    "First_Name", "first_name", "First Name",
    "Last_Name", "last_name", "Last Name",
    "Will_Give_Again_Probability", "predicted_prob",
    "Gave_Again_In_2025", "Gave_Again_In_2024", "actual_gave",
    "total_giving", "Lifetime_Giving", "lifetime_giving", "Lifetime Giving",
    "avg_gift", "Average_Gift", "average_gift",
    "Last_Gift", "last_gift", "LastGift", "last_gift_amount",
    "gift_count", "Num_Gifts", "num_gifts",
    "segment", "region", "donor_type", "Primary_Constituent_Type",
    "days_since_last", "Days_Since_Last_Gift", "days_since_last_gift",
    "Last_Gift_Date", "last_gift_date",
    "Primary_Manager",
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
        "Days_Since_Last_Gift": "days_since_last",
        "days_since_last_gift": "days_since_last",
        "Primary_Constituent_Type": "donor_type",
    }
    existing = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=existing)


def main() -> None:
    source = _resolve_source_parquet()
    if source is None:
        raise FileNotFoundError("No source donor parquet file found.")

    out_dir = settings.get_data_paths()["summary_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(source, engine="pyarrow")
    keep = [c for c in SLIM_COLUMNS if c in df.columns]
    slim = df[keep].copy()
    slim = _normalize_columns(slim)
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

