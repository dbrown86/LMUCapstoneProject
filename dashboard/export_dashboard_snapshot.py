#!/usr/bin/env python3
"""
Export a JSON snapshot of all dashboard metrics and chart-driving data.
Run locally and on EC2, then diff the two JSON files to verify parity.

Usage (from project root, with venv active):
  python -m dashboard.export_dashboard_snapshot
  python -m dashboard.export_dashboard_snapshot --output external_snapshot.json

Output: dashboard_snapshot_<timestamp>.json (or path given by --output)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import numpy as np


def _safe_float(x):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _safe_int(x):
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _to_jsonable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, float)):
        return obj
    return obj


def export_snapshot(output_path: str = None) -> dict:
    """Build snapshot dict of all metrics and chart data. No Streamlit required."""
    from dashboard.config import settings
    from dashboard.data.loader import _resolve_existing_parquet_path, _load_dataset_for_page_internal
    from dashboard.models.metrics import try_load_saved_metrics, get_model_metrics

    out = {
        "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "environment": {
            "cwd": str(Path.cwd()),
            "project_root": str(settings.get_project_root()),
            "USE_SAVED_METRICS_ONLY": getattr(settings, "USE_SAVED_METRICS_ONLY", None),
        },
        "data_source": {},
        "saved_metrics": None,
        "model_metrics": None,
        "executive_summary": {},
        "sidebar_metrics": {},
        "charts": {},
    }

    # Which parquet is used
    parquet_path = _resolve_existing_parquet_path()
    if parquet_path:
        out["data_source"]["parquet_path"] = str(parquet_path)
        out["data_source"]["parquet_exists"] = True
        out["data_source"]["parquet_size_mb"] = round(parquet_path.stat().st_size / (1024 * 1024), 2)
    else:
        out["data_source"]["parquet_path"] = None
        out["data_source"]["parquet_exists"] = False

    # Saved metrics (training_summary.json)
    saved = try_load_saved_metrics()
    if saved:
        out["saved_metrics"] = {k: _safe_float(v) if isinstance(v, (int, float)) else v for k, v in saved.items()}
    else:
        out["saved_metrics"] = None

    # Load dataframe once (dashboard page columns)
    try:
        df = _load_dataset_for_page_internal("dashboard")
    except Exception as e:
        out["error"] = f"Failed to load dataset: {e}"
        return out

    if df is None or not isinstance(df, pd.DataFrame):
        out["error"] = "Dataset is not a DataFrame"
        return out
    out["data_source"]["rows"] = len(df)
    out["data_source"]["columns_sample"] = list(df.columns)[:30]

    # Model metrics (same as sidebar and Executive Summary)
    metrics = get_model_metrics(df, use_cache=False)
    out["model_metrics"] = {k: _safe_float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics.items()}

    # Sidebar displays: auc, f1, baseline_auc, lift
    out["sidebar_metrics"] = {
        "auc": metrics.get("auc"),
        "f1": metrics.get("f1"),
        "baseline_auc": metrics.get("baseline_auc"),
        "lift": metrics.get("lift"),
    }

    # Executive Summary hero metrics and card
    prob_col = "Will_Give_Again_Probability" if "Will_Give_Again_Probability" in df.columns else "predicted_prob"
    outcome_col = "Gave_Again_In_2025" if "Gave_Again_In_2025" in df.columns else ("Gave_Again_In_2024" if "Gave_Again_In_2024" in df.columns else "actual_gave")

    out["executive_summary"]["auc_display"] = f"{metrics['auc']:.2%}" if metrics.get("auc") is not None else "N/A"
    out["executive_summary"]["f1_display"] = f"{metrics['f1']:.2%}" if metrics.get("f1") is not None else "N/A"
    out["executive_summary"]["baseline_auc_display"] = f"{metrics['baseline_auc']:.2%}" if metrics.get("baseline_auc") is not None else "50.29%"

    if prob_col in df.columns:
        high_confidence_count = (pd.to_numeric(df[prob_col], errors="coerce") >= 0.7).sum()
        out["executive_summary"]["high_confidence_count"] = _safe_int(high_confidence_count)
        high_prob = df[df[prob_col] >= 0.5]
        if "avg_gift" in df.columns and outcome_col in df.columns and len(high_prob) > 0:
            avg_gift_mean = pd.to_numeric(high_prob["avg_gift"], errors="coerce").fillna(0).clip(lower=0).median()
            conv = high_prob[outcome_col].mean()
            estimated_revenue = avg_gift_mean * len(high_prob) * conv
            out["executive_summary"]["estimated_revenue"] = _safe_float(estimated_revenue)
            out["executive_summary"]["revenue_display"] = f"${estimated_revenue:,.0f}" if estimated_revenue < 1e6 else f"${estimated_revenue/1e6:,.0f}M"
        else:
            out["executive_summary"]["estimated_revenue"] = None
            out["executive_summary"]["revenue_display"] = None
    else:
        out["executive_summary"]["high_confidence_count"] = None
        out["executive_summary"]["estimated_revenue"] = None
        out["executive_summary"]["revenue_display"] = None

    # Chart data: segment breakdown (Executive Summary)
    if "segment" in df.columns and prob_col in df.columns:
        seg_df = df[["segment", prob_col]].dropna(subset=["segment"])
        seg_df = seg_df[seg_df["segment"] != "Prospects/New"]
        if not seg_df.empty:
            summary = seg_df.groupby("segment", observed=False).agg(Count=("segment", "size"), Avg_Prob=(prob_col, "mean")).reset_index()
            summary["Percentage"] = (summary["Count"] / summary["Count"].sum() * 100).round(1)
            out["charts"]["segment_breakdown"] = summary.to_dict(orient="records")
            out["charts"]["segment_totals"] = summary["Count"].sum()
        else:
            out["charts"]["segment_breakdown"] = []
            out["charts"]["segment_totals"] = 0
    else:
        out["charts"]["segment_breakdown"] = []
        out["charts"]["segment_totals"] = 0

    # Chart data: tier (Low/Medium/High) counts
    if prob_col in df.columns:
        probs = pd.to_numeric(df[prob_col], errors="coerce").dropna()
        if len(probs):
            tiers = pd.cut(probs, bins=[0.0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)
            tier_counts = tiers.value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
            out["charts"]["tier_counts"] = tier_counts.to_dict()
        else:
            out["charts"]["tier_counts"] = {"Low": 0, "Medium": 0, "High": 0}
    else:
        out["charts"]["tier_counts"] = {"Low": 0, "Medium": 0, "High": 0}

    # Convert any remaining numpy/pandas types for JSON
    out = _to_jsonable(out)

    # training_summary.json path check (re-add after _to_jsonable; it's already in out)
    # (saved_metrics_resolved_path is set below and out is already converted, so set it after)
    out["saved_metrics_path_found"] = saved is not None
    for p in getattr(settings, "SAVED_METRICS_CANDIDATES", []):
        candidate = settings.get_project_root() / p
        if candidate.resolve().exists() and candidate.resolve().is_file():
            out["saved_metrics_resolved_path"] = str(candidate.resolve())
            break
    else:
        out["saved_metrics_resolved_path"] = None

    return out


def main():
    parser = argparse.ArgumentParser(description="Export dashboard metrics/chart snapshot for parity checks.")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path (default: dashboard_snapshot_<timestamp>.json)")
    args = parser.parse_args()

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(__file__).parent / f"dashboard_snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

    try:
        snapshot = export_snapshot()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        print(f"Snapshot written to: {out_path}")
        print(f"  Rows: {snapshot.get('data_source', {}).get('rows', 'N/A')}")
        print(f"  Parquet: {snapshot.get('data_source', {}).get('parquet_path', 'N/A')}")
        print(f"  Saved metrics loaded: {snapshot.get('saved_metrics_path_found', False)}")
        print(f"  AUC: {snapshot.get('model_metrics', {}).get('auc')}")
        print(f"  F1: {snapshot.get('model_metrics', {}).get('f1')}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
