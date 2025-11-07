#!/usr/bin/env python3
"""
Standalone script to help troubleshoot the dashboard page visuals.
It loads the dataset, builds the key Plotly figures (segment, region, tiers),
prints figure metadata, and optionally saves HTML snapshots.
"""

import json
from pathlib import Path

import pandas as pd

from dashboard.data.loader import load_full_dataset

try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception as e:
    raise SystemExit(f"Plotly import failed: {e}")


def inspect_fig(fig: go.Figure, name: str) -> None:
    print(f"--- {name} ---")
    print("type:", type(fig))
    traces = len(fig.data)
    print("traces:", traces)
    for i, trace in enumerate(fig.data, start=1):
        if hasattr(trace, "x") and trace.x is not None:
            x_len = len(trace.x)
        else:
            x_len = "n/a"
        if hasattr(trace, "y") and trace.y is not None:
            y_len = len(trace.y)
        else:
            y_len = "n/a"
        if hasattr(trace, "values") and trace.values is not None:
            val_len = len(trace.values)
        else:
            val_len = "n/a"
        print(f" trace {i}: type={trace.type}, len(x)={x_len}, len(y)={y_len}, len(values)={val_len}")
    layout_json = fig.layout.to_plotly_json()
    print("layout keys:", list(layout_json.keys()))


def build_segment_chart(df: pd.DataFrame) -> go.Figure:
    df = df.loc[:, ~df.columns.duplicated()]
    if {"segment", "predicted_prob"}.issubset(df.columns):
        seg_df = df[["segment", "predicted_prob"]].dropna(subset=["segment"])
        counts = seg_df.groupby("segment", observed=False).size().reset_index(name="Count")
        fig = px.bar(counts.sort_values("segment"),
                     x="segment", y="Count",
                     color="segment",
                     color_discrete_sequence=['#4caf50', '#8bc34a', '#ffc107', '#ff5722', '#9e9e9e'])
        fig.update_traces(texttemplate="%{y:,}", textposition="outside")
        return fig
    return go.Figure()


def build_region_chart(df: pd.DataFrame) -> go.Figure:
    df = df.loc[:, ~df.columns.duplicated()]
    if "region" in df.columns:
        reg_series = df["region"].dropna()
        if not reg_series.empty:
            region_counts = reg_series.value_counts().reset_index()
            region_counts.columns = ["Region", "Count"]
            fig = px.pie(region_counts, names="Region", values="Count", hole=0.4,
                         color_discrete_sequence=['#2196f3', '#4caf50', '#ff9800', '#9c27b0', '#e91e63'])
            fig.update_traces(textposition="inside", textinfo="percent+label")
            return fig
    return go.Figure()


def build_tiers_chart(df: pd.DataFrame) -> go.Figure:
    df = df.loc[:, ~df.columns.duplicated()]
    if "predicted_prob" in df.columns:
        probs = pd.to_numeric(df["predicted_prob"], errors="coerce").dropna()
        total = len(probs)
        if total:
            bins = [0.0, 0.4, 0.7, 1.0]
            labels = ["Low", "Medium", "High"]
            tiers = pd.cut(probs, bins=bins, labels=labels, include_lowest=True)
            counts = tiers.value_counts().reindex(labels, fill_value=0).reset_index()
            counts.columns = ["Tier", "Count"]
            counts["Percent"] = counts["Count"] / total * 100.0
            fig = px.bar(counts, x="Percent", y=["Tier"], orientation="h",
                         color="Tier", barmode="stack",
                         color_discrete_map={'Low': '#f44336', 'Medium': '#ffc107', 'High': '#4caf50'})
            return fig
    return go.Figure()


def main() -> None:
    print("Loading dataset...")
    df = load_full_dataset(use_cache=False)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print()

    segment_fig = build_segment_chart(df)
    inspect_fig(segment_fig, "Segment Chart")

    region_fig = build_region_chart(df)
    inspect_fig(region_fig, "Region Chart")

    tiers_fig = build_tiers_chart(df)
    inspect_fig(tiers_fig, "Tiers Chart")

    # Optionally write HTML debug files
    out_dir = Path("debug_dashboard_figures")
    out_dir.mkdir(exist_ok=True)
    for fig, name in [(segment_fig, "segment"), (region_fig, "region"), (tiers_fig, "tiers")]:
        out_path = out_dir / f"{name}.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

