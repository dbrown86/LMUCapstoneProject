#!/usr/bin/env python3
"""
Compare two dashboard snapshots (e.g. local vs EC2) for parity.
Ignores environment-specific fields: exported_at, cwd, project_root, parquet_path (full path), saved_metrics_resolved_path.
Compares metrics, executive summary, and chart data.

Usage:
  python -m dashboard.compare_snapshots dashboard/local_snapshot.json dashboard/external_snapshot.json
  python -m dashboard.compare_snapshots  (defaults: local_snapshot.json, external_snapshot.json)
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Keys we ignore when comparing (environment-specific)
IGNORE_KEYS = frozenset({
    "exported_at", "environment", "parquet_path", "saved_metrics_resolved_path",
    "data_source",  # we compare data_source.rows and columns_sample manually; path differs
})

# Within data_source we only compare these
DATA_SOURCE_COMPARE = frozenset({"rows", "parquet_exists", "columns_sample"})


def _norm(x):
    """Normalize for comparison: NaN -> None, float tolerance."""
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _float_eq(a, b, rel_tol=1e-9):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        if math.isnan(fa) or math.isnan(fb):
            return False
        return math.isclose(fa, fb, rel_tol=rel_tol)
    except (TypeError, ValueError):
        return a == b


def _deep_compare(path, a, b, diffs, ignore_keys=None):
    """Compare two values; append differences to diffs. path is the key path for messages."""
    ignore_keys = ignore_keys or set()
    if path and path.split(".")[-1] in (IGNORE_KEYS | (ignore_keys or set())):
        return
    if type(a) != type(b) and not (_norm(a) == _norm(b)):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not _float_eq(a, b):
                diffs.append(f"{path}: {a} != {b}")
        else:
            diffs.append(f"{path}: type or value mismatch ({type(a).__name__}) {a} vs ({type(b).__name__}) {b}")
        return
    if isinstance(a, dict):
        all_keys = set(a.keys()) | set(b.keys())
        for k in all_keys:
            if k in IGNORE_KEYS and path == "":
                if k == "data_source":
                    # Compare only rows, parquet_exists, columns_sample
                    va, vb = a.get(k), b.get(k)
                    if isinstance(va, dict) and isinstance(vb, dict):
                        for dk in DATA_SOURCE_COMPARE:
                            if dk in va or dk in vb:
                                _deep_compare(f"{path}.data_source.{dk}", va.get(dk), vb.get(dk), diffs)
                continue
            _deep_compare(f"{path}.{k}" if path else k, a.get(k), b.get(k), diffs)
        return
    if isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: list length {len(a)} != {len(b)}")
            return
        for i, (ea, eb) in enumerate(zip(a, b)):
            _deep_compare(f"{path}[{i}]", ea, eb, diffs)
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not _float_eq(a, b):
            diffs.append(f"{path}: {a} != {b}")
        return
    if a != b:
        # NaN handling
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return
        try:
            if math.isnan(a) and math.isnan(b):
                return
        except (TypeError, ValueError):
            pass
        diffs.append(f"{path}: {a!r} != {b!r}")


def main():
    parser = argparse.ArgumentParser(description="Compare two dashboard snapshots for parity.")
    parser.add_argument("local", nargs="?", default="dashboard/local_snapshot.json", help="Local snapshot JSON path")
    parser.add_argument("external", nargs="?", default="dashboard/external_snapshot.json", help="External (EC2) snapshot JSON path")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    local_path = base / args.local if not Path(args.local).is_absolute() else Path(args.local)
    external_path = base / args.external if not Path(args.external).is_absolute() else Path(args.external)

    if not local_path.exists():
        print(f"Local snapshot not found: {local_path}", file=sys.stderr)
        print("Run: python -m dashboard.export_dashboard_snapshot --output dashboard/local_snapshot.json", file=sys.stderr)
        sys.exit(2)
    if not external_path.exists():
        print(f"External snapshot not found: {external_path}", file=sys.stderr)
        print("To check EC2 parity:", file=sys.stderr)
        print("  1. SSH to EC2 and run:", file=sys.stderr)
        print("     cd ~/capstone-app && source ~/venv/bin/activate", file=sys.stderr)
        print("     python -m dashboard.export_dashboard_snapshot --output external_snapshot.json", file=sys.stderr)
        print("  2. Copy external_snapshot.json to your machine into LMUCapstoneProject/dashboard/external_snapshot.json", file=sys.stderr)
        print("  3. Run this script again.", file=sys.stderr)
        sys.exit(2)

    with open(local_path, encoding="utf-8") as f:
        local = json.load(f)
    with open(external_path, encoding="utf-8") as f:
        external = json.load(f)

    diffs = []
    _deep_compare("", local, external, diffs)

    if not diffs:
        print("PARITY OK: Local and external snapshots match (metrics and chart data).")
        print(f"  Rows: {local.get('data_source', {}).get('rows')}")
        print(f"  AUC: {local.get('model_metrics', {}).get('auc')}")
        print(f"  F1:  {local.get('model_metrics', {}).get('f1')}")
        sys.exit(0)

    print("PARITY DIFFERENCES:", file=sys.stderr)
    for d in diffs:
        print(f"  {d}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
