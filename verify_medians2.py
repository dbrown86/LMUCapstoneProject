import pandas as pd
from pathlib import Path
import sys, traceback

try:
    f = Path("data/parquet_export/donors_with_network_features.parquet")
    if not f.exists():
        print("Parquet not found:", f); sys.exit(1)

    df = pd.read_parquet(f, engine="pyarrow")

    prob = "Will_Give_Again_Probability" if "Will_Give_Again_Probability" in df.columns else ("predicted_prob" if "predicted_prob" in df.columns else None)
    if prob is None:
        print("No prediction probability column found"); sys.exit(1)

    if "donor_type" not in df.columns:
        for c in ("Donor_Type","Primary_Constituent_Type","type"):
            if c in df.columns:
                df["donor_type"] = df[c]
                break
    if "donor_type" not in df.columns:
        print("donor_type column not found"); sys.exit(1)

    med = df.groupby("donor_type")[prob].median().sort_values(ascending=False)
    print("=== Median probability by donor_type ===")
    for k, v in med.items():
        print(f"{k}: {v:.4f} ({v:.1%})")
except Exception:
    print("Error while computing medians:\n" + traceback.format_exc())
