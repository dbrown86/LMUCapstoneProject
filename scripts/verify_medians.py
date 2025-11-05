import pandas as pd
from pathlib import Path
import sys

f = Path('data/parquet_export/donors_with_network_features.parquet')
if not f.exists():
    print(' Parquet not found:', f); sys.exit(1)

df = pd.read_parquet(f, engine='pyarrow')
prob = 'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df.columns else ('predicted_prob' if 'predicted_prob' in df.columns else None)
if prob is None:
    print(' No prediction probability column found'); sys.exit(1)
if 'donor_type' not in df.columns:
    for c in ['Donor_Type','Primary_Constituent_Type','type']:
        if c in df.columns:
            df['donor_type'] = df[c]
            break
if 'donor_type' not in df.columns:
    print(' donor_type column not found'); sys.exit(1)

s = pd.to_numeric(df[prob], errors='coerce')
med = (pd.DataFrame({'donor_type': df['donor_type'], 'prob': s})
         .dropna(subset=['prob'])
         .groupby('donor_type')['prob']
         .median()
         .sort_values(ascending=False))

cnt = df.groupby('donor_type').size().reindex(med.index)
print('\n=== Median Will-Give-Again Probability by Donor Type ===')
for idx, val in med.items():
    n = int(cnt.loc[idx]) if idx in cnt.index else 0
    print(f"{idx}: median={val:.4f} ({val:.1%}), n={n}")
