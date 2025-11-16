import pandas as pd
import numpy as np
from pathlib import Path

path = Path('data/parquet_export/donors_with_network_features.parquet')
if not path.exists():
    raise SystemExit('Dataset missing')

df = pd.read_parquet(path)
prob_col = 'predicted_prob' if 'predicted_prob' in df.columns else (
    'Will_Give_Again_Probability' if 'Will_Give_Again_Probability' in df.columns else None
)
if prob_col is None:
    raise SystemExit('Probability column missing')

prob = pd.to_numeric(df[prob_col], errors='coerce').fillna(0).values
segments = df['segment'].astype(str).values if 'segment' in df.columns else None
lg_cols = [c for c in ('Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount') if c in df.columns]
if lg_cols:
    last_gift_series = pd.to_numeric(df[lg_cols[0]], errors='coerce').fillna(0).clip(lower=0)
else:
    last_gift_series = pd.Series(np.zeros(len(df)))

quick_mask = (prob >= 0.7) & (segments == 'Recent (0-6mo)') if segments is not None else np.zeros(len(df), dtype=bool)
if 'total_giving' in df.columns:
    tg = pd.to_numeric(df['total_giving'], errors='coerce').fillna(0)
    tg75 = tg.quantile(0.75)
    cultivation_mask = (prob >= 0.4) & (prob < 0.7) & (tg >= tg75)
else:
    cultivation_mask = np.zeros(len(df), dtype=bool)
reeng_mask = np.zeros(len(df), dtype=bool)
if segments is not None:
    reeng_mask = (prob >= 0.6) & np.isin(segments, ['Lapsed (1-2yr)', 'Very Lapsed (2yr+)'])

fusion_rate = df.loc[pd.to_numeric(df[prob_col], errors='coerce') >= 0.5, 'actual_gave'].mean() if 'actual_gave' in df.columns else 0.2
if np.isnan(fusion_rate):
    fusion_rate = 0.2

def describe(name, mask):
    subset = df.loc[mask]
    if subset.empty:
        print(f"{name}: no rows")
        return
    gift = pd.to_numeric(subset[lg_cols[0]], errors='coerce').fillna(0).clip(lower=0) if lg_cols else pd.Series([0])
    median = gift.median()
    mean = gift.mean()
    avg_used = median if median > 0 else (mean if mean > 0 else 500)
    revenue = len(subset) * avg_used * fusion_rate
    print(f"{name}: count={len(subset):,}, median=, mean=, avg_used=, revenue_est=")

for name, mask in [
    ('Quick Wins', quick_mask),
    ('Cultivation Targets', cultivation_mask),
    ('Re-engagement', reeng_mask),
]:
    describe(name, mask)
