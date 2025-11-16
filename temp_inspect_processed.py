import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('.')
from dashboard.data.loader import process_dataframe

raw_df = pd.read_parquet('data/parquet_export/donors_with_network_features.parquet')
processed = process_dataframe(raw_df)

prob = pd.to_numeric(processed['predicted_prob'], errors='coerce').fillna(0)
segments = processed['segment'].astype(str)
last_gift_cols = [c for c in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount'] if c in processed.columns]
if last_gift_cols:
    lg_col = processed[last_gift_cols[0]]
    if isinstance(lg_col, pd.DataFrame):
        lg_col = lg_col.iloc[:, 0]
    last_gift = pd.to_numeric(lg_col, errors='coerce').fillna(0).clip(lower=0)
else:
    last_gift = pd.Series(np.zeros(len(processed)))

tg_col = processed['total_giving']
if isinstance(tg_col, pd.DataFrame):
    tg_col = tg_col.iloc[:, 0]
total_giving = pd.to_numeric(tg_col, errors='coerce').fillna(0)
tg75 = total_giving.quantile(0.75)

quick_mask = (prob >= 0.7) & (segments == 'Recent (0-6mo)')
cultivation_mask = (prob >= 0.4) & (prob < 0.7) & (total_giving >= tg75)
reeng_mask = (prob >= 0.6) & segments.isin(['Lapsed (1-2yr)', 'Very Lapsed (2yr+)'])

fusion_rate = processed.loc[processed['predicted_prob'] >= 0.5, 'actual_gave'].mean()
if np.isnan(fusion_rate):
    fusion_rate = 0.2

for name, mask in [('Quick Wins', quick_mask), ('Cultivation Targets', cultivation_mask), ('Re-engagement', reeng_mask)]:
    subset = processed.loc[mask]
    if subset.empty:
        print(f"{name}: no rows")
        continue
    gifts = last_gift.loc[mask]
    median = float(gifts.median())
    mean = float(gifts.mean())
    avg_used = median if median > 0 else (mean if mean > 0 else 500)
    revenue = len(subset) * avg_used * fusion_rate
    print(f"{name}: count={len(subset):,}, median={median}, mean={mean}, avg_used={avg_used}, fusion_rate={fusion_rate}, revenue_est={revenue}")
