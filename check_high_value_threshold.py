import pandas as pd

df = pd.read_parquet('data/parquet_export/donors_with_network_features.parquet')

# Filter unassigned donors
unassigned = df[(df['Primary_Manager'].isna()) | (df['Primary_Manager'] == '')].copy()

# Get columns
prob_col = 'Will_Give_Again_Probability'
giving_col = None
for col in ['total_giving', 'Lifetime_Giving', 'LifetimeGiving', 'Lifetime Giving']:
    if col in unassigned.columns:
        giving_col = col
        break

if prob_col in unassigned.columns and giving_col:
    prob_values = pd.to_numeric(unassigned[prob_col], errors='coerce').fillna(0)
    giving_values = pd.to_numeric(unassigned[giving_col], errors='coerce').fillna(0)
    
    prob_threshold = 0.5
    giving_threshold = giving_values.quantile(0.75)
    
    high_value_mask = (prob_values >= prob_threshold) & (giving_values >= giving_threshold)
    high_value_count = high_value_mask.sum()
    
    print('='*70)
    print('HIGH VALUE PROSPECT DEFINITION')
    print('='*70)
    print(f'\nCriteria (both must be met):')
    print(f'1. Will Give Again Probability >= {prob_threshold:.1%} (50%)')
    print(f'2. Lifetime Giving >= ${giving_threshold:,.2f} (75th percentile)')
    
    print(f'\nTotal unassigned donors: {len(unassigned):,}')
    print(f'High-value prospects meeting criteria: {high_value_count:,}')
    print(f'Percentage of unassigned: {high_value_count/len(unassigned)*100:.1f}%')
    
    print(f'\nLifetime Giving Statistics (unassigned donors):')
    print(f'  25th percentile: ${giving_values.quantile(0.25):,.2f}')
    print(f'  50th percentile (median): ${giving_values.quantile(0.50):,.2f}')
    print(f'  75th percentile (threshold): ${giving_threshold:,.2f}')
    print(f'  90th percentile: ${giving_values.quantile(0.90):,.2f}')
    print(f'  Maximum: ${giving_values.max():,.2f}')
    
    print(f'\nWill Give Again Probability Statistics (unassigned donors):')
    print(f'  Minimum: {prob_values.min():.1%}')
    print(f'  25th percentile: {prob_values.quantile(0.25):.1%}')
    print(f'  50th percentile (median): {prob_values.quantile(0.50):.1%}')
    print(f'  75th percentile: {prob_values.quantile(0.75):.1%}')
    print(f'  Maximum: {prob_values.max():.1%}')
else:
    print("Required columns not found")

