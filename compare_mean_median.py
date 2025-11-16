import pandas as pd
import numpy as np

# Load data
df = pd.read_parquet('data/parquet_export/donors_with_network_features.parquet')

# Filter assigned donors
assigned = df[df['Primary_Manager'].notna() & (df['Primary_Manager'] != '')].copy()
prob_col = 'Will_Give_Again_Probability'

# Convert to numeric
assigned[prob_col] = pd.to_numeric(assigned[prob_col], errors='coerce')

# Calculate both mean and median per officer
officer_stats = assigned.groupby('Primary_Manager', observed=False).agg(
    Avg_Probability=(prob_col, 'mean'),
    Median_Probability=(prob_col, 'median'),
    Count=(prob_col, 'size'),
    Std_Deviation=(prob_col, 'std')
).reset_index()

# Calculate differences
officer_stats['Difference'] = officer_stats['Avg_Probability'] - officer_stats['Median_Probability']
officer_stats['Abs_Difference'] = officer_stats['Difference'].abs()
officer_stats['Pct_Difference'] = (officer_stats['Difference'] / officer_stats['Median_Probability'] * 100)

# Sort by average probability
officer_stats = officer_stats.sort_values('Avg_Probability', ascending=False)

print('='*100)
print('COMPARISON: Average vs Median Will Give Again Probability per Officer')
print('='*100)
print()

# Display detailed comparison
display_cols = ['Primary_Manager', 'Count', 'Avg_Probability', 'Median_Probability', 
                'Difference', 'Abs_Difference', 'Std_Deviation']
for col in ['Avg_Probability', 'Median_Probability', 'Difference', 'Abs_Difference']:
    officer_stats[col] = officer_stats[col].apply(lambda x: f"{x:.1%}")

print(officer_stats[display_cols].to_string(index=False))

print()
print('='*100)
print('SUMMARY STATISTICS')
print('='*100)

# Convert back to numeric for calculations
officer_stats['Difference'] = pd.to_numeric(officer_stats['Difference'].str.rstrip('%').astype(float) / 100, errors='coerce')
officer_stats['Abs_Difference'] = pd.to_numeric(officer_stats['Abs_Difference'].str.rstrip('%').astype(float) / 100, errors='coerce')

print(f'Mean absolute difference: {officer_stats["Abs_Difference"].mean():.4f} ({officer_stats["Abs_Difference"].mean()*100:.2f} percentage points)')
print(f'Median absolute difference: {officer_stats["Abs_Difference"].median():.4f} ({officer_stats["Abs_Difference"].median()*100:.2f} percentage points)')
print(f'Max absolute difference: {officer_stats["Abs_Difference"].max():.4f} ({officer_stats["Abs_Difference"].max()*100:.2f} percentage points)')
print(f'Min absolute difference: {officer_stats["Abs_Difference"].min():.4f} ({officer_stats["Abs_Difference"].min()*100:.2f} percentage points)')
print(f'Std deviation of absolute differences: {officer_stats["Abs_Difference"].std():.4f} ({officer_stats["Abs_Difference"].std()*100:.2f} percentage points)')
print()

# Check if differences are significant (e.g., > 5 percentage points)
significant_diff = officer_stats[officer_stats['Abs_Difference'] > 0.05]
print(f'Officers with difference > 5 percentage points: {len(significant_diff)} out of {len(officer_stats)}')
if len(significant_diff) > 0:
    print('\nOfficers with largest differences:')
    print(significant_diff[['Primary_Manager', 'Abs_Difference']].head(10).to_string(index=False))

