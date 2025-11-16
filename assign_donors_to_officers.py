"""
Script to assign high-value donors to gift officers.
Each gift officer gets 100-150 donors based on high lifetime giving and high ratings.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def assign_donors_to_officers():
    """Assign high-value donors to gift officers"""
    
    # Load the dataset
    data_path = Path('data/parquet_export/donors_with_network_features.parquet')
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        return None
    
    print("Loading donor dataset...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} total records")
    
    # Get list of gift officers (from existing Primary_Manager column or create default list)
    if 'Primary_Manager' in df.columns:
        officers = df['Primary_Manager'].dropna().unique()
        officers = [o for o in officers if str(o).strip() != 'Unassigned' and str(o).strip() != '']
        print(f"Found {len(officers)} gift officers: {list(officers)[:10]}...")
    else:
        # Default list if no officers exist
        officers = [
            'Emily Zhang', 'Lisa Thompson', 'Sarah Chen', 'Rachel Johnson', 
            'Kevin Wu', 'Nicole Brown', 'Amanda Davis', 'Michael Rodriguez',
            'David Park', 'Christopher Lee', 'Jennifer Liu', 'Maria Gonzalez',
            'James Wilson', 'Robert Kim', 'Mark Williams'
        ]
        print(f"Using default list of {len(officers)} gift officers")
    
    # Filter for high-value donors
    # Criteria: High lifetime giving (top quartile) AND high rating
    print("\nFiltering high-value donors...")
    
    # Get lifetime giving column
    giving_col = None
    for col in ['Lifetime_Giving', 'total_giving', 'LifetimeGiving', 'Lifetime Giving']:
        if col in df.columns:
            giving_col = col
            break
    
    if giving_col is None:
        print("ERROR: Could not find lifetime giving column")
        return None
    
    # Get rating column
    rating_col = None
    for col in ['Rating', 'rating', 'Donor_Rating', 'donor_rating']:
        if col in df.columns:
            rating_col = col
            break
    
    # Convert giving to numeric
    df[giving_col] = pd.to_numeric(df[giving_col], errors='coerce').fillna(0)
    
    # Calculate thresholds
    giving_threshold = df[giving_col].quantile(0.75)  # Top quartile
    print(f"Lifetime giving threshold (75th percentile): ${giving_threshold:,.2f}")
    
    # Filter for high lifetime giving
    high_giving = df[df[giving_col] >= giving_threshold].copy()
    print(f"Found {len(high_giving):,} donors with high lifetime giving")
    
    # Calculate assignment parameters (needed for validation)
    num_officers = len(officers)
    min_per_officer = 100
    max_per_officer = 150
    
    # If rating column exists, filter for high ratings
    if rating_col:
        # Get unique ratings and identify high ratings
        unique_ratings = high_giving[rating_col].dropna().unique()
        print(f"Available ratings: {sorted(unique_ratings)[:10]}")
        
        # Try to identify high ratings
        if high_giving[rating_col].dtype == 'object':
            # String ratings - use A, B, C (top 3 ratings alphabetically) to get enough donors
            # Get unique ratings and sort them to find top ones (A, B, C are typically highest)
            unique_ratings_sorted = sorted(high_giving[rating_col].dropna().unique())
            # Take top 3 ratings (or more if needed to get enough donors)
            top_ratings = unique_ratings_sorted[:3]
            print(f"Using top ratings: {top_ratings}")
            high_value = high_giving[high_giving[rating_col].isin(top_ratings)].copy()
            
            # If not enough with top 3, expand to top 5
            if len(high_value) < num_officers * min_per_officer and len(unique_ratings_sorted) > 3:
                top_ratings = unique_ratings_sorted[:5]
                print(f"Expanding to top 5 ratings: {top_ratings}")
                high_value = high_giving[high_giving[rating_col].isin(top_ratings)].copy()
        else:
            # Numeric ratings - top 50% (median) to get enough donors
            rating_threshold = high_giving[rating_col].quantile(0.5)
            high_value = high_giving[high_giving[rating_col] >= rating_threshold].copy()
        
        print(f"Found {len(high_value):,} donors with high lifetime giving AND high ratings")
        
        # If still not enough, relax rating requirement
        if len(high_value) < num_officers * min_per_officer:
            print(f"Not enough donors with both criteria. Using only high lifetime giving...")
            high_value = high_giving.copy()
    else:
        # No rating column - use only high lifetime giving
        high_value = high_giving.copy()
        print(f"WARNING: No rating column found - using only lifetime giving criteria")
    
    # Sort by lifetime giving (descending) to prioritize highest value donors
    high_value = high_value.sort_values(giving_col, ascending=False).reset_index(drop=True)
    
    # Calculate assignment parameters
    target_per_officer = (min_per_officer + max_per_officer) // 2  # 125
    
    total_to_assign = min(len(high_value), num_officers * max_per_officer)
    high_value = high_value.head(total_to_assign)
    
    print(f"\nAssignment parameters:")
    print(f"   - Number of officers: {num_officers}")
    print(f"   - Target per officer: {target_per_officer}")
    print(f"   - Range: {min_per_officer}-{max_per_officer} per officer")
    print(f"   - Total donors to assign: {len(high_value):,}")
    
    # Assign donors to officers
    # Round-robin assignment to distribute evenly, but sorted by value
    assignments = []
    officer_counts = {officer: 0 for officer in officers}
    
    for idx, row in high_value.iterrows():
        # Round-robin assignment
        officer_idx = idx % num_officers
        officer = officers[officer_idx]
        
        # Check if officer has reached max capacity
        if officer_counts[officer] >= max_per_officer:
            # Find next available officer
            for i in range(num_officers):
                next_officer_idx = (officer_idx + i + 1) % num_officers
                next_officer = officers[next_officer_idx]
                if officer_counts[next_officer] < max_per_officer:
                    officer = next_officer
                    break
        
        assignments.append({
            'Donor_ID': row.get('ID', row.get('Donor_ID', idx)),
            'Full_Name': row.get('Full_Name', ''),
            'Lifetime_Giving': row[giving_col],
            'Rating': row.get(rating_col, 'N/A') if rating_col else 'N/A',
            'Assigned_Officer': officer
        })
        officer_counts[officer] += 1
    
    # Create assignment dataframe
    assignment_df = pd.DataFrame(assignments)
    
    # Display results
    print("\n" + "="*60)
    print("ASSIGNMENT RESULTS")
    print("="*60)
    
    summary = assignment_df.groupby('Assigned_Officer').agg(
        Count=('Assigned_Officer', 'size'),
        Avg_Lifetime_Giving=('Lifetime_Giving', 'mean'),
        Total_Lifetime_Giving=('Lifetime_Giving', 'sum')
    ).reset_index()
    
    summary = summary.sort_values('Count', ascending=False)
    
    print("\nAssignments per Officer:")
    print(summary.to_string(index=False))
    
    print(f"\nSummary Statistics:")
    print(f"   - Total assigned: {len(assignment_df):,}")
    print(f"   - Average per officer: {summary['Count'].mean():.1f}")
    print(f"   - Min per officer: {summary['Count'].min()}")
    print(f"   - Max per officer: {summary['Count'].max()}")
    print(f"   - Total lifetime giving assigned: ${summary['Total_Lifetime_Giving'].sum():,.2f}")
    print(f"   - Average lifetime giving per donor: ${assignment_df['Lifetime_Giving'].mean():,.2f}")
    
    # Verify all officers have 100-150 assignments
    print(f"\nVerification:")
    all_valid = True
    for _, row in summary.iterrows():
        count = row['Count']
        if count < min_per_officer or count > max_per_officer:
            print(f"   WARNING: {row['Assigned_Officer']}: {count} assignments (outside {min_per_officer}-{max_per_officer} range)")
            all_valid = False
    
    if all_valid:
        print(f"   SUCCESS: All officers have {min_per_officer}-{max_per_officer} assignments")
    
    # Save results
    output_path = Path('data/processed/gift_officer_assignments.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_df.to_csv(output_path, index=False)
    print(f"\nSaved assignments to: {output_path}")
    
    # Also save summary
    summary_path = Path('data/processed/gift_officer_assignment_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")
    
    return assignment_df, summary

if __name__ == "__main__":
    assignment_df, summary = assign_donors_to_officers()
    
    if assignment_df is not None:
        print("\nAssignment complete! Review the results above.")
        print("\nTo use these assignments, you can update the Primary_Manager column in your dataset.")
    else:
        print("\nAssignment failed. Please check the errors above.")

