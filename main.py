# Main entry point for the synthetic donor dataset generation and GNN analysis
import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import date
from faker import Faker

# Add src to path
sys.path.append('src')

from data_generation.donor_generator import (
    generate_random_ids, 
    generate_core_donors_with_families,
    generate_giving_history,
    generate_enhanced_fields,
    generate_contact_reports
)
from data_generation.validation import validate_dataset, create_dataset_visualizations
from gnn_models.gnn_pipeline import main_gnn_pipeline

# Configuration
TOTAL_DONORS = 50000
OUTPUT_DIR = "synthetic_donor_dataset"
START_DATE = date(1940, 1, 1)
END_DATE = date(2025, 12, 31)

def setup_environment():
    """Set up the environment with seeds and output directory"""
    random.seed(42)
    rng = np.random.default_rng(42)
    fake = Faker()
    Faker.seed(42)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Configuration set for {TOTAL_DONORS:,} donors")
    print(f"Output directory: {OUTPUT_DIR}")
    
    return rng

def create_final_tables(donors_df, relationships_df, contact_reports_df, giving_history_df, enhanced_df):
    """Create final normalized tables"""
    print("Creating final normalized tables...")
    
    # Table 1: Core Donors (already includes family fields)
    if enhanced_df is not None:
        donors_final = donors_df.merge(enhanced_df, left_on='ID', right_on='Donor_ID', how='left')
        donors_final = donors_final.drop('Donor_ID', axis=1)
    else:
        donors_final = donors_df.copy()
    
    # Table 2: Family Relationships (created during donor generation)
    relationships_final = relationships_df.copy() if not relationships_df.empty else pd.DataFrame()
    
    # Table 3: Contact Reports
    contact_reports_final = contact_reports_df.copy() if not contact_reports_df.empty else pd.DataFrame()
    
    # Table 4: Giving History
    giving_history_final = giving_history_df.copy() if not giving_history_df.empty else pd.DataFrame()
    
    return donors_final, relationships_final, contact_reports_final, giving_history_final

def export_datasets(donors_final, relationships_final, contact_reports_final, giving_history_final, enhanced_df, output_dir):
    """Export all datasets to CSV files"""
    print("=" * 50)
    print("EXPORTING DATASETS TO CSV FILES")
    print("=" * 50)
    
    # Save main tables
    donors_final.to_csv(f'{output_dir}/donors.csv', index=False)
    relationships_final.to_csv(f'{output_dir}/relationships.csv', index=False)
    contact_reports_final.to_csv(f'{output_dir}/contact_reports.csv', index=False)
    giving_history_final.to_csv(f'{output_dir}/giving_history.csv', index=False)
    enhanced_df.to_csv(f'{output_dir}/enhanced_fields.csv', index=False)
    
    print(f"✅ CSV files saved to: {output_dir}/")
    print("Files created:")
    print(f"  - donors.csv ({len(donors_final):,} records)")
    print(f"  - relationships.csv ({len(relationships_final):,} records)")
    print(f"  - contact_reports.csv ({len(contact_reports_final):,} records)")
    print(f"  - giving_history.csv ({len(giving_history_final):,} records)")
    print(f"  - enhanced_fields.csv ({len(enhanced_df):,} records)")

def main():
    """Main execution function"""
    print("=" * 80)
    print("SYNTHETIC DONOR DATASET GENERATION AND GNN ANALYSIS")
    print("=" * 80)
    
    # Setup environment
    rng = setup_environment()
    
    # Generate unique random ID pool
    DONOR_IDS = generate_random_ids(TOTAL_DONORS)
    print(f"ID range: {min(DONOR_IDS)} to {max(DONOR_IDS)}")
    
    # Step 1: Generate core donor data with family relationships
    print("\n" + "=" * 50)
    print("STEP 1: GENERATING CORE DONOR DATA WITH FAMILY RELATIONSHIPS")
    print("=" * 50)
    
    donors_df, relationships_df = generate_core_donors_with_families(TOTAL_DONORS, DONOR_IDS, rng)
    
    print(f"\nGenerated {len(donors_df):,} donor records")
    print(f"Non-donors: {(donors_df['Lifetime_Giving'] == 0).sum():,}")
    print(f"Donors: {(donors_df['Lifetime_Giving'] > 0).sum():,}")
    print(f"Major donors (>$100K): {(donors_df['Lifetime_Giving'] > 100000).sum():,}")
    print(f"Donors in families: {donors_df['Family_ID'].notna().sum():,}")
    print(f"Number of families: {donors_df['Family_ID'].nunique()}")
    
    # Step 2: Generate detailed giving history
    print("\n" + "=" * 50)
    print("STEP 2: GENERATING DETAILED GIVING HISTORY")
    print("=" * 50)
    
    giving_history_df = generate_giving_history(donors_df, rng)
    
    print(f"Created {len(giving_history_df):,} individual gift records")
    print(f"Average gifts per donor: {len(giving_history_df) / (donors_df['Lifetime_Giving'] > 0).sum():.1f}")
    
    # Verify totals match
    sample_donor = donors_df[donors_df['Lifetime_Giving'] > 0].iloc[0]
    donor_gifts = giving_history_df[giving_history_df['Donor_ID'] == sample_donor['ID']]
    print(f"\nSample verification - Donor {sample_donor['ID']}:")
    print(f"  Expected lifetime giving: ${sample_donor['Lifetime_Giving']:,.2f}")
    print(f"  Sum of individual gifts: ${donor_gifts['Gift_Amount'].sum():,.2f}")
    print(f"  Expected total years: {sample_donor['Total_Yr_Giving_Count']}")
    print(f"  Actual gift records: {len(donor_gifts)}")
    
    # Step 3: Generate enhanced fields
    print("\n" + "=" * 50)
    print("STEP 3: GENERATING ENHANCED FIELDS")
    print("=" * 50)
    
    enhanced_df = generate_enhanced_fields(donors_df)
    
    print(f"Generated enhanced fields for {len(enhanced_df):,} donors")
    print(f"Legacy intent (binary): {enhanced_df['Legacy_Intent_Binary'].sum():,} donors ({enhanced_df['Legacy_Intent_Binary'].sum()/len(enhanced_df)*100:.1f}%)")
    print(f"Board affiliations: {enhanced_df['Board_Affiliations'].notna().sum():,} donors")
    
    # Step 4: Generate contact reports
    print("\n" + "=" * 50)
    print("STEP 4: GENERATING CONTACT REPORTS")
    print("=" * 50)
    
    contact_reports_df = generate_contact_reports(donors_df)
    
    print(f"Created {len(contact_reports_df):,} contact report records")
    print(f"Coverage: {len(contact_reports_df) / len(donors_df) * 100:.1f}% of donors have contact reports")
    
    # Outcome distribution
    outcome_counts = contact_reports_df['Outcome_Category'].value_counts()
    print("\nContact Report Outcomes:")
    for outcome, count in outcome_counts.items():
        print(f"  {outcome}: {count:,} ({count/len(contact_reports_df)*100:.1f}%)")
    
    # Step 5: Create final tables
    print("\n" + "=" * 50)
    print("STEP 5: CREATING FINAL TABLES")
    print("=" * 50)
    
    donors_final, relationships_final, contact_reports_final, giving_history_final = create_final_tables(
        donors_df, relationships_df, contact_reports_df, giving_history_df, enhanced_df
    )
    
    print("Final table sizes:")
    print(f"  Donors: {len(donors_final):,} rows, {len(donors_final.columns)} columns")
    print(f"  Relationships: {len(relationships_final):,} rows")
    print(f"  Contact Reports: {len(contact_reports_final):,} rows")
    print(f"  Giving History: {len(giving_history_final):,} rows")
    
    # Show family integration
    family_donors = donors_final['Family_ID'].notna().sum()
    print(f"\nFamily Integration:")
    print(f"  Donors with family info: {family_donors:,} ({family_donors/len(donors_final)*100:.1f}%)")
    print(f"  Unique families: {donors_final['Family_ID'].nunique()}")
    print(f"  Family relationship types: {donors_final['Relationship_Type'].value_counts().to_dict()}")
    
    # Step 6: Data validation
    print("\n" + "=" * 50)
    print("STEP 6: DATA QUALITY VALIDATION")
    print("=" * 50)
    
    validation_results = validate_dataset(donors_final, relationships_final, contact_reports_final, giving_history_final)
    
    print("Validation Results:")
    print("\nErrors:")
    if validation_results['errors']:
        for error in validation_results['errors']:
            print(f"  ❌ {error}")
    else:
        print("  ✅ No errors found")
    
    print("\nWarnings:")
    if validation_results['warnings']:
        for warning in validation_results['warnings']:
            print(f"  ⚠️ {warning}")
    else:
        print("  ✅ No warnings")
    
    # Step 7: Create visualizations
    print("\n" + "=" * 50)
    print("STEP 7: CREATING VISUALIZATIONS")
    print("=" * 50)
    
    create_dataset_visualizations(donors_final, giving_history_final, contact_reports_final, OUTPUT_DIR)
    
    print(f"\n✅ Analysis complete! Check {OUTPUT_DIR}/dataset_analysis.png for visualizations")
    
    # Step 8: Export datasets
    export_datasets(donors_final, relationships_final, contact_reports_final, giving_history_final, enhanced_df, OUTPUT_DIR)
    
    # Step 9: GNN Analysis (optional)
    print("\n" + "=" * 50)
    print("STEP 8: GRAPH NEURAL NETWORK ANALYSIS")
    print("=" * 50)
    
    run_gnn = input("Would you like to run GNN analysis? (y/n): ").lower().strip() == 'y'
    
    if run_gnn:
        try:
            print("Starting GNN analysis...")
            gnn_results = main_gnn_pipeline(
                donors_final, 
                relationships_final, 
                contact_reports_final, 
                giving_history_final
            )
            print("\n✅ GNN analysis completed successfully!")
        except Exception as e:
            print(f"❌ GNN analysis failed: {e}")
            print("Make sure you have installed the required packages:")
            print("pip install torch torch-geometric scikit-learn matplotlib seaborn networkx")
    else:
        print("Skipping GNN analysis.")
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"All files saved to: {OUTPUT_DIR}/")
    print("You can now use the generated datasets for your analysis.")

if __name__ == "__main__":
    main()
