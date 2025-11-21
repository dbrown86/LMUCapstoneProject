#!/usr/bin/env python3
"""
Script to clean up the Kaggle export CSV files.
- Removes duplicate prediction columns
- Standardizes column names (region, donor_type)
- Creates cleaned version ready for Kaggle
"""

import pandas as pd
import sys
import zipfile
from pathlib import Path
import shutil

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def clean_donors_file(input_path, output_path):
    """Clean the donors.csv file."""
    print("\n" + "=" * 80)
    print("CLEANING: donors.csv")
    print("=" * 80)
    
    df = pd.read_csv(input_path)
    
    print(f"\n📊 Original file:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    
    # Find duplicate prediction columns
    duplicate_cols = []
    if 'Will_Give_Again_Probability_x' in df.columns:
        duplicate_cols.append('Will_Give_Again_Probability_x')
        print(f"\n   Found duplicate: Will_Give_Again_Probability_x")
    if 'Will_Give_Again_Probability_y' in df.columns:
        duplicate_cols.append('Will_Give_Again_Probability_y')
        print(f"   Found duplicate: Will_Give_Again_Probability_y")
    
    # Remove duplicate columns
    if duplicate_cols:
        print(f"\n🗑️  Removing duplicate columns: {duplicate_cols}")
        df = df.drop(columns=duplicate_cols)
        print(f"   ✅ Removed {len(duplicate_cols)} duplicate column(s)")
    
    # Check for region column (might be named differently)
    region_cols = [c for c in df.columns if 'region' in c.lower() or 'geographic' in c.lower()]
    if region_cols:
        print(f"\n🌍 Region columns found: {region_cols}")
        # Use Geographic_Region if it exists, otherwise keep as is
        if 'Geographic_Region' in df.columns and 'region' not in df.columns:
            print(f"   ✅ Renaming 'Geographic_Region' to 'region' for consistency")
            df = df.rename(columns={'Geographic_Region': 'region'})
    else:
        print(f"\n   ⚠️  No region column found")
    
    # Check for donor_type column (might be named differently)
    type_cols = [c for c in df.columns if 'type' in c.lower() and 'constituent' in c.lower()]
    if type_cols:
        print(f"\n👤 Donor type columns found: {type_cols}")
        # Use Primary_Constituent_Type if it exists, otherwise keep as is
        if 'Primary_Constituent_Type' in df.columns and 'donor_type' not in df.columns:
            print(f"   ✅ Renaming 'Primary_Constituent_Type' to 'donor_type' for consistency")
            df = df.rename(columns={'Primary_Constituent_Type': 'donor_type'})
    else:
        print(f"\n   ⚠️  No donor_type column found")
    
    # Verify final columns
    print(f"\n📋 Final columns: {len(df.columns)}")
    
    # Verify key columns still exist
    required_cols = ['ID', 'Will_Give_Again_Probability', 'Gave_Again_In_2025']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"\n❌ ERROR: Missing required columns: {missing}")
        return None
    else:
        print(f"\n✅ All required columns present")
    
    # Save cleaned file
    print(f"\n💾 Saving cleaned file to: {output_path}")
    df.to_csv(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"   ✅ Saved {len(df):,} rows, {len(df.columns)} columns ({file_size_mb:.2f} MB)")
    
    # Verify the cleaned file
    print(f"\n🔍 Verifying cleaned file...")
    verify_df = pd.read_csv(output_path)
    
    # Check predictions
    if 'Will_Give_Again_Probability' in verify_df.columns:
        prob_col = verify_df['Will_Give_Again_Probability']
        print(f"   ✅ Will_Give_Again_Probability: {prob_col.notna().sum():,} values")
        print(f"      Range: {prob_col.min():.6f} - {prob_col.max():.6f}")
    
    if 'Gave_Again_In_2025' in verify_df.columns:
        outcome_col = verify_df['Gave_Again_In_2025']
        print(f"   ✅ Gave_Again_In_2025: {outcome_col.sum():,} donors gave ({(outcome_col.mean()*100):.1f}%)")
    
    # Check for duplicates
    if 'Will_Give_Again_Probability_x' in verify_df.columns or 'Will_Give_Again_Probability_y' in verify_df.columns:
        print(f"   ⚠️  Warning: Duplicate columns still present!")
    else:
        print(f"   ✅ No duplicate prediction columns")
    
    return df

def update_zip_archive(export_dir, base_path):
    """Update the zip archive with cleaned files."""
    zip_path = base_path / 'kaggle_dataset.zip'
    
    print(f"\n📦 Updating zip archive: {zip_path.name}")
    
    # Remove old zip if exists
    if zip_path.exists():
        zip_path.unlink()
        print(f"   Removed old archive")
    
    # Create new zip with cleaned files
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all CSV files from export directory
        for csv_file in sorted(export_dir.glob('*.csv')):
            print(f"   Adding {csv_file.name}...")
            zipf.write(csv_file, arcname=csv_file.name)
        
        # Add README if exists
        readme_path = export_dir / 'README.md'
        if readme_path.exists():
            print(f"   Adding README.md...")
            zipf.write(readme_path, arcname='README.md')
    
    file_size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"\n✅ Created updated {zip_path.name} ({file_size_mb:.2f} MB)")

def main():
    """Main cleaning function."""
    print("=" * 80)
    print("KAGGLE EXPORT CLEANUP SCRIPT")
    print("=" * 80)
    
    base_path = Path(__file__).parent
    export_dir = base_path / 'kaggle_export'
    
    if not export_dir.exists():
        print(f"\n❌ Error: {export_dir} directory not found!")
        return
    
    # Clean donors file
    donors_input = export_dir / 'donors.csv'
    donors_output = export_dir / 'donors.csv'
    
    if not donors_input.exists():
        print(f"\n❌ Error: {donors_input} not found!")
        return
    
    # Create backup
    backup_path = export_dir / 'donors_backup.csv'
    print(f"\n💾 Creating backup: {backup_path.name}")
    shutil.copy2(donors_input, backup_path)
    print(f"   ✅ Backup created")
    
    # Clean the file
    cleaned_df = clean_donors_file(donors_input, donors_output)
    
    if cleaned_df is None:
        print(f"\n❌ Cleaning failed! Restoring from backup...")
        shutil.copy2(backup_path, donors_output)
        return
    
    # Update zip archive
    update_zip_archive(export_dir, base_path)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ CLEANUP COMPLETE!")
    print("=" * 80)
    print(f"\n📋 Changes made:")
    print(f"   ✅ Removed duplicate prediction columns")
    print(f"   ✅ Standardized column names (region, donor_type)")
    print(f"   ✅ Updated kaggle_dataset.zip")
    print(f"   ✅ Backup saved: {backup_path.name}")
    
    print(f"\n🚀 Ready for Kaggle upload!")
    print(f"   File: {base_path / 'kaggle_dataset.zip'}")

if __name__ == '__main__':
    main()

