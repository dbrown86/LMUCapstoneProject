#!/usr/bin/env python3
"""
One-time preprocessing: Export SQLite to Parquet with all joins resolved
This avoids SQL parameter limits by doing the joins once and saving to efficient format
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from src.sql_data_loader import SQLDataLoader
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

def export_to_parquet(db_path: str = "data/synthetic_donor_dataset_500k_dense/donor_database.db",
                     output_dir: str = "data/parquet_export",
                     batch_size: int = 10000):
    """
    Export SQL database to Parquet format with all joins resolved
    
    Args:
        db_path: Path to SQLite database
        output_dir: Directory to save Parquet files
        batch_size: Number of records to process at a time
    """
    print("🚀 EXPORTING SQL DATABASE TO PARQUET FORMAT")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with SQLDataLoader(db_path) as loader:
        # Get database statistics
        print("\n📊 Database Statistics:")
        stats = loader.get_database_stats()
        for table, count in stats.items():
            print(f"   {table}: {count:,} records")
        
        # Export core donors with enhanced fields (joined)
        print("\n📂 Exporting donors with enhanced fields...")
        donors_df = loader.get_donors()
        enhanced_df = loader.get_enhanced_fields()
        
        print(f"   Merging {len(donors_df):,} donors with {len(enhanced_df):,} enhanced records...")
        # Merge on donor ID
        merged_df = donors_df.merge(
            enhanced_df, 
            left_on='ID', 
            right_on='Donor_ID', 
            how='left',
            suffixes=('', '_enhanced')
        )
        
        # Drop duplicate ID column
        if 'Donor_ID' in merged_df.columns:
            merged_df = merged_df.drop(columns=['Donor_ID'])
        
        print(f"   ✅ Merged dataset: {len(merged_df):,} records, {len(merged_df.columns)} columns")
        
        # Save to Parquet
        parquet_file = output_path / 'donors_with_features.parquet'
        merged_df.to_parquet(parquet_file, index=False, compression='snappy')
        print(f"   💾 Saved to {parquet_file}")
        print(f"   📊 File size: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Export relationships (in batches to avoid memory issues)
        print("\n📂 Exporting relationships...")
        relationships_df = loader.get_relationships(limit=None)
        
        if relationships_df is not None and len(relationships_df) > 0:
            parquet_file = output_path / 'relationships.parquet'
            relationships_df.to_parquet(parquet_file, index=False, compression='snappy')
            print(f"   ✅ Saved {len(relationships_df):,} relationships")
            print(f"   💾 Saved to {parquet_file}")
            print(f"   📊 File size: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Export giving history
        print("\n📂 Exporting giving history...")
        giving_df = loader.get_giving_history()
        
        if giving_df is not None and len(giving_df) > 0:
            parquet_file = output_path / 'giving_history.parquet'
            giving_df.to_parquet(parquet_file, index=False, compression='snappy')
            print(f"   ✅ Saved {len(giving_df):,} giving records")
            print(f"   💾 Saved to {parquet_file}")
            print(f"   📊 File size: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Export event attendance
        print("\n📂 Exporting event attendance...")
        events_df = loader.get_event_attendance()
        
        if events_df is not None and len(events_df) > 0:
            parquet_file = output_path / 'event_attendance.parquet'
            events_df.to_parquet(parquet_file, index=False, compression='snappy')
            print(f"   ✅ Saved {len(events_df):,} event records")
            print(f"   💾 Saved to {parquet_file}")
            print(f"   📊 File size: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Export contact reports
        print("\n📂 Exporting contact reports...")
        contacts_df = loader.get_contact_reports()
        
        if contacts_df is not None and len(contacts_df) > 0:
            parquet_file = output_path / 'contact_reports.parquet'
            contacts_df.to_parquet(parquet_file, index=False, compression='snappy')
            print(f"   ✅ Saved {len(contacts_df):,} contact records")
            print(f"   💾 Saved to {parquet_file}")
            print(f"   📊 File size: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Export family relationships
        print("\n📂 Exporting family relationships...")
        family_df = loader.get_family_relationships()
        
        if family_df is not None and len(family_df) > 0:
            parquet_file = output_path / 'family_relationships.parquet'
            family_df.to_parquet(parquet_file, index=False, compression='snappy')
            print(f"   ✅ Saved {len(family_df):,} family records")
            print(f"   💾 Saved to {parquet_file}")
            print(f"   📊 File size: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_path.glob('*.parquet'))
    print(f"\n✅ EXPORT COMPLETE!")
    print(f"   📊 Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"   📁 Output directory: {output_path.absolute()}")
    
    return output_path

def verify_export(output_dir: str = "data/parquet_export"):
    """Verify the exported Parquet files"""
    print("\n🔍 VERIFYING PARQUET EXPORT")
    print("=" * 60)
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ Output directory not found: {output_path}")
        return False
    
    # Check each file
    expected_files = [
        'donors_with_features.parquet',
        'relationships.parquet',
        'giving_history.parquet',
        'event_attendance.parquet',
        'contact_reports.parquet',
        'family_relationships.parquet'
    ]
    
    for filename in expected_files:
        filepath = output_path / filename
        if filepath.exists():
            df = pd.read_parquet(filepath)
            print(f"✅ {filename}: {len(df):,} records, {len(df.columns)} columns")
        else:
            print(f"⚠️ {filename}: Not found")
    
    print("\n✅ Verification complete!")
    return True

if __name__ == "__main__":
    # Export database to Parquet
    output_path = export_to_parquet()
    
    # Verify export
    verify_export(str(output_path))



