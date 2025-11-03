#!/usr/bin/env python3
"""
Quick helper to clear cache and check file modification dates
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def clear_streamlit_cache():
    """Clear Streamlit cache"""
    print("=" * 80)
    print("CLEARING STREAMLIT CACHE")
    print("=" * 80)
    
    cache_dirs = [
        Path(".streamlit/cache"),
        Path(".streamlit"),
        Path("__pycache__"),
    ]
    
    cleared = False
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            try:
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir)
                    print(f"✅ Deleted: {cache_dir}")
                    cleared = True
                else:
                    cache_dir.unlink()
                    print(f"✅ Deleted: {cache_dir}")
                    cleared = True
            except Exception as e:
                print(f"⚠️ Could not delete {cache_dir}: {e}")
    
    if not cleared:
        print("ℹ️ No cache directories found to delete.")
        print("   You may need to restart Streamlit manually.")
    
    print("\n" + "-" * 80 + "\n")

def check_file_modification_date():
    """Check when the data file was last modified"""
    print("=" * 80)
    print("CHECKING FILE MODIFICATION DATE")
    print("=" * 80)
    
    data_file = Path("data/parquet_export/donors_with_network_features.parquet")
    
    if data_file.exists():
        mod_time = datetime.fromtimestamp(data_file.stat().st_mtime)
        now = datetime.now()
        age = now - mod_time
        
        print(f"\n📁 File: {data_file}")
        print(f"   Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Age: {age.days} days, {age.seconds // 3600} hours ago")
        
        if age.days < 1:
            print("\n   ⚠️ WARNING: File was modified within the last 24 hours!")
            print("      This suggests:")
            print("      - A model may have recently run and overwritten probabilities")
            print("      - Or data was recently regenerated/updated")
            print("\n   🔍 RECOMMENDATION:")
            print("      - Check if you ran model training/inference recently")
            print("      - Verify the model used predict_proba(), not predict()")
            print("      - Check if probabilities were clipped at 1.0")
        elif age.days < 7:
            print("\n   ⚠️ File was modified within the last week")
            print("      Check if model training occurred recently")
        else:
            print("\n   ✅ File hasn't been modified recently")
            print("      Less likely that model predictions overwrote it")
            
        # Check file size
        file_size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"\n   File size: {file_size_mb:.2f} MB")
        
    else:
        print(f"\n❌ File not found: {data_file}")
    
    print("\n" + "-" * 80 + "\n")

def main():
    print("\n🔧 QUICK FIX CHECKER\n")
    
    # Ask user what they want to do
    print("Choose an option:")
    print("1. Clear Streamlit cache only")
    print("2. Check file modification date only")
    print("3. Both")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1" or choice == "3":
        clear_streamlit_cache()
    
    if choice == "2" or choice == "3":
        check_file_modification_date()
    
    if choice == "3":
        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("\n1. Restart Streamlit:")
        print("   streamlit run dashboard/alternate_dashboard.py")
        print("\n2. Check the dashboard - look for:")
        print("   - Sidebar warnings about 1.0 values")
        print("   - Diagnostic panel in Top 10 Prospects section")
        print("\n3. If 1.0 values still appear:")
        print("   - The issue is in the source data file")
        print("   - Check if model inference recently ran")
        print("   - Review the files identified in trace_probability_source.py")
        print("\n" + "=" * 80 + "\n")
    
    if choice == "4":
        print("Exiting...")
        return
    
    if choice not in ["1", "2", "3", "4"]:
        print("Invalid choice. Running both checks by default...\n")
        clear_streamlit_cache()
        check_file_modification_date()

if __name__ == "__main__":
    main()

