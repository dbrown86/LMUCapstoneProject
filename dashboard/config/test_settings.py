"""
Test script for dashboard config module.
Run this to verify config extraction works correctly.
"""

from dashboard.config import settings

def test_settings():
    """Test that all settings are accessible."""
    print("=" * 60)
    print("Testing Dashboard Configuration Module")
    print("=" * 60)
    print()
    
    # Test PAGE_CONFIG
    print("📄 PAGE_CONFIG:")
    print(f"   Page Title: {settings.PAGE_CONFIG['page_title']}")
    print(f"   Page Icon: {settings.PAGE_CONFIG['page_icon']}")
    print(f"   Layout: {settings.PAGE_CONFIG['layout']}")
    assert len(settings.PAGE_CONFIG) > 0
    print("   ✅ PAGE_CONFIG is valid")
    print()
    
    # Test data paths
    print("📁 DATA PATHS:")
    data_paths = settings.get_data_paths()
    print(f"   Parquet paths: {len(data_paths['parquet_paths'])} candidates")
    print(f"   SQLite paths: {len(data_paths['sqlite_paths'])} candidates")
    print(f"   CSV directories: {len(data_paths['csv_dir_candidates'])} candidates")
    print(f"   Giving history paths: {len(data_paths['giving_paths'])} candidates")
    assert len(data_paths['parquet_paths']) > 0
    assert len(data_paths['sqlite_paths']) > 0
    print("   ✅ Data paths are configured")
    print()
    
    # Test saved metrics
    print("📊 SAVED METRICS:")
    print(f"   Metrics candidates: {len(settings.SAVED_METRICS_CANDIDATES)}")
    print(f"   Use saved metrics only: {settings.USE_SAVED_METRICS_ONLY}")
    assert len(settings.SAVED_METRICS_CANDIDATES) > 0
    print("   ✅ Saved metrics config is valid")
    print()
    
    # Test column mappings
    print("🔗 COLUMN MAPPINGS:")
    print(f"   Column mapping keys: {len(settings.COLUMN_MAPPING)}")
    print(f"   Probability variants: {len(settings.PROBABILITY_COLUMN_VARIANTS)}")
    print(f"   Outcome variants: {len(settings.OUTCOME_COLUMN_VARIANTS)}")
    assert len(settings.COLUMN_MAPPING) > 0
    assert len(settings.PROBABILITY_COLUMN_VARIANTS) > 0
    assert len(settings.OUTCOME_COLUMN_VARIANTS) > 0
    print("   ✅ Column mappings are configured")
    print()
    
    # Test project root
    print("📂 PROJECT ROOT:")
    root = settings.get_project_root()
    print(f"   Root path: {root}")
    assert root.exists()
    print("   ✅ Project root is valid")
    print()
    
    print("=" * 60)
    print("✅ All configuration tests passed!")
    print("=" * 60)
    print()
    print("Next step: Extract data loader module")
    print()

if __name__ == "__main__":
    test_settings()

