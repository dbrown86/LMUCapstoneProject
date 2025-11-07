#!/usr/bin/env python3
"""
Dependency Installation Script for Enhanced Pipeline
Handles missing packages gracefully and provides fallback options
"""

import subprocess
import sys
import importlib

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package_name, import_name=None):
    """Check if a package is available"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def main():
    """Install dependencies with fallback options"""
    print("=" * 60)
    print("ENHANCED PIPELINE DEPENDENCY INSTALLER")
    print("=" * 60)
    
    # Core packages (required)
    core_packages = [
        ("torch", "torch"),
        ("scikit-learn", "sklearn"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
        ("joblib", "joblib")
    ]
    
    # Optional packages
    optional_packages = [
        ("imbalanced-learn", "imblearn"),
        ("torch-geometric", "torch_geometric"),
        ("transformers", "transformers"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("plotly", "plotly"),
        ("optuna", "optuna"),
        ("shap", "shap")
    ]
    
    print("\n📦 Installing core packages...")
    core_success = []
    core_failed = []
    
    for package, import_name in core_packages:
        print(f"Checking {package}...", end=" ")
        if check_package(package, import_name):
            print("✅ Already installed")
            core_success.append(package)
        else:
            print("❌ Not found, installing...", end=" ")
            if install_package(package):
                print("✅ Installed successfully")
                core_success.append(package)
            else:
                print("❌ Failed to install")
                core_failed.append(package)
    
    print(f"\n📊 Core packages: {len(core_success)}/{len(core_packages)} installed")
    if core_failed:
        print(f"❌ Failed to install: {', '.join(core_failed)}")
    
    print("\n🔧 Installing optional packages...")
    optional_success = []
    optional_failed = []
    
    for package, import_name in optional_packages:
        print(f"Checking {package}...", end=" ")
        if check_package(package, import_name):
            print("✅ Already installed")
            optional_success.append(package)
        else:
            print("❌ Not found, installing...", end=" ")
            if install_package(package):
                print("✅ Installed successfully")
                optional_success.append(package)
            else:
                print("❌ Failed to install (optional)")
                optional_failed.append(package)
    
    print(f"\n📊 Optional packages: {len(optional_success)}/{len(optional_packages)} installed")
    
    # Summary
    print("\n" + "=" * 60)
    print("INSTALLATION SUMMARY")
    print("=" * 60)
    
    total_success = len(core_success) + len(optional_success)
    total_packages = len(core_packages) + len(optional_packages)
    
    print(f"Total packages installed: {total_success}/{total_packages}")
    print(f"Core packages: {len(core_success)}/{len(core_packages)}")
    print(f"Optional packages: {len(optional_success)}/{len(optional_packages)}")
    
    if len(core_success) == len(core_packages):
        print("\n✅ All core packages installed successfully!")
        print("The enhanced pipeline should work with full functionality.")
    else:
        print(f"\n⚠️  Some core packages failed to install: {', '.join(core_failed)}")
        print("The pipeline will work with limited functionality.")
    
    if optional_failed:
        print(f"\nℹ️  Optional packages that failed (not critical): {', '.join(optional_failed)}")
        print("These provide additional features but are not required.")
    
    # Test import
    print("\n🧪 Testing imports...")
    try:
        from src.class_imbalance_handler import AdvancedClassImbalanceHandler
        print("✅ Class imbalance handler imported successfully")
    except ImportError as e:
        print(f"❌ Class imbalance handler import failed: {e}")
    
    try:
        from src.enhanced_ensemble_model import AdvancedEnsembleModel
        print("✅ Enhanced ensemble model imported successfully")
    except ImportError as e:
        print(f"❌ Enhanced ensemble model import failed: {e}")
    
    try:
        from src.business_metrics_evaluator import DonorLegacyBusinessEvaluator
        print("✅ Business metrics evaluator imported successfully")
    except ImportError as e:
        print(f"❌ Business metrics evaluator import failed: {e}")
    
    print("\n🎉 Installation completed!")
    print("You can now run: python run_enhanced_pipeline.py")

if __name__ == "__main__":
    main()
