#!/usr/bin/env python3
"""
Quick test to verify training pipeline is set up correctly
"""

import sys
import importlib

def test_imports():
    """Test if all modules can be imported"""
    print("=" * 80)
    print("TESTING TRAINING PIPELINE SETUP")
    print("=" * 80)
    
    modules_to_test = [
        ('Core Pipeline', 'src.training_pipeline'),
        ('Integrated Trainer', 'src.integrated_trainer'),
    ]
    
    components_to_test = [
        ('DataSplitter', 'src.training_pipeline', 'DataSplitter'),
        ('CrossValidationFramework', 'src.training_pipeline', 'CrossValidationFramework'),
        ('EarlyStoppingCallback', 'src.training_pipeline', 'EarlyStoppingCallback'),
        ('ModelCheckpointer', 'src.training_pipeline', 'ModelCheckpointer'),
        ('PerformanceMetrics', 'src.training_pipeline', 'PerformanceMetrics'),
        ('ClassBalancer', 'src.training_pipeline', 'ClassBalancer'),
        ('IntegratedTrainer', 'src.integrated_trainer', 'IntegratedTrainer'),
    ]
    
    print("\n1. Testing Module Imports...")
    print("-" * 80)
    
    all_passed = True
    
    for name, module_path in modules_to_test:
        try:
            importlib.import_module(module_path)
            print(f"  ✅ {name}: SUCCESS")
        except Exception as e:
            print(f"  ❌ {name}: FAILED - {e}")
            all_passed = False
    
    print("\n2. Testing Component Imports...")
    print("-" * 80)
    
    for name, module_path, component_name in components_to_test:
        try:
            module = importlib.import_module(module_path)
            getattr(module, component_name)
            print(f"  ✅ {name}: SUCCESS")
        except Exception as e:
            print(f"  ❌ {name}: FAILED - {e}")
            all_passed = False
    
    print("\n3. Testing Dependencies...")
    print("-" * 80)
    
    dependencies = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('torch', 'torch'),
        ('sklearn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('tqdm', 'tqdm'),
    ]
    
    optional_dependencies = [
        ('imbalanced-learn', 'imblearn'),
    ]
    
    for name, import_name in dependencies:
        try:
            importlib.import_module(import_name)
            print(f"  ✅ {name}: INSTALLED")
        except Exception as e:
            print(f"  ❌ {name}: MISSING - {e}")
            all_passed = False
    
    print("\n4. Testing Optional Dependencies...")
    print("-" * 80)
    
    for name, import_name in optional_dependencies:
        try:
            importlib.import_module(import_name)
            print(f"  ✅ {name}: INSTALLED (will use advanced balancing)")
        except Exception:
            print(f"  ⚠️  {name}: NOT INSTALLED (will use class weights)")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Pipeline is set up correctly!")
        print("=" * 80)
        return True
    else:
        print("❌ SOME TESTS FAILED - See errors above")
        print("=" * 80)
        return False


def quick_functionality_test():
    """Quick test of basic functionality"""
    print("\n" + "=" * 80)
    print("QUICK FUNCTIONALITY TEST")
    print("=" * 80)
    
    try:
        import numpy as np
        from sklearn.datasets import make_classification
        from src.training_pipeline import DataSplitter, PerformanceMetrics
        
        print("\n1. Testing DataSplitter...")
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, 
                                   weights=[0.7, 0.3], random_state=42)
        
        splitter = DataSplitter(test_size=0.2, val_size=0.2, random_state=42)
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data(X, y, stratify=True)
        
        print(f"  ✅ Data split successfully")
        print(f"     Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
        
        print("\n2. Testing PerformanceMetrics...")
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics_calc = PerformanceMetrics()
        metrics = metrics_calc.calculate_metrics(y_test, y_pred, y_pred_proba, prefix='test_')
        
        print(f"  ✅ Metrics calculated successfully")
        print(f"     Accuracy: {metrics['test_accuracy']:.3f}")
        print(f"     AUC: {metrics['test_roc_auc']:.3f}")
        print(f"     F1: {metrics['test_f1']:.3f}")
        
        print("\n✅ FUNCTIONALITY TEST PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ FUNCTIONALITY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_trainer():
    """Test the IntegratedTrainer with a simple model"""
    print("\n" + "=" * 80)
    print("INTEGRATED TRAINER TEST")
    print("=" * 80)
    
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from sklearn.datasets import make_classification
        from src.integrated_trainer import IntegratedTrainer
        
        print("\n1. Creating test data...")
        X, y = make_classification(n_samples=500, n_features=10, n_classes=2,
                                   weights=[0.7, 0.3], random_state=42)
        print(f"  ✅ Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        print("\n2. Defining simple PyTorch model...")
        class SimpleModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 16)
                self.fc2 = nn.Linear(16, 2)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        model = SimpleModel(input_dim=X.shape[1])
        print(f"  ✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        print("\n3. Training with IntegratedTrainer...")
        trainer = IntegratedTrainer(model, model_type='pytorch', device='cpu', random_state=42)
        
        results = trainer.fit(
            X, y,
            test_size=0.2,
            val_size=0.2,
            stratify=True,
            balance_strategy='smote',
            epochs=5,  # Very short for testing
            batch_size=32,
            patience=3,
            checkpoint_dir='test_checkpoints',
            verbose=0  # Suppress output for test
        )
        
        print(f"  ✅ Training completed successfully")
        print(f"     Test Accuracy: {results['test_results']['test_accuracy']:.3f}")
        print(f"     Test AUC: {results['test_results']['test_roc_auc']:.3f}")
        print(f"     Epochs trained: {results['training_results']['total_epochs']}")
        
        print("\n4. Testing predictions...")
        predictions, probabilities = trainer.predict(X[:10])
        print(f"  ✅ Predictions working")
        print(f"     Sample predictions: {predictions[:5]}")
        
        print("\n✅ INTEGRATED TRAINER TEST PASSED")
        
        # Cleanup
        import shutil
        import os
        if os.path.exists('test_checkpoints'):
            shutil.rmtree('test_checkpoints')
            print("  🧹 Cleaned up test checkpoints")
        
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATED TRAINER TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "🧪" * 40)
    print("TRAINING PIPELINE TEST SUITE")
    print("🧪" * 40)
    
    results = []
    
    # Test 1: Imports
    print("\n" + "=" * 80)
    print("TEST 1: IMPORTS AND DEPENDENCIES")
    print("=" * 80)
    results.append(("Imports", test_imports()))
    
    # Test 2: Basic functionality
    if results[0][1]:  # Only run if imports passed
        results.append(("Functionality", quick_functionality_test()))
    
    # Test 3: Integrated trainer
    if all(r[1] for r in results):  # Only run if all previous tests passed
        results.append(("Integrated Trainer", test_integrated_trainer()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Training pipeline is ready to use!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Run: python examples/basic_training_example.py")
        print("  2. Run: python examples/donor_prediction_with_pipeline.py")
        print("  3. Check documentation in: docs/TRAINING_PIPELINE_README.md")
    else:
        print("⚠️  SOME TESTS FAILED - Please fix the issues above")
        print("=" * 80)
        print("\nTroubleshooting:")
        print("  1. Install missing dependencies: pip install -r requirements_enhanced.txt")
        print("  2. Check Python version: Python 3.8+ required")
        print("  3. Verify file paths are correct")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)





