# Project Refactoring Guide

## Overview
This document describes the project restructuring completed on November 5, 2025 to align with industry-standard Python project organization.

## New Directory Structure

```
LMUCapstoneProject/
├── data/                        # All data files
│   ├── raw/                    # Original, immutable data
│   ├── processed/              # Cleaned, transformed data
│   │   └── parquet_export/    # Processed parquet files
│   ├── interim/                # Intermediate transformations
│   └── external/               # External reference data
│
├── models/                      # Model artifacts
│   ├── saved_models/           # Trained model files (.pt, .h5, .pkl)
│   │   └── best_influential_donor_model.pt
│   ├── checkpoints/            # Training checkpoints
│   │   └── donor_model_checkpoints/
│   └── exports/                # Exported models (ONNX, TFLite, etc.)
│
├── src/                         # Source code (Python package)
│   ├── __init__.py
│   ├── data/                   # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── export_sql_to_parquet.py
│   │   └── generate_enhanced_dataset.py
│   ├── models/                 # Model training and inference
│   │   ├── __init__.py
│   │   ├── train_will_give_again.py  # Main training script
│   │   └── generate_predictions.py   # Inference script
│   ├── features/               # Feature engineering
│   │   ├── __init__.py
│   │   └── extract_network_features.py
│   ├── evaluation/             # Model evaluation utilities
│   │   └── __init__.py
│   └── utils/                  # Helper functions
│       ├── __init__.py
│       └── generate_visualizations.py
│
├── dashboard/                   # Streamlit dashboard
│   ├── alternate_dashboard.py  # Main dashboard (DO NOT BREAK!)
│   ├── components/             # Reusable UI components
│   ├── assets/                 # CSS, images, etc.
│   ├── pages/                  # Multi-page dashboard sections
│   └── page_modules/           # Modular page components
│
├── notebooks/                   # Jupyter notebooks
│   ├── exploratory/            # EDA and experiments
│   └── reports/                # Analysis notebooks
│
├── tests/                       # Unit and integration tests
│   └── test_environment_setup.py
│
├── configs/                     # Configuration files
│   └── config/                 # Configuration documentation
│
├── scripts/                     # Utility scripts
│   ├── setup_environment.sh
│   ├── train.sh
│   └── [other model scripts]
│
├── docs/                        # Documentation
├── examples/                    # Example usage
├── results/                     # Training results and plots
└── requirements.txt             # Python dependencies
```

## Key Changes

### Data Organization
- **OLD**: `data/parquet_export/`
- **NEW**: `data/processed/parquet_export/`
- **Backwards Compatibility**: Both paths are supported with fallback logic

### Model Storage
- **OLD**: `models/best_influential_donor_model.pt`
- **NEW**: `models/saved_models/best_influential_donor_model.pt`
- **OLD**: `models/donor_model_checkpoints/`
- **NEW**: `models/checkpoints/donor_model_checkpoints/`

### Source Code
- **Training Script**: 
  - **OLD**: `multi_target_production/src/simplified_single_target_training.py`
  - **NEW**: `src/models/train_will_give_again.py`
- **Prediction Script**:
  - **OLD**: `final_model/src/generate_will_give_again_predictions.py`
  - **NEW**: `src/models/generate_predictions.py`
- **Data Loader**:
  - **OLD**: `dashboard/utils/data_loader.py`
  - **NEW**: `src/data/data_loader.py`

### Configuration
- **OLD**: `config/`
- **NEW**: `configs/`

## Critical Files (DO NOT BREAK!)

1. **dashboard/alternate_dashboard.py** - Main dashboard interface
2. **src/models/train_will_give_again.py** - Primary training script
3. **src/models/generate_predictions.py** - Inference pipeline

All path references in these files have been updated with backwards compatibility fallbacks.

## Migration Strategy

### For Existing Data
If you have existing data in the old structure:
```powershell
# PowerShell commands to migrate data
Copy-Item -Path "data\parquet_export" -Destination "data\processed\parquet_export" -Recurse
```

### For Existing Models
Models in the old location will continue to work due to fallback logic, but new models should be saved to:
- `models/saved_models/` for final trained models
- `models/checkpoints/` for training checkpoints

## Benefits

1. **Industry Standard**: Follows Python packaging best practices
2. **Clear Separation**: Data, code, models, and configs are clearly separated
3. **Scalability**: Easy to add new components (tests, docs, notebooks)
4. **Git History Preserved**: All moves done with `git mv` to preserve history
5. **Backwards Compatible**: Legacy paths still work as fallbacks

## Testing

After refactoring, verify:
1. Dashboard loads correctly: `streamlit run dashboard/alternate_dashboard.py`
2. Training script runs: `python src/models/train_will_give_again.py`
3. Predictions work: `python src/models/generate_predictions.py`

## Git History

All refactoring commits are in the `refactor-project-structure` branch:
- Backup branch: `backup-before-refactor` (safe restoration point)
- Refactoring branch: `refactor-project-structure` (new structure)

## Rollback

If issues occur, switch back to the backup:
```bash
git checkout backup-before-refactor
```

## Questions?

See `PROJECT_STRUCTURE.md` for the original structure documentation.

