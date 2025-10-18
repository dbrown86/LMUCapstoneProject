# LMU CS Capstone Project - Organized Structure

## 📁 Directory Organization

```
LMUCapstoneProject/
├── 📁 config/                    # Configuration files and documentation
│   ├── README.md                 # Main project documentation
│   ├── requirements_enhanced.txt # Python dependencies
│   ├── TESTING_GUIDE.md         # Testing instructions
│   ├── TRAINING_PIPELINE_QUICKSTART.md
│   ├── TRAINING_RESULTS_SUMMARY.md
│   ├── MULTIMODAL_ARCHITECTURE_CHECKLIST.txt
│   └── PROJECT_EVALUATION_REPORT.txt
│
├── 📁 data/                      # All data files and embeddings
│   ├── synthetic_donor_dataset/  # Generated synthetic data
│   │   ├── donors.csv
│   │   ├── relationships.csv
│   │   ├── contact_reports.csv
│   │   ├── giving_history.csv
│   │   ├── enhanced_fields.csv
│   │   ├── challenging_test_cases.csv
│   │   └── dataset_analysis.png
│   ├── bert_embeddings_real.npy  # Pre-computed BERT embeddings
│   ├── gnn_embeddings_real.npy   # Pre-computed GNN embeddings
│   └── enhanced_pipeline_real_embeddings_results.pkl
│
├── 📁 models/                    # Trained models and checkpoints
│   ├── best_contact_classifier.pt
│   ├── best_donor_gnn_model.pt
│   └── donor_model_checkpoints/
│       ├── best_model.pt
│       └── training_summary.json
│
├── 📁 scripts/                   # All executable scripts
│   ├── main.py                   # Main entry point
│   ├── run_interpretability_pipeline.py
│   ├── run_improved_pipeline.py
│   ├── run_enhanced_pipeline.py
│   ├── run_final_optimized_pipeline.py
│   ├── run_with_real_embeddings.py
│   ├── run_donor_training_simple.py
│   ├── extract_real_embeddings.py
│   ├── create_capstone_visualizations.py
│   ├── install_dependencies.py
│   └── test_pipeline_setup.py
│
├── 📁 src/                       # Source code modules
│   ├── __init__.py
│   ├── bert_pipeline.py          # BERT text processing
│   ├── enhanced_ensemble_model.py # Main ensemble model
│   ├── enhanced_feature_engineering.py
│   ├── enhanced_multimodal_pipeline.py
│   ├── model_interpretability.py # SHAP, attention, etc.
│   ├── interpretability_integration.py
│   ├── multimodal_arch.py        # Multimodal architecture
│   ├── training_pipeline.py
│   ├── integrated_trainer.py
│   ├── business_metrics_evaluator.py
│   ├── class_imbalance_handler.py
│   ├── data_generation/          # Synthetic data generation
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── data_generation.py
│   │   ├── donor_generator.py
│   │   └── validation.py
│   └── gnn_models/               # Graph Neural Network models
│       ├── __init__.py
│       ├── gnn_models.py
│       ├── gnn_pipeline.py
│       ├── gnn_analysis.py
│       └── dataset_diagnostics.py
│
├── 📁 examples/                  # Example usage scripts
│   ├── basic_training_example.py
│   ├── donor_prediction_with_pipeline.py
│   └── interpretability_example.py
│
├── 📁 notebooks/                 # Jupyter notebooks
│   └── colab_multimodal_pipeline.ipynb
│
├── 📁 visualizations/            # Generated plots and charts
│   ├── dataset_predictability_analysis.png
│   ├── donor_training_curves.png
│   ├── multimodal_separation_analysis.png
│   └── multimodal_separation_results.csv
│
├── 📁 docs/                      # Detailed documentation
│   ├── INTERPRETABILITY_GUIDE.md
│   ├── TRAINING_PIPELINE_GUIDE.md
│   └── TRAINING_PIPELINE_README.md
│
├── 📁 results/                   # Experiment results (empty, ready for new results)
├── 📁 tests/                     # Unit tests (empty, ready for test files)
└── 📁 venv/                      # Python virtual environment
```

## 🚀 Quick Start Guide

### 1. Setup Environment
```bash
# Navigate to project directory
cd "C:\Desktop\LMU CS Capstone Project\LMUCapstoneProject"

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r config\requirements_enhanced.txt
```

### 2. Run Main Scripts
```bash
# Run the main interpretability pipeline
python scripts\run_interpretability_pipeline.py

# Run the improved pipeline with enhanced features
python scripts\run_improved_pipeline.py

# Extract embeddings (if needed)
python scripts\extract_real_embeddings.py
```

### 3. View Results
- **Models**: Check `models/` for trained model files
- **Visualizations**: Check `visualizations/` for generated plots
- **Results**: Check `results/` for experiment outputs
- **Data**: Check `data/` for datasets and embeddings

## 📋 Key Features by Directory

### `scripts/` - Main Execution Scripts
- **run_interpretability_pipeline.py**: Complete interpretability analysis
- **run_improved_pipeline.py**: Enhanced pipeline with better features
- **extract_real_embeddings.py**: Generate BERT and GNN embeddings

### `src/` - Core Implementation
- **enhanced_ensemble_model.py**: Main ensemble model with calibration
- **model_interpretability.py**: SHAP, attention, graph importance
- **bert_pipeline.py**: Text processing and BERT integration
- **gnn_models/**: Graph neural network implementations

### `data/` - All Data Files
- **synthetic_donor_dataset/**: Generated synthetic donor data
- **bert_embeddings_real.npy**: Pre-computed text embeddings
- **gnn_embeddings_real.npy**: Pre-computed graph embeddings

### `config/` - Configuration & Documentation
- **README.md**: Main project documentation
- **requirements_enhanced.txt**: Python dependencies
- **TESTING_GUIDE.md**: Testing instructions

## 🔧 Maintenance

### Adding New Scripts
- Place new executable scripts in `scripts/`
- Update this structure document

### Adding New Data
- Place datasets in `data/`
- Place embeddings in `data/`
- Update data loading paths in scripts

### Adding New Models
- Place trained models in `models/`
- Update model loading paths in scripts

### Adding New Visualizations
- Place generated plots in `visualizations/`
- Update visualization saving paths in scripts

## 📊 Current Status
- ✅ Project structure organized
- ✅ Files moved to appropriate directories
- ✅ Clear separation of concerns
- 🔄 Import paths may need updating in some scripts
- 🔄 Documentation updated to reflect new structure



