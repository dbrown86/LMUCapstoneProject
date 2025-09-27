# Synthetic Donor Dataset Generation and GNN Analysis

This project generates a comprehensive synthetic donor dataset and provides Graph Neural Network (GNN) analysis capabilities for donor relationship modeling and legacy intent prediction.

## Project Structure

```
LMUCapstoneProject/
├── main.py                          # Main entry point
├── src/
│   ├── data_generation/             # Data generation modules
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration settings
│   │   ├── data_generation.py      # Core generator classes
│   │   ├── donor_generator.py      # Main generation functions
│   │   └── validation.py           # Data validation and visualization
│   └── gnn_models/                 # Graph Neural Network modules
│       ├── __init__.py
│       ├── gnn_models.py           # GNN model definitions
│       ├── gnn_analysis.py         # Analysis and visualization functions
│       └── gnn_pipeline.py         # Main GNN pipeline
├── synthetic_donor_dataset/         # Generated dataset (created after running)
└── requirements.txt
```

## Features

### Data Generation
- **50,000 synthetic donor records** with realistic demographics
- **Family relationship modeling** (30% of donors in families)
- **Comprehensive giving history** with individual gift records
- **Contact reports** with realistic outcomes
- **Enhanced fields** for deep learning applications
- **Data quality validation** and comprehensive reporting

### Graph Neural Network Analysis
- **GraphSAGE and GCN models** for donor classification
- **Family network analysis** and relationship modeling
- **Legacy intent prediction** using graph structure
- **Feature importance analysis** and model interpretability
- **Node embedding visualization** using t-SNE
- **Hyperparameter optimization** support

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python main.py
```

This will:
- Generate 50,000 donor records with family relationships
- Create detailed giving history and contact reports
- Validate data quality and create visualizations
- Export all datasets to CSV files
- Optionally run GNN analysis

### 3. Run GNN Analysis Separately

```python
from src.gnn_models import main_gnn_pipeline
import pandas as pd

# Load generated datasets
donors_df = pd.read_csv('synthetic_donor_dataset/donors.csv')
relationships_df = pd.read_csv('synthetic_donor_dataset/relationships.csv')
contact_reports_df = pd.read_csv('synthetic_donor_dataset/contact_reports.csv')
giving_history_df = pd.read_csv('synthetic_donor_dataset/giving_history.csv')

# Run GNN analysis
results = main_gnn_pipeline(donors_df, relationships_df, contact_reports_df, giving_history_df)
```

## Generated Datasets

The system creates the following CSV files in `synthetic_donor_dataset/`:

1. **donors.csv** - Main donor records (50,000 rows)
2. **relationships.csv** - Family relationship mappings (15,000 rows)
3. **contact_reports.csv** - Contact interaction records (32,947 rows)
4. **giving_history.csv** - Individual gift transactions (489,345 rows)
5. **enhanced_fields.csv** - Deep learning features (50,000 rows)
6. **dataset_analysis.png** - Comprehensive visualizations

## Configuration

Key parameters can be modified in `src/data_generation/config.py`:

- `TOTAL_DONORS`: Number of donor records to generate
- `FAMILY_PERCENTAGE`: Percentage of donors in families
- `NON_DONOR_PERCENTAGE`: Percentage of non-donors
- `RATING_DISTRIBUTION`: Wealth capacity rating distribution

## GNN Models

The system implements two GNN architectures:

1. **GraphSAGE** - Inductive learning with neighborhood sampling
2. **GCN** - Graph Convolutional Networks for node classification

Both models are trained for binary classification (legacy intent prediction) and include:
- Multi-layer architecture with batch normalization
- Dropout for regularization
- Early stopping to prevent overfitting
- Comprehensive evaluation metrics

## Analysis Features

### Family Network Analysis
- Family size distribution analysis
- Giving patterns within families
- Legacy intent correlation analysis
- Embedding similarity within family groups

### Model Interpretability
- Feature importance analysis using gradients
- Node-level prediction explanations
- Attention weight visualization
- t-SNE embedding plots

### Performance Metrics
- Accuracy, AUC, and F1 scores
- Training/validation curve analysis
- Cross-validation results
- Hyperparameter optimization

## Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.0+
- scikit-learn 1.0+
- pandas 1.3+
- numpy 1.21+
- matplotlib 3.5+
- seaborn 0.11+
- tqdm 4.62+

## Usage Examples

### Basic Data Generation
```python
from src.data_generation import generate_core_donors_with_families
import numpy as np

# Generate donors
rng = np.random.default_rng(42)
donor_ids = generate_random_ids(1000)
donors_df, relationships_df = generate_core_donors_with_families(1000, donor_ids, rng)
```

### GNN Training
```python
from src.gnn_models import DonorGraphPreprocessor, GraphSAGE, DonorGNNTrainer

# Create graph data
preprocessor = DonorGraphPreprocessor(donors_df, relationships_df)
graph_data = preprocessor.create_graph_data()

# Train model
model = GraphSAGE(input_dim=graph_data.x.shape[1], hidden_dim=64, output_dim=2)
trainer = DonorGNNTrainer(model, device='cpu')
results = trainer.train(graph_data)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the LMU CS Capstone Project and is intended for educational and research purposes.