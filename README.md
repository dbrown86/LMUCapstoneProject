# LMU CS Capstone Project: Synthetic Donor Dataset with Dense Relationships

## 🎯 Project Overview

This project generates a comprehensive **500,000 donor synthetic dataset** with **1.8 million dense relationships** for advanced donor analytics and machine learning research. The dataset includes multiple relationship types, event attendance, giving history, and contact reports.

## 📊 Dataset Summary

- **Total Donors**: 500,000
- **Dense Relationships**: 1,805,144
- **Giving History Records**: 3,836,541
- **Event Attendance**: 156,065
- **Contact Reports**: 329,299
- **Family Relationships**: 91,592

### Relationship Types:
- **Professional**: 500,964 relationships
- **Alumni**: 400,840 relationships  
- **Geographic**: 300,170 relationships
- **Giving**: 300,000 relationships
- **Activity**: 200,000 relationships
- **Family**: 93,170 relationships
- **Social**: 10,000 relationships

## 🗂️ Project Structure

```
LMUCapstoneProject/
├── data/
│   └── synthetic_donor_dataset_500k_dense/    # Main 500K dataset
│       ├── donors.csv                         # Core donor data
│       ├── dense_relationships.csv            # All relationship types
│       ├── dense_relationships.parquet        # Optimized format
│       ├── donor_database.db                  # SQLite database
│       ├── family_relationships.csv           # Family connections
│       ├── event_attendance.csv               # Event participation
│       ├── giving_history.csv                 # Donation records
│       ├── contact_reports.csv                # Contact interactions
│       ├── enhanced_fields.csv                # ML features
│       └── parts/                             # Checkpoint files
├── scripts/
│   └── generate_enhanced_500k_dataset_with_dense_relationships.py  # Main generator
├── src/
│   ├── data_generation/                       # Data generation modules
│   ├── gnn_models/                           # Graph neural network models
│   └── *.py                                  # Core ML pipeline modules
├── dashboard/                                 # Streamlit web interface
├── docs/                                     # Documentation and analysis
├── examples/                                 # Usage examples
├── models/                                   # Trained model checkpoints
└── visualizations/                           # Generated plots and charts
```

## 🚀 Quick Start

### 1. Generate the Dataset
```bash
python scripts/generate_enhanced_500k_dataset_with_dense_relationships.py
```

### 2. Use the SQL Database (Recommended)
```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('data/synthetic_donor_dataset_500k_dense/donor_database.db')

# Query donors
donors = pd.read_sql_query("SELECT * FROM donors LIMIT 10", conn)

# Query relationships
relationships = pd.read_sql_query("""
    SELECT * FROM relationships 
    WHERE Relationship_Category = 'Alumni' 
    LIMIT 10
""", conn)
```

### 3. Use CSV Files
```python
import pandas as pd

# Load main datasets
donors = pd.read_csv('data/synthetic_donor_dataset_500k_dense/donors.csv')
relationships = pd.read_csv('data/synthetic_donor_dataset_500k_dense/dense_relationships.csv')
```

## 🔧 Technical Features

### Memory Optimization
- **Incremental disk writing** prevents OOM errors
- **Vectorized NumPy operations** for 10-100x speedup
- **Chunked processing** with hard caps on relationship counts
- **SQLite database** with 24 performance indexes

### Data Quality
- **Referential integrity** across all tables
- **Realistic relationship densities** (0.0014% graph density)
- **Comprehensive validation** and QA checks
- **Checkpoint system** for resumable generation

### Performance
- **Query speed**: Complex joins in ~3 seconds
- **Database size**: 936 MB (compressed)
- **Memory usage**: <2 GB during generation
- **Generation time**: ~10-15 minutes

## 📈 Use Cases

- **Donor Analytics**: Relationship network analysis
- **Machine Learning**: Graph neural networks, recommendation systems
- **Research**: Social network analysis, fundraising optimization
- **Development**: Testing ML pipelines with realistic data

## 🛠️ Requirements

- Python 3.8+
- pandas, numpy, sqlite3
- Optional: polars (for faster CSV processing)

## 📚 Documentation

- `docs/TRAINING_PIPELINE_GUIDE.md` - ML pipeline setup
- `docs/INTERPRETABILITY_GUIDE.md` - Model interpretability
- `docs/OPTIMIZATION_GUIDE.md` - Performance optimization
- `examples/` - Usage examples and tutorials

## 🎉 Success Metrics

✅ **Dataset Generation**: 500K donors with 1.8M relationships  
✅ **Memory Management**: No OOM errors with optimized processing  
✅ **Data Quality**: All integrity checks passed  
✅ **Performance**: Sub-3-second complex queries  
✅ **Documentation**: Comprehensive guides and examples  

## 📞 Support

For questions or issues, refer to the documentation in the `docs/` folder or check the example scripts in `examples/`.

---

**Generated**: 2025-01-22  
**Version**: 1.0  
**Status**: Production Ready ✅






