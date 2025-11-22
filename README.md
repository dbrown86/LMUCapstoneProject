# LMU CS Capstone Project: Synthetic Donor Dataset & Multimodal Fusion Model for Fundraising Analytics

## 🎯 Project Overview

Fundraising and philanthropy research suffers from a scarcity of **open-source datasets**—privacy concerns and donor confidentiality make real-world data almost impossible to share. This project addresses that gap by generating a **synthetic donor dataset** (500K records, 1.8M dense relationships) expressly designed for experimenting with **machine learning (ML)** and **deep learning (DL)** approaches in advancement analytics.

The dataset includes contact reports, gift histories, event attendance, household linkages, and graph-based influence signals. On top of the data layer, we implement a **novel multimodal fusion model** that unifies tabular, sequential, network, and text-derived features through a deep learning pipeline (MLP + LSTM + attention + fusion layers). The result is an end-to-end DL pipeline for exploring how modern AI techniques can elevate prospect prioritization, revenue forecasting, and gift officer workflows.

### Key Innovations
- **Open source synthetic fundraising dataset** with rich, interleaved donor behaviors.
- **Multimodal fusion architecture** combining tabular encoders, sequence models, graph/network embeddings, and text aggregates with cross-modal attention.
- **Temporal validation** (1980–2025) to mimic real deployment scenarios for 2025 predictions.
- **Streamlit dashboard** that surfaces KPIs, feature insights, and business impact for practitioner review.

## 📊 Dataset Snapshot

- **Total Donors**: 500,000  
- **Dense Relationships**: 1,805,144  
- **Giving History Records**: 3,836,541  
- **Event Attendance**: 156,065  
- **Contact Reports**: 329,299  
- **Relationship Mix**: Professional (500K), Alumni (401K), Geographic (300K), Giving (300K), Activity (200K), Family (93K), Social (10K)

## 🗂️ Project Structure

```
LMUCapstoneProject/
├── data/
│   └── synthetic_donor_dataset_500k_dense/      # Core synthetic corpus
│       ├── donors.csv / donor_database.db       # Tabular + SQL formats
│       ├── dense_relationships.(csv|parquet)    # Network edges
│       ├── giving_history.csv                   # Longitudinal gifts
│       └── parts/                               # Generation checkpoints
├── dashboard/                                   # Streamlit app + assets
│   ├── app.py                                   # Main entry point
│   ├── architecture_diagram.html                # Visual overview (used in README)
│   └── pages/ / components/ / models/           # Modular UI + metrics
├── docs/                                        # Deep dives + guides
├── scripts/                                     # Dataset generation utilities
├── src/                                         # ML/DL training pipelines
├── models/                                      # Saved checkpoints
└── visualizations/                              # Plots, analyses, figures
```

## 🧠 Architecture Diagram

The full multimodal training flow (data generation → feature engineering → fusion model → evaluation → dashboard deployment) is captured in `dashboard/architecture_diagram.html`. Open that file in a browser to view the interactive layered diagram referenced throughout the documentation.

## 🖥️ Streamlit Dashboard 

To access the Streamlit dashboard: https://fictitiousuniversity.streamlit.app/ 

## 🚀 Quick Start

### 1. Generate / refresh the dataset
```bash
python scripts/generate_enhanced_500k_dataset_with_dense_relationships.py
```

### 2. Explore via SQLite (recommended)
```python
import sqlite3, pandas as pd
conn = sqlite3.connect('data/synthetic_donor_dataset_500k_dense/donor_database.db')
donors = pd.read_sql_query("SELECT * FROM donors LIMIT 10", conn)
relationships = pd.read_sql_query("""
    SELECT * FROM relationships
    WHERE Relationship_Category = 'Alumni'
    LIMIT 10
""", conn)
```

### 3. Work with CSV/Parquet extracts
```python
import pandas as pd
donors = pd.read_csv('data/synthetic_donor_dataset_500k_dense/donors.csv')
relationships = pd.read_csv('data/synthetic_donor_dataset_500k_dense/dense_relationships.csv')
```

### 4. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

## 🔧 Technical Highlights

**Memory + Performance**
- Incremental disk writes and chunked processing prevent OOM on standard laptops.
- Vectorized NumPy/Pandas flow yields 10–100× faster generation.
- SQLite backend ships with 24 tuned indexes → complex network joins in ~3 seconds.

**Data Fidelity**
- Referential integrity enforced across donors, gifts, relationships, and events.
- Graph density calibrated (≈0.0014%) to mirror enterprise advancement CRMs.
- QA pipeline plus resumable checkpoints for long-running jobs.

**Modeling Approach**
- Feature store covers RFM, engagement streaks, network centrality, capacity indicators, and synthetic contact report stats.
- Multimodal pipeline (`src/models/train_will_give_again.py`) fuses 60+ engineered features with sequential gift histories, network embeddings, and SVD-based text signals.
- Training splits: **1980–2023** (train), **2024** (validation), **2025** (test target) with AdamW + BCE-with-logits + ReduceLROnPlateau + batch size 2048 + hidden dim 256.

## 📚 Documentation & Support

- `docs/TRAINING_PIPELINE_GUIDE.md` – Training + evaluation steps  
- `docs/INTERPRETABILITY_GUIDE.md` – Explaining model decisions  
- `docs/OPTIMIZATION_GUIDE.md` – Performance + scaling tips  
- `examples/` – Quick notebooks and scripts to jump-start analysis

For questions, explore the `docs/` folder or open an issue/PR with reproducible steps. This repository is intentionally open so other institutions can build upon the synthetic dataset and multimodal modeling blueprint.

---

## 👤 Credits

**Created by**  
Danielle Brown  
Loyola Marymount University  
M.S. in Computer Science Senior Capstone Project  
2025






