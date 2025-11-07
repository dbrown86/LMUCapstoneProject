# DonorAI Analytics Dashboard

## 🚀 Quick Start

To run the dashboard:

```bash
streamlit run dashboard/app.py
```

## 📁 Project Structure

The dashboard has been refactored into a modular architecture for better maintainability and scalability:

```
dashboard/
├── app.py                      # Main entry point (USE THIS)
├── alternate_dashboard.py.archived  # Legacy monolithic dashboard (archived)
│
├── config/                     # Configuration settings
│   ├── __init__.py
│   └── settings.py            # PAGE_CONFIG, data paths, column mappings
│
├── data/                       # Data loading and processing
│   ├── __init__.py
│   └── loader.py              # load_full_dataset()
│
├── models/                     # Model metrics and feature importance
│   ├── __init__.py
│   └── metrics.py             # get_model_metrics(), get_feature_importance()
│
├── components/                 # Reusable UI components
│   ├── __init__.py
│   ├── styles.py              # CSS styles
│   ├── charts.py              # Chart utilities
│   ├── sidebar.py             # Sidebar navigation and filters
│   └── metric_cards.py        # Metric card rendering
│
└── pages/                      # Page modules (one per dashboard page)
    ├── __init__.py
    ├── utils.py               # Shared page utilities
    ├── dashboard.py           # 🏠 Executive Dashboard
    ├── model_comparison.py    # 🔬 Model Comparison
    ├── business_impact.py     # 💰 Business Impact
    ├── donor_insights.py      # 💎 Donor Insights
    ├── features.py            # 🔬 Features
    ├── predictions.py         # 🎲 Predictions
    ├── performance.py         # 📈 Performance
    └── take_action.py         # ⚡ Take Action
```

## 🎯 Key Features

- **Modular Architecture**: Each page is a separate module for easy maintenance
- **Reusable Components**: Shared UI components reduce code duplication
- **Centralized Configuration**: All settings in one place
- **Clean Separation**: Data, models, UI, and business logic are separated
- **Easy Testing**: Each module can be tested independently

## 📊 Dashboard Pages

1. **🏠 Dashboard** - Executive summary with key metrics and visualizations
2. **🔬 Model Comparison** - Compare baseline vs. multimodal fusion model
3. **💰 Business Impact** - Revenue analysis and ROI calculations
4. **💎 Donor Insights** - Segment analysis and tactical recommendations
5. **🔬 Features** - Feature importance and distribution analysis
6. **🎲 Predictions** - Interactive prediction tool for donor profiles
7. **📈 Performance** - Model performance metrics (ROC, PR curves, confusion matrix)
8. **⚡ Take Action** - Prioritized outreach recommendations and export

## 🔧 Development

### Running Tests

```bash
# Test config module
python dashboard/config/test_settings.py

# Test data loader
python dashboard/data/test_loader.py

# Test components
python dashboard/components/test_components.py

# Test pages
python dashboard/pages/test_pages.py
```

### Adding a New Page

1. Create a new file in `dashboard/pages/` (e.g., `my_page.py`)
2. Define a `render(df, ...)` function
3. Import and add to `dashboard/pages/__init__.py`
4. Add routing in `dashboard/app.py`
5. Add navigation option in `dashboard/components/sidebar.py`

## 📝 Migration Notes

The dashboard was refactored from a monolithic `alternate_dashboard.py` (4500+ lines) into a modular architecture:

- **Before**: Single file with all logic, hard to maintain and test
- **After**: Modular structure with clear separation of concerns

The archived `alternate_dashboard.py.archived` is kept for reference but should not be used for development.

## 🐛 Troubleshooting

### Charts Not Rendering
- Ensure you're using `st.plotly_chart(fig)` without deprecated parameters
- Clear Streamlit cache: `streamlit cache clear`

### Import Errors
- Ensure project root is in Python path
- Check that all `__init__.py` files exist

### Data Not Loading
- Verify data paths in `dashboard/config/settings.py`
- Check that parquet files exist in `data/processed/parquet_export/`

## 📚 Related Documentation

- `REFACTORING_GUIDE.md` - Detailed refactoring process
- `MIGRATION_LOG.md` - Step-by-step migration log
- `CONFIG_EXTRACTION_SUMMARY.md` - Config module extraction details

## 🎉 Contributors

LMU Capstone Project Team - University Advancement Donor Prediction Dashboard

