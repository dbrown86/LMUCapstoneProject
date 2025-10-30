# 🎯 Donor Prediction Dashboard

Interactive Streamlit web interface for the LMU Capstone donor prediction model.

## 🚀 Features

### 📊 Core Functionality
- **Donor Search & Filtering** - Advanced search with multiple criteria
- **Prediction Display** - Individual donor predictions with confidence scores
- **Model Explanations** - SHAP values, attention patterns, and feature importance
- **Analytics Dashboard** - Performance metrics and cohort analysis
- **Export Capabilities** - CSV, PDF, and JSON export options

### 🧠 Interpretability Features
- **SHAP Values** - Feature importance for individual predictions
- **BERT Attention** - Text analysis attention patterns
- **GNN Importance** - Graph network feature importance
- **Confidence Intervals** - Prediction uncertainty quantification
- **Feature Contribution** - Detailed breakdown of model decisions

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Existing model pipeline (run `scripts/advanced_multimodal_ensemble.py` first)

### Setup
```bash
# Install dashboard dependencies
pip install -r dashboard/requirements.txt

# Run the dashboard
streamlit run dashboard/app.py
```

## 📁 Project Structure

```
dashboard/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dashboard dependencies
├── README.md             # This file
├── pages/                # Dashboard pages
│   ├── overview.py       # Dashboard overview
│   ├── search.py         # Donor search & filtering
│   ├── predictions.py    # Prediction display
│   ├── explanations.py   # Model explanations
│   └── analytics.py      # Analytics & insights
├── components/           # Reusable components
│   ├── donor_card.py     # Individual donor display
│   ├── filters.py        # Search & filter widgets
│   ├── charts.py         # Visualization components
│   └── exports.py        # Export functionality
└── utils/                # Utility functions
    ├── data_loader.py    # Data loading utilities
    ├── model_integration.py # Model integration
    └── visualization.py  # Chart generation
```

## 🎯 Usage

### 1. Start the Dashboard
```bash
streamlit run dashboard/app.py
```

### 2. Navigate the Interface
- **🏠 Overview** - Key metrics and model status
- **🔍 Search** - Find and filter donors
- **📊 Predictions** - View model predictions
- **🧠 Explanations** - Understand model decisions
- **📈 Analytics** - Performance analysis

### 3. Key Features

#### Search & Filter
- Search by name, ID, email, or phone
- Filter by age, giving amount, legacy intent
- Sort by various criteria
- Export filtered results

#### Predictions
- View individual donor predictions
- See confidence scores and probabilities
- Filter by prediction type and confidence
- Export prediction data

#### Explanations
- SHAP values for feature importance
- BERT attention patterns for text
- GNN importance for network features
- Individual donor explanations

#### Analytics
- Model performance metrics
- Cohort analysis by age and giving
- Feature correlation analysis
- Business impact assessment

## 🔧 Configuration

### Data Sources
The dashboard automatically loads data from:
- `data/synthetic_donor_dataset/donors.csv`
- `data/synthetic_donor_dataset/contact_reports.csv`
- `data/bert_embeddings_real.npy`
- `data/gnn_embeddings_real.npy`

### Model Integration
The dashboard integrates with:
- `scripts/advanced_multimodal_ensemble.py`
- `fast_multimodal_results.pkl`

### Caching
- Data is cached using Streamlit's `@st.cache_data`
- Model is cached using `@st.cache_resource`
- Cache is automatically invalidated when data changes

## 📊 Performance

### Optimization Features
- **Lazy Loading** - Data loaded only when needed
- **Caching** - Results cached for faster access
- **Pagination** - Large datasets paginated for performance
- **Background Processing** - Heavy computations run asynchronously

### Expected Performance
- **App Load Time** - <5 seconds
- **Search Response** - <1 second
- **Prediction Display** - <2 seconds
- **Export Generation** - <10 seconds

## 🚀 Deployment

### Local Development
```bash
streamlit run dashboard/app.py --server.port 8501
```

### Production Deployment
```bash
# Using Streamlit Cloud
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0

# Using Docker
docker build -t donor-dashboard .
docker run -p 8501:8501 donor-dashboard
```

## 🔍 Troubleshooting

### Common Issues

#### Model Not Ready
- **Error**: "Model Not Ready" message
- **Solution**: Run `python scripts/advanced_multimodal_ensemble.py` first

#### Data Not Found
- **Error**: "No donor data available"
- **Solution**: Ensure data files exist in `data/synthetic_donor_dataset/`

#### Import Errors
- **Error**: Module not found
- **Solution**: Install requirements with `pip install -r dashboard/requirements.txt`

#### Performance Issues
- **Error**: Slow loading
- **Solution**: Check cache directory permissions and available memory

### Debug Mode
```bash
streamlit run dashboard/app.py --logger.level debug
```

## 📈 Future Enhancements

### Planned Features
- **Real-time Updates** - Live data synchronization
- **User Authentication** - Multi-user access control
- **Advanced Visualizations** - Interactive charts and graphs
- **API Integration** - REST API for external access
- **Mobile Support** - Responsive design for mobile devices

### Customization
- **Themes** - Custom color schemes and layouts
- **Widgets** - Additional visualization components
- **Integrations** - Connect to external data sources
- **Automation** - Scheduled reports and alerts

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions
- Include unit tests for new features

## 📄 License

This project is part of the LMU Capstone Project and follows the same licensing terms.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the project documentation
3. Create an issue in the repository
4. Contact the development team

---

**Built with ❤️ using Streamlit and the LMU Capstone Project**

