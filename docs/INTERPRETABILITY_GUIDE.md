# Model Interpretability Guide

## Overview

This guide provides comprehensive documentation for the model interpretability features implemented in the LMU CS Capstone Project. The interpretability system supports all three modalities in our multimodal architecture: tabular features, text (BERT), and graph (GNN) data.

## Table of Contents

1. [Features Overview](#features-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Tabular Interpretability](#tabular-interpretability)
4. [Text Interpretability](#text-interpretability)
5. [Graph Interpretability](#graph-interpretability)
6. [Confidence Intervals](#confidence-intervals)
7. [Feature Contribution Breakdown](#feature-contribution-breakdown)
8. [Report Generation](#report-generation)
9. [Integration Examples](#integration-examples)
10. [Best Practices](#best-practices)

## Features Overview

### 🎯 **Core Capabilities**

- **SHAP Integration**: Comprehensive SHAP values for tabular features
- **Attention Visualization**: BERT attention heatmaps for text analysis
- **Graph Importance**: Node-level importance scoring for GNN models
- **Confidence Intervals**: Bootstrap and binomial confidence interval calculation
- **Feature Breakdown**: Detailed contribution analysis across modalities
- **Interactive Reports**: HTML reports with Plotly visualizations

### 🏗️ **Architecture**

```
MultimodalModelInterpreter
├── Tabular SHAP Analysis
├── Text Attention Analysis
├── Graph Importance Analysis
├── Confidence Interval Calculation
├── Feature Contribution Breakdown
└── Interactive Report Generation
```

## Installation and Setup

### Prerequisites

```bash
pip install -r requirements_enhanced.txt
```

### Key Dependencies

- `shap>=0.42.0` - SHAP values for tabular features
- `plotly>=5.15.0` - Interactive visualizations
- `networkx>=3.1.0` - Graph visualization
- `captum>=0.6.0` - PyTorch model interpretability
- `transformers>=4.30.0` - BERT attention analysis

## Tabular Interpretability

### SHAP Values for Tabular Features

```python
from src.model_interpretability import MultimodalModelInterpreter

# Initialize interpreter
interpreter = MultimodalModelInterpreter(
    ensemble_model=your_ensemble_model,
    feature_names=feature_names
)

# Compute SHAP values
shap_values, indices = interpreter.compute_tabular_shap_values(
    X_tabular, sample_size=100
)

# Create summary plot
interpreter.plot_shap_summary(X_tabular[indices], max_display=20)
```

### Feature Importance Analysis

```python
# Analyze feature importance across models
importance_scores = interpreter.analyze_feature_importance(X_test, y_test)

# Get top features for specific donor
top_features = interpreter.get_top_features_for_donor(
    donor_idx=0, 
    X_test=donor_features, 
    top_n=10
)
```

## Text Interpretability

### BERT Attention Visualization

```python
from transformers import AutoTokenizer, AutoModel

# Setup BERT components
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased', output_attentions=True)

# Create attention heatmap
attention_weights, tokens = interpreter.create_attention_heatmap(
    text="Sample contact report text...",
    tokenizer=tokenizer,
    model=model,
    layer_idx=-1  # Last layer
)
```

### Attention Pattern Analysis

The attention heatmap shows:
- **Query-Key Relationships**: Which tokens attend to which other tokens
- **Attention Strength**: Intensity of attention weights
- **Important Tokens**: Most attended tokens in the sequence

## Graph Interpretability

### Graph Importance Scoring

```python
# Compute graph-level importance scores
graph_importance = interpreter.compute_graph_importance_scores(
    graph_data=your_graph_data,
    model=your_gnn_model,
    node_idx=None  # Analyze all nodes
)

# Visualize graph importance
interpreter.visualize_graph_importance(
    graph_data, 
    graph_importance['node_importance'],
    top_k=20
)
```

### Node Importance Analysis

The graph importance analysis provides:
- **Node Importance Scores**: Gradient-based importance for each node
- **Network Visualization**: Highlighted important nodes in the graph
- **Relationship Analysis**: Understanding of influential connections

## Confidence Intervals

### Bootstrap Method

```python
# Compute bootstrap confidence intervals
ci_bootstrap = interpreter.compute_confidence_intervals(
    y_pred_proba=predictions,
    method='bootstrap',
    n_bootstrap=1000,
    confidence_level=0.95
)
```

### Binomial Method (Wilson Score Interval)

```python
# Compute binomial confidence intervals
ci_binomial = interpreter.compute_confidence_intervals(
    y_pred_proba=predictions,
    method='binomial',
    confidence_level=0.95
)
```

### Confidence Level Classification

- **High Confidence**: Probability > 0.8 or < 0.2
- **Medium Confidence**: Probability 0.4-0.6 or 0.6-0.8
- **Low Confidence**: Probability 0.2-0.4

## Feature Contribution Breakdown

### Comprehensive Analysis

```python
# Create feature contribution breakdown
breakdown = interpreter.create_feature_contribution_breakdown(
    donor_features=donor_feature_vector,
    prediction_proba=0.75,
    tabular_features=tabular_feature_names,
    text_features=text_feature_names,  # Optional
    graph_features=graph_feature_names  # Optional
)
```

### Breakdown Structure

```python
{
    'prediction_probability': 0.75,
    'confidence_level': 'High',
    'modality_contributions': {
        'tabular': {...},
        'text': {...},
        'graph': {...}
    },
    'top_features': [...]
}
```

## Report Generation

### Interactive HTML Reports

```python
# Generate comprehensive HTML report
fig = interpreter.generate_interpretability_report(
    donor_id=12345,
    breakdown=breakdown,
    save_path='donor_12345_interpretability_report.html'
)
```

### Report Components

1. **Prediction Overview**: Gauge chart with probability and confidence
2. **Modality Contributions**: Bar chart showing feature counts per modality
3. **Top Features**: Horizontal bar chart of most important features
4. **Confidence Analysis**: Scatter plot showing confidence levels

## Integration Examples

### Complete Pipeline Integration

```python
from src.interpretability_integration import InterpretabilityPipeline

# Create interpretability pipeline
pipeline = InterpretabilityPipeline(ensemble_model=your_ensemble_model)

# Setup components
pipeline.setup_bert_components()
pipeline.setup_gnn_components(graph_data, gnn_model)

# Run comprehensive analysis
results = pipeline.run_comprehensive_interpretability(
    donors_df, contact_reports_df,
    bert_embeddings=bert_embeddings,
    gnn_embeddings=gnn_embeddings,
    sample_donor_ids=[12345, 67890, 11111],
    save_reports=True
)
```

### Individual Donor Analysis

```python
# Analyze specific donor
pipeline._analyze_individual_donor(
    donor_id=12345,
    donors_df=donors_df,
    contact_reports_df=contact_reports_df,
    tabular_features=tabular_features,
    text_embeddings=text_embeddings,
    graph_embeddings=graph_embeddings,
    save_reports=True
)
```

## Best Practices

### 1. **Performance Optimization**

- Use sampling for SHAP computation on large datasets
- Cache attention weights for repeated text analysis
- Pre-compute graph importance for static networks

### 2. **Visualization Guidelines**

- Limit SHAP summary plots to top 20 features
- Use consistent color schemes across modalities
- Include confidence intervals in all probability visualizations

### 3. **Report Generation**

- Generate reports for high-value donors first
- Include business context in report titles
- Save reports with descriptive filenames

### 4. **Error Handling**

```python
try:
    # Interpretability analysis
    results = interpreter.compute_tabular_shap_values(X_tabular)
except Exception as e:
    print(f"Interpretability analysis failed: {e}")
    # Fallback to basic feature importance
    results = interpreter.analyze_feature_importance(X_tabular, y)
```

### 5. **Memory Management**

- Process large datasets in batches
- Clear intermediate results when not needed
- Use appropriate data types (float32 vs float64)

## Troubleshooting

### Common Issues

1. **SHAP Memory Issues**
   - Reduce `sample_size` parameter
   - Use `shap.TreeExplainer` for tree-based models
   - Process data in smaller batches

2. **BERT Attention Errors**
   - Ensure model has `output_attentions=True`
   - Check tokenizer compatibility
   - Verify input text length limits

3. **Graph Visualization Problems**
   - Install `networkx` and `matplotlib`
   - Check graph data format
   - Reduce `top_k` parameter for large graphs

### Performance Tips

- Use GPU acceleration for BERT models when available
- Cache SHAP explainers for repeated use
- Pre-compute embeddings for static text data

## Advanced Usage

### Custom Interpretability Methods

```python
class CustomInterpreter(MultimodalModelInterpreter):
    def custom_analysis_method(self, data):
        # Implement custom interpretability logic
        pass
```

### Integration with Business Metrics

```python
# Combine interpretability with business metrics
from src.business_metrics_evaluator import create_business_metrics_evaluator

business_evaluator = create_business_metrics_evaluator()
roi_metrics = business_evaluator.calculate_roi(y_true, y_pred, y_pred_proba)

# Include in interpretability report
breakdown['business_metrics'] = roi_metrics
```

## Conclusion

The interpretability system provides comprehensive analysis capabilities for all modalities in the multimodal donor prediction pipeline. By following this guide and using the provided examples, you can gain deep insights into model behavior and create actionable reports for stakeholders.

For additional support or questions, refer to the example scripts in the `examples/` directory or the integration pipeline in `src/interpretability_integration.py`.



