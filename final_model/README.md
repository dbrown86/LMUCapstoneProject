# Multi-Target Donor Engagement Prediction System

## 🎯 **Project Overview**

This project implements a **multi-target deep learning system** for predicting various donor engagement behaviors using a 500K synthetic donor dataset. Instead of predicting a single, poorly-defined target, we predict 5 well-defined, actionable engagement metrics.

## 🚀 **New Project Scope**

### **Core Concept**
- **Multi-Target Prediction**: 5 different donor engagement behaviors
- **500K Dataset**: Full synthetic donor dataset with network features
- **Deep Learning**: Multimodal fusion with GNN integration
- **Production Ready**: Phase 3 as baseline (F1 = 53.69%)

### **5 Target Definitions**

| Target | Definition | Business Value | Expected F1 |
|--------|------------|----------------|-------------|
| **High Engagement** | Top 20% by Engagement_Score | Identify responsive donors | 65-75% |
| **Network Influencers** | Top 20% by network influence | Find peer influencers | 60-70% |
| **Consistent Givers** | Top 20% by giving consistency | Reliable supporters | 70-80% |
| **High-Value Prospects** | Top 15% by Lifetime_Giving | Major gift potential | 55-65% |
| **Recent Engagers** | Top 25% by recency | Active giving phase | 60-70% |

## 📁 **Project Structure**

```
multi_target_production/
├── src/
│   ├── quick_multi_target_500k.py    # Quick implementation
│   ├── multimodal_fusion.py          # Core architecture
│   └── gnn_integration.py            # GNN enhancement
├── models/
│   ├── phase3_baseline.pt            # Production baseline
│   └── multi_target_models/          # Multi-target models
├── data/
│   ├── parquet_export/               # Processed data
│   └── network_features/             # GNN data
├── results/
│   ├── phase3_results/               # Baseline results
│   └── multi_target_results/         # New results
└── docs/
    ├── architecture_guide.md         # Technical docs
    └── deployment_guide.md           # Production guide
```

## 🏗️ **Architecture**

### **Multimodal Fusion Model**
- **Tabular Branch**: Giving, engagement, network features
- **Sequence Branch**: LSTM for giving history
- **Network Branch**: GNN for relationship data
- **Multi-Target Heads**: 5 specialized prediction heads
- **Cross-Target Attention**: Learn from target interactions

### **Key Features**
- **No Data Leakage**: Proper target definitions
- **Class Balancing**: Focal loss + class weights
- **Robust Training**: Gradient clipping, early stopping
- **Ensemble Methods**: Multiple model architectures

## 🚀 **Quick Start**

### **1. Run Quick Implementation**
```bash
cd multi_target_production
python src/quick_multi_target_500k.py
```

### **2. Expected Results**
- **Training Time**: 2-3 hours
- **Average F1**: 50-70% across targets
- **Best Targets**: Engagement, consistency
- **GNN Integration**: Next phase

## 📊 **Performance Baseline**

| Model | Dataset | F1 Score | Status |
|-------|---------|----------|--------|
| **Phase 3** | 40K | **53.69%** | ✅ Production Ready |
| Multi-Target | 500K | **50-70%** | 🔄 In Development |

## 🎯 **Business Value**

### **Immediate Benefits**
- **Actionable Insights**: 5 different donor types
- **Segmentation**: Target specific donor behaviors
- **Prioritization**: Rank donors by multiple criteria
- **Scalability**: 500K record capacity

### **Use Cases**
1. **Engagement Campaigns**: Target high-engagement donors
2. **Peer Influence**: Identify network influencers
3. **Stewardship**: Focus on consistent givers
4. **Major Gifts**: Target high-value prospects
5. **Reactivation**: Engage recent engagers

## 🔧 **Technical Implementation**

### **Phase 1: Core Multi-Target (Today)**
- ✅ Basic multi-target model
- ✅ 5 targets on 500K records
- ✅ Initial performance assessment

### **Phase 2: GNN Integration (Tomorrow)**
- 🔄 Add GNN layers for network targets
- 🔄 Enhance with relationship data
- 🔄 Optimize performance

### **Phase 3: Production Deployment**
- 🔄 Model optimization
- 🔄 API development
- 🔄 Monitoring and maintenance

## 📈 **Success Metrics**

- **Primary**: Average F1 > 60% across targets
- **Secondary**: Individual target F1 > 50%
- **Tertiary**: Production deployment ready

## 🛠️ **Development Status**

- **Phase 3 Baseline**: ✅ Complete (F1 = 53.69%)
- **Multi-Target Core**: 🔄 In Progress
- **GNN Integration**: ⏳ Planned
- **Production Deploy**: ⏳ Planned

---

**This represents a significant pivot from single-target prediction to multi-target engagement prediction, leveraging the full 500K dataset with deep learning techniques.**