# Project Cleanup Plan

## 🎯 Goal
Clean up experimental/duplicate files while preserving important code and history.

## 📦 Step 1: Archive Old Experimental Scripts

### Files to Archive (Move to `archive/experimental/`)

**In `src/` root:**
```
- advanced_feature_engineering_v2.py
- advanced_multimodal_arch_v3.py
- advanced_multimodal_arch.py
- advanced_parquet_multimodal.py
- bert_pipeline.py
- business_metrics_evaluator.py
- class_imbalance_handler.py
- dense_graph_multimodal.py
- enhanced_ensemble_model_v2.py
- enhanced_ensemble_model.py
- enhanced_feature_engineering.py
- enhanced_multimodal_pipeline.py
- improved_multimodal_arch.py
- integrated_trainer.py
- interpretability_integration.py
- model_interpretability.py
- multimodal_arch_sql.py
- multimodal_arch.py
- optimized_sql_training.py
- parquet_multimodal_training.py
- robust_parquet_multimodal.py
- simple_sql_training.py
- sql_data_loader.py
- stable_multimodal_training.py
- training_pipeline_sql.py
- training_pipeline.py
- working_multimodal_final.py
```

**Rationale**: These are experimental versions superseded by `src/models/train_will_give_again.py`

### PowerShell Commands:
```powershell
# Create archive directory
New-Item -ItemType Directory -Path "archive\experimental" -Force

# Move old experimental scripts
$filesToArchive = @(
    "advanced_feature_engineering_v2.py",
    "advanced_multimodal_arch_v3.py",
    "advanced_multimodal_arch.py",
    "advanced_parquet_multimodal.py",
    "bert_pipeline.py",
    "business_metrics_evaluator.py",
    "class_imbalance_handler.py",
    "dense_graph_multimodal.py",
    "enhanced_ensemble_model_v2.py",
    "enhanced_ensemble_model.py",
    "enhanced_feature_engineering.py",
    "enhanced_multimodal_pipeline.py",
    "improved_multimodal_arch.py",
    "integrated_trainer.py",
    "interpretability_integration.py",
    "model_interpretability.py",
    "multimodal_arch_sql.py",
    "multimodal_arch.py",
    "optimized_sql_training.py",
    "parquet_multimodal_training.py",
    "robust_parquet_multimodal.py",
    "simple_sql_training.py",
    "sql_data_loader.py",
    "stable_multimodal_training.py",
    "training_pipeline_sql.py",
    "training_pipeline.py",
    "working_multimodal_final.py"
)

foreach ($file in $filesToArchive) {
    if (Test-Path "src\$file") {
        git mv "src\$file" "archive\experimental\$file"
    }
}
```

---

## 📦 Step 2: Organize `scripts/` Directory

### Files in `scripts/`:
```
scripts/
  - advanced_multimodal_ensemble.py      → archive/experimental/
  - generate_bert_gnn_embeddings.py      → archive/experimental/
  - improved_multimodal_ensemble.py      → archive/experimental/
  - interpretable_ml_ensemble.py         → archive/experimental/
  - multimodal_deep_learning.py          → archive/experimental/
  - simple_neural_network_baseline.py    → archive/experimental/
  - setup_environment.sh                 ✅ Keep (utility script)
  - train.sh                             ✅ Keep (entry point)
```

### PowerShell Commands:
```powershell
# Move old model scripts from scripts/
git mv scripts/advanced_multimodal_ensemble.py archive/experimental/
git mv scripts/generate_bert_gnn_embeddings.py archive/experimental/
git mv scripts/improved_multimodal_ensemble.py archive/experimental/
git mv scripts/interpretable_ml_ensemble.py archive/experimental/
git mv scripts/multimodal_deep_learning.py archive/experimental/
git mv scripts/simple_neural_network_baseline.py archive/experimental/
```

---

## 📦 Step 3: Clean Up `final_model/` Directory

The `final_model/` directory has **duplicates** of files we've already moved:

```
final_model/
  ├── src/
  │   ├── simplified_single_target_training.py  ❌ DUPLICATE (moved to src/models/)
  │   ├── enhanced_temporal_multimodal_training.py  → archive/experimental/
  │   ├── model_value_segmentation.py          → archive/experimental/
  │   ├── temporal_*.py                        → archive/experimental/
  │   ├── models/                              ❌ DUPLICATE (moved to models/)
  │   ├── results/                             → Move to root results/
  │   └── cache/                               ❌ Delete (in .gitignore)
  ├── monitor_training.py                      → src/utils/
  ├── performance_comparison.py                → src/evaluation/
  ├── config/requirements.txt                  → configs/model_requirements.txt
  └── README.md                                → docs/FINAL_MODEL_README.md
```

### PowerShell Commands:
```powershell
# Move useful utilities
git mv final_model/monitor_training.py src/utils/monitor_training.py
git mv final_model/performance_comparison.py src/evaluation/performance_comparison.py

# Move documentation
git mv final_model/README.md docs/FINAL_MODEL_README.md
git mv final_model/config/requirements.txt configs/model_requirements.txt

# Archive experimental scripts
git mv final_model/src/enhanced_temporal_multimodal_training.py archive/experimental/
git mv final_model/src/model_value_segmentation.py archive/experimental/
git mv final_model/src/temporal_cross_validation.py archive/experimental/
git mv final_model/src/temporal_leakage_test.py archive/experimental/
git mv final_model/src/temporal_validation_test.py archive/experimental/

# Copy results to main results folder (if needed)
# Note: Don't use git mv for results as they might be in .gitignore
Copy-Item -Path "final_model\src\results\*" -Destination "results\" -Recurse -Force

# Delete the now-empty final_model directory structure
# (Do this manually after verifying everything is moved)
```

---

## 📦 Step 4: Organize `examples/` Directory

```
examples/
  - basic_training_example.py          ✅ Keep (useful reference)
  - donor_prediction_with_pipeline.py  ✅ Keep (useful reference)
  - interpretability_example.py        ✅ Keep (useful reference)
```

**Action**: Keep as-is! These are good reference examples.

---

## 📦 Step 5: Root-Level Files

### Temporary/Utility Files to Remove:
```powershell
# Remove temporary helper scripts (if they exist)
git rm commit_changes.ps1 -f
git rm verify_medians.py -f
git rm verify_medians2.py -f
git rm create_results_visualization.py -f  # Or move to src/utils/
```

### Documentation Files - Keep & Organize:
- ✅ `README.md` - Keep at root
- ✅ `REFACTORING_GUIDE.md` - Keep at root
- ✅ `REFACTORING_SUMMARY.md` - Keep at root
- ✅ `PROJECT_STRUCTURE.md` - Keep at root
- Move detailed docs to `docs/`:
  ```powershell
  git mv MODEL_PERFORMANCE_ASSESSMENT.md docs/
  git mv FEATURE_IMPORTANCE_GUIDE.md docs/
  git mv TEMPORAL_VALIDATION_*.md docs/
  git mv TRAINING_STATUS_CHECK.md docs/
  git mv PROJECT_EXECUTION_FLOW.md docs/
  git mv EXECUTION_FLOW_DIAGRAM.txt docs/
  ```

---

## 📦 Step 6: Clean Up `src/gnn_models/` and `src/data_generation/`

These look like they have useful utilities:

```
src/gnn_models/          ✅ Keep (GNN utilities)
src/data_generation/     ✅ Keep (data generation utilities)
```

**Action**: Keep as-is! These are modular utilities.

---

## 📦 Step 7: Final Directory Structure

After cleanup, you'll have:

```
LMUCapstoneProject/
├── src/                        # Clean, production code only
│   ├── data/                  # ✅ 3 files (loader, export, generate)
│   ├── models/                # ✅ 2 files (train, predict)
│   ├── features/              # ✅ 1 file (extract_network)
│   ├── utils/                 # ✅ 2 files (visualizations, monitor)
│   ├── evaluation/            # ✅ 1 file (performance_comparison)
│   ├── gnn_models/            # ✅ GNN utilities
│   └── data_generation/       # ✅ Data generation utilities
├── archive/
│   └── experimental/          # 📦 All old experimental scripts
├── scripts/                   # ✅ Just utilities (setup, train)
├── examples/                  # ✅ Reference examples
├── docs/                      # ✅ All documentation
├── dashboard/                 # ✅ Your working dashboard
├── models/                    # ✅ Trained models only
├── data/                      # ✅ Organized data
├── results/                   # ✅ Training results
├── tests/                     # ✅ Tests
└── configs/                   # ✅ Configuration files
```

---

## ⚠️ Important Notes

1. **Test After Each Major Step**: 
   ```powershell
   streamlit run dashboard/alternate_dashboard.py
   ```

2. **Commit After Each Step**: Use atomic commits
   ```powershell
   git add .
   git commit -m "chore: archive experimental scripts to archive/experimental/"
   ```

3. **Don't Delete, Archive**: Keep experimental code in `archive/` for reference

4. **Verify Before Deleting**: Double-check that moved files aren't imported anywhere

---

## 🎯 Priority Order

1. **HIGH**: Archive `src/` experimental files (biggest cleanup)
2. **MEDIUM**: Clean up `scripts/` directory
3. **MEDIUM**: Reorganize `final_model/` directory
4. **LOW**: Move documentation files to `docs/`
5. **LOW**: Remove temporary utility scripts

---

## 🔍 How to Verify Imports

Before archiving, check if any file is still imported:

```powershell
# Example: Check if any file imports advanced_multimodal_arch.py
Get-ChildItem -Recurse -Include *.py | Select-String "from.*advanced_multimodal_arch" | Select-Object -Unique Path
```

---

## ✅ Final Checklist

- [ ] Created `archive/experimental/` directory
- [ ] Moved old experimental scripts from `src/`
- [ ] Moved old model scripts from `scripts/`
- [ ] Cleaned up `final_model/` directory
- [ ] Moved documentation to `docs/`
- [ ] Removed temporary utility scripts
- [ ] Tested dashboard still works
- [ ] Committed all changes
- [ ] Pushed to GitHub

---

## 📚 Reference

See `REFACTORING_GUIDE.md` for the structural changes already completed.

