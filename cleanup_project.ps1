# Project Cleanup Script
# Run this to archive old experimental files and organize the project

Write-Host "🧹 Starting Project Cleanup..." -ForegroundColor Cyan
Write-Host ""

# Step 1: Create archive directory
Write-Host "📦 Step 1: Creating archive directory..." -ForegroundColor Yellow
if (-not (Test-Path "archive\experimental")) {
    New-Item -ItemType Directory -Path "archive\experimental" -Force | Out-Null
    Write-Host "   ✅ Created archive\experimental\" -ForegroundColor Green
} else {
    Write-Host "   ✅ Archive directory already exists" -ForegroundColor Green
}
Write-Host ""

# Step 2: Archive experimental scripts from src/
Write-Host "📦 Step 2: Archiving experimental scripts from src/..." -ForegroundColor Yellow

$srcFiles = @(
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

$movedCount = 0
foreach ($file in $srcFiles) {
    $srcPath = "src\$file"
    $destPath = "archive\experimental\$file"
    
    if (Test-Path $srcPath) {
        git mv $srcPath $destPath 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ Moved $file" -ForegroundColor Green
            $movedCount++
        } else {
            Write-Host "   ⚠️  Could not move $file (may not be tracked)" -ForegroundColor Yellow
        }
    }
}
Write-Host "   📊 Archived $movedCount files from src/" -ForegroundColor Cyan
Write-Host ""

# Step 3: Archive old model scripts from scripts/
Write-Host "📦 Step 3: Archiving old model scripts from scripts/..." -ForegroundColor Yellow

$scriptsFiles = @(
    "advanced_multimodal_ensemble.py",
    "generate_bert_gnn_embeddings.py",
    "improved_multimodal_ensemble.py",
    "interpretable_ml_ensemble.py",
    "multimodal_deep_learning.py",
    "simple_neural_network_baseline.py"
)

$movedCount = 0
foreach ($file in $scriptsFiles) {
    $srcPath = "scripts\$file"
    $destPath = "archive\experimental\$file"
    
    if (Test-Path $srcPath) {
        git mv $srcPath $destPath 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ Moved $file" -ForegroundColor Green
            $movedCount++
        }
    }
}
Write-Host "   📊 Archived $movedCount files from scripts/" -ForegroundColor Cyan
Write-Host ""

# Step 4: Move utilities from final_model/
Write-Host "📦 Step 4: Moving utilities from final_model/..." -ForegroundColor Yellow

if (Test-Path "final_model\monitor_training.py") {
    git mv "final_model\monitor_training.py" "src\utils\monitor_training.py" 2>&1 | Out-Null
    Write-Host "   ✅ Moved monitor_training.py to src/utils/" -ForegroundColor Green
}

if (Test-Path "final_model\performance_comparison.py") {
    git mv "final_model\performance_comparison.py" "src\evaluation\performance_comparison.py" 2>&1 | Out-Null
    Write-Host "   ✅ Moved performance_comparison.py to src/evaluation/" -ForegroundColor Green
}
Write-Host ""

# Step 5: Archive experimental scripts from final_model/src/
Write-Host "📦 Step 5: Archiving experimental scripts from final_model/src/..." -ForegroundColor Yellow

$finalModelFiles = @(
    "enhanced_temporal_multimodal_training.py",
    "model_value_segmentation.py",
    "temporal_cross_validation.py",
    "temporal_leakage_test.py",
    "temporal_validation_test.py"
)

$movedCount = 0
foreach ($file in $finalModelFiles) {
    $srcPath = "final_model\src\$file"
    $destPath = "archive\experimental\$file"
    
    if (Test-Path $srcPath) {
        git mv $srcPath $destPath 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ Moved $file" -ForegroundColor Green
            $movedCount++
        }
    }
}
Write-Host "   📊 Archived $movedCount files from final_model/src/" -ForegroundColor Cyan
Write-Host ""

# Step 6: Move documentation
Write-Host "📦 Step 6: Moving documentation to docs/..." -ForegroundColor Yellow

if (-not (Test-Path "docs")) {
    New-Item -ItemType Directory -Path "docs" -Force | Out-Null
}

$docFiles = @(
    "MODEL_PERFORMANCE_ASSESSMENT.md",
    "FEATURE_IMPORTANCE_GUIDE.md",
    "TEMPORAL_VALIDATION_COMPLETE.md",
    "TEMPORAL_VALIDATION_SUMMARY.txt",
    "TRAINING_STATUS_CHECK.md",
    "PROJECT_EXECUTION_FLOW.md",
    "EXECUTION_FLOW_DIAGRAM.txt"
)

$movedCount = 0
foreach ($file in $docFiles) {
    if (Test-Path $file) {
        git mv $file "docs\$file" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ Moved $file" -ForegroundColor Green
            $movedCount++
        }
    }
}

# Move final_model docs
if (Test-Path "final_model\README.md") {
    git mv "final_model\README.md" "docs\FINAL_MODEL_README.md" 2>&1 | Out-Null
    Write-Host "   ✅ Moved final_model/README.md" -ForegroundColor Green
    $movedCount++
}

if (Test-Path "final_model\config\requirements.txt") {
    if (-not (Test-Path "configs")) {
        New-Item -ItemType Directory -Path "configs" -Force | Out-Null
    }
    git mv "final_model\config\requirements.txt" "configs\model_requirements.txt" 2>&1 | Out-Null
    Write-Host "   ✅ Moved final_model/config/requirements.txt" -ForegroundColor Green
    $movedCount++
}

Write-Host "   📊 Moved $movedCount documentation files" -ForegroundColor Cyan
Write-Host ""

# Step 7: Remove temporary files
Write-Host "📦 Step 7: Removing temporary files..." -ForegroundColor Yellow

$tempFiles = @(
    "commit_changes.ps1",
    "verify_medians.py",
    "verify_medians2.py",
    "create_results_visualization.py"
)

$removedCount = 0
foreach ($file in $tempFiles) {
    if (Test-Path $file) {
        git rm $file -f 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ Removed $file" -ForegroundColor Green
            $removedCount++
        }
    }
}
Write-Host "   📊 Removed $removedCount temporary files" -ForegroundColor Cyan
Write-Host ""

# Step 8: Commit changes
Write-Host "💾 Step 8: Committing changes..." -ForegroundColor Yellow
git add .
git commit -m "chore: archive experimental code and organize project structure

- Archived 27+ experimental scripts to archive/experimental/
- Moved utilities to src/utils/ and src/evaluation/
- Organized documentation in docs/
- Removed temporary helper scripts
- Clean project structure ready for production"

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Changes committed successfully" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Commit failed or no changes to commit" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "✅ Cleanup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Summary:" -ForegroundColor Cyan
Write-Host "   • Experimental code archived to archive/experimental/"
Write-Host "   • Utilities organized in src/"
Write-Host "   • Documentation moved to docs/"
Write-Host "   • Temporary files removed"
Write-Host ""
Write-Host "🧪 Next Step: Test the dashboard" -ForegroundColor Yellow
Write-Host "   Run: streamlit run dashboard\alternate_dashboard.py"
Write-Host ""
Write-Host "📤 Don't forget to push:" -ForegroundColor Yellow
Write-Host "   git push"

