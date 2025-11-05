# Simpler version - just do the essential cleanup
Write-Host "Starting cleanup..." -ForegroundColor Cyan

# Create archive
New-Item -ItemType Directory -Path "archive\experimental" -Force | Out-Null

# Move files one at a time
$files = @(
    "src\advanced_feature_engineering_v2.py",
    "src\advanced_multimodal_arch_v3.py",
    "src\advanced_multimodal_arch.py",
    "src\advanced_parquet_multimodal.py",
    "src\bert_pipeline.py",
    "src\business_metrics_evaluator.py",
    "src\class_imbalance_handler.py",
    "src\dense_graph_multimodal.py",
    "src\enhanced_ensemble_model_v2.py",
    "src\enhanced_ensemble_model.py",
    "src\enhanced_feature_engineering.py",
    "src\enhanced_multimodal_pipeline.py",
    "src\improved_multimodal_arch.py",
    "src\integrated_trainer.py",
    "src\interpretability_integration.py",
    "src\model_interpretability.py",
    "src\multimodal_arch_sql.py",
    "src\multimodal_arch.py",
    "src\optimized_sql_training.py",
    "src\parquet_multimodal_training.py",
    "src\robust_parquet_multimodal.py",
    "src\simple_sql_training.py",
    "src\sql_data_loader.py",
    "src\stable_multimodal_training.py",
    "src\training_pipeline_sql.py",
    "src\training_pipeline.py",
    "src\working_multimodal_final.py"
)

$count = 0
foreach ($file in $files) {
    if (Test-Path $file) {
        $filename = Split-Path $file -Leaf
        git mv $file "archive\experimental\$filename"
        if ($?) {
            $count++
            Write-Host "Moved $filename" -ForegroundColor Green
        }
    }
}

Write-Host "`nMoved $count files" -ForegroundColor Cyan
Write-Host "Now run: git add . && git commit -m 'chore: archive experimental code'" -ForegroundColor Yellow

