# Complete Phase 0 - Add missing steps (Version 2 - Simplified)
$env:GIT_PAGER = ''
$env:PAGER = ''
git config --global core.pager '' 2>&1 | Out-Null

Write-Host "Completing Phase 0 missing steps..." -ForegroundColor Yellow
Write-Host ""

# Step 1: Create safety commit
Write-Host "1. Creating safety commit..." -ForegroundColor Yellow
git add dashboard/alternate_dashboard.py 2>&1 | Out-Null
$commitResult = git commit -m "Save working dashboard before refactoring" 2>&1 | Out-String

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Safety commit created" -ForegroundColor Green
}
elseif ($commitResult -match "nothing to commit") {
    Write-Host "   ✓ No changes to commit (already up to date)" -ForegroundColor Gray
}
else {
    Write-Host "   ⚠️  Commit issue" -ForegroundColor Yellow
}
Write-Host ""

# Step 2: Create physical backup
Write-Host "2. Creating physical backup..." -ForegroundColor Yellow
$dateStr = Get-Date -Format "yyyyMMdd"
$backupPath = "dashboard_backup_$dateStr"

if (Test-Path $backupPath) {
    Write-Host "   ✓ Backup already exists: $backupPath" -ForegroundColor Gray
}
else {
    Copy-Item -Path "dashboard" -Destination $backupPath -Recurse -Force
    Write-Host "   ✅ Physical backup created: $backupPath" -ForegroundColor Green
}
Write-Host ""

Write-Host "✅ Phase 0 completion done!" -ForegroundColor Green
Write-Host ""
Write-Host "Run verification again: .\verify_phase0.ps1" -ForegroundColor Yellow

