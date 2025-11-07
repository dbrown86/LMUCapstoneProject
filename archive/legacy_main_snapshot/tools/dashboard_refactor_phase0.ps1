# Phase 0: Safety Net - Dashboard Refactoring Setup
# This script sets up backups and feature branch

# Disable git pager for all commands
$env:GIT_PAGER = ''
$env:PAGER = ''
git config --global core.pager ''

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     Phase 0: Safety Net - Dashboard Refactoring Setup    ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1: Commit current working state
Write-Host "Step 1: Committing current working state..." -ForegroundColor Yellow
git add dashboard/alternate_dashboard.py
if ($LASTEXITCODE -eq 0) {
    git commit -m "Save working dashboard before refactoring" 2>&1 | Out-String
    Write-Host "   ✅ Current state committed" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  No changes to commit" -ForegroundColor Yellow
}
Write-Host ""

# Step 2: Create backup branch
Write-Host "Step 2: Creating backup branch..." -ForegroundColor Yellow
git checkout -b dashboard-refactor-backup 2>&1 | Out-String
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Backup branch created: dashboard-refactor-backup" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Branch may already exist" -ForegroundColor Yellow
}
Write-Host ""

# Step 3: Get current branch name
Write-Host "Step 3: Determining base branch..." -ForegroundColor Yellow
$currentBranch = git branch --show-current 2>&1 | Out-String
Write-Host "   Current branch: $currentBranch" -ForegroundColor Cyan
Write-Host ""

# Step 4: Switch back to base branch
Write-Host "Step 4: Switching back to base branch..." -ForegroundColor Yellow
if ($currentBranch -match "dashboard-refactor-backup") {
    git checkout main 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        git checkout refactor-project-structure 2>&1 | Out-String
    }
}
Write-Host ""

# Step 5: Create feature branch
Write-Host "Step 5: Creating feature branch for refactoring..." -ForegroundColor Yellow
git checkout -b feature/dashboard-refactor 2>&1 | Out-String
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Feature branch created: feature/dashboard-refactor" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Branch may already exist, checking out..." -ForegroundColor Yellow
    git checkout feature/dashboard-refactor 2>&1 | Out-String
}
Write-Host ""

# Step 6: Create physical backup
Write-Host "Step 6: Creating physical backup of dashboard folder..." -ForegroundColor Yellow
$dateStr = Get-Date -Format "yyyyMMdd"
$backupPath = "dashboard_backup_$dateStr"
if (Test-Path $backupPath) {
    Write-Host "   ⚠️  Backup already exists: $backupPath" -ForegroundColor Yellow
} else {
    Copy-Item -Path "dashboard" -Destination $backupPath -Recurse -Force
    Write-Host "   ✅ Physical backup created: $backupPath" -ForegroundColor Green
}
Write-Host ""

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                  Phase 0 Complete! ✅                       ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  • Current state committed" -ForegroundColor White
Write-Host "  • Backup branch: dashboard-refactor-backup" -ForegroundColor White
Write-Host "  • Feature branch: feature/dashboard-refactor" -ForegroundColor White
Write-Host "  • Physical backup: $backupPath" -ForegroundColor White
Write-Host ""
Write-Host "Next: Run Phase 1 script to create directory structure" -ForegroundColor Yellow
Write-Host ""

