# Verify Phase 0 was completed correctly
# This script disables git pager for all commands

# Disable git pager completely
$env:GIT_PAGER = ''
$env:PAGER = ''
git config --global core.pager '' 2>&1 | Out-Null

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           Verifying Phase 0 Completion                     ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check current branch
Write-Host "1. Checking current branch..." -ForegroundColor Yellow
$currentBranch = (git branch --show-current 2>&1 | Out-String).Trim()
Write-Host "   Current branch: $currentBranch" -ForegroundColor Cyan
Write-Host ""

# Check if backup branch exists
Write-Host "2. Checking for backup branch..." -ForegroundColor Yellow
$backupBranch = (git branch 2>&1 | Select-String "dashboard-refactor-backup" | Out-String).Trim()
if ($backupBranch) {
    Write-Host "   ✅ Backup branch found: dashboard-refactor-backup" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Backup branch NOT found" -ForegroundColor Yellow
}
Write-Host ""

# Check if feature branch exists
Write-Host "3. Checking for feature branch..." -ForegroundColor Yellow
$featureBranch = (git branch 2>&1 | Select-String "feature/dashboard-refactor" | Out-String).Trim()
if ($featureBranch) {
    Write-Host "   ✅ Feature branch found: feature/dashboard-refactor" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Feature branch NOT found" -ForegroundColor Yellow
}
Write-Host ""

# Check for physical backup
Write-Host "4. Checking for physical backup..." -ForegroundColor Yellow
$backupDirs = Get-ChildItem -Directory -Filter "dashboard_backup_*" -ErrorAction SilentlyContinue
if ($backupDirs -and $backupDirs.Count -gt 0) {
    Write-Host "   ✅ Physical backup found: $($backupDirs[0].Name)" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Physical backup NOT found" -ForegroundColor Yellow
}
Write-Host ""

# Check recent commits
Write-Host "5. Checking recent commits..." -ForegroundColor Yellow
$recentCommits = git log --oneline -5 --no-pager 2>&1 | Out-String
$hasBackupCommit = $recentCommits -match "Save working dashboard" -or $recentCommits -match "before refactoring"
if ($hasBackupCommit) {
    Write-Host "   ✅ Found 'Save working dashboard' commit" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  'Save working dashboard' commit NOT found" -ForegroundColor Yellow
    Write-Host "   Recent commits:" -ForegroundColor Gray
    $recentCommits -split "`n" | Select-Object -First 5 | ForEach-Object {
        Write-Host "     $_" -ForegroundColor Gray
    }
}
Write-Host ""

# Summary
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                    Verification Summary                     ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$checks = @(
    @{Name="On correct branch"; Status=($currentBranch -eq "feature/dashboard-refactor" -or $currentBranch -eq "main" -or $currentBranch -eq "refactor-project-structure")},
    @{Name="Backup branch exists"; Status=($backupBranch -ne "")},
    @{Name="Feature branch exists"; Status=($featureBranch -ne "")},
    @{Name="Physical backup exists"; Status=($backupDirs.Count -gt 0)},
    @{Name="Safety commit made"; Status=$hasBackupCommit}
)

$allPassed = $true
foreach ($check in $checks) {
    if ($check.Status) {
        Write-Host "   ✅ $($check.Name)" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $($check.Name)" -ForegroundColor Red
        $allPassed = $false
    }
}

Write-Host ""
if ($allPassed) {
    Write-Host "🎉 Phase 0 completed successfully!" -ForegroundColor Green
    Write-Host "   You can proceed to Phase 1: .\dashboard_refactor_phase1.ps1" -ForegroundColor Yellow
} else {
    Write-Host "⚠️  Some checks failed. Please run Phase 0 again:" -ForegroundColor Yellow
    Write-Host "   .\dashboard_refactor_phase0.ps1" -ForegroundColor Yellow
}
Write-Host ""

