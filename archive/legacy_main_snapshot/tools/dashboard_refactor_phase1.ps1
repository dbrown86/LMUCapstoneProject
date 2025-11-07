# Phase 1: Create Directory Structure for Side-by-Side Development

# Disable git pager
$env:GIT_PAGER = ''
$env:PAGER = ''
git config --global core.pager ''

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Phase 1: Creating Directory Structure (Side-by-Side)    ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Create new directory structure
Write-Host "Creating new directory structure..." -ForegroundColor Yellow
Write-Host ""

$directories = @(
    "dashboard/config",
    "dashboard/data",
    "dashboard/models",
    "dashboard/pages",
    "dashboard/components",
    "dashboard/tests"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   ✅ Created $dir" -ForegroundColor Green
    } else {
        Write-Host "   ✓ $dir already exists" -ForegroundColor Gray
    }
}
Write-Host ""

# Create __init__.py files
Write-Host "Creating Python package files..." -ForegroundColor Yellow
Write-Host ""

$initFiles = @(
    "dashboard/config/__init__.py",
    "dashboard/data/__init__.py",
    "dashboard/models/__init__.py",
    "dashboard/pages/__init__.py",
    "dashboard/components/__init__.py",
    "dashboard/tests/__init__.py"
)

foreach ($file in $initFiles) {
    if (-not (Test-Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "   ✅ Created $file" -ForegroundColor Green
    } else {
        Write-Host "   ✓ $file already exists" -ForegroundColor Gray
    }
}
Write-Host ""

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              Phase 1 Complete! ✅                           ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Directory structure created:" -ForegroundColor Cyan
Write-Host "  dashboard/" -ForegroundColor White
Write-Host "    ├── alternate_dashboard.py        (OLD - Keep working!)" -ForegroundColor Gray
Write-Host "    ├── app_new.py                    (NEW - Build alongside)" -ForegroundColor White
Write-Host "    ├── config/                       (NEW)" -ForegroundColor White
Write-Host "    ├── data/                         (NEW)" -ForegroundColor White
Write-Host "    ├── models/                       (NEW)" -ForegroundColor White
Write-Host "    ├── pages/                        (NEW)" -ForegroundColor White
Write-Host "    ├── components/                   (NEW)" -ForegroundColor White
Write-Host "    └── tests/                        (NEW)" -ForegroundColor White
Write-Host ""
Write-Host "Next: Extract config module (Step 1)" -ForegroundColor Yellow
Write-Host ""

