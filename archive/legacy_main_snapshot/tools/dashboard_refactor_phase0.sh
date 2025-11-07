#!/bin/bash
# Phase 0: Safety Net - Dashboard Refactoring Setup
# This script sets up backups and feature branch

set -e  # Exit on error

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Phase 0: Safety Net - Dashboard Refactoring Setup    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Disable git pager
export GIT_PAGER=''
export PAGER=''
git config --global core.pager '' 2>/dev/null || true

# Step 1: Commit current working state
echo "Step 1: Committing current working state..."
git add dashboard/alternate_dashboard.py 2>/dev/null || true
if git diff --staged --quiet dashboard/alternate_dashboard.py 2>/dev/null; then
    echo "   ⚠️  No changes to commit"
else
    git commit -m "Save working dashboard before refactoring" 2>&1
    if [ $? -eq 0 ]; then
        echo "   ✅ Current state committed"
    else
        echo "   ⚠️  Commit failed or nothing to commit"
    fi
fi
echo ""

# Step 2: Create backup branch
echo "Step 2: Creating backup branch..."
if git branch --list | grep -q "dashboard-refactor-backup"; then
    echo "   ⚠️  Backup branch already exists"
else
    git checkout -b dashboard-refactor-backup 2>&1
    if [ $? -eq 0 ]; then
        echo "   ✅ Backup branch created: dashboard-refactor-backup"
    else
        echo "   ⚠️  Failed to create backup branch"
    fi
fi
echo ""

# Step 3: Get current branch name
echo "Step 3: Determining base branch..."
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
echo "   Current branch: $CURRENT_BRANCH"
echo ""

# Step 4: Switch back to base branch (if we created backup branch)
if [ "$CURRENT_BRANCH" = "dashboard-refactor-backup" ]; then
    echo "Step 4: Switching back to base branch..."
    if git branch --list | grep -q "main"; then
        git checkout main 2>&1
    elif git branch --list | grep -q "refactor-project-structure"; then
        git checkout refactor-project-structure 2>&1
    else
        echo "   ⚠️  Could not find base branch, staying on backup branch"
    fi
    CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
    echo "   Switched to: $CURRENT_BRANCH"
    echo ""
fi

# Step 5: Create feature branch
echo "Step 5: Creating feature branch for refactoring..."
if git branch --list | grep -q "feature/dashboard-refactor"; then
    echo "   ⚠️  Feature branch already exists, checking out..."
    git checkout feature/dashboard-refactor 2>&1
    echo "   ✅ Switched to feature branch: feature/dashboard-refactor"
else
    git checkout -b feature/dashboard-refactor 2>&1
    if [ $? -eq 0 ]; then
        echo "   ✅ Feature branch created: feature/dashboard-refactor"
    else
        echo "   ⚠️  Failed to create feature branch"
    fi
fi
echo ""

# Step 6: Create physical backup
echo "Step 6: Creating physical backup of dashboard folder..."
DATE_STR=$(date +%Y%m%d)
BACKUP_PATH="dashboard_backup_$DATE_STR"

if [ -d "$BACKUP_PATH" ]; then
    echo "   ⚠️  Backup already exists: $BACKUP_PATH"
else
    cp -r dashboard "$BACKUP_PATH" 2>&1
    if [ $? -eq 0 ]; then
        echo "   ✅ Physical backup created: $BACKUP_PATH"
    else
        echo "   ⚠️  Failed to create physical backup"
    fi
fi
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  Phase 0 Complete! ✅                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary:"
echo "  • Current state committed"
echo "  • Backup branch: dashboard-refactor-backup"
echo "  • Feature branch: feature/dashboard-refactor"
echo "  • Physical backup: $BACKUP_PATH"
echo ""
echo "Next: Run Phase 1 script to create directory structure"
echo "      bash dashboard_refactor_phase1.sh"
echo ""

