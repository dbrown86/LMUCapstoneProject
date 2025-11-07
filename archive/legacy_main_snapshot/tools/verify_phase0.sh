#!/bin/bash
# Verify Phase 0 was completed correctly

export GIT_PAGER=''
export PAGER=''

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║           Verifying Phase 0 Completion                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check current branch
echo "1. Checking current branch..."
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
echo "   Current branch: $CURRENT_BRANCH"
echo ""

# Check if backup branch exists
echo "2. Checking for backup branch..."
if git branch --list | grep -q "dashboard-refactor-backup"; then
    echo "   ✅ Backup branch found: dashboard-refactor-backup"
    BACKUP_EXISTS=true
else
    echo "   ⚠️  Backup branch NOT found"
    BACKUP_EXISTS=false
fi
echo ""

# Check if feature branch exists
echo "3. Checking for feature branch..."
if git branch --list | grep -q "feature/dashboard-refactor"; then
    echo "   ✅ Feature branch found: feature/dashboard-refactor"
    FEATURE_EXISTS=true
else
    echo "   ⚠️  Feature branch NOT found"
    FEATURE_EXISTS=false
fi
echo ""

# Check for physical backup
echo "4. Checking for physical backup..."
BACKUP_DIRS=$(find . -maxdepth 1 -type d -name "dashboard_backup_*" 2>/dev/null)
if [ -n "$BACKUP_DIRS" ]; then
    BACKUP_NAME=$(echo "$BACKUP_DIRS" | head -1 | xargs basename)
    echo "   ✅ Physical backup found: $BACKUP_NAME"
    PHYSICAL_BACKUP_EXISTS=true
else
    echo "   ⚠️  Physical backup NOT found"
    PHYSICAL_BACKUP_EXISTS=false
fi
echo ""

# Check recent commits
echo "5. Checking recent commits..."
RECENT_COMMITS=$(git log --oneline -5 2>/dev/null)
if echo "$RECENT_COMMITS" | grep -q "Save working dashboard\|before refactoring"; then
    echo "   ✅ Found 'Save working dashboard' commit"
    COMMIT_EXISTS=true
else
    echo "   ⚠️  'Save working dashboard' commit NOT found"
    echo "   Recent commits:"
    echo "$RECENT_COMMITS" | head -5 | sed 's/^/     /'
    COMMIT_EXISTS=false
fi
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Verification Summary                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

ALL_PASSED=true

if [ "$CURRENT_BRANCH" = "feature/dashboard-refactor" ] || [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "refactor-project-structure" ]; then
    echo "   ✅ On correct branch"
else
    echo "   ❌ On correct branch"
    ALL_PASSED=false
fi

if [ "$BACKUP_EXISTS" = true ]; then
    echo "   ✅ Backup branch exists"
else
    echo "   ❌ Backup branch exists"
    ALL_PASSED=false
fi

if [ "$FEATURE_EXISTS" = true ]; then
    echo "   ✅ Feature branch exists"
else
    echo "   ❌ Feature branch exists"
    ALL_PASSED=false
fi

if [ "$PHYSICAL_BACKUP_EXISTS" = true ]; then
    echo "   ✅ Physical backup exists"
else
    echo "   ❌ Physical backup exists"
    ALL_PASSED=false
fi

if [ "$COMMIT_EXISTS" = true ]; then
    echo "   ✅ Safety commit made"
else
    echo "   ❌ Safety commit made"
    ALL_PASSED=false
fi

echo ""
if [ "$ALL_PASSED" = true ]; then
    echo "🎉 Phase 0 completed successfully!"
    echo "   You can proceed to Phase 1: bash dashboard_refactor_phase1.sh"
else
    echo "⚠️  Some checks failed. Please run Phase 0 again:"
    echo "   bash dashboard_refactor_phase0.sh"
fi
echo ""

