#!/bin/bash
# Phase 1: Create Directory Structure for Side-by-Side Development

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Phase 1: Creating Directory Structure (Side-by-Side)     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Create new directory structure
echo "Creating new directory structure..."
echo ""

DIRECTORIES=(
    "dashboard/config"
    "dashboard/data"
    "dashboard/models"
    "dashboard/pages"
    "dashboard/components"
    "dashboard/tests"
)

for dir in "${DIRECTORIES[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "   ✅ Created $dir"
    else
        echo "   ✓ $dir already exists"
    fi
done
echo ""

# Create __init__.py files
echo "Creating Python package files..."
echo ""

INIT_FILES=(
    "dashboard/config/__init__.py"
    "dashboard/data/__init__.py"
    "dashboard/models/__init__.py"
    "dashboard/pages/__init__.py"
    "dashboard/components/__init__.py"
    "dashboard/tests/__init__.py"
)

for file in "${INIT_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        touch "$file"
        echo "   ✅ Created $file"
    else
        echo "   ✓ $file already exists"
    fi
done
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              Phase 1 Complete! ✅                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Directory structure created:"
echo "  dashboard/"
echo "    ├── alternate_dashboard.py        (OLD - Keep working!)"
echo "    ├── app_new.py                    (NEW - Build alongside)"
echo "    ├── config/                       (NEW)"
echo "    ├── data/                         (NEW)"
echo "    ├── models/                       (NEW)"
echo "    ├── pages/                        (NEW)"
echo "    ├── components/                   (NEW)"
echo "    └── tests/                        (NEW)"
echo ""
echo "Next: Extract config module (Step 1)"
echo ""

