#!/bin/bash
# Commit script for Phase 6: Extract pages module

set -e

echo "=========================================="
echo "Committing Phase 6: Extract Pages Module"
echo "=========================================="
echo ""

# Suppress git pager
export GIT_PAGER=''

# Add all page files
git add dashboard/pages/__init__.py
git add dashboard/pages/utils.py
git add dashboard/pages/performance.py
git add dashboard/pages/features.py
git add dashboard/pages/donor_insights.py
git add dashboard/pages/predictions.py
git add dashboard/pages/test_pages.py

# Commit
git commit -m "Phase 6: Extract page modules (performance, features, donor_insights, predictions)

- Created dashboard/pages/ directory structure
- Extracted page_performance to pages/performance.py
- Extracted page_features to pages/features.py
- Extracted page_donor_insights to pages/donor_insights.py
- Extracted page_predictions to pages/predictions.py
- Created pages/utils.py for shared helper functions
- Added test script for page modules
- All pages maintain backward compatibility with alternate_dashboard.py
- Old dashboard still works (side-by-side development)"

echo ""
echo "✅ Phase 6 committed successfully!"
echo ""
echo "Next steps:"
echo "1. Test the page modules: python dashboard/pages/test_pages.py"
echo "2. Extract remaining complex pages (dashboard, business_impact, take_action, model_comparison)"
echo "3. Update alternate_dashboard.py to use new page modules"

