#!/bin/bash
# Commit components module

export GIT_PAGER=''
git add dashboard/components/
git commit -m "Add components module - sidebar, metric cards, charts, styles"
echo "✅ Components module committed!"

