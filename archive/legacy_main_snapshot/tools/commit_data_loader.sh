#!/bin/bash
# Commit data loader module

export GIT_PAGER=''
git add dashboard/data/
git commit -m "Add data loader module - tested independently"
echo "✅ Data loader module committed!"

