#!/bin/bash
# Commit metrics module

export GIT_PAGER=''
git add dashboard/models/
git commit -m "Add metrics module - tested independently"
echo "✅ Metrics module committed!"

