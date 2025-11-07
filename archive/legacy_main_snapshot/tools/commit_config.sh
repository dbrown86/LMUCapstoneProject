#!/bin/bash
# Commit config module

export GIT_PAGER=''
git add dashboard/config/
git commit -m "Add config module - old dashboard still works"
echo "✅ Config module committed!"

