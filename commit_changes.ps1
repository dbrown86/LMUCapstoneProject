# PowerShell script to commit all modified files except the large pkl file
# The file is already in .gitignore, so it will be automatically excluded

Write-Host "Step 1: Switching to or creating branch feature/LGCP-180-Model-Performance-Monitoring..."
git checkout feature/LGCP-180-Model-Performance-Monitoring 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Branch doesn't exist, creating it..."
    git checkout -b feature/LGCP-180-Model-Performance-Monitoring
}

Write-Host "`nStep 2: Staging all modified files (pkl file will be excluded by .gitignore)..."
git add .gitignore
git add dashboard/alternate_dashboard.py
git add final_model/src/generate_will_give_again_predictions.py
git add .

Write-Host "`nStep 3: Showing what will be committed (verifying pkl file is excluded)..."
git status

Write-Host "`nStep 4: Committing changes..."
git commit -m "Update dashboard: Fix revenue calculation, remove debug panels, adjust segment visualization

- Fixed revenue calculation to use Last_Gift median instead of corrupted avg_gift_amount
- Removed debug panels from donor segment visualization  
- Adjusted y-axis padding to prevent text cutoff on bar chart
- Removed segment performance summary table/chart
- Updated .gitignore to exclude large cache files"

Write-Host "`nStep 5: Pushing to remote..."
git push origin feature/LGCP-180-Model-Performance-Monitoring

Write-Host "`nDone! All changes committed and pushed."

