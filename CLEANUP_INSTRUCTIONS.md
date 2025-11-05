# How to Run the Cleanup

The terminal is experiencing some issues, so I've created a PowerShell script for you to run manually.

## 🚀 Quick Start

**Option 1: Run the script (Recommended)**
```powershell
# In PowerShell, from your project root:
.\cleanup_project.ps1
```

**Option 2: Manual cleanup (if script doesn't work)**
See the detailed commands in `CLEANUP_PLAN.md`

## 📋 What the Script Does

1. ✅ Creates `archive/experimental/` directory
2. ✅ Moves 27+ old experimental scripts from `src/`
3. ✅ Moves old model scripts from `scripts/`
4. ✅ Reorganizes `final_model/` utilities
5. ✅ Moves documentation to `docs/`
6. ✅ Removes temporary helper files
7. ✅ Commits all changes with a clear message

## ⚠️ If PowerShell Blocks the Script

If you get an execution policy error:
```powershell
# Run this first:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Then run the script:
.\cleanup_project.ps1
```

## 🧪 After Cleanup

1. **Test the dashboard:**
   ```powershell
   streamlit run dashboard\alternate_dashboard.py
   ```

2. **If everything works, push:**
   ```powershell
   git push
   ```

3. **If something breaks:**
   ```powershell
   git reset --hard HEAD~1  # Undo the cleanup commit
   ```

## 📊 Expected Result

Your `src/` directory will be clean with only:
```
src/
├── data/               ✅ 3 production files
├── models/             ✅ 2 production files  
├── features/           ✅ 1 production file
├── utils/              ✅ 2 utility files
├── evaluation/         ✅ 1 evaluation file
├── gnn_models/         ✅ GNN utilities
└── data_generation/    ✅ Data generation utilities
```

All experimental code will be in:
```
archive/
└── experimental/       📦 27+ old experimental scripts
```

## ❓ Questions?

- **"Will this break my dashboard?"** - No! We're only archiving unused experimental code
- **"Can I undo this?"** - Yes! `git reset --hard HEAD~1` or checkout `backup-before-refactor` branch
- **"Should I delete the archived files?"** - Not yet! Keep them for reference, delete later if not needed

## ✅ You're Ready!

Run the script and then test your dashboard. Good luck! 🚀

