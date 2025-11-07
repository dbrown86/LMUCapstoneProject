# Feature Importance File Guide

## Why This Happens

The inference script (`generate_will_give_again_predictions.py`) is **correctly preventing data leakage** by refusing to recompute feature importance. Here's why:

### 🛡️ Data Leakage Prevention

1. **Feature selection requires outcomes**: To select the "best" 60 features, the training script uses `mutual_info_classif`, which requires:
   - Feature values (X)
   - Target outcomes (y) - "gave again in 2024"

2. **During inference**: We're predicting "will give again" for 2024, so we can't use 2024 outcomes to select features - that would be cheating!

3. **Training-time selection**: The training script selects features using only **historical data (2021-2023)** and saves the selection to `results/feature_importance_influential_donor.csv`

4. **Inference-time requirement**: The inference script must use the **exact same features** selected during training to match the model architecture

## The Problem

The file `results/feature_importance_influential_donor.csv` is missing because:
- Training script hasn't been run yet, OR
- Training script was run but the file wasn't saved to the expected location, OR
- Training script was run from a different directory

## Solutions

### ✅ **Option 1: Run Training Script (RECOMMENDED)**

This is the proper solution - run the training script to generate the feature importance file:

```bash
cd final_model/src
python simplified_single_target_training.py
```

The training script will:
1. Select top 60 features using historical data (2021-2023)
2. Train the model
3. Save feature importance to `results/feature_importance_influential_donor.csv`

**Note**: Training can take a while (~30-60 minutes depending on your hardware).

### ⚠️ **Option 2: Use Fallback (Temporary)**

The inference script will use the **first 60 features** as a fallback, but this may cause:
- **Feature mismatch errors** if the model was trained with different features
- **Poor performance** if important features are excluded
- **Runtime errors** if feature dimensions don't match

**Only use this if**:
- You're testing the inference pipeline
- You're certain the first 60 features match what was used in training
- You'll run training properly later

### 🔍 **Option 3: Extract from Trained Model (Advanced)**

If you have a trained model but the feature importance file is missing, you could:
1. Check model checkpoint metadata (if saved)
2. Re-run just the feature selection part of the training script (not full training)

## File Locations

The inference script looks for the file in these locations (in order):
1. `results/feature_importance_influential_donor.csv` (relative to script)
2. `../results/feature_importance_influential_donor.csv` (one level up)
3. `../../results/feature_importance_influential_donor.csv` (two levels up)

The training script saves it to: `results/feature_importance_influential_donor.csv` (relative to where it's run from)

## Verification

To check if the file exists:
```bash
# From project root
ls results/feature_importance_influential_donor.csv

# Or from final_model/src
ls ../results/feature_importance_influential_donor.csv
```

## Next Steps

1. **If you haven't trained yet**: Run the training script first
2. **If training failed**: Check the training logs to see why feature importance wasn't saved
3. **If training completed successfully**: Verify the file exists in the `results/` directory

## Why Not Just Recompute?

You might ask: "Why not just recompute feature importance during inference?"

**Answer**: That would be data leakage because:
- Feature selection would use 2024 outcomes (future data)
- The model would "see" answers before making predictions
- Metrics would be artificially inflated
- Model wouldn't work on new, unseen data

The inference script correctly prevents this! ✅

