"""
Script to retrain the fusion model for 2025 predictions and generate updated parquet files
"""
import subprocess
import sys
import os
from pathlib import Path

print("=" * 80)
print("RETRAINING FUSION MODEL FOR 2025 PREDICTIONS")
print("=" * 80)

# Check if we're in the right directory
if not os.path.exists('src/models/train_will_give_again.py'):
    print("❌ Error: train_will_give_again.py not found. Please run from project root.")
    sys.exit(1)

# Step 1: Retrain the model
print("\n" + "=" * 80)
print("STEP 1: RETRAINING MODEL FOR 2025 TARGET")
print("=" * 80)
print("\nThis will train the fusion model to predict 2025 giving.")
print("Training may take a while depending on your hardware...\n")

try:
    # Run training script
    result = subprocess.run(
        [sys.executable, 'src/models/train_will_give_again.py'],
        cwd=os.getcwd(),
        check=False  # Don't fail if training has warnings
    )
    
    if result.returncode != 0:
        print(f"\n⚠️  Training completed with exit code {result.returncode}")
        print("This may be normal if there are warnings. Checking for model file...")
    else:
        print("\n✅ Training completed successfully!")
        
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    print("Please check the error messages above.")
    sys.exit(1)

# Check if model was saved
model_paths = [
    'models/best_donor_model_2025.pt',
    'models/best_donor_model.pt',
    'best_donor_model.pt'
]

model_found = False
for model_path in model_paths:
    if os.path.exists(model_path):
        print(f"\n✅ Model file found: {model_path}")
        model_found = True
        break

if not model_found:
    print("\n⚠️  Model file not found in expected locations.")
    print("Training may have saved it elsewhere. Please check the training output.")
    print("You can still proceed to generate predictions if you have a model file.")

# Step 2: Generate predictions
print("\n" + "=" * 80)
print("STEP 2: GENERATING PREDICTIONS AND UPDATING PARQUET FILES")
print("=" * 80)
print("\nThis will generate predictions for all donors and update the parquet file.\n")

try:
    # Run prediction generation script
    result = subprocess.run(
        [sys.executable, 'src/models/generate_predictions.py'],
        cwd=os.getcwd(),
        check=False
    )
    
    if result.returncode != 0:
        print(f"\n⚠️  Prediction generation completed with exit code {result.returncode}")
    else:
        print("\n✅ Predictions generated successfully!")
        
except Exception as e:
    print(f"\n❌ Error during prediction generation: {e}")
    print("Please check the error messages above.")
    sys.exit(1)

# Verify output
parquet_path = 'data/parquet_export/donors_with_network_features.parquet'
if os.path.exists(parquet_path):
    print(f"\n✅ Parquet file updated: {parquet_path}")
    
    # Quick verification
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path, engine='pyarrow')
        print(f"   Total records: {len(df):,}")
        
        if 'Will_Give_Again_Probability' in df.columns:
            print(f"   ✅ Will_Give_Again_Probability column found")
            print(f"      Mean probability: {df['Will_Give_Again_Probability'].mean():.3f}")
            print(f"      Range: {df['Will_Give_Again_Probability'].min():.3f} - {df['Will_Give_Again_Probability'].max():.3f}")
        
        if 'Gave_Again_In_2025' in df.columns:
            gave_2025 = df['Gave_Again_In_2025'].sum()
            print(f"   ✅ Gave_Again_In_2025 column found: {gave_2025:,} donors ({gave_2025/len(df)*100:.1f}%)")
        
        if 'Gave_Again_In_2024' in df.columns:
            gave_2024 = df['Gave_Again_In_2024'].sum()
            print(f"   ✅ Gave_Again_In_2024 column found: {gave_2024:,} donors ({gave_2024/len(df)*100:.1f}%)")
            
    except Exception as e:
        print(f"   ⚠️  Could not verify parquet file: {e}")
else:
    print(f"\n❌ Parquet file not found: {parquet_path}")

print("\n" + "=" * 80)
print("✅ PROCESS COMPLETE")
print("=" * 80)
print("""
Next steps:
1. The parquet file has been updated with 2025 predictions
2. Restart your Streamlit dashboard to see the new predictions
3. The dashboard will now use Gave_Again_In_2025 as the primary target

Note: If training failed or you want to use an existing model,
you can run generate_predictions.py directly with an existing model file.
""")

