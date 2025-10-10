#!/usr/bin/env python3
"""
Enhanced Pipeline with Real BERT and GNN Embeddings
This should significantly improve performance over dummy embeddings
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import os

def main():
    """Run enhanced pipeline with real embeddings"""
    print("="*100)
    print("ENHANCED MULTIMODAL PIPELINE WITH REAL EMBEDDINGS")
    print("="*100)
    
    try:
        # Import enhanced pipeline
        from enhanced_multimodal_pipeline import EnhancedMultimodalPipeline
        
        # Load data
        print("\nLoading data...")
        donors_df = pd.read_csv('synthetic_donor_dataset/donors.csv')
        contact_reports_df = pd.read_csv('synthetic_donor_dataset/contact_reports.csv')
        relationships_df = pd.read_csv('synthetic_donor_dataset/relationships.csv')
        
        print(f"Loaded {len(donors_df):,} donors")
        print(f"Loaded {len(contact_reports_df):,} contact reports")
        print(f"Loaded {len(relationships_df):,} relationships")
        
        # Load real embeddings
        print("\nLoading real embeddings...")
        
        if os.path.exists('bert_embeddings_real.npy'):
            bert_embeddings = np.load('bert_embeddings_real.npy')
            print(f"SUCCESS: Loaded BERT embeddings: {bert_embeddings.shape}")
        else:
            print("ERROR: bert_embeddings_real.npy not found")
            print("   Run: python extract_real_embeddings.py first")
            bert_embeddings = None
        
        if os.path.exists('gnn_embeddings_real.npy'):
            gnn_embeddings = np.load('gnn_embeddings_real.npy')
            print(f"SUCCESS: Loaded GNN embeddings: {gnn_embeddings.shape}")
        else:
            print("ERROR: gnn_embeddings_real.npy not found")
            print("   Run: python extract_real_embeddings.py first")
            gnn_embeddings = None
        
        if bert_embeddings is None and gnn_embeddings is None:
            print("\nWARNING: No real embeddings found. Please run extract_real_embeddings.py first")
            print("   python extract_real_embeddings.py")
            return None, None
        
        # Initialize pipeline
        print("\nInitializing enhanced pipeline...")
        pipeline = EnhancedMultimodalPipeline(random_state=42)
        
        # Run with REAL embeddings
        print("\n" + "="*80)
        print("RUNNING ENHANCED PIPELINE WITH REAL MULTIMODAL EMBEDDINGS")
        print("="*80)
        
        results = pipeline.run_complete_pipeline(
            donors_df=donors_df,
            contact_reports_df=contact_reports_df,
            relationships_df=relationships_df,
            giving_history_df=None,
            bert_embeddings=bert_embeddings,      # REAL embeddings!
            gnn_embeddings=gnn_embeddings         # REAL embeddings!
        )
        
        # Save results
        pipeline.save_results("enhanced_pipeline_real_embeddings_results.pkl")
        
        # Print performance comparison
        print("\n" + "="*100)
        print("PERFORMANCE COMPARISON: DUMMY vs REAL EMBEDDINGS")
        print("="*100)
        
        print("\nPREVIOUS (Dummy Embeddings):")
        print("  Accuracy:  61.9%")
        print("  AUC:       0.688")
        print("  Precision: 29.4%")
        print("  Recall:    65.9%")
        print("  F1 Score:  0.399")
        
        ensemble_metrics = results['ensemble_metrics']
        print("\nCURRENT (Real Embeddings):")
        print(f"  Accuracy:  {ensemble_metrics['accuracy']:.1%}")
        print(f"  AUC:       {ensemble_metrics['auc']:.3f}")
        print(f"  Precision: {ensemble_metrics['precision']:.1%}")
        print(f"  Recall:    {ensemble_metrics['recall']:.1%}")
        print(f"  F1 Score:  {ensemble_metrics['f1']:.3f}")
        
        # Calculate improvements
        acc_improvement = (ensemble_metrics['accuracy'] - 0.619) * 100
        auc_improvement = ensemble_metrics['auc'] - 0.688
        precision_improvement = (ensemble_metrics['precision'] - 0.294) * 100
        f1_improvement = ensemble_metrics['f1'] - 0.399
        
        print("\nIMPROVEMENT:")
        print(f"  Accuracy:  {acc_improvement:+.1f} percentage points")
        print(f"  AUC:       {auc_improvement:+.3f}")
        print(f"  Precision: {precision_improvement:+.1f} percentage points")
        print(f"  F1 Score:  {f1_improvement:+.3f}")
        
        # Assessment
        print("\n" + "="*80)
        print("ASSESSMENT")
        print("="*80)
        
        if ensemble_metrics['accuracy'] > 0.68:
            print("SUCCESS: EXCELLENT - Real embeddings significantly improved performance!")
            print("   Metrics now competitive with original GNN models")
        elif ensemble_metrics['accuracy'] > 0.65:
            print("SUCCESS: GOOD - Real embeddings improved performance")
            print("   Consider threshold adjustment for further gains")
        else:
            print("WARNING: MODERATE - Some improvement but still below target")
            print("   Recommended: Adjust threshold to 0.35-0.40")
        
        # Comparison with original models
        print("\nVS ORIGINAL GNN MODELS:")
        print(f"  GraphSAGE: 71.9% accuracy, 0.726 AUC")
        print(f"  GCN:       70.3% accuracy, 0.716 AUC")
        print(f"  Enhanced:  {ensemble_metrics['accuracy']:.1%} accuracy, {ensemble_metrics['auc']:.3f} AUC")
        
        if ensemble_metrics['auc'] >= 0.716:
            print("\nSUCCESS: AUC matches or exceeds original GCN!")
        if ensemble_metrics['accuracy'] >= 0.700:
            print("SUCCESS: Accuracy matches or exceeds original GCN!")
        
        # Next steps
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        
        if ensemble_metrics['accuracy'] < 0.68:
            print("1. Adjust threshold from 0.10 to 0.35")
            print("   This will improve accuracy by 6-8 percentage points")
            print("2. Add SHAP interpretability")
            print("3. Create visualizations for capstone")
        else:
            print("1. SUCCESS: Performance looks good!")
            print("2. Add SHAP interpretability")
            print("3. Create visualizations for capstone")
            print("4. Document results for capstone presentation")
        
        return pipeline, results
        
    except ImportError as e:
        print(f"\nERROR: Import Error: {e}")
        print("\nPlease ensure you have the required packages installed:")
        print("  pip install torch transformers")
        return None, None
        
    except FileNotFoundError as e:
        print(f"\nERROR: File Not Found: {e}")
        print("\nPlease ensure:")
        print("  1. Real embeddings exist (run extract_real_embeddings.py)")
        print("  2. Data files are in synthetic_donor_dataset/")
        return None, None
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Check if embeddings exist
    if not os.path.exists('bert_embeddings_real.npy') or not os.path.exists('gnn_embeddings_real.npy'):
        print("WARNING: Real embeddings not found!")
        print("\nPlease run the extraction script first:")
        print("  python extract_real_embeddings.py")
        print("\nThis will:")
        print("  1. Load the trained BERT model")
        print("  2. Extract embeddings for all 50,000 donors")
        print("  3. Run GNN pipeline for graph embeddings")
        print("  4. Save embeddings for use in this script")
        print("\nEstimated time: 10-20 minutes")
    else:
        print("SUCCESS: Real embeddings found! Running enhanced pipeline...\n")
        pipeline, results = main()

