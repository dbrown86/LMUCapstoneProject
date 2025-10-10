#!/usr/bin/env python3
"""
Run Enhanced Pipeline with REAL BERT and GNN Embeddings
This will significantly improve metrics vs dummy embeddings
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append('src')

def main():
    print("="*100)
    print("ENHANCED PIPELINE WITH REAL MULTIMODAL EMBEDDINGS")
    print("="*100)
    
    # Load data
    print("\nLoading data...")
    donors_df = pd.read_csv('synthetic_donor_dataset/donors.csv')
    contact_reports_df = pd.read_csv('synthetic_donor_dataset/contact_reports.csv')
    relationships_df = pd.read_csv('synthetic_donor_dataset/relationships.csv')
    
    print(f"Loaded {len(donors_df):,} donors")
    print(f"Loaded {len(contact_reports_df):,} contact reports")
    print(f"Loaded {len(relationships_df):,} relationships")
    
    # Step 1: Get REAL BERT Embeddings
    print("\n" + "="*80)
    print("STEP 1: TRAINING BERT AND EXTRACTING REAL TEXT EMBEDDINGS")
    print("="*80)
    
    try:
        from bert_pipeline import run_bert_pipeline_on_contact_reports, EmbeddingExtractor
        import torch
        
        # Check if we already have trained model
        if os.path.exists('best_contact_classifier.pt'):
            print("Found existing BERT model, loading...")
            
            from bert_pipeline import setup_transformer_environment, select_model
            device = setup_transformer_environment()
            model_name = select_model('bert')
            
            # Load tokenizer and model
            from transformers import AutoTokenizer, AutoModel
            from bert_pipeline import ContactReportClassifier
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = ContactReportClassifier(model_name, num_labels=3)
            
            checkpoint = torch.load('best_contact_classifier.pt', map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)
            
            extractor = EmbeddingExtractor(model, tokenizer, device)
            
        else:
            print("No existing model found, training BERT...")
            bert_results = run_bert_pipeline_on_contact_reports(
                data_dir="synthetic_donor_dataset",
                model_choice='bert',
                batch_size=16,
                epochs=3,
                learning_rate=2e-5
            )
            extractor = bert_results['extractor']
        
        # Create donor texts
        print("\nAggregating contact reports by donor...")
        donor_texts = []
        for donor_id in donors_df['ID']:
            donor_reports = contact_reports_df[contact_reports_df['Donor_ID'] == donor_id]
            if len(donor_reports) > 0:
                combined_text = ' '.join(donor_reports['Report_Text'].fillna('').astype(str))
                donor_texts.append(combined_text)
            else:
                donor_texts.append('')
        
        # Extract embeddings
        print("Extracting BERT embeddings...")
        bert_embeddings = extractor.extract_embeddings(donor_texts, batch_size=32)
        print(f"✅ Real BERT embeddings extracted: {bert_embeddings.shape}")
        
    except Exception as e:
        print(f"Error loading BERT embeddings: {e}")
        print("Falling back to dummy embeddings...")
        bert_embeddings = None
    
    # Step 2: Get REAL GNN Embeddings
    print("\n" + "="*80)
    print("STEP 2: TRAINING GNN AND EXTRACTING REAL GRAPH EMBEDDINGS")
    print("="*80)
    
    try:
        from gnn_models.gnn_pipeline import main_gnn_pipeline
        
        print("Running GNN pipeline...")
        gnn_results = main_gnn_pipeline(
            donors_df=donors_df,
            relationships_df=relationships_df,
            contact_reports_df=None,
            giving_history_df=None
        )
        
        gnn_embeddings = gnn_results['embeddings']
        print(f"✅ Real GNN embeddings extracted: {gnn_embeddings.shape}")
        
    except Exception as e:
        print(f"Error loading GNN embeddings: {e}")
        print("Falling back to dummy embeddings...")
        gnn_embeddings = None
    
    # Step 3: Run Enhanced Pipeline with REAL Embeddings
    print("\n" + "="*80)
    print("STEP 3: RUNNING ENHANCED PIPELINE WITH REAL MULTIMODAL EMBEDDINGS")
    print("="*80)
    
    from enhanced_multimodal_pipeline import EnhancedMultimodalPipeline
    
    pipeline = EnhancedMultimodalPipeline(random_state=42)
    
    # Temporarily adjust threshold for better balance
    # Modify the pipeline to use threshold 0.35 instead of 0.10
    
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
    
    # Print comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: DUMMY vs REAL EMBEDDINGS")
    print("="*80)
    
    print("\nDummy Embeddings Results (Previous):")
    print("  Accuracy: 61.9%")
    print("  AUC: 0.688")
    print("  Precision: 29.4%")
    print("  Recall: 65.9%")
    print("  F1: 0.399")
    
    ensemble_metrics = results['ensemble_metrics']
    print("\nReal Embeddings Results (Current):")
    print(f"  Accuracy: {ensemble_metrics['accuracy']:.1%}")
    print(f"  AUC: {ensemble_metrics['auc']:.3f}")
    print(f"  Precision: {ensemble_metrics['precision']:.1%}")
    print(f"  Recall: {ensemble_metrics['recall']:.1%}")
    print(f"  F1: {ensemble_metrics['f1']:.3f}")
    
    # Calculate improvement
    acc_improvement = (ensemble_metrics['accuracy'] - 0.619) * 100
    auc_improvement = ensemble_metrics['auc'] - 0.688
    
    print(f"\nImprovement:")
    print(f"  Accuracy: {acc_improvement:+.1f} percentage points")
    print(f"  AUC: {auc_improvement:+.3f}")
    
    if ensemble_metrics['accuracy'] > 0.68:
        print("\n✅ SUCCESS: Real embeddings significantly improved performance!")
        print("   Metrics now competitive with original GNN models")
    else:
        print("\n⚠️  Metrics still lower - consider threshold adjustment")
    
    return pipeline, results

if __name__ == "__main__":
    pipeline, results = main()
    
    print("\n" + "="*100)
    print("NEXT STEPS:")
    print("="*100)
    print("1. Review results above")
    print("2. If accuracy < 68%, adjust threshold to 0.35 in source code")
    print("3. Add SHAP interpretability (see RECOMMENDED_ACTION_PLAN.md)")
    print("4. Create visualizations for capstone presentation")


