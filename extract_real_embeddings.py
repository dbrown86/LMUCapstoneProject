#!/usr/bin/env python3
"""
Extract Real BERT and GNN Embeddings
Step 1 of improving the enhanced pipeline
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import torch
import os

def extract_bert_embeddings():
    """Extract real BERT embeddings from trained model"""
    print("="*80)
    print("EXTRACTING REAL BERT EMBEDDINGS")
    print("="*80)
    
    try:
        # Load data
        print("\nLoading data...")
        donors_df = pd.read_csv('synthetic_donor_dataset/donors.csv')
        contact_reports_df = pd.read_csv('synthetic_donor_dataset/contact_reports.csv')
        print(f"Loaded {len(donors_df):,} donors and {len(contact_reports_df):,} contact reports")
        
        # Import BERT components
        from bert_pipeline import (
            setup_transformer_environment,
            select_model,
            ContactReportClassifier,
            EmbeddingExtractor
        )
        from transformers import AutoTokenizer
        
        # Setup environment
        device = setup_transformer_environment()
        model_name = select_model('bert')
        
        # Load tokenizer
        print(f"\nLoading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create model
        print("Initializing BERT model...")
        model = ContactReportClassifier(
            model_name=model_name,
            num_labels=3,  # Positive, Negative, Unresponsive
            dropout=0.3
        )
        
        # Load trained weights
        if os.path.exists('best_contact_classifier.pt'):
            print("Loading trained model weights...")
            state_dict = torch.load('best_contact_classifier.pt', map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print("✅ BERT model loaded successfully")
        else:
            print("❌ No trained model found at best_contact_classifier.pt")
            return None
        
        # Create embedding extractor
        extractor = EmbeddingExtractor(model, tokenizer, device)
        
        # Aggregate contact reports by donor
        print("\nAggregating contact reports by donor...")
        donor_texts = []
        donor_ids = []
        
        for donor_id in donors_df['ID']:
            donor_reports = contact_reports_df[contact_reports_df['Donor_ID'] == donor_id]
            
            if len(donor_reports) > 0:
                # Combine all reports for this donor
                combined_text = ' [SEP] '.join(donor_reports['Report_Text'].fillna('').astype(str))
                donor_texts.append(combined_text)
            else:
                # Empty text for donors with no reports
                donor_texts.append('')
            
            donor_ids.append(donor_id)
        
        print(f"Created text for {len(donor_texts):,} donors")
        print(f"Donors with contact reports: {sum(1 for text in donor_texts if text):,}")
        print(f"Donors without reports: {sum(1 for text in donor_texts if not text):,}")
        
        # Extract embeddings
        print("\nExtracting BERT embeddings (this may take a few minutes)...")
        bert_embeddings = extractor.extract_embeddings(donor_texts, batch_size=32)
        
        print(f"\n✅ BERT embeddings extracted successfully!")
        print(f"   Shape: {bert_embeddings.shape}")
        print(f"   Mean: {bert_embeddings.mean():.4f}")
        print(f"   Std: {bert_embeddings.std():.4f}")
        
        # Save embeddings
        np.save('bert_embeddings_real.npy', bert_embeddings)
        print(f"   Saved to: bert_embeddings_real.npy")
        
        return bert_embeddings
        
    except Exception as e:
        print(f"\n❌ Error extracting BERT embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_gnn_embeddings():
    """Extract real GNN embeddings from graph neural network"""
    print("\n" + "="*80)
    print("EXTRACTING REAL GNN EMBEDDINGS")
    print("="*80)
    
    try:
        # Load data
        print("\nLoading data...")
        donors_df = pd.read_csv('synthetic_donor_dataset/donors.csv')
        relationships_df = pd.read_csv('synthetic_donor_dataset/relationships.csv')
        print(f"Loaded {len(donors_df):,} donors and {len(relationships_df):,} relationships")
        
        # Import GNN components
        from gnn_models.gnn_pipeline import main_gnn_pipeline
        
        # Run GNN pipeline to get embeddings
        print("\nRunning GNN pipeline (this will take several minutes)...")
        gnn_results = main_gnn_pipeline(
            donors_df=donors_df,
            relationships_df=relationships_df,
            contact_reports_df=None,
            giving_history_df=None
        )
        
        # Extract embeddings
        gnn_embeddings = gnn_results['embeddings']
        
        print(f"\n✅ GNN embeddings extracted successfully!")
        print(f"   Shape: {gnn_embeddings.shape}")
        print(f"   Mean: {gnn_embeddings.mean():.4f}")
        print(f"   Std: {gnn_embeddings.std():.4f}")
        
        # Save embeddings
        np.save('gnn_embeddings_real.npy', gnn_embeddings)
        print(f"   Saved to: gnn_embeddings_real.npy")
        
        return gnn_embeddings
        
    except Exception as e:
        print(f"\n❌ Error extracting GNN embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution function"""
    print("="*100)
    print("REAL EMBEDDINGS EXTRACTION PIPELINE")
    print("="*100)
    print("\nThis script will:")
    print("1. Load the trained BERT model and extract real text embeddings")
    print("2. Run the GNN pipeline and extract real graph embeddings")
    print("3. Save embeddings for use in enhanced pipeline")
    print("\nEstimated time: 10-20 minutes (depends on GPU availability)")
    
    # Extract BERT embeddings
    bert_embeddings = extract_bert_embeddings()
    
    # Extract GNN embeddings
    gnn_embeddings = extract_gnn_embeddings()
    
    # Summary
    print("\n" + "="*100)
    print("EXTRACTION SUMMARY")
    print("="*100)
    
    if bert_embeddings is not None:
        print(f"✅ BERT embeddings: {bert_embeddings.shape} - Saved to bert_embeddings_real.npy")
    else:
        print("❌ BERT embeddings: Failed to extract")
    
    if gnn_embeddings is not None:
        print(f"✅ GNN embeddings: {gnn_embeddings.shape} - Saved to gnn_embeddings_real.npy")
    else:
        print("❌ GNN embeddings: Failed to extract")
    
    if bert_embeddings is not None and gnn_embeddings is not None:
        print("\n🎉 SUCCESS: All real embeddings extracted!")
        print("\nNext step: Run enhanced pipeline with real embeddings")
        print("  python run_enhanced_with_real_embeddings.py")
    else:
        print("\n⚠️  Some embeddings failed to extract")
        print("Check error messages above for details")
    
    return bert_embeddings, gnn_embeddings

if __name__ == "__main__":
    bert_embeddings, gnn_embeddings = main()


