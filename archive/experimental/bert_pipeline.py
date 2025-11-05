# BERT/RoBERTa Text Analysis Pipeline for Contact Reports
# Complete implementation for your synthetic donor dataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    BertModel,
    RobertaModel
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("Transformers libraries loaded successfully!")

# =============================================================================
# STEP 1: SETUP AND MODEL SELECTION
# =============================================================================

def setup_transformer_environment():
    """Setup environment and check GPU availability"""
    print("=" * 60)
    print("TRANSFORMER ENVIRONMENT SETUP")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device

def select_model(model_choice='bert'):
    """
    Select pre-trained model for fine-tuning
    
    Options:
    - 'bert': BERT-base (110M params, faster)
    - 'roberta': RoBERTa-base (125M params, often better performance)
    - 'distilbert': DistilBERT (66M params, fastest, 95% of BERT performance)
    """
    
    model_configs = {
        'bert': {
            'model_name': 'bert-base-uncased',
            'description': 'BERT-base: Good balance, 110M parameters'
        },
        'roberta': {
            'model_name': 'roberta-base',
            'description': 'RoBERTa-base: Often better performance, 125M parameters'
        },
        'distilbert': {
            'model_name': 'distilbert-base-uncased',
            'description': 'DistilBERT: Faster, 66M parameters, 95% BERT performance'
        }
    }
    
    config = model_configs.get(model_choice, model_configs['bert'])
    
    print(f"Selected model: {config['description']}")
    print(f"Model name: {config['model_name']}")
    
    return config['model_name']

# =============================================================================
# STEP 2: CONTACT REPORT PREPROCESSING PIPELINE
# =============================================================================

class ContactReportDataset(Dataset):
    """PyTorch Dataset for contact reports"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ContactReportPreprocessor:
    """Preprocessing pipeline for contact reports"""
    
    def __init__(self, contact_reports_df):
        self.df = contact_reports_df.copy()
        
    def clean_text(self, text):
        """Clean contact report text"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Basic cleaning
        text = text.strip()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_labels(self):
        """Create numerical labels from outcome categories"""
        # Map your existing Outcome_Category to numerical labels
        label_mapping = {
            'Positive': 0,
            'Negative': 1,
            'Unresponsive': 2
        }
        
        self.df['label'] = self.df['Outcome_Category'].map(label_mapping)
        
        # Handle any unmapped values
        self.df['label'] = self.df['label'].fillna(0).astype(int)
        
        return label_mapping
    
    def prepare_data(self):
        """Prepare cleaned text and labels"""
        print("Preprocessing contact reports...")
        
        # Clean text
        self.df['cleaned_text'] = self.df['Report_Text'].apply(self.clean_text)
        
        # Create labels
        label_mapping = self.create_labels()
        
        # Remove empty texts
        self.df = self.df[self.df['cleaned_text'].str.len() > 10].reset_index(drop=True)
        
        print(f"Prepared {len(self.df):,} contact reports")
        print(f"Label distribution:")
        for outcome, label in label_mapping.items():
            count = (self.df['label'] == label).sum()
            print(f"  {outcome} ({label}): {count:,} ({count/len(self.df)*100:.1f}%)")
        
        return self.df['cleaned_text'].values, self.df['label'].values, label_mapping

def create_data_loaders(texts, labels, tokenizer, batch_size=16, max_length=128, test_size=0.2):
    """Create train/val/test data loaders"""
    print(f"Creating data loaders with batch size {batch_size}...")
    
    # Split into train/temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=test_size*2, random_state=42, stratify=labels
    )
    
    # Split temp into val/test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Create datasets
    train_dataset = ContactReportDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = ContactReportDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = ContactReportDataset(X_test, y_test, tokenizer, max_length)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Train: {len(train_dataset):,} samples")
    print(f"Val: {len(val_dataset):,} samples")
    print(f"Test: {len(test_dataset):,} samples")
    
    return train_loader, val_loader, test_loader, (X_test, y_test)

# =============================================================================
# STEP 3: FINE-TUNING STRATEGY AND MODEL ARCHITECTURE
# =============================================================================

class ContactReportClassifier(nn.Module):
    """
    BERT/RoBERTa classifier with custom head for contact reports
    """
    
    def __init__(self, model_name, num_labels, dropout=0.3, freeze_base=False):
        super(ContactReportClassifier, self).__init__()
        
        # Load pre-trained model
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Optionally freeze base model for faster training
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Get hidden size from model config
        hidden_size = self.transformer.config.hidden_size
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # Store for attention visualization
        self.attention_weights = None
        
    def forward(self, input_ids, attention_mask, output_attentions=False):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Store attention weights for visualization
        if output_attentions:
            self.attention_weights = outputs.attentions
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits, outputs

# =============================================================================
# STEP 4: TRAINING PIPELINE WITH DOMAIN ADAPTATION
# =============================================================================

class ContactReportTrainer:
    """Training pipeline for contact report classification"""
    
    def __init__(self, model, device, learning_rate=2e-5, warmup_steps=500):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with weight decay (AdamW is standard for transformers)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            eps=1e-8,
            weight_decay=0.01
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        self.warmup_steps = warmup_steps
        self.scheduler = None
    
    def setup_scheduler(self, num_training_steps):
        """Setup learning rate scheduler with warmup"""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits, _ = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, all_predictions, all_labels, all_probs
    
    def train(self, train_loader, val_loader, epochs=5, early_stopping_patience=2):
        """Complete training loop"""
        print(f"Training for {epochs} epochs...")
        
        # Setup scheduler
        num_training_steps = len(train_loader) * epochs
        self.setup_scheduler(num_training_steps)
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, _, _, _ = self.evaluate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_contact_classifier.pt')
                print("Saved best model!")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_contact_classifier.pt'))
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
        
        return best_val_acc
    
    def plot_training_curves(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss
        axes[0].plot(self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(self.val_losses, label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.train_accs, label='Train Acc', marker='o')
        axes[1].plot(self.val_accs, label='Val Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# =============================================================================
# STEP 5: ATTENTION VISUALIZATION SYSTEM
# =============================================================================

class AttentionVisualizer:
    """Visualize attention mechanisms in transformer models"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def get_attention_weights(self, text, layer=-1, head=0):
        """Extract attention weights for a specific text"""
        self.model.eval()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Forward pass with attention output
        with torch.no_grad():
            logits, outputs = self.model(input_ids, attention_mask, output_attentions=True)
        
        # Get attention weights from specified layer and head
        attention = outputs.attentions[layer][0, head].cpu().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Find actual length (before padding)
        actual_length = attention_mask.sum().item()
        
        return attention[:actual_length, :actual_length], tokens[:actual_length]
    
    def visualize_attention(self, text, layer=-1, head=0, figsize=(12, 10)):
        """Visualize attention heatmap"""
        attention, tokens = self.get_attention_weights(text, layer, head)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        ax.set_title(f'Attention Weights - Layer {layer}, Head {head}')
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def highlight_important_words(self, text, layer=-1, threshold=0.1):
        """Highlight words with high attention to [CLS] token"""
        attention, tokens = self.get_attention_weights(text, layer, head=0)
        
        # Get attention from [CLS] token (first token) to all others
        cls_attention = attention[0, :]
        
        print(f"Important words (attention > {threshold}):")
        print("-" * 60)
        
        important_words = []
        for token, weight in zip(tokens, cls_attention):
            if weight > threshold and token not in ['[CLS]', '[SEP]', '[PAD]']:
                important_words.append((token, weight))
                print(f"  {token:20s}: {weight:.4f}")
        
        return important_words

# =============================================================================
# STEP 6: EMBEDDING EXTRACTION PIPELINE
# =============================================================================

class EmbeddingExtractor:
    """Extract and analyze embeddings from trained model"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def extract_embeddings(self, texts, batch_size=32):
        """Extract [CLS] embeddings for a list of texts"""
        self.model.eval()
        
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                _, outputs = self.model(input_ids, attention_mask)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        
        print(f"Extracted embeddings shape: {all_embeddings.shape}")
        return all_embeddings
    
    def visualize_embeddings(self, embeddings, labels, label_names=None):
        """Visualize embeddings using t-SNE"""
        from sklearn.manifold import TSNE
        
        print("Reducing dimensionality with t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = label_names[label] if label_names else f'Class {label}'
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=label_name,
                alpha=0.6,
                s=50
            )
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Contact Report Embeddings Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def analyze_embedding_clusters(self, embeddings, labels):
        """Analyze clustering quality of embeddings"""
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        
        print("Embedding Clustering Analysis:")
        print("-" * 60)
        
        # Silhouette score (higher is better, range: -1 to 1)
        silhouette = silhouette_score(embeddings, labels)
        print(f"Silhouette Score: {silhouette:.4f}")
        
        # Davies-Bouldin score (lower is better)
        db_score = davies_bouldin_score(embeddings, labels)
        print(f"Davies-Bouldin Score: {db_score:.4f}")
        
        # Intra-class and inter-class distances
        unique_labels = np.unique(labels)
        
        print("\nClass-wise statistics:")
        for label in unique_labels:
            class_embeddings = embeddings[labels == label]
            class_center = class_embeddings.mean(axis=0)
            
            # Average intra-class distance
            intra_dist = np.linalg.norm(class_embeddings - class_center, axis=1).mean()
            print(f"  Class {label} - Intra-class distance: {intra_dist:.4f}")

# =============================================================================
# STEP 7: COMPREHENSIVE EVALUATION AND ANALYSIS
# =============================================================================

def comprehensive_evaluation(trainer, test_loader, label_mapping):
    """Perform comprehensive evaluation on test set"""
    print("=" * 60)
    print("COMPREHENSIVE TEST SET EVALUATION")
    print("=" * 60)
    
    # Get predictions
    test_loss, test_acc, predictions, true_labels, probs = trainer.evaluate(test_loader)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Classification report
    label_names = {v: k for k, v in label_mapping.items()}
    target_names = [label_names[i] for i in sorted(label_names.keys())]
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title('Confusion Matrix - Contact Report Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # ROC-AUC for multi-class
    try:
        probs_array = np.array(probs)
        auc_scores = {}
        for i, name in enumerate(target_names):
            if len(np.unique(true_labels)) > 2:
                # One-vs-rest AUC
                binary_labels = (np.array(true_labels) == i).astype(int)
                auc = roc_auc_score(binary_labels, probs_array[:, i])
                auc_scores[name] = auc
                print(f"AUC for {name}: {auc:.4f}")
            else:
                auc = roc_auc_score(true_labels, probs_array[:, 1])
                print(f"AUC: {auc:.4f}")
                break
    except Exception as e:
        print(f"Could not calculate AUC: {e}")
    
    return test_acc, predictions, true_labels

# =============================================================================
# STEP 8: MAIN EXECUTION PIPELINE
# =============================================================================

def run_bert_pipeline_on_contact_reports(
    data_dir="synthetic_donor_dataset",
    model_choice='bert',
    batch_size=16,
    epochs=5,
    learning_rate=2e-5
):
    """
    Complete BERT/RoBERTa pipeline for contact report analysis
    
    Args:
        data_dir: Directory containing your CSV files
        model_choice: 'bert', 'roberta', or 'distilbert'
        batch_size: Batch size for training (reduce if OOM errors)
        epochs: Number of training epochs
        learning_rate: Learning rate for fine-tuning
    """
    
    print("=" * 80)
    print("BERT/RoBERTa CONTACT REPORT ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Step 1: Setup
    device = setup_transformer_environment()
    model_name = select_model(model_choice)
    
    # Step 2: Load data
    print("\n" + "=" * 60)
    print("LOADING CONTACT REPORTS")
    print("=" * 60)
    
    try:
        # Try to load improved data first, fallback to original
        try:
            contact_reports_df = pd.read_csv(f"{data_dir}/contact_reports_improved.csv")
            print(f"Loaded {len(contact_reports_df):,} improved contact reports")
        except FileNotFoundError:
            contact_reports_df = pd.read_csv(f"{data_dir}/contact_reports.csv")
            print(f"Loaded {len(contact_reports_df):,} original contact reports")
    except FileNotFoundError:
        print(f"Error: Could not find contact_reports.csv or contact_reports_improved.csv in {data_dir}")
        return None
    
    # Step 3: Preprocess
    print("\n" + "=" * 60)
    print("PREPROCESSING CONTACT REPORTS")
    print("=" * 60)
    
    preprocessor = ContactReportPreprocessor(contact_reports_df)
    texts, labels, label_mapping = preprocessor.prepare_data()
    
    # Step 4: Create tokenizer and data loaders
    print("\n" + "=" * 60)
    print("CREATING DATA LOADERS")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, val_loader, test_loader, (X_test, y_test) = create_data_loaders(
        texts, labels, tokenizer, batch_size=batch_size
    )
    
    # Step 5: Initialize model
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)
    
    num_labels = len(label_mapping)
    model = ContactReportClassifier(
        model_name=model_name,
        num_labels=num_labels,
        dropout=0.3,
        freeze_base=False  # Set to True for faster training with frozen base
    )
    
    print(f"Model initialized with {num_labels} output classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Step 6: Train
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    trainer = ContactReportTrainer(model, device, learning_rate=learning_rate)
    best_val_acc = trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Step 7: Evaluate
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    
    test_acc, predictions, true_labels = comprehensive_evaluation(
        trainer, test_loader, label_mapping
    )
    
    # Step 8: Attention visualization
    print("\n" + "=" * 60)
    print("ATTENTION VISUALIZATION")
    print("=" * 60)
    
    visualizer = AttentionVisualizer(model, tokenizer, device)
    
    # Visualize attention for a sample text
    sample_text = X_test[0]
    print(f"Sample text: {sample_text[:100]}...")
    
    # Visualize attention
    visualizer.visualize_attention(sample_text, layer=-1, head=0)
    
    # Highlight important words
    important_words = visualizer.highlight_important_words(sample_text, layer=-1, threshold=0.1)
    
    # Step 9: Extract embeddings
    print("\n" + "=" * 60)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 60)
    
    extractor = EmbeddingExtractor(model, tokenizer, device)
    test_embeddings = extractor.extract_embeddings(X_test.tolist())
    
    # Visualize embeddings
    label_names = {v: k for k, v in label_mapping.items()}
    extractor.visualize_embeddings(test_embeddings, y_test, label_names)
    
    # Analyze clustering
    extractor.analyze_embedding_clusters(test_embeddings, y_test)
    
    # Step 10: Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    results = {
        'model_name': model_name,
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'label_mapping': label_mapping,
        'num_train_samples': len(train_loader.dataset),
        'num_test_samples': len(test_loader.dataset)
    }
    
    # Save model and results
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
        'label_mapping': label_mapping
    }, 'contact_classifier_final.pt')
    
    print("Model and results saved!")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return {
        'model': model,
        'trainer': trainer,
        'tokenizer': tokenizer,
        'visualizer': visualizer,
        'extractor': extractor,
        'results': results,
        'test_embeddings': test_embeddings
    }

# =============================================================================
# STEP 9: ADVANCED ANALYSIS FUNCTIONS
# =============================================================================

def analyze_misclassifications(model, tokenizer, device, X_test, y_test, predictions, label_mapping):
    """Analyze misclassified examples"""
    print("=" * 60)
    print("ANALYZING MISCLASSIFICATIONS")
    print("=" * 60)
    
    label_names = {v: k for k, v in label_mapping.items()}
    
    # Find misclassified examples
    misclassified_idx = np.where(np.array(predictions) != y_test)[0]
    
    print(f"Total misclassified: {len(misclassified_idx)} ({len(misclassified_idx)/len(y_test)*100:.1f}%)")
    
    # Show some examples
    print("\nSample Misclassifications:")
    print("-" * 60)
    
    for i in misclassified_idx[:5]:  # Show first 5
        text = X_test[i]
        true_label = label_names[y_test[i]]
        pred_label = label_names[predictions[i]]
        
        print(f"\nText: {text[:100]}...")
        print(f"True: {true_label} | Predicted: {pred_label}")
        print("-" * 60)

def compare_outcomes_by_donor_type(contact_reports_df, donors_df, predictions):
    """Analyze how predictions vary by donor characteristics"""
    print("=" * 60)
    print("ANALYZING PREDICTIONS BY DONOR TYPE")
    print("=" * 60)
    
    # Merge contact reports with donor data
    merged = contact_reports_df.merge(donors_df[['ID', 'Primary_Constituent_Type', 'Rating', 'Lifetime_Giving']], 
                                      left_on='Donor_ID', right_on='ID', how='left')
    
    # Add predictions to first N rows
    merged['Predicted_Outcome'] = None
    merged.loc[:len(predictions)-1, 'Predicted_Outcome'] = predictions
    
    # Analyze by constituent type
    print("\nPredictions by Constituent Type:")
    outcome_by_type = merged.groupby(['Primary_Constituent_Type', 'Predicted_Outcome']).size().unstack(fill_value=0)
    print(outcome_by_type)
    
    # Analyze by giving level
    merged['Giving_Category'] = pd.cut(merged['Lifetime_Giving'], 
                                       bins=[0, 1000, 10000, 100000, float('inf')],
                                       labels=['Non-Donor', 'Small', 'Medium', 'Major'])
    
    print("\nPredictions by Giving Level:")
    outcome_by_giving = merged.groupby(['Giving_Category', 'Predicted_Outcome']).size().unstack(fill_value=0)
    print(outcome_by_giving)

def extract_key_phrases_by_outcome(texts, labels, label_mapping, top_n=10):
    """Extract most distinctive phrases for each outcome"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print("=" * 60)
    print("KEY PHRASES BY OUTCOME")
    print("=" * 60)
    
    label_names = {v: k for k, v in label_mapping.items()}
    
    for label_id, label_name in label_names.items():
        print(f"\n{label_name} Contact Reports:")
        print("-" * 40)
        
        # Get texts for this label
        label_texts = [texts[i] for i in range(len(texts)) if labels[i] == label_id]
        
        if len(label_texts) < 5:
            print("Not enough samples")
            continue
        
        # TF-IDF to find distinctive phrases
        vectorizer = TfidfVectorizer(max_features=50, ngram_range=(2, 3), stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(label_texts)
        
        # Get top features
        feature_names = vectorizer.get_feature_names_out()
        avg_tfidf = tfidf_matrix.mean(axis=0).A1
        top_indices = avg_tfidf.argsort()[-top_n:][::-1]
        
        print("Top distinctive phrases:")
        for idx in top_indices:
            print(f"  - {feature_names[idx]}")

# =============================================================================
# STEP 10: INTEGRATION WITH GNN PIPELINE
# =============================================================================

def create_multimodal_features(bert_embeddings, gnn_embeddings):
    """Combine BERT text embeddings with GNN graph embeddings"""
    print("=" * 60)
    print("CREATING MULTIMODAL FEATURES")
    print("=" * 60)
    
    # Ensure same number of samples
    min_samples = min(len(bert_embeddings), len(gnn_embeddings))
    
    bert_subset = bert_embeddings[:min_samples]
    gnn_subset = gnn_embeddings[:min_samples]
    
    # Concatenate embeddings
    multimodal_embeddings = np.hstack([bert_subset, gnn_subset])
    
    print(f"BERT embeddings shape: {bert_subset.shape}")
    print(f"GNN embeddings shape: {gnn_subset.shape}")
    print(f"Multimodal embeddings shape: {multimodal_embeddings.shape}")
    
    return multimodal_embeddings

class MultimodalClassifier(nn.Module):
    """Classifier that combines text and graph features"""
    
    def __init__(self, bert_dim, gnn_dim, num_classes, dropout=0.3):
        super(MultimodalClassifier, self).__init__()
        
        combined_dim = bert_dim + gnn_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, bert_features, gnn_features):
        combined = torch.cat([bert_features, gnn_features], dim=1)
        return self.classifier(combined)

# =============================================================================
# USAGE EXAMPLES AND BEST PRACTICES
# =============================================================================

def quick_start_example():
    """Quick start example for running the pipeline"""
    print("""
    ===================================================================
    QUICK START GUIDE - BERT/RoBERTa Contact Report Analysis
    ===================================================================
    
    1. BASIC USAGE (Recommended for most users):
    
    results = run_bert_pipeline_on_contact_reports(
        data_dir="synthetic_donor_dataset",
        model_choice='distilbert',  # Fastest option
        batch_size=16,
        epochs=3,
        learning_rate=2e-5
    )
    
    2. HIGH PERFORMANCE (If you have GPU with 8GB+ memory):
    
    results = run_bert_pipeline_on_contact_reports(
        data_dir="synthetic_donor_dataset",
        model_choice='roberta',     # Best performance
        batch_size=32,
        epochs=5,
        learning_rate=2e-5
    )
    
    3. CPU/LOW MEMORY (If running on CPU or limited memory):
    
    results = run_bert_pipeline_on_contact_reports(
        data_dir="synthetic_donor_dataset",
        model_choice='distilbert',
        batch_size=8,               # Smaller batch
        epochs=3,
        learning_rate=3e-5
    )
    
    4. ACCESS RESULTS:
    
    model = results['model']
    trainer = results['trainer']
    tokenizer = results['tokenizer']
    visualizer = results['visualizer']
    extractor = results['extractor']
    embeddings = results['test_embeddings']
    
    5. ANALYZE SPECIFIC EXAMPLES:
    
    # Visualize attention for custom text
    custom_text = "Met with donor to discuss planned giving..."
    visualizer.visualize_attention(custom_text)
    
    # Extract embeddings for new texts
    new_texts = ["text1", "text2", "text3"]
    new_embeddings = extractor.extract_embeddings(new_texts)
    
    ===================================================================
    TROUBLESHOOTING
    ===================================================================
    
    ERROR: CUDA out of memory
    SOLUTION: Reduce batch_size to 8 or 4
    
    ERROR: Slow training on CPU
    SOLUTION: Use 'distilbert' model and reduce epochs to 2-3
    
    ERROR: Low accuracy (<60%)
    SOLUTION: Check class imbalance, increase epochs, adjust learning rate
    
    ERROR: Model not improving
    SOLUTION: Try different learning rates (1e-5, 3e-5, 5e-5)
    
    ===================================================================
    EXPECTED PERFORMANCE
    ===================================================================
    
    With your 50K synthetic dataset (~33K contact reports):
    
    DistilBERT (3 epochs): 
        - Training time: 15-20 min (GPU) / 1-2 hours (CPU)
        - Expected accuracy: 80-85%
        - Expected AUC: 0.85-0.90
    
    BERT-base (5 epochs):
        - Training time: 30-40 min (GPU) / 3-4 hours (CPU)
        - Expected accuracy: 82-87%
        - Expected AUC: 0.87-0.92
    
    RoBERTa-base (5 epochs):
        - Training time: 35-45 min (GPU) / 3-4 hours (CPU)
        - Expected accuracy: 83-88%
        - Expected AUC: 0.88-0.93
    
    ===================================================================
    """)

def hyperparameter_recommendations():
    """Guide for hyperparameter tuning"""
    print("""
    ===================================================================
    HYPERPARAMETER TUNING GUIDE
    ===================================================================
    
    1. LEARNING RATE (Most important):
       - Default: 2e-5 (good starting point)
       - Try: [1e-5, 2e-5, 3e-5, 5e-5]
       - Higher LR if: Model converges too slowly
       - Lower LR if: Training loss oscillates
    
    2. BATCH SIZE:
       - Default: 16 (good balance)
       - Larger (32-64): Faster training, more memory
       - Smaller (8-4): Less memory, more stable gradients
       - Rule: Largest that fits in memory
    
    3. EPOCHS:
       - Default: 3-5 epochs
       - More if: Validation accuracy still improving
       - Less if: Model overfits (train >> val accuracy)
       - Use early stopping: patience=2
    
    4. DROPOUT:
       - Default: 0.3
       - Increase (0.4-0.5): If overfitting
       - Decrease (0.1-0.2): If underfitting
    
    5. MAX_LENGTH:
       - Default: 128 tokens
       - Your contact reports are short, 128 is fine
       - Increase to 256 if you have longer texts
       - Longer = more memory, slower training
    
    6. WARMUP_STEPS:
       - Default: 500 steps
       - Formula: 10% of total training steps
       - Helps stability in early training
    
    7. WEIGHT_DECAY:
       - Default: 0.01
       - Regularization parameter
       - Increase if overfitting
    
    ===================================================================
    TUNING STRATEGY
    ===================================================================
    
    Phase 1: Find best learning rate
        - Fix all other params
        - Try: [1e-5, 2e-5, 3e-5, 5e-5]
        - Pick LR with best validation accuracy
    
    Phase 2: Optimize batch size
        - Use best LR from Phase 1
        - Try largest batch that fits memory
        - Balance speed vs accuracy
    
    Phase 3: Fine-tune epochs and dropout
        - Use best LR and batch size
        - Adjust epochs based on convergence
        - Tune dropout if over/underfitting
    
    ===================================================================
    """)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("BERT/RoBERTa Contact Report Analysis Pipeline")
    print("=" * 80)
    
    # Show usage guide
    quick_start_example()
    
    # Run the actual pipeline:
    
    results = run_bert_pipeline_on_contact_reports(
        data_dir="synthetic_donor_dataset",
        model_choice='distilbert',  # Start with fastest model
        batch_size=16,
        epochs=5,  # Increased epochs for more realistic training
        learning_rate=2e-5
    )
    
    # Access components
    model = results['model']
    visualizer = results['visualizer']
    extractor = results['extractor']
    
    # Example: Analyze a specific contact report
    sample_text = "Met with donor about planned giving. Very interested in legacy gift."
    visualizer.visualize_attention(sample_text)
    important_words = visualizer.highlight_important_words(sample_text)
    
    
    print("\n" + "=" * 80)
    print("Pipeline completed! Check results above.")
    print("=" * 80)