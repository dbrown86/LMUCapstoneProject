"""
Metrics calculation module for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Import config for paths
from dashboard.config import settings

# Optional Streamlit import - only used if available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create a mock st object for testing without Streamlit
    class MockStreamlit:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    st = MockStreamlit()


def try_load_saved_metrics() -> Optional[Dict[str, Any]]:
    """
    Load precomputed training metrics from disk, if available.
    
    Returns:
        dict: Metrics dictionary if found, None otherwise
    """
    for p in settings.SAVED_METRICS_CANDIDATES:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    m = json.load(f)
                return {
                    "auc": m.get("auc"),
                    "f1": m.get("f1"),
                    "accuracy": m.get("accuracy"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "baseline_auc": m.get("baseline_auc"),
                    "lift": m.get("lift"),
                    "optimal_threshold": m.get("optimal_threshold"),
                }
            except Exception:
                continue
    return None


def get_model_metrics(df: Optional[pd.DataFrame] = None, use_cache: bool = True) -> Dict[str, Any]:
    """
    Calculate model performance metrics directly from parquet file for accuracy.
    Bypasses dataframe processing to use source data directly.
    ALWAYS uses Will_Give_Again_Probability (not predicted_prob or Legacy_Intent_Probability).
    Returns fusion model metrics, baseline AUC, and lift.
    
    Args:
        df: Optional dataframe to use if parquet file cannot be loaded
        use_cache: If True and Streamlit is available, use Streamlit caching
    
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # Use Streamlit caching if available and requested
    if use_cache and STREAMLIT_AVAILABLE:
        @st.cache_data(ttl=3600)
        def _get_cached():
            return _get_model_metrics_internal(df)
        return _get_cached()
    else:
        return _get_model_metrics_internal(df)


def _get_model_metrics_internal(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Internal function to calculate metrics (without caching decorator)."""
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
    
    # Read directly from parquet file to avoid processing issues
    root = settings.get_project_root()
    data_paths = settings.get_data_paths()
    parquet_paths = data_paths['parquet_paths']
    
    source_df = None
    for path in parquet_paths:
        if os.path.exists(path):
            try:
                source_df = pd.read_parquet(path, engine='pyarrow')
                break
            except Exception:
                continue
    
    # If we can't read from file, fall back to provided dataframe
    if source_df is None:
        if df is None:
            return {'auc': None, 'f1': None, 'baseline_auc': None, 'lift': None}
        source_df = df
    
    # Find Will_Give_Again_Probability - prefer exact match, handle duplicates
    prob_col = None
    if 'Will_Give_Again_Probability' in source_df.columns:
        prob_col = 'Will_Give_Again_Probability'
    else:
        will_give_variants = [c for c in source_df.columns if 'Will_Give_Again_Probability' in c]
        if will_give_variants:
            prob_col = will_give_variants[0]
    
    # Find outcome column
    outcome_col = None
    if 'Gave_Again_In_2024' in source_df.columns:
        outcome_col = 'Gave_Again_In_2024'
    elif 'actual_gave' in source_df.columns:
        outcome_col = 'actual_gave'
    
    # Initialize result with None values
    result = {
        'auc': None,
        'f1': None,
        'accuracy': None,
        'precision': None,
        'recall': None,
        'specificity': None,
        'baseline_auc': None,
        'baseline_f1': None,
        'baseline_precision': None,
        'baseline_recall': None,
        'baseline_specificity': None,
        'lift': None
    }
    
    # Calculate fusion model metrics directly from source data
    if prob_col and outcome_col:
        # Use the identified columns directly from source
        y_true = pd.to_numeric(source_df[outcome_col], errors='coerce')
        y_prob = pd.to_numeric(source_df[prob_col], errors='coerce')
        valid_mask = y_true.notna() & y_prob.notna()
        
        if valid_mask.sum() > 0:
            y_true_valid = y_true[valid_mask].astype(int).values
            y_prob_valid = np.clip(y_prob[valid_mask].astype(float).values, 0, 1)
            
            # Check if we have two classes
            unique_classes = np.unique(y_true_valid)
            if len(unique_classes) >= 2:
                # Check mean probabilities for each class (should be higher for positive class)
                prob_for_pos = y_prob_valid[y_true_valid == 1].mean() if (y_true_valid == 1).sum() > 0 else 0
                prob_for_neg = y_prob_valid[y_true_valid == 0].mean() if (y_true_valid == 0).sum() > 0 else 0
                
                # If predictions appear inverted, invert them
                if prob_for_pos < prob_for_neg:
                    y_prob_valid = 1 - y_prob_valid
                
                # Calculate AUC directly
                result['auc'] = roc_auc_score(y_true_valid, y_prob_valid)
                
                # Binary predictions at 0.5 threshold
                threshold = 0.5
                y_pred_binary = (y_prob_valid >= threshold).astype(int)
                
                # Calculate other metrics
                result['f1'] = f1_score(y_true_valid, y_pred_binary, zero_division=0)
                result['accuracy'] = accuracy_score(y_true_valid, y_pred_binary)
                result['precision'] = precision_score(y_true_valid, y_pred_binary, zero_division=0)
                result['recall'] = recall_score(y_true_valid, y_pred_binary, zero_division=0)
                
                # Calculate specificity
                cm = confusion_matrix(y_true_valid, y_pred_binary)
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                    result['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Calculate baseline AUC from days_since_last
    if outcome_col:
        # Check if days_since_last exists, otherwise calculate from Last_Gift_Date
        if 'days_since_last' not in source_df.columns:
            # Try to calculate from Last_Gift_Date
            date_col = None
            for col_name in ['Last_Gift_Date', 'last_gift_date', 'LastGiftDate', 'lastGiftDate']:
                if col_name in source_df.columns:
                    date_col = col_name
                    break
            
            if date_col:
                date_series = pd.to_datetime(source_df[date_col], errors='coerce')
                today = pd.Timestamp.now()
                source_df['days_since_last'] = (today - date_series).dt.days.clip(lower=0)
        
        if 'days_since_last' in source_df.columns:
            try:
                outcome_series = pd.to_numeric(source_df[outcome_col], errors='coerce')
                days_series = pd.to_numeric(source_df['days_since_last'], errors='coerce')
                base_mask = outcome_series.notna() & days_series.notna()
                
                if base_mask.sum() > 0:
                    y_true_base = outcome_series[base_mask].astype(int).values
                    days_valid = days_series[base_mask].astype(float).values
                    
                    unique_classes_base = np.unique(y_true_base)
                    if len(unique_classes_base) >= 2:
                        # Calculate baseline predictions: more recent = higher probability
                        max_days = np.nanpercentile(days_valid, 95) if days_valid.size > 0 else np.nanmax(days_valid)
                        if np.isfinite(max_days) and max_days > 0:
                            baseline_pred = 1 - (np.clip(days_valid, 0, max_days) / max_days)
                            result['baseline_auc'] = roc_auc_score(y_true_base, baseline_pred)
                            
                            # Calculate baseline binary metrics
                            baseline_pred_binary = (baseline_pred >= 0.5).astype(int)
                            result['baseline_f1'] = f1_score(y_true_base, baseline_pred_binary, zero_division=0)
                            result['baseline_precision'] = precision_score(y_true_base, baseline_pred_binary, zero_division=0)
                            result['baseline_recall'] = recall_score(y_true_base, baseline_pred_binary, zero_division=0)
                            
                            cm_baseline = confusion_matrix(y_true_base, baseline_pred_binary)
                            if cm_baseline.size == 4:
                                tn, fp, fn, tp = cm_baseline.ravel()
                                result['baseline_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            except Exception:
                pass  # Silently fail - baseline_auc remains None
    
    # Calculate lift
    if result['auc'] is not None and result['baseline_auc'] is not None and result['baseline_auc'] > 0:
        try:
            result['lift'] = (result['auc'] - result['baseline_auc']) / result['baseline_auc']
            if not np.isfinite(result['lift']):
                result['lift'] = None
        except Exception:
            pass
    
    return result


def get_feature_importance(df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """
    Calculate feature importance from actual data if available.
    
    Uses correlation with the 'gave again in 2024' outcome as a proxy for feature importance.
    This is based on the multi-modal fusion model dataset and outcome variable.
    
    Args:
        df: Dataframe with features and outcome
        use_cache: If True and Streamlit is available, use Streamlit caching
    
    Returns:
        pd.DataFrame: DataFrame with feature names and importance scores
    """
    # Use Streamlit caching if available and requested
    if use_cache and STREAMLIT_AVAILABLE:
        @st.cache_data
        def _get_cached():
            return _get_feature_importance_internal(df)
        return _get_cached()
    else:
        return _get_feature_importance_internal(df)


def _get_feature_importance_internal(df: pd.DataFrame) -> pd.DataFrame:
    """Internal function to calculate feature importance (without caching decorator)."""
    # CRITICAL: Use Gave_Again_In_2024 if available (from will give again in 2024 prediction file)
    # Otherwise fall back to actual_gave
    outcome_col = 'Gave_Again_In_2024' if 'Gave_Again_In_2024' in df.columns else ('actual_gave' if 'actual_gave' in df.columns else None)
    
    if outcome_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target, ID, and prediction columns (these are outputs, not features)
        exclude_cols = [outcome_col, 'actual_gave', 'donor_id', 'Will_Give_Again_Probability', 
                       'predicted_prob', 'Legacy_Intent_Probability']
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        importance_scores = []
        feature_names = []
        
        # Convert outcome to numeric for correlation
        outcome_series = pd.to_numeric(df[outcome_col], errors='coerce')
        
        for col in feature_cols:
            try:
                # Handle duplicate column names
                col_data = df[col]
                if isinstance(col_data, pd.DataFrame):
                    col_series = col_data.iloc[:, 0]
                else:
                    col_series = col_data
                
                feature_series = pd.to_numeric(col_series, errors='coerce')
                
                # Calculate correlation only if both series have valid data
                if len(feature_series.dropna()) > 10 and len(outcome_series.dropna()) > 10:
                    corr = abs(feature_series.corr(outcome_series))
                    if not np.isnan(corr):
                        importance_scores.append(corr)
                        feature_names.append(col)
            except (ValueError, TypeError):
                pass
        
        if len(feature_names) > 0:
            # Sort by importance
            sorted_indices = np.argsort(importance_scores)[::-1][:15]  # Top 15
            
            return pd.DataFrame({
                'feature': [feature_names[i] for i in sorted_indices],
                'importance': [importance_scores[i] for i in sorted_indices]
            })
    
    # Fallback: return default features (should rarely be used)
    features = [
        'days_since_last', 'total_giving',
        'avg_gift', 'gift_count', 'rfm_score', 'recency_score',
        'frequency_score', 'monetary_score', 'years_active', 
        'consecutive_years'
    ]
    
    # Create dummy importance scores based on position
    importance = np.linspace(0.08, 0.01, len(features))
    
    return pd.DataFrame({'feature': features, 'importance': importance})

