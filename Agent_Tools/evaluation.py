"""
Evaluation module for comparing predictions against ground truth.

This module provides functions to evaluate predictions using R2 score
and other metrics, similar to the evaluation in the training notebook.
"""

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


def calculate_r2_metrics(y_true, y_pred, per_gene=False):
    """
    Calculate R2 score metrics.
    
    Args:
        y_true: Ground truth values (cells × features or flattened)
        y_pred: Predicted values (cells × features or flattened)
        per_gene: If True, calculate per-gene R2 scores
    
    Returns:
        Dict with R2 metrics
    """
    # Flatten if needed
    if y_true.ndim > 1:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
    else:
        y_true_flat = y_true
        y_pred_flat = y_pred
    
    # Remove NaN values
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return {'r2': np.nan, 'mean_r2': np.nan, 'median_r2': np.nan}
    
    # Overall R2
    overall_r2 = r2_score(y_true_clean, y_pred_clean)
    
    metrics = {'r2': overall_r2}
    
    # Per-gene R2 if requested and data is 2D
    if per_gene and y_true.ndim == 2:
        per_gene_r2 = []
        for i in range(y_true.shape[1]):
            true_col = y_true[:, i]
            pred_col = y_pred[:, i]
            mask = ~(np.isnan(true_col) | np.isnan(pred_col))
            if mask.sum() > 0:
                r2 = r2_score(true_col[mask], pred_col[mask])
                per_gene_r2.append(r2)
        
        if len(per_gene_r2) > 0:
            metrics['mean_r2'] = np.mean(per_gene_r2)
            metrics['median_r2'] = np.median(per_gene_r2)
            metrics['per_gene_r2'] = per_gene_r2
        else:
            metrics['mean_r2'] = np.nan
            metrics['median_r2'] = np.nan
    
    return metrics


def calculate_correlation_metrics(y_true, y_pred):
    """
    Calculate Pearson correlation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dict with correlation metrics
    """
    # Flatten if needed
    if y_true.ndim > 1:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
    else:
        y_true_flat = y_true
        y_pred_flat = y_pred
    
    # Remove NaN values
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) < 2:
        return {'pearson_r': np.nan, 'pearson_p': np.nan}
    
    # Pearson correlation
    r, p = pearsonr(y_true_clean, y_pred_clean)
    
    return {'pearson_r': r, 'pearson_p': p}


def calculate_error_metrics(y_true, y_pred):
    """
    Calculate error metrics (MSE, RMSE, MAE).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dict with error metrics
    """
    # Flatten if needed
    if y_true.ndim > 1:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
    else:
        y_true_flat = y_true
        y_pred_flat = y_pred
    
    # Remove NaN values
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan}
    
    # Calculate metrics
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae}


def evaluate_predictions(y_true, y_pred, feature_names=None, per_feature=False):
    """
    Comprehensive evaluation of predictions against ground truth.
    
    Args:
        y_true: Ground truth values (cells × features or array)
        y_pred: Predicted values (cells × features or array)
        feature_names: Names of features (for per-feature evaluation)
        per_feature: If True, calculate per-feature metrics
    
    Returns:
        Dict with all metrics
    """
    print(f"\n  Evaluating predictions...")
    print(f"    True shape: {y_true.shape if hasattr(y_true, 'shape') else 'N/A'}")
    print(f"    Pred shape: {y_pred.shape if hasattr(y_pred, 'shape') else 'N/A'}")
    
    # Convert to numpy if needed
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    
    # Ensure same shape
    if y_true.shape != y_pred.shape:
        print(f"    Warning: Shape mismatch, aligning...")
        # Try to align if possible
        if isinstance(y_true, pd.DataFrame) and isinstance(y_pred, pd.DataFrame):
            common_index = y_true.index.intersection(y_pred.index)
            common_cols = y_true.columns.intersection(y_pred.columns)
            y_true = y_true.loc[common_index, common_cols].values
            y_pred = y_pred.loc[common_index, common_cols].values
        else:
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(y_true.shape, y_pred.shape))
            y_true = y_true[:min_shape[0], :min_shape[1]]
            y_pred = y_pred[:min_shape[0], :min_shape[1]]
    
    # Calculate all metrics
    metrics = {}
    
    # R2 metrics
    r2_metrics = calculate_r2_metrics(y_true, y_pred, per_gene=per_feature)
    metrics.update(r2_metrics)
    
    # Correlation metrics
    corr_metrics = calculate_correlation_metrics(y_true, y_pred)
    metrics.update(corr_metrics)
    
    # Error metrics
    error_metrics = calculate_error_metrics(y_true, y_pred)
    metrics.update(error_metrics)
    
    # Print summary
    print(f"    ✓ R2 Score: {metrics['r2']:.4f}")
    if 'mean_r2' in metrics and not np.isnan(metrics['mean_r2']):
        print(f"    ✓ Mean R2 (per feature): {metrics['mean_r2']:.4f}")
        print(f"    ✓ Median R2 (per feature): {metrics['median_r2']:.4f}")
    print(f"    ✓ Pearson R: {metrics['pearson_r']:.4f}")
    print(f"    ✓ RMSE: {metrics['rmse']:.4f}")
    print(f"    ✓ MAE: {metrics['mae']:.4f}")
    
    return metrics


def evaluate_by_perturbation(y_true, y_pred, perturbations, feature_names=None):
    """
    Evaluate predictions grouped by perturbation.
    
    Args:
        y_true: Ground truth values (cells × features)
        y_pred: Predicted values (cells × features)
        perturbations: Perturbation labels for each cell (array-like)
        feature_names: Names of features
    
    Returns:
        Dict with per-perturbation metrics and summary
    """
    print(f"\n  Evaluating by perturbation...")
    
    # Convert to numpy if needed
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    
    perturbations = np.array(perturbations)
    unique_perts = np.unique(perturbations)
    
    print(f"    Found {len(unique_perts)} unique perturbations")
    
    per_pert_metrics = {}
    
    for pert in unique_perts:
        mask = perturbations == pert
        if mask.sum() == 0:
            continue
        
        pert_true = y_true[mask]
        pert_pred = y_pred[mask]
        
        # Calculate metrics for this perturbation
        pert_metrics = evaluate_predictions(pert_true, pert_pred, per_feature=False)
        per_pert_metrics[pert] = pert_metrics
    
    # Create summary DataFrame
    if per_pert_metrics:
        metrics_df = pd.DataFrame.from_dict(per_pert_metrics, orient='index')
        metrics_df = metrics_df.sort_values(by='r2', ascending=False)
        
        print(f"\n    Per-perturbation R2 scores:")
        for pert, row in metrics_df.head(10).iterrows():
            print(f"      {pert}: {row['r2']:.4f}")
        
        if len(metrics_df) > 10:
            print(f"      ... and {len(metrics_df) - 10} more")
        
        print(f"\n    Summary statistics:")
        print(f"      Mean R2: {metrics_df['r2'].mean():.4f}")
        print(f"      Median R2: {metrics_df['r2'].median():.4f}")
        print(f"      Std R2: {metrics_df['r2'].std():.4f}")
        
        return {
            'per_perturbation': per_pert_metrics,
            'summary': metrics_df,
            'mean_r2': metrics_df['r2'].mean(),
            'median_r2': metrics_df['r2'].median()
        }
    else:
        return {'per_perturbation': {}, 'summary': None}

