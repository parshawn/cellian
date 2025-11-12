"""Functions for comparing predicted vs real profiles and calculating error propagation."""
import math
from typing import Dict
from .metrics import pearson, spearman, cosine_similarity


def compare_profiles(
    pred: Dict[str, float],
    real: Dict[str, float],
    metric: str = "pearson"
) -> Dict[str, float]:
    """
    Compare predicted and real profiles.
    
    Args:
        pred: Dictionary mapping feature names to predicted values
        real: Dictionary mapping feature names to real values
        metric: Metric to use ("pearson", "spearman", "cosine", "mse", "mae")
    
    Returns:
        Dictionary with comparison metrics
    """
    common_features = set(pred.keys()) & set(real.keys())
    
    if not common_features:
        return {
            "metric": metric,
            "value": float("nan"),
            "n_features": 0,
            "mse": float("nan"),
            "mae": float("nan")
        }
    
    sorted_features = sorted(common_features)
    pred_vals = [pred[f] for f in sorted_features]
    real_vals = [real[f] for f in sorted_features]
    
    result = {
        "metric": metric,
        "n_features": len(common_features)
    }
    
    if metric == "pearson":
        result["value"] = pearson(pred_vals, real_vals)
    elif metric == "spearman":
        result["value"] = spearman(pred_vals, real_vals)
    elif metric == "cosine":
        result["value"] = cosine_similarity(pred_vals, real_vals)
    else:
        result["value"] = float("nan")
    
    mse = sum((p - r) ** 2 for p, r in zip(pred_vals, real_vals)) / len(pred_vals)
    mae = sum(abs(p - r) for p, r in zip(pred_vals, real_vals)) / len(pred_vals)
    
    result["mse"] = mse
    result["mae"] = mae
    
    return result


def calculate_error_propagation(
    rna_error: float,
    protein_error: float,
    rna_to_protein_error: float = 0.0
) -> Dict[str, float]:
    """
    Calculate error propagation through the 3-node chain.
    
    Args:
        rna_error: Error at RNA prediction step (Node 2)
        protein_error: Error at protein prediction step (Node 3)
        rna_to_protein_error: Error in RNA->Protein translation
    
    Returns:
        Dictionary with error propagation metrics
    """
    total_error = math.sqrt(rna_error ** 2 + protein_error ** 2 + rna_to_protein_error ** 2)
    
    if rna_error > 0:
        amplification = protein_error / rna_error
    else:
        amplification = float("nan")
    
    rna_contribution = rna_error / total_error if total_error > 0 else 0.0
    protein_contribution = protein_error / total_error if total_error > 0 else 0.0
    translation_contribution = rna_to_protein_error / total_error if total_error > 0 else 0.0
    
    return {
        "total_error": total_error,
        "error_amplification": amplification,
        "rna_error": rna_error,
        "protein_error": protein_error,
        "translation_error": rna_to_protein_error,
        "rna_contribution": rna_contribution,
        "protein_contribution": protein_contribution,
        "translation_contribution": translation_contribution
    }

