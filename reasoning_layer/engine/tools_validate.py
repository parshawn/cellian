"""Validation tool: compute metrics and edge-sign accuracy."""
from .metrics import pearson, spearman
from .registry import Tool


def validate_all_fn(args: dict) -> dict:
    """
    Compute RNA/Protein metrics and edge-sign accuracy.
    """
    pred_rna = args.get("pred_rna", {})
    obs_rna = args.get("obs_rna", {})
    pred_prot = args.get("pred_prot", {})
    obs_prot = args.get("obs_prot", {})
    path_edges = args.get("path_edges", [])
    
    # RNA Spearman: intersection of genes
    rna_genes = set(pred_rna.keys()) & set(obs_rna.keys())
    if rna_genes:
        rna_pred_vals = [pred_rna[g] for g in sorted(rna_genes)]
        rna_obs_vals = [obs_rna[g] for g in sorted(rna_genes)]
        rna_spearman = spearman(rna_pred_vals, rna_obs_vals)
    else:
        rna_spearman = float("nan")
    
    # Protein Pearson: intersection of markers
    prot_markers = set(pred_prot.keys()) & set(obs_prot.keys())
    if prot_markers:
        prot_pred_vals = [pred_prot[m] for m in sorted(prot_markers)]
        prot_obs_vals = [obs_prot[m] for m in sorted(prot_markers)]
        prot_pearson = pearson(prot_pred_vals, prot_obs_vals)
        
        # MSE
        prot_mse = sum((p - o) ** 2 for p, o in zip(prot_pred_vals, prot_obs_vals)) / len(prot_pred_vals)
    else:
        prot_pearson = float("nan")
        prot_mse = float("nan")
    
    # Edge-sign accuracy
    if path_edges and pred_rna:
        correct = 0
        total = 0
        for edge in path_edges:
            dst = edge.get("dst")
            sign = edge.get("sign", "+")
            if dst in pred_rna:
                total += 1
                pred_val = pred_rna[dst]
                if sign == "+":
                    if pred_val >= 0:
                        correct += 1
                else:  # sign == "-"
                    if pred_val < 0:
                        correct += 1
        
        if total > 0:
            edge_sign_accuracy = correct / total
        else:
            edge_sign_accuracy = float("nan")
    else:
        edge_sign_accuracy = float("nan")
    
    return {
        "rna_spearman": rna_spearman,
        "prot_pearson": prot_pearson,
        "prot_mse": prot_mse,
        "edge_sign_accuracy": edge_sign_accuracy
    }


validate_all = Tool(
    "validate.all",
    "Compute RNA/Protein metrics and edge-sign accuracy",
    validate_all_fn
)

