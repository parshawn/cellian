"""STATE tool: predict RNA from pre-computed aligned embedding (Node 2)."""
import random
import numpy as np
from .registry import Tool


_data_loader = None


def _get_data_loader():
    """Get or create data loader instance (lazy import to avoid circular dependency)."""
    global _data_loader
    if _data_loader is None:
        from .data_loader import DataLoader
        _data_loader = DataLoader()
    return _data_loader


def state_predict_fn(args: dict) -> dict:
    """
    Predict RNA changes from pre-computed aligned perturbation embedding (Node 2).
    Assumes embedding is already aligned and ready to use.
    """
    target = args.get("target", "")
    context = args.get("context", {})
    embedding = args.get("embedding")  # Pre-computed aligned embedding
    
    data_loader = _get_data_loader()
    
    # Get control RNA profile for baseline
    control_rna = data_loader.get_control_rna_profile()
    
    # Use pre-computed aligned embedding directly (no alignment needed)
    # In real implementation, would use STATE model with aligned embedding
    if embedding is not None:
        # Convert embedding if needed
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        # Embedding is already aligned, use directly
        # For mock: use deterministic prediction based on embedding
        rng = random.Random(7)
        if isinstance(embedding, np.ndarray) and len(embedding) > 0:
            # Use embedding hash as seed for reproducibility
            seed = int(abs(embedding[0] * 1000)) % 10000
            rng = random.Random(seed)
    else:
        rng = random.Random(7)
    
    # Get gene set from control profile
    if control_rna:
        genes = list(control_rna.keys())[:100]  # Limit to first 100 genes
    else:
        genes = ["STAT1", "IRF1", "CXCL10", "HLA-A", "B2M", "SOCS1", "IFIT1"]
    
    # Predict RNA delta (mock - replace with real STATE model)
    delta_rna = {gene: rng.uniform(-1.0, 1.0) for gene in genes}
    
    # Top-k up and down
    sorted_genes = sorted(delta_rna.items(), key=lambda x: x[1], reverse=True)
    topk_up = [[gene, val] for gene, val in sorted_genes[:3]]
    topk_down = [[gene, val] for gene, val in sorted(reversed(sorted_genes), key=lambda x: x[1])[:3]]
    
    return {
        "delta_rna": delta_rna,
        "topk_up": topk_up,
        "topk_down": topk_down,
        "embedding_used": embedding is not None,
        "embedding_aligned": True  # Assumed to be already aligned
    }


state_predict = Tool(
    "state.predict",
    "Predict ΔRNA from pre-computed aligned perturbation embedding (Node 2)",
    state_predict_fn
)
