"""Utilities for working with pre-computed aligned embeddings."""
import numpy as np
from typing import Dict, Optional, Any
import torch


def embedding_to_dict(emb: Any) -> Dict[str, Any]:
    """
    Convert embedding to dictionary representation.
    Assumes embedding is already aligned and ready to use.
    """
    if emb is None:
        return {"embedding": None, "dim": 0}
    
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    
    if isinstance(emb, np.ndarray):
        return {
            "embedding": emb.tolist(),
            "dim": len(emb),
            "mean": float(emb.mean()),
            "std": float(emb.std())
        }
    
    return {"embedding": None, "dim": 0}


def compute_embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    Assumes embeddings are already in aligned space.
    """
    if emb1 is None or emb2 is None:
        return 0.0
    
    if isinstance(emb1, torch.Tensor):
        emb1 = emb1.detach().cpu().numpy()
    if isinstance(emb2, torch.Tensor):
        emb2 = emb2.detach().cpu().numpy()
    
    # Normalize
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(emb1, emb2) / (norm1 * norm2))
