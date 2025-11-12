"""Tool for getting pre-computed aligned perturbation embeddings (Node 1)."""
from .registry import Tool
from .data_loader import DataLoader
from .embedding_utils import embedding_to_dict


# Global data loader instance (lazy initialization)
_data_loader = None


def get_data_loader() -> DataLoader:
    """Get or create data loader instance."""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader


def perturbation_embedding_fn(args: dict) -> dict:
    """
    Get pre-computed aligned perturbation embedding for a given perturbation name.
    This is Node 1 in the 3-node chain.
    Assumes embedding is already aligned and ready to use.
    
    Args:
        args: Dictionary with "perturbation_name" key
    
    Returns:
        Dictionary with "embedding" and metadata
    """
    perturbation_name = args.get("perturbation_name", "")
    
    if not perturbation_name:
        return {
            "embedding": None,
            "perturbation_name": "",
            "error": "perturbation_name is required"
        }
    
    data_loader = get_data_loader()
    embedding = data_loader.get_perturbation_embedding(perturbation_name)
    
    if embedding is None:
        return {
            "embedding": None,
            "perturbation_name": perturbation_name,
            "error": f"Perturbation '{perturbation_name}' not found or embeddings not available",
            "note": "Embeddings should be pre-computed and aligned before using reasoning_layer"
        }
    
    # Convert to dictionary (embedding is already aligned)
    emb_dict = embedding_to_dict(embedding)
    emb_dict["perturbation_name"] = perturbation_name
    emb_dict["aligned"] = True  # Assumed to be already aligned
    
    return emb_dict


perturbation_embedding = Tool(
    "perturbation.get_embedding",
    "Get pre-computed aligned perturbation embedding (Node 1)",
    perturbation_embedding_fn
)
