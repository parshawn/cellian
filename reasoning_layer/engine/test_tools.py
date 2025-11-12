"""Test-mode tools that don't require DataLoader."""
import random
import numpy as np
from .registry import Tool
from .embedding_utils import embedding_to_dict


def test_perturbation_embedding_fn(args: dict) -> dict:
    """
    Test-mode perturbation embedding tool.
    Returns embedding from args if provided, otherwise generates a test embedding.
    """
    perturbation_name = args.get("perturbation_name", "")
    embedding = args.get("embedding")  # Can be provided directly in test mode
    
    if embedding is not None:
        # Convert to numpy array if needed
        if isinstance(embedding, list):
            embedding_arr = np.array(embedding)
        else:
            embedding_arr = np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding
        
        # Convert to dictionary format
        emb_dict = embedding_to_dict(embedding_arr)
        emb_dict["perturbation_name"] = perturbation_name
        emb_dict["aligned"] = True
        emb_dict["test_mode"] = True
        # Also store raw embedding as list for easy access by state.predict
        emb_dict["embedding"] = embedding_arr.tolist() if hasattr(embedding_arr, 'tolist') else list(embedding_arr)
        return emb_dict
    
    # Generate test embedding if not provided
    if not perturbation_name:
        perturbation_name = "TEST_GENE"
    
    # Generate deterministic embedding based on perturbation name
    rng = np.random.RandomState(hash(perturbation_name) % 2**32)
    embedding = rng.randn(1024).astype(np.float32)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    
    emb_dict = embedding_to_dict(embedding)
    emb_dict["perturbation_name"] = perturbation_name
    emb_dict["aligned"] = True
    emb_dict["test_mode"] = True
    # Also store raw embedding as list for easy access by state.predict
    emb_dict["embedding"] = embedding.tolist()
    return emb_dict


def test_state_predict_fn(args: dict) -> dict:
    """
    Test-mode STATE predict tool.
    Uses genes from sample or generates test genes.
    """
    target = args.get("target", "")
    context = args.get("context", {})
    embedding = args.get("embedding")
    test_genes = args.get("test_genes")  # Genes from test sample
    
    # Use test genes if provided, otherwise use default
    if test_genes:
        genes = test_genes
    else:
        genes = ["STAT1", "IRF1", "CXCL10", "HLA-A", "B2M", "SOCS1", "IFIT1"]
    
    # Generate deterministic predictions based on embedding
    if embedding is not None:
        # Handle different embedding formats
        if isinstance(embedding, list):
            embedding_arr = np.array(embedding)
        elif isinstance(embedding, dict):
            # Handle embedding_dict format
            if "values" in embedding:
                embedding_arr = np.array(embedding["values"])
            elif "embedding" in embedding:
                embedding_arr = np.array(embedding["embedding"])
            else:
                embedding_arr = np.array(list(embedding.values()))
        elif isinstance(embedding, np.ndarray):
            embedding_arr = embedding
        else:
            embedding_arr = np.array(embedding)
        
        if len(embedding_arr) > 0:
            # Use embedding hash as seed for reproducibility
            seed = int(abs(embedding_arr[0] * 1000)) % 10000
            rng = random.Random(seed)
        else:
            rng = random.Random(7)
    else:
        rng = random.Random(7)
    
    # Predict RNA delta
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
        "embedding_aligned": True,
        "test_mode": True
    }


# Test-mode tools
test_perturbation_embedding = Tool(
    "perturbation.get_embedding",
    "Get test-mode perturbation embedding (Node 1)",
    test_perturbation_embedding_fn
)

test_state_predict = Tool(
    "state.predict",
    "Test-mode RNA prediction (Node 2)",
    test_state_predict_fn
)

