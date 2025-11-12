"""Test data generator for testing the reasoning engine with arbitrary inputs."""
import random
import numpy as np
from typing import Dict, List, Optional, Any


def generate_embedding(dim: int = 1024, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate an arbitrary embedding vector.
    
    Args:
        dim: Embedding dimension
        seed: Random seed for reproducibility
    
    Returns:
        numpy array of embedding
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Generate normalized embedding
    embedding = rng.randn(dim).astype(np.float32)
    # Normalize to unit vector
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    return embedding


def generate_rna_deltas(genes: List[str], seed: Optional[int] = None, 
                        min_val: float = -1.0, max_val: float = 1.0) -> Dict[str, float]:
    """
    Generate arbitrary RNA delta values for given genes.
    
    Args:
        genes: List of gene names
        seed: Random seed for reproducibility
        min_val: Minimum delta value
        max_val: Maximum delta value
    
    Returns:
        Dictionary mapping gene names to delta values
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    return {gene: rng.uniform(min_val, max_val) for gene in genes}


def generate_protein_deltas(proteins: List[str], seed: Optional[int] = None,
                            min_val: float = -1.0, max_val: float = 1.0) -> Dict[str, float]:
    """
    Generate arbitrary protein delta values for given proteins.
    
    Args:
        proteins: List of protein names
        seed: Random seed for reproducibility
        min_val: Minimum delta value
        max_val: Maximum delta value
    
    Returns:
        Dictionary mapping protein names to delta values
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    return {protein: rng.uniform(min_val, max_val) for protein in proteins}


def generate_test_sample(
    perturbation_name: str = "TEST_GENE",
    perturbation_type: str = "KO",
    cell_line: str = "A375",
    condition: str = "IFNg+",
    genes: Optional[List[str]] = None,
    proteins: Optional[List[str]] = None,
    embedding_dim: int = 1024,
    embedding_seed: Optional[int] = 42,
    rna_seed: Optional[int] = 123,
    protein_seed: Optional[int] = 456,
    obs_rna_seed: Optional[int] = 789,
    obs_protein_seed: Optional[int] = 101112,
    use_same_seed_for_pred_obs: bool = False
) -> Dict[str, Any]:
    """
    Generate a complete test sample with arbitrary data.
    
    Args:
        perturbation_name: Name of perturbation
        perturbation_type: Type of perturbation (KO, KD, OE)
        cell_line: Cell line name
        condition: Condition name
        genes: List of gene names (default: common test genes)
        proteins: List of protein names (default: common test proteins)
        embedding_dim: Embedding dimension
        embedding_seed: Seed for embedding generation
        rna_seed: Seed for predicted RNA deltas
        protein_seed: Seed for predicted protein deltas
        obs_rna_seed: Seed for observed RNA deltas
        obs_protein_seed: Seed for observed protein deltas
        use_same_seed_for_pred_obs: If True, use same seed for pred and obs (for testing perfect matches)
    
    Returns:
        Complete test sample dictionary
    """
    # Default genes and proteins
    if genes is None:
        genes = ["STAT1", "IRF1", "CXCL10", "HLA-A", "B2M", "SOCS1", "IFIT1", 
                 "JAK1", "JAK2", "STAT2", "IFNGR1", "IFNGR2"]
    
    if proteins is None:
        proteins = ["HLA-A", "CD58", "CD274", "CD155"]
    
    # Generate embedding
    embedding = generate_embedding(dim=embedding_dim, seed=embedding_seed)
    
    # Generate predicted RNA deltas
    pred_rna_seed = rna_seed
    pred_rna = generate_rna_deltas(genes, seed=pred_rna_seed)
    
    # Generate predicted protein deltas
    pred_protein_seed = protein_seed
    pred_protein = generate_protein_deltas(proteins, seed=pred_protein_seed)
    
    # Generate observed RNA deltas
    if use_same_seed_for_pred_obs:
        obs_rna = pred_rna.copy()  # Perfect match for testing
    else:
        obs_rna = generate_rna_deltas(genes, seed=obs_rna_seed)
    
    # Generate observed protein deltas
    if use_same_seed_for_pred_obs:
        obs_protein = pred_protein.copy()  # Perfect match for testing
    else:
        obs_protein = generate_protein_deltas(proteins, seed=obs_protein_seed)
    
    # Build sample dictionary
    sample = {
        "context": {
            "cell_line": cell_line,
            "condition": condition
        },
        "perturbation": {
            "target": perturbation_name,
            "type": perturbation_type
        },
        "rna": {
            "obs_delta": obs_rna
        },
        "protein": {
            "panel": proteins,
            "obs_delta": obs_protein
        },
        "embedding": embedding.tolist(),  # Convert to list for JSON serialization
        "test_mode": True,
        "metadata": {
            "embedding_dim": embedding_dim,
            "embedding_seed": embedding_seed,
            "rna_seed": rna_seed,
            "protein_seed": protein_seed,
            "obs_rna_seed": obs_rna_seed,
            "obs_protein_seed": obs_protein_seed,
            "genes": genes,
            "proteins": proteins
        }
    }
    
    return sample


def load_test_sample_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load test sample from JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Test sample dictionary
    """
    import json
    with open(filepath, 'r') as f:
        sample = json.load(f)
    
    # Convert embedding list back to numpy array if present
    if "embedding" in sample and isinstance(sample["embedding"], list):
        sample["embedding"] = np.array(sample["embedding"])
    
    return sample


def save_test_sample_to_json(sample: Dict[str, Any], filepath: str):
    """
    Save test sample to JSON file.
    
    Args:
        sample: Test sample dictionary
        filepath: Path to save JSON file
    """
    import json
    
    # Create a copy and convert numpy arrays to lists
    sample_copy = sample.copy()
    if "embedding" in sample_copy and isinstance(sample_copy["embedding"], np.ndarray):
        sample_copy["embedding"] = sample_copy["embedding"].tolist()
    
    with open(filepath, 'w') as f:
        json.dump(sample_copy, f, indent=2)


def create_test_sample_from_args(
    perturbation: Optional[str] = None,
    genes: Optional[str] = None,
    proteins: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    seed: Optional[int] = None,
    perfect_match: bool = False
) -> Dict[str, Any]:
    """
    Create test sample from command-line arguments.
    
    Args:
        perturbation: Perturbation name (e.g., "JAK1_KO")
        genes: Comma-separated gene names
        proteins: Comma-separated protein names
        embedding_dim: Embedding dimension
        seed: Base seed for all random generation
        perfect_match: If True, predicted and observed will match perfectly
    
    Returns:
        Test sample dictionary
    """
    # Parse perturbation
    if perturbation:
        if "_" in perturbation:
            pert_name, pert_type = perturbation.rsplit("_", 1)
        else:
            pert_name = perturbation
            pert_type = "KO"
    else:
        pert_name = "TEST_GENE"
        pert_type = "KO"
    
    # Parse genes
    gene_list = None
    if genes:
        gene_list = [g.strip() for g in genes.split(",")]
    
    # Parse proteins
    protein_list = None
    if proteins:
        protein_list = [p.strip() for p in proteins.split(",")]
    
    # Use seed for all if provided
    if seed is not None:
        embedding_seed = seed
        rna_seed = seed + 1
        protein_seed = seed + 2
        obs_rna_seed = seed + 3 if not perfect_match else seed + 1
        obs_protein_seed = seed + 4 if not perfect_match else seed + 2
    else:
        embedding_seed = 42
        rna_seed = 123
        protein_seed = 456
        obs_rna_seed = 789 if not perfect_match else 123
        obs_protein_seed = 101112 if not perfect_match else 456
    
    return generate_test_sample(
        perturbation_name=pert_name,
        perturbation_type=pert_type,
        genes=gene_list,
        proteins=protein_list,
        embedding_dim=embedding_dim or 1024,
        embedding_seed=embedding_seed,
        rna_seed=rna_seed,
        protein_seed=protein_seed,
        obs_rna_seed=obs_rna_seed,
        obs_protein_seed=obs_protein_seed,
        use_same_seed_for_pred_obs=perfect_match
    )

