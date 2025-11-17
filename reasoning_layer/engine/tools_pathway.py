"""Pathway tools for finding and traversing pathways."""
from .registry import Tool
from .pathway_loader import PathwayLoader


# Global pathway loader instance (lazy initialization)
_pathway_loader = None


def get_pathway_loader() -> PathwayLoader:
    """Get or create pathway loader instance."""
    global _pathway_loader
    if _pathway_loader is None:
        _pathway_loader = PathwayLoader()
    return _pathway_loader


def pathway_find_affected_fn(args: dict) -> dict:
    """
    Find pathways affected by a perturbation.
    
    Args:
        args: Dictionary with "perturbation" (gene name) and optional "max_hops"
    
    Returns:
        Dictionary with pathway_ids, pathway_names, and n_pathways
    """
    perturbation = args.get("perturbation", "")
    max_hops = args.get("max_hops", 3)
    
    if not perturbation:
        return {
            "pathway_ids": [],
            "pathway_names": [],
            "n_pathways": 0,
            "error": "perturbation is required"
        }
    
    loader = get_pathway_loader()
    pathway_ids = loader.find_affected_pathways(perturbation, max_hops)
    
    pathway_names = []
    for pathway_id in pathway_ids:
        pathway_data = loader.get_pathway(pathway_id)
        if pathway_data:
            pathway_names.append(pathway_data.get("name", pathway_id))
        else:
            pathway_names.append(pathway_id)
    
    return {
        "pathway_ids": pathway_ids,
        "pathway_names": pathway_names,
        "n_pathways": len(pathway_ids),
        "perturbation": perturbation
    }


def pathway_get_genes_fn(args: dict) -> dict:
    """
    Get all genes in specified pathways.
    
    Args:
        args: Dictionary with "pathway_ids" (list of pathway IDs)
    
    Returns:
        Dictionary with genes list and n_genes
    """
    pathway_ids = args.get("pathway_ids", [])
    
    if not pathway_ids:
        return {
            "genes": [],
            "n_genes": 0,
            "error": "pathway_ids is required"
        }
    
    loader = get_pathway_loader()
    genes = loader.get_all_pathway_genes(pathway_ids)
    
    return {
        "genes": sorted(list(genes)),  # Sort for consistency
        "n_genes": len(genes),
        "pathway_ids": pathway_ids
    }


def pathway_traverse_fn(args: dict) -> dict:
    """
    Traverse pathway cascades from a starting gene.
    
    Args:
        args: Dictionary with "start_gene", optional "pathway_id", and "max_hops"
    
    Returns:
        Dictionary with genes list and paths
    """
    start_gene = args.get("start_gene", "")
    pathway_id = args.get("pathway_id", None)
    max_hops = args.get("max_hops", 3)
    
    if not start_gene:
        return {
            "genes": [],
            "paths": [],
            "error": "start_gene is required"
        }
    
    loader = get_pathway_loader()
    result = loader.traverse_pathway(start_gene, pathway_id, max_hops)
    
    return {
        "genes": sorted(result.get("genes", [])),
        "paths": result.get("paths", []),
        "start_gene": start_gene,
        "n_genes": len(result.get("genes", []))
    }


# Register tools
pathway_find_affected = Tool(
    "pathway.find_affected",
    "Find pathways affected by a perturbation",
    pathway_find_affected_fn
)

pathway_get_genes = Tool(
    "pathway.get_genes",
    "Get all genes in specified pathways",
    pathway_get_genes_fn
)

pathway_traverse = Tool(
    "pathway.traverse",
    "Traverse pathway cascades from a starting gene",
    pathway_traverse_fn
)

