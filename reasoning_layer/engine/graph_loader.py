"""Graph loader for loading pre-computed or dummy graphs."""
import os
from typing import Optional
from .causal_graph import CausalGraph


def load_graph(filepath: Optional[str] = None, use_dummy: bool = False, 
               perturbation_name: str = "JAK1") -> CausalGraph:
    """
    Load a causal graph from file or create a dummy graph.
    
    Args:
        filepath: Path to pre-computed graph JSON file
        use_dummy: If True, create a dummy graph instead of loading from file
        perturbation_name: Name of perturbation for dummy graph
    
    Returns:
        CausalGraph instance
    """
    if use_dummy:
        return CausalGraph.create_dummy_graph(perturbation_name)
    
    if filepath and os.path.exists(filepath):
        try:
            return CausalGraph.load(filepath)
        except Exception as e:
            print(f"Warning: Could not load graph from {filepath}: {e}")
            print("  Creating dummy graph instead")
            return CausalGraph.create_dummy_graph(perturbation_name)
    
    # Default: create empty graph
    return CausalGraph()


def get_graph_path(perturbation_name: str, 
                   base_dir: str = "/home/nebius/cellian/outputs") -> str:
    """
    Get expected path for a pre-computed graph.
    
    Args:
        perturbation_name: Name of perturbation
        base_dir: Base directory for graph files
    
    Returns:
        Path to graph file
    """
    return os.path.join(base_dir, f"graphs", f"{perturbation_name}_graph.json")


def save_graph(graph: CausalGraph, filepath: str):
    """
    Save graph to file.
    
    Args:
        graph: CausalGraph instance to save
        filepath: Path to save graph
    """
    graph.save(filepath)

