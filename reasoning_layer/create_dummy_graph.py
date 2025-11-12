"""Script to create and save a dummy graph for testing."""
import json
import sys
from engine.causal_graph import CausalGraph
from engine.graph_loader import save_graph, get_graph_path


def main():
    """Create and save a dummy graph."""
    if len(sys.argv) > 1:
        perturbation_name = sys.argv[1]
    else:
        perturbation_name = "JAK1"
    
    print(f"Creating dummy graph for perturbation: {perturbation_name}")
    
    # Create dummy graph
    graph = CausalGraph.create_dummy_graph(perturbation_name)
    
    # Save graph
    graph_path = get_graph_path(perturbation_name)
    save_graph(graph, graph_path)
    
    print(f"Saved dummy graph to: {graph_path}")
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Print graph structure
    graph_dict = graph.to_dict()
    print("\nGraph structure:")
    print(json.dumps(graph_dict, indent=2))


if __name__ == "__main__":
    main()

