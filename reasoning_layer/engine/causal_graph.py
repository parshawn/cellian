"""Causal graph structure for 3-node chain: Perturbation → RNA → Protein."""
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GraphNode:
    """Represents a node in the causal graph."""
    node_id: int
    name: str
    node_type: str  # "perturbation", "rna", "protein"
    embedding: Optional[Any] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Represents an edge in the causal graph."""
    source: int
    target: int
    edge_type: str  # "embeds_to", "translates_to"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CausalGraph:
    """
    Causal graph representing the 3-node chain:
    Node 1: Perturbation (embedding)
    Node 2: RNA (prediction)
    Node 3: Protein (prediction)
    
    Supports both dynamic building and loading pre-computed graphs.
    """
    
    def __init__(self, graph_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize causal graph.
        
        Args:
            graph_dict: Optional pre-computed graph dictionary to load
        """
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.node_counter = 0
        
        if graph_dict:
            self.from_dict(graph_dict)
    
    def add_node(self, name: str, node_type: str, embedding: Optional[Any] = None, 
                 data: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a node to the graph."""
        self.node_counter += 1
        node = GraphNode(
            node_id=self.node_counter,
            name=name,
            node_type=node_type,
            embedding=embedding,
            data=data or {},
            metadata=metadata or {}
        )
        self.nodes[self.node_counter] = node
        return self.node_counter
    
    def add_edge(self, source_id: int, target_id: int, edge_type: str, 
                 weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        """Add an edge to the graph."""
        edge = GraphEdge(
            source=source_id,
            target=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {}
        )
        self.edges.append(edge)
    
    def get_node(self, node_id: int) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_node_by_name(self, name: str) -> Optional[GraphNode]:
        """Get a node by name."""
        for node in self.nodes.values():
            if node.name == name:
                return node
        return None
    
    def get_edges_from(self, source_id: int) -> List[GraphEdge]:
        """Get all edges from a source node."""
        return [e for e in self.edges if e.source == source_id]
    
    def get_edges_to(self, target_id: int) -> List[GraphEdge]:
        """Get all edges to a target node."""
        return [e for e in self.edges if e.target == target_id]
    
    def get_path(self, start_id: int, end_id: int) -> List[int]:
        """Get path from start to end node (BFS)."""
        if start_id == end_id:
            return [start_id]
        
        queue = [(start_id, [start_id])]
        visited = {start_id}
        
        while queue:
            node_id, path = queue.pop(0)
            
            for edge in self.get_edges_from(node_id):
                next_id = edge.target
                if next_id == end_id:
                    return path + [next_id]
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))
        
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        # Convert embeddings to lists for JSON serialization
        nodes_data = []
        for node in self.nodes.values():
            node_dict = {
                "id": node.node_id,
                "name": node.name,
                "type": node.node_type,
                "data": node.data,
                "metadata": node.metadata.copy()
            }
            # Handle embedding (convert to list if numpy array)
            if node.embedding is not None:
                if isinstance(node.embedding, np.ndarray):
                    node_dict["embedding"] = node.embedding.tolist()
                    node_dict["metadata"]["embedding_dim"] = len(node.embedding)
                elif isinstance(node.embedding, list):
                    node_dict["embedding"] = node.embedding
                else:
                    # Skip embedding if not serializable
                    node_dict["metadata"]["has_embedding"] = True
            nodes_data.append(node_dict)
        
        edges_data = [
            {
                "source": edge.source,
                "target": edge.target,
                "type": edge.edge_type,
                "weight": edge.weight,
                "metadata": edge.metadata
            }
            for edge in self.edges
        ]
        
        return {
            "nodes": nodes_data,
            "edges": edges_data
        }
    
    def from_dict(self, graph_dict: Dict[str, Any]):
        """Load graph from dictionary representation."""
        self.nodes = {}
        self.edges = []
        
        # Load nodes
        for node_data in graph_dict.get("nodes", []):
            node_id = node_data["id"]
            # Convert embedding back from list if present
            embedding = None
            if "embedding" in node_data:
                embedding = np.array(node_data["embedding"]) if isinstance(node_data["embedding"], list) else None
            
            node = GraphNode(
                node_id=node_id,
                name=node_data["name"],
                node_type=node_data["type"],
                embedding=embedding,
                data=node_data.get("data", {}),
                metadata=node_data.get("metadata", {})
            )
            self.nodes[node_id] = node
            self.node_counter = max(self.node_counter, node_id)
        
        # Load edges
        for edge_data in graph_dict.get("edges", []):
            edge = GraphEdge(
                source=edge_data["source"],
                target=edge_data["target"],
                edge_type=edge_data["type"],
                weight=edge_data.get("weight", 1.0),
                metadata=edge_data.get("metadata", {})
            )
            self.edges.append(edge)
    
    def save(self, filepath: str):
        """Save graph to JSON file."""
        graph_dict = self.to_dict()
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(graph_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CausalGraph':
        """Load graph from JSON file."""
        with open(filepath, 'r') as f:
            graph_dict = json.load(f)
        return cls(graph_dict=graph_dict)
    
    @classmethod
    def create_dummy_graph(cls, perturbation_name: str = "JAK1") -> 'CausalGraph':
        """
        Create a dummy graph for testing.
        
        Args:
            perturbation_name: Name of perturbation for dummy graph
        
        Returns:
            CausalGraph instance with dummy nodes and edges
        """
        graph = cls()
        
        # Node 1: Perturbation
        node1_id = graph.add_node(
            name=f"perturbation_{perturbation_name}",
            node_type="perturbation",
            embedding=np.random.randn(1024).astype(np.float32),  # Dummy embedding
            data={"perturbation_name": perturbation_name},
            metadata={"embedding_dim": 1024, "dummy": True}
        )
        
        # Node 2: RNA prediction
        node2_id = graph.add_node(
            name="rna_prediction",
            node_type="rna",
            data={
                "delta_rna": {
                    "STAT1": -0.7,
                    "IRF1": -0.5,
                    "CXCL10": -0.4,
                    "HLA-A": -0.6,
                    "B2M": -0.5
                }
            },
            metadata={"n_genes": 5, "dummy": True}
        )
        
        # Node 3: Protein prediction
        node3_id = graph.add_node(
            name="protein_prediction",
            node_type="protein",
            data={
                "delta_protein": {
                    "HLA-A": -0.6,
                    "CD58": -0.4
                }
            },
            metadata={"n_proteins": 2, "dummy": True}
        )
        
        # Edge 1: Perturbation -> RNA
        graph.add_edge(
            source_id=node1_id,
            target_id=node2_id,
            edge_type="embeds_to",
            weight=1.0,
            metadata={"transformation": "perturbation_to_rna", "dummy": True}
        )
        
        # Edge 2: RNA -> Protein
        graph.add_edge(
            source_id=node2_id,
            target_id=node3_id,
            edge_type="translates_to",
            weight=1.0,
            metadata={"transformation": "rna_to_protein", "dummy": True}
        )
        
        return graph
    
    def is_dummy(self) -> bool:
        """Check if graph is a dummy graph."""
        if not self.nodes:
            return False
        # Check if any node has dummy metadata
        return any(node.metadata.get("dummy", False) for node in self.nodes.values())
    
    def get_3_node_chain(self) -> Optional[List[int]]:
        """
        Get the standard 3-node chain: perturbation -> RNA -> protein.
        
        Returns:
            List of node IDs in order, or None if chain doesn't exist
        """
        # Find perturbation node
        pert_node = None
        for node in self.nodes.values():
            if node.node_type == "perturbation":
                pert_node = node
                break
        
        if not pert_node:
            return None
        
        # Find path to protein node
        protein_node = None
        for node in self.nodes.values():
            if node.node_type == "protein":
                protein_node = node
                break
        
        if not protein_node:
            return None
        
        # Get path
        path = self.get_path(pert_node.node_id, protein_node.node_id)
        return path if len(path) == 3 else None
