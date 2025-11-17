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
    
    def visualize(self, filepath: Optional[str] = None, show_changes: bool = True):
        """
        Create a visual diagram of the causal graph.
        
        Args:
            filepath: Path to save the visualization (if None, returns figure)
            show_changes: If True, show RNA/protein changes in node labels
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        except ImportError:
            print("Warning: matplotlib not available, cannot visualize graph")
            return None
        
        # Determine layout: pathway-aware or simple 3-node
        has_pathways = any(node.node_type == "pathway" for node in self.nodes.values())
        has_genes = any(node.node_type == "gene" for node in self.nodes.values())
        
        if has_pathways or has_genes:
            # Pathway-aware layout: hierarchical
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))
            ax.set_xlim(-1, 15)
            ax.set_ylim(-1, 8)
        else:
            # Simple 3-node layout
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.set_xlim(-1, 11)
            ax.set_ylim(-1, 3)
        
        ax.axis('off')
        
        # Find nodes by type
        pert_node = None
        rna_node = None
        prot_node = None
        pathway_nodes = []
        gene_nodes = []
        
        for node in self.nodes.values():
            if node.node_type == "perturbation":
                pert_node = node
            elif node.node_type == "rna":
                rna_node = node
            elif node.node_type == "protein":
                prot_node = node
            elif node.node_type == "pathway":
                pathway_nodes.append(node)
            elif node.node_type == "gene":
                gene_nodes.append(node)
        
        # Create position mapping
        node_positions = {}
        
        if has_pathways or has_genes:
            # Hierarchical layout: perturbation at top, pathways below, genes below pathways, RNA/protein at bottom
            if pert_node:
                node_positions[pert_node.node_id] = (7, 7)
            
            # Pathway nodes in a row
            for i, pathway_node in enumerate(pathway_nodes[:3]):  # Max 3 pathways
                node_positions[pathway_node.node_id] = (3 + i * 4, 5)
            
            # Gene nodes (show up to 6 genes)
            for i, gene_node in enumerate(gene_nodes[:6]):
                row = i // 3
                col = i % 3
                node_positions[gene_node.node_id] = (2 + col * 4, 3 - row * 1.5)
            
            # RNA and protein nodes at bottom
            if rna_node:
                node_positions[rna_node.node_id] = (5, 1)
            if prot_node:
                node_positions[prot_node.node_id] = (9, 1)
        else:
            # Simple horizontal layout
            x_positions = [1, 5, 9]
            nodes_in_order = [pert_node, rna_node, prot_node]
            for i, node in enumerate(nodes_in_order):
                if node:
                    node_positions[node.node_id] = (x_positions[i], 1.5)
        
        # Color scheme
        node_colors = {
            "perturbation": "#FF6B6B",  # Red
            "rna": "#4ECDC4",           # Teal
            "protein": "#45B7D1",        # Blue
            "pathway": "#9B59B6",        # Purple
            "gene": "#F39C12"           # Orange
        }
        
        # Draw edges first (so they're behind nodes)
        for edge in self.edges:
            source_id = edge.source
            target_id = edge.target
            if source_id in node_positions and target_id in node_positions:
                x1, y1 = node_positions[source_id]
                x2, y2 = node_positions[target_id]
                
                # Draw arrow
                arrow = FancyArrowPatch(
                    (x1 + 0.4, y1), (x2 - 0.4, y2),
                    arrowstyle='->', mutation_scale=20,
                    linewidth=2, color='#666666', zorder=1
                )
                ax.add_patch(arrow)
                
                # Edge label
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                edge_label = edge.edge_type.replace('_', ' ').title()
                ax.text(mid_x, mid_y + 0.3, edge_label, 
                       ha='center', va='bottom', fontsize=9, 
                       style='italic', color='#666666')
        
        # Draw nodes
        for node_id, node in self.nodes.items():
            if node_id not in node_positions:
                continue
                
            x, y = node_positions[node_id]
            color = node_colors.get(node.node_type, "#CCCCCC")
            
            # Create rounded rectangle for node
            box = FancyBboxPatch(
                (x - 0.5, y - 0.4), 1.0, 0.8,
                boxstyle="round,pad=0.1",
                linewidth=2,
                edgecolor='#333333',
                facecolor=color,
                alpha=0.8,
                zorder=2
            )
            ax.add_patch(box)
            
            # Node name
            node_name = node.name.replace('_', ' ').title()
            ax.text(x, y + 0.15, node_name, 
                   ha='center', va='center', fontsize=11, 
                   fontweight='bold', color='white')
            
            # Show changes if available
            if show_changes:
                change_text = ""
                if node.node_type == "rna" and "delta_rna" in node.data:
                    delta_rna = node.data["delta_rna"]
                    # Show top 3 changes
                    top_changes = sorted(delta_rna.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    change_parts = [f"{gene}: {val:+.2f}" for gene, val in top_changes]
                    change_text = "\n".join(change_parts)
                elif node.node_type == "protein" and "delta_protein" in node.data:
                    delta_prot = node.data["delta_protein"]
                    change_parts = [f"{marker}: {val:+.2f}" for marker, val in delta_prot.items()]
                    change_text = "\n".join(change_parts)
                elif node.node_type == "perturbation":
                    pert_name = node.data.get("perturbation_name", "")
                    change_text = f"Perturbation:\n{pert_name}"
                elif node.node_type == "pathway":
                    pathway_id = node.data.get("pathway_id", "")
                    n_genes = len(node.data.get("genes", []))
                    change_text = f"{n_genes} genes"
                elif node.node_type == "gene":
                    gene_name = node.data.get("gene_name", node.name)
                    delta = node.data.get("delta_rna", 0.0)
                    change_text = f"{gene_name}\n{delta:+.2f}"
                
                if change_text:
                    ax.text(x, y - 0.2, change_text, 
                           ha='center', va='top', fontsize=8, 
                           color='white', style='italic')
        
        # Title
        title_y = 7.5 if (has_pathways or has_genes) else 2.5
        title_x = 7 if (has_pathways or has_genes) else 5
        ax.text(title_x, title_y, 'Causal Reasoning Graph', 
               ha='center', va='top', fontsize=16, fontweight='bold')
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=node_colors["perturbation"], label='Perturbation', alpha=0.8),
            mpatches.Patch(facecolor=node_colors["rna"], label='RNA Prediction', alpha=0.8),
            mpatches.Patch(facecolor=node_colors["protein"], label='Protein Prediction', alpha=0.8),
        ]
        if has_pathways:
            legend_elements.append(mpatches.Patch(facecolor=node_colors["pathway"], label='Pathway', alpha=0.8))
        if has_genes:
            legend_elements.append(mpatches.Patch(facecolor=node_colors["gene"], label='Gene', alpha=0.8))
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            return fig
    
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
