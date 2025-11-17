"""Extended causal graph with pathway-level reasoning."""
from typing import Dict, List, Optional, Any, Set
from .causal_graph import CausalGraph, GraphNode, GraphEdge
from .pathway_loader import PathwayLoader


class PathwayGraph(CausalGraph):
    """
    Extended causal graph with pathway-level reasoning.
    Inherits from CausalGraph, adds pathway-specific methods.
    """
    
    def __init__(self, graph_dict: Optional[Dict[str, Any]] = None, 
                 load_pathway_data: bool = True):
        """
        Initialize pathway graph.
        
        Args:
            graph_dict: Optional pre-computed graph dictionary to load
            load_pathway_data: If True, load pathway data and build pathway nodes/edges
        """
        super().__init__(graph_dict)
        self.pathway_loader = PathwayLoader() if load_pathway_data else None
        self.pathways: Dict[str, Dict] = {}
        self.gene_to_pathways: Dict[str, List[str]] = {}
        self.pathway_to_genes: Dict[str, List[str]] = {}
        
        if load_pathway_data and self.pathway_loader:
            self.pathways = self.pathway_loader.pathways
            self.gene_to_pathways = self.pathway_loader.gene_to_pathways
            self.pathway_to_genes = self.pathway_loader.pathway_to_genes
            self._add_pathway_nodes()
            self._add_pathway_edges()
    
    def _add_pathway_nodes(self):
        """Add pathway nodes to graph."""
        for pathway_id, pathway_data in self.pathways.items():
            pathway_name = pathway_data.get("name", pathway_id)
            
            # Check if pathway node already exists
            existing_node = self.get_node_by_name(pathway_name)
            if existing_node:
                continue
            
            self.add_node(
                name=pathway_name,
                node_type="pathway",
                data={
                    "pathway_id": pathway_id,
                    "genes": pathway_data.get("genes", []),
                    "category": pathway_data.get("category", "unknown")
                },
                metadata={
                    "n_genes": len(pathway_data.get("genes", [])),
                    "source": "KEGG"
                }
            )
    
    def _add_pathway_edges(self):
        """Add pathway relationship edges."""
        for pathway_id, pathway_data in self.pathways.items():
            pathway_name = pathway_data.get("name", pathway_id)
            pathway_node = self.get_node_by_name(pathway_name)
            if not pathway_node:
                continue
            
            # Add edges: gene → pathway (membership)
            for gene in pathway_data.get("genes", []):
                gene_node = self.get_node_by_name(gene)
                if not gene_node:
                    # Create gene node if it doesn't exist
                    gene_node_id = self.add_node(
                        name=gene,
                        node_type="gene",
                        data={"gene_name": gene},
                        metadata={}
                    )
                else:
                    gene_node_id = gene_node.node_id
                
                # Check if edge already exists
                existing_edges = [
                    e for e in self.get_edges_from(gene_node_id)
                    if e.target == pathway_node.node_id and e.edge_type == "pathway_member"
                ]
                if not existing_edges:
                    self.add_edge(
                        source_id=gene_node_id,
                        target_id=pathway_node.node_id,
                        edge_type="pathway_member",
                        weight=1.0,
                        metadata={"pathway_id": pathway_id}
                    )
            
            # Add edges: gene → gene (within pathway)
            for edge_data in pathway_data.get("edges", []):
                if isinstance(edge_data, dict):
                    src = edge_data.get("source") or edge_data.get("src")
                    dst = edge_data.get("target") or edge_data.get("dst")
                    edge_type = edge_data.get("type") or edge_data.get("rel", "regulates")
                    sign = edge_data.get("sign", "+")
                elif isinstance(edge_data, (list, tuple)) and len(edge_data) >= 2:
                    src, dst = edge_data[0], edge_data[1]
                    edge_type = edge_data[2] if len(edge_data) > 2 else "regulates"
                    sign = edge_data[3] if len(edge_data) > 3 else "+"
                else:
                    continue
                
                src_node = self.get_node_by_name(src)
                dst_node = self.get_node_by_name(dst)
                
                if not src_node:
                    src_node_id = self.add_node(name=src, node_type="gene", data={"gene_name": src})
                else:
                    src_node_id = src_node.node_id
                
                if not dst_node:
                    dst_node_id = self.add_node(name=dst, node_type="gene", data={"gene_name": dst})
                else:
                    dst_node_id = dst_node.node_id
                
                # Check if edge already exists
                existing_edges = [
                    e for e in self.get_edges_from(src_node_id)
                    if e.target == dst_node_id and e.edge_type == edge_type
                ]
                if not existing_edges:
                    self.add_edge(
                        source_id=src_node_id,
                        target_id=dst_node_id,
                        edge_type=edge_type,
                        weight=1.0,
                        metadata={"sign": sign, "pathway_id": pathway_id}
                    )
    
    def find_affected_pathways(self, perturbation_gene: str, max_hops: int = 3) -> List[str]:
        """
        Find all pathways affected by a perturbation.
        
        Args:
            perturbation_gene: Gene name (e.g., "JAK1")
            max_hops: Maximum distance to traverse
        
        Returns:
            List of pathway IDs
        """
        if not self.pathway_loader:
            return []
        
        return self.pathway_loader.find_affected_pathways(perturbation_gene, max_hops)
    
    def get_pathway_genes(self, pathway_ids: List[str]) -> Set[str]:
        """Get all genes in given pathways."""
        if not self.pathway_loader:
            return set()
        
        return self.pathway_loader.get_all_pathway_genes(pathway_ids)
    
    def traverse_pathway(self, start_gene: str, pathway_id: Optional[str] = None,
                        max_hops: int = 3) -> Dict[str, Any]:
        """
        Traverse pathway from a starting gene.
        Returns all affected genes through pathway cascades.
        """
        if not self.pathway_loader:
            return {"genes": [], "paths": []}
        
        return self.pathway_loader.traverse_pathway(start_gene, pathway_id, max_hops)

