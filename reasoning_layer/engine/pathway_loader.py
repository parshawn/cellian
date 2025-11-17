"""Pathway data loader for KEGG pathways and related biological networks."""
import os
import json
from typing import Dict, List, Set, Tuple, Optional


class PathwayLoader:
    """
    Loads pathway data from KEGG and other sources.
    Supports loading from JSON files (pre-downloaded data).
    """
    
    def __init__(self, data_dir: str = "/home/nebius/cellian/reasoning_layer/data/pathways"):
        """
        Initialize pathway loader.
        
        Args:
            data_dir: Directory containing pathway data files
        """
        self.data_dir = data_dir
        self.pathways: Dict[str, Dict] = {}
        self.gene_to_pathways: Dict[str, List[str]] = {}
        self.pathway_to_genes: Dict[str, List[str]] = {}
        self.ppi_edges: List[Tuple[str, str, str]] = []  # (gene1, gene2, interaction_type)
        self.regulatory_edges: List[Tuple[str, str, str, str]] = []  # (tf, target, type, sign)
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load pathway data
        self._load_pathway_data()
    
    def _load_pathway_data(self):
        """Load pathway data from JSON files."""
        # Load KEGG pathways
        kegg_file = os.path.join(self.data_dir, "kegg_pathways.json")
        if os.path.exists(kegg_file):
            try:
                with open(kegg_file, 'r') as f:
                    self.pathways = json.load(f)
                self._build_gene_pathway_mappings()
                print(f"Loaded {len(self.pathways)} pathways from {kegg_file}")
            except Exception as e:
                print(f"Warning: Could not load pathway data from {kegg_file}: {e}")
                self.pathways = {}
        else:
            print(f"Warning: Pathway data file not found: {kegg_file}")
            print(f"  Run download_pathway_data.py to download pathway data")
            self.pathways = {}
        
        # Load PPI data (optional)
        ppi_file = os.path.join(self.data_dir, "ppi_network.json")
        if os.path.exists(ppi_file):
            try:
                with open(ppi_file, 'r') as f:
                    ppi_data = json.load(f)
                    self.ppi_edges = [
                        (item["protein1"], item["protein2"], item.get("interaction_type", "interacts_with"))
                        for item in ppi_data
                    ]
                print(f"Loaded {len(self.ppi_edges)} PPI edges from {ppi_file}")
            except Exception as e:
                print(f"Warning: Could not load PPI data: {e}")
        
        # Load regulatory relationships (optional)
        reg_file = os.path.join(self.data_dir, "regulatory_network.json")
        if os.path.exists(reg_file):
            try:
                with open(reg_file, 'r') as f:
                    reg_data = json.load(f)
                    self.regulatory_edges = [
                        (item["tf"], item["target"], item.get("type", "transcription_factor"), item.get("sign", "+"))
                        for item in reg_data
                    ]
                print(f"Loaded {len(self.regulatory_edges)} regulatory edges from {reg_file}")
            except Exception as e:
                print(f"Warning: Could not load regulatory data: {e}")
    
    def _build_gene_pathway_mappings(self):
        """Build gene → pathways and pathway → genes mappings."""
        self.gene_to_pathways = {}
        self.pathway_to_genes = {}
        
        for pathway_id, pathway_data in self.pathways.items():
            genes = pathway_data.get("genes", [])
            self.pathway_to_genes[pathway_id] = genes
            
            for gene in genes:
                if gene not in self.gene_to_pathways:
                    self.gene_to_pathways[gene] = []
                self.gene_to_pathways[gene].append(pathway_id)
    
    def get_pathway(self, pathway_id: str) -> Optional[Dict]:
        """Get pathway data by ID."""
        return self.pathways.get(pathway_id)
    
    def get_pathways_for_gene(self, gene_name: str) -> List[str]:
        """Get list of pathway IDs that contain this gene."""
        return self.gene_to_pathways.get(gene_name, [])
    
    def get_genes_in_pathway(self, pathway_id: str) -> List[str]:
        """Get list of genes in a pathway."""
        return self.pathway_to_genes.get(pathway_id, [])
    
    def get_pathway_edges(self, pathway_id: str) -> List[Dict]:
        """Get edges (interactions) within a pathway."""
        pathway = self.get_pathway(pathway_id)
        if pathway:
            return pathway.get("edges", [])
        return []
    
    def find_affected_pathways(self, perturbation_gene: str, max_hops: int = 3) -> List[str]:
        """
        Find all pathways affected by a perturbation.
        
        Args:
            perturbation_gene: Gene name (e.g., "JAK1")
            max_hops: Maximum distance to traverse (not used in simple version)
        
        Returns:
            List of pathway IDs
        """
        # Direct pathways (gene is a member)
        affected_pathways = set(self.get_pathways_for_gene(perturbation_gene))
        
        # Could extend to find indirect pathways through interactions
        # For now, return direct pathways
        return list(affected_pathways)
    
    def get_all_pathway_genes(self, pathway_ids: List[str]) -> Set[str]:
        """Get all genes in given pathways."""
        all_genes = set()
        for pathway_id in pathway_ids:
            genes = self.get_genes_in_pathway(pathway_id)
            all_genes.update(genes)
        return all_genes
    
    def traverse_pathway(self, start_gene: str, pathway_id: Optional[str] = None, 
                        max_hops: int = 3) -> Dict[str, any]:
        """
        Traverse pathway from a starting gene.
        Returns all affected genes through pathway cascades.
        
        Args:
            start_gene: Starting gene name
            pathway_id: Optional pathway ID to limit traversal
            max_hops: Maximum distance to traverse
        
        Returns:
            Dict with "genes" (list) and "paths" (list of paths)
        """
        affected_genes = {start_gene}
        paths = [[start_gene]]
        
        # Get pathway edges
        if pathway_id:
            edges = self.get_pathway_edges(pathway_id)
        else:
            # Get edges from all pathways containing start_gene
            pathway_ids = self.get_pathways_for_gene(start_gene)
            edges = []
            for pid in pathway_ids:
                edges.extend(self.get_pathway_edges(pid))
        
        # Simple traversal: follow edges from start_gene
        current_level = {start_gene}
        visited = {start_gene}
        
        for hop in range(max_hops):
            next_level = set()
            for gene in current_level:
                # Find edges starting from this gene
                for edge in edges:
                    if isinstance(edge, dict):
                        src = edge.get("source") or edge.get("src")
                        dst = edge.get("target") or edge.get("dst")
                    elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                        src, dst = edge[0], edge[1]
                    else:
                        continue
                    
                    if src == gene and dst not in visited:
                        next_level.add(dst)
                        visited.add(dst)
                        affected_genes.add(dst)
            
            if not next_level:
                break
            current_level = next_level
        
        return {
            "genes": list(affected_genes),
            "paths": paths  # Simplified for now
        }

