"""Unified phenotype knowledge base integrating multiple data sources."""

import math
from typing import List, Dict, Any, Optional
from collections import defaultdict


class PhenotypeKB:
    """
    Unified phenotype knowledge base built from:
    - Gene2Phenotype (G2P)
    - Human Phenotype Ontology (HPO)
    - CTD (gene–phenotype + gene–biological process)
    - DisGeNET (gene–disease, disease→phenotype mapping)
    - CellMarker or PanglaoDB (cell-state phenotypes)
    """
    
    def __init__(
        self,
        g2p_data: Optional[Dict[str, List[Dict]]] = None,
        hpo_data: Optional[Dict[str, List[Dict]]] = None,
        ctd_data: Optional[Dict[str, List[Dict]]] = None,
        disgenet_data: Optional[Dict[str, List[Dict]]] = None,
        cellmarker_data: Optional[Dict[str, List[Dict]]] = None
    ):
        """
        Initialize phenotype knowledge base with preprocessed data.
        
        Args:
            g2p_data: Dict mapping gene -> list of phenotype dicts from G2P
            hpo_data: Dict mapping gene -> list of phenotype dicts from HPO
            ctd_data: Dict mapping gene -> list of phenotype dicts from CTD
            disgenet_data: Dict mapping gene -> list of phenotype dicts from DisGeNET
            cellmarker_data: Dict mapping gene -> list of cell-state phenotypes
        """
        self.g2p_data = g2p_data or {}
        self.hpo_data = hpo_data or {}
        self.ctd_data = ctd_data or {}
        self.disgenet_data = disgenet_data or {}
        self.cellmarker_data = cellmarker_data or {}
    
    def get_gene_phenotypes(self, gene: str) -> List[Dict[str, Any]]:
        """
        Return all phenotype entries for a gene from all sources.
        
        Args:
            gene: Gene symbol (e.g., "TP53")
        
        Returns:
            List of phenotype dictionaries:
            [
                {
                    "phenotype_id": "HP:0001903",
                    "name": "Increased apoptosis",
                    "source": "G2P|HPO|CTD|DisGeNET|CellMarker",
                    "evidence_type": "direct|indirect",
                    "confidence": float (0.0-1.0)
                },
                ...
            ]
        """
        phenotypes = []
        seen = set()  # Track (gene, phenotype_id) pairs to avoid duplicates
        
        # Collect from G2P
        for pheno in self.g2p_data.get(gene.upper(), []):
            key = (gene.upper(), pheno.get("phenotype_id", ""))
            if key not in seen:
                pheno["source"] = "G2P"
                phenotypes.append(pheno)
                seen.add(key)
        
        # Collect from HPO
        for pheno in self.hpo_data.get(gene.upper(), []):
            key = (gene.upper(), pheno.get("phenotype_id", ""))
            if key not in seen:
                pheno["source"] = "HPO"
                phenotypes.append(pheno)
                seen.add(key)
        
        # Collect from CTD
        for pheno in self.ctd_data.get(gene.upper(), []):
            key = (gene.upper(), pheno.get("phenotype_id", ""))
            if key not in seen:
                pheno["source"] = "CTD"
                phenotypes.append(pheno)
                seen.add(key)
        
        # Collect from DisGeNET
        for pheno in self.disgenet_data.get(gene.upper(), []):
            key = (gene.upper(), pheno.get("phenotype_id", ""))
            if key not in seen:
                pheno["source"] = "DisGeNET"
                phenotypes.append(pheno)
                seen.add(key)
        
        # Collect from CellMarker
        for pheno in self.cellmarker_data.get(gene.upper(), []):
            key = (gene.upper(), pheno.get("phenotype_id", ""))
            if key not in seen:
                pheno["source"] = "CellMarker"
                phenotypes.append(pheno)
                seen.add(key)
        
        return phenotypes
    
    def get_pathway_phenotypes(
        self,
        pathway_id: str,
        member_genes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Aggregate phenotypes from all member genes in a pathway.
        
        Args:
            pathway_id: Pathway identifier (e.g., "KEGG_APOPTOSIS")
            member_genes: List of gene symbols in the pathway
        
        Returns:
            List of aggregated phenotypes with gene support counts
        """
        phenotype_counts = defaultdict(lambda: {
            "phenotype_id": "",
            "name": "",
            "sources": set(),
            "supporting_genes": [],
            "gene_count": 0
        })
        
        # Collect phenotypes from all member genes
        for gene in member_genes:
            gene_phenotypes = self.get_gene_phenotypes(gene)
            for pheno in gene_phenotypes:
                pheno_id = pheno.get("phenotype_id", "")
                name = pheno.get("name", "")
                
                if not pheno_id:
                    continue
                
                if pheno_id not in phenotype_counts:
                    phenotype_counts[pheno_id]["phenotype_id"] = pheno_id
                    phenotype_counts[pheno_id]["name"] = name
                
                phenotype_counts[pheno_id]["sources"].add(pheno.get("source", "Unknown"))
                phenotype_counts[pheno_id]["supporting_genes"].append(gene)
                phenotype_counts[pheno_id]["gene_count"] += 1
        
        # Convert to list format
        aggregated = []
        for pheno_id, data in phenotype_counts.items():
            aggregated.append({
                "phenotype_id": data["phenotype_id"],
                "name": data["name"],
                "source": "|".join(sorted(data["sources"])),
                "supporting_genes": sorted(list(set(data["supporting_genes"]))),
                "gene_count": data["gene_count"],
                "pathway_id": pathway_id
            })
        
        # Sort by gene count (most supported first)
        aggregated.sort(key=lambda x: x["gene_count"], reverse=True)
        
        return aggregated
    
    def score_phenotypes(
        self,
        deg_list: List[Dict[str, Any]],
        pathways: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compute phenotype scores using DEGs and pathway enrichment.
        
        Args:
            deg_list: List of DEG dicts with keys: gene, log2fc, pval
            pathways: List of pathway dicts with keys: id, name, NES, FDR, member_genes
        
        Returns:
            List of scored phenotypes:
            [
                {
                    "phenotype_id": "HP:0001903",
                    "name": "Increased apoptosis",
                    "score": 0.0-1.0,
                    "direction": "increase|decrease|mixed",
                    "supporting_genes": [...],
                    "supporting_pathways": [...],
                    "supporting_up_genes": [...],
                    "supporting_down_genes": [...]
                },
                ...
            ]
        """
        # Build gene->DEG mapping
        deg_dict = {deg["gene"].upper(): deg for deg in deg_list}
        
        # Collect all phenotypes from DEGs and pathways
        phenotype_scores = defaultdict(lambda: {
            "phenotype_id": "",
            "name": "",
            "supporting_genes": [],
            "supporting_up_genes": [],
            "supporting_down_genes": [],
            "supporting_pathways": [],
            "pathway_scores": [],
            "gene_scores": []
        })
        
        # Score phenotypes from individual DEGs
        for deg in deg_list:
            gene = deg["gene"].upper()
            log2fc = deg.get("log2fc", 0.0)
            pval = deg.get("pval", 1.0)
            
            # Get phenotypes for this gene
            gene_phenotypes = self.get_gene_phenotypes(gene)
            
            for pheno in gene_phenotypes:
                pheno_id = pheno.get("phenotype_id", "")
                name = pheno.get("name", "")
                
                if not pheno_id:
                    continue
                
                if not phenotype_scores[pheno_id]["phenotype_id"]:
                    phenotype_scores[pheno_id]["phenotype_id"] = pheno_id
                    phenotype_scores[pheno_id]["name"] = name
                
                phenotype_scores[pheno_id]["supporting_genes"].append(gene)
                
                # Score contribution: -log10(pval) * abs(log2fc)
                gene_score = -math.log10(max(pval, 1e-10)) * abs(log2fc)
                phenotype_scores[pheno_id]["gene_scores"].append(gene_score)
                
                if log2fc > 0:
                    phenotype_scores[pheno_id]["supporting_up_genes"].append(gene)
                elif log2fc < 0:
                    phenotype_scores[pheno_id]["supporting_down_genes"].append(gene)
        
        # Score phenotypes from pathways
        for pathway in pathways:
            pathway_id = pathway.get("id", "")
            pathway_name = pathway.get("name", "")
            nes = pathway.get("NES", 0.0)
            fdr = pathway.get("FDR", 1.0)
            member_genes = [g.upper() for g in pathway.get("member_genes", [])]
            
            if not pathway_id or fdr > 0.05:
                continue  # Skip non-significant pathways
            
            # Get phenotypes for this pathway
            pathway_phenotypes = self.get_pathway_phenotypes(pathway_id, member_genes)
            
            for pheno in pathway_phenotypes:
                pheno_id = pheno.get("phenotype_id", "")
                name = pheno.get("name", "")
                
                if not pheno_id:
                    continue
                
                if not phenotype_scores[pheno_id]["phenotype_id"]:
                    phenotype_scores[pheno_id]["phenotype_id"] = pheno_id
                    phenotype_scores[pheno_id]["name"] = name
                
                phenotype_scores[pheno_id]["supporting_pathways"].append({
                    "id": pathway_id,
                    "name": pathway_name,
                    "NES": nes,
                    "FDR": fdr
                })
                
                # Pathway score: NES * (1 - FDR)
                pathway_score = abs(nes) * (1 - fdr)
                phenotype_scores[pheno_id]["pathway_scores"].append(pathway_score)
        
        # Compute final scores and directions
        scored_phenotypes = []
        for pheno_id, data in phenotype_scores.items():
            if not data["phenotype_id"]:
                continue
            
            # Aggregate scores
            gene_score = sum(data["gene_scores"]) / max(len(data["gene_scores"]), 1) if data["gene_scores"] else 0.0
            pathway_score = sum(data["pathway_scores"]) / max(len(data["pathway_scores"]), 1) if data["pathway_scores"] else 0.0
            
            # Combined score (normalize to 0-1)
            combined_score = (gene_score + pathway_score) / 2.0
            # Simple normalization (can be refined)
            normalized_score = min(combined_score / 10.0, 1.0)  # Rough normalization
            
            # Determine direction
            up_count = len(data["supporting_up_genes"])
            down_count = len(data["supporting_down_genes"])
            
            if up_count > down_count * 1.5:
                direction = "increase"
            elif down_count > up_count * 1.5:
                direction = "decrease"
            else:
                direction = "mixed"
            
            scored_phenotypes.append({
                "phenotype_id": data["phenotype_id"],
                "name": data["name"],
                "score": normalized_score,
                "direction": direction,
                "supporting_genes": sorted(list(set(data["supporting_genes"]))),
                "supporting_up_genes": sorted(list(set(data["supporting_up_genes"]))),
                "supporting_down_genes": sorted(list(set(data["supporting_down_genes"]))),
                "supporting_pathways": data["supporting_pathways"]
            })
        
        # Sort by score (highest first)
        scored_phenotypes.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_phenotypes


# Import numpy for scoring (with fallback)
try:
    import numpy as np
except ImportError:
    import math
    # Fallback implementation
    class np_module:
        @staticmethod
        def log10(x):
            return math.log10(x) if x > 0 else -10
        @staticmethod
        def log(x):
            return math.log(x) if x > 0 else -10
    np = np_module

