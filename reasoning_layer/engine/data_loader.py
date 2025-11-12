"""Data loader for loading pre-computed aligned embeddings and nodes."""
import os
import sys
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional


class DataLoader:
    """
    Loads pre-computed aligned embeddings and nodes.
    Assumes embeddings are already aligned and nodes are ready.
    """
    
    def __init__(self, data_dir: str = "/home/nebius/cellian/data/perturb-cite-seq/SCP1064",
                 embeddings_path: str = "/home/nebius/cellian/outputs/cell_embeddings.pt"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Base directory for perturbation data
            embeddings_path: Path to pre-computed aligned embeddings
        """
        self.data_dir = data_dir
        self.embeddings_path = embeddings_path
        self.rna_meta = None
        self.rna_exp = None
        self.prot_exp = None
        self.embeddings = None
        self.cells = None
        self._load_data()
    
    def _load_data(self):
        """Load pre-computed aligned embeddings and expression data."""
        # Load metadata
        meta_path = os.path.join(self.data_dir, "metadata", "RNA_metadata.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        self.rna_meta = pd.read_csv(meta_path)
        self.rna_meta = self.rna_meta.rename(columns={"NAME": "cell"}).set_index("cell")
        
        # Load RNA expression
        rna_path = os.path.join(self.data_dir, "other", "RNA_expression.csv")
        if not os.path.exists(rna_path):
            raise FileNotFoundError(f"RNA expression file not found: {rna_path}")
        self.rna_exp = pd.read_csv(rna_path, index_col=0).T
        
        # Load protein expression
        prot_path = os.path.join(self.data_dir, "expression", "Protein_expression.csv")
        if not os.path.exists(prot_path):
            raise FileNotFoundError(f"Protein expression file not found: {prot_path}")
        self.prot_exp = pd.read_csv(prot_path, index_col=0).T
        
        # Align indices
        common = self.rna_meta.index.intersection(self.rna_exp.index).intersection(self.prot_exp.index)
        if len(common) == 0:
            raise ValueError("No common cells found between metadata, RNA, and protein data")
        
        self.rna_meta = self.rna_meta.loc[common]
        self.rna_exp = self.rna_exp.loc[common]
        self.prot_exp = self.prot_exp.loc[common]
        
        # Load pre-computed aligned embeddings (assumed to be already aligned)
        if os.path.exists(self.embeddings_path):
            try:
                self.embeddings = torch.load(self.embeddings_path, map_location="cpu")
                self.cells = self.embeddings.get("cells", list(common))
                # Embeddings are assumed to be already aligned across modalities
            except Exception as e:
                print(f"Warning: Could not load embeddings: {e}", file=sys.stderr)
                self.embeddings = None
                self.cells = list(common)
        else:
            print(f"Warning: Embeddings file not found: {self.embeddings_path}", file=sys.stderr)
            print("  Embeddings should be pre-computed and aligned before using reasoning_layer", file=sys.stderr)
            self.embeddings = None
            self.cells = list(common)
    
    def get_control_cells(self) -> List[str]:
        """Get list of control cell IDs."""
        if self.rna_meta is None:
            return []
        ctrl_mask = self.rna_meta["sgRNA"].fillna("CTRL").isin(["CTRL", "control", "Control"])
        return self.rna_meta[ctrl_mask].index.tolist()
    
    def get_control_rna_profile(self) -> Dict[str, float]:
        """Get average RNA profile for control cells."""
        control_cells = self.get_control_cells()
        if not control_cells:
            return {}
        control_rna = self.rna_exp.loc[control_cells]
        return control_rna.mean(axis=0).to_dict()
    
    def get_control_protein_profile(self) -> Dict[str, float]:
        """Get average protein profile for control cells."""
        control_cells = self.get_control_cells()
        if not control_cells:
            return {}
        control_prot = self.prot_exp.loc[control_cells]
        return control_prot.mean(axis=0).to_dict()
    
    def get_perturbation_cells(self, perturbation_name: str) -> List[str]:
        """Get list of cell IDs for a specific perturbation."""
        if self.rna_meta is None:
            return []
        pert_col = self.rna_meta["sgRNA"].fillna("CTRL")
        # Try exact match first
        mask = pert_col == perturbation_name
        if not mask.any():
            # Try prefix match (e.g., "JAK1" matches "JAK1_KO")
            mask = pert_col.str.startswith(perturbation_name, na=False)
        return self.rna_meta[mask].index.tolist()
    
    def get_real_rna_profile(self, perturbation_name: str) -> Dict[str, float]:
        """Get average real RNA profile for a perturbation."""
        pert_cells = self.get_perturbation_cells(perturbation_name)
        if not pert_cells:
            return {}
        pert_rna = self.rna_exp.loc[pert_cells]
        return pert_rna.mean(axis=0).to_dict()
    
    def get_real_protein_profile(self, perturbation_name: str) -> Dict[str, float]:
        """Get average real protein profile for a perturbation."""
        pert_cells = self.get_perturbation_cells(perturbation_name)
        if not pert_cells:
            return {}
        pert_prot = self.prot_exp.loc[pert_cells]
        return pert_prot.mean(axis=0).to_dict()
    
    def get_rna_delta(self, perturbation_name: str) -> Dict[str, float]:
        """Get RNA delta (perturbation - control) for a perturbation."""
        control_rna = self.get_control_rna_profile()
        real_rna = self.get_real_rna_profile(perturbation_name)
        delta = {}
        all_genes = set(control_rna.keys()) | set(real_rna.keys())
        for gene in all_genes:
            control_val = control_rna.get(gene, 0.0)
            real_val = real_rna.get(gene, 0.0)
            delta[gene] = real_val - control_val
        return delta
    
    def get_protein_delta(self, perturbation_name: str) -> Dict[str, float]:
        """Get protein delta (perturbation - control) for a perturbation."""
        control_prot = self.get_control_protein_profile()
        real_prot = self.get_real_protein_profile(perturbation_name)
        delta = {}
        all_markers = set(control_prot.keys()) | set(real_prot.keys())
        for marker in all_markers:
            control_val = control_prot.get(marker, 0.0)
            real_val = real_prot.get(marker, 0.0)
            delta[marker] = real_val - control_val
        return delta
    
    def get_perturbation_embedding(self, perturbation_name: str) -> Optional[np.ndarray]:
        """
        Get pre-computed aligned perturbation embedding.
        Assumes embedding is already aligned and ready to use.
        """
        if self.embeddings is None or "perturb_emb" not in self.embeddings:
            return None
        pert_cells = self.get_perturbation_cells(perturbation_name)
        if not pert_cells:
            return None
        cell_to_idx = {cell: i for i, cell in enumerate(self.cells)}
        pert_indices = [cell_to_idx[cell] for cell in pert_cells if cell in cell_to_idx]
        if not pert_indices:
            return None
        pert_embs = self.embeddings["perturb_emb"][pert_indices]
        if isinstance(pert_embs, torch.Tensor):
            pert_embs = pert_embs.numpy()
        # Return average embedding (already aligned, no transformation needed)
        return pert_embs.mean(axis=0)
    
    def get_rna_embedding(self, perturbation_name: str) -> Optional[np.ndarray]:
        """
        Get pre-computed aligned RNA embedding.
        Assumes embedding is already aligned with perturbation embedding.
        """
        if self.embeddings is None or "rna_emb" not in self.embeddings:
            return None
        pert_cells = self.get_perturbation_cells(perturbation_name)
        if not pert_cells:
            return None
        cell_to_idx = {cell: i for i, cell in enumerate(self.cells)}
        pert_indices = [cell_to_idx[cell] for cell in pert_cells if cell in cell_to_idx]
        if not pert_indices:
            return None
        rna_embs = self.embeddings["rna_emb"][pert_indices]
        if isinstance(rna_embs, torch.Tensor):
            rna_embs = rna_embs.numpy()
        # Return average embedding (already aligned)
        return rna_embs.mean(axis=0)
    
    def get_protein_embedding(self, perturbation_name: str) -> Optional[np.ndarray]:
        """
        Get pre-computed aligned protein embedding.
        Assumes embedding is already aligned with RNA and perturbation embeddings.
        """
        if self.embeddings is None or "prot_emb" not in self.embeddings:
            return None
        pert_cells = self.get_perturbation_cells(perturbation_name)
        if not pert_cells:
            return None
        cell_to_idx = {cell: i for i, cell in enumerate(self.cells)}
        pert_indices = [cell_to_idx[cell] for cell in pert_cells if cell in cell_to_idx]
        if not pert_indices:
            return None
        prot_embs = self.embeddings["prot_emb"][pert_indices]
        if isinstance(prot_embs, torch.Tensor):
            prot_embs = prot_embs.numpy()
        # Return average embedding (already aligned)
        return prot_embs.mean(axis=0)
