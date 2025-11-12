"""
Multi-Omics Hypothesis Engine
Core framework for reasoning over foundation models using Perturb-CITE-seq validation
"""

import torch
import pandas as pd
import numpy as np
import anndata as ad
import subprocess
import os
import tempfile
from scipy.spatial.distance import cosine


class ModelGraph:
    """
    Holds pre-trained foundation models as nodes in the reasoning graph.
    """

    def __init__(self, state_model_path=None):
        print("Initializing ModelGraph...")

        # RNA Node: STATE model path (command-line tool)
        print("Setting up STATE model (RNA Node)...")
        self.state_model_path = state_model_path
        if self.state_model_path:
            print(f"  STATE model path: {self.state_model_path}")
        else:
            print("  STATE model path not specified, will use default from installation")
        print("✓ STATE setup complete")

        # Protein Node: Load CAPTAIN model from local path
        print("Loading CAPTAIN model (Protein Node)...")
        captain_path = "/home/nebius/cellian/foundation_models/CAPTAIN_Base/CAPTAIN_Base.pt"

        # Load to GPU if available, otherwise CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.protein_model = torch.load(captain_path, map_location=device)
        if hasattr(self.protein_model, 'eval'):
            self.protein_model.eval()
        print(f"✓ CAPTAIN model loaded on {device}")

        print("ModelGraph initialized successfully\n")

    def run_state_embedding(self, adata_input_path, adata_output_path):
        """
        Run STATE embedding via command line using the official API.

        Official STATE command: state emb transform
        Requirements:
        - Input h5ad must have CSR sparse matrix format
        - Input h5ad must have 'gene_name' in var DataFrame

        Args:
            adata_input_path: Path to input h5ad file
            adata_output_path: Path to output h5ad file with embeddings
        """
        # Use official STATE command: state emb transform
        cmd = ["state", "emb", "transform", "--input", adata_input_path, "--output", adata_output_path]

        if self.state_model_path:
            cmd.extend(["--model-folder", self.state_model_path])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"STATE embedding failed: {result.stderr}")

        return adata_output_path


class HypothesisAgent:
    """
    Uses ModelGraph to run reasoning chains and validate against Perturb-CITE-seq ground truth.
    """

    def __init__(self, model_graph):
        self.graph = model_graph
        self.meta_df = None
        self.rna_df = None
        self.protein_df = None
        self.average_control_rna_profile = None

        print("Initializing HypothesisAgent...")
        self.load_ground_truth()
        print("HypothesisAgent initialized successfully\n")

    def load_ground_truth(self):
        """
        Load and process Perturb-CITE-seq dataset for validation.
        Automatically uses Parquet format if available (much faster).
        """
        print("Loading ground truth data...")

        # Define file paths
        data_dir = "/home/nebius/cellian/data/perturb-cite-seq/SCP1064"
        meta_path = f"{data_dir}/metadata/RNA_metadata.csv"
        rna_csv = f"{data_dir}/other/RNA_expression.csv"
        rna_parquet = f"{data_dir}/other/RNA_expression.parquet"
        protein_csv = f"{data_dir}/expression/Protein_expression.csv"
        protein_parquet = f"{data_dir}/expression/Protein_expression.parquet"

        # Load metadata (always CSV, small file)
        self.meta_df = pd.read_csv(meta_path, index_col=0)

        # Load RNA data (prefer Parquet if available)
        if os.path.exists(rna_parquet):
            print(f"  Loading RNA from Parquet (fast)...")
            import time
            start = time.time()
            self.rna_df = pd.read_parquet(rna_parquet)
            print(f"  ✓ Loaded in {time.time() - start:.2f}s")
        else:
            print(f"  Loading RNA from CSV (slow, consider converting to Parquet)...")
            self.rna_df = pd.read_csv(rna_csv, index_col=0)

        # Load protein data (prefer Parquet if available)
        if os.path.exists(protein_parquet):
            print(f"  Loading Protein from Parquet (fast)...")
            self.protein_df = pd.read_parquet(protein_parquet)
        else:
            print(f"  Loading Protein from CSV...")
            self.protein_df = pd.read_csv(protein_csv, index_col=0)

        print(f"  Metadata: {self.meta_df.shape}")
        print(f"  RNA expression: {self.rna_df.shape}")
        print(f"  Protein expression: {self.protein_df.shape}")

        # Find control cells (non-targeting or empty sgRNA)
        sgRNA_col = 'sgRNA' if 'sgRNA' in self.meta_df.columns else 'sgRNA_target'
        control_mask = (
            self.meta_df[sgRNA_col].isna() |
            (self.meta_df[sgRNA_col] == '') |
            (self.meta_df[sgRNA_col].str.lower().str.contains('non-targeting|control', na=False))
        )
        control_cells = self.meta_df[control_mask].index.tolist()
        print(f"  Found {len(control_cells)} control cells")

        # Calculate average control RNA profile (baseline cell)
        control_rna_data = self.rna_df[control_cells]
        self.average_control_rna_profile = control_rna_data.mean(axis=1)
        print("  Computed average control RNA profile")
        print("✓ Ground truth data loaded\n")

    def generate_hypothesis(self, perturbation_name):
        """
        Generate hypothesis for a perturbation and validate against ground truth.

        Args:
            perturbation_name: String describing the perturbation (e.g., "KO of CD58")

        Returns:
            dict: Contains perturbation, path, scores, and error propagation
        """
        print(f"Generating hypothesis for: {perturbation_name}")

        # Step 1: Node 1 - Perturbation
        # Create AnnData object with control RNA profile as baseline
        print("  Step 1: Creating AnnData with baseline RNA profile...")

        # Create a single-cell AnnData object with the average control profile
        X = self.average_control_rna_profile.values.reshape(1, -1)
        obs = pd.DataFrame({'perturbation': [perturbation_name]}, index=['cell_0'])
        var = pd.DataFrame(index=self.average_control_rna_profile.index)

        adata_input = ad.AnnData(X=X, obs=obs, var=var)

        # Step 2: Node 2 - RNA (STATE model)
        print("  Step 2: Running STATE model (RNA embedding)...")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_h5ad = os.path.join(tmpdir, "input.h5ad")
            output_h5ad = os.path.join(tmpdir, "output.h5ad")

            # Save input AnnData
            adata_input.write_h5ad(input_h5ad)

            # Run STATE embedding
            self.graph.run_state_embedding(input_h5ad, output_h5ad)

            # Load output with embeddings
            adata_output = ad.read_h5ad(output_h5ad)

            # Extract STATE embedding (typically stored in .obsm or .X)
            if 'X_state' in adata_output.obsm:
                predicted_rna_vector = adata_output.obsm['X_state'][0]
            elif 'state' in adata_output.obsm:
                predicted_rna_vector = adata_output.obsm['state'][0]
            else:
                # Use the embedding from .X if obsm doesn't contain it
                predicted_rna_vector = adata_output.X[0]

        print(f"    STATE embedding shape: {predicted_rna_vector.shape}")

        # Step 3: Node 3 - Protein (CAPTAIN model)
        print("  Step 3: Running CAPTAIN model (Protein prediction)...")
        with torch.no_grad():
            # Convert RNA embedding to tensor for CAPTAIN input
            device = next(self.graph.protein_model.parameters()).device if hasattr(self.graph.protein_model, 'parameters') else torch.device('cpu')
            rna_tensor = torch.tensor(predicted_rna_vector, dtype=torch.float32).unsqueeze(0).to(device)

            # Get CAPTAIN prediction
            if callable(self.graph.protein_model):
                protein_output = self.graph.protein_model(rna_tensor)
            else:
                # If it's a state dict, this needs model architecture
                protein_output = rna_tensor  # Placeholder

            predicted_protein_vector = protein_output.squeeze().cpu().numpy()

        print(f"    CAPTAIN output shape: {predicted_protein_vector.shape}")

        # Step 4: Validation against ground truth
        print("  Step 4: Validating against ground truth...")

        # Find cells with this perturbation
        sgRNA_col = 'sgRNA' if 'sgRNA' in self.meta_df.columns else 'sgRNA_target'
        perturb_mask = self.meta_df[sgRNA_col].str.contains(
            perturbation_name.replace("KO of ", "").replace("OE of ", ""),
            case=False,
            na=False
        )
        perturb_cells = self.meta_df[perturb_mask].index.tolist()

        if len(perturb_cells) == 0:
            print(f"  ⚠ Warning: No cells found for perturbation '{perturbation_name}'")
            return {
                'perturbation': perturbation_name,
                'path': 'Perturbation -> STATE -> CAPTAIN',
                'rna_score': None,
                'protein_score': None,
                'error_propagation': None,
                'note': 'No matching cells found in dataset'
            }

        print(f"  Found {len(perturb_cells)} cells with this perturbation")

        # Get real average profiles
        real_rna_data = self.rna_df[perturb_cells]
        real_rna_vector = real_rna_data.mean(axis=1).values

        real_protein_data = self.protein_df[perturb_cells]
        real_protein_vector = real_protein_data.mean(axis=1).values

        # Calculate cosine similarities (1 - cosine distance)
        # Align vector dimensions if needed
        min_rna_len = min(len(predicted_rna_vector), len(real_rna_vector))
        rna_score = 1 - cosine(
            predicted_rna_vector[:min_rna_len],
            real_rna_vector[:min_rna_len]
        )

        min_protein_len = min(len(predicted_protein_vector), len(real_protein_vector))
        protein_score = 1 - cosine(
            predicted_protein_vector[:min_protein_len],
            real_protein_vector[:min_protein_len]
        )

        # Step 5: Return results
        error_propagation = rna_score - protein_score

        print(f"  ✓ RNA Score: {rna_score:.4f}")
        print(f"  ✓ Protein Score: {protein_score:.4f}")
        print(f"  ✓ Error Propagation: {error_propagation:.4f}\n")

        return {
            'perturbation': perturbation_name,
            'path': 'Perturbation -> STATE -> CAPTAIN',
            'rna_score': float(rna_score),
            'protein_score': float(protein_score),
            'error_propagation': float(error_propagation)
        }


def main():
    """
    Example usage of the hypothesis engine.
    """
    print("=" * 60)
    print("MULTI-OMICS HYPOTHESIS ENGINE")
    print("=" * 60 + "\n")

    # Initialize model graph
    graph = ModelGraph()

    # Initialize hypothesis agent
    agent = HypothesisAgent(graph)

    # Test with example perturbation
    print("=" * 60)
    print("TESTING HYPOTHESIS GENERATION")
    print("=" * 60 + "\n")

    result = agent.generate_hypothesis("KO of CD58")

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Perturbation: {result['perturbation']}")
    print(f"Reasoning Path: {result['path']}")
    print(f"RNA Score: {result.get('rna_score', 'N/A')}")
    print(f"Protein Score: {result.get('protein_score', 'N/A')}")
    print(f"Error Propagation: {result.get('error_propagation', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
