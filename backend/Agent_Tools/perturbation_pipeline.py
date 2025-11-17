#!/usr/bin/env python3
"""
Perturbation → Transcriptomics → Proteomics Pipeline

This is the main CLI tool that orchestrates the entire pipeline:
1. Takes a perturbation (gene name) as input
2. Runs state inference: perturbation → transcriptomics (RNA)
3. Runs scTranslator inference: transcriptomics (RNA) → proteomics
4. Evaluates predictions at each step against ground truth
5. Saves results in temporary folder

Usage:
    python perturbation_pipeline.py --perturbation GENE_NAME
    
Example:
    python perturbation_pipeline.py --target-gene ACTB
    python perturbation_pipeline.py --target-gene AARS
"""

import os
import sys
import argparse
import tempfile
import shutil
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from state_inference import run_state_inference, load_hvg_genes, filter_to_hvg, load_perturbation_names
from drug_inference import run_drug_inference as run_st_tahoe_inference
from sctranslator_inference import (
    run_id_mapping,
    prepare_protein_data,
    create_dummy_protein_file,
    run_sctranslator_inference,
    save_predictions_as_adata
)

# Try importing evaluation module
try:
    from evaluation import evaluate_predictions, evaluate_by_perturbation
except ImportError:
    # If import fails, define a simple version
    def evaluate_predictions(y_true, y_pred, **kwargs):
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        from scipy.stats import pearsonr
        
        # Flatten if needed
        if y_true.ndim > 1:
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
        else:
            y_true_flat = y_true
            y_pred_flat = y_pred
        
        # Remove NaN
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]
        
        if len(y_true_clean) == 0:
            return {'r2': np.nan, 'pearson_r': np.nan, 'rmse': np.nan, 'mae': np.nan}
        
        r2 = r2_score(y_true_clean, y_pred_clean)
        r, _ = pearsonr(y_true_clean, y_pred_clean) if len(y_true_clean) > 1 else (np.nan, np.nan)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        
        print(f"    R2 Score: {r2:.4f}")
        print(f"    Pearson R: {r:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAE: {mae:.4f}")
        
        return {'r2': r2, 'pearson_r': r, 'rmse': rmse, 'mae': mae}
    
    def evaluate_by_perturbation(y_true, y_pred, perturbations, **kwargs):
        return {}


# Default paths for gene perturbation
DEFAULT_STATE_MODEL_DIR = "/home/nebius/state/test_replogle/hepg2_holdout"
DEFAULT_STATE_CHECKPOINT = "/home/nebius/state/test_replogle/hepg2_holdout/checkpoints/last.ckpt"
DEFAULT_CONDITION_DATA_DIR = "/home/nebius/cellian/data/perturb-cite-seq/SCP1064/processed/condition_splits"

# Default paths for drug perturbation (ST-Tahoe)
DEFAULT_ST_TAHOE_MODEL_DIR = "/home/nebius/ST-Tahoe"
DEFAULT_ST_TAHOE_CHECKPOINT = "/home/nebius/ST-Tahoe/final.ckpt"

# Common paths
DEFAULT_SCTRANSLATOR_CHECKPOINT = "/home/nebius/scTranslator/checkpoint/expression_fine-tuned_scTranslator.pt"
# Ground truth paths - different files for genetic vs drug perturbations
DEFAULT_GROUND_TRUTH_GENETIC = "/home/nebius/cellian/data/perturb-cite-seq/SCP1064/processed/scp1064_aligned_to_model.h5ad"
DEFAULT_GROUND_TRUTH_DRUG = "/home/nebius/cellian/data/perturb-cite-seq/SCP1064/processed/scp1064_aligned_to_model_tahoe.h5ad"
# Legacy paths (kept for backward compatibility)
DEFAULT_GROUND_TRUTH_RNA = "/home/nebius/cellian/data/perturb-cite-seq/SCP1064/expression/RNA_expression_combined_mapped.h5ad"
DEFAULT_PROTEIN_FILE = "/home/nebius/cellian/data/perturb-cite-seq/SCP1064/expression/protein_expression.h5ad"
DEFAULT_GROUND_TRUTH_PROTEIN = "/home/nebius/cellian/data/perturb-cite-seq/SCP1064/expression/protein_expression_mapped.h5ad"
DEFAULT_SCTRANSLATOR_DIR = "/home/nebius/scTranslator"


def sanitize_filename(name):
    """
    Sanitize a string to be used as a filename by replacing special characters.
    
    Args:
        name: String to sanitize
    
    Returns:
        Sanitized string safe for use in filenames
    """
    import re
    # Replace special characters with underscores
    sanitized = re.sub(r'[^\w\s-]', '_', str(name))
    # Replace spaces with underscores
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Limit length to avoid filesystem issues
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized


def get_condition_adata_path(condition, perturbation_type, condition_data_dir=DEFAULT_CONDITION_DATA_DIR):
    """
    Get the adata file path for a given condition and perturbation type.
    
    Args:
        condition: Condition name (IFNγ, Control, Co-Culture)
        perturbation_type: 'gene' or 'drug'
        condition_data_dir: Base directory for condition-specific adata files
    
    Returns:
        Path to condition-specific adata file
    """
    if perturbation_type == 'gene':
        filename = f"scp1064_genetic_{condition}.h5ad"
    else:  # drug
        filename = f"scp1064_drug_{condition}.h5ad"
    
    return os.path.join(condition_data_dir, filename)


def create_perturbation_data(condition_adata_path, target_gene, output_path, var_dims_path=None):
    """
    Create perturbation data from condition-specific adata.
    
    IMPORTANT BEHAVIOR:
    - If target_gene IS in the target_gene column: Filter to those cells (normal protocol)
    - If target_gene IS NOT in the target_gene column: Follow drug protocol:
        * Filter to control cells (non-targeting)
        * Set 80% to target_gene (perturbed)
        * Set 20% to 'non-targeting' (control)
        * This allows for GSEA analysis after inference (comparing perturbed vs control)
    
    The adata file should already be aligned to the model genes.
    
    Args:
        condition_adata_path: Path to condition-specific adata file
        target_gene: Target gene name
        output_path: Path to save perturbation data
        var_dims_path: Optional path to var_dims.pkl to validate if gene is in training set
    
    Returns:
        Tuple of (Path to created file, bool: whether gene was in target_gene column)
    """
    print(f"\n{'='*70}")
    print("PREPARING GENE PERTURBATION DATA")
    print(f"{'='*70}")
    print(f"Condition adata: {condition_adata_path}")
    print(f"Target gene: {target_gene}")
    print(f"Output: {output_path}")
    
    # Check if gene is in training set (if var_dims_path provided)
    if var_dims_path and os.path.exists(var_dims_path):
        try:
            pert_names_trained = load_perturbation_names(var_dims_path)
            pert_set_trained = set(pert_names_trained)
            if target_gene not in pert_set_trained:
                print(f"\n{'!'*70}")
                print("⚠️  WARNING: Gene NOT in Model Training Set")
                print(f"{'!'*70}")
                print(f"\nGene '{target_gene}' is NOT in the model's training perturbations.")
                print(f"Model was trained on {len(pert_names_trained)} perturbations.")
                print(f"\n⚠️  CRITICAL: STATE will treat '{target_gene}' as a CONTROL perturbation!")
                print(f"   Predictions for this gene will be identical to control predictions.")
                print(f"   Results will NOT be meaningful for this gene.")
                print(f"\n{'!'*70}\n")
        except Exception as e:
            print(f"  Warning: Could not validate gene against training set: {e}")
    
    # Load condition adata
    condition_adata = ad.read_h5ad(condition_adata_path)
    print(f"  Condition adata shape: {condition_adata.shape} (cells × genes)")
    
    # Check if target_gene exists in the target_gene column
    gene_found_in_target_gene = False
    
    if 'target_gene' in condition_adata.obs.columns:
        print(f"  Found 'target_gene' column in condition adata")
        unique_genes = condition_adata.obs['target_gene'].unique()
        print(f"  Available target_gene values (first 10): {list(unique_genes[:10])}")
        
        mask = condition_adata.obs['target_gene'] == target_gene
        if mask.sum() > 0:
            # Gene IS in target_gene column - use normal protocol
            gene_found_in_target_gene = True
            condition_adata = condition_adata[mask].copy()
            print(f"  ✓ Gene '{target_gene}' found in target_gene column")
            print(f"  Filtered to {mask.sum()} cells with target_gene: {target_gene}")
            print(f"  Using normal protocol: Will validate against perturb-cite-seq data")
        else:
            # Gene NOT in target_gene column - use drug protocol
            print(f"  ✗ Gene '{target_gene}' NOT found in target_gene column")
            print(f"  Using drug-like protocol: Filter to control cells, apply 80/20 split")
            print(f"  Will NOT validate against perturb-cite-seq data")
            
            # Filter to control cells (non-targeting)
            control_mask = condition_adata.obs['target_gene'] == 'non-targeting'
            if control_mask.sum() > 0:
                condition_adata = condition_adata[control_mask].copy()
                print(f"  Filtered to {control_mask.sum()} control cells (non-targeting)")
            else:
                print(f"  Warning: No 'non-targeting' cells found in target_gene column")
                print(f"  Available values: {unique_genes[:10]}")
                raise ValueError("No control cells (non-targeting) found for drug-like protocol")
            
            # Create 80/20 split: 80% perturbed, 20% control
            n_cells = condition_adata.n_obs
            n_perturbed = int(n_cells * 0.8)
            n_control = n_cells - n_perturbed
            
            # Initialize all as control
            condition_adata.obs['target_gene'] = 'non-targeting'
            
            # Randomly select 80% of cells for perturbation
            np.random.seed(42)  # For reproducibility
            perturbed_indices = np.random.choice(condition_adata.obs_names, size=n_perturbed, replace=False)
            condition_adata.obs.loc[perturbed_indices, 'target_gene'] = target_gene
            
            print(f"  Applied gene perturbation to {n_perturbed} cells (80%)")
            print(f"  Set {n_control} cells (20%) as control: 'non-targeting'")
            print(f"  This allows for GSEA analysis after inference (comparing perturbed vs control)")
            
    elif 'perturbation' in condition_adata.obs.columns:
        # Fallback to old column name
        print(f"  Found 'perturbation' column in condition adata (using old column name)")
        unique_genes = condition_adata.obs['perturbation'].unique()
        print(f"  Available perturbation values (first 10): {list(unique_genes[:10])}")
        
        mask = condition_adata.obs['perturbation'] == target_gene
        if mask.sum() > 0:
            # Gene IS in perturbation column - use normal protocol
            gene_found_in_target_gene = True
            condition_adata = condition_adata[mask].copy()
            condition_adata.obs['target_gene'] = target_gene
            print(f"  ✓ Gene '{target_gene}' found in perturbation column")
            print(f"  Filtered to {mask.sum()} cells with perturbation: {target_gene}")
            print(f"  Using normal protocol: Will validate against perturb-cite-seq data")
        else:
            # Gene NOT in perturbation column - use drug protocol
            print(f"  ✗ Gene '{target_gene}' NOT found in perturbation column")
            print(f"  Using drug-like protocol: Filter to control cells, apply 80/20 split")
            print(f"  Will NOT validate against perturb-cite-seq data")
            
            # Filter to control cells
            control_mask = condition_adata.obs['perturbation'] == 'non-targeting'
            if control_mask.sum() == 0:
                control_mask = condition_adata.obs['perturbation'] == 'control'
            
            if control_mask.sum() > 0:
                condition_adata = condition_adata[control_mask].copy()
                print(f"  Filtered to {control_mask.sum()} control cells")
            else:
                print(f"  Warning: No control cells found")
                raise ValueError("No control cells found for drug-like protocol")
            
            # Create 80/20 split
            n_cells = condition_adata.n_obs
            n_perturbed = int(n_cells * 0.8)
            n_control = n_cells - n_perturbed
            
            condition_adata.obs['target_gene'] = 'non-targeting'
            np.random.seed(42)
            perturbed_indices = np.random.choice(condition_adata.obs_names, size=n_perturbed, replace=False)
            condition_adata.obs.loc[perturbed_indices, 'target_gene'] = target_gene
            
            print(f"  Applied gene perturbation to {n_perturbed} cells (80%)")
            print(f"  Set {n_control} cells (20%) as control: 'non-targeting'")
    else:
        print(f"  Warning: 'target_gene' or 'perturbation' column not found in condition adata")
        print(f"  Available columns: {list(condition_adata.obs.columns)}")
        raise ValueError("'target_gene' or 'perturbation' column required in condition adata")
    
    # Save perturbation data
    condition_adata.write_h5ad(output_path)
    print(f"  ✓ Created perturbation data: {output_path}")
    print(f"    Shape: {condition_adata.shape} (cells × genes)")
    print(f"    Number of cells: {condition_adata.n_obs:,}")
    
    return output_path, gene_found_in_target_gene


def load_available_genes(var_dims_path="/home/nebius/state/test_replogle/hepg2_holdout/var_dims.pkl"):
    """
    Load available gene perturbations from STATE model var_dims.pkl.
    
    Args:
        var_dims_path: Path to var_dims.pkl file
    
    Returns:
        List of available gene perturbations
    """
    try:
        with open(var_dims_path, 'rb') as f:
            var = pickle.load(f)
        genes = var.get("pert_names", [])
        # Convert numpy strings to regular strings
        return [str(g) for g in genes] if genes is not None else []
    except Exception as e:
        print(f"  Warning: Could not load available genes from {var_dims_path}: {str(e)}")
        return []


def load_available_drugs(var_dims_path="/home/nebius/ST-Tahoe/var_dims.pkl"):
    """
    Load available drug perturbations from ST-Tahoe var_dims.pkl.
    
    Args:
        var_dims_path: Path to var_dims.pkl file
    
    Returns:
        List of available drug perturbations in tuple format
    """
    try:
        with open(var_dims_path, 'rb') as f:
            var = pickle.load(f)
        drugs = var.get("pert_names", [])
        return list(drugs) if drugs is not None else []
    except Exception as e:
        print(f"  Warning: Could not load available drugs from {var_dims_path}: {str(e)}")
        return []


def create_drug_perturbation_data(condition_adata_path, drug_name, output_path, var_dims_path="/home/nebius/ST-Tahoe/var_dims.pkl"):
    """
    Create drug perturbation data from condition-specific adata.
    
    IMPORTANT: 
    - Filters to control cells (non-targeting) first
    - Sets 80% of cells to the user's desired drug perturbation
    - Sets 20% of cells to control: "[('DMSO_TF', 0.0, 'uM')]"
    - This allows for GSEA analysis after inference (comparing perturbed vs control)
    
    The drug_name should be in the format: "[('drugname', concentration, 'uM')]"
    Example: "[('(R)-Verapamil (hydrochloride)', 0.05, 'uM')]"
    
    Args:
        condition_adata_path: Path to condition-specific adata file (already aligned to ST-Tahoe)
        drug_name: Drug perturbation in tuple format (e.g., "[('(R)-Verapamil (hydrochloride)', 0.05, 'uM')]")
        output_path: Path to save perturbation data
        var_dims_path: Path to var_dims.pkl to validate drug name
    
    Returns:
        Path to created file
    """
    print(f"\n{'='*70}")
    print("PREPARING DRUG PERTURBATION DATA")
    print(f"{'='*70}")
    print(f"Condition adata: {condition_adata_path}")
    print(f"Drug perturbation: {drug_name}")
    print(f"Output: {output_path}")
    
    # Load available drugs to validate
    available_drugs = load_available_drugs(var_dims_path)
    if available_drugs:
        print(f"  Loaded {len(available_drugs)} available drug perturbations from ST-Tahoe model")
        # Check if drug_name is in the available list
        drug_name_clean = drug_name.strip()
        if drug_name_clean not in available_drugs:
            print(f"  Warning: Drug perturbation '{drug_name_clean}' not found in available drugs")
            print(f"  Sample available drugs (first 5):")
            for d in available_drugs[:5]:
                print(f"    {d}")
    
    # Load condition adata
    drug_adata = ad.read_h5ad(condition_adata_path)
    print(f"  Condition adata shape: {drug_adata.shape} (cells × genes)")
    
    # FIRST: Filter to control cells (non-targeting) - this makes biological sense
    if 'target_gene' in drug_adata.obs.columns:
        control_mask = drug_adata.obs['target_gene'] == 'non-targeting'
        if control_mask.sum() > 0:
            drug_adata = drug_adata[control_mask].copy()
            print(f"  Filtered to {control_mask.sum()} control cells (non-targeting)")
        else:
            print(f"  Warning: No 'non-targeting' cells found in target_gene column")
            print(f"  Available values: {drug_adata.obs['target_gene'].unique()[:10]}")
            raise ValueError("No control cells (non-targeting) found in drug data")
    else:
        print(f"  Warning: 'target_gene' column not found in drug data")
        print(f"  Available columns: {list(drug_adata.obs.columns)}")
        raise ValueError("'target_gene' column required to filter control cells")
    
    # Now create the drugname_drugconc column
    # Format should be: "[('(R)-Verapamil (hydrochloride)', 0.05, 'uM')]"
    # Use the drug_name as-is (should already be in correct format)
    drug_tuple_str = drug_name.strip()
    control_tuple_str = "[('DMSO_TF', 0.0, 'uM')]"
    
    # Set 80% to drug perturbation, 20% to control
    n_cells = drug_adata.n_obs
    n_perturbed = int(n_cells * 0.8)
    n_control = n_cells - n_perturbed
    
    # Create drugname_drugconc column
    drug_adata.obs['drugname_drugconc'] = np.str_(control_tuple_str)  # Initialize all as control
    
    # Randomly select 80% of cells for perturbation
    np.random.seed(42)  # For reproducibility
    perturbed_indices = np.random.choice(drug_adata.obs_names, size=n_perturbed, replace=False)
    drug_adata.obs.loc[perturbed_indices, 'drugname_drugconc'] = np.str_(drug_tuple_str)
    
    print(f"  Applied drug perturbation to {n_perturbed} cells (80%)")
    print(f"  Set {n_control} cells (20%) as control: {control_tuple_str}")
    print(f"  Drug perturbation format: {drug_tuple_str}")
    
    # Save perturbation data
    drug_adata.write_h5ad(output_path)
    print(f"  ✓ Created drug perturbation data: {output_path}")
    print(f"    Shape: {drug_adata.shape} (cells × genes)")
    print(f"    Number of cells: {drug_adata.n_obs:,}")
    print(f"    Perturbed cells: {n_perturbed} (80%)")
    print(f"    Control cells: {n_control} (20%)")
    
    return output_path


def load_ground_truth(ground_truth_rna_path, ground_truth_protein_path, 
                      target_gene=None, perturbation_type='gene'):
    """
    Load ground truth data for evaluation.
    
    Args:
        ground_truth_rna_path: Path to ground truth RNA h5ad (or combined file for genetic/drug)
        ground_truth_protein_path: Path to ground truth protein h5ad (or combined file for genetic/drug)
        target_gene: Optional target gene to filter by
        perturbation_type: 'gene' or 'drug' to determine which file to use
    
    Returns:
        Tuple of (RNA adata, protein adata)
    """
    print(f"\n  Loading ground truth data...")
    
    # For genetic and drug perturbations, use the specified aligned files
    # These files contain both RNA and protein data
    if perturbation_type == 'gene':
        # Use genetic perturbation ground truth file
        if ground_truth_rna_path == DEFAULT_GROUND_TRUTH_RNA:
            # Use the genetic perturbation file for both RNA and protein
            ground_truth_path = DEFAULT_GROUND_TRUTH_GENETIC
            print(f"    Using genetic perturbation ground truth: {ground_truth_path}")
            ground_truth_combined = ad.read_h5ad(ground_truth_path)
            print(f"    Ground truth shape: {ground_truth_combined.shape}")
            # Use the same file for both RNA and protein
            ground_truth_rna = ground_truth_combined
            ground_truth_protein = ground_truth_combined
        else:
            # Custom path provided, use as specified
            ground_truth_rna = ad.read_h5ad(ground_truth_rna_path)
            print(f"    Ground truth RNA shape: {ground_truth_rna.shape}")
            if ground_truth_protein_path and ground_truth_protein_path != ground_truth_rna_path:
                ground_truth_protein = ad.read_h5ad(ground_truth_protein_path)
                print(f"    Ground truth protein shape: {ground_truth_protein.shape}")
            else:
                ground_truth_protein = ground_truth_rna
    elif perturbation_type == 'drug':
        # Use drug perturbation ground truth file
        if ground_truth_rna_path == DEFAULT_GROUND_TRUTH_RNA:
            # Use the drug perturbation file for both RNA and protein
            ground_truth_path = DEFAULT_GROUND_TRUTH_DRUG
            print(f"    Using drug perturbation ground truth: {ground_truth_path}")
            ground_truth_combined = ad.read_h5ad(ground_truth_path)
            print(f"    Ground truth shape: {ground_truth_combined.shape}")
            # Use the same file for both RNA and protein
            ground_truth_rna = ground_truth_combined
            ground_truth_protein = ground_truth_combined
        else:
            # Custom path provided, use as specified
            ground_truth_rna = ad.read_h5ad(ground_truth_rna_path)
            print(f"    Ground truth RNA shape: {ground_truth_rna.shape}")
            if ground_truth_protein_path and ground_truth_protein_path != ground_truth_rna_path:
                ground_truth_protein = ad.read_h5ad(ground_truth_protein_path)
                print(f"    Ground truth protein shape: {ground_truth_protein.shape}")
            else:
                ground_truth_protein = ground_truth_rna
    else:
        # Fallback to original behavior
        ground_truth_rna = ad.read_h5ad(ground_truth_rna_path)
        print(f"    Ground truth RNA shape: {ground_truth_rna.shape}")
        ground_truth_protein = ad.read_h5ad(ground_truth_protein_path)
        print(f"    Ground truth protein shape: {ground_truth_protein.shape}")
    
    # Filter by target gene/drug if provided
    if target_gene is not None:
        if perturbation_type == 'gene':
            # For genetic perturbations, filter by target_gene
            if 'target_gene' in ground_truth_rna.obs.columns:
                mask = ground_truth_rna.obs['target_gene'] == target_gene
                if mask.sum() > 0:
                    ground_truth_rna = ground_truth_rna[mask].copy()
                    ground_truth_protein = ground_truth_protein[mask].copy()
                    print(f"    Filtered to {mask.sum()} cells with target gene: {target_gene}")
                else:
                    print(f"    Warning: No cells found with target gene: {target_gene}")
            elif 'perturbation' in ground_truth_rna.obs.columns:
                # Fallback to old column name for backward compatibility
                mask = ground_truth_rna.obs['perturbation'] == target_gene
                if mask.sum() > 0:
                    ground_truth_rna = ground_truth_rna[mask].copy()
                    ground_truth_protein = ground_truth_protein[mask].copy()
                    print(f"    Filtered to {mask.sum()} cells with perturbation: {target_gene} (using old column name)")
                else:
                    print(f"    Warning: No cells found with perturbation: {target_gene}")
            else:
                print(f"    Warning: 'target_gene' or 'perturbation' column not found in ground truth RNA")
        else:  # drug perturbation
            # For drug perturbations, filter by drugname_drugconc
            if 'drugname_drugconc' in ground_truth_rna.obs.columns:
                mask = ground_truth_rna.obs['drugname_drugconc'] == target_gene
                if mask.sum() > 0:
                    ground_truth_rna = ground_truth_rna[mask].copy()
                    ground_truth_protein = ground_truth_protein[mask].copy()
                    print(f"    Filtered to {mask.sum()} cells with drug: {target_gene}")
                else:
                    print(f"    Warning: No cells found with drug: {target_gene}")
            else:
                print(f"    Warning: 'drugname_drugconc' column not found in ground truth RNA")
    
    return ground_truth_rna, ground_truth_protein


def fix_sctranslator_checkpoint_loading(checkpoint_path, sctranslator_dir):
    """
    Fix scTranslator inference script to handle checkpoint format correctly.
    We need to modify the inference script temporarily or create a wrapper.
    
    For now, we'll create a modified version that handles both formats.
    """
    # This will be handled in the sctranslator_inference module
    pass


def main():
    parser = argparse.ArgumentParser(
        description='Perturbation → Transcriptomics → Proteomics Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline for ACTB target gene (gene perturbation)
  python perturbation_pipeline.py --target-gene ACTB
  
  # Run pipeline for AARS target gene (gene perturbation)
  python perturbation_pipeline.py --target-gene AARS
  
  # Run pipeline for a drug perturbation
  python perturbation_pipeline.py --drug DRUG_NAME --perturbation-type drug
  
  # Run with custom paths
  python perturbation_pipeline.py --target-gene ACTB \\
    --state-model-dir /path/to/state/model \\
    --sctranslator-checkpoint /path/to/sctranslator.pt
        """
    )
    
    # Perturbation type selection
    parser.add_argument('--perturbation-type', type=str, choices=['gene', 'drug'], default='gene',
                        help='Type of perturbation: gene or drug (default: gene)')
    
    # Gene perturbation arguments
    parser.add_argument('--target-gene', '--perturbation', type=str, default=None,
                        dest='target_gene',
                        help='Target gene name (e.g., ACTB, AARS). Required for gene perturbation.')
    
    # Drug perturbation arguments
    parser.add_argument('--drug', type=str, default=None,
                        help='Drug name. Required for drug perturbation.')
    
    # Gene perturbation paths
    parser.add_argument('--state-model-dir', type=str, default=DEFAULT_STATE_MODEL_DIR,
                        help=f'Path to state model directory for gene perturbation (default: {DEFAULT_STATE_MODEL_DIR})')
    parser.add_argument('--state-checkpoint', type=str, default=DEFAULT_STATE_CHECKPOINT,
                        help=f'Path to state checkpoint for gene perturbation (default: {DEFAULT_STATE_CHECKPOINT})')
    
    # Drug perturbation paths
    parser.add_argument('--st-tahoe-model-dir', type=str, default=DEFAULT_ST_TAHOE_MODEL_DIR,
                        help=f'Path to ST-Tahoe model directory for drug perturbation (default: {DEFAULT_ST_TAHOE_MODEL_DIR})')
    parser.add_argument('--st-tahoe-checkpoint', type=str, default=DEFAULT_ST_TAHOE_CHECKPOINT,
                        help=f'Path to ST-Tahoe checkpoint for drug perturbation (default: {DEFAULT_ST_TAHOE_CHECKPOINT})')
    
    # Condition data directory
    parser.add_argument('--condition-data-dir', type=str, default=DEFAULT_CONDITION_DATA_DIR,
                        help=f'Path to condition-specific adata files directory (default: {DEFAULT_CONDITION_DATA_DIR})')
    
    # Common arguments
    parser.add_argument('--sctranslator-checkpoint', type=str, default=DEFAULT_SCTRANSLATOR_CHECKPOINT,
                        help=f'Path to scTranslator checkpoint (default: {DEFAULT_SCTRANSLATOR_CHECKPOINT})')
    parser.add_argument('--ground-truth-rna', type=str, default=DEFAULT_GROUND_TRUTH_RNA,
                        help=f'Path to ground truth RNA (default: {DEFAULT_GROUND_TRUTH_RNA})')
    parser.add_argument('--ground-truth-protein', type=str, default=DEFAULT_GROUND_TRUTH_PROTEIN,
                        help=f'Path to ground truth protein (default: {DEFAULT_GROUND_TRUTH_PROTEIN})')
    parser.add_argument('--protein-file', type=str, default=DEFAULT_PROTEIN_FILE,
                        help=f'Path to protein expression file for scTranslator (default: {DEFAULT_PROTEIN_FILE})')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: temp directory in Agent_Tools)')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary files after completion')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip evaluation against ground truth')
    
    args = parser.parse_args()
    
    # Interactive mode: ask user for perturbation type and condition if not specified via command line
    if args.target_gene is None and args.drug is None:
        print("\n" + "="*70)
        print("PERTURBATION TYPE SELECTION")
        print("="*70)
        print("\nPlease select perturbation type:")
        print("  1. Gene perturbation (e.g., ACTB, AARS)")
        print("  2. Drug perturbation")
        
        while True:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice == '1':
                args.perturbation_type = 'gene'
                # Load and show available genes
                var_dims_path = "/home/nebius/state/test_replogle/hepg2_holdout/var_dims.pkl"
                available_genes = load_available_genes(var_dims_path)
                if available_genes:
                    print(f"\n  Found {len(available_genes)} available gene perturbations")
                    print(f"  Sample genes (first 10):")
                    for i, gene in enumerate(available_genes[:10], 1):
                        print(f"    {i}. {gene}")
                    print(f"  ... and {len(available_genes) - 10} more")
                    print(f"\n  Note: Only genes in the training set can produce meaningful predictions")
                else:
                    print(f"\n  Warning: Could not load available genes")
                
                args.target_gene = input("\nEnter target gene name (e.g., ACTB, AARS): ").strip()
                if not args.target_gene:
                    print("Error: Target gene name is required")
                    sys.exit(1)
                break
            elif choice == '2':
                args.perturbation_type = 'drug'
                # Load and show available drugs
                var_dims_path = "/home/nebius/ST-Tahoe/var_dims.pkl"
                available_drugs = load_available_drugs(var_dims_path)
                if available_drugs:
                    print(f"\n  Found {len(available_drugs)} available drug perturbations")
                    print(f"  Sample drugs (first 10):")
                    for i, drug in enumerate(available_drugs[:10], 1):
                        print(f"    {i}. {drug}")
                    print(f"  ... and {len(available_drugs) - 10} more")
                    print(f"\n  Format: [('drugname', concentration, 'uM')]")
                    print(f"  Example: [('(R)-Verapamil (hydrochloride)', 0.05, 'uM')]")
                else:
                    print(f"\n  Warning: Could not load available drugs")
                    print(f"  Format: [('drugname', concentration, 'uM')]")
                    print(f"  Example: [('(R)-Verapamil (hydrochloride)', 0.05, 'uM')]")
                
                args.drug = input("\nEnter drug perturbation in format [('drugname', concentration, 'uM')]: ").strip()
                if not args.drug:
                    print("Error: Drug perturbation is required")
                    sys.exit(1)
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    # Ask for condition if not specified
    if not hasattr(args, 'condition') or args.condition is None:
        print("\n" + "="*70)
        print("CONDITION SELECTION")
        print("="*70)
        print("\nPlease select condition:")
        print("  1. IFNγ")
        print("  2. Control")
        print("  3. Co-Culture")
        
        while True:
            choice = input("\nEnter choice (1, 2, or 3): ").strip()
            if choice == '1':
                args.condition = 'IFNγ'
                break
            elif choice == '2':
                args.condition = 'Control'
                break
            elif choice == '3':
                args.condition = 'Co-Culture'
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Validate arguments based on perturbation type
    if args.perturbation_type == 'gene':
        if args.target_gene is None:
            parser.error("--target-gene is required for gene perturbation")
        perturbation_name = args.target_gene
    elif args.perturbation_type == 'drug':
        if args.drug is None:
            parser.error("--drug is required for drug perturbation")
        perturbation_name = args.drug
    else:
        parser.error(f"Invalid perturbation type: {args.perturbation_type}")
    
    # Validate condition
    if not hasattr(args, 'condition') or args.condition is None:
        parser.error("--condition is required (IFNγ, Control, or Co-Culture)")
    
    # Get condition-specific adata path
    condition_adata_path = get_condition_adata_path(
        args.condition, 
        args.perturbation_type,
        args.condition_data_dir
    )
    
    if not os.path.exists(condition_adata_path):
        raise FileNotFoundError(
            f"Condition adata file not found: {condition_adata_path}\n"
            f"Please ensure the file exists in {args.condition_data_dir}"
        )
    
    # Create output directory
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "temp_output")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("PERTURBATION → TRANSCRIPTOMICS → PROTEOMICS PIPELINE")
    print("="*70)
    print(f"\nPerturbation type: {args.perturbation_type}")
    print(f"Perturbation name: {perturbation_name}")
    print(f"Condition: {args.condition}")
    print(f"Condition adata: {condition_adata_path}")
    print(f"Output directory: {output_dir}")
    
    if args.perturbation_type == 'gene':
        print(f"State model: {args.state_model_dir}")
    else:
        print(f"ST-Tahoe model: {args.st_tahoe_model_dir}")
    print(f"scTranslator checkpoint: {args.sctranslator_checkpoint}")
    
    try:
        # Sanitize perturbation name for use in filenames
        sanitized_pert_name = sanitize_filename(perturbation_name)
        
        # Step 1: Create perturbation data
        if args.perturbation_type == 'gene':
            pert_data_path = os.path.join(output_dir, f"perturbation_{sanitized_pert_name}_{args.condition}.h5ad")
            # Get var_dims path for validation
            state_var_dims_path = os.path.join(args.state_model_dir, "var_dims.pkl")
            pert_data_path, gene_in_target_gene = create_perturbation_data(
                condition_adata_path,
                perturbation_name,
                pert_data_path,
                var_dims_path=state_var_dims_path if os.path.exists(state_var_dims_path) else None
            )
            # Track whether we should skip validation (gene not in target_gene column)
            skip_validation_gene_not_found = not gene_in_target_gene
            if skip_validation_gene_not_found:
                print(f"\n  NOTE: Gene '{perturbation_name}' not found in target_gene column.")
                print(f"  Using drug-like protocol (80/20 split). Validation will be SKIPPED.")
            # Set inference parameters for gene perturbation
            model_dir = args.state_model_dir
            checkpoint_path = args.state_checkpoint
            pert_col = "target_gene"
        else:  # drug perturbation
            skip_validation_gene_not_found = False  # For drug perturbations, already handled
            pert_data_path = os.path.join(output_dir, f"drug_perturbation_{sanitized_pert_name}_{args.condition}.h5ad")
            create_drug_perturbation_data(
                condition_adata_path,
                perturbation_name,
                pert_data_path
            )
            # Set inference parameters for drug perturbation
            model_dir = args.st_tahoe_model_dir
            checkpoint_path = args.st_tahoe_checkpoint
            pert_col = "drugname_drugconc"
        
        # Step 2: Run inference (perturbation → RNA)
        if args.perturbation_type == 'gene':
            rna_pred_path = os.path.join(output_dir, f"rna_predictions_{sanitized_pert_name}_{args.condition}.h5ad")
            run_state_inference(
                model_dir=model_dir,
                checkpoint_path=checkpoint_path,
                input_adata_path=pert_data_path,
                output_path=rna_pred_path,
                pert_col=pert_col
            )
        else:  # drug perturbation - output should be a file, not directory
            rna_pred_path = os.path.join(output_dir, f"rna_predictions_{sanitized_pert_name}_{args.condition}.h5ad")
            rna_pred_path = run_st_tahoe_inference(
                model_dir=model_dir,
                checkpoint_path=checkpoint_path,
                input_adata_path=pert_data_path,
                output_path=rna_pred_path,  # Pass file path, not directory
                pert_col=pert_col
            )
        
        # Step 3: Load ground truth RNA for evaluation
        ground_truth_rna = None
        ground_truth_protein = None
        
        if not args.skip_evaluation and args.perturbation_type == 'gene' and not skip_validation_gene_not_found:
            # Only load ground truth if gene was found in target_gene column
            # For genes not in target_gene, skip validation
            filter_value = perturbation_name
            ground_truth_rna, ground_truth_protein = load_ground_truth(
                args.ground_truth_rna,
                args.ground_truth_protein,
                target_gene=filter_value,
                perturbation_type=args.perturbation_type
            )
        elif not args.skip_evaluation and args.perturbation_type == 'gene' and skip_validation_gene_not_found:
            print(f"\n  Skipping ground truth loading: Gene '{perturbation_name}' not in target_gene column")
            ground_truth_rna = None
            ground_truth_protein = None
        elif not args.skip_evaluation and args.perturbation_type == 'drug':
            # Load ground truth for drug perturbations
            filter_value = perturbation_name
            ground_truth_rna, ground_truth_protein = load_ground_truth(
                args.ground_truth_rna,
                args.ground_truth_protein,
                target_gene=filter_value,
                perturbation_type=args.perturbation_type
            )
        
        # Step 3b: Evaluate RNA predictions (only if ground truth available)
        if not args.skip_evaluation and ground_truth_rna is not None:
            print(f"\n{'='*70}")
            print("EVALUATING RNA PREDICTIONS")
            print(f"{'='*70}")
            
            pred_rna = ad.read_h5ad(rna_pred_path)
            
            # Filter ground truth to only cells with perturbation
            if args.perturbation_type == 'gene':
                filter_col = 'target_gene'
            else:  # drug
                filter_col = 'drugname_drugconc'
            
            if filter_col in ground_truth_rna.obs.columns:
                gt_mask = ground_truth_rna.obs[filter_col] == perturbation_name
                if gt_mask.sum() > 0:
                    ground_truth_rna_filtered = ground_truth_rna[gt_mask].copy()
                    print(f"  Filtered ground truth RNA to {gt_mask.sum()} cells with {filter_col}: {perturbation_name}")
                else:
                    print(f"  Warning: No cells found with {filter_col}: {perturbation_name} in ground truth RNA")
                    print(f"  Using all ground truth cells")
                    ground_truth_rna_filtered = ground_truth_rna.copy()
            elif 'perturbation' in ground_truth_rna.obs.columns:
                # Fallback to old column name
                gt_mask = ground_truth_rna.obs['perturbation'] == perturbation_name
                if gt_mask.sum() > 0:
                    ground_truth_rna_filtered = ground_truth_rna[gt_mask].copy()
                    print(f"  Filtered ground truth RNA to {gt_mask.sum()} cells with perturbation: {perturbation_name} (using old column name)")
                else:
                    print(f"  Warning: No cells found with perturbation: {perturbation_name} in ground truth RNA")
                    print(f"  Using all ground truth cells")
                    ground_truth_rna_filtered = ground_truth_rna.copy()
            else:
                print(f"  Warning: '{filter_col}' or 'perturbation' column not found in ground truth RNA")
                print(f"  Using all ground truth cells")
                ground_truth_rna_filtered = ground_truth_rna.copy()
            
            # Align genes
            common_genes = pred_rna.var.index.intersection(ground_truth_rna_filtered.var.index)
            if len(common_genes) > 0:
                # Align cells - use common cells between predictions and filtered ground truth
                common_cells = pred_rna.obs_names.intersection(ground_truth_rna_filtered.obs_names)
                if len(common_cells) > 0:
                    print(f"  Evaluating on {len(common_cells)} cells with {filter_col}: {perturbation_name}")
                    pred_rna_aligned = pred_rna[common_cells, common_genes].X
                    truth_rna_aligned = ground_truth_rna_filtered[common_cells, common_genes].X
                    
                    # Convert sparse to dense if needed
                    if hasattr(pred_rna_aligned, 'toarray'):
                        pred_rna_aligned = pred_rna_aligned.toarray()
                    if hasattr(truth_rna_aligned, 'toarray'):
                        truth_rna_aligned = truth_rna_aligned.toarray()
                    
                    rna_metrics = evaluate_predictions(truth_rna_aligned, pred_rna_aligned)
                    
                    print(f"\n  RNA Prediction Metrics (cells with {perturbation_name} {args.perturbation_type} perturbation):")
                    print(f"    R2 Score: {rna_metrics['r2']:.4f}")
                    print(f"    Pearson R: {rna_metrics['pearson_r']:.4f}")
                    print(f"    RMSE: {rna_metrics['rmse']:.4f}")
                    print(f"    MAE: {rna_metrics['mae']:.4f}")
                else:
                    print(f"  Warning: No common cells between predictions and filtered ground truth")
            else:
                print(f"  Warning: No common genes between predictions and ground truth")
        
        # Step 4: Prepare RNA for scTranslator (map gene IDs)
        print(f"\n{'='*70}")
        print("PREPARING RNA FOR SCTRANSLATOR")
        print(f"{'='*70}")
        
        rna_mapped_path = os.path.join(output_dir, f"rna_predictions_{sanitized_pert_name}_{args.condition}_mapped.h5ad")
        
        # Check if already mapped
        pred_rna = ad.read_h5ad(rna_pred_path)
        if 'my_Id' not in pred_rna.var.columns:
            print(f"  RNA predictions need ID mapping...")
            run_id_mapping(
                rna_pred_path,
                rna_mapped_path,
                gene_type='human_gene_symbol',
                gene_column='index',
                sctranslator_dir=DEFAULT_SCTRANSLATOR_DIR
            )
        else:
            print(f"  RNA predictions already have my_Id column")
            shutil.copy(rna_pred_path, rna_mapped_path)
        
        # Step 5: Prepare protein data from real protein file
        print(f"\n{'='*70}")
        print("PREPARING PROTEIN DATA FOR SCTRANSLATOR")
        print(f"{'='*70}")
        
        protein_file_path = args.protein_file
        protein_prepared_path = os.path.join(output_dir, f"protein_prepared_{sanitized_pert_name}_{args.condition}.h5ad")
        
        # Prepare protein data: filter to match RNA predictions (same perturbation) and align cells
        # Also adds all proteins from RNA gene names (in addition to original 14)
        protein_prepared_path, original_protein_names = prepare_protein_data(
            rna_adata_path=rna_mapped_path,
            protein_file_path=protein_file_path,
            target_gene=perturbation_name,
            output_path=protein_prepared_path,
            is_drug=(args.perturbation_type == 'drug')
        )
        
        print(f"\n  Original protein names for validation ({len(original_protein_names)}):")
        print(f"    {original_protein_names}")
        
        # Map protein if needed
        protein_mapped_path = os.path.join(output_dir, f"protein_prepared_{sanitized_pert_name}_{args.condition}_mapped.h5ad")
        
        protein_prepared = ad.read_h5ad(protein_prepared_path)
        if 'my_Id' not in protein_prepared.var.columns:
            print(f"  Protein needs ID mapping...")
            run_id_mapping(
                protein_prepared_path,
                protein_mapped_path,
                gene_type='human_gene_symbol',
                gene_column='index',
                sctranslator_dir=DEFAULT_SCTRANSLATOR_DIR
            )
        else:
            print(f"  Protein already has my_Id column")
            shutil.copy(protein_prepared_path, protein_mapped_path)
        
        # Step 6: Run scTranslator inference (RNA → protein)
        # We need to modify the inference script to handle checkpoint format
        # For now, we'll create a wrapper script
        
        # Create a modified inference script that handles checkpoint format
        protein_pred_path = os.path.join(output_dir, f"protein_predictions_{sanitized_pert_name}_{args.condition}.h5ad")
        
        print(f"\n{'='*70}")
        print("RUNNING SCTRANSLATOR INFERENCE")
        print(f"{'='*70}")
        
        # Run scTranslator inference with custom function
        from sctranslator_inference_custom import run_sctranslator_inference_custom
        
        pred_protein_adata, protein_metrics = run_sctranslator_inference_custom(
            rna_path=rna_mapped_path,
            protein_path=protein_mapped_path,
            checkpoint_path=args.sctranslator_checkpoint,
            output_path=protein_pred_path,
            sctranslator_dir=DEFAULT_SCTRANSLATOR_DIR,
            test_batch_size=64,  # Reduced to 64 - model requires large contiguous allocations (20+ GiB)
            fix_set=True
        )
        
        print(f"\n  Protein Prediction Metrics (from scTranslator):")
        if 'mse' in protein_metrics:
            print(f"    MSE: {protein_metrics['mse']:.4f}")
        if 'ccc' in protein_metrics:
            print(f"    CCC: {protein_metrics['ccc']:.4f}")
        
        # Extract predictions DataFrame
        y_pred = pd.DataFrame(
            pred_protein_adata.X,
            index=pred_protein_adata.obs_names,
            columns=pred_protein_adata.var_names
        )
        
        # Extract truth if available
        if 'truth' in pred_protein_adata.layers:
            y_truth = pd.DataFrame(
                pred_protein_adata.layers['truth'],
                index=pred_protein_adata.obs_names,
                columns=pred_protein_adata.var_names
            )
        else:
            y_truth = None
        
        # Step 7: Evaluate protein predictions
        # IMPORTANT: Only validate on the original 14 proteins from perturb-cite-seq, NOT the additional predicted proteins
        if not args.skip_evaluation and y_pred is not None and ground_truth_protein is not None:
            print(f"\n{'='*70}")
            print("EVALUATING PROTEIN PREDICTIONS")
            print(f"{'='*70}")
            print(f"  NOTE: Validation will ONLY use the {len(original_protein_names)} original proteins")
            print(f"        (from perturb-cite-seq: {original_protein_names[:5]}{'...' if len(original_protein_names) > 5 else ''})")
            # Count how many original proteins are actually in predictions
            predicted_original_proteins = [p for p in original_protein_names if p in y_pred.columns]
            additional_proteins = len(y_pred.columns) - len(predicted_original_proteins)
            print(f"        {len(predicted_original_proteins)} original proteins are in predictions (out of {len(original_protein_names)})")
            print(f"        Additional {additional_proteins} proteins are predicted but NOT validated")
            
            # Filter ground truth to only cells with perturbation
            if args.perturbation_type == 'gene':
                filter_col = 'target_gene'
            else:  # drug
                filter_col = 'drugname_drugconc'
            
            if filter_col in ground_truth_protein.obs.columns:
                gt_mask = ground_truth_protein.obs[filter_col] == perturbation_name
                if gt_mask.sum() > 0:
                    ground_truth_protein_filtered = ground_truth_protein[gt_mask].copy()
                    print(f"  Filtered ground truth protein to {gt_mask.sum()} cells with {filter_col}: {perturbation_name}")
                else:
                    print(f"  Warning: No cells found with {filter_col}: {perturbation_name} in ground truth protein")
                    print(f"  Using all ground truth cells")
                    ground_truth_protein_filtered = ground_truth_protein.copy()
            elif 'perturbation' in ground_truth_protein.obs.columns:
                # Fallback to old column name
                gt_mask = ground_truth_protein.obs['perturbation'] == perturbation_name
                if gt_mask.sum() > 0:
                    ground_truth_protein_filtered = ground_truth_protein[gt_mask].copy()
                    print(f"  Filtered ground truth protein to {gt_mask.sum()} cells with perturbation: {perturbation_name} (using old column name)")
                else:
                    print(f"  Warning: No cells found with perturbation: {perturbation_name} in ground truth protein")
                    print(f"  Using all ground truth cells")
                    ground_truth_protein_filtered = ground_truth_protein.copy()
            else:
                print(f"  Warning: '{filter_col}' or 'perturbation' column not found in ground truth protein")
                print(f"  Using all ground truth cells")
                ground_truth_protein_filtered = ground_truth_protein.copy()
            
            # CRITICAL: Only use the original proteins for validation
            # Filter to proteins that are both in predictions and in original protein list
            validation_proteins = [p for p in original_protein_names if p in y_pred.columns]
            print(f"  Proteins available for validation: {len(validation_proteins)} out of {len(original_protein_names)} original proteins")
            
            if len(validation_proteins) == 0:
                print(f"  Warning: None of the original {len(original_protein_names)} proteins are in predictions")
                print(f"    Original proteins: {original_protein_names}")
                print(f"    Prediction proteins (first 20): {list(y_pred.columns[:20])}")
                print(f"  Skipping protein validation")
            else:
                # Also check if these proteins are in ground truth
                available_in_ground_truth = [p for p in validation_proteins if p in ground_truth_protein_filtered.var.index]
                print(f"  Proteins in both predictions and ground truth: {len(available_in_ground_truth)}")
                
                if len(available_in_ground_truth) == 0:
                    print(f"  Warning: None of the validation proteins are in ground truth")
                    print(f"  Skipping protein validation")
                else:
                    # Use only the original proteins that are in both predictions and ground truth
                    common_proteins = pd.Index(available_in_ground_truth)
                    common_cells = y_pred.index.intersection(ground_truth_protein_filtered.obs.index)
                    if len(common_cells) > 0:
                        print(f"  Evaluating on {len(common_cells)} cells and {len(common_proteins)} proteins")
                        print(f"    Validation proteins: {list(common_proteins[:10])}{'...' if len(common_proteins) > 10 else ''}")
                        pred_protein_aligned = y_pred.loc[common_cells, common_proteins].values
                        truth_protein_aligned = ground_truth_protein_filtered[common_cells, common_proteins].X
                        
                        # Convert sparse to dense if needed
                        if hasattr(truth_protein_aligned, 'toarray'):
                            truth_protein_aligned = truth_protein_aligned.toarray()
                        
                        protein_metrics = evaluate_predictions(truth_protein_aligned, pred_protein_aligned)
                        
                        print(f"\n  Protein Prediction Metrics (cells with {perturbation_name} {args.perturbation_type} perturbation):")
                        print(f"    Evaluated on {len(common_proteins)} original proteins only")
                        print(f"    R2 Score: {protein_metrics['r2']:.4f}")
                        print(f"    Pearson R: {protein_metrics['pearson_r']:.4f}")
                        print(f"    RMSE: {protein_metrics['rmse']:.4f}")
                        print(f"    MAE: {protein_metrics['mae']:.4f}")
                    else:
                        print(f"  Warning: No common cells between predictions and filtered ground truth")
        
        # Step 8: Pathway and enrichment analysis
        # For gene perturbations: use ground truth as control
        # For drug perturbations: run analysis without control (correlation only)
        if not args.skip_evaluation:
            print(f"\n{'='*70}")
            print("PATHWAY AND ENRICHMENT ANALYSIS")
            print(f"{'='*70}")
            
            try:
                from pathway_analysis import comprehensive_analysis
                
                analysis_output_dir = os.path.join(output_dir, 'pathway_analysis')
                os.makedirs(analysis_output_dir, exist_ok=True)
                
                # For gene perturbations, use ground truth as control
                # For drug perturbations or genes not in target_gene, control cells are in predictions
                # comprehensive_analysis will automatically extract control from predictions if no external control provided
                if args.perturbation_type == 'gene' and ground_truth_rna is not None and ground_truth_protein is not None and not skip_validation_gene_not_found:
                    # Gene found in target_gene: use ground truth as control
                    analysis_results = comprehensive_analysis(
                        rna_predictions_path=rna_pred_path,
                        protein_predictions_path=protein_pred_path,
                        control_rna_path=args.ground_truth_rna,
                        control_protein_path=args.ground_truth_protein,
                        target_gene=perturbation_name,
                        output_dir=analysis_output_dir,
                        control_label='non-targeting'
                    )
                else:
                    # Drug perturbation OR gene not in target_gene: control cells are in predictions
                    # For genes not in target_gene: 20% non-targeting, 80% target_gene
                    # For drugs: 20% DMSO_TF, 80% drug
                    # comprehensive_analysis will automatically extract control from predictions
                    analysis_results = comprehensive_analysis(
                        rna_predictions_path=rna_pred_path,
                        protein_predictions_path=protein_pred_path,
                        control_rna_path=None,
                        control_protein_path=None,
                        target_gene=perturbation_name,
                        output_dir=analysis_output_dir,
                        control_label='non-targeting'
                    )
                
                print(f"\n  ✓ Pathway analysis completed")
                print(f"    Results saved to: {analysis_output_dir}")
            except Exception as e:
                print(f"  Warning: Pathway analysis failed: {str(e)}")
                print(f"    You can run it separately using:")
                print(f"    python pathway_analysis.py \\")
                print(f"      --rna-predictions {rna_pred_path} \\")
                print(f"      --protein-predictions {protein_pred_path} \\")
                print(f"      --control-rna {args.ground_truth_rna} \\")
                print(f"      --control-protein {args.ground_truth_protein} \\")
                print(f"      --target-gene {perturbation_name} \\")
                print(f"      --output-dir {os.path.join(output_dir, 'pathway_analysis')}")
        
        # Step 9: Save final results
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"\nResults saved to: {output_dir}")
        print(f"  - Perturbation data: {pert_data_path}")
        print(f"  - RNA predictions: {rna_pred_path}")
        print(f"  - Protein predictions: {protein_pred_path}")
        print(f"  - Prepared protein data: {protein_prepared_path}")
        if not args.skip_evaluation:
            print(f"  - Pathway analysis: {os.path.join(output_dir, 'pathway_analysis')}")
        print(f"\nPerturbation type: {args.perturbation_type}")
        print(f"Perturbation name: {perturbation_name}")
        print(f"Protein file used: {args.protein_file}")
        
        if not args.keep_temp:
            print(f"\nNote: Temporary files will be kept in {output_dir}")
            print(f"      Use --keep-temp to preserve them explicitly")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR: Pipeline failed")
        print(f"{'='*70}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

