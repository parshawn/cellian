"""
scTranslator inference module: Transcriptomics (RNA) → Proteomics

This module handles running scTranslator to predict protein expression
from RNA expression data.
"""

import os
import subprocess
import tempfile
import shutil
import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path


def run_id_mapping(input_path, output_path, gene_type='human_gene_symbol', gene_column='index', 
                   sctranslator_dir="/home/nebius/scTranslator"):
    """
    Run scTranslator ID mapping preprocessing.
    
    Args:
        input_path: Path to input h5ad file
        output_path: Path to save mapped h5ad file
        gene_type: Type of gene identifiers
        gene_column: Column containing gene names
        sctranslator_dir: Path to scTranslator directory
    
    Returns:
        Path to mapped file
    """
    print(f"\n  Running ID mapping for: {os.path.basename(input_path)}")
    
    # Convert input_path to absolute path
    input_path_abs = os.path.abspath(input_path)
    
    # The script needs to be run from scTranslator directory to find ID_dic files
    cmd = [
        "conda", "run", "-n", "new_env", "python",
        "code/model/data_preprocessing_ID_convert.py",
        f"--origin_gene_type={gene_type}",
        f"--origin_gene_column={gene_column}",
        f"--data_path={input_path_abs}"
    ]
    
    # Run from scTranslator directory
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=sctranslator_dir)
    
    if result.returncode != 0:
        print(f"  ID mapping failed: {result.stderr}")
        print(f"  STDOUT: {result.stdout}")
        raise RuntimeError(f"ID mapping failed: {result.stderr}")
    
    # The script creates a file with _mapped suffix in the same directory as input
    input_dir = os.path.dirname(input_path_abs)
    input_basename = os.path.basename(input_path_abs)
    expected_output = os.path.join(input_dir, input_basename.replace('.h5ad', '_mapped.h5ad'))
    
    if not os.path.exists(expected_output):
        raise FileNotFoundError(f"Expected output {expected_output} not found after ID mapping")
    
    # Move to desired output path if different
    if expected_output != output_path:
        shutil.move(expected_output, output_path)
    
    print(f"  ✓ ID mapping completed: {output_path}")
    return output_path


def prepare_protein_data(rna_adata_path, protein_file_path, target_gene, output_path, is_drug=False):
    """
    Prepare protein data from real protein file to match RNA predictions.
    
    IMPORTANT: This function now does TWO things:
    1. Keeps the original 14 proteins from perturb-cite-seq (for validation)
    2. Adds ALL additional proteins that match RNA gene names (for inference)
    
    Filters protein data to match cells from RNA predictions (same target_gene/drug)
    and aligns cell indices. Then adds all proteins from RNA gene names that aren't
    already in the protein file.
    
    Args:
        rna_adata_path: Path to RNA AnnData file (predictions)
        protein_file_path: Path to protein expression h5ad file (real data)
        target_gene: Target gene name or drug name
        output_path: Path to save prepared protein file
        is_drug: Whether this is a drug perturbation (default: False)
    
    Returns:
        Tuple of (Path to prepared protein file, list of 14 mapped protein names for validation)
    """
    print(f"\n  Preparing protein data from real protein file...")
    print(f"    RNA predictions: {os.path.basename(rna_adata_path)}")
    print(f"    Protein file: {os.path.basename(protein_file_path)}")
    print(f"    {'Drug' if is_drug else 'Target gene'}: {target_gene}")
    
    # Load RNA predictions to get cell indices and gene names
    rna_adata = ad.read_h5ad(rna_adata_path)
    print(f"    RNA predictions shape: {rna_adata.shape} (cells × genes)")
    print(f"    RNA cell IDs: {len(rna_adata.obs_names)} cells")
    print(f"    RNA gene names: {len(rna_adata.var_names)} genes")
    
    # Load protein file
    protein_adata = ad.read_h5ad(protein_file_path)
    print(f"    Protein file shape: {protein_adata.shape} (cells × proteins)")
    print(f"    Protein cell IDs: {len(protein_adata.obs_names)} cells")
    print(f"    Original proteins: {len(protein_adata.var_names)}")
    
    # Keep track of the original 14 proteins for validation
    original_protein_names = list(protein_adata.var_names)
    print(f"    Original protein names (first 10): {original_protein_names[:10]}")
    
    # Filter protein data to match target_gene/drug
    # For drug perturbations, we don't filter (user said no validation)
    if is_drug:
        print(f"    Drug perturbation: Using all cells from protein file (no filtering)")
        protein_filtered = protein_adata.copy()
    elif 'target_gene' in protein_adata.obs.columns:
        mask = protein_adata.obs['target_gene'] == target_gene
        if mask.sum() > 0:
            protein_filtered = protein_adata[mask].copy()
            print(f"    Filtered to {mask.sum()} cells with target_gene: {target_gene}")
        else:
            print(f"    Warning: No cells found with target_gene: {target_gene} in protein file")
            print(f"    Using all cells from protein file")
            protein_filtered = protein_adata.copy()
    elif 'perturbation' in protein_adata.obs.columns:
        # Fallback to old column name
        mask = protein_adata.obs['perturbation'] == target_gene
        if mask.sum() > 0:
            protein_filtered = protein_adata[mask].copy()
            print(f"    Filtered to {mask.sum()} cells with perturbation: {target_gene} (using old column name)")
        else:
            print(f"    Warning: No cells found with perturbation: {target_gene} in protein file")
            print(f"    Using all cells from protein file")
            protein_filtered = protein_adata.copy()
    else:
        print(f"    Warning: 'target_gene' or 'perturbation' column not found in protein file")
        print(f"    Using all cells from protein file")
        protein_filtered = protein_adata.copy()
    
    # Align cells: try to match cell indices with RNA predictions
    # Priority: Use cells that match RNA predictions exactly
    common_cells = rna_adata.obs_names.intersection(protein_filtered.obs_names)
    
    print(f"    Cell alignment: {len(common_cells)} common cells out of {len(rna_adata)} RNA cells")
    
    if len(common_cells) == len(rna_adata) and len(common_cells) == len(protein_filtered):
        # Perfect match - use common cells in RNA order
        print(f"    Perfect match: All {len(common_cells)} RNA cells found in protein file")
        protein_aligned = protein_filtered[common_cells].copy()
        # Reorder to match RNA order exactly
        protein_aligned = protein_aligned[rna_adata.obs_names].copy()
    elif len(common_cells) > 0:
        # Partial match - use common cells
        print(f"    Partial match: Found {len(common_cells)}/{len(rna_adata)} common cells")
        if len(common_cells) < len(rna_adata):
            print(f"    Warning: {len(rna_adata) - len(common_cells)} RNA cells not found in protein file")
        print(f"    Using common cells and reordering to match RNA cell order")
        protein_aligned = protein_filtered[common_cells].copy()
        # Reorder to match RNA order (only common cells)
        common_in_rna_order = [cell for cell in rna_adata.obs_names if cell in common_cells]
        protein_aligned = protein_aligned[common_in_rna_order].copy()
        
        # If we have fewer cells than RNA, we need to subset RNA too or handle mismatch
        # For now, we'll use the common cells and scTranslator will handle it
        if len(protein_aligned) < len(rna_adata):
            print(f"    Note: Will use {len(protein_aligned)} cells for scTranslator inference")
    else:
        # No match - cells don't overlap
        # This can happen if RNA predictions come from a different source
        # In this case, we'll create protein structure with RNA cells and use protein structure
        print(f"    Warning: No common cells found between RNA and protein")
        print(f"    This may happen if RNA predictions come from a different source")
        print(f"    Creating protein structure with RNA cell IDs")
        print(f"    Note: Protein values will be zeros (scTranslator will predict them)")
        
        # Create protein adata with RNA cells and protein structure
        if sp.issparse(protein_filtered.X):
            protein_aligned = ad.AnnData(
                X=sp.csr_matrix((len(rna_adata), protein_filtered.n_vars)),
                obs=rna_adata.obs.copy(),
                var=protein_filtered.var.copy()
            )
        else:
            protein_aligned = ad.AnnData(
                X=np.zeros((len(rna_adata), protein_filtered.n_vars), dtype=np.float32),
                obs=rna_adata.obs.copy(),
                var=protein_filtered.var.copy()
            )
        protein_aligned.obs_names = rna_adata.obs_names
    
    # NEW: Add all proteins from RNA gene names that aren't already in protein file
    print(f"\n    Adding proteins from RNA gene names...")
    rna_gene_names = set(rna_adata.var_names)
    existing_protein_names = set(protein_aligned.var_names)
    
    # Find proteins in RNA that aren't in protein file
    new_protein_names = sorted(list(rna_gene_names - existing_protein_names))
    print(f"    Found {len(new_protein_names)} additional proteins from RNA gene names")
    print(f"    (First 10 new proteins: {new_protein_names[:10]})")
    
    if len(new_protein_names) > 0:
        # Create new protein entries with zeros (they'll be predicted by scTranslator)
        n_cells = protein_aligned.n_obs
        
        # Create var DataFrame for new proteins
        new_protein_var = pd.DataFrame(
            index=new_protein_names,
            columns=protein_aligned.var.columns if len(protein_aligned.var.columns) > 0 else []
        )
        
        # CRITICAL: Copy my_Id from RNA - all new proteins should match RNA gene names
        # Since we're adding proteins that match RNA gene names, they should have my_Id from RNA
        if 'my_Id' in rna_adata.var.columns:
            my_ids = []
            for gene in new_protein_names:
                if gene in rna_adata.var_names:
                    my_id = rna_adata.var.loc[gene, 'my_Id']
                    # Handle NaN - fill with 0 if not found
                    if pd.isna(my_id):
                        print(f"    Warning: my_Id is NaN for gene {gene}, using 0")
                        my_ids.append(0)
                    else:
                        my_ids.append(my_id)
                else:
                    print(f"    Warning: Gene {gene} not found in RNA var_names, using 0 for my_Id")
                    my_ids.append(0)
            new_protein_var['my_Id'] = my_ids
        else:
            print(f"    Warning: 'my_Id' column not found in RNA adata")
            print(f"    Setting my_Id to 0 for all new proteins (they may not work with scTranslator)")
            new_protein_var['my_Id'] = 0
        
        # Create zeros for new proteins
        if sp.issparse(protein_aligned.X):
            new_protein_X = sp.csr_matrix((n_cells, len(new_protein_names)), dtype=np.float32)
        else:
            new_protein_X = np.zeros((n_cells, len(new_protein_names)), dtype=np.float32)
        
        # Concatenate existing and new proteins
        # Combine var
        combined_var = pd.concat([protein_aligned.var, new_protein_var])
        
        # Combine X
        if sp.issparse(protein_aligned.X):
            from scipy.sparse import hstack
            combined_X = hstack([protein_aligned.X, new_protein_X], format='csr')
        else:
            combined_X = np.hstack([protein_aligned.X, new_protein_X])
        
        # Create new adata with combined proteins
        protein_aligned = ad.AnnData(
            X=combined_X,
            obs=protein_aligned.obs.copy(),
            var=combined_var
        )
        protein_aligned.obs_names = protein_aligned.obs_names  # Keep cell names
        
        print(f"    ✓ Added {len(new_protein_names)} new proteins to protein data")
    
    # CRITICAL: Ensure all proteins have valid my_Id values (no NaN)
    # Fill any NaN values with 0
    if 'my_Id' in protein_aligned.var.columns:
        nan_count = protein_aligned.var['my_Id'].isna().sum()
        if nan_count > 0:
            print(f"    Warning: {nan_count} proteins have NaN my_Id values, filling with 0")
            protein_aligned.var['my_Id'] = protein_aligned.var['my_Id'].fillna(0)
        # Also ensure my_Id is numeric (handle any string types)
        try:
            protein_aligned.var['my_Id'] = pd.to_numeric(protein_aligned.var['my_Id'], errors='coerce').fillna(0).astype(int)
        except Exception as e:
            print(f"    Warning: Could not convert my_Id to int: {e}")
            protein_aligned.var['my_Id'] = 0
    else:
        print(f"    Warning: 'my_Id' column not found in protein var, creating with 0 values")
        protein_aligned.var['my_Id'] = 0
    
    print(f"    Total proteins now: {len(protein_aligned.var_names)} (original {len(original_protein_names)} + {len(new_protein_names) if 'new_protein_names' in locals() else 0} new)")
    
    # Ensure target_gene is set correctly
    if 'target_gene' not in protein_aligned.obs.columns:
        protein_aligned.obs['target_gene'] = target_gene
    else:
        protein_aligned.obs['target_gene'] = target_gene
    
    # Save prepared protein file
    protein_aligned.write_h5ad(output_path)
    print(f"  ✓ Prepared protein data: {output_path}")
    print(f"    Shape: {protein_aligned.shape} (cells × proteins)")
    print(f"    Number of cells: {protein_aligned.n_obs:,}")
    print(f"    Number of proteins: {protein_aligned.n_vars:,}")
    
    # IMPORTANT: Tell user about validation and model limitations
    print(f"\n    ⚠️  IMPORTANT NOTES:")
    print(f"       - scTranslator model has a max_seq_len limit of 1000 proteins")
    print(f"       - If you have more than 1000 proteins, only the first 1000 will be predicted")
    print(f"       - Original {len(original_protein_names)} proteins are first (guaranteed to be predicted)")
    print(f"         (from perturb-cite-seq: {original_protein_names[:5]}{'...' if len(original_protein_names) > 5 else ''})")
    if len(protein_aligned.var_names) > 1000:
        print(f"       - WARNING: {len(protein_aligned.var_names)} total proteins, but only first 1000 will be predicted")
        print(f"       - {len(protein_aligned.var_names) - 1000} proteins will be excluded due to model limitation")
    else:
        print(f"       - All {len(protein_aligned.var_names)} proteins will be predicted")
    print(f"       - Validation will ONLY use the {len(original_protein_names)} original proteins")
    
    # Check if values are all zeros (dummy data) or real data
    if hasattr(protein_aligned.X, 'toarray'):
        protein_values = protein_aligned.X.toarray()
    else:
        protein_values = protein_aligned.X
    
    non_zero_count = np.count_nonzero(protein_values)
    total_count = protein_values.size
    if non_zero_count == 0:
        print(f"    Note: Protein values are all zeros (will be predicted by scTranslator)")
    else:
        sparsity = 1 - (non_zero_count / total_count)
        print(f"    Note: Protein values are real data (sparsity: {sparsity:.2%})")
    
    return output_path, original_protein_names


def create_dummy_protein_file(rna_adata_path, protein_template_path, output_path, 
                               gene_type='human_gene_symbol'):
    """
    Create a dummy protein file with same cells as RNA and protein structure from template.
    
    DEPRECATED: Use prepare_protein_data() instead to use real protein data.
    
    Args:
        rna_adata_path: Path to RNA AnnData file
        protein_template_path: Path to protein template file (for structure)
        output_path: Path to save dummy protein file
        gene_type: Gene identifier type for mapping
    
    Returns:
        Path to created dummy protein file
    """
    print(f"\n  Creating dummy protein file from template...")
    print(f"    Warning: This creates dummy data. Consider using prepare_protein_data() for real data.")
    
    # Load RNA and protein template
    rna_adata = ad.read_h5ad(rna_adata_path)
    protein_template = ad.read_h5ad(protein_template_path)
    
    # Create dummy protein adata with same cells as RNA
    dummy_protein = ad.AnnData(
        X=np.zeros((rna_adata.n_obs, protein_template.n_vars)),
        obs=rna_adata.obs.copy(),
        var=protein_template.var.copy()
    )
    
    # Ensure cell indices match
    dummy_protein.obs_names = rna_adata.obs_names
    
    # Save dummy protein file
    dummy_protein.write_h5ad(output_path)
    print(f"  ✓ Created dummy protein file: {output_path}")
    print(f"    Shape: {dummy_protein.shape} (cells × proteins)")
    
    return output_path


def load_checkpoint_model(checkpoint_path, device='cuda'):
    """
    Load model from checkpoint, handling both direct model and checkpoint dict formats.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    import torch
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    
    # Check if checkpoint is a dict with 'net' key (from fine-tuning) or direct model
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        model = checkpoint['net']
        print(f"  ✓ Loaded model from checkpoint dict (fine-tuned model)")
    else:
        model = checkpoint
        print(f"  ✓ Loaded model directly from checkpoint")
    
    return model


def run_sctranslator_inference(
    rna_path,
    protein_path,
    checkpoint_path,
    output_dir,
    test_batch_size=4,
    fix_set=True,
    sctranslator_dir="/home/nebius/scTranslator"
):
    """
    Run scTranslator inference to predict protein from RNA.
    
    Args:
        rna_path: Path to RNA h5ad file (with my_Id column)
        protein_path: Path to protein h5ad file (with my_Id column, can be dummy)
        checkpoint_path: Path to scTranslator checkpoint
        output_dir: Directory to save results
        test_batch_size: Batch size for inference
        fix_set: Whether to use fixed (aligned) dataset
        sctranslator_dir: Path to scTranslator directory
    
    Returns:
        Tuple of (predictions DataFrame, truth DataFrame, metrics dict)
    """
    print(f"\n{'='*70}")
    print("STEP 2: Running scTranslator Inference (RNA → Protein)")
    print(f"{'='*70}")
    print(f"RNA input: {rna_path}")
    print(f"Protein template: {protein_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    # Change to scTranslator directory for running the script
    original_dir = os.getcwd()
    
    try:
        os.chdir(sctranslator_dir)
        
        # Build command
        cmd = [
            "conda", "run", "-n", "new_env", "python",
            "code/main_scripts/stage3_inference_without_finetune.py",
            f"--pretrain_checkpoint={checkpoint_path}",
            f"--RNA_path={rna_path}",
            f"--Pro_path={protein_path}",
            f"--test_batch_size={test_batch_size}",
        ]
        
        if fix_set:
            cmd.append("--fix_set")
        
        print(f"\n  Running scTranslator inference command...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  scTranslator inference failed!")
            print(f"  STDERR: {result.stderr}")
            raise RuntimeError(f"scTranslator inference failed: {result.stderr}")
        
        print(f"  ✓ scTranslator inference completed")
        
        # Parse output to extract metrics
        metrics = {}
        for line in result.stdout.split('\n'):
            if 'AVG mse' in line:
                try:
                    # Extract MSE value
                    mse_str = line.split('AVG mse')[1].strip().split()[0]
                    metrics['mse'] = float(mse_str)
                except:
                    pass
            elif 'AVG ccc' in line:
                try:
                    # Extract CCC value
                    ccc_str = line.split('AVG ccc')[1].strip().split()[0]
                    metrics['ccc'] = float(ccc_str)
                except:
                    pass
        
        # Load predictions from output directory
        # The script saves to result/test/new_data-without_fine-tune/
        result_dir = os.path.join(sctranslator_dir, "result/test/new_data-without_fine-tune")
        
        if os.path.exists(os.path.join(result_dir, "y_pred.csv")):
            y_pred = pd.read_csv(os.path.join(result_dir, "y_pred.csv"), index_col=0)
            y_truth = pd.read_csv(os.path.join(result_dir, "y_truth.csv"), index_col=0)
            
            print(f"  ✓ Loaded predictions: {y_pred.shape}")
            print(f"    Predictions shape: {y_pred.shape} (cells × proteins)")
            
            return y_pred, y_truth, metrics
        else:
            print(f"  Warning: Prediction files not found in {result_dir}")
            return None, None, metrics
            
    finally:
        os.chdir(original_dir)


def save_predictions_as_adata(y_pred, y_truth, rna_adata_path, protein_template_path, output_path):
    """
    Save predictions as AnnData file.
    
    Args:
        y_pred: Predictions DataFrame (cells × proteins)
        y_truth: Truth DataFrame (cells × proteins) - can be None
        rna_adata_path: Path to RNA AnnData (for cell metadata)
        protein_template_path: Path to protein template (for protein metadata)
        output_path: Path to save predictions as h5ad
    
    Returns:
        Path to saved file
    """
    print(f"\n  Saving predictions as AnnData...")
    
    # Load RNA adata for cell metadata
    rna_adata = ad.read_h5ad(rna_adata_path)
    
    # Load protein template for protein metadata
    protein_template = ad.read_h5ad(protein_template_path)
    
    # Create AnnData from predictions
    pred_adata = ad.AnnData(
        X=y_pred.values,
        obs=rna_adata.obs.copy(),
        var=protein_template.var.copy()
    )
    
    # Ensure cell names match
    if len(pred_adata) == len(y_pred):
        pred_adata.obs_names = y_pred.index
    
    # Add truth as a layer if available
    if y_truth is not None:
        # Align truth with predictions
        common_cells = y_pred.index.intersection(y_truth.index)
        if len(common_cells) > 0:
            truth_aligned = y_truth.loc[common_cells, y_pred.columns]
            pred_adata.layers['truth'] = truth_aligned.values
            print(f"  ✓ Added ground truth as layer (aligned {len(common_cells)} cells)")
    
    # Save
    pred_adata.write_h5ad(output_path)
    print(f"  ✓ Saved predictions to: {output_path}")
    print(f"    Shape: {pred_adata.shape} (cells × proteins)")
    
    return output_path
