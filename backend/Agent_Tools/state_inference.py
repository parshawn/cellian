"""
State inference module: Perturbation → Transcriptomics (RNA)

This module handles running the state model to predict RNA expression
from perturbation data.
"""

import os
import subprocess
import tempfile
import anndata as ad
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


def load_hvg_genes(var_dims_path):
    """Load highly variable genes from state model."""
    with open(var_dims_path, 'rb') as f:
        var_dims = pickle.load(f)
    return var_dims['gene_names']


def load_perturbation_names(var_dims_path):
    """Load perturbation names from state model training data."""
    with open(var_dims_path, 'rb') as f:
        var_dims = pickle.load(f)
    pert_names = var_dims.get('pert_names', [])
    # Convert numpy strings to regular strings
    return [str(p) for p in pert_names] if pert_names is not None else []


def run_state_inference(
    model_dir,
    checkpoint_path,
    input_adata_path,
    output_path,
    pert_col="target_gene",
    var_dims_path=None
):
    """
    Run state model inference to predict RNA from perturbation.
    
    Uses obsm['X_hvg'] with 2000-dimensional gene expression as input.
    The model expects X_hvg to have shape (n_cells, 2000).
    
    Args:
        model_dir: Path to state model directory
        checkpoint_path: Path to checkpoint file
        input_adata_path: Path to input AnnData file with perturbation info
        output_path: Path to save predictions
        pert_col: Column name for perturbation in obs
        var_dims_path: Path to var_dims.pkl file (auto-detected if None)
    
    Returns:
        Path to output file with predictions
    """
    print(f"\n{'='*70}")
    print("STEP 1: Running State Inference (Perturbation → RNA)")
    print(f"{'='*70}")
    print(f"Model directory: {model_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input data: {input_adata_path}")
    print(f"Output: {output_path}")
    print(f"Perturbation column: {pert_col}")
    
    # Auto-detect var_dims path if not provided
    if var_dims_path is None:
        var_dims_path = os.path.join(model_dir, "var_dims.pkl")
    
    if not os.path.exists(var_dims_path):
        raise FileNotFoundError(f"var_dims.pkl not found at {var_dims_path}")
    
    # Load HVG genes
    hvg_genes = load_hvg_genes(var_dims_path)
    print(f"\n✓ Loaded {len(hvg_genes)} highly variable genes from model")
    
    # Check if requested perturbations are in training set
    pert_names_trained = load_perturbation_names(var_dims_path)
    pert_set_trained = set(pert_names_trained)
    print(f"\n✓ Loaded {len(pert_names_trained)} perturbations from model training data")
    
    # Load input data to check perturbations
    input_adata = ad.read_h5ad(input_adata_path)
    if pert_col in input_adata.obs.columns:
        unique_perts = set(input_adata.obs[pert_col].unique())
        unknown_perts = unique_perts - pert_set_trained
        unknown_perts = unknown_perts - {None, np.nan, ''}  # Remove empty values
        
        if len(unknown_perts) > 0:
            print(f"\n{'!'*70}")
            print("⚠️  WARNING: Unknown Perturbations Detected")
            print(f"{'!'*70}")
            print(f"\nThe following perturbations are NOT in the model's training data:")
            for pert in sorted(unknown_perts):
                print(f"  - {pert}")
            print(f"\n⚠️  CRITICAL: STATE will treat these as CONTROL perturbations!")
            print(f"   This means predictions for these perturbations will be identical to control predictions.")
            print(f"   The model will use the control perturbation vector for unknown perturbations.")
            print(f"\n⚠️  Results for these perturbations will NOT be meaningful.")
            print(f"\n{'!'*70}\n")
        else:
            print(f"\n✓ All perturbations in input data are in the training set")
    
    # Build command - use X_hvg with 2000-dimensional gene expression
    cmd = [
        "state", "tx", "infer",
        "--model-dir", model_dir,
        "--checkpoint", checkpoint_path,
        "--pert-col", pert_col,
        "--batch-col", "library_preparation_protocol",
        "--control-pert", "non-targeting",
        "--adata", input_adata_path,
        "--output", output_path,
        "--embed-key", "X_hvg"
    ]
    
    print(f"\nRunning state inference command...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"State inference failed!")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"State inference failed: {result.stderr}")
    
    print(f"✓ State inference completed successfully")
    print(f"  Predictions saved to: {output_path}")
    
    # Load and verify predictions
    pred_adata = ad.read_h5ad(output_path)
    print(f"  Predicted RNA shape: {pred_adata.shape} (cells × genes)")
    print(f"  Number of cells: {pred_adata.n_obs:,}")
    print(f"  Number of genes: {pred_adata.n_vars:,}")
    
    return output_path


def filter_to_hvg(pred_adata, hvg_genes):
    """
    Filter predicted RNA to only include HVG genes.
    
    Args:
        pred_adata: AnnData object with predictions
        hvg_genes: List of HVG gene names
    
    Returns:
        Filtered AnnData object
    """
    # Get intersection of predicted genes and HVG genes
    pred_genes = pred_adata.var.index.tolist()
    hvg_set = set(hvg_genes)
    pred_set = set(pred_genes)
    
    overlap = pred_set.intersection(hvg_set)
    print(f"  HVG overlap: {len(overlap)}/{len(hvg_genes)} genes")
    
    # Filter to overlapping genes
    if len(overlap) < len(hvg_genes):
        print(f"  Warning: Only {len(overlap)}/{len(hvg_genes)} HVG genes found in predictions")
    
    # Filter to genes that exist in both
    genes_to_keep = [g for g in pred_genes if g in hvg_set]
    filtered_adata = pred_adata[:, genes_to_keep].copy()
    
    return filtered_adata

