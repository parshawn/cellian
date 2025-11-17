"""
ST-Tahoe drug inference module: Drug Perturbation → Transcriptomics (RNA)

This module handles running the ST-Tahoe model to predict RNA expression
from drug perturbation data.
"""

import os
import subprocess
import tempfile
import anndata as ad
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


def run_drug_inference(
    model_dir,
    checkpoint_path,
    input_adata_path,
    output_path,
    pert_col="drugname_drugconc"
):
    """
    Run ST-Tahoe model inference to predict RNA from drug perturbation.
    
    Uses the command: state tx infer --model-dir <ST-Tahoe_PATH> --checkpoint <CHECKPOINT> 
                     --pert-col drugname_drugconc --control-pert "[('DMSO_TF', 0.0, 'uM')]"
                     --adata <INPUT_ADATA>.h5ad --output <OUTPUT_PATH>
    
    Note: The perturbations are already in the adata file's pert_col column.
    The --control-pert parameter specifies the control perturbation format.
    
    Args:
        model_dir: Path to ST-Tahoe model directory
        checkpoint_path: Path to checkpoint file
        input_adata_path: Path to input AnnData file with drug perturbation info
        output_path: Path to save predictions (can be file or directory)
        pert_col: Column name for drug perturbation in obs (format: drugname_drugconc)
    
    Returns:
        Path to output file or directory with predictions
    """
    print(f"\n{'='*70}")
    print("STEP 1: Running ST-Tahoe Drug Inference (Drug Perturbation → RNA)")
    print(f"{'='*70}")
    print(f"Model directory: {model_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input data: {input_adata_path}")
    print(f"Output file: {output_path}")
    print(f"Drug perturbation column: {pert_col}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run state inference command with checkpoint
    # Note: --control-pert specifies the control perturbation format
    # The actual perturbations are already in the adata file's pert_col column
    cmd = [
        "state", "tx", "infer",
        "--model-dir", model_dir,
        "--checkpoint", checkpoint_path,
        "--pert-col", pert_col,
        "--batch-col", "library_preparation_protocol",
        "--control-pert", "[('DMSO_TF', 0.0, 'uM')]",
        "--adata", input_adata_path,
        "--output", output_path
    ]
    
    print(f"\nRunning ST-Tahoe drug inference command...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ST-Tahoe drug inference failed!")
        print(f"STDERR: {result.stderr}")
        print(f"STDOUT: {result.stdout}")
        raise RuntimeError(f"ST-Tahoe drug inference failed: {result.stderr}")
    
    print(f"✓ ST-Tahoe drug inference completed successfully")
    print(f"  Predictions saved to: {output_path}")
    
    # Verify the output file exists
    if os.path.exists(output_path) and os.path.isfile(output_path):
        pred_adata = ad.read_h5ad(output_path)
        print(f"  Predicted RNA shape: {pred_adata.shape} (cells × genes)")
        print(f"  Number of cells: {pred_adata.n_obs:,}")
        print(f"  Number of genes: {pred_adata.n_vars:,}")
        return output_path
    else:
        raise FileNotFoundError(f"Output file not found at expected path: {output_path}")



