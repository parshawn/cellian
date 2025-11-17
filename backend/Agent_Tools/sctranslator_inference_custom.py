"""
Custom scTranslator inference module that handles fine-tuned checkpoints
and removes the [:100] limitation from the original script.
"""

import os
import sys
import time
import warnings
import tempfile
import subprocess
import shutil

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import torch

# Add scTranslator to path
sctranslator_dir = "/home/nebius/scTranslator"
sys.path.insert(0, os.path.join(sctranslator_dir, 'code/model'))

from performer_enc_dec import *
from utils import *


def load_checkpoint_model(checkpoint_path, device='cuda'):
    """
    Load model from checkpoint, handling both direct model and checkpoint dict formats.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    print(f"  Loading checkpoint from: {checkpoint_path}")
    
    with torch.serialization.safe_globals([scPerformerEncDec]):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    
    # Check if checkpoint is a dict with 'net' key (from fine-tuning) or direct model
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        model = checkpoint['net']
        print(f"  ✓ Loaded fine-tuned model from checkpoint dict")
        if 'epoch' in checkpoint:
            print(f"    Checkpoint epoch: {checkpoint['epoch']}")
    else:
        model = checkpoint
        print(f"  ✓ Loaded model directly from checkpoint")
    
    return model


def run_sctranslator_inference_custom(
    rna_path,
    protein_path,
    checkpoint_path,
    output_path,
    test_batch_size=32,  # Reduced to 32 to avoid GPU memory issues
    fix_set=True,
    enc_max_seq_len=20000,
    dec_max_seq_len=1000,  # Model was trained with 1000 - cannot be changed (positional embeddings are fixed)
    device='cuda',
    sctranslator_dir="/home/nebius/scTranslator"
):
    """
    Run scTranslator inference with custom handling for fine-tuned checkpoints.
    
    Args:
        rna_path: Path to RNA h5ad file (with my_Id column)
        protein_path: Path to protein h5ad file (with my_Id column)
        checkpoint_path: Path to scTranslator checkpoint
        output_path: Path to save predictions as h5ad
        test_batch_size: Batch size for inference
        fix_set: Whether to use fixed (aligned) dataset
        enc_max_seq_len: Encoder max sequence length
        dec_max_seq_len: Decoder max sequence length
        device: Device to run inference on
        sctranslator_dir: Path to scTranslator directory
    
    Returns:
        Tuple of (predictions AnnData, metrics dict)
    """
    print(f"\n  Running scTranslator inference (custom)...")
    print(f"    RNA: {rna_path}")
    print(f"    Protein: {protein_path}")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    Output: {output_path}")
    print(f"    Device: {device}")
    
    warnings.filterwarnings('ignore')
    
    # IMPORTANT: Set this environment variable before running to reduce memory fragmentation:
    # export PYTORCH_ALLOC_CONF=expandable_segments:True
    # (Note: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead)
    # This helps reduce memory fragmentation by allowing PyTorch to expand memory segments
    # The model requires large contiguous allocations (20+ GiB), so this is critical
    import os
    if 'PYTORCH_ALLOC_CONF' not in os.environ:
        print(f"    WARNING: PYTORCH_ALLOC_CONF not set. Consider setting:")
        print(f"    export PYTORCH_ALLOC_CONF=expandable_segments:True")
        print(f"    This will help with memory fragmentation for large allocations (20+ GiB)")
    
    # Load model and verify GPU
    requested_device = device
    use_cuda = device == 'cuda' and torch.cuda.is_available()
    if use_cuda:
        device = 'cuda'
        print(f"    Using device: {device}")
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"    CUDA Available: {torch.cuda.is_available()}")
        print(f"    CUDA Version: {torch.version.cuda}")
    else:
        device = 'cpu'
        print(f"    Using device: {device}")
        if requested_device == 'cuda' and not torch.cuda.is_available():
            print(f"    Warning: CUDA requested but not available, falling back to CPU")
    
    model = load_checkpoint_model(checkpoint_path, device=device)
    model = model.to(device)
    model.eval()
    
    # Clear GPU cache after model loading
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Memory fraction limit removed to allow full GPU memory usage
        # With 52+ GB free memory available, we can use larger batches
    
    # Load data (REMOVED [:100] limitation)
    print(f"    Loading data...")
    scRNA_adata = sc.read_h5ad(rna_path)
    scP_adata = sc.read_h5ad(protein_path)
    
    print(f"    RNA shape: {scRNA_adata.shape} (cells × genes)")
    print(f"    Protein shape: {scP_adata.shape} (cells × proteins)")
    print(f"    Total cells: {scRNA_adata.n_obs:,}")
    print(f"    Total proteins: {scP_adata.n_vars:,}")
    
    # CRITICAL: The model was trained with a fixed dec_max_seq_len (default 1000)
    # We CANNOT dynamically change this because the model's positional embeddings are fixed
    # If we have more proteins than dec_max_seq_len, we must truncate to fit within the limit
    num_proteins = scP_adata.n_vars
    if num_proteins > dec_max_seq_len:
        print(f"    WARNING: Dataset has {num_proteins} proteins, but model max_seq_len is {dec_max_seq_len}")
        print(f"    Model was trained with dec_max_seq_len={dec_max_seq_len} - cannot be changed dynamically")
        print(f"    Truncating to first {dec_max_seq_len} proteins for inference")
        # Truncate protein adata to first dec_max_seq_len proteins
        # IMPORTANT: The original 14 proteins should be first, so they'll be included
        scP_adata = scP_adata[:, :dec_max_seq_len].copy()
        num_proteins = scP_adata.n_vars
        print(f"    After truncation: {num_proteins} proteins will be predicted")
    else:
        print(f"    Using dec_max_seq_len={dec_max_seq_len} (enough for {num_proteins} proteins)")
    
    # Ensure cells match
    if not scRNA_adata.obs.index.equals(scP_adata.obs.index):
        print(f"    Warning: Cell indices don't match, aligning...")
        common_cells = scRNA_adata.obs.index.intersection(scP_adata.obs.index)
        scRNA_adata = scRNA_adata[common_cells].copy()
        scP_adata = scP_adata[common_cells].copy()
        print(f"    Aligned to {len(common_cells):,} common cells")
    
    # Create dataset
    print(f"    Creating dataset...")
    if fix_set:
        my_testset = fix_SCDataset(scRNA_adata, scP_adata, enc_max_seq_len, dec_max_seq_len)
    else:
        my_testset = SCDataset(scRNA_adata, scP_adata, enc_max_seq_len, dec_max_seq_len)
    
    # Create dataloader with num_workers for faster data loading
    # Note: pin_memory can use extra memory, but helps with GPU transfer speed
    # Consider reducing num_workers if memory is tight
    # For memory-constrained scenarios, reduce num_workers or set to 0
    test_loader = torch.utils.data.DataLoader(
        my_testset,
        batch_size=test_batch_size,
        drop_last=False,  # Don't drop last batch to process all cells
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid memory overhead from worker processes
        pin_memory=True if use_cuda else False,  # Re-enabled for faster GPU transfer
        persistent_workers=False  # Don't keep workers alive between epochs (saves memory)
    )
    
    print(f"    Created dataloader with {len(test_loader)} batches (batch size: {test_batch_size})")
    
    # Verify model is on correct device
    if use_cuda:
        next_param = next(model.parameters())
        actual_device = next_param.device
        print(f"    Model device: {actual_device}")
        if actual_device.type != 'cuda':
            print(f"    Warning: Model is not on GPU! Moving to GPU...")
            model = model.to(device)
    
    # Run inference
    print(f"    Running inference...")
    if use_cuda:
        print(f"    GPU Memory before inference: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"    GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    start_time = time.time()
    
    # Import test function from utils
    from utils import test
    
    test_loss, test_ccc, y_hat, y = test(model, device, test_loader)
    
    # Clear GPU cache after inference to free up memory
    if use_cuda:
        torch.cuda.empty_cache()
        print(f"    GPU Memory after inference: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"    GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    elapsed_time = time.time() - start_time
    print(f"    ✓ Inference completed in {elapsed_time:.2f}s")
    
    # After truncation, predictions should match the number of proteins in the dataset
    # (which should be <= dec_max_seq_len)
    num_proteins_processed = scP_adata.n_vars
    print(f"    Proteins processed: {num_proteins_processed} (truncated to fit model max_seq_len={dec_max_seq_len})")
    print(f"    Predictions shape: {y_hat.shape} (should be [n_cells, {num_proteins_processed}])")
    
    # Verify prediction shape matches expected number of proteins
    if y_hat.shape[1] != num_proteins_processed:
        print(f"    Warning: Prediction shape {y_hat.shape[1]} doesn't match expected {num_proteins_processed} proteins")
        # Truncate if needed
        if y_hat.shape[1] > num_proteins_processed:
            print(f"    Truncating predictions to {num_proteins_processed} proteins")
            y_hat = y_hat[:, :num_proteins_processed]
            y = y[:, :num_proteins_processed] if y.shape[1] > num_proteins_processed else y
        elif y_hat.shape[1] < num_proteins_processed:
            print(f"    Error: Predictions have fewer proteins than expected!")
            raise ValueError(f"Predictions have {y_hat.shape[1]} proteins but expected {num_proteins_processed}")
    
    # Create predictions DataFrame for processed proteins
    y_pred = pd.DataFrame(y_hat, columns=scP_adata.var.index.tolist())
    y_truth = pd.DataFrame(y, columns=scP_adata.var.index.tolist())
    
    # Clear intermediate numpy arrays if they're large
    del y_hat, y
    if use_cuda:
        torch.cuda.empty_cache()
    
    # Set cell indices
    if len(y_pred) == len(scRNA_adata):
        y_pred.index = scRNA_adata.obs.index[:len(y_pred)]
        y_truth.index = scRNA_adata.obs.index[:len(y_truth)]
    
    print(f"    Predictions shape: {y_pred.shape} (cells × proteins)")
    
    # Calculate metrics
    metrics = {
        'test_loss': test_loss,
        'test_ccc': test_ccc,
        'mse': test_loss,
        'ccc': test_ccc
    }
    
    print(f"    Metrics:")
    print(f"      MSE: {metrics['mse']:.4f}")
    print(f"      CCC: {metrics['ccc']:.4f}")
    
    # Save predictions as AnnData
    print(f"    Saving predictions as AnnData...")
    
    # Create AnnData from predictions - include all proteins
    pred_adata = ad.AnnData(
        X=y_pred.values.copy(),  # Explicit copy to avoid reference issues
        obs=scRNA_adata.obs.iloc[:len(y_pred)].copy() if len(y_pred) <= len(scRNA_adata) else scRNA_adata.obs.copy(),
        var=scP_adata.var.copy()  # Include all proteins
    )
    
    # Ensure cell names match
    if len(pred_adata) == len(y_pred):
        pred_adata.obs_names = y_pred.index
    
    # Add truth as layer if available
    if len(y_truth) > 0 and len(y_truth) == len(pred_adata):
        pred_adata.layers['truth'] = y_truth.loc[pred_adata.obs_names, pred_adata.var_names].values.copy()
        print(f"    ✓ Added ground truth as layer")
    
    # Clear DataFrames before saving to free memory
    del y_pred, y_truth
    if use_cuda:
        torch.cuda.empty_cache()
    
    # Save
    pred_adata.write_h5ad(output_path)
    print(f"    ✓ Saved predictions to: {output_path}")
    
    # Final cleanup
    if use_cuda:
        torch.cuda.empty_cache()
    
    return pred_adata, metrics


# test function is imported from utils.py, no need to redefine

