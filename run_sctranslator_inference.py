"""
Helper script to run scTranslator inference with proper data preprocessing.

This script:
1. Takes RNA and protein h5ad files as input
2. Runs ID mapping to scTranslator vocabulary
3. Calls scTranslator inference
4. Returns predictions in standard format

Usage:
    python run_sctranslator_inference.py \
        --rna_path /path/to/rna.h5ad \
        --protein_path /path/to/protein.h5ad \
        --output_path /path/to/predictions.h5ad \
        --checkpoint /path/to/scTranslator_2M.pt
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import anndata as ad
import pandas as pd
import numpy as np


def run_id_mapping(input_path, output_path, gene_type='human_gene_symbol', gene_column='index'):
    """
    Run scTranslator ID mapping preprocessing.

    Args:
        input_path: Path to input h5ad file
        output_path: Path to save mapped h5ad file
        gene_type: Type of gene identifiers (default: human_gene_symbol)
        gene_column: Column containing gene names (default: index)
    """
    sctranslator_dir = "/home/nebius/scTranslator"

    cmd = [
        "conda", "run", "-n", "new_env", "python",
        f"{sctranslator_dir}/code/model/data_preprocessing_ID_convert.py",
        f"--origin_gene_type={gene_type}",
        f"--origin_gene_column={gene_column}",
        f"--data_path={input_path}"
    ]

    print(f"Running ID mapping for {input_path}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ID mapping failed: {result.stderr}")
        raise RuntimeError(f"ID mapping failed: {result.stderr}")

    # The script creates a file with _mapped suffix
    expected_output = input_path.replace('.h5ad', '_mapped.h5ad')

    if not os.path.exists(expected_output):
        raise FileNotFoundError(f"Expected output {expected_output} not found after ID mapping")

    # Move to desired output path if different
    if expected_output != output_path:
        shutil.move(expected_output, output_path)

    print(f"✓ ID mapping completed: {output_path}")
    return output_path


def run_sctranslator_inference(rna_path, protein_path, checkpoint_path, output_dir):
    """
    Run scTranslator inference.

    Args:
        rna_path: Path to RNA h5ad (with my_Id column)
        protein_path: Path to protein h5ad (with my_Id column)
        checkpoint_path: Path to scTranslator checkpoint
        output_dir: Directory to save results
    """
    sctranslator_dir = "/home/nebius/scTranslator"

    cmd = [
        "conda", "run", "-n", "new_env", "python",
        f"{sctranslator_dir}/code/main_scripts/stage3_inference_without_finetune.py",
        f"--pretrain_checkpoint={checkpoint_path}",
        f"--RNA_path={rna_path}",
        f"--Pro_path={protein_path}"
    ]

    print(f"\nRunning scTranslator inference...")
    print(f"  RNA: {rna_path}")
    print(f"  Protein: {protein_path}")
    print(f"  Checkpoint: {checkpoint_path}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=sctranslator_dir)

    print(result.stdout)

    if result.returncode != 0:
        print(f"Inference failed: {result.stderr}")
        raise RuntimeError(f"Inference failed: {result.stderr}")

    print("✓ Inference completed")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description='Run scTranslator inference with automatic preprocessing')
    parser.add_argument('--rna_path', type=str, required=True,
                        help='Path to RNA expression h5ad file')
    parser.add_argument('--protein_path', type=str, required=True,
                        help='Path to protein expression h5ad file')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save predictions (optional)')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/nebius/scTranslator/checkpoint/scTranslator_2M.pt',
                        help='Path to scTranslator checkpoint')
    parser.add_argument('--gene_type', type=str, default='human_gene_symbol',
                        help='Gene identifier type')
    parser.add_argument('--gene_column', type=str, default='index',
                        help='Column containing gene names')
    parser.add_argument('--skip_id_mapping', action='store_true',
                        help='Skip ID mapping if already done')

    args = parser.parse_args()

    print("=" * 70)
    print("scTranslator Inference Runner")
    print("=" * 70)

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:

        # Check if ID mapping is needed
        if args.skip_id_mapping:
            print("\nSkipping ID mapping (--skip_id_mapping flag set)")
            rna_mapped = args.rna_path
            protein_mapped = args.protein_path

            # Verify my_Id column exists
            adata_rna = ad.read_h5ad(rna_mapped)
            adata_protein = ad.read_h5ad(protein_mapped)

            if 'my_Id' not in adata_rna.var.columns or 'my_Id' not in adata_protein.var.columns:
                raise ValueError("Input files missing 'my_Id' column. Remove --skip_id_mapping flag.")

            print(f"✓ Validated: my_Id column present in both files")
        else:
            # Run ID mapping for RNA
            print("\n" + "=" * 70)
            print("Step 1: Mapping RNA gene IDs to scTranslator vocabulary")
            print("=" * 70)
            rna_mapped = os.path.join(tmpdir, "rna_mapped.h5ad")
            run_id_mapping(args.rna_path, rna_mapped, args.gene_type, args.gene_column)

            # Run ID mapping for protein
            print("\n" + "=" * 70)
            print("Step 2: Mapping protein IDs to scTranslator vocabulary")
            print("=" * 70)
            protein_mapped = os.path.join(tmpdir, "protein_mapped.h5ad")
            run_id_mapping(args.protein_path, protein_mapped, args.gene_type, args.gene_column)

        # Run inference
        print("\n" + "=" * 70)
        print("Step 3: Running scTranslator inference")
        print("=" * 70)

        output = run_sctranslator_inference(
            rna_mapped,
            protein_mapped,
            args.checkpoint,
            tmpdir
        )

        # Parse results from output
        print("\n" + "=" * 70)
        print("Results")
        print("=" * 70)

        # Extract performance metrics from output
        for line in output.split('\n'):
            if 'AVG mse' in line or 'AVG ccc' in line:
                print(line)

        if args.output_path:
            print(f"\nNote: Predictions are not automatically saved by scTranslator")
            print(f"      You may need to modify the inference script to save predictions")

        print("\n" + "=" * 70)
        print("Completed!")
        print("=" * 70)


if __name__ == "__main__":
    main()
