#!/usr/bin/env python3
"""
Quick script to run pathway analysis on already generated predictions.
Usage:
    python run_pathway_analysis_debug.py
"""

import sys
import os
from pathway_analysis import comprehensive_analysis

# Use the most recent drug perturbation files
rna_pred = 'temp_output/rna_predictions_R_Verapamil_hydrochloride_0_05_uM_IFNγ.h5ad'
protein_pred = 'temp_output/protein_predictions_R_Verapamil_hydrochloride_0_05_uM_IFNγ.h5ad'
target_name = "('(R)-Verapamil (hydrochloride)', 0.05, 'uM')"
output_dir = 'temp_output/pathway_analysis_debug'

print(f'Running pathway analysis...')
print(f'  RNA predictions: {rna_pred}')
print(f'  Protein predictions: {protein_pred}')
print(f'  Target: {target_name}')
print(f'  Output: {output_dir}')
print()

results = comprehensive_analysis(
    rna_predictions_path=rna_pred,
    protein_predictions_path=protein_pred,
    control_rna_path=None,  # Control is in predictions (drug perturbation)
    control_protein_path=None,  # Control is in predictions (drug perturbation)
    target_gene=target_name,
    output_dir=output_dir,
    control_label='non-targeting'
)
print('\nDone!')

