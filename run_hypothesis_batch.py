"""
Batch Hypothesis Engine Runner
Processes multiple perturbations in parallel on GPU
"""

import torch
import pandas as pd
import json
import argparse
from datetime import datetime
from pathlib import Path
from hypothesis_engine import ModelGraph, HypothesisAgent


def get_all_perturbations(metadata_path):
    """Extract all unique perturbations from metadata."""
    meta_df = pd.read_csv(metadata_path, index_col=0)

    # Get sgRNA column
    sgRNA_col = 'sgRNA' if 'sgRNA' in meta_df.columns else 'sgRNA_target'

    # Filter out control/empty cells
    perturbations = meta_df[sgRNA_col].dropna()
    perturbations = perturbations[perturbations != '']
    perturbations = perturbations[~perturbations.str.lower().str.contains('non-targeting|control', na=False)]

    # Extract unique gene names (remove guide number suffix)
    unique_perturbations = perturbations.str.replace(r'_\d+$', '', regex=True).unique()

    return sorted(unique_perturbations.tolist())


def main():
    parser = argparse.ArgumentParser(description='Run hypothesis engine on multiple perturbations')
    parser.add_argument('--state-model-path', type=str, default=None,
                        help='Path to STATE model folder')
    parser.add_argument('--output-dir', type=str, default='/home/nebius/cellian/results',
                        help='Directory to save results')
    parser.add_argument('--perturbations', type=str, nargs='+', default=None,
                        help='Specific perturbations to test (default: all)')
    parser.add_argument('--max-perturbations', type=int, default=None,
                        help='Maximum number of perturbations to test')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MULTI-OMICS HYPOTHESIS ENGINE - BATCH MODE")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 80)
    print()

    # Initialize model graph
    print("Initializing ModelGraph...")
    graph = ModelGraph(state_model_path=args.state_model_path)

    # Move CAPTAIN model to GPU if available
    if torch.cuda.is_available() and hasattr(graph.protein_model, 'to'):
        graph.protein_model = graph.protein_model.to('cuda')
        print("✓ CAPTAIN model moved to GPU\n")

    # Initialize hypothesis agent
    print("Initializing HypothesisAgent...")
    agent = HypothesisAgent(graph)

    # Get perturbations to test
    if args.perturbations:
        perturbations = args.perturbations
    else:
        metadata_path = "/home/nebius/cellian/data/perturb-cite-seq/SCP1064/metadata/RNA_metadata.csv"
        perturbations = get_all_perturbations(metadata_path)

    if args.max_perturbations:
        perturbations = perturbations[:args.max_perturbations]

    print(f"\nTesting {len(perturbations)} perturbations")
    print("=" * 80)
    print()

    # Run hypothesis generation for all perturbations
    results = []
    successful = 0
    failed = 0

    for i, perturbation in enumerate(perturbations, 1):
        print(f"\n[{i}/{len(perturbations)}] Processing: {perturbation}")
        print("-" * 80)

        try:
            result = agent.generate_hypothesis(perturbation)
            results.append(result)

            if result.get('rna_score') is not None:
                successful += 1
            else:
                failed += 1

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            failed += 1
            results.append({
                'perturbation': perturbation,
                'path': 'Perturbation -> STATE -> CAPTAIN',
                'rna_score': None,
                'protein_score': None,
                'error_propagation': None,
                'error': str(e)
            })

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save JSON
    json_path = output_dir / f"hypothesis_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {json_path}")

    # Save CSV
    df_results = pd.DataFrame(results)
    csv_path = output_dir / f"hypothesis_results_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Perturbations: {len(perturbations)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        valid_results = [r for r in results if r.get('rna_score') is not None]
        avg_rna_score = sum(r['rna_score'] for r in valid_results) / len(valid_results)
        avg_protein_score = sum(r['protein_score'] for r in valid_results) / len(valid_results)
        avg_error_prop = sum(r['error_propagation'] for r in valid_results) / len(valid_results)

        print(f"\nAverage Scores:")
        print(f"  RNA Score: {avg_rna_score:.4f}")
        print(f"  Protein Score: {avg_protein_score:.4f}")
        print(f"  Error Propagation: {avg_error_prop:.4f}")

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
