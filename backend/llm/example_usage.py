"""Example usage of the LLM module with Agent_Tools results."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.input import process_user_query
from llm.output import interpret_results, generate_summary, collect_results_from_pipeline


def example_usage():
    """Example of how to use the LLM module."""
    
    # Example 1: Process user query
    query = "What happens if I knock down TP53?"
    print("=" * 70)
    print("EXAMPLE 1: Processing User Query")
    print("=" * 70)
    print(f"Query: {query}")
    
    perturbation_info = process_user_query(query)
    print(f"\nExtracted perturbation info:")
    print(f"  Target: {perturbation_info['target']}")
    print(f"  Type: {perturbation_info['type']}")
    print(f"  Confidence: {perturbation_info['confidence']}")
    
    # Example 2: Collect results from Agent_Tools pipeline
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Collecting Results from Agent_Tools Pipeline")
    print("=" * 70)
    
    # Option A: Collect from output directory (if pipeline has been run)
    # output_dir = "Agent_Tools/temp_output"
    # results = collect_results_from_pipeline(
    #     output_dir=output_dir,
    #     target_gene="TP53",
    #     rna_metrics=None,  # Will try to load from files
    #     protein_metrics=None  # Will try to load from files
    # )
    
    # Option B: Manually construct results (when you have metrics from pipeline)
    # This matches what evaluate_predictions() returns and comprehensive_analysis() returns
    # Note: perturbation_pipeline.py calculates these but doesn't return them,
    # so you need to capture them manually or use collect_results_from_pipeline()
    mock_results = {
        "target_gene": "TP53",
        "rna_metrics": {
            "r2": 0.75,
            "pearson_r": 0.82,
            "rmse": 0.15,
            "mae": 0.12
        },
        "protein_metrics": {
            "r2": 0.68,
            "pearson_r": 0.74,
            "rmse": 0.18,
            "mae": 0.14
        },
        "pathway_analysis": {
            "correlation": {
                "correlation": 0.65,
                "pvalue": 0.001,
                "n_common_genes": 150
            },
            "differential_rna": None,  # Would be a DataFrame in real usage
            "differential_protein": None,  # Would be a DataFrame in real usage
            "gsea_rna": None,  # Would be a DataFrame in real usage
            "kegg_enrichment": None,  # Would be a DataFrame in real usage
            "reactome_enrichment": None,  # Would be a DataFrame in real usage
            "go_enrichment": None  # Would be a DataFrame in real usage
        }
    }
    
    interpretation = interpret_results(
        mock_results,
        query=query,
        perturbation_info=perturbation_info
    )
    
    print("\nLLM Interpretation:")
    print(interpretation)
    
    # Example 3: Generate complete summary
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Generating Complete Summary")
    print("=" * 70)
    
    summary = generate_summary(
        mock_results,
        query=query,
        perturbation_info=perturbation_info
    )
    
    print("\nComplete Summary Structure:")
    print(f"  Query: {summary['query']}")
    print(f"  Perturbation Info: {summary['perturbation_info']}")
    print(f"  Interpretation length: {len(summary['interpretation'])} characters")


if __name__ == "__main__":
    example_usage()

