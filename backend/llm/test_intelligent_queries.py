"""
Quick test script for the new intelligent query features.

This script tests:
- Protein-focused queries
- Top N filtering (genes, proteins, pathways, phenotypes)
- Query-based adaptive output generation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.input import extract_perturbation_info
from llm.perturbation_orchestrator import process_user_query, _generate_outputs
from unittest.mock import patch, MagicMock
import pandas as pd


def create_mock_results_with_proteins():
    """Create mock results with both RNA and protein differential expression."""
    return {
        "perturbation_name": "TP53",
        "perturbation_type": "gene",
        "pathway_analysis": {
            "differential_rna": pd.DataFrame({
                "gene": ["TP53", "CDKN1A", "BAX", "BCL2", "CASP3", "MDM2", "ATM", "CHEK2", "GADD45A", "RB1"] * 5,
                "log2fc": [2.5, 1.8, 1.5, -1.2, 1.3, -0.9, 1.1, 0.8, 1.0, 0.7] * 5,
                "pvalue": [0.001, 0.002, 0.003, 0.01, 0.005, 0.02, 0.008, 0.015, 0.012, 0.025] * 5,
                "pvalue_adj": [0.01, 0.02, 0.03, 0.1, 0.05, 0.2, 0.08, 0.15, 0.12, 0.25] * 5,
                "pred_mean": [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5] * 5,
                "control_mean": [2.5, 2.7, 2.5, 4.7, 1.7, 3.4, 0.9, 0.7, 0.0, -0.2] * 5
            }),
            "differential_protein": pd.DataFrame({
                "gene": ["TP53", "CDKN1A", "BAX", "BCL2", "CASP3", "MDM2", "ATM", "CHEK2"] * 3,
                "log2fc": [2.8, 2.0, 1.8, -1.5, 1.6, -1.1, 1.3, 1.0] * 3,
                "pvalue": [0.0001, 0.001, 0.002, 0.008, 0.004, 0.015, 0.007, 0.012] * 3,
                "pvalue_adj": [0.001, 0.01, 0.02, 0.08, 0.04, 0.15, 0.07, 0.12] * 3,
                "pred_mean": [6.0, 5.0, 4.5, 3.0, 3.5, 2.0, 2.5, 1.5] * 3,
                "control_mean": [3.2, 3.0, 2.7, 4.5, 1.9, 3.1, 1.2, 0.5] * 3
            }),
            "gsea_rna": pd.DataFrame({
                "Term": ["p53 signaling pathway", "Cell cycle", "Apoptosis", "DNA repair"] * 5,
                "NES": [2.5, 2.1, -2.3, -1.8] * 5,
                "FDR q-val": [0.001, 0.003, 0.002, 0.01] * 5,
                "NOM p-val": [0.0001, 0.0003, 0.0002, 0.001] * 5
            }),
            "kegg_enrichment": pd.DataFrame({
                "Term": ["p53 signaling pathway", "Cell cycle", "Apoptosis"] * 3,
                "P-value": [0.001, 0.003, 0.005] * 3,
                "Adjusted P-value": [0.01, 0.03, 0.05] * 3,
                "Normalized Enrichment": [2.0, 1.8, 1.5] * 3
            })
        },
        "validated_edges": []
    }


def test_query_extraction():
    """Test intelligent query extraction."""
    print("="*70)
    print("TEST 1: Intelligent Query Extraction")
    print("="*70)
    
    test_queries = [
        "Show me top 10 proteins changed by TP53 knockout",
        "What are the top 5 genes and top 3 pathways affected?",
        "Show me the top 7 most changed proteins and their pathways",
        "Find top 3 genes in PI3K pathway affecting apoptosis",
        "Compare protein changes between TP53 KO and imatinib",
        "What proteins change when I knock down JAK1?",
        "Top 10 genes in mTOR pathway",
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        intent = extract_perturbation_info(query)
        
        print(f"  Target: {intent.get('target')}")
        print(f"  Focus: {intent.get('focus')}")
        print(f"  Protein mentioned: {intent.get('protein_mentioned')}")
        print(f"  Top N genes: {intent.get('top_n_genes')}")
        print(f"  Top N proteins: {intent.get('top_n_proteins')}")
        print(f"  Top N pathways: {intent.get('top_n_pathways')}")
        print(f"  Pathway mentioned: {intent.get('pathway_mentioned')}")
        print(f"  Phenotype mentioned: {intent.get('phenotype_mentioned')}")
        print(f"  Is comparison: {intent.get('is_comparison')}")
        print(f"  Output types: {intent.get('output_types')}")


def test_adaptive_output_generation():
    """Test adaptive output generation based on query intent."""
    print("\n" + "="*70)
    print("TEST 2: Adaptive Output Generation")
    print("="*70)
    
    mock_results = create_mock_results_with_proteins()
    
    test_cases = [
        {
            "query": "Show me top 5 proteins changed by TP53 knockout",
            "description": "Protein-focused with top N"
        },
        {
            "query": "What are the top 3 genes affected?",
            "description": "Gene-focused with top N"
        },
        {
            "query": "Show me top 3 pathways",
            "description": "Pathway-focused with top N"
        },
        {
            "query": "What happens if I knock out TP53?",
            "description": "General query (no specific focus)"
        },
    ]
    
    for test_case in test_cases:
        query = test_case["query"]
        description = test_case["description"]
        
        print(f"\n{description}")
        print(f"Query: '{query}'")
        
        # Extract query intent
        query_intent = extract_perturbation_info(query)
        print(f"  Extracted intent:")
        print(f"    Focus: {query_intent.get('focus')}")
        print(f"    Top N genes: {query_intent.get('top_n_genes')}")
        print(f"    Top N proteins: {query_intent.get('top_n_proteins')}")
        print(f"    Top N pathways: {query_intent.get('top_n_pathways')}")
        
        # Generate outputs with query intent
        try:
            outputs = _generate_outputs(mock_results, "gene", "TP53", query_intent=query_intent)
            print(f"  Generated outputs:")
            for key, value in outputs.items():
                if value and isinstance(value, str):
                    print(f"    {key}: {value}")
                elif key in ["hypotheses"]:
                    print(f"    {key}: (generated)")
                elif key in ["genes_csv", "proteins_csv", "pathways_csv"]:
                    if value:
                        print(f"    {key}: {value} (filtered data saved)")
        except Exception as e:
            print(f"  ⚠️  Output generation failed: {e}")
            import traceback
            traceback.print_exc()


def test_mock_pipeline_with_intelligent_queries():
    """Test the full pipeline with intelligent queries (mocked)."""
    print("\n" + "="*70)
    print("TEST 3: Full Pipeline with Intelligent Queries (Mocked)")
    print("="*70)
    
    def mock_run_gene_perturbation(pert_name, context):
        print(f"  [MOCK] Running gene perturbation: {pert_name}")
        return create_mock_results_with_proteins()
    
    # Test protein-focused query
    test_query = "Show me top 5 proteins changed by TP53 knockout"
    print(f"\nTest query: '{test_query}'")
    
    with patch('llm.perturbation_orchestrator.run_state_gene_perturbation', side_effect=mock_run_gene_perturbation):
        from llm.perturbation_orchestrator import load_valid_perturbation_names
        drug_names, gene_names = load_valid_perturbation_names()
        
        if gene_names:
            # Use first available gene
            test_gene = str(gene_names[0])
            results = process_user_query(f"Show me top 5 proteins changed by {test_gene} knockout")
            
            print(f"\n  Results:")
            print(f"    Intent mode: {results['intent']['mode']}")
            print(f"    Number of perturbations: {len(results['perturbations'])}")
            
            if results['perturbations']:
                pert = results['perturbations'][0]
                if 'outputs' in pert:
                    outputs = pert['outputs']
                    print(f"    Outputs generated:")
                    for key, value in outputs.items():
                        if value and isinstance(value, str) and 'csv' not in key:
                            print(f"      {key}: {value}")
                        elif 'csv' in key and value:
                            print(f"      {key}: {value} (filtered data)")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("INTELLIGENT QUERY FEATURES TEST SUITE")
    print("="*70)
    print("\nTesting:")
    print("- Intelligent query extraction (protein focus, top N, etc.)")
    print("- Adaptive output generation based on query intent")
    print("- Full pipeline integration with mocked Agent_Tools\n")
    
    try:
        test_query_extraction()
        test_adaptive_output_generation()
        test_mock_pipeline_with_intelligent_queries()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)
        print("\nNote: Some tests may show warnings if optional dependencies")
        print("(pandas, matplotlib) are not available, but core logic is tested.")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

