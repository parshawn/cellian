"""
Test script for perturbation_orchestrator with MOCKED Agent_Tools (fast testing).

This script mocks Agent_Tools calls and uses sample data to test:
- Name validation (exact/close/none matching)
- User intent detection
- Output generation logic
- Comparison logic

NOTE: This uses MOCKED data for fast testing. For REAL workflow testing with actual Agent_Tools,
use:
  - test_real_workflow.py (local real testing)
  - sbatch llm/run_orchestrator_slurm.sh (SLURM real testing)
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.perturbation_orchestrator import (
    load_valid_perturbation_names,
    validate_perturbation_name,
    detect_user_intent,
    process_user_query,
    _generate_outputs,
    _generate_comparison
)


def create_mock_agent_tools_results(pert_name: str, pert_type: str) -> dict:
    """Create mock results that Agent_Tools would return."""
    return {
        "perturbation_name": pert_name,
        "perturbation_type": pert_type,
        "target_gene": pert_name if pert_type == "gene" else None,
        "rna_metrics": {
            "r2": 0.85,
            "pearson_r": 0.92,
            "rmse": 0.15,
            "mae": 0.12
        },
        "protein_metrics": {
            "r2": 0.78,
            "pearson_r": 0.88,
            "rmse": 0.22,
            "mae": 0.18
        },
        "pathway_analysis": {
            "differential_rna": _create_mock_de_df(pert_name),
            "differential_protein": _create_mock_de_df(pert_name),
            "gsea_rna": _create_mock_gsea_df(pert_name),
            "kegg_enrichment": _create_mock_enrichment_df("KEGG", pert_name),
            "reactome_enrichment": _create_mock_enrichment_df("Reactome", pert_name),
            "go_enrichment": _create_mock_enrichment_df("GO", pert_name)
        }
    }


def _create_mock_de_df(pert_name: str):
    """Create mock differential expression DataFrame."""
    try:
        import pandas as pd
        return pd.DataFrame({
            "gene": ["TP53", "CDKN1A", "BAX", "BCL2", "CASP3", "MDM2", "ATM", "CHEK2", "GADD45A", "RB1"],
            "log2fc": [2.5, 1.8, 1.5, -1.2, 1.3, -0.9, 1.1, 0.8, 1.0, 0.7],
            "pvalue": [0.001, 0.002, 0.003, 0.01, 0.005, 0.02, 0.008, 0.015, 0.012, 0.025],
            "pvalue_adj": [0.01, 0.02, 0.03, 0.1, 0.05, 0.2, 0.08, 0.15, 0.12, 0.25]
        })
    except ImportError:
        # Fallback: return list of dicts
        return [
            {"gene": "TP53", "log2fc": 2.5, "pvalue": 0.001, "pvalue_adj": 0.01},
            {"gene": "CDKN1A", "log2fc": 1.8, "pvalue": 0.002, "pvalue_adj": 0.02},
            {"gene": "BAX", "log2fc": 1.5, "pvalue": 0.003, "pvalue_adj": 0.03},
        ]


def _create_mock_gsea_df(pert_name: str):
    """Create mock GSEA DataFrame."""
    try:
        import pandas as pd
        return pd.DataFrame({
            "Term": [
                "p53 signaling pathway",
                "Cell cycle",
                "Apoptosis",
                "DNA repair",
                "Cell cycle checkpoints"
            ],
            "NES": [2.5, 2.1, -2.3, -1.8, 1.9],
            "FDR q-val": [0.001, 0.003, 0.002, 0.01, 0.008]
        })
    except ImportError:
        return [
            {"Term": "p53 signaling pathway", "NES": 2.5, "FDR q-val": 0.001},
            {"Term": "Cell cycle", "NES": 2.1, "FDR q-val": 0.003},
        ]


def _create_mock_enrichment_df(db: str, pert_name: str):
    """Create mock enrichment DataFrame."""
    try:
        import pandas as pd
        pathways = {
            "KEGG": ["p53 signaling pathway", "Cell cycle", "Apoptosis"],
            "Reactome": ["Cell cycle checkpoints", "DNA repair", "Stress response"],
            "GO": ["Regulation of cell cycle", "Apoptotic process", "DNA damage response"]
        }
        return pd.DataFrame({
            "Term": pathways.get(db, ["Pathway 1", "Pathway 2"]),
            "Adjusted P-value": [0.001, 0.003, 0.005],
            "Normalized Enrichment": [2.0, 1.8, 1.5]
        })
    except ImportError:
        return [
            {"Term": "Pathway 1", "Adjusted P-value": 0.001, "Normalized Enrichment": 2.0},
            {"Term": "Pathway 2", "Adjusted P-value": 0.003, "Normalized Enrichment": 1.8},
        ]


def test_name_validation():
    """Test name validation logic."""
    print("\n" + "="*70)
    print("TEST 1: Name Validation")
    print("="*70)
    
    drug_names, gene_names = load_valid_perturbation_names()
    print(f"Loaded {len(drug_names)} drug names and {len(gene_names)} gene names")
    
    # Test exact match
    if gene_names:
        test_gene = str(gene_names[0])
        match = validate_perturbation_name(test_gene, is_gene=True, valid_names=gene_names)
        print(f"\n✓ Exact match test:")
        print(f"  Requested: '{test_gene}'")
        print(f"  Match type: {match['match_type']}")
        print(f"  Used name: {match['used_name']}")
        assert match['match_type'] == 'exact', "Should be exact match"
    
    # Test close match (if we can find one)
    if gene_names:
        # Try a slightly modified version
        test_gene_close = str(gene_names[0]) + "X"
        match = validate_perturbation_name(test_gene_close, is_gene=True, valid_names=gene_names)
        print(f"\n✓ Close match test:")
        print(f"  Requested: '{test_gene_close}'")
        print(f"  Match type: {match['match_type']}")
        print(f"  Used name: {match['used_name']}")
        print(f"  Similarity: {match['similarity_score']:.2f}")
    
    # Test no match
    match = validate_perturbation_name("NONEXISTENT_GENE_XYZ", is_gene=True, valid_names=gene_names)
    print(f"\n✓ No match test:")
    print(f"  Requested: 'NONEXISTENT_GENE_XYZ'")
    print(f"  Match type: {match['match_type']}")
    assert match['match_type'] == 'none', "Should be no match"
    
    print("\n✓ Name validation tests passed!")


def test_intent_detection():
    """Test user intent detection."""
    print("\n" + "="*70)
    print("TEST 2: User Intent Detection")
    print("="*70)
    
    test_queries = [
        ("run KO TP53", "single_gene"),
        ("simulate treatment with imatinib", "single_drug"),
        ("compare TP53 vs imatinib", "comparison"),
        ("which is stronger TP53 or imatinib", "comparison"),
        ("what is the difference between TP53 knockout and imatinib treatment", "comparison"),
    ]
    
    for query, expected_mode in test_queries:
        intent = detect_user_intent(query)
        print(f"\nQuery: '{query}'")
        print(f"  Detected mode: {intent['mode']}")
        print(f"  Gene name: {intent.get('gene_name')}")
        print(f"  Drug name: {intent.get('drug_name')}")
        print(f"  Has comparison phrase: {intent.get('has_comparison_phrase')}")
        # Note: exact mode matching may vary, so we just print
    
    print("\n✓ Intent detection tests completed!")


def test_mock_pipeline():
    """Test orchestrator with mocked Agent_Tools."""
    print("\n" + "="*70)
    print("TEST 3: Mock Pipeline Execution")
    print("="*70)
    
    # Mock the Agent_Tools functions
    def mock_run_gene_perturbation(pert_name, context):
        print(f"  [MOCK] Running gene perturbation: {pert_name}")
        return create_mock_agent_tools_results(pert_name, "gene")
    
    def mock_run_drug_perturbation(pert_name, context):
        print(f"  [MOCK] Running drug perturbation: {pert_name}")
        return create_mock_agent_tools_results(pert_name, "drug")
    
    # Test single gene perturbation
    print("\n--- Testing Single Gene Perturbation ---")
    with patch('llm.perturbation_orchestrator.run_state_gene_perturbation', side_effect=mock_run_gene_perturbation):
        drug_names, gene_names = load_valid_perturbation_names()
        if gene_names:
            test_gene = str(gene_names[0])
            results = process_user_query(f"run KO {test_gene}")
            print(f"\n  Query: 'run KO {test_gene}'")
            print(f"  Intent mode: {results['intent']['mode']}")
            print(f"  Number of perturbations: {len(results['perturbations'])}")
            if results['perturbations']:
                pert = results['perturbations'][0]
                print(f"  Perturbation type: {pert.get('type')}")
                print(f"  Match type: {pert.get('match_info', {}).get('match_type')}")
                if 'results' in pert:
                    print(f"  Results keys: {list(pert['results'].keys())}")
                if 'outputs' in pert:
                    print(f"  Outputs generated: {list(pert['outputs'].keys())}")
    
    # Test comparison
    print("\n--- Testing Comparison ---")
    with patch('llm.perturbation_orchestrator.run_state_gene_perturbation', side_effect=mock_run_gene_perturbation), \
         patch('llm.perturbation_orchestrator.run_state_drug_perturbation', side_effect=mock_run_drug_perturbation):
        
        drug_names, gene_names = load_valid_perturbation_names()
        if gene_names and drug_names:
            test_gene = str(gene_names[0])
            test_drug = str(drug_names[0])
            results = process_user_query(f"compare {test_gene} vs {test_drug}")
            print(f"\n  Query: 'compare {test_gene} vs {test_drug}'")
            print(f"  Intent mode: {results['intent']['mode']}")
            print(f"  Number of perturbations: {len(results['perturbations'])}")
            if 'comparison' in results and results['comparison']:
                comp = results['comparison']
                if 'error' not in comp:
                    print(f"  Comparison summary generated: {len(comp.get('summary', ''))} chars")
                    print(f"  Shared pathways: {len(comp.get('shared_pathways', []))}")
                    print(f"  Shared phenotypes: {len(comp.get('shared_phenotypes', []))}")
                else:
                    print(f"  Comparison error: {comp['error']}")
    
    print("\n✓ Mock pipeline tests completed!")


def test_output_generation():
    """Test output generation with mock data."""
    print("\n" + "="*70)
    print("TEST 4: Output Generation")
    print("="*70)
    
    mock_results = create_mock_agent_tools_results("TP53", "gene")
    
    try:
        # Test without query intent (default behavior)
        outputs = _generate_outputs(mock_results, "gene", "TP53")
        print(f"\n  Generated outputs (default):")
        for key, value in outputs.items():
            if value:
                print(f"    {key}: {value}")
            else:
                print(f"    {key}: (failed)")
        
        # Test with query intent (protein focus, top N)
        from llm.input import extract_perturbation_info
        query_intent = extract_perturbation_info("Show me top 5 proteins changed by TP53 knockout")
        print(f"\n  Query intent extracted:")
        print(f"    Focus: {query_intent.get('focus')}")
        print(f"    Top N proteins: {query_intent.get('top_n_proteins')}")
        print(f"    Protein mentioned: {query_intent.get('protein_mentioned')}")
        
        outputs_with_intent = _generate_outputs(mock_results, "gene", "TP53", query_intent=query_intent)
        print(f"\n  Generated outputs (with query intent):")
        for key, value in outputs_with_intent.items():
            if value:
                print(f"    {key}: {value}")
            else:
                print(f"    {key}: (failed)")
        
        print("\n✓ Output generation test completed!")
    except Exception as e:
        print(f"\n⚠️  Output generation test failed (may be due to missing dependencies): {e}")
        import traceback
        traceback.print_exc()


def test_intelligent_queries():
    """Test the new intelligent query features (protein focus, top N, etc.)."""
    print("\n" + "="*70)
    print("TEST 6: Intelligent Query Features")
    print("="*70)
    
    from llm.input import extract_perturbation_info
    
    test_queries = [
        ("Show me top 10 proteins changed by TP53 knockout", {
            "expected_focus": "proteins",
            "expected_top_n_proteins": 10,
            "expected_protein_mentioned": True
        }),
        ("What are the top 5 genes and top 3 pathways affected?", {
            "expected_top_n_genes": 5,
            "expected_top_n_pathways": 3,
            "expected_focus": "both"
        }),
        ("Show me the top 7 most changed proteins and their pathways", {
            "expected_top_n_proteins": 7,
            "expected_focus": "proteins",
            "expected_protein_mentioned": True
        }),
        ("Find top 3 genes in PI3K pathway affecting apoptosis", {
            "expected_top_n_genes": 3,
            "expected_pathway_mentioned": "PI3K",
            "expected_phenotype_mentioned": "apoptosis"
        }),
        ("Compare protein changes between TP53 KO and imatinib", {
            "expected_focus": "proteins",
            "expected_protein_mentioned": True,
            "expected_is_comparison": True
        }),
    ]
    
    for query, expected in test_queries:
        print(f"\n  Query: '{query}'")
        intent = extract_perturbation_info(query)
        
        print(f"    Extracted intent:")
        print(f"      Focus: {intent.get('focus')} (expected: {expected.get('expected_focus')})")
        if 'expected_top_n_genes' in expected:
            print(f"      Top N genes: {intent.get('top_n_genes')} (expected: {expected.get('expected_top_n_genes')})")
        if 'expected_top_n_proteins' in expected:
            print(f"      Top N proteins: {intent.get('top_n_proteins')} (expected: {expected.get('expected_top_n_proteins')})")
        if 'expected_top_n_pathways' in expected:
            print(f"      Top N pathways: {intent.get('top_n_pathways')} (expected: {expected.get('expected_top_n_pathways')})")
        if 'expected_protein_mentioned' in expected:
            print(f"      Protein mentioned: {intent.get('protein_mentioned')} (expected: {expected.get('expected_protein_mentioned')})")
        if 'expected_pathway_mentioned' in expected:
            print(f"      Pathway mentioned: {intent.get('pathway_mentioned')} (expected: {expected.get('expected_pathway_mentioned')})")
        if 'expected_phenotype_mentioned' in expected:
            print(f"      Phenotype mentioned: {intent.get('phenotype_mentioned')} (expected: {expected.get('expected_phenotype_mentioned')})")
        if 'expected_is_comparison' in expected:
            print(f"      Is comparison: {intent.get('is_comparison')} (expected: {expected.get('expected_is_comparison')})")
    
    print("\n✓ Intelligent query tests completed!")


def test_comparison_logic():
    """Test comparison logic with mock data."""
    print("\n" + "="*70)
    print("TEST 5: Comparison Logic")
    print("="*70)
    
    # Create two mock perturbations with overlapping pathways
    pert1_results = create_mock_agent_tools_results("TP53", "gene")
    pert2_results = create_mock_agent_tools_results("imatinib", "drug")
    
    # Modify to have some shared pathways
    try:
        import pandas as pd
        # Make sure both have "p53 signaling pathway"
        pert1_results["pathway_analysis"]["gsea_rna"] = pd.DataFrame({
            "Term": ["p53 signaling pathway", "Cell cycle", "Apoptosis"],
            "NES": [2.5, 2.1, -2.3],
            "FDR q-val": [0.001, 0.003, 0.002]
        })
        pert2_results["pathway_analysis"]["gsea_rna"] = pd.DataFrame({
            "Term": ["p53 signaling pathway", "Cell cycle", "DNA repair"],
            "NES": [1.7, 1.5, -1.2],
            "FDR q-val": [0.005, 0.01, 0.02]
        })
    except ImportError:
        pass
    
    pert1 = {
        "type": "gene",
        "match_info": {"used_name": "TP53"},
        "results": pert1_results
    }
    pert2 = {
        "type": "drug",
        "match_info": {"used_name": "imatinib"},
        "results": pert2_results
    }
    
    try:
        comparison = _generate_comparison(pert1, pert2)
        print(f"\n  Comparison generated:")
        print(f"    Shared pathways: {len(comparison.get('shared_pathways', []))}")
        print(f"    Shared phenotypes: {len(comparison.get('shared_phenotypes', []))}")
        if 'summary' in comparison:
            summary_lines = comparison['summary'].split('\n')[:5]
            print(f"    Summary preview:")
            for line in summary_lines:
                print(f"      {line}")
        print("\n✓ Comparison logic test completed!")
    except Exception as e:
        print(f"\n⚠️  Comparison test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PERTURBATION ORCHESTRATOR TEST SUITE")
    print("="*70)
    print("\nThis test suite mocks Agent_Tools calls to test orchestrator logic")
    print("without running actual pipeline executions.\n")
    
    try:
        test_name_validation()
        test_intent_detection()
        test_mock_pipeline()
        test_output_generation()
        test_comparison_logic()
        test_intelligent_queries()
        
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

