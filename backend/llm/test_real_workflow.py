"""
Test script for the REAL workflow using actual Agent_Tools.

This script tests the complete orchestrator workflow by:
1. Validating perturbation names
2. Detecting user intent
3. Actually calling Agent_Tools perturbation_pipeline
4. Collecting and processing results from Agent_Tools
5. Generating outputs (plots, hypotheses, reports)

Usage:
    # Test locally (if you have Agent_Tools set up):
    python llm/test_real_workflow.py "run KO TP53"
    
    # Or test on SLURM (recommended):
    sbatch llm/run_orchestrator_slurm.sh "run KO TP53"
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.perturbation_orchestrator import (
    process_user_query,
    load_valid_perturbation_names,
    validate_perturbation_name,
    detect_user_intent
)


def test_agent_tools_integration():
    """Test that Agent_Tools is available and can be called."""
    print("="*70)
    print("TEST 1: Agent_Tools Integration Check")
    print("="*70)
    
    agent_tools_script = Path(__file__).parent.parent / "Agent_Tools" / "perturbation_pipeline.py"
    
    if not agent_tools_script.exists():
        print(f"\n❌ Agent_Tools not found at: {agent_tools_script}")
        print("   Make sure Agent_Tools directory exists in the project root.")
        return False
    
    print(f"\n✓ Agent_Tools found at: {agent_tools_script}")
    
    # Check if required Agent_Tools modules exist
    agent_tools_path = agent_tools_script.parent
    required_modules = [
        "perturbation_pipeline.py",
        "state_inference.py",
        "sctranslator_inference.py",
        "pathway_analysis.py"
    ]
    
    missing = []
    for module in required_modules:
        module_path = agent_tools_path / module
        if not module_path.exists():
            missing.append(module)
        else:
            print(f"  ✓ {module} found")
    
    if missing:
        print(f"\n⚠️  Missing modules: {missing}")
        return False
    
    print("\n✓ Agent_Tools integration check passed!")
    return True


def test_query_validation():
    """Test query validation and intent detection with real data."""
    print("\n" + "="*70)
    print("TEST 2: Query Validation and Intent Detection")
    print("="*70)
    
    # Load valid perturbation names
    print("\nLoading valid perturbation names...")
    drug_names, gene_names = load_valid_perturbation_names()
    print(f"  ✓ Loaded {len(drug_names)} drug names")
    print(f"  ✓ Loaded {len(gene_names)} gene names")
    
    # Test queries
    test_queries = [
        "run KO TP53",
        "Show me top 10 proteins changed by TP53 knockout",
        "What are the top 5 genes affected?",
        "Compare TP53 vs imatinib",
        "Compare protein changes between TP53 KO and imatinib",
    ]
    
    print("\nTesting query validation and intent detection:")
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        
        # Detect intent
        intent = detect_user_intent(query)
        print(f"    Mode: {intent['mode']}")
        
        if intent.get('gene_name'):
            gene_name = intent['gene_name']
            match_info = validate_perturbation_name(gene_name, is_gene=True, valid_names=gene_names)
            print(f"    Gene: {gene_name} -> {match_info.get('used_name')} (match: {match_info.get('match_type')})")
        
        if intent.get('drug_name'):
            drug_name = intent['drug_name']
            match_info = validate_perturbation_name(drug_name, is_gene=False, valid_names=drug_names)
            used_name = match_info.get('used_name', '')[:80] + '...' if len(str(match_info.get('used_name', ''))) > 80 else match_info.get('used_name')
            print(f"    Drug: {drug_name} -> {used_name} (match: {match_info.get('match_type')})")
    
    print("\n✓ Query validation test completed!")
    return True


def test_real_workflow(query: str, condition: str = "Control"):
    """
    Test the complete real workflow with Agent_Tools.
    
    Args:
        query: User query string
        condition: Condition to use (default: "Control")
    """
    print("\n" + "="*70)
    print("TEST 3: Real Workflow Execution")
    print("="*70)
    print(f"\nQuery: '{query}'")
    print(f"Condition: {condition}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*70 + "\n")
    
    # Create context
    context = {
        "condition": condition
    }
    
    # Process query (this will actually call Agent_Tools)
    try:
        results = process_user_query(query, context=context)
        
        print("\n" + "="*70)
        print("WORKFLOW RESULTS SUMMARY")
        print("="*70)
        print(f"Query: {query}")
        print(f"Intent mode: {results.get('intent', {}).get('mode', 'unknown')}")
        print(f"Number of perturbations: {len(results.get('perturbations', []))}")
        
        # Check perturbations
        for i, pert in enumerate(results.get('perturbations', []), 1):
            print(f"\nPerturbation {i}:")
            print(f"  Type: {pert.get('type')}")
            match_info = pert.get('match_info', {})
            print(f"  Requested: {match_info.get('requested_name')}")
            print(f"  Used: {match_info.get('used_name')}")
            print(f"  Match type: {match_info.get('match_type')}")
            
            if 'error' in pert:
                print(f"  ❌ Error: {pert['error']}")
            elif 'outputs' in pert:
                outputs = pert['outputs']
                print(f"  ✓ Outputs generated:")
                for key, value in outputs.items():
                    if value and isinstance(value, str):
                        # Show if file exists
                        file_path = Path(value)
                        if file_path.exists():
                            size = file_path.stat().st_size / 1024  # KB
                            print(f"    {key}: {value} ({size:.1f} KB)")
                        else:
                            print(f"    {key}: {value} (file not found)")
                    elif key == "hypotheses" and value:
                        num_hyp = len(value.get('hypotheses', []))
                        print(f"    {key}: {num_hyp} hypotheses generated")
        
        # Check comparison
        if 'comparison' in results and results['comparison']:
            comp = results['comparison']
            if 'error' not in comp:
                print(f"\n✓ Comparison generated:")
                print(f"  Shared pathways: {len(comp.get('shared_pathways', []))}")
                print(f"  Shared phenotypes: {len(comp.get('shared_phenotypes', []))}")
                if 'comparison_report' in comp:
                    report_path = Path(comp['comparison_report'])
                    if report_path.exists():
                        size = report_path.stat().st_size / 1024
                        print(f"  Report: {comp['comparison_report']} ({size:.1f} KB)")
            else:
                print(f"\n❌ Comparison error: {comp['error']}")
        
        print("\n" + "="*70)
        print("WORKFLOW COMPLETED")
        print("="*70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Print output locations
        output_dir = Path(__file__).parent / "perturbation_outputs"
        if output_dir.exists():
            print(f"\nResults saved in: {output_dir}")
            print(f"Generated directories:")
            for pert_dir in sorted(output_dir.glob("*")):
                if pert_dir.is_dir():
                    num_files = len(list(pert_dir.glob("*")))
                    print(f"  {pert_dir.name}/ ({num_files} files)")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the real perturbation orchestrator workflow")
    parser.add_argument("query", nargs="?", help="User query (e.g., 'run KO TP53')")
    parser.add_argument("--condition", default="Control", choices=["Control", "IFNγ", "Co-Culture"],
                        help="Condition to use (default: Control)")
    parser.add_argument("--skip-integration-check", action="store_true",
                        help="Skip Agent_Tools integration check")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip query validation test")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("REAL WORKFLOW TEST SUITE")
    print("="*70)
    print("\nThis test suite tests the COMPLETE workflow with ACTUAL Agent_Tools.")
    print("Warning: This will run real inference pipelines which may take 20-40 minutes per perturbation.")
    print("For faster testing, use test_orchestrator.py with mocked Agent_Tools.\n")
    
    # Test 1: Integration check
    if not args.skip_integration_check:
        if not test_agent_tools_integration():
            print("\n❌ Agent_Tools integration check failed. Cannot proceed with real workflow test.")
            print("   Make sure Agent_Tools is properly set up.")
            sys.exit(1)
    
    # Test 2: Query validation
    if not args.skip_validation:
        if not test_query_validation():
            print("\n⚠️  Query validation test failed.")
    
    # Test 3: Real workflow (if query provided)
    if args.query:
        print("\n⚠️  WARNING: Running REAL workflow with Agent_Tools.")
        print("   This will execute actual inference pipelines and may take 20-40 minutes.")
        print("   To cancel, press Ctrl+C within the next 5 seconds...\n")
        
        try:
            import time
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user.")
            sys.exit(0)
        
        results = test_real_workflow(args.query, condition=args.condition)
        
        if results:
            print("\n✓ Real workflow test completed!")
            sys.exit(0)
        else:
            print("\n❌ Real workflow test failed!")
            sys.exit(1)
    else:
        print("\n" + "="*70)
        print("INTEGRATION AND VALIDATION TESTS COMPLETED")
        print("="*70)
        print("\nTo test the full workflow, provide a query:")
        print("  python llm/test_real_workflow.py \"run KO TP53\"")
        print("\nOr use SLURM (recommended):")
        print("  sbatch llm/run_orchestrator_slurm.sh \"run KO TP53\"")


if __name__ == "__main__":
    main()

