"""CLI entrypoint for the reasoning engine demo."""
import json
import math
import os
import sys
import argparse
from pathlib import Path

# Load .env file if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

from engine.registry import ToolRegistry
from engine import tools_kg
from engine import tools_state
from engine import tools_captain
from engine import tools_validate
from engine import tools_perturbation
from engine import planner
from engine.reasoner import Reasoner
from engine.test_data_generator import (
    generate_test_sample,
    create_test_sample_from_args,
    load_test_sample_from_json,
    save_test_sample_to_json
)
from engine.test_tools import test_perturbation_embedding, test_state_predict
import numpy as np


def convert_nan_to_none(obj):
    """Recursively convert NaN values to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def print_results_summary(result):
    """Print a human-readable summary of results."""
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    
    # Plan and rationale
    if "rationale" in result:
        print(f"\n📋 Rationale:")
        print(f"   {result['rationale']}")
    
    # Metrics
    if "metrics" in result:
        metrics = result["metrics"]
        print(f"\n📈 Validation Metrics:")
        
        rna_spearman = metrics.get("RNA_spearman")
        if rna_spearman is not None and not (isinstance(rna_spearman, float) and math.isnan(rna_spearman)):
            print(f"   RNA Spearman:        {rna_spearman:.3f}")
        
        prot_pearson = metrics.get("Protein_pearson")
        if prot_pearson is not None and not (isinstance(prot_pearson, float) and math.isnan(prot_pearson)):
            print(f"   Protein Pearson:     {prot_pearson:.3f}")
        
        prot_mse = metrics.get("Protein_mse")
        if prot_mse is not None and not (isinstance(prot_mse, float) and math.isnan(prot_mse)):
            print(f"   Protein MSE:         {prot_mse:.3f}")
        
        edge_acc = metrics.get("Edge_sign_accuracy")
        if edge_acc is not None and not (isinstance(edge_acc, float) and math.isnan(edge_acc)):
            print(f"   Edge-Sign Accuracy:  {edge_acc:.3f}")
    
    # Arrays
    if "arrays" in result:
        arrays = result["arrays"]
        print(f"\n📊 Predictions vs Observations:")
        
        if "rna" in arrays:
            rna = arrays["rna"]
            genes = rna.get("genes", [])
            pred_vals = rna.get("pred", [])
            obs_vals = rna.get("obs", [])
            
            if genes:
                print(f"   RNA (first {len(genes)} genes):")
                for i, gene in enumerate(genes):
                    pred = pred_vals[i] if i < len(pred_vals) else 0.0
                    obs = obs_vals[i] if i < len(obs_vals) else 0.0
                    diff = abs(pred - obs)
                    print(f"     {gene:10s}: pred={pred:+.3f}, obs={obs:+.3f}, diff={diff:.3f}")
        
        if "protein" in arrays:
            protein = arrays["protein"]
            markers = protein.get("markers", [])
            pred_vals = protein.get("pred", [])
            obs_vals = protein.get("obs", [])
            
            if markers:
                print(f"   Protein ({len(markers)} markers):")
                for i, marker in enumerate(markers):
                    pred = pred_vals[i] if i < len(pred_vals) else 0.0
                    obs = obs_vals[i] if i < len(obs_vals) else 0.0
                    diff = abs(pred - obs)
                    print(f"     {marker:10s}: pred={pred:+.3f}, obs={obs:+.3f}, diff={diff:.3f}")
    
    # Citations
    if "citations" in result and result["citations"]:
        citations = result["citations"]
        print(f"\n🔗 Knowledge Graph Citations ({len(citations)} edges):")
        for citation in citations[:5]:  # Show first 5
            edge = citation.get("edge", "")
            sign = citation.get("sign", "+")
            print(f"     {edge} [{sign}]")
        if len(citations) > 5:
            print(f"     ... and {len(citations) - 5} more")
    
    # Tool calls
    if "tool_calls" in result:
        tool_calls = result["tool_calls"]
        print(f"\n🔧 Tool Execution:")
        for tool_call in tool_calls:
            name = tool_call.get("name", "unknown")
            status = tool_call.get("status", "unknown")
            status_icon = "✓" if status == "ok" else "✗"
            print(f"     {status_icon} {name}")
    
    # LLM Interpretation
    if "llm_interpretation" in result:
        llm_interp = result["llm_interpretation"]
        if llm_interp:
            print(f"\n🤖 LLM Interpretation:")
            # Print first few lines of interpretation
            lines = llm_interp.split('\n')
            for line in lines[:10]:  # Show first 10 lines
                if line.strip():
                    print(f"   {line}")
            if len(lines) > 10:
                print(f"   ... ({len(lines) - 10} more lines)")
    
    # Graph info
    if "graph" in result:
        graph = result["graph"]
        nodes_count = len(graph.get("nodes", []))
        edges_count = len(graph.get("edges", []))
        print(f"\n📊 Causal Graph:")
        print(f"   Nodes: {nodes_count}, Edges: {edges_count}")
        if nodes_count > 0:
            print(f"   Graph type: {'Dummy graph' if any(n.get('metadata', {}).get('dummy') for n in graph.get('nodes', [])) else 'Pre-computed graph'}")
        if "graph_filepath" in result:
            print(f"   Saved to: {result['graph_filepath']}")
        if "graph_viz_filepath" in result:
            print(f"   Visualization: {result['graph_viz_filepath']}")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """Run the demo."""
    parser = argparse.ArgumentParser(
        description="Run the reasoning engine demo with test mode support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode with default arbitrary data
  python run_demo.py --test

  # Test mode with custom perturbation and genes
  python run_demo.py --test --perturbation JAK1_KO --genes "STAT1,IRF1,HLA-A" --proteins "HLA-A,CD58"

  # Test mode with perfect match (for testing metrics)
  python run_demo.py --test --perfect-match

  # Test mode with custom seed for reproducibility
  python run_demo.py --test --seed 42

  # Test mode with custom embedding dimension
  python run_demo.py --test --embedding-dim 2048

  # Load test sample from JSON file
  python run_demo.py --test --sample-file test_sample.json

  # Save generated test sample to file
  python run_demo.py --test --save-sample test_sample.json

  # Normal mode (requires real data)
  python run_demo.py --query "What happens if I knock out JAK1?"
        """
    )
    
    # Test mode arguments
    parser.add_argument("--test", action="store_true",
                       help="Enable test mode (use arbitrary data, no DataLoader required)")
    parser.add_argument("--perturbation", type=str, default="JAK1_KO",
                       help="Perturbation name (e.g., JAK1_KO, STAT1_KD)")
    parser.add_argument("--genes", type=str,
                       help="Comma-separated list of genes (e.g., 'STAT1,IRF1,HLA-A')")
    parser.add_argument("--proteins", type=str,
                       help="Comma-separated list of proteins (e.g., 'HLA-A,CD58')")
    parser.add_argument("--embedding-dim", type=int, default=1024,
                       help="Embedding dimension (default: 1024)")
    parser.add_argument("--seed", type=int,
                       help="Random seed for reproducibility")
    parser.add_argument("--perfect-match", action="store_true",
                       help="Make predicted and observed data match perfectly (for testing)")
    parser.add_argument("--sample-file", type=str,
                       help="Load test sample from JSON file")
    parser.add_argument("--save-sample", type=str,
                       help="Save generated test sample to JSON file")
    parser.add_argument("--dummy-graph", action="store_true",
                       help="Create and include dummy graph in output")
    
    # Normal mode arguments
    parser.add_argument("--query", type=str, default="What happens if I knock out JAK1?",
                       help="Query string for LLM planner")
    parser.add_argument("--output-file", type=str,
                       help="Save results to JSON file")
    parser.add_argument("--no-summary", action="store_true",
                       help="Don't print human-readable summary")
    parser.add_argument("--json-only", action="store_true",
                       help="Only output JSON (no summary, useful for scripts)")
    
    args = parser.parse_args()
    
    # Check for API key (required for LLM planner)
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please set it in .env file or export it:", file=sys.stderr)
        print("  Option 1: Create .env file with: GEMINI_API_KEY=your_key_here", file=sys.stderr)
        print("  Option 2: export GEMINI_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)
    
    # Build registry and register tools
    registry = ToolRegistry()
    
    # Import pathway tools
    from engine import tools_pathway
    
    if args.test:
        # Test mode: use test tools that don't require DataLoader
        registry.register(tools_kg.kg_find_path)
        registry.register(tools_pathway.pathway_find_affected)
        registry.register(tools_pathway.pathway_get_genes)
        registry.register(tools_pathway.pathway_traverse)
        registry.register(test_state_predict)  # Test-mode STATE tool
        registry.register(tools_captain.captain_translate)
        registry.register(tools_validate.validate_all)
        registry.register(test_perturbation_embedding)  # Test-mode perturbation tool
    else:
        # Normal mode: use real tools
        registry.register(tools_kg.kg_find_path)
        registry.register(tools_pathway.pathway_find_affected)
        registry.register(tools_pathway.pathway_get_genes)
        registry.register(tools_pathway.pathway_traverse)
        registry.register(tools_state.state_predict)
        registry.register(tools_captain.captain_translate)
        registry.register(tools_validate.validate_all)
        registry.register(tools_perturbation.perturbation_embedding)
    
    if args.test:
        # Test mode: generate or load test sample
        if args.sample_file:
            print(f"Loading test sample from: {args.sample_file}", file=sys.stderr)
            sample = load_test_sample_from_json(args.sample_file)
        else:
            print("Generating test sample with arbitrary data...", file=sys.stderr)
            sample = create_test_sample_from_args(
                perturbation=args.perturbation,
                genes=args.genes,
                proteins=args.proteins,
                embedding_dim=args.embedding_dim,
                seed=args.seed,
                perfect_match=args.perfect_match
            )
            print(f"  Perturbation: {sample['perturbation']['target']} ({sample['perturbation']['type']})", file=sys.stderr)
            print(f"  Genes: {len(sample['rna']['obs_delta'])}", file=sys.stderr)
            print(f"  Proteins: {len(sample['protein']['panel'])}", file=sys.stderr)
            print(f"  Embedding dim: {sample['metadata']['embedding_dim']}", file=sys.stderr)
        
        # Save sample if requested
        if args.save_sample:
            save_test_sample_to_json(sample, args.save_sample)
            print(f"Test sample saved to: {args.save_sample}", file=sys.stderr)
        
        # Prepare test data for Reasoner
        test_embedding = sample.get("embedding")
        if test_embedding is not None:
            # Convert list to numpy array if needed
            if isinstance(test_embedding, list):
                test_embedding = np.array(test_embedding)
        
        # Prepare test data dict for Reasoner
        test_data = {
            "embedding": test_embedding,
            "test_genes": list(sample["rna"]["obs_delta"].keys()) if "rna" in sample and "obs_delta" in sample["rna"] else None
        }
        
        # Create reasoner in test mode
        reasoner = Reasoner(registry, planner, test_mode=True, test_data=test_data, 
                           use_dummy_graph=args.dummy_graph)
        
    else:
        # Normal mode: use hardcoded sample (requires real data)
        sample = {
            "context": {"cell_line": "A375", "condition": "IFNg+"},
            "perturbation": {"target": "JAK1", "type": "KO"},
            "rna": {
                "obs_delta": {
                    "STAT1": -0.7,
                    "IRF1": -0.5,
                    "CXCL10": -0.4,
                    "HLA-A": -0.6,
                    "B2M": -0.5,
                    "SOCS1": 0.2,
                    "IFIT1": -0.3
                }
            },
            "protein": {
                "panel": ["HLA-A", "CD58"],
                "obs_delta": {"HLA-A": -0.6, "CD58": -0.4}
            }
        }
        reasoner = Reasoner(registry, planner, use_dummy_graph=args.dummy_graph)
    
    # Run reasoner
    query = args.query
    if args.test:
        # Extract perturbation name for query if using default
        default_query = "What happens if I knock out JAK1?"
        if query == default_query or (not args.query and query == parser.get_default("query")):
            pert_name = sample["perturbation"]["target"]
            pert_type = sample["perturbation"]["type"]
            # Convert perturbation type to action verb
            type_to_verb = {
                "KO": "knock out",
                "KD": "knock down", 
                "OE": "overexpress"
            }
            verb = type_to_verb.get(pert_type.upper(), "perturb")
            query = f"What happens if I {verb} {pert_name}?"
    
    print(f"Processing query: {query}", file=sys.stderr)
    try:
        result = reasoner.run(query, sample)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print results
    result_clean = convert_nan_to_none(result)
    
    if not args.json_only:
        if not args.no_summary:
            print_results_summary(result_clean)
        else:
            print(json.dumps(result_clean, indent=2))
    else:
        # JSON only mode (for scripts/SLURM)
        print(json.dumps(result_clean, indent=2))
    
    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(result_clean, f, indent=2)
        print(f"Results saved to: {args.output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()

