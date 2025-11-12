"""CLI entrypoint for HypothesisAgent demo."""
import json
import math
import os
import sys

from engine.registry import ToolRegistry
from engine import tools_kg
from engine import tools_state
from engine import tools_captain
from engine import tools_validate
from engine import tools_perturbation
from engine.hypothesis_agent import HypothesisAgent
from engine.data_loader import DataLoader


def convert_nan_to_none(obj):
    """Recursively convert NaN values to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def main():
    """Run the HypothesisAgent demo."""
    # Build registry and register tools
    registry = ToolRegistry()
    registry.register(tools_kg.kg_find_path)
    registry.register(tools_state.state_predict)
    registry.register(tools_captain.captain_translate)
    registry.register(tools_validate.validate_all)
    registry.register(tools_perturbation.perturbation_embedding)
    
    # Initialize data loader
    try:
        data_loader = DataLoader()
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        print("Make sure data files are available at:", file=sys.stderr)
        print("  /home/nebius/cellian/data/perturb-cite-seq/SCP1064/", file=sys.stderr)
        sys.exit(1)
    
    # Check for dummy graph flag
    use_dummy = "--dummy" in sys.argv or "-d" in sys.argv
    if use_dummy:
        sys.argv = [arg for arg in sys.argv if arg not in ["--dummy", "-d"]]
        print("Using dummy graph for testing", file=sys.stderr)
    
    # Instantiate HypothesisAgent
    agent = HypothesisAgent(registry, data_loader, use_dummy_graph=use_dummy)
    
    # Get perturbation name from command line or use default
    if len(sys.argv) > 1:
        perturbation_name = sys.argv[1]
    else:
        # Try to find a non-control perturbation
        import pandas as pd
        perturbations = data_loader.rna_meta["sgRNA"].fillna("CTRL").unique()
        non_ctrl = [p for p in perturbations if p != "CTRL" and not pd.isna(p)]
        if non_ctrl:
            perturbation_name = non_ctrl[0]
            print(f"Using perturbation: {perturbation_name}", file=sys.stderr)
        else:
            print("Error: No perturbations found in data", file=sys.stderr)
            print("Usage: python run_hypothesis_agent.py <perturbation_name>", file=sys.stderr)
            sys.exit(1)
    
    # Generate hypothesis
    print(f"Generating hypothesis for perturbation: {perturbation_name}", file=sys.stderr)
    result = agent.generate_hypothesis(perturbation_name)
    
    # Print JSON (convert NaN to None for JSON compatibility)
    result_clean = convert_nan_to_none(result)
    print(json.dumps(result_clean, indent=2))


if __name__ == "__main__":
    main()

