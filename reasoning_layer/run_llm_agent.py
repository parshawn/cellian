"""CLI entrypoint for LLM-powered agent with natural language queries."""
import json
import math
import os
import sys
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
from engine.llm_agent import LLMAgent
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
    """Run the LLM agent with natural language queries."""
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please set it in .env file or export it:", file=sys.stderr)
        print("  Option 1: Create .env file with: GEMINI_API_KEY=your_key_here", file=sys.stderr)
        print("  Option 2: export GEMINI_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)
    
    # Build registry and register tools
    from engine import tools_pathway
    
    registry = ToolRegistry()
    registry.register(tools_kg.kg_find_path)
    registry.register(tools_pathway.pathway_find_affected)
    registry.register(tools_pathway.pathway_get_genes)
    registry.register(tools_pathway.pathway_traverse)
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
    
    # Instantiate LLM agent
    llm_agent = LLMAgent(registry, data_loader, use_dummy_graph=use_dummy)
    
    # Get query from command line or use default
    if len(sys.argv) > 1:
        # Join all arguments as query (in case query has spaces)
        query = " ".join(sys.argv[1:])
    else:
        # Default query
        query = "What happens if I knock down JAK1?"
        print(f"Using default query: {query}", file=sys.stderr)
        print("Usage: python run_llm_agent.py 'your question here'", file=sys.stderr)
        print("Example: python run_llm_agent.py 'What happens if I knock out STAT1?'", file=sys.stderr)
    
    # Answer query
    print(f"Processing query: {query}", file=sys.stderr)
    result = llm_agent.answer_query(query)
    
    # Print JSON (convert NaN to None for JSON compatibility)
    result_clean = convert_nan_to_none(result)
    print(json.dumps(result_clean, indent=2))


if __name__ == "__main__":
    main()

