"""LLM-powered agent that handles natural language queries and generates interpretations."""
import os
from typing import Dict, Any, Optional
from .hypothesis_agent import HypothesisAgent
from .registry import ToolRegistry
from .data_loader import DataLoader
from .llm_interpreter import interpret_results, extract_perturbation_info
from .tools_perturbation import get_data_loader


class LLMAgent:
    """
    LLM-powered agent that:
    1. Takes natural language queries
    2. Extracts perturbation information
    3. Executes HypothesisAgent
    4. Generates LLM interpretation of results
    """
    
    def __init__(self, registry: ToolRegistry, data_loader: Optional[DataLoader] = None,
                 use_dummy_graph: bool = False):
        """
        Initialize LLMAgent.
        
        Args:
            registry: Tool registry with available tools
            data_loader: Data loader instance (optional)
            use_dummy_graph: If True, use dummy graph for testing
        """
        self.registry = registry
        self.data_loader = data_loader or get_data_loader()
        self.hypothesis_agent = HypothesisAgent(registry, data_loader, use_dummy_graph=use_dummy_graph)
    
    def answer_query(self, query: str, perturbation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer a natural language query about a perturbation.
        
        Args:
            query: Natural language query (e.g., "What happens if I knock down JAK1?")
            perturbation_name: Optional explicit perturbation name (overrides extraction)
        
        Returns:
            Dictionary with results and LLM interpretation
        """
        # Step 1: Extract perturbation information from query
        if perturbation_name:
            # Use explicit perturbation name
            pert_info = {
                "target": perturbation_name.split("_")[0],
                "type": "unknown",
                "confidence": 1.0
            }
            pert_name = perturbation_name
        else:
            # Extract from query
            pert_info = extract_perturbation_info(query)
            target = pert_info.get("target", "UNKNOWN")
            pert_type = pert_info.get("type", "unknown")
            
            # Construct perturbation name
            if pert_type != "unknown":
                pert_name = f"{target}_{pert_type}"
            else:
                pert_name = target
        
        # Step 2: Generate hypothesis using HypothesisAgent
        # Try multiple perturbation name formats if needed
        results = None
        tried_names = []
        
        for candidate_name in [pert_name, pert_info.get("target"), pert_name.upper(), pert_info.get("target", "").upper()]:
            if not candidate_name or candidate_name == "UNKNOWN" or candidate_name in tried_names:
                continue
            tried_names.append(candidate_name)
            
            try:
                results = self.hypothesis_agent.generate_hypothesis(candidate_name)
                pert_name = candidate_name  # Update to successful name
                break
            except Exception as e:
                continue
        
        if results is None:
            # If all attempts failed, create error response
            return {
                "query": query,
                "perturbation_info": pert_info,
                "perturbation_name": pert_name,
                "error": f"Could not find perturbation '{pert_name}' in data. Tried: {tried_names}",
                "llm_interpretation": f"Error: Could not process query about '{pert_info.get('target', 'unknown')}'. Please check if the perturbation exists in the data."
            }
        
        # Step 3: Generate LLM interpretation
        llm_interpretation = interpret_results(results, query)
        
        # Step 4: Combine results with LLM interpretation
        final_results = {
            "query": query,
            "perturbation_info": pert_info,
            "perturbation_name": pert_name,
            "llm_interpretation": llm_interpretation,
            **results  # Include all HypothesisAgent results
        }
        
        return final_results

