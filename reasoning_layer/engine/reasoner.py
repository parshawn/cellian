"""Reasoner: executes LLM plans and assembles final results."""
from typing import Dict, Any, Optional


class Reasoner:
    """Main reasoner that executes plans and computes results."""
    
    def __init__(self, registry, planner_module, test_mode: bool = False, test_data: Optional[Dict] = None):
        self.registry = registry
        self.planner = planner_module
        self.test_mode = test_mode
        self.test_data = test_data or {}
    
    def run(self, query: str, sample: dict) -> dict:
        """
        Run the reasoning pipeline.
        
        Args:
            query: User query string
            sample: Sample data dict
        
        Returns:
            Final JSON result dict
        """
        # Get plan from LLM
        plan_json = self.planner.plan(query, sample, self.registry.specs())
        
        plan = plan_json.get("plan", [])
        tool_calls_spec = plan_json.get("tool_calls", [])
        rationale = plan_json.get("rationale", "")
        
        # Execute tools in order
        memory: Dict[str, Any] = {}
        executed_tool_calls = []
        
        for tool_spec in tool_calls_spec:
            tool_name = tool_spec.get("name")
            tool_args = tool_spec.get("args", {})
            
            try:
                tool = self.registry.get(tool_name)
                
                # Special handling for tool execution order
                if tool_name == "perturbation.get_embedding":
                    # Inject test embedding if in test mode
                    if self.test_mode and "embedding" in self.test_data:
                        # Convert numpy array to list for JSON compatibility
                        test_emb = self.test_data["embedding"]
                        if hasattr(test_emb, 'tolist'):
                            tool_args["embedding"] = test_emb.tolist()
                        elif isinstance(test_emb, (list, tuple)):
                            tool_args["embedding"] = list(test_emb)
                        else:
                            tool_args["embedding"] = test_emb
                    result = tool(tool_args)
                    # Store embedding in memory for later use
                    # Extract embedding from result (could be in different formats)
                    if "embedding" in result:
                        memory["embedding"] = result["embedding"]
                    elif "embedding_dict" in result:
                        # Handle embedding_dict format
                        emb_dict = result["embedding_dict"]
                        if "values" in emb_dict:
                            memory["embedding"] = emb_dict["values"]
                
                elif tool_name == "kg.find_path":
                    result = tool(tool_args)
                    memory["path_edges"] = result.get("paths", [])[0] if result.get("paths") else []
                
                elif tool_name == "state.predict":
                    # Inject test data if in test mode
                    if self.test_mode:
                        # Add test genes from sample if available
                        if "test_genes" in self.test_data:
                            tool_args["test_genes"] = self.test_data["test_genes"]
                        # Ensure embedding is passed if available (from memory or test_data)
                        if "embedding" not in tool_args:
                            # First try to get embedding from memory (from perturbation.get_embedding)
                            if "embedding" in memory:
                                tool_args["embedding"] = memory["embedding"]
                            # Otherwise use test_data embedding
                            elif "embedding" in self.test_data:
                                tool_args["embedding"] = self.test_data["embedding"]
                    else:
                        # Normal mode: get embedding from memory if available
                        if "embedding" not in tool_args and "embedding" in memory:
                            tool_args["embedding"] = memory["embedding"]
                    result = tool(tool_args)
                    memory["pred_rna"] = result.get("delta_rna", {})
                
                elif tool_name == "captain.translate":
                    # Force args to use memory["pred_rna"] + sample["protein"]["panel"]
                    forced_args = {
                        "delta_rna": memory.get("pred_rna", {}),
                        "panel": sample.get("protein", {}).get("panel", [])
                    }
                    result = tool(forced_args)
                    memory["pred_prot"] = result.get("delta_protein", {})
                
                elif tool_name == "validate.all":
                    # Compose args from memory + sample obs
                    validate_args = {
                        "pred_rna": memory.get("pred_rna", {}),
                        "obs_rna": sample.get("rna", {}).get("obs_delta", {}),
                        "pred_prot": memory.get("pred_prot", {}),
                        "obs_prot": sample.get("protein", {}).get("obs_delta", {}),
                        "path_edges": memory.get("path_edges", [])
                    }
                    result = tool(validate_args)
                    memory["metrics"] = result
                
                else:
                    result = tool(tool_args)
                
                executed_tool_calls.append({
                    "name": tool_name,
                    "args": tool_args,
                    "status": "ok",
                    "time_ms": 10  # Placeholder
                })
            
            except Exception as e:
                executed_tool_calls.append({
                    "name": tool_name,
                    "args": tool_args,
                    "status": "error",
                    "time_ms": 5
                })
        
        # Build arrays
        pred_rna = memory.get("pred_rna", {})
        obs_rna = sample.get("rna", {}).get("obs_delta", {})
        
        # RNA arrays: first 5 genes from pred_rna
        rna_genes_list = list(pred_rna.keys())[:5]
        rna_pred_vals = [pred_rna.get(g, 0.0) for g in rna_genes_list]
        rna_obs_vals = [obs_rna.get(g, 0.0) for g in rna_genes_list]
        
        # Protein arrays: all markers in predicted set
        pred_prot = memory.get("pred_prot", {})
        obs_prot = sample.get("protein", {}).get("obs_delta", {})
        prot_markers_list = sorted(pred_prot.keys())
        prot_pred_vals = [pred_prot.get(m, 0.0) for m in prot_markers_list]
        prot_obs_vals = [obs_prot.get(m, 0.0) for m in prot_markers_list]
        
        # Get metrics
        metrics = memory.get("metrics", {})
        
        # Build citations from path edges
        path_edges = memory.get("path_edges", [])
        citations = []
        for edge in path_edges:
            citations.append({
                "edge": f"{edge.get('src')}→{edge.get('dst')}",
                "sign": edge.get("sign", "+")
            })
        
        # Assemble final JSON
        final_json = {
            "plan": plan,
            "tool_calls": executed_tool_calls,
            "rationale": rationale,
            "metrics": {
                "RNA_spearman": metrics.get("rna_spearman", float("nan")),
                "Protein_pearson": metrics.get("prot_pearson", float("nan")),
                "Protein_mse": metrics.get("prot_mse", float("nan")),
                "Edge_sign_accuracy": metrics.get("edge_sign_accuracy", float("nan"))
            },
            "arrays": {
                "rna": {
                    "genes": rna_genes_list,
                    "pred": rna_pred_vals,
                    "obs": rna_obs_vals
                },
                "protein": {
                    "markers": prot_markers_list,
                    "pred": prot_pred_vals,
                    "obs": prot_obs_vals
                }
            },
            "citations": citations
        }
        
        return final_json

