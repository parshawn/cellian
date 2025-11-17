"""Reasoner: executes LLM plans and assembles final results."""
import math
from typing import Dict, Any, Optional
from .causal_graph import CausalGraph
from .llm_interpreter import interpret_results, extract_perturbation_info
from .graph_loader import save_graph, get_graph_path


class Reasoner:
    """Main reasoner that executes plans and computes results."""
    
    def __init__(self, registry, planner_module, test_mode: bool = False, test_data: Optional[Dict] = None, 
                 use_dummy_graph: bool = False):
        self.registry = registry
        self.planner = planner_module
        self.test_mode = test_mode
        self.test_data = test_data or {}
        self.use_dummy_graph = use_dummy_graph
    
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
                
                elif tool_name == "pathway.find_affected":
                    result = tool(tool_args)
                    memory["affected_pathways"] = result.get("pathway_ids", [])
                    memory["pathway_names"] = result.get("pathway_names", [])
                
                elif tool_name == "pathway.get_genes":
                    # Use pathway_ids from memory if not provided in args
                    if "pathway_ids" not in tool_args and "affected_pathways" in memory:
                        tool_args["pathway_ids"] = memory["affected_pathways"]
                    result = tool(tool_args)
                    memory["pathway_genes"] = result.get("genes", [])
                    # Store pathway genes for use in state.predict
                    if "pathway_genes" in memory:
                        memory["target_genes"] = memory["pathway_genes"]
                
                elif tool_name == "pathway.traverse":
                    result = tool(tool_args)
                    memory["traversed_genes"] = result.get("genes", [])
                    memory["traversal_paths"] = result.get("paths", [])
                    # Store traversed genes for use in state.predict
                    if "traversed_genes" in memory:
                        memory["target_genes"] = memory["traversed_genes"]
                
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
                    
                    # If pathway genes are available, pass them to state.predict
                    # (Note: state.predict tool may need to be updated to handle target_genes)
                    if "target_genes" in memory and "target_genes" not in tool_args:
                        tool_args["target_genes"] = memory["target_genes"]
                    
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
                
                # Store tool call with results
                # Use a copy of tool_args to show what was actually used (after memory injection)
                tool_call_record = {
                    "name": tool_name,
                    "args": tool_args.copy() if isinstance(tool_args, dict) else tool_args,
                    "status": "ok",
                    "time_ms": 10  # Placeholder
                }
                
                # Include key results for pathway tools
                if tool_name == "pathway.find_affected" and result:
                    tool_call_record["result"] = {
                        "pathway_ids": result.get("pathway_ids", []),
                        "pathway_names": result.get("pathway_names", []),
                        "n_pathways": result.get("n_pathways", 0)
                    }
                elif tool_name == "pathway.get_genes" and result:
                    tool_call_record["result"] = {
                        "genes": result.get("genes", [])[:20],  # First 20 genes
                        "n_genes": result.get("n_genes", 0)
                    }
                elif tool_name == "pathway.traverse" and result:
                    tool_call_record["result"] = {
                        "genes": result.get("genes", [])[:20],  # First 20 genes
                        "n_genes": result.get("n_genes", 0)
                    }
                
                executed_tool_calls.append(tool_call_record)
            
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
        
        # Extract perturbation info from query
        perturbation_info = extract_perturbation_info(query)
        perturbation_name = perturbation_info.get("target", sample.get("perturbation", {}).get("target", "unknown"))
        
        # Create graph with pathway information if available
        graph = None
        graph_dict = None
        graph_filepath = None
        graph_viz_filepath = None
        
        if self.use_dummy_graph:
            # Use PathwayGraph to include pathway relationships
            from .pathway_graph import PathwayGraph
            graph = PathwayGraph(load_pathway_data=True)
            
            # Add perturbation node
            pert_node_id = graph.add_node(
                name=f"perturbation_{perturbation_name}",
                node_type="perturbation",
                data={"perturbation_name": perturbation_name},
                metadata={}
            )
            
            # Add RNA prediction node with pathway genes if available
            rna_node_id = graph.add_node(
                name="rna_prediction",
                node_type="rna",
                data={
                    "delta_rna": pred_rna,
                    "pathway_genes": memory.get("pathway_genes", [])
                },
                metadata={
                    "n_genes": len(pred_rna),
                    "n_pathway_genes": len(memory.get("pathway_genes", []))
                }
            )
            
            # Add protein prediction node
            prot_node_id = graph.add_node(
                name="protein_prediction",
                node_type="protein",
                data={"delta_protein": pred_prot},
                metadata={"n_proteins": len(pred_prot)}
            )
            
            # Add standard edges
            graph.add_edge(pert_node_id, rna_node_id, "embeds_to", metadata={"transformation": "perturbation_to_rna"})
            graph.add_edge(rna_node_id, prot_node_id, "translates_to", metadata={"transformation": "rna_to_protein"})
            
            # Add pathway relationships if available
            affected_pathways = memory.get("affected_pathways", [])
            pathway_genes = memory.get("pathway_genes", [])
            
            if affected_pathways:
                # Add pathway nodes and connect to genes
                from .pathway_loader import PathwayLoader
                loader = PathwayLoader()
                
                for pathway_id in affected_pathways:
                    pathway_data = loader.get_pathway(pathway_id)
                    if pathway_data:
                        pathway_name = pathway_data.get("name", pathway_id)
                        pathway_node = graph.get_node_by_name(pathway_name)
                        
                        if not pathway_node:
                            pathway_node_id = graph.add_node(
                                name=pathway_name,
                                node_type="pathway",
                                data={
                                    "pathway_id": pathway_id,
                                    "genes": pathway_data.get("genes", [])
                                },
                                metadata={"source": "KEGG"}
                            )
                        else:
                            pathway_node_id = pathway_node.node_id
                        
                        # Connect perturbation to pathway
                        graph.add_edge(
                            pert_node_id, 
                            pathway_node_id, 
                            "affects_pathway",
                            metadata={"pathway_id": pathway_id}
                        )
                        
                        # Connect pathway to affected genes (if they have predictions)
                        for gene in pathway_genes:
                            if gene in pred_rna:
                                gene_node = graph.get_node_by_name(gene)
                                if not gene_node:
                                    gene_node_id = graph.add_node(
                                        name=gene,
                                        node_type="gene",
                                        data={
                                            "gene_name": gene,
                                            "delta_rna": pred_rna.get(gene, 0.0)
                                        },
                                        metadata={}
                                    )
                                else:
                                    gene_node_id = gene_node.node_id
                                
                                # Connect pathway to gene
                                graph.add_edge(
                                    pathway_node_id,
                                    gene_node_id,
                                    "pathway_member",
                                    metadata={"pathway_id": pathway_id}
                                )
                                
                                # Connect gene to RNA prediction
                                graph.add_edge(
                                    gene_node_id,
                                    rna_node_id,
                                    "contributes_to",
                                    metadata={"delta": pred_rna.get(gene, 0.0)}
                                )
            
            graph_dict = graph.to_dict()
            # Save graph to file
            graph_filepath = get_graph_path(perturbation_name)
            save_graph(graph, graph_filepath)
            # Create visualization
            import os
            graph_viz_filepath = graph_filepath.replace('.json', '_viz.png')
            graph.visualize(graph_viz_filepath, show_changes=True)
        
        # Prepare results for LLM interpretation (adapt Reasoner format to HypothesisAgent-like format)
        # Convert NaN to None for JSON compatibility
        rna_spearman = metrics.get("rna_spearman", float("nan"))
        prot_pearson = metrics.get("prot_pearson", float("nan"))
        prot_mse = metrics.get("prot_mse", float("nan"))
        
        results_for_llm = {
            "hypothesis": rationale,
            "perturbation_name": perturbation_name,
            "validation_scores": {
                "rna": {
                    "spearman": None if (isinstance(rna_spearman, float) and math.isnan(rna_spearman)) else rna_spearman
                },
                "protein": {
                    "pearson": None if (isinstance(prot_pearson, float) and math.isnan(prot_pearson)) else prot_pearson,
                    "mse": None if (isinstance(prot_mse, float) and math.isnan(prot_mse)) else prot_mse
                }
            },
            "predictions": {
                "rna": {
                    "delta": pred_rna
                },
                "protein": {
                    "delta": pred_prot
                }
            },
            "path": path_edges
        }
        
        # Generate LLM interpretation
        llm_interpretation = interpret_results(results_for_llm, query)
        
        # Assemble final JSON
        final_json = {
            "query": query,
            "perturbation_info": perturbation_info,
            "perturbation_name": perturbation_name,
            "plan": plan,
            "tool_calls": executed_tool_calls,
            "rationale": rationale,
            "llm_interpretation": llm_interpretation,
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
        
        # Add graph if created
        if graph_dict is not None:
            final_json["graph"] = graph_dict
            final_json["graph_filepath"] = graph_filepath
            if graph_viz_filepath:
                final_json["graph_viz_filepath"] = graph_viz_filepath
        
        return final_json

