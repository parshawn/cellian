"""HypothesisAgent: Autonomous agent that traverses the graph and generates hypotheses."""
import math
import time
import numpy as np
from typing import Dict, Any, Optional
from .data_loader import DataLoader
from .registry import ToolRegistry
from .causal_graph import CausalGraph
from .graph_loader import load_graph, get_graph_path
from .profile_comparison import compare_profiles, calculate_error_propagation
from .tools_perturbation import get_data_loader
from .embedding_utils import embedding_to_dict


class HypothesisAgent:
    """
    Autonomous agent that executes the full reasoning chain using pre-computed aligned embeddings.
    Assumes embeddings are already aligned and nodes are ready.
    
    Executes:
    1. Fetches control cell baseline
    2. Gets pre-computed aligned perturbation embedding (Node 1)
    3. Predicts RNA using aligned embedding (Node 2)
    4. Predicts protein from RNA prediction (Node 3)
    5. Compares predictions to ground truth
    6. Calculates error propagation
    7. Builds causal graph with nodes and edges
    """
    
    def __init__(self, registry: ToolRegistry, data_loader: Optional[DataLoader] = None,
                 graph: Optional[CausalGraph] = None, use_dummy_graph: bool = False):
        """
        Initialize HypothesisAgent.
        
        Args:
            registry: Tool registry with available tools
            data_loader: Data loader instance (optional, will create if not provided)
            graph: Pre-computed graph to use (optional)
            use_dummy_graph: If True, use dummy graph for testing
        """
        self.registry = registry
        self.data_loader = data_loader or get_data_loader()
        self.graph = graph if graph is not None else CausalGraph()
        self.use_dummy_graph = use_dummy_graph
    
    def generate_hypothesis(self, perturbation_name: str) -> Dict[str, Any]:
        """
        Generate hypothesis for a perturbation by executing the full reasoning chain.
        
        Args:
            perturbation_name: Name of perturbation (e.g., "JAK1", "JAK1_KO")
        
        Returns:
            Structured JSON with hypothesis, path, validation scores, and error propagation
        """
        start_time = time.time()
        
        # Try to load pre-computed graph, or create new one
        graph_pre_loaded = len(self.graph.nodes) > 0
        
        if not graph_pre_loaded:
            graph_path = get_graph_path(perturbation_name)
            self.graph = load_graph(
                filepath=graph_path,
                use_dummy=self.use_dummy_graph,
                perturbation_name=perturbation_name
            )
            graph_pre_loaded = len(self.graph.nodes) > 0
        
        # Step 1: Fetch control cell baseline
        control_rna = self.data_loader.get_control_rna_profile()
        control_prot = self.data_loader.get_control_protein_profile()
        
        # Step 2: Get perturbation embedding (Node 1)
        # If graph is pre-loaded, try to get existing node; otherwise create new one
        node1_start = time.time()
        node1_id = None
        pert_embedding = None
        
        if graph_pre_loaded:
            # Try to get existing perturbation node
            pert_node = self.graph.get_node_by_name(f"perturbation_{perturbation_name}")
            if pert_node:
                node1_id = pert_node.node_id
                pert_embedding = pert_node.embedding
                if isinstance(pert_embedding, list):
                    pert_embedding = np.array(pert_embedding)
            node1_time = (time.time() - node1_start) * 1000
        else:
            # Create new node
            try:
                pert_emb_tool = self.registry.get("perturbation.get_embedding")
                pert_emb_result = pert_emb_tool({"perturbation_name": perturbation_name})
                pert_embedding = pert_emb_result.get("embedding")
                
                # Add Node 1 to graph
                node1_id = self.graph.add_node(
                    name=f"perturbation_{perturbation_name}",
                    node_type="perturbation",
                    embedding=pert_embedding,
                    data={"perturbation_name": perturbation_name},
                    metadata={"embedding_dim": len(pert_embedding) if pert_embedding else 0}
                )
                node1_time = (time.time() - node1_start) * 1000
            except Exception as e:
                pert_embedding = None
                node1_id = None
                node1_time = (time.time() - node1_start) * 1000
        
        # Step 3: Predict RNA (Node 2)
        node2_start = time.time()
        node2_id = None
        pred_rna_delta = {}
        
        if graph_pre_loaded:
            # Try to get existing RNA node
            rna_node = self.graph.get_node_by_name("rna_prediction")
            if rna_node:
                node2_id = rna_node.node_id
                pred_rna_delta = rna_node.data.get("delta_rna", {})
            node2_time = (time.time() - node2_start) * 1000
        else:
            # Create new node
            try:
                state_tool = self.registry.get("state.predict")
                gene_name = perturbation_name.split("_")[0]
                state_result = state_tool({
                    "target": gene_name,
                    "context": {"cell_line": "A375", "condition": "IFNg+"},
                    "embedding": pert_embedding
                })
                pred_rna_delta = state_result.get("delta_rna", {})
                
                # Add Node 2 to graph
                node2_id = self.graph.add_node(
                    name="rna_prediction",
                    node_type="rna",
                    data={"delta_rna": pred_rna_delta},
                    metadata={"n_genes": len(pred_rna_delta)}
                )
                
                # Add edge from Node 1 to Node 2
                if node1_id is not None:
                    self.graph.add_edge(
                        source_id=node1_id,
                        target_id=node2_id,
                        edge_type="embeds_to",
                        metadata={"transformation": "perturbation_to_rna"}
                    )
                
                node2_time = (time.time() - node2_start) * 1000
            except Exception as e:
                pred_rna_delta = {}
                node2_id = None
                node2_time = (time.time() - node2_start) * 1000
        
        # Step 4: Predict protein (Node 3)
        node3_start = time.time()
        node3_id = None
        pred_prot_delta = {}
        protein_panel = []
        
        if graph_pre_loaded:
            # Try to get existing protein node
            prot_node = self.graph.get_node_by_name("protein_prediction")
            if prot_node:
                node3_id = prot_node.node_id
                pred_prot_delta = prot_node.data.get("delta_protein", {})
                protein_panel = list(pred_prot_delta.keys())
            node3_time = (time.time() - node3_start) * 1000
        else:
            # Create new node
            try:
                captain_tool = self.registry.get("captain.translate")
                real_prot = self.data_loader.get_real_protein_profile(perturbation_name)
                protein_panel = list(real_prot.keys()) if real_prot else []
                
                captain_result = captain_tool({
                    "delta_rna": pred_rna_delta,
                    "panel": protein_panel
                })
                pred_prot_delta = captain_result.get("delta_protein", {})
                
                # Add Node 3 to graph
                node3_id = self.graph.add_node(
                    name="protein_prediction",
                    node_type="protein",
                    data={"delta_protein": pred_prot_delta},
                    metadata={"n_proteins": len(pred_prot_delta)}
                )
                
                # Add edge from Node 2 to Node 3
                if node2_id is not None:
                    self.graph.add_edge(
                        source_id=node2_id,
                        target_id=node3_id,
                        edge_type="translates_to",
                        metadata={"transformation": "rna_to_protein"}
                    )
                
                node3_time = (time.time() - node3_start) * 1000
            except Exception as e:
                pred_prot_delta = {}
                node3_id = None
                node3_time = (time.time() - node3_start) * 1000
        
        # Step 5: Get real RNA and protein profiles
        real_rna_delta = self.data_loader.get_rna_delta(perturbation_name)
        real_prot_delta = self.data_loader.get_protein_delta(perturbation_name)
        
        # Step 6: Compare predictions to ground truth
        rna_comparison = compare_profiles(pred_rna_delta, real_rna_delta, metric="spearman")
        prot_comparison = compare_profiles(pred_prot_delta, real_prot_delta, metric="pearson")
        
        # compute cosine similarity for both RNA and protein
        rna_cosine = compare_profiles(pred_rna_delta, real_rna_delta, metric="cosine")
        prot_cosine = compare_profiles(pred_prot_delta, real_prot_delta, metric="cosine")
        
        # Step 6.5: Compute edge-sign accuracy (if KG edges available)
        edge_sign_accuracy = float("nan")
        kg_path_edges = []
        
        if pred_rna_delta:
            # Try to query KG for path edges
            try:
                kg_tool = self.registry.get("kg.find_path")
                if kg_tool:
                    # Extract gene name from perturbation (e.g., "JAK1_KO" -> "JAK1")
                    gene_name = perturbation_name.split("_")[0]
                    # Get predicted genes as targets
                    target_genes = list(pred_rna_delta.keys())
                    
                    kg_result = kg_tool({
                        "source": gene_name,
                        "targets": target_genes,
                        "max_hops": 3
                    })
                    
                    paths = kg_result.get("paths", [])
                    if paths:
                        # Flatten paths to get all edges
                        kg_path_edges = paths[0] if isinstance(paths[0], list) else []
                        
                        # Compute edge-sign accuracy
                        if kg_path_edges:
                            correct = 0
                            total = 0
                            for edge in kg_path_edges:
                                dst = edge.get("dst")
                                sign = edge.get("sign", "+")
                                if dst in pred_rna_delta:
                                    total += 1
                                    pred_val = pred_rna_delta[dst]
                                    # Check if predicted sign matches expected sign
                                    if sign == "+":
                                        if pred_val >= 0:
                                            correct += 1
                                    else:  # sign == "-"
                                        if pred_val < 0:
                                            correct += 1
                            
                            if total > 0:
                                edge_sign_accuracy = correct / total
            except Exception:
                # KG query failed or not available - skip edge-sign accuracy
                edge_sign_accuracy = float("nan")
                kg_path_edges = []
        
        # Step 7: Calculate error propagation
        rna_error = rna_comparison.get("mse", float("nan"))
        prot_error = prot_comparison.get("mse", float("nan"))
        
        # Calculate RNA->Protein translation error
        translation_error = max(0.0, prot_error - rna_error) if not (math.isnan(rna_error) or math.isnan(prot_error)) else 0.0
        
        error_prop = calculate_error_propagation(
            rna_error=rna_error if not math.isnan(rna_error) else 0.0,
            protein_error=prot_error if not math.isnan(prot_error) else 0.0,
            rna_to_protein_error=translation_error
        )
        
        # Build 3-node path
        path = []
        if node1_id:
            path.append({
                "node": 1,
                "node_id": node1_id,
                "name": "perturbation.get_embedding",
                "type": "perturbation",
                "input": {"perturbation_name": perturbation_name},
                "output": {"embedding_dim": len(pert_embedding) if pert_embedding else 0},
                "time_ms": int(node1_time)
            })
        if node2_id:
            path.append({
                "node": 2,
                "node_id": node2_id,
                "name": "state.predict",
                "type": "rna",
                "input": {"target": perturbation_name, "embedding_used": pert_embedding is not None},
                "output": {"n_genes": len(pred_rna_delta)},
                "time_ms": int(node2_time)
            })
        if node3_id:
            path.append({
                "node": 3,
                "node_id": node3_id,
                "name": "captain.translate",
                "type": "protein",
                "input": {"n_genes": len(pred_rna_delta), "n_proteins": len(protein_panel)},
                "output": {"n_proteins": len(pred_prot_delta)},
                "time_ms": int(node3_time)
            })
        
        # Generate hypothesis statement
        hypothesis = self._generate_hypothesis_text(
            perturbation_name,
            pred_rna_delta,
            pred_prot_delta,
            rna_comparison,
            prot_comparison
        )
        
        total_time = (time.time() - start_time) * 1000
        
        # Get graph representation
        graph_dict = self.graph.to_dict()
        
        # Assemble final JSON
        result = {
            "hypothesis": hypothesis,
            "perturbation_name": perturbation_name,
            "path": path,
            "graph": graph_dict,
            "predictions": {
                "rna": {
                    "delta": pred_rna_delta,
                    "n_genes": len(pred_rna_delta)
                },
                "protein": {
                    "delta": pred_prot_delta,
                    "n_proteins": len(pred_prot_delta)
                }
            },
            "ground_truth": {
                "rna": {
                    "delta": real_rna_delta,
                    "n_genes": len(real_rna_delta)
                },
                "protein": {
                    "delta": real_prot_delta,
                    "n_proteins": len(real_prot_delta)
                }
            },
            "validation_scores": {
                "rna": {
                    "spearman": rna_comparison.get("value"),
                    "cosine": rna_cosine.get("value"),
                    "mse": rna_comparison.get("mse"),
                    "mae": rna_comparison.get("mae"),
                    "n_features": rna_comparison.get("n_features")
                },
                "protein": {
                    "pearson": prot_comparison.get("value"),
                    "cosine": prot_cosine.get("value"),
                    "mse": prot_comparison.get("mse"),
                    "mae": prot_comparison.get("mae"),
                    "n_features": prot_comparison.get("n_features")
                },
                "edge_sign_accuracy": edge_sign_accuracy,
                "kg_edges_used": len(kg_path_edges)
            },
            "error_propagation": error_prop,
            "metadata": {
                "total_time_ms": int(total_time),
                "control_cells": len(self.data_loader.get_control_cells()),
                "perturbation_cells": len(self.data_loader.get_perturbation_cells(perturbation_name)),
                "embeddings_aligned": True,  # Assumed to be pre-aligned
                "embeddings_loaded": pert_embedding is not None
            }
        }
        
        return result
    
    def _generate_hypothesis_text(
        self,
        perturbation_name: str,
        pred_rna: Dict[str, float],
        pred_prot: Dict[str, float],
        rna_comparison: Dict[str, float],
        prot_comparison: Dict[str, float]
    ) -> str:
        """Generate hypothesis text from predictions and comparisons."""
        gene_name = perturbation_name.split("_")[0]
        
        top_rna_up = sorted(pred_rna.items(), key=lambda x: x[1], reverse=True)[:3]
        top_rna_down = sorted(pred_rna.items(), key=lambda x: x[1])[:3]
        
        rna_spearman = rna_comparison.get("value", float("nan"))
        prot_pearson = prot_comparison.get("value", float("nan"))
        
        hypothesis_parts = [
            f"Perturbation of {gene_name} is predicted to cause:",
            f"  - RNA changes in {len(pred_rna)} genes",
            f"  - Protein changes in {len(pred_prot)} markers"
        ]
        
        if top_rna_up:
            genes_up = ", ".join([g for g, v in top_rna_up])
            hypothesis_parts.append(f"  - Top upregulated genes: {genes_up}")
        
        if top_rna_down:
            genes_down = ", ".join([g for g, v in top_rna_down])
            hypothesis_parts.append(f"  - Top downregulated genes: {genes_down}")
        
        if not math.isnan(rna_spearman):
            hypothesis_parts.append(f"  - RNA prediction accuracy (Spearman): {rna_spearman:.3f}")
        
        if not math.isnan(prot_pearson):
            hypothesis_parts.append(f"  - Protein prediction accuracy (Pearson): {prot_pearson:.3f}")
        
        return "\n".join(hypothesis_parts)

