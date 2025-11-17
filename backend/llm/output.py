"""LLM output processing: interprets Agent_Tools results and generates natural language summaries."""
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import pandas as pd
    import anndata as ad
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from dotenv import load_dotenv
    # Load .env file from llm directory first
    local_env_path = Path(__file__).parent / '.env'
    if local_env_path.exists():
        load_dotenv(local_env_path)
    # Also try loading from project root
    root_env_path = Path(__file__).parent.parent / '.env'
    if root_env_path.exists():
        load_dotenv(root_env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

try:
    import google.generativeai as genai
except ImportError:
    genai = None


SYSTEM_PROMPT = """You are a biological reasoning interpreter. You receive the results of a perturbation analysis pipeline (perturbation → RNA → protein) and generate a comprehensive natural language summary.

Your task is to:
1. Interpret the perturbation and its predicted effects
2. Explain the validation scores (R2, Pearson correlation, RMSE, MAE) and what they mean
3. Describe pathway enrichment results and biological significance
4. Summarize key findings in 2-4 sentences
5. Provide biological insights based on the results

Be clear, concise, and focus on biological significance. Use scientific terminology appropriately."""

MODEL_NAME = "gemini-2.0-flash-exp"  # Best Gemini model (falls back to gemini-1.5-pro if unavailable)


def interpret_results(
    results: Dict[str, Any],
    query: Optional[str] = None,
    perturbation_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Interpret Agent_Tools results and generate natural language summary using LLM.
    
    Args:
        results: Results dictionary from Agent_Tools pipeline. Expected structure:
            - "target_gene": str, target gene name
            - "rna_metrics": dict with keys like 'r2', 'pearson_r', 'rmse', 'mae' 
              (from evaluate_predictions() in Agent_Tools/evaluation.py)
            - "protein_metrics": dict with keys like 'r2', 'pearson_r', 'rmse', 'mae'
              (from evaluate_predictions() in Agent_Tools/evaluation.py)
            - "pathway_analysis": dict (from comprehensive_analysis() in Agent_Tools/pathway_analysis.py) with:
              - "correlation": dict with 'correlation', 'pvalue', 'n_common_genes'
              - "differential_rna": pd.DataFrame or None
              - "differential_protein": pd.DataFrame or None
              - "gsea_rna": pd.DataFrame or None
              - "kegg_enrichment": pd.DataFrame or None (protein)
              - "reactome_enrichment": pd.DataFrame or None (protein)
              - "go_enrichment": pd.DataFrame or None (protein)
              - "kegg_enrichment_rna": pd.DataFrame or None (RNA, NEW)
              - "reactome_enrichment_rna": pd.DataFrame or None (RNA, NEW)
              - "go_enrichment_rna": pd.DataFrame or None (RNA, NEW)
        
        Note: perturbation_pipeline.py doesn't return a structured dict. You need to:
        1. Manually collect rna_metrics and protein_metrics from evaluate_predictions() calls
        2. Manually collect pathway_analysis from comprehensive_analysis() return value
        3. Or use collect_results_from_pipeline() helper function to load from output directory
        
        query: Original user query (optional)
        perturbation_info: Extracted perturbation information (optional)
    
    Returns:
        Natural language interpretation of results
    """
    if genai is None:
        return _generate_fallback_summary(results, query, perturbation_info)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _generate_fallback_summary(results, query, perturbation_info)
    
    # Strip whitespace to avoid authentication issues
    api_key = api_key.strip()
    
    # Validate key format (Gemini keys start with "AIza")
    if not api_key.startswith("AIza"):
        return _generate_fallback_summary(results, query, perturbation_info)
    
    # Debug: Log key format (first/last few chars only for security)
    if len(api_key) < 20:
        return _generate_fallback_summary(results, query, perturbation_info)
    
    try:
        genai.configure(api_key=api_key)
    except Exception:
        return _generate_fallback_summary(results, query, perturbation_info)
    
    # Prepare results summary for LLM
    target_gene = results.get("target_gene") or (perturbation_info.get("target") if perturbation_info else "unknown")
    pert_type = (perturbation_info.get("type") if perturbation_info else "unknown")
    
    results_summary = {
        "perturbation": {
            "target": target_gene,
            "type": pert_type
        },
        "rna_metrics": results.get("rna_metrics", {}),
        "protein_metrics": results.get("protein_metrics", {}),
        "pathway_analysis": _summarize_pathway_analysis(results.get("pathway_analysis", {}))
    }
    
    user_message = f"""Original query: {query or "Not provided"}

Perturbation: {target_gene} ({pert_type})

Results Summary:
{json.dumps(results_summary, indent=2, default=str)}

Generate a comprehensive natural language interpretation of these results, focusing on:
1. What the perturbation predicts (RNA and protein changes)
2. How accurate the predictions are (based on validation scores: R2, Pearson correlation, RMSE, MAE)
3. What pathway enrichment results tell us about biological significance
4. Key biological insights and implications"""
    
    try:
        # Try the best model first, fall back to gemini-1.5-pro if unavailable
        try:
            model = genai.GenerativeModel(MODEL_NAME)
        except Exception:
            model = genai.GenerativeModel("gemini-1.5-pro")
        
        # Combine system prompt and user message for Gemini
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_message}"
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000
            )
        )
        
        return response.text
    except Exception as e:
        return _generate_fallback_summary(results, query, perturbation_info)


def generate_summary(
    results: Dict[str, Any],
    query: Optional[str] = None,
    perturbation_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive summary including both structured results and LLM interpretation.
    
    Args:
        results: Results dictionary from Agent_Tools pipeline
        query: Original user query (optional)
        perturbation_info: Extracted perturbation information (optional)
    
    Returns:
        Dictionary with:
        - "query": original query
        - "perturbation_info": perturbation information
        - "results": original results
        - "interpretation": LLM-generated natural language interpretation
    """
    interpretation = interpret_results(results, query, perturbation_info)
    
    return {
        "query": query,
        "perturbation_info": perturbation_info,
        "results": results,
        "interpretation": interpretation
    }


def _summarize_pathway_analysis(pathway_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize pathway analysis results for LLM consumption."""
    summary = {}
    
    # Correlation
    if "correlation" in pathway_analysis:
        corr = pathway_analysis["correlation"]
        summary["correlation"] = {
            "correlation": corr.get("correlation"),
            "pvalue": corr.get("pvalue"),
            "n_common_genes": corr.get("n_common_genes", 0)
        }
    
    # Differential expression - RNA
    if "differential_rna" in pathway_analysis:
        de_rna = pathway_analysis["differential_rna"]
        if hasattr(de_rna, '__len__') and len(de_rna) > 0:
            summary["differential_rna"] = {
                "n_significant": len(de_rna) if hasattr(de_rna, '__len__') else 0,
                "top_genes": de_rna.head(10).to_dict('records') if hasattr(de_rna, 'head') else []
            }
    
    # Differential expression - Protein
    if "differential_protein" in pathway_analysis:
        de_protein = pathway_analysis["differential_protein"]
        if hasattr(de_protein, '__len__') and len(de_protein) > 0:
            summary["differential_protein"] = {
                "n_significant": len(de_protein) if hasattr(de_protein, '__len__') else 0,
                "top_proteins": de_protein.head(10).to_dict('records') if hasattr(de_protein, 'head') else []
            }
    
    # GSEA - RNA
    if "gsea_rna" in pathway_analysis and pathway_analysis["gsea_rna"] is not None:
        gsea = pathway_analysis["gsea_rna"]
        if hasattr(gsea, '__len__') and len(gsea) > 0:
            summary["gsea_rna"] = {
                "n_pathways": len(gsea) if hasattr(gsea, '__len__') else 0,
                "top_pathways": gsea.head(10).to_dict('records') if hasattr(gsea, 'head') else []
            }
    
    # Pathway enrichments - Protein
    for db in ["kegg_enrichment", "reactome_enrichment", "go_enrichment"]:
        if db in pathway_analysis and pathway_analysis[db] is not None:
            enrichment = pathway_analysis[db]
            if hasattr(enrichment, '__len__') and len(enrichment) > 0:
                summary[db] = {
                    "n_pathways": len(enrichment) if hasattr(enrichment, '__len__') else 0,
                    "top_pathways": enrichment.head(10).to_dict('records') if hasattr(enrichment, 'head') else []
                }
    
    # Pathway enrichments - RNA (NEW in updated Agent_Tools)
    for db in ["kegg_enrichment_rna", "reactome_enrichment_rna", "go_enrichment_rna"]:
        if db in pathway_analysis and pathway_analysis[db] is not None:
            enrichment = pathway_analysis[db]
            if hasattr(enrichment, '__len__') and len(enrichment) > 0:
                summary[db] = {
                    "n_pathways": len(enrichment) if hasattr(enrichment, '__len__') else 0,
                    "top_pathways": enrichment.head(10).to_dict('records') if hasattr(enrichment, 'head') else []
                }
    
    return summary


def collect_results_from_pipeline(
    output_dir: str,
    target_gene: str,
    rna_metrics: Optional[Dict[str, Any]] = None,
    protein_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Collect results from Agent_Tools pipeline output directory.
    
    This helper function constructs a results dictionary compatible with interpret_results()
    from the pipeline's output directory. Since perturbation_pipeline.py doesn't return
    a structured dict, this function helps collect results from files and metrics.
    
    Args:
        output_dir: Directory where pipeline saved results (e.g., Agent_Tools/temp_output)
        target_gene: Target gene name
        rna_metrics: Optional RNA metrics dict from evaluate_predictions()
        protein_metrics: Optional protein metrics dict from evaluate_predictions()
    
    Returns:
        Results dictionary compatible with interpret_results()
    """
    results = {
        "target_gene": target_gene,
        "rna_metrics": rna_metrics or {},
        "protein_metrics": protein_metrics or {},
        "pathway_analysis": {}
    }
    
    # Try to load pathway analysis results if available
    if PANDAS_AVAILABLE:
        pathway_analysis_dir = os.path.join(output_dir, 'pathway_analysis')
        
        if os.path.exists(pathway_analysis_dir):
            # Try to load pathway analysis CSV files
            try:
                # Use target_gene for filename if provided, otherwise try to find files without suffix
                gene_suffix = f"_{target_gene}" if target_gene else ""
                
                # Correlation analysis
                corr_files = [
                    os.path.join(pathway_analysis_dir, f'correlation_analysis{gene_suffix}.csv'),
                    os.path.join(pathway_analysis_dir, 'correlation_analysis.csv')
                ]
                for corr_file in corr_files:
                    if os.path.exists(corr_file):
                        corr_df = pd.read_csv(corr_file)
                        # Extract correlation stats (implementation depends on CSV structure)
                        results["pathway_analysis"]["correlation"] = {
                            "correlation": None,  # Would need to extract from CSV
                            "n_common_genes": len(corr_df) if len(corr_df) > 0 else 0
                        }
                        break
                
                # Differential expression - RNA
                # Handle both gene and drug perturbations (drugs have different filename format)
                de_rna_files = [
                    os.path.join(pathway_analysis_dir, f'differential_expression_rna{gene_suffix}.csv'),
                    os.path.join(pathway_analysis_dir, 'differential_expression_rna.csv')
                ]
                # For drug perturbations, also check for files with drug name pattern
                # Pattern: differential_expression_rna_[('DrugName', conc, unit)].csv
                import glob
                drug_pattern = os.path.join(pathway_analysis_dir, 'differential_expression_rna_*.csv')
                drug_files = glob.glob(drug_pattern)
                if drug_files:
                    de_rna_files.extend(drug_files)
                    # Only log once if multiple drug files found (not for every call)
                    # Removed verbose debug message
                
                for de_rna_file in de_rna_files:
                    if os.path.exists(de_rna_file):
                        results["pathway_analysis"]["differential_rna"] = pd.read_csv(de_rna_file)
                        # Only print once, not on every call
                        if not hasattr(collect_results_from_pipeline, '_rna_loaded'):
                            print(f"  ✓ Loaded differential expression RNA from: {os.path.basename(de_rna_file)}")
                            collect_results_from_pipeline._rna_loaded = True
                        break
                else:
                    # If no file was found, provide more detailed debugging info
                    if not de_rna_files or not any(os.path.exists(f) for f in de_rna_files):
                        print(f"  ⚠️  No differential expression RNA files found")
                        print(f"  [DEBUG] Searched in: {pathway_analysis_dir}")
                        print(f"  [DEBUG] Searched {len(de_rna_files)} file patterns")
                        # List what files actually exist
                        if os.path.exists(pathway_analysis_dir):
                            actual_files = [f for f in os.listdir(pathway_analysis_dir) if 'differential' in f.lower() and 'rna' in f.lower()]
                            if actual_files:
                                print(f"  [DEBUG] Found files in directory: {actual_files[:5]}")
                            else:
                                print(f"  [DEBUG] No differential expression RNA files found in directory")
                
                # Differential expression - Protein
                # Handle both gene and drug perturbations
                de_protein_files = [
                    os.path.join(pathway_analysis_dir, f'differential_expression_protein{gene_suffix}.csv'),
                    os.path.join(pathway_analysis_dir, 'differential_expression_protein.csv')
                ]
                # For drug perturbations, also check for files with drug name pattern
                import glob
                drug_pattern = os.path.join(pathway_analysis_dir, 'differential_expression_protein_*.csv')
                drug_files = glob.glob(drug_pattern)
                if drug_files:
                    de_protein_files.extend(drug_files)
                
                for de_protein_file in de_protein_files:
                    if os.path.exists(de_protein_file):
                        results["pathway_analysis"]["differential_protein"] = pd.read_csv(de_protein_file)
                        break
                
                # GSEA - RNA (in gsea_rna/ subdirectory)
                gsea_dir = os.path.join(pathway_analysis_dir, 'gsea_rna')
                if os.path.exists(gsea_dir):
                    # Look for GSEA report CSV
                    gsea_files = [
                        os.path.join(gsea_dir, 'gseapy.gene_set.prerank.report.csv'),
                        os.path.join(gsea_dir, f'gsea_rna{gene_suffix}.csv'),
                        os.path.join(gsea_dir, 'gsea_rna.csv'),
                        os.path.join(pathway_analysis_dir, f'gsea_rna{gene_suffix}.csv'),
                        os.path.join(pathway_analysis_dir, 'gsea_rna.csv')
                    ]
                    for gsea_file in gsea_files:
                        if os.path.exists(gsea_file):
                            results["pathway_analysis"]["gsea_rna"] = pd.read_csv(gsea_file)
                            break
                    else:
                        results["pathway_analysis"]["gsea_rna"] = None
                else:
                    # Try in main directory
                    gsea_files = [
                        os.path.join(pathway_analysis_dir, f'gsea_rna{gene_suffix}.csv'),
                        os.path.join(pathway_analysis_dir, 'gsea_rna.csv')
                    ]
                    for gsea_file in gsea_files:
                        if os.path.exists(gsea_file):
                            results["pathway_analysis"]["gsea_rna"] = pd.read_csv(gsea_file)
                            break
                    else:
                        results["pathway_analysis"]["gsea_rna"] = None
                
                # Pathway enrichments - Protein (in subdirectories with .txt files)
                enrichment_patterns = {
                    "kegg_enrichment": ["KEGG_2021_Human.Human.enrichr.reports.txt", f"kegg_enrichment{gene_suffix}.csv", f"kegg_enrichment{gene_suffix}.txt", "kegg_enrichment.csv"],
                    "reactome_enrichment": ["Reactome_2022.Human.enrichr.reports.txt", f"reactome_enrichment{gene_suffix}.csv", f"reactome_enrichment{gene_suffix}.txt", "reactome_enrichment.csv"],
                    "go_enrichment": ["GO_Biological_Process_2021.Human.enrichr.reports.txt", f"go_enrichment{gene_suffix}.csv", f"go_enrichment{gene_suffix}.txt", "go_enrichment.csv"]
                }
                
                for db, patterns in enrichment_patterns.items():
                    db_dir = os.path.join(pathway_analysis_dir, db)
                    enrich_file = None
                    
                    if os.path.exists(db_dir):
                        # Look in subdirectory
                        for pattern in patterns:
                            potential_file = os.path.join(db_dir, pattern)
                            if os.path.exists(potential_file):
                                enrich_file = potential_file
                                break
                    else:
                        # Try directly in pathway_analysis_dir
                        for pattern in patterns:
                            potential_file = os.path.join(pathway_analysis_dir, pattern)
                            if os.path.exists(potential_file):
                                enrich_file = potential_file
                                break
                    
                    if enrich_file:
                        # Read as CSV or TSV (tab-separated)
                        try:
                            if enrich_file.endswith('.csv'):
                                results["pathway_analysis"][db] = pd.read_csv(enrich_file)
                            else:
                                # Try tab-separated (enrichr reports are usually tab-separated)
                                results["pathway_analysis"][db] = pd.read_csv(enrich_file, sep='\t')
                        except Exception:
                            # Fallback: try default separator
                            try:
                                results["pathway_analysis"][db] = pd.read_csv(enrich_file, sep=None, engine='python')
                            except Exception:
                                results["pathway_analysis"][db] = None
                    else:
                        results["pathway_analysis"][db] = None
                
                # Pathway enrichments - RNA (NEW in updated Agent_Tools)
                rna_enrichment_patterns = {
                    "kegg_enrichment_rna": ["KEGG_2021_Human.Human.enrichr.reports.txt", f"kegg_enrichment_rna{gene_suffix}.csv", f"kegg_enrichment_rna{gene_suffix}.txt", "kegg_enrichment_rna.csv"],
                    "reactome_enrichment_rna": ["Reactome_2022.Human.enrichr.reports.txt", f"reactome_enrichment_rna{gene_suffix}.csv", f"reactome_enrichment_rna{gene_suffix}.txt", "reactome_enrichment_rna.csv"],
                    "go_enrichment_rna": ["GO_Biological_Process_2021.Human.enrichr.reports.txt", f"go_enrichment_rna{gene_suffix}.csv", f"go_enrichment_rna{gene_suffix}.txt", "go_enrichment_rna.csv"]
                }
                
                for db, patterns in rna_enrichment_patterns.items():
                    db_dir = os.path.join(pathway_analysis_dir, db)
                    enrich_file = None
                    
                    if os.path.exists(db_dir):
                        # Look in subdirectory (e.g., kegg_enrichment_rna/)
                        for pattern in patterns:
                            potential_file = os.path.join(db_dir, pattern)
                            if os.path.exists(potential_file):
                                enrich_file = potential_file
                                break
                    else:
                        # Try directly in pathway_analysis_dir
                        for pattern in patterns:
                            potential_file = os.path.join(pathway_analysis_dir, pattern)
                            if os.path.exists(potential_file):
                                enrich_file = potential_file
                                break
                    
                    if enrich_file:
                        # Read as CSV or TSV (tab-separated)
                        try:
                            if enrich_file.endswith('.csv'):
                                results["pathway_analysis"][db] = pd.read_csv(enrich_file)
                            else:
                                # Try tab-separated (enrichr reports are usually tab-separated)
                                results["pathway_analysis"][db] = pd.read_csv(enrich_file, sep='\t')
                        except Exception:
                            # Fallback: try default separator
                            try:
                                results["pathway_analysis"][db] = pd.read_csv(enrich_file, sep=None, engine='python')
                            except Exception:
                                results["pathway_analysis"][db] = None
                    else:
                        results["pathway_analysis"][db] = None
                        
            except Exception as e:
                # If loading fails, just continue with empty pathway_analysis
                pass
    
    return results


def _generate_fallback_summary(
    results: Dict[str, Any],
    query: Optional[str] = None,
    perturbation_info: Optional[Dict[str, Any]] = None
) -> str:
    """Generate fallback summary without LLM."""
    target_gene = results.get("target_gene") or (perturbation_info.get("target") if perturbation_info else "unknown")
    pert_type = (perturbation_info.get("type") if perturbation_info else "unknown")
    
    summary = f"Perturbation Analysis: {target_gene} ({pert_type})\n\n"
    
    if query:
        summary += f"Query: {query}\n\n"
    
    # RNA metrics
    rna_metrics = results.get("rna_metrics", {})
    if rna_metrics:
        summary += "RNA Prediction Metrics:\n"
        if "r2" in rna_metrics:
            summary += f"  R2 Score: {rna_metrics['r2']:.4f}\n"
        if "pearson_r" in rna_metrics:
            summary += f"  Pearson R: {rna_metrics['pearson_r']:.4f}\n"
        if "rmse" in rna_metrics:
            summary += f"  RMSE: {rna_metrics['rmse']:.4f}\n"
        if "mae" in rna_metrics:
            summary += f"  MAE: {rna_metrics['mae']:.4f}\n"
        summary += "\n"
    
    # Protein metrics
    protein_metrics = results.get("protein_metrics", {})
    if protein_metrics:
        summary += "Protein Prediction Metrics:\n"
        if "r2" in protein_metrics:
            summary += f"  R2 Score: {protein_metrics['r2']:.4f}\n"
        if "pearson_r" in protein_metrics:
            summary += f"  Pearson R: {protein_metrics['pearson_r']:.4f}\n"
        if "rmse" in protein_metrics:
            summary += f"  RMSE: {protein_metrics['rmse']:.4f}\n"
        if "mae" in protein_metrics:
            summary += f"  MAE: {protein_metrics['mae']:.4f}\n"
        summary += "\n"
    
    # Pathway analysis summary
    pathway_analysis = results.get("pathway_analysis", {})
    if pathway_analysis:
        summary += "Pathway Analysis:\n"
        if "correlation" in pathway_analysis:
            corr = pathway_analysis["correlation"]
            if "correlation" in corr:
                summary += f"  RNA-Protein Correlation: {corr['correlation']:.4f}\n"
        if "gsea_rna" in pathway_analysis and pathway_analysis["gsea_rna"] is not None:
            gsea = pathway_analysis["gsea_rna"]
            if hasattr(gsea, '__len__'):
                summary += f"  GSEA Pathways (RNA): {len(gsea)} pathways identified\n"
        summary += "\n"
    
    return summary

