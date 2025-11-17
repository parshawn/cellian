"""Report generation for Virtual Cell analysis."""

from typing import Dict, List, Any, Optional
from datetime import datetime


def build_report(
    context: Dict[str, Any],
    pathways: List[Dict[str, Any]],
    phenotypes: List[Dict[str, Any]],
    hypotheses: Dict[str, Any],
    plot_paths: Dict[str, str]
) -> str:
    """
    Build a human-readable report summarizing the analysis.
    
    Args:
        context: Context dict with perturbation, cell_type, species, user_question
        pathways: List of pathway dicts
        phenotypes: List of phenotype dicts
        hypotheses: Dict from generate_hypotheses() with "hypotheses" key
        plot_paths: Dict with keys: volcano, pathway_enrichment, phenotype_scores, rna_gsea, protein_psea, phenotype_enrichment
    
    Returns:
        Markdown string with complete report
    """
    lines = []
    
    # Title
    perturbation = context.get("perturbation", "Unknown")
    cell_type = context.get("cell_type", "")
    lines.append("# Virtual Cell Analysis Report")
    lines.append("")
    lines.append(f"**Perturbation:** {perturbation}")
    if cell_type:
        lines.append(f"**Cell Type:** {cell_type}")
    if context.get("species"):
        lines.append(f"**Species:** {context.get('species')}")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # User question (if provided)
    user_question = context.get("user_question")
    if user_question:
        lines.append("## Query")
        lines.append(f"{user_question}")
        lines.append("")
    
    # Methods summary
    lines.append("## Methods")
    lines.append("")
    lines.append("This analysis was performed using the Virtual Cell system, which integrates:")
    lines.append("- **Differential Expression Analysis:** RNA-seq data analysis")
    lines.append("- **Pathway Enrichment:** GSEA (Gene Set Enrichment Analysis) and PSEA (Pathway Set Enrichment Analysis)")
    lines.append("- **Literature Integration:** Edison Scientific RAG system (PaperQA) for evidence retrieval")
    lines.append("- **Hypothesis Generation:** Mechanistic hypothesis generation with literature support")
    lines.append("")
    
    # Key DE + Enrichment Results
    lines.append("## Key Results")
    lines.append("")
    
    # DEG summary
    deg_list = context.get("deg_list", [])
    deg_count = len(deg_list)
    if deg_count > 0:
        significant_degs = [d for d in deg_list if d.get("pval", 1.0) <= 0.05]
        num_significant = len(significant_degs)
        lines.append(f"**Differential Expression:** {num_significant} significantly differentially expressed genes (p ≤ 0.05) out of {deg_count} total genes")
        lines.append("")
        
        # Show top up/down regulated genes
        if significant_degs:
            # Sort by log2FC
            sorted_degs = sorted(significant_degs, key=lambda x: x.get("log2fc", 0.0), reverse=True)
            
            # Top upregulated (positive log2FC)
            upregulated = [d for d in sorted_degs if d.get("log2fc", 0.0) > 0]
            if upregulated:
                lines.append("**Top Upregulated Genes:**")
                for i, deg in enumerate(upregulated[:10], 1):
                    gene = deg.get("gene", "Unknown")
                    log2fc = deg.get("log2fc", 0.0)
                    pval = deg.get("pval", 1.0)
                    pval_adj = deg.get("pval_adj", 1.0)
                    pred_mean = deg.get("pred_mean")
                    control_mean = deg.get("control_mean")
                    
                    if pred_mean is not None and control_mean is not None:
                        lines.append(f"{i}. **{gene}**: log2FC={log2fc:.2f}, p={pval:.3e}, p_adj={pval_adj:.3f} (perturbed: {pred_mean:.2f}, control: {control_mean:.2f})")
                    else:
                        lines.append(f"{i}. **{gene}**: log2FC={log2fc:.2f}, p={pval:.3e}, p_adj={pval_adj:.3f}")
                lines.append("")
            
            # Top downregulated (negative log2FC)
            downregulated = [d for d in sorted_degs if d.get("log2fc", 0.0) < 0]
            if downregulated:
                lines.append("**Top Downregulated Genes:**")
                for i, deg in enumerate(downregulated[:10], 1):
                    gene = deg.get("gene", "Unknown")
                    log2fc = deg.get("log2fc", 0.0)
                    pval = deg.get("pval", 1.0)
                    pval_adj = deg.get("pval_adj", 1.0)
                    pred_mean = deg.get("pred_mean")
                    control_mean = deg.get("control_mean")
                    
                    if pred_mean is not None and control_mean is not None:
                        lines.append(f"{i}. **{gene}**: log2FC={log2fc:.2f}, p={pval:.3e}, p_adj={pval_adj:.3f} (perturbed: {pred_mean:.2f}, control: {control_mean:.2f})")
                    else:
                        lines.append(f"{i}. **{gene}**: log2FC={log2fc:.2f}, p={pval:.3e}, p_adj={pval_adj:.3f}")
                lines.append("")
        
        # Volcano plot
        if plot_paths.get("volcano"):
            lines.append("")
            lines.append(f"![Volcano Plot]({plot_paths['volcano']})")
            lines.append("")
    
    # Pathway enrichment (use p-value, not FDR)
    if pathways:
        # Use p-value <= 0.05 (not FDR)
        significant_pathways = []
        for p in pathways:
            pval = p.get("pval") or p.get("pvalue") or p.get("P-value") or p.get("P_value") or 1.0
            if isinstance(pval, (int, float)) and pval <= 0.05:
                significant_pathways.append(p)
        lines.append(f"**Pathway Enrichment:** {len(significant_pathways)} significantly enriched pathways (p ≤ 0.05)")
        
        if significant_pathways:
            lines.append("")
            lines.append("Top enriched pathways:")
            for i, pathway in enumerate(significant_pathways[:10], 1):
                name = pathway.get("name", "Unknown")
                nes = pathway.get("NES", 0.0)
                pval = pathway.get("pval") or pathway.get("pvalue") or pathway.get("P-value") or pathway.get("P_value") or 1.0
                fdr = pathway.get("FDR", 1.0)  # Keep FDR for reference
                lines.append(f"{i}. {name} (NES={nes:.2f}, p={pval:.3e}, FDR={fdr:.3f})")
        
        # Pathway plot
        if plot_paths.get("pathway_enrichment"):
            lines.append("")
            lines.append(f"![Pathway Enrichment]({plot_paths['pathway_enrichment']})")
            lines.append("")
        
        # RNA GSEA plot
        if plot_paths.get("rna_gsea"):
            lines.append("")
            lines.append(f"![RNA GSEA Enrichment Plot]({plot_paths['rna_gsea']})")
            lines.append("")
        
        # Protein/PPI PSEA plot
        if plot_paths.get("protein_psea"):
            lines.append("")
            lines.append(f"![Protein/PPI PSEA Enrichment Plot]({plot_paths['protein_psea']})")
            lines.append("")
    
    # Hypotheses
    lines.append("## Mechanistic Hypotheses")
    lines.append("")
    
    hypothesis_list = hypotheses.get("hypotheses", [])
    
    if hypothesis_list:
        lines.append(f"Generated {len(hypothesis_list)} testable mechanistic hypotheses:")
        lines.append("")
        
        for hyp in hypothesis_list:
            hyp_id = hyp.get("id", "Unknown")
            statement = hyp.get("statement", "")
            
            lines.append(f"### Hypothesis {hyp_id}: {statement}")
            lines.append("")
            
            # Mechanism
            mechanism = hyp.get("mechanism", [])
            if mechanism:
                lines.append("**Mechanism:**")
                for step in mechanism:
                    lines.append(f"- {step}")
                lines.append("")
            
            # Phenotype support
            pheno_support = hyp.get("phenotype_support")
            if pheno_support:
                primary_phenotype = pheno_support.get("primary_phenotype")
                if primary_phenotype:
                    score = pheno_support.get("score", 0.0)
                    direction = pheno_support.get("direction", "unknown")
                    lines.append(f"**Phenotype Support:** {primary_phenotype} ({direction}, score={score:.2f})")
                    lines.append("")
            
            # Literature support
            lit_support = hyp.get("literature_support", {})
            overall = lit_support.get("overall", "unknown")
            summary = lit_support.get("summary", "")
            supporting_papers_full = lit_support.get("supporting_papers_full", [])
            supporting_pmids = lit_support.get("supporting_papers", [])
            
            lines.append(f"**Literature Support:** {overall.upper()}")
            lines.append(f"{summary}")
            
            # Use full formatted citations if available, otherwise fall back to PMIDs
            if supporting_papers_full:
                lines.append("**Supporting Papers:**")
                from .futurehouse_client import format_citation
                for paper in supporting_papers_full[:5]:  # Top 5
                    citation = format_citation(paper)
                    lines.append(f"- {citation}")
            elif supporting_pmids:
                lines.append("**Supporting Papers:**")
                for pmid in supporting_pmids[:5]:  # Top 5
                    lines.append(f"- {pmid}")
            lines.append("")
            
            # Predicted readouts
            readouts = hyp.get("predicted_readouts", [])
            if readouts:
                lines.append("**Predicted Experimental Readouts:**")
                for readout in readouts:
                    lines.append(f"- {readout}")
                lines.append("")
            
            # Suggested experiments
            experiments = hyp.get("experiments", [])
            if experiments:
                lines.append("**Suggested Experiments:**")
                for exp in experiments:
                    lines.append(f"- {exp}")
                lines.append("")
            
            # Speculation notes
            speculation = hyp.get("speculation_notes", "")
            if speculation:
                lines.append("**Notes:**")
                lines.append(f"{speculation}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
    else:
        # Check if there's an explanation for why no hypotheses were generated
        no_hyp_explanation = hypotheses.get("no_hypotheses_explanation")
        literature_papers = hypotheses.get("supporting_papers_full", [])
        literature_search_performed = hypotheses.get("literature_search_performed", False)
        
        if no_hyp_explanation:
            lines.append("**No hypotheses generated.**")
            lines.append("")
            lines.append(f"**Explanation:** {no_hyp_explanation}")
            lines.append("")
            
            if literature_search_performed:
                num_papers = hypotheses.get("literature_papers_found", 0)
                lines.append(f"**Literature Search:** Performed ({num_papers} papers found)")
                if literature_papers:
                    lines.append("**Relevant Papers Found:**")
                    from .futurehouse_client import format_citation
                    for paper in literature_papers[:5]:
                        citation = format_citation(paper)
                        lines.append(f"- {citation}")
                    lines.append("")
        else:
            lines.append("No hypotheses generated.")
            lines.append("")
    
    # Limitations / Notes
    lines.append("## Limitations & Notes")
    lines.append("")
    lines.append("- **Computational Predictions:** This analysis is based on computational predictions and requires experimental validation.")
    lines.append("- **Literature Evidence:** Literature support classification is automated and should be manually reviewed for critical hypotheses.")
    lines.append("- **Tissue/Cell Context:** Results are specific to the analyzed cell type and may not generalize to other contexts.")
    lines.append("- **Dynamic Processes:** This analysis represents a snapshot and does not capture temporal dynamics of cellular responses.")
    lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by Virtual Cell system*")
    lines.append("")
    
    return "\n".join(lines)


def build_html_report(
    context: Dict[str, Any],
    pathways: List[Dict[str, Any]],
    phenotypes: List[Dict[str, Any]],
    hypotheses: Dict[str, Any],
    plot_paths: Dict[str, str]
) -> str:
    """
    Build an HTML report (alternative to Markdown).
    
    Args:
        Same as build_report()
    
    Returns:
        HTML string with complete report
    """
    # Convert plot paths to base64 or use relative paths
    import os
    from pathlib import Path
    
    def get_image_tag(path_key: str) -> str:
        path = plot_paths.get(path_key)
        if not path or not os.path.exists(path):
            return ""
        
        # Use relative path or convert to base64
        return f'<img src="{path}" alt="{path_key}" style="max-width: 100%; height: auto;">'
    
    html_lines = []
    html_lines.append("<!DOCTYPE html>")
    html_lines.append("<html>")
    html_lines.append("<head>")
    html_lines.append('<meta charset="UTF-8">')
    html_lines.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    html_lines.append("<title>Virtual Cell Analysis Report</title>")
    html_lines.append("<style>")
    html_lines.append("body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }")
    html_lines.append("h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }")
    html_lines.append("h2 { color: #34495e; margin-top: 30px; }")
    html_lines.append("h3 { color: #7f8c8d; }")
    html_lines.append("code { background-color: #ecf0f1; padding: 2px 5px; border-radius: 3px; }")
    html_lines.append("img { margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }")
    html_lines.append("hr { border: 1px solid #ecf0f1; margin: 30px 0; }")
    html_lines.append("</style>")
    html_lines.append("</head>")
    html_lines.append("<body>")
    
    # Convert markdown to HTML (simple conversion)
    markdown = build_report(context, pathways, phenotypes, hypotheses, plot_paths)
    
    # Simple markdown to HTML conversion
    html_content = markdown
    
    # Convert basic markdown
    html_content = html_content.replace("### ", "<h3>").replace("\n", "</h3>\n")
    html_content = html_content.replace("## ", "<h2>").replace("\n", "</h2>\n")
    html_content = html_content.replace("# ", "<h1>").replace("\n", "</h1>\n")
    html_content = html_content.replace("**", "<strong>").replace("**", "</strong>")
    html_content = html_content.replace("\n\n", "</p><p>")
    html_content = html_content.replace("\n", "<br>")
    html_content = html_content.replace("---", "<hr>")
    
    # Fix image tags (already in HTML format)
    html_content = html_content.replace('![', '<img src="').replace('](', '" alt="').replace(')', '">')
    
    html_lines.append(f"<p>{html_content}</p>")
    
    html_lines.append("</body>")
    html_lines.append("</html>")
    
    return "\n".join(html_lines)

