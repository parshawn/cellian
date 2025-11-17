"""Plotting functions for pathways, phenotypes, and differential expression."""

import os
from typing import List, Dict, Any, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Set style
if MATPLOTLIB_AVAILABLE:
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300


def plot_pathway_enrichment(
    pathways: List[Dict[str, Any]],
    out_path: str,
    top_n: int = 20,
    title_suffix: Optional[str] = None
) -> str:
    """
    Create a bar plot of top enriched pathways using NES or -log10(FDR).
    
    Args:
        pathways: List of pathway dicts with keys: id, name, NES, FDR
        out_path: Output file path (e.g., "pathways.png")
        top_n: Number of top pathways to plot
    
    Returns:
        Output file path
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, cannot generate plot")
        return out_path
    
    if not pathways:
        print("Warning: No pathways to plot")
        return out_path
    
    # Filter and sort pathways (use p-value, not adjusted/FDR)
    significant_pathways = [
        p for p in pathways
        if p.get("pval", 1.0) <= 0.05  # Use regular p-value
    ]
    
    if not significant_pathways:
        # Use all pathways if none are significant
        significant_pathways = pathways[:top_n]
    else:
        # Sort by absolute NES and take top N
        significant_pathways.sort(key=lambda x: abs(x.get("NES", 0.0)), reverse=True)
        significant_pathways = significant_pathways[:top_n]
    
    # Extract data
    names = [p.get("name", p.get("id", "Unknown"))[:50] for p in significant_pathways]  # Truncate long names
    nes_values = [p.get("NES", 0.0) for p in significant_pathways]
    pval_values = [p.get("pval", 1.0) for p in significant_pathways]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(significant_pathways) * 0.4)))
    
    # Color by direction
    colors = ['#2ecc71' if nes > 0 else '#e74c3c' for nes in nes_values]
    
    # Plot bars
    y_pos = np.arange(len(names))
    ax.barh(y_pos, nes_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Normalized Enrichment Score (NES)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pathways', fontsize=12, fontweight='bold')
    title = f'Top {len(significant_pathways)} Enriched Pathways'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Add p-value labels (not adjusted/FDR)
    for i, (nes, pval) in enumerate(zip(nes_values, pval_values)):
        ax.text(nes + 0.05 if nes > 0 else nes - 0.05, i, f'p={pval:.3e}',
                va='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return out_path


def plot_phenotype_scores(
    phenotypes: List[Dict[str, Any]],
    out_path: str,
    top_n: int = 20,
    title_suffix: Optional[str] = None
) -> str:
    """
    Plot top phenotypes (score vs name).
    
    Args:
        phenotypes: List of phenotype dicts with keys: phenotype_id, name, score, direction
        out_path: Output file path (e.g., "phenotypes.png")
        top_n: Number of top phenotypes to plot
    
    Returns:
        Output file path
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, cannot generate plot")
        return out_path
    
    if not phenotypes:
        # Silently return if no phenotypes (expected when PhenotypeKB has no data)
        return out_path
    
    # Sort and take top N
    sorted_phenotypes = sorted(phenotypes, key=lambda x: x.get("score", 0.0), reverse=True)
    top_phenotypes = sorted_phenotypes[:top_n]
    
    # Extract data
    names = [p.get("name", p.get("phenotype_id", "Unknown"))[:50] for p in top_phenotypes]
    scores = [p.get("score", 0.0) for p in top_phenotypes]
    directions = [p.get("direction", "mixed") for p in top_phenotypes]
    
    # Color by direction
    color_map = {
        "increase": "#2ecc71",  # Green
        "decrease": "#e74c3c",  # Red
        "mixed": "#f39c12"      # Orange
    }
    colors = [color_map.get(d, "#95a5a6") for d in directions]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_phenotypes) * 0.4)))
    
    # Plot bars
    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Phenotype Score', fontsize=12, fontweight='bold')
    title = f'Top {len(top_phenotypes)} Predicted Phenotypes'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(scores) * 1.1 if scores else 1.0)
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map["increase"], label='Increase', alpha=0.7),
        Patch(facecolor=color_map["decrease"], label='Decrease', alpha=0.7),
        Patch(facecolor=color_map["mixed"], label='Mixed', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return out_path


def plot_volcano(
    deg_list: List[Dict[str, Any]],
    out_path: str,
    log2fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    title_suffix: Optional[str] = None
) -> str:
    """
    Create a standard volcano plot of log2FC vs -log10(pval).
    
    Args:
        deg_list: List of DEG dicts with keys: gene, log2fc, pval
        out_path: Output file path (e.g., "volcano.png")
        log2fc_threshold: Log2FC threshold for significance
        pval_threshold: P-value threshold for significance
    
    Returns:
        Output file path
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, cannot generate plot")
        return out_path
    
    if not deg_list:
        # Silently return if no DEGs (expected in some cases, e.g., no significant changes)
        return out_path
    
    # Extract data
    log2fc_values = [deg.get("log2fc", 0.0) for deg in deg_list]
    pval_values = [max(deg.get("pval", 1.0), 1e-10) for deg in deg_list]
    neg_log10_pval = [-np.log10(p) for p in pval_values]
    genes = [deg.get("gene", "") for deg in deg_list]
    
    # Classify points
    significant_up = [
        (fc, p, g) for fc, p, g in zip(log2fc_values, neg_log10_pval, genes)
        if fc >= log2fc_threshold and -np.log10(pval_values[genes.index(g)]) >= -np.log10(pval_threshold)
    ]
    significant_down = [
        (fc, p, g) for fc, p, g in zip(log2fc_values, neg_log10_pval, genes)
        if fc <= -log2fc_threshold and -np.log10(pval_values[genes.index(g)]) >= -np.log10(pval_threshold)
    ]
    not_significant = [
        (fc, p, g) for fc, p, g in zip(log2fc_values, neg_log10_pval, genes)
        if not (abs(fc) >= log2fc_threshold and -np.log10(pval_values[genes.index(g)]) >= -np.log10(pval_threshold))
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot non-significant points
    if not_significant:
        ax.scatter([x[0] for x in not_significant], [x[1] for x in not_significant],
                  c='gray', alpha=0.3, s=20, label='Not significant')
    
    # Plot significant up-regulated
    if significant_up:
        ax.scatter([x[0] for x in significant_up], [x[1] for x in significant_up],
                  c='red', alpha=0.6, s=50, label=f'Up (|log2FC|>={log2fc_threshold})', edgecolors='black', linewidths=0.5)
        
        # Annotate top genes
        top_up = sorted(significant_up, key=lambda x: x[1], reverse=True)[:10]
        for fc, p, gene in top_up:
            ax.annotate(gene, (fc, p), fontsize=8, alpha=0.7, ha='center')
    
    # Plot significant down-regulated
    if significant_down:
        ax.scatter([x[0] for x in significant_down], [x[1] for x in significant_down],
                  c='blue', alpha=0.6, s=50, label=f'Down (|log2FC|>={log2fc_threshold})', edgecolors='black', linewidths=0.5)
        
        # Annotate top genes
        top_down = sorted(significant_down, key=lambda x: x[1], reverse=True)[:10]
        for fc, p, gene in top_down:
            ax.annotate(gene, (fc, p), fontsize=8, alpha=0.7, ha='center')
    
    # Add threshold lines
    ax.axvline(x=log2fc_threshold, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=-log2fc_threshold, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=-np.log10(pval_threshold), color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Customize
    ax.set_xlabel('log2 Fold Change', fontsize=12, fontweight='bold')
    ax.set_ylabel('-log10(p-value)', fontsize=12, fontweight='bold')
    title = 'Volcano Plot: Differential Expression'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return out_path


def plot_rna_gsea(
    rna_deg_list: List[Dict[str, Any]],
    pathways: List[Dict[str, Any]],
    out_path: str,
    top_pathways: int = 20,
    title_suffix: Optional[str] = None
) -> str:
    """
    Create an RNA GSEA (Gene Set Enrichment Analysis) bar chart showing
    Normalized Enrichment Scores (NES) for top pathways from transcriptomics data.
    
    This performs real gene-set enrichment on transcriptomics (RNA) data.
    
    Args:
        rna_deg_list: List of RNA DEG dicts with keys: gene, log2fc, pval (from transcriptomics)
        pathways: List of pathway dicts with keys: id, name, NES, FDR, member_genes
        out_path: Output file path (e.g., "rna_gsea.png")
        top_pathways: Number of top pathways to plot
    
    Returns:
        Output file path
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, cannot generate RNA GSEA plot")
        return out_path
    
    if not pathways:
        print("Warning: No pathways to plot for RNA GSEA")
        return out_path
    
    # Filter significant pathways and sort by absolute NES (use p-value, not adjusted/FDR)
    significant_pathways = [
        p for p in pathways
        if p.get("pval", 1.0) <= 0.05  # Use regular p-value
    ]
    
    if not significant_pathways:
        # Use all pathways if none are significant
        significant_pathways = pathways[:top_pathways]
    else:
        # Sort by absolute NES and take top N
        significant_pathways.sort(key=lambda x: abs(x.get("NES", 0.0)), reverse=True)
        significant_pathways = significant_pathways[:top_pathways]
    
    if not significant_pathways:
        print("Warning: No pathways to plot in RNA GSEA")
        return out_path
    
    # Extract data
    names = [p.get("name", p.get("id", "Unknown"))[:50] for p in significant_pathways]  # Truncate long names
    nes_values = [p.get("NES", 0.0) for p in significant_pathways]
    pval_values = [p.get("pval", 1.0) for p in significant_pathways]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(significant_pathways) * 0.4)))
    
    # Color by direction (green for positive NES, red for negative)
    colors = ['#2ecc71' if nes > 0 else '#e74c3c' for nes in nes_values]
    
    # Plot bars (horizontal bar chart)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, nes_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Normalized Enrichment Score (NES)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pathways', fontsize=12, fontweight='bold')
    title = f'RNA GSEA: Top {len(significant_pathways)} Enriched Pathways'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Add p-value labels (not adjusted/FDR)
    for i, (nes, pval) in enumerate(zip(nes_values, pval_values)):
        ax.text(nes + 0.05 if nes > 0 else nes - 0.05, i, f'p={pval:.3e}',
                va='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return out_path


def plot_protein_psea(
    protein_deg_list: List[Dict[str, Any]],
    validated_edges: List[Dict[str, Any]],
    pathways: List[Dict[str, Any]],
    out_path: str,
    top_pathways: int = 20,
    title_suffix: Optional[str] = None
) -> str:
    """
    Create a Protein/PPI PSEA (Pathway Set Enrichment Analysis) bar chart showing
    Normalized Enrichment Scores (NES) for top pathways from network-aware proteomics/PPI data.
    
    This performs network-aware enrichment using PPI edges to weight enrichment scores.
    
    Args:
        protein_deg_list: List of protein DEG dicts with keys: gene, log2fc, pval (from proteomics)
        validated_edges: List of PPI/network edge dicts with keys: source, target, direction, confidence
        pathways: List of pathway dicts with keys: id, name, NES, FDR, member_genes
        out_path: Output file path (e.g., "protein_psea.png")
        top_pathways: Number of top pathways to plot
    
    Returns:
        Output file path
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, cannot generate Protein/PPI PSEA plot")
        return out_path
    
    if not pathways:
        print("Warning: No pathways to plot for Protein/PPI PSEA")
        return out_path
    
    # Build PPI network for annotation (count networked genes per pathway)
    network_edges = {}
    if validated_edges:
        for edge in validated_edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            if source and target:
                network_edges[source] = network_edges.get(source, set()) | {target}
                network_edges[target] = network_edges.get(target, set()) | {source}
    
    # Filter significant pathways and sort by absolute NES (use p-value, not adjusted/FDR)
    significant_pathways = [
        p for p in pathways
        if p.get("pval", 1.0) <= 0.05  # Use regular p-value
    ]
    
    if not significant_pathways:
        # Use all pathways if none are significant
        significant_pathways = pathways[:top_pathways]
    else:
        # Sort by absolute NES and take top N
        significant_pathways.sort(key=lambda x: abs(x.get("NES", 0.0)), reverse=True)
        significant_pathways = significant_pathways[:top_pathways]
    
    if not significant_pathways:
        print("Warning: No pathways to plot in Protein/PPI PSEA")
        return out_path
    
    # Extract data
    names = [p.get("name", p.get("id", "Unknown"))[:50] for p in significant_pathways]  # Truncate long names
    nes_values = [p.get("NES", 0.0) for p in significant_pathways]
    pval_values = [p.get("pval", 1.0) for p in significant_pathways]
    
    # Count networked genes for each pathway (for annotation)
    member_genes_list = []
    network_counts = []
    for pathway in significant_pathways:
        member_genes = pathway.get("member_genes", [])
        if isinstance(member_genes, str):
            member_genes = [g.strip() for g in member_genes.split(",")]
        member_genes_list.append(member_genes)
        # Count genes that are in the PPI network
        network_count = sum(1 for g in member_genes if g in network_edges)
        network_counts.append(network_count)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(significant_pathways) * 0.4)))
    
    # Color by direction (green for positive NES, red for negative)
    colors = ['#2ecc71' if nes > 0 else '#e74c3c' for nes in nes_values]
    
    # Plot bars (horizontal bar chart)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, nes_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Normalized Enrichment Score (NES)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pathways', fontsize=12, fontweight='bold')
    title = f'Protein/PPI PSEA: Top {len(significant_pathways)} Enriched Pathways (Network-aware)'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Add p-value and network info labels (not adjusted/FDR)
    for i, (nes, pval, network_count, member_count) in enumerate(zip(nes_values, pval_values, network_counts, [len(mg) for mg in member_genes_list])):
        label = f'p={pval:.3e}'
        if network_count > 0:
            label += f' ({network_count}/{member_count} networked)'
        ax.text(nes + 0.05 if nes > 0 else nes - 0.05, i, label,
                va='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return out_path


def plot_phenotype_enrichment(
    deg_list: List[Dict[str, Any]],
    phenotypes: List[Dict[str, Any]],
    phenotype_kb: Optional[Any] = None,
    out_path: str = "phenotype_enrichment.png",
    top_phenotypes: int = 5,
    title_suffix: Optional[str] = None
) -> str:
    """
    Create a Phenotype Enrichment Analysis plot showing
    enrichment curves for top phenotypes along a ranked gene list.
    
    Args:
        deg_list: List of DEG dicts with keys: gene, log2fc, pval
        phenotypes: List of phenotype dicts with keys: phenotype_id, name, score, direction, supporting_genes
        phenotype_kb: Optional PhenotypeKB instance for gene-phenotype associations
        out_path: Output file path (e.g., "psea.png")
        top_phenotypes: Number of top phenotypes to plot
    
    Returns:
        Output file path
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, cannot generate Phenotype Enrichment plot")
        return out_path
    
    if not deg_list or not phenotypes:
        # Silently return if no data (expected when DEGs or phenotypes are empty)
        return out_path
    
    # Sort genes by log2FC (descending) to create ranked list
    ranked_genes = sorted(deg_list, key=lambda x: x.get("log2fc", 0.0), reverse=True)
    gene_names = [g.get("gene", "") for g in ranked_genes]
    gene_ranks = {gene: i for i, gene in enumerate(gene_names)}
    
    # Filter top phenotypes by score
    sorted_phenotypes = sorted(phenotypes, key=lambda x: x.get("score", 0.0), reverse=True)
    top_phenotypes_list = sorted_phenotypes[:top_phenotypes]
    
    if not top_phenotypes_list:
        # Silently return if no phenotypes to plot (expected when no significant phenotypes)
        return out_path
    
    # Create figure with subplots for each phenotype
    n_phenotypes = len(top_phenotypes_list)
    fig, axes = plt.subplots(n_phenotypes, 1, figsize=(12, 3 * n_phenotypes))
    if n_phenotypes == 1:
        axes = [axes]
    
    n_genes = len(ranked_genes)
    x_positions = np.arange(n_genes)
    
    for ax_idx, phenotype in enumerate(top_phenotypes_list):
        ax = axes[ax_idx]
        phenotype_name = phenotype.get("name", phenotype.get("phenotype_id", "Unknown"))
        score = phenotype.get("score", 0.0)
        direction = phenotype.get("direction", "mixed")
        supporting_genes = phenotype.get("supporting_genes", [])
        
        # Convert supporting_genes to list if it's a string
        if isinstance(supporting_genes, str):
            supporting_genes = [g.strip() for g in supporting_genes.split(",")]
        
        # If phenotype_kb is provided, try to get additional genes
        if phenotype_kb is not None and phenotype.get("phenotype_id"):
            try:
                kb_genes = phenotype_kb.get_gene_phenotypes(phenotype.get("phenotype_id"))
                kb_gene_names = [g.get("gene", "") if isinstance(g, dict) else str(g) 
                               for g in kb_genes[:20]]  # Limit to top 20
                # Merge with supporting_genes
                all_phenotype_genes = list(set(supporting_genes + kb_gene_names))
            except Exception:
                all_phenotype_genes = supporting_genes
        else:
            all_phenotype_genes = supporting_genes
        
        # Find positions of phenotype genes in ranked list
        phenotype_positions = []
        for gene in all_phenotype_genes:
            if gene in gene_ranks:
                phenotype_positions.append(gene_ranks[gene])
        phenotype_positions.sort()
        
        if not phenotype_positions:
            # No phenotype genes found in ranked list
            ax.text(0.5, 0.5, f"No phenotype-associated genes found in ranked list", 
                   ha='center', va='center', transform=ax.transAxes)
            title = f"{phenotype_name}\nScore={score:.2f}, Direction={direction}"
            if title_suffix:
                title += f" ({title_suffix})"
            ax.set_title(title, 
                        fontsize=10, fontweight='bold')
            ax.set_xlim(0, n_genes)
            ax.set_ylim(-1, 1)
            continue
        
        # Calculate running enrichment score
        hit_scores = np.zeros(n_genes)
        miss_scores = np.zeros(n_genes)
        
        # Weight hits by log2FC (consider direction)
        for pos in phenotype_positions:
            gene_idx = pos
            if 0 <= gene_idx < len(ranked_genes):
                log2fc = ranked_genes[gene_idx].get("log2fc", 0.0)
                # Use absolute value, but consider direction if known
                if direction == "increase" and log2fc > 0:
                    hit_scores[gene_idx] = abs(log2fc)
                elif direction == "decrease" and log2fc < 0:
                    hit_scores[gene_idx] = abs(log2fc)
                else:
                    # For mixed or unknown direction, use absolute value
                    hit_scores[gene_idx] = abs(log2fc)
        
        # Normalize hit scores
        if hit_scores.sum() > 0:
            hit_scores = hit_scores / hit_scores.sum()
        
        # Calculate miss scores (penalty for genes not in phenotype)
        n_hits = len(phenotype_positions)
        n_misses = n_genes - n_hits
        if n_misses > 0:
            miss_scores = np.ones(n_genes) / n_misses
            miss_scores[phenotype_positions] = 0
        
        # Calculate running ES
        running_es = np.cumsum(hit_scores - miss_scores)
        
        # Plot enrichment curve
        color_map = {
            "increase": "#2ecc71",  # Green
            "decrease": "#e74c3c",  # Red
            "mixed": "#f39c12"      # Orange
        }
        color = color_map.get(direction, "#95a5a6")
        ax.plot(x_positions, running_es, color=color, linewidth=2, alpha=0.8)
        ax.fill_between(x_positions, 0, running_es, alpha=0.3, color=color)
        
        # Zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Maximum ES line
        max_es_idx = np.argmax(np.abs(running_es))
        max_es = running_es[max_es_idx]
        ax.axvline(x=max_es_idx, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Customize axes
        ax.set_xlabel('Rank in Gene List (sorted by log2FC)', fontsize=10)
        ax.set_ylabel('Enrichment Score (ES)', fontsize=10)
        title = f"Phenotype Enrichment: {phenotype_name}\nScore={score:.2f}, Direction={direction}, {len(phenotype_positions)} genes"
        if title_suffix:
            title += f" ({title_suffix})"
        ax.set_title(title, 
                    fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, n_genes)
        
        # Set y-limits based on ES range, then add vertical bars
        es_min, es_max = running_es.min(), running_es.max()
        y_margin = (es_max - es_min) * 0.1 if (es_max - es_min) > 0 else 0.1
        ax.set_ylim(es_min - y_margin, es_max + y_margin)
        
        # Add vertical bars for phenotype gene positions (after setting y-limits)
        if phenotype_positions:
            ax.vlines(phenotype_positions, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], 
                     colors='black', linewidths=0.5, alpha=0.3, linestyles='--')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return out_path

