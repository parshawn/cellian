"""
Pathway and Enrichment Analysis Module

This module performs comprehensive analysis on predicted RNA and protein outputs:
1. Gene Set Enrichment Analysis (GSEA) on RNA
2. Pathway enrichment analysis (KEGG, Reactome, GO) on RNA and protein
3. Comparison between perturbation and control
4. Pathway identification from perturbation

For drug perturbations, control cells are included in the predictions (20% control, 80% perturbed),
so GSEA and pathway enrichment can be performed by filtering control cells from predictions.

Usage:
    python pathway_analysis.py \
        --rna-predictions /path/to/rna_predictions.h5ad \
        --protein-predictions /path/to/protein_predictions.h5ad \
        --control-rna /path/to/control_rna.h5ad \
        --control-protein /path/to/control_protein.h5ad \
        --target-gene ACTB \
        --output-dir /path/to/output
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Try importing enrichment analysis libraries
try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    print("Warning: gseapy not available. GSEA analysis will be skipped.")

# Try importing goatools - check multiple possible import methods
GOATOOLS_AVAILABLE = False
try:
    import goatools
    from goatools import GOEnrichmentStudy
    from goatools.obo_parser import GODag
    GOATOOLS_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        # Try alternative import
        from goatools.go_enrichment import GOEnrichmentStudy
        from goatools.obo_parser import GODag
        GOATOOLS_AVAILABLE = True
    except (ImportError, AttributeError):
        GOATOOLS_AVAILABLE = False
        print("Warning: goatools not available. GO enrichment will use alternative methods.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available. PPI network analysis will be skipped.")



def calculate_correlation(rna_data, protein_data, method='pearson'):
    """
    Calculate correlation between RNA and protein predictions.
    
    Args:
        rna_data: RNA expression data (cells × genes) as DataFrame or array
        protein_data: Protein expression data (cells × proteins) as DataFrame or array
        method: Correlation method ('pearson' or 'spearman')
    
    Returns:
        Dict with correlation metrics
    """
    print(f"\n{'='*70}")
    print("CORRELATION ANALYSIS: RNA vs PROTEIN")
    print(f"{'='*70}")
    
    # Convert to DataFrame if needed
    if isinstance(rna_data, ad.AnnData):
        rna_df = pd.DataFrame(rna_data.X, index=rna_data.obs_names, columns=rna_data.var_names)
    elif isinstance(rna_data, pd.DataFrame):
        rna_df = rna_data
    else:
        rna_df = pd.DataFrame(rna_data)
    
    if isinstance(protein_data, ad.AnnData):
        protein_df = pd.DataFrame(protein_data.X, index=protein_data.obs_names, columns=protein_data.var_names)
    elif isinstance(protein_data, pd.DataFrame):
        protein_df = protein_data
    else:
        protein_df = pd.DataFrame(protein_data)
    
    # Get common genes (RNA genes that match protein names)
    common_genes = set(rna_df.columns).intersection(set(protein_df.columns))
    
    if len(common_genes) == 0:
        print("  Warning: No common genes between RNA and protein data")
        return {'correlation': np.nan, 'pvalue': np.nan, 'n_common_genes': 0}
    
    print(f"  Found {len(common_genes)} common genes/proteins")
    
    # Calculate correlation for each common gene
    correlations = []
    pvalues = []
    gene_names = []
    
    for gene in common_genes:
        if gene in rna_df.columns and gene in protein_df.columns:
            # Align cells
            common_cells = rna_df.index.intersection(protein_df.index)
            if len(common_cells) > 1:
                rna_values = rna_df.loc[common_cells, gene].values
                protein_values = protein_df.loc[common_cells, gene].values
                
                # Remove NaN
                mask = ~(np.isnan(rna_values) | np.isnan(protein_values))
                if mask.sum() > 1:
                    rna_clean = rna_values[mask]
                    protein_clean = protein_values[mask]
                    
                    if method == 'pearson':
                        corr, pval = pearsonr(rna_clean, protein_clean)
                    else:
                        corr, pval = spearmanr(rna_clean, protein_clean)
                    
                    correlations.append(corr)
                    pvalues.append(pval)
                    gene_names.append(gene)
    
    if len(correlations) == 0:
        print("  Warning: No valid correlations calculated")
        return {'correlation': np.nan, 'pvalue': np.nan, 'n_common_genes': 0}
    
    correlations = np.array(correlations)
    pvalues = np.array(pvalues)
    
    # Calculate summary statistics
    mean_corr = np.mean(correlations)
    median_corr = np.median(correlations)
    std_corr = np.std(correlations)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'gene': gene_names,
        'correlation': correlations,
        'pvalue': pvalues
    })
    results_df = results_df.sort_values('correlation', ascending=False)
    
    print(f"\n  Correlation Summary ({method}):")
    print(f"    Mean correlation: {mean_corr:.4f}")
    print(f"    Median correlation: {median_corr:.4f}")
    print(f"    Std correlation: {std_corr:.4f}")
    print(f"    Number of significant correlations (p<0.05): {(pvalues < 0.05).sum()}")
    print(f"    Number of strong correlations (|r|>0.5): {(np.abs(correlations) > 0.5).sum()}")
    
    # Top correlations
    print(f"\n  Top 10 positive correlations:")
    for _, row in results_df.head(10).iterrows():
        print(f"    {row['gene']}: {row['correlation']:.4f} (p={row['pvalue']:.4e})")
    
    return {
        'correlation': mean_corr,
        'median_correlation': median_corr,
        'std_correlation': std_corr,
        'pvalue': np.mean(pvalues),
        'n_common_genes': len(correlations),
        'results_df': results_df,
        'method': method
    }


def filter_control_data(control_data, control_label='non-targeting'):
    """
    Filter control data to get control cells.
    
    Args:
        control_data: Control expression data (AnnData)
        control_label: Label for control cells (default: 'non-targeting')
    
    Returns:
        Filtered control data
    """
    if isinstance(control_data, ad.AnnData):
        # Check for target_gene column first, then perturbation
        if 'target_gene' in control_data.obs.columns:
            mask = control_data.obs['target_gene'] == control_label
            if mask.sum() > 0:
                return control_data[mask].copy()
        elif 'perturbation' in control_data.obs.columns:
            mask = control_data.obs['perturbation'] == control_label
            if mask.sum() > 0:
                return control_data[mask].copy()
        # If no matching column or no control cells found, return all data
        return control_data
    return control_data


def calculate_differential_expression(pred_data, control_data, data_type='RNA', control_label='non-targeting'):
    """
    Calculate differential expression between perturbation and control.
    
    Args:
        pred_data: Predicted expression data (perturbation)
        control_data: Control expression data
        data_type: Type of data ('RNA' or 'protein')
        control_label: Label for control cells (default: 'non-targeting')
    
    Returns:
        DataFrame with differential expression results
    """
    print(f"\n{'='*70}")
    print(f"DIFFERENTIAL EXPRESSION ANALYSIS: {data_type}")
    print(f"{'='*70}")
    
    # Filter control data to get actual control cells
    control_data_filtered = filter_control_data(control_data, control_label=control_label)
    
    # Convert to DataFrame if needed
    if isinstance(pred_data, ad.AnnData):
        pred_df = pd.DataFrame(pred_data.X, index=pred_data.obs_names, columns=pred_data.var_names)
    elif isinstance(pred_data, pd.DataFrame):
        pred_df = pred_data
    else:
        pred_df = pd.DataFrame(pred_data)
    
    if isinstance(control_data_filtered, ad.AnnData):
        control_df = pd.DataFrame(control_data_filtered.X, index=control_data_filtered.obs_names, columns=control_data_filtered.var_names)
    elif isinstance(control_data_filtered, pd.DataFrame):
        control_df = control_data_filtered
    else:
        control_df = pd.DataFrame(control_data_filtered)
    
    # Get common genes
    # For proteins: Use ALL predicted proteins, not just those in control
    # Control will be extended with zeros/mean for missing proteins
    if data_type == 'protein':
        # Use ALL proteins from predictions
        common_genes = set(pred_df.columns)
        print(f"  Using ALL {len(common_genes)} predicted proteins for DEA")
        
        # Extend control_df to include all predicted proteins (fill missing with mean/median of control)
        missing_proteins = common_genes - set(control_df.columns)
        if len(missing_proteins) > 0:
            print(f"  Extending control data with {len(missing_proteins)} additional proteins (using mean values)")
            for protein in missing_proteins:
                # Use mean of all control proteins as baseline for missing proteins
                control_mean = control_df.mean(axis=1).values if len(control_df.columns) > 0 else np.zeros(len(control_df))
                control_df[protein] = control_mean
    else:
        # For RNA: Use intersection as before
        common_genes = set(pred_df.columns).intersection(set(control_df.columns))
        print(f"  Found {len(common_genes)} common genes/proteins")
    
    # Calculate differential expression
    results = []
    
    for gene in common_genes:
        if gene in pred_df.columns and gene in control_df.columns:
            pred_values = pred_df[gene].values
            control_values = control_df[gene].values
            
            # Remove NaN
            pred_clean = pred_values[~np.isnan(pred_values)]
            control_clean = control_values[~np.isnan(control_values)]
            
            if len(pred_clean) > 0 and len(control_clean) > 0:
                # Calculate fold change
                pred_mean = np.mean(pred_clean)
                control_mean = np.mean(control_clean)
                
                if control_mean != 0:
                    log2fc = np.log2((pred_mean + 1e-10) / (control_mean + 1e-10))
                else:
                    log2fc = np.nan
                
                # Calculate t-test
                try:
                    t_stat, pval = stats.ttest_ind(pred_clean, control_clean)
                except:
                    t_stat, pval = np.nan, np.nan
                
                results.append({
                    'gene': gene,
                    'pred_mean': pred_mean,
                    'control_mean': control_mean,
                    'log2fc': log2fc,
                    't_statistic': t_stat,
                    'pvalue': pval,
                    'abs_log2fc': np.abs(log2fc)
                })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("  Warning: No differential expression results calculated")
        return pd.DataFrame()
    
    # Sort by absolute log2 fold change
    results_df = results_df.sort_values('abs_log2fc', ascending=False)
    
    # Calculate adjusted p-values (Bonferroni correction)
    try:
        from statsmodels.stats.multitest import multipletests
        _, results_df['pvalue_adj'], _, _ = multipletests(
            results_df['pvalue'].fillna(1),
            method='bonferroni'
        )
    except ImportError:
        # If statsmodels not available, use simple Bonferroni correction
        n_tests = len(results_df)
        results_df['pvalue_adj'] = results_df['pvalue'] * n_tests
        results_df['pvalue_adj'] = results_df['pvalue_adj'].clip(upper=1.0)
    except Exception:
        results_df['pvalue_adj'] = results_df['pvalue']
    
    # Filter significant results using p-value only (not adjusted p-value, no log2FC threshold)
    significant = results_df[results_df['pvalue'] < 0.05]
    
    print(f"\n  Differential Expression Summary:")
    print(f"    Total genes/proteins: {len(results_df)}")
    print(f"    Significant (p<0.05): {len(significant)}")
    print(f"    Upregulated: {(significant['log2fc'] > 0).sum()}")
    print(f"    Downregulated: {(significant['log2fc'] < 0).sum()}")
    
    # Top differential genes
    print(f"\n  Top 10 upregulated:")
    for _, row in significant.nlargest(10, 'log2fc').iterrows():
        print(f"    {row['gene']}: log2FC={row['log2fc']:.4f}, p={row['pvalue']:.4e}")
    
    print(f"\n  Top 10 downregulated:")
    for _, row in significant.nsmallest(10, 'log2fc').iterrows():
        print(f"    {row['gene']}: log2FC={row['log2fc']:.4f}, p={row['pvalue']:.4e}")
    
    return results_df


def run_gsea(rna_data, control_data, gene_sets='KEGG_2021_Human', output_dir=None, control_label='non-targeting'):
    """
    Run Gene Set Enrichment Analysis (GSEA) on RNA data.
    
    Args:
        rna_data: Predicted RNA expression data
        control_data: Control RNA expression data
        gene_sets: Gene sets to use (default: 'KEGG_2021_Human')
        output_dir: Directory to save results
        control_label: Label for control cells (default: 'non-targeting')
    
    Returns:
        GSEA results DataFrame
    """
    if not GSEAPY_AVAILABLE:
        print("\n  Warning: gseapy not available. Skipping GSEA analysis.")
        return None
    
    print(f"\n{'='*70}")
    print("GENE SET ENRICHMENT ANALYSIS (GSEA): RNA")
    print(f"{'='*70}")
    
    # Calculate differential expression first
    de_results = calculate_differential_expression(rna_data, control_data, data_type='RNA', control_label=control_label)
    
    if len(de_results) == 0:
        print("  Warning: No differential expression results for GSEA")
        return None
    
    # Prepare data for GSEA
    # GSEA needs a ranked list of genes (by log2FC or other metric)
    ranked_genes = de_results.set_index('gene')['log2fc'].sort_values(ascending=False)
    ranked_genes = ranked_genes.dropna()
    
    if len(ranked_genes) == 0:
        print("  Warning: No valid ranked genes for GSEA")
        return None
    
    print(f"  Running GSEA with {len(ranked_genes)} genes...")
    print(f"  Gene sets: {gene_sets}")
    
    try:
        # Run GSEA
        print(f"  Running GSEA prerank analysis...")
        gsea_results = gp.prerank(
            rnk=ranked_genes,
            gene_sets=gene_sets,
            processes=4,
            permutation_num=100,
            outdir=output_dir if output_dir else None,
            format='png',
            seed=42,
            verbose=False
        )
        
        # Get enriched pathways - gseapy returns a dict with 'res' key
        if isinstance(gsea_results, dict):
            if 'res' in gsea_results:
                enriched_pathways = gsea_results['res']
            elif 'res2d' in gsea_results:
                enriched_pathways = gsea_results['res2d']
            else:
                # Try to get the first value that looks like a DataFrame
                for key, value in gsea_results.items():
                    if isinstance(value, pd.DataFrame):
                        enriched_pathways = value
                        break
                else:
                    print("  Warning: GSEA results not in expected format")
                    return None
        elif hasattr(gsea_results, 'res2d'):
            enriched_pathways = gsea_results.res2d
        elif isinstance(gsea_results, pd.DataFrame):
            enriched_pathways = gsea_results
        else:
            print(f"  Warning: GSEA results type: {type(gsea_results)}")
            return None
        
        # Sort by p-value (not FDR)
        # Find p-value column
        pval_col = None
        for col in ['NOM p-val', 'pval', 'p-value', 'P-value', 'p_value']:
            if col in enriched_pathways.columns:
                pval_col = col
                break
        
        if pval_col:
            enriched_pathways = enriched_pathways.sort_values(pval_col, ascending=True)
        elif 'NES' in enriched_pathways.columns:
            enriched_pathways = enriched_pathways.sort_values('NES', ascending=False)
        
        print(f"\n  GSEA Results:")
        print(f"    Total pathways tested: {len(enriched_pathways)}")
        if pval_col:
            significant_count = (enriched_pathways[pval_col] < 0.05).sum()
            print(f"    Significantly enriched (p<0.05): {significant_count}")
        else:
            print(f"    Warning: No p-value column found. Available columns: {enriched_pathways.columns.tolist()}")
        
        # Top enriched pathways
        print(f"\n  Top 10 enriched pathways:")
        for idx, (pathway_idx, row) in enumerate(enriched_pathways.head(10).iterrows()):
            # Get pathway name from appropriate column
            pathway_name = 'N/A'
            for name_col in ['Term', 'NAME', 'pathway', 'Pathway', 'Name', 'term']:
                if name_col in row:
                    pathway_name = row[name_col]
                    break
            if pathway_name == 'N/A':
                # If no name column found, use index (might be numeric)
                pathway_name = str(pathway_idx) if not isinstance(pathway_idx, str) else pathway_idx
            
            nes = row['NES'] if 'NES' in row else 'N/A'
            pval = row[pval_col] if pval_col and pval_col in row else 'N/A'
            # Format values properly - can't use conditional in f-string format specifier
            if isinstance(nes, (int, float)) and not pd.isna(nes):
                nes_str = f"{nes:.4f}"
            else:
                nes_str = str(nes)
            if isinstance(pval, (int, float)) and not pd.isna(pval):
                pval_str = f"{pval:.4e}"
            else:
                pval_str = str(pval)
            print(f"    {pathway_name}: NES={nes_str}, p={pval_str}")
        
        return enriched_pathways
    
    except Exception as e:
        print(f"  Error running GSEA: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_pathway_enrichment(protein_data, control_data, database='KEGG', output_dir=None, control_label='non-targeting', data_type='protein'):
    """
    Run pathway enrichment analysis on expression data (RNA or protein).
    
    Args:
        protein_data: Predicted expression data (RNA or protein)
        control_data: Control expression data (RNA or protein)
        database: Database to use ('KEGG', 'Reactome', 'GO')
        output_dir: Directory to save results
        control_label: Label for control cells (default: 'non-targeting')
        data_type: Type of data ('RNA' or 'protein') - for display purposes only
    
    Returns:
        Enrichment results DataFrame
    """
    if not GSEAPY_AVAILABLE:
        print(f"\n  Warning: gseapy not available. Skipping {database} enrichment analysis.")
        return None
    
    print(f"\n{'='*70}")
    print(f"PATHWAY ENRICHMENT ANALYSIS: {data_type.upper()} ({database})")
    print(f"{'='*70}")
    
    # Calculate differential expression first
    de_results = calculate_differential_expression(protein_data, control_data, data_type=data_type, control_label=control_label)
    
    if len(de_results) == 0:
        print(f"  Warning: No differential expression results for {database} enrichment")
        return None
    
    # Get significant genes/proteins using p-value only (no log2FC threshold)
    significant_genes = de_results[de_results['pvalue'] < 0.05]
    
    if len(significant_genes) == 0:
        print(f"  Warning: No significant {data_type} genes/proteins for {database} enrichment")
        return None
    
    # Prepare gene list
    gene_list = significant_genes['gene'].tolist()
    print(f"  Running {database} enrichment with {len(gene_list)} significant {data_type} genes/proteins...")
    
    # Map database names
    database_map = {
        'KEGG': 'KEGG_2021_Human',
        'Reactome': 'Reactome_2022',
        'GO': 'GO_Biological_Process_2021'
    }
    
    gene_sets = database_map.get(database, database)
    
    try:
        # Run enrichment analysis
        # Use cutoff=1.0 to get all results, then filter manually (prevents plotting errors)
        print(f"  Running {database} enrichment analysis...")
        try:
            # Don't pass format parameter - let it default, but set outdir=None to prevent file writing
            # This avoids the "replace() argument 2 must be str, not None" error
            enr_results = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gene_sets,
                organism='Human',
                outdir=None,  # Don't save plots/files to avoid errors
                verbose=False,
                cutoff=1.0  # Get all results, filter manually
            )
        except ValueError as e:
            # If plotting fails due to no enriched terms, try to get results anyway
            if "No enrich terms" in str(e) or "cutoff" in str(e).lower():
                print(f"  Note: No enriched terms at default cutoff - trying to get all results...")
                try:
                    enr_results = gp.enrichr(
                        gene_list=gene_list,
                        gene_sets=gene_sets,
                        organism='Human',
                        outdir=None,  # Don't save plots
                        verbose=False,
                        cutoff=1.0  # Get all results
                    )
                except Exception as e2:
                    print(f"  No enrichment results found for {database}")
                    print(f"  This means none of the {len(gene_list)} significant genes/proteins map to enriched pathways")
                    return None
            else:
                raise
        
        # Get enriched pathways - gseapy enrichr returns a dict
        enriched_pathways = None
        if isinstance(enr_results, dict):
            if 'res' in enr_results:
                enriched_pathways = enr_results['res']
            elif 'res2d' in enr_results:
                enriched_pathways = enr_results['res2d']
            else:
                # Try to get the first value that looks like a DataFrame
                for key, value in enr_results.items():
                    if isinstance(value, pd.DataFrame):
                        enriched_pathways = value
                        break
                if enriched_pathways is None:
                    print(f"  Warning: {database} enrichment results not in expected format")
                    return None
        elif hasattr(enr_results, 'res2d'):
            enriched_pathways = enr_results.res2d
        elif isinstance(enr_results, pd.DataFrame):
            enriched_pathways = enr_results
        else:
            print(f"  Warning: {database} enrichment results type: {type(enr_results)}")
            return None
        
        # Check if we have any results
        if enriched_pathways is None or len(enriched_pathways) == 0:
            print(f"  No enrichment results found for {database}")
            print(f"  This means none of the {len(gene_list)} significant genes/proteins map to enriched pathways")
            return None
        
        # Sort by p-value (use unadjusted for filtering, as requested)
        # Check for both adjusted and unadjusted p-value columns
        has_adjusted = 'Adjusted P-value' in enriched_pathways.columns
        has_unadjusted = 'P-value' in enriched_pathways.columns
        
        # Use unadjusted p-value for filtering (as requested by user)
        filter_pval_col = 'P-value' if has_unadjusted else 'Adjusted P-value'
        # Use adjusted p-value for sorting/display if available, otherwise unadjusted
        sort_pval_col = 'Adjusted P-value' if has_adjusted else 'P-value'
        
        if filter_pval_col in enriched_pathways.columns:
            # Sort by adjusted p-value if available, otherwise unadjusted
            if sort_pval_col in enriched_pathways.columns and sort_pval_col != filter_pval_col:
                enriched_pathways = enriched_pathways.sort_values(sort_pval_col)
            else:
                enriched_pathways = enriched_pathways.sort_values(filter_pval_col)
            
            print(f"\n  {database} Enrichment Results:")
            print(f"    Total pathways tested: {len(enriched_pathways)}")
            # Filter by unadjusted p-value < 0.05 (as requested)
            significant_count = (enriched_pathways[filter_pval_col] < 0.05).sum()
            print(f"    Significantly enriched (P-value<0.05): {significant_count}")
            
            # Show results or message if no significant pathways
            if significant_count == 0:
                print(f"  No significant pathways found (P-value<0.05)")
                # Still return the results (with non-significant pathways) for reference
                return enriched_pathways
            else:
                # Top enriched pathways - filter by unadjusted p-value
                print(f"\n  Top 10 enriched pathways:")
                significant_pathways = enriched_pathways[enriched_pathways[filter_pval_col] < 0.05].head(10)
                if len(significant_pathways) == 0:
                    # Fallback to top 10 overall if none meet cutoff
                    significant_pathways = enriched_pathways.head(10)
                
                for idx, (_, row) in enumerate(significant_pathways.iterrows()):
                    term = row['Term'] if 'Term' in row else row.index[0] if len(row) > 0 else 'N/A'
                    # Show unadjusted p-value for filtering
                    pval = row[filter_pval_col] if filter_pval_col in row else 'N/A'
                    # Also show adjusted p-value if available
                    adj_pval = row['Adjusted P-value'] if 'Adjusted P-value' in row else None
                    odds_ratio = row['Odds Ratio'] if 'Odds Ratio' in row else 'N/A'
                    # Format values properly - can't use conditional in f-string format specifier
                    if isinstance(pval, (int, float)) and not pd.isna(pval):
                        pval_str = f"{pval:.4e}"
                    else:
                        pval_str = str(pval)
                    if adj_pval is not None and isinstance(adj_pval, (int, float)) and not pd.isna(adj_pval):
                        adj_pval_str = f", Adjusted P={adj_pval:.4e}"
                    else:
                        adj_pval_str = ""
                    if isinstance(odds_ratio, (int, float)) and not pd.isna(odds_ratio):
                        odds_ratio_str = f"{odds_ratio:.4f}"
                    else:
                        odds_ratio_str = str(odds_ratio)
                    print(f"    {term}: P-value={pval_str}{adj_pval_str}, Odds Ratio={odds_ratio_str}")
                
                return enriched_pathways
        else:
            print(f"  Warning: {database} enrichment results missing p-value column. Available columns: {enriched_pathways.columns.tolist()}")
            return enriched_pathways
    
    except Exception as e:
        error_msg = str(e)
        # Check if this is the "no enriched terms" error
        if "No enrich terms" in error_msg or "cutoff" in error_msg.lower():
            print(f"  No significant pathways found for {database}")
            print(f"  This means none of the {len(gene_list)} significant genes/proteins map to enriched pathways (p-value < 0.05)")
            return None
        else:
            print(f"  Error running {database} enrichment: {error_msg}")
            import traceback
            traceback.print_exc()
            return None


def analyze_pathway_network(enrichment_results, output_dir=None):
    """
    Analyze pathway networks from enrichment results.
    
    Args:
        enrichment_results: Enrichment results DataFrame
        output_dir: Directory to save results
    
    Returns:
        Network analysis results
    """
    if not NETWORKX_AVAILABLE:
        print("\n  Warning: networkx not available. Skipping network analysis.")
        return None
    
    print(f"\n{'='*70}")
    print("PATHWAY NETWORK ANALYSIS")
    print(f"{'='*70}")
    
    if enrichment_results is None or len(enrichment_results) == 0:
        print("  Warning: No enrichment results for network analysis")
        return None
    
    # Create pathway network (simplified - would need pathway interaction data)
    print("  Network analysis would require pathway interaction data")
    print("  This is a placeholder for future implementation")
    
    return None


def comprehensive_analysis(
    rna_predictions_path,
    protein_predictions_path,
    control_rna_path=None,
    control_protein_path=None,
    target_gene=None,
    output_dir=None,
    control_label='non-targeting'
):
    """
    Run comprehensive analysis on predictions.
    
    Args:
        rna_predictions_path: Path to RNA predictions h5ad
        protein_predictions_path: Path to protein predictions h5ad
        control_rna_path: Path to control RNA h5ad (optional, for comparison)
        control_protein_path: Path to control protein h5ad (optional, for comparison)
        target_gene: Target gene/drug name (optional, for file naming)
        output_dir: Output directory for results
        control_label: Label for control cells (default: 'non-targeting')
    """
    print("="*70)
    print("COMPREHENSIVE PATHWAY ANALYSIS")
    print("="*70)
    if target_gene:
        print(f"\nTarget: {target_gene}")
    if control_rna_path and control_protein_path:
        print(f"Control label: {control_label}")
        print(f"Note: Using external control data files")
    else:
        print(f"Note: No external control data provided - control cells will be extracted from predictions")
        print(f"      (looking for control cells based on 'drugname_drugconc' or 'target_gene' columns)")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data...")
    rna_pred = ad.read_h5ad(rna_predictions_path)
    protein_pred = ad.read_h5ad(protein_predictions_path)
    
    print(f"  RNA predictions: {rna_pred.shape}")
    print(f"  Protein predictions: {protein_pred.shape}")
    
    # Load control data if provided
    control_rna_filtered = None
    control_protein_filtered = None
    
    if control_rna_path and control_protein_path:
        control_rna = ad.read_h5ad(control_rna_path)
        control_protein = ad.read_h5ad(control_protein_path)
        print(f"  Control RNA: {control_rna.shape}")
        print(f"  Control protein: {control_protein.shape}")
        
        # Filter control data
        control_rna_filtered = filter_control_data(control_rna, control_label=control_label)
        control_protein_filtered = filter_control_data(control_protein, control_label=control_label)
        
        print(f"  Control RNA (filtered): {control_rna_filtered.shape}")
        print(f"  Control protein (filtered): {control_protein_filtered.shape}")
    
    results = {}
    
    # 1. Differential expression - RNA
    # For drug perturbations, control cells are in the predictions (drugname_drugconc == "[('DMSO_TF', 0.0, 'uM')]")
    # So we need to extract control from predictions if no external control provided
    if control_rna_filtered is not None:
        print(f"\n{'='*70}")
        print("STEP 1: DIFFERENTIAL EXPRESSION - RNA")
        print(f"{'='*70}")
        de_rna = calculate_differential_expression(rna_pred, control_rna_filtered, data_type='RNA', control_label=control_label)
        results['differential_rna'] = de_rna
        
        if len(de_rna) > 0:
            filename = f'differential_expression_rna_{target_gene}.csv' if target_gene else 'differential_expression_rna.csv'
            de_rna.to_csv(
                os.path.join(output_dir, filename),
                index=False
            )
    else:
        # No external control data - try to extract control from predictions
        # For drug perturbations: drugname_drugconc == "[('DMSO_TF', 0.0, 'uM')]"
        # For gene perturbations with 80-20 split: target_gene == 'non-targeting'
        control_mask = None
        control_label = None
        
        if 'drugname_drugconc' in rna_pred.obs.columns:
            control_label = "[('DMSO_TF', 0.0, 'uM')]"
            control_mask = rna_pred.obs['drugname_drugconc'] == control_label
        elif 'target_gene' in rna_pred.obs.columns:
            # For gene perturbations with 80-20 split
            control_label = 'non-targeting'
            control_mask = rna_pred.obs['target_gene'] == control_label
        
        if control_mask is not None and control_mask.sum() > 0:
                print(f"\n{'='*70}")
                print("STEP 1: DIFFERENTIAL EXPRESSION - RNA (Using control cells from predictions)")
                print(f"{'='*70}")
                print(f"  Found {control_mask.sum()} control cells in predictions")
                
                # Split predictions into perturbed and control
                rna_perturbed = rna_pred[~control_mask].copy()
                control_rna_from_pred = rna_pred[control_mask].copy()
                
                de_rna = calculate_differential_expression(rna_perturbed, control_rna_from_pred, data_type='RNA', control_label=control_label)
                results['differential_rna'] = de_rna
                
                if len(de_rna) > 0:
                    filename = f'differential_expression_rna_{target_gene}.csv' if target_gene else 'differential_expression_rna.csv'
                    de_rna.to_csv(
                        os.path.join(output_dir, filename),
                        index=False
                    )
        else:
            print(f"\n{'='*70}")
            print("STEP 1: DIFFERENTIAL EXPRESSION - RNA (SKIPPED)")
            print(f"{'='*70}")
            if control_mask is None:
                print("  No control column found in predictions (need 'drugname_drugconc' or 'target_gene')")
            else:
                print(f"  No control cells found in predictions (looked for: {control_label})")
                print(f"  Available values: {list(rna_pred.obs.get('drugname_drugconc' if 'drugname_drugconc' in rna_pred.obs.columns else 'target_gene', pd.Series()).unique()[:10])}")
            results['differential_rna'] = None
    
    # 3. Differential expression - Protein
    # For drug perturbations, control cells are in the predictions (drugname_drugconc == "[('DMSO_TF', 0.0, 'uM')]")
    # So we need to extract control from predictions if no external control provided
    if control_protein_filtered is not None:
        print(f"\n{'='*70}")
        print("STEP 2: DIFFERENTIAL EXPRESSION - PROTEIN")
        print(f"{'='*70}")
        de_protein = calculate_differential_expression(protein_pred, control_protein_filtered, data_type='protein', control_label=control_label)
        results['differential_protein'] = de_protein
        
        if len(de_protein) > 0:
            filename = f'differential_expression_protein_{target_gene}.csv' if target_gene else 'differential_expression_protein.csv'
            de_protein.to_csv(
                os.path.join(output_dir, filename),
                index=False
            )
    else:
        # No external control data - try to extract control from predictions
        # For drug perturbations: drugname_drugconc == "[('DMSO_TF', 0.0, 'uM')]"
        # For gene perturbations with 80-20 split: target_gene == 'non-targeting'
        control_mask = None
        control_label = None
        
        if 'drugname_drugconc' in protein_pred.obs.columns:
            control_label = "[('DMSO_TF', 0.0, 'uM')]"
            control_mask = protein_pred.obs['drugname_drugconc'] == control_label
        elif 'target_gene' in protein_pred.obs.columns:
            # For gene perturbations with 80-20 split
            control_label = 'non-targeting'
            control_mask = protein_pred.obs['target_gene'] == control_label
        
        if control_mask is not None and control_mask.sum() > 0:
                print(f"\n{'='*70}")
                print("STEP 2: DIFFERENTIAL EXPRESSION - PROTEIN (Using control cells from predictions)")
                print(f"{'='*70}")
                print(f"  Found {control_mask.sum()} control cells in predictions")
                
                # Split predictions into perturbed and control
                protein_perturbed = protein_pred[~control_mask].copy()
                control_protein_from_pred = protein_pred[control_mask].copy()
                
                de_protein = calculate_differential_expression(protein_perturbed, control_protein_from_pred, data_type='protein', control_label=control_label)
                results['differential_protein'] = de_protein
                
                if len(de_protein) > 0:
                    filename = f'differential_expression_protein_{target_gene}.csv' if target_gene else 'differential_expression_protein.csv'
                    de_protein.to_csv(
                        os.path.join(output_dir, filename),
                        index=False
                    )
        else:
            print(f"\n{'='*70}")
            print("STEP 2: DIFFERENTIAL EXPRESSION - PROTEIN (SKIPPED)")
            print(f"{'='*70}")
            if control_mask is None:
                print("  No control column found in predictions (need 'drugname_drugconc' or 'target_gene')")
            else:
                print(f"  No control cells found in predictions (looked for: {control_label})")
                print(f"  Available values: {list(protein_pred.obs.get('drugname_drugconc' if 'drugname_drugconc' in protein_pred.obs.columns else 'target_gene', pd.Series()).unique()[:10])}")
            results['differential_protein'] = None
    
    # 4. GSEA - RNA
    # For drug perturbations, control cells are in the predictions (drugname_drugconc == "[('DMSO_TF', 0.0, 'uM')]")
    # So we need to extract control from predictions if no external control provided
    if control_rna_filtered is not None:
        print(f"\n{'='*70}")
        print("STEP 3: GSEA - RNA")
        print(f"{'='*70}")
        gsea_dir = os.path.join(output_dir, 'gsea_rna')
        os.makedirs(gsea_dir, exist_ok=True)
        try:
            gsea_results = run_gsea(rna_pred, control_rna_filtered, gene_sets='KEGG_2021_Human', output_dir=gsea_dir, control_label=control_label)
            results['gsea_rna'] = gsea_results
            
            if gsea_results is not None and len(gsea_results) > 0:
                filename = f'gsea_rna_{target_gene}.csv' if target_gene else 'gsea_rna.csv'
                gsea_results.to_csv(
                    os.path.join(output_dir, filename)
                )
        except Exception as e:
            print(f"  Warning: GSEA failed: {str(e)}")
            results['gsea_rna'] = None
    else:
        # No external control data - try to extract control from predictions
        # For drug perturbations: drugname_drugconc == "[('DMSO_TF', 0.0, 'uM')]"
        # For gene perturbations with 80-20 split: target_gene == 'non-targeting'
        control_mask = None
        control_label = None
        
        if 'drugname_drugconc' in rna_pred.obs.columns:
            control_label = "[('DMSO_TF', 0.0, 'uM')]"
            control_mask = rna_pred.obs['drugname_drugconc'] == control_label
        elif 'target_gene' in rna_pred.obs.columns:
            # For gene perturbations with 80-20 split
            control_label = 'non-targeting'
            control_mask = rna_pred.obs['target_gene'] == control_label
        
        if control_mask is not None and control_mask.sum() > 0:
                print(f"\n{'='*70}")
                print("STEP 3: GSEA - RNA (Using control cells from predictions)")
                print(f"{'='*70}")
                print(f"  Found {control_mask.sum()} control cells in predictions")
                
                # Split predictions into perturbed and control
                rna_perturbed = rna_pred[~control_mask].copy()
                control_rna_from_pred = rna_pred[control_mask].copy()
                
                gsea_dir = os.path.join(output_dir, 'gsea_rna')
                os.makedirs(gsea_dir, exist_ok=True)
                try:
                    gsea_results = run_gsea(rna_perturbed, control_rna_from_pred, gene_sets='KEGG_2021_Human', output_dir=gsea_dir, control_label=control_label)
                    results['gsea_rna'] = gsea_results
                    
                    if gsea_results is not None and len(gsea_results) > 0:
                        filename = f'gsea_rna_{target_gene}.csv' if target_gene else 'gsea_rna.csv'
                        gsea_results.to_csv(
                            os.path.join(output_dir, filename)
                        )
                except Exception as e:
                    print(f"  Warning: GSEA failed: {str(e)}")
                    results['gsea_rna'] = None
        else:
            print(f"\n{'='*70}")
            print("STEP 3: GSEA - RNA (SKIPPED)")
            print(f"{'='*70}")
            if control_mask is None:
                print("  No control column found in predictions (need 'drugname_drugconc' or 'target_gene')")
            else:
                print(f"  No control cells found in predictions (looked for: {control_label})")
                print(f"  Available values: {list(rna_pred.obs.get('drugname_drugconc' if 'drugname_drugconc' in rna_pred.obs.columns else 'target_gene', pd.Series()).unique()[:10])}")
            results['gsea_rna'] = None
    
    # 4. Pathway enrichment - RNA (KEGG)
    # Run KEGG/Reactome/GO ORA for RNA as well
    if control_rna_filtered is not None:
        print(f"\n{'='*70}")
        print("STEP 4a: PATHWAY ENRICHMENT - RNA (KEGG)")
        print(f"{'='*70}")
        kegg_dir_rna = os.path.join(output_dir, 'kegg_enrichment_rna')
        os.makedirs(kegg_dir_rna, exist_ok=True)
        try:
            kegg_results_rna = run_pathway_enrichment(rna_pred, control_rna_filtered, database='KEGG', output_dir=kegg_dir_rna, control_label=control_label, data_type='RNA')
            results['kegg_enrichment_rna'] = kegg_results_rna
            
            if kegg_results_rna is not None and len(kegg_results_rna) > 0:
                filename = f'kegg_enrichment_rna_{target_gene}.csv' if target_gene else 'kegg_enrichment_rna.csv'
                kegg_results_rna.to_csv(
                    os.path.join(output_dir, filename)
                )
        except Exception as e:
            print(f"  Warning: KEGG enrichment (RNA) failed: {str(e)}")
            results['kegg_enrichment_rna'] = None
    else:
        # No external control data - try to extract control from predictions
        control_mask = None
        control_label_local = None
        
        if 'drugname_drugconc' in rna_pred.obs.columns:
            control_label_local = "[('DMSO_TF', 0.0, 'uM')]"
            control_mask = rna_pred.obs['drugname_drugconc'] == control_label_local
        elif 'target_gene' in rna_pred.obs.columns:
            control_label_local = 'non-targeting'
            control_mask = rna_pred.obs['target_gene'] == control_label_local
        
        if control_mask is not None and control_mask.sum() > 0:
            print(f"\n{'='*70}")
            print("STEP 4a: PATHWAY ENRICHMENT - RNA (KEGG) (Using control cells from predictions)")
            print(f"{'='*70}")
            print(f"  Found {control_mask.sum()} control cells in predictions")
            
            rna_perturbed = rna_pred[~control_mask].copy()
            control_rna_from_pred = rna_pred[control_mask].copy()
            
            kegg_dir_rna = os.path.join(output_dir, 'kegg_enrichment_rna')
            os.makedirs(kegg_dir_rna, exist_ok=True)
            try:
                kegg_results_rna = run_pathway_enrichment(rna_perturbed, control_rna_from_pred, database='KEGG', output_dir=kegg_dir_rna, control_label=control_label_local, data_type='RNA')
                results['kegg_enrichment_rna'] = kegg_results_rna
                
                if kegg_results_rna is not None and len(kegg_results_rna) > 0:
                    filename = f'kegg_enrichment_rna_{target_gene}.csv' if target_gene else 'kegg_enrichment_rna.csv'
                    kegg_results_rna.to_csv(
                        os.path.join(output_dir, filename)
                    )
            except Exception as e:
                print(f"  Warning: KEGG enrichment (RNA) failed: {str(e)}")
                results['kegg_enrichment_rna'] = None
        else:
            print(f"\n{'='*70}")
            print("STEP 4a: PATHWAY ENRICHMENT - RNA (KEGG) (SKIPPED)")
            print(f"{'='*70}")
            if control_mask is None:
                print("  No control column found in predictions")
            else:
                print(f"  No control cells found in predictions")
            results['kegg_enrichment_rna'] = None
    
    # 4b. Pathway enrichment - RNA (Reactome)
    if control_rna_filtered is not None:
        print(f"\n{'='*70}")
        print("STEP 4b: PATHWAY ENRICHMENT - RNA (REACTOME)")
        print(f"{'='*70}")
        reactome_dir_rna = os.path.join(output_dir, 'reactome_enrichment_rna')
        os.makedirs(reactome_dir_rna, exist_ok=True)
        try:
            reactome_results_rna = run_pathway_enrichment(rna_pred, control_rna_filtered, database='Reactome', output_dir=reactome_dir_rna, control_label=control_label, data_type='RNA')
            results['reactome_enrichment_rna'] = reactome_results_rna
            
            if reactome_results_rna is not None and len(reactome_results_rna) > 0:
                filename = f'reactome_enrichment_rna_{target_gene}.csv' if target_gene else 'reactome_enrichment_rna.csv'
                reactome_results_rna.to_csv(
                    os.path.join(output_dir, filename)
                )
        except Exception as e:
            print(f"  Warning: Reactome enrichment (RNA) failed: {str(e)}")
            results['reactome_enrichment_rna'] = None
    else:
        control_mask = None
        control_label_local = None
        
        if 'drugname_drugconc' in rna_pred.obs.columns:
            control_label_local = "[('DMSO_TF', 0.0, 'uM')]"
            control_mask = rna_pred.obs['drugname_drugconc'] == control_label_local
        elif 'target_gene' in rna_pred.obs.columns:
            control_label_local = 'non-targeting'
            control_mask = rna_pred.obs['target_gene'] == control_label_local
        
        if control_mask is not None and control_mask.sum() > 0:
            print(f"\n{'='*70}")
            print("STEP 4b: PATHWAY ENRICHMENT - RNA (REACTOME) (Using control cells from predictions)")
            print(f"{'='*70}")
            print(f"  Found {control_mask.sum()} control cells in predictions")
            
            rna_perturbed = rna_pred[~control_mask].copy()
            control_rna_from_pred = rna_pred[control_mask].copy()
            
            reactome_dir_rna = os.path.join(output_dir, 'reactome_enrichment_rna')
            os.makedirs(reactome_dir_rna, exist_ok=True)
            try:
                reactome_results_rna = run_pathway_enrichment(rna_perturbed, control_rna_from_pred, database='Reactome', output_dir=reactome_dir_rna, control_label=control_label_local, data_type='RNA')
                results['reactome_enrichment_rna'] = reactome_results_rna
                
                if reactome_results_rna is not None and len(reactome_results_rna) > 0:
                    filename = f'reactome_enrichment_rna_{target_gene}.csv' if target_gene else 'reactome_enrichment_rna.csv'
                    reactome_results_rna.to_csv(
                        os.path.join(output_dir, filename)
                    )
            except Exception as e:
                print(f"  Warning: Reactome enrichment (RNA) failed: {str(e)}")
                results['reactome_enrichment_rna'] = None
        else:
            print(f"\n{'='*70}")
            print("STEP 4b: PATHWAY ENRICHMENT - RNA (REACTOME) (SKIPPED)")
            print(f"{'='*70}")
            results['reactome_enrichment_rna'] = None
    
    # 4c. Pathway enrichment - RNA (GO)
    if control_rna_filtered is not None:
        print(f"\n{'='*70}")
        print("STEP 4c: PATHWAY ENRICHMENT - RNA (GO)")
        print(f"{'='*70}")
        go_dir_rna = os.path.join(output_dir, 'go_enrichment_rna')
        os.makedirs(go_dir_rna, exist_ok=True)
        try:
            go_results_rna = run_pathway_enrichment(rna_pred, control_rna_filtered, database='GO', output_dir=go_dir_rna, control_label=control_label, data_type='RNA')
            results['go_enrichment_rna'] = go_results_rna
            
            if go_results_rna is not None and len(go_results_rna) > 0:
                filename = f'go_enrichment_rna_{target_gene}.csv' if target_gene else 'go_enrichment_rna.csv'
                go_results_rna.to_csv(
                    os.path.join(output_dir, filename)
                )
        except Exception as e:
            print(f"  Warning: GO enrichment (RNA) failed: {str(e)}")
            results['go_enrichment_rna'] = None
    else:
        control_mask = None
        control_label_local = None
        
        if 'drugname_drugconc' in rna_pred.obs.columns:
            control_label_local = "[('DMSO_TF', 0.0, 'uM')]"
            control_mask = rna_pred.obs['drugname_drugconc'] == control_label_local
        elif 'target_gene' in rna_pred.obs.columns:
            control_label_local = 'non-targeting'
            control_mask = rna_pred.obs['target_gene'] == control_label_local
        
        if control_mask is not None and control_mask.sum() > 0:
            print(f"\n{'='*70}")
            print("STEP 4c: PATHWAY ENRICHMENT - RNA (GO) (Using control cells from predictions)")
            print(f"{'='*70}")
            print(f"  Found {control_mask.sum()} control cells in predictions")
            
            rna_perturbed = rna_pred[~control_mask].copy()
            control_rna_from_pred = rna_pred[control_mask].copy()
            
            go_dir_rna = os.path.join(output_dir, 'go_enrichment_rna')
            os.makedirs(go_dir_rna, exist_ok=True)
            try:
                go_results_rna = run_pathway_enrichment(rna_perturbed, control_rna_from_pred, database='GO', output_dir=go_dir_rna, control_label=control_label_local, data_type='RNA')
                results['go_enrichment_rna'] = go_results_rna
                
                if go_results_rna is not None and len(go_results_rna) > 0:
                    filename = f'go_enrichment_rna_{target_gene}.csv' if target_gene else 'go_enrichment_rna.csv'
                    go_results_rna.to_csv(
                        os.path.join(output_dir, filename)
                    )
            except Exception as e:
                print(f"  Warning: GO enrichment (RNA) failed: {str(e)}")
                results['go_enrichment_rna'] = None
        else:
            print(f"\n{'='*70}")
            print("STEP 4c: PATHWAY ENRICHMENT - RNA (GO) (SKIPPED)")
            print(f"{'='*70}")
            results['go_enrichment_rna'] = None
    
    # 5. Pathway enrichment - Protein (KEGG, Reactome, GO)
    # For drug perturbations, control cells are in the predictions
    if control_protein_filtered is not None:
        print(f"\n{'='*70}")
        print("STEP 5: PATHWAY ENRICHMENT - PROTEIN (KEGG)")
        print(f"{'='*70}")
        kegg_dir = os.path.join(output_dir, 'kegg_enrichment')
        os.makedirs(kegg_dir, exist_ok=True)
        try:
            kegg_results = run_pathway_enrichment(protein_pred, control_protein_filtered, database='KEGG', output_dir=kegg_dir, control_label=control_label, data_type='protein')
            results['kegg_enrichment'] = kegg_results
            
            if kegg_results is not None and len(kegg_results) > 0:
                filename = f'kegg_enrichment_{target_gene}.csv' if target_gene else 'kegg_enrichment.csv'
                kegg_results.to_csv(
                    os.path.join(output_dir, filename)
                )
        except Exception as e:
            print(f"  Warning: KEGG enrichment failed: {str(e)}")
            results['kegg_enrichment'] = None
    else:
        # No external control data - try to extract control from predictions
        # For drug perturbations: drugname_drugconc == "[('DMSO_TF', 0.0, 'uM')]"
        # For gene perturbations with 80-20 split: target_gene == 'non-targeting'
        control_mask = None
        control_label = None
        
        if 'drugname_drugconc' in protein_pred.obs.columns:
            control_label = "[('DMSO_TF', 0.0, 'uM')]"
            control_mask = protein_pred.obs['drugname_drugconc'] == control_label
        elif 'target_gene' in protein_pred.obs.columns:
            # For gene perturbations with 80-20 split
            control_label = 'non-targeting'
            control_mask = protein_pred.obs['target_gene'] == control_label
        
        if control_mask is not None and control_mask.sum() > 0:
                print(f"\n{'='*70}")
                print("STEP 5: PATHWAY ENRICHMENT - PROTEIN (KEGG) (Using control cells from predictions)")
                print(f"{'='*70}")
                print(f"  Found {control_mask.sum()} control cells in predictions")
                
                # Split predictions into perturbed and control
                protein_perturbed = protein_pred[~control_mask].copy()
                control_protein_from_pred = protein_pred[control_mask].copy()
                
                kegg_dir = os.path.join(output_dir, 'kegg_enrichment')
                os.makedirs(kegg_dir, exist_ok=True)
                try:
                    kegg_results = run_pathway_enrichment(protein_perturbed, control_protein_from_pred, database='KEGG', output_dir=kegg_dir, control_label=control_label, data_type='protein')
                    results['kegg_enrichment'] = kegg_results
                    
                    if kegg_results is not None and len(kegg_results) > 0:
                        filename = f'kegg_enrichment_{target_gene}.csv' if target_gene else 'kegg_enrichment.csv'
                        kegg_results.to_csv(
                            os.path.join(output_dir, filename)
                        )
                except Exception as e:
                    print(f"  Warning: KEGG enrichment failed: {str(e)}")
                    results['kegg_enrichment'] = None
        else:
            print(f"\n{'='*70}")
            print("STEP 5: PATHWAY ENRICHMENT - PROTEIN (KEGG) (SKIPPED)")
            print(f"{'='*70}")
            if control_mask is None:
                print("  No control column found in predictions (need 'drugname_drugconc' or 'target_gene')")
            else:
                print(f"  No control cells found in predictions (looked for: {control_label})")
                print(f"  Available values: {list(protein_pred.obs.get('drugname_drugconc' if 'drugname_drugconc' in protein_pred.obs.columns else 'target_gene', pd.Series()).unique()[:10])}")
            results['kegg_enrichment'] = None
    
    # 6. Pathway enrichment - Protein (Reactome)
    # For drug perturbations, control cells are in the predictions
    if control_protein_filtered is not None:
        print(f"\n{'='*70}")
        print("STEP 6: PATHWAY ENRICHMENT - PROTEIN (REACTOME)")
        print(f"{'='*70}")
        reactome_dir = os.path.join(output_dir, 'reactome_enrichment')
        os.makedirs(reactome_dir, exist_ok=True)
        try:
            reactome_results = run_pathway_enrichment(protein_pred, control_protein_filtered, database='Reactome', output_dir=reactome_dir, control_label=control_label, data_type='protein')
            results['reactome_enrichment'] = reactome_results
            
            if reactome_results is not None and len(reactome_results) > 0:
                filename = f'reactome_enrichment_{target_gene}.csv' if target_gene else 'reactome_enrichment.csv'
                reactome_results.to_csv(
                    os.path.join(output_dir, filename)
                )
        except Exception as e:
            print(f"  Warning: Reactome enrichment failed: {str(e)}")
            results['reactome_enrichment'] = None
    else:
        # No external control data - try to extract control from predictions
        # For drug perturbations: drugname_drugconc == "[('DMSO_TF', 0.0, 'uM')]"
        # For gene perturbations with 80-20 split: target_gene == 'non-targeting'
        control_mask = None
        control_label = None
        
        if 'drugname_drugconc' in protein_pred.obs.columns:
            control_label = "[('DMSO_TF', 0.0, 'uM')]"
            control_mask = protein_pred.obs['drugname_drugconc'] == control_label
        elif 'target_gene' in protein_pred.obs.columns:
            # For gene perturbations with 80-20 split
            control_label = 'non-targeting'
            control_mask = protein_pred.obs['target_gene'] == control_label
        
        if control_mask is not None and control_mask.sum() > 0:
                print(f"\n{'='*70}")
                print("STEP 6: PATHWAY ENRICHMENT - PROTEIN (REACTOME) (Using control cells from predictions)")
                print(f"{'='*70}")
                print(f"  Found {control_mask.sum()} control cells in predictions")
                
                # Split predictions into perturbed and control
                protein_perturbed = protein_pred[~control_mask].copy()
                control_protein_from_pred = protein_pred[control_mask].copy()
                
                reactome_dir = os.path.join(output_dir, 'reactome_enrichment')
                os.makedirs(reactome_dir, exist_ok=True)
                try:
                    reactome_results = run_pathway_enrichment(protein_perturbed, control_protein_from_pred, database='Reactome', output_dir=reactome_dir, control_label=control_label, data_type='protein')
                    results['reactome_enrichment'] = reactome_results
                    
                    if reactome_results is not None and len(reactome_results) > 0:
                        filename = f'reactome_enrichment_{target_gene}.csv' if target_gene else 'reactome_enrichment.csv'
                        reactome_results.to_csv(
                            os.path.join(output_dir, filename)
                        )
                except Exception as e:
                    print(f"  Warning: Reactome enrichment failed: {str(e)}")
                    results['reactome_enrichment'] = None
        else:
            print(f"\n{'='*70}")
            print("STEP 6: PATHWAY ENRICHMENT - PROTEIN (REACTOME) (SKIPPED)")
            print(f"{'='*70}")
            if control_mask is None:
                print("  No control column found in predictions (need 'drugname_drugconc' or 'target_gene')")
            else:
                print(f"  No control cells found in predictions (looked for: {control_label})")
                print(f"  Available values: {list(protein_pred.obs.get('drugname_drugconc' if 'drugname_drugconc' in protein_pred.obs.columns else 'target_gene', pd.Series()).unique()[:10])}")
            results['reactome_enrichment'] = None
    
    # 7. Pathway enrichment - Protein (GO)
    # For drug perturbations, control cells are in the predictions
    if control_protein_filtered is not None:
        print(f"\n{'='*70}")
        print("STEP 7: PATHWAY ENRICHMENT - PROTEIN (GO)")
        print(f"{'='*70}")
        go_dir = os.path.join(output_dir, 'go_enrichment')
        os.makedirs(go_dir, exist_ok=True)
        try:
            go_results = run_pathway_enrichment(protein_pred, control_protein_filtered, database='GO', output_dir=go_dir, control_label=control_label, data_type='protein')
            results['go_enrichment'] = go_results
            
            if go_results is not None and len(go_results) > 0:
                filename = f'go_enrichment_{target_gene}.csv' if target_gene else 'go_enrichment.csv'
                go_results.to_csv(
                    os.path.join(output_dir, filename)
                )
        except Exception as e:
            print(f"  Warning: GO enrichment failed: {str(e)}")
            results['go_enrichment'] = None
    else:
        # No external control data - try to extract control from predictions
        # For drug perturbations: drugname_drugconc == "[('DMSO_TF', 0.0, 'uM')]"
        # For gene perturbations with 80-20 split: target_gene == 'non-targeting'
        control_mask = None
        control_label = None
        
        if 'drugname_drugconc' in protein_pred.obs.columns:
            control_label = "[('DMSO_TF', 0.0, 'uM')]"
            control_mask = protein_pred.obs['drugname_drugconc'] == control_label
        elif 'target_gene' in protein_pred.obs.columns:
            # For gene perturbations with 80-20 split
            control_label = 'non-targeting'
            control_mask = protein_pred.obs['target_gene'] == control_label
        
        if control_mask is not None and control_mask.sum() > 0:
                print(f"\n{'='*70}")
                print("STEP 7: PATHWAY ENRICHMENT - PROTEIN (GO) (Using control cells from predictions)")
                print(f"{'='*70}")
                print(f"  Found {control_mask.sum()} control cells in predictions")
                
                # Split predictions into perturbed and control
                protein_perturbed = protein_pred[~control_mask].copy()
                control_protein_from_pred = protein_pred[control_mask].copy()
                
                go_dir = os.path.join(output_dir, 'go_enrichment')
                os.makedirs(go_dir, exist_ok=True)
                try:
                    go_results = run_pathway_enrichment(protein_perturbed, control_protein_from_pred, database='GO', output_dir=go_dir, control_label=control_label, data_type='protein')
                    results['go_enrichment'] = go_results
                    
                    if go_results is not None and len(go_results) > 0:
                        filename = f'go_enrichment_{target_gene}.csv' if target_gene else 'go_enrichment.csv'
                        go_results.to_csv(
                            os.path.join(output_dir, filename)
                        )
                except Exception as e:
                    print(f"  Warning: GO enrichment failed: {str(e)}")
                    results['go_enrichment'] = None
        else:
            print(f"\n{'='*70}")
            print("STEP 7: PATHWAY ENRICHMENT - PROTEIN (GO) (SKIPPED)")
            print(f"{'='*70}")
            if control_mask is None:
                print("  No control column found in predictions (need 'drugname_drugconc' or 'target_gene')")
            else:
                print(f"  No control cells found in predictions (looked for: {control_label})")
                print(f"  Available values: {list(protein_pred.obs.get('drugname_drugconc' if 'drugname_drugconc' in protein_pred.obs.columns else 'target_gene', pd.Series()).unique()[:10])}")
            results['go_enrichment'] = None
    
    # Summary
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETED")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    if results.get('differential_rna') is not None:
        de_rna_filename = f'differential_expression_rna_{target_gene}.csv' if target_gene else 'differential_expression_rna.csv'
        print(f"  - Differential expression (RNA): {de_rna_filename}")
    if results.get('differential_protein') is not None:
        de_prot_filename = f'differential_expression_protein_{target_gene}.csv' if target_gene else 'differential_expression_protein.csv'
        print(f"  - Differential expression (Protein): {de_prot_filename}")
    if results.get('gsea_rna') is not None:
        gsea_filename = f'gsea_rna_{target_gene}.csv' if target_gene else 'gsea_rna.csv'
        print(f"  - GSEA (RNA): {gsea_filename}")
    if results.get('kegg_enrichment_rna') is not None:
        kegg_filename_rna = f'kegg_enrichment_rna_{target_gene}.csv' if target_gene else 'kegg_enrichment_rna.csv'
        print(f"  - KEGG enrichment (RNA): {kegg_filename_rna}")
    if results.get('reactome_enrichment_rna') is not None:
        reactome_filename_rna = f'reactome_enrichment_rna_{target_gene}.csv' if target_gene else 'reactome_enrichment_rna.csv'
        print(f"  - Reactome enrichment (RNA): {reactome_filename_rna}")
    if results.get('go_enrichment_rna') is not None:
        go_filename_rna = f'go_enrichment_rna_{target_gene}.csv' if target_gene else 'go_enrichment_rna.csv'
        print(f"  - GO enrichment (RNA): {go_filename_rna}")
    if results.get('kegg_enrichment') is not None:
        kegg_filename = f'kegg_enrichment_{target_gene}.csv' if target_gene else 'kegg_enrichment.csv'
        print(f"  - KEGG enrichment: {kegg_filename}")
    if results.get('reactome_enrichment') is not None:
        reactome_filename = f'reactome_enrichment_{target_gene}.csv' if target_gene else 'reactome_enrichment.csv'
        print(f"  - Reactome enrichment: {reactome_filename}")
    if results.get('go_enrichment') is not None:
        go_filename = f'go_enrichment_{target_gene}.csv' if target_gene else 'go_enrichment.csv'
        print(f"  - GO enrichment: {go_filename}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Pathway and Enrichment Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--rna-predictions', type=str, required=True,
                        help='Path to RNA predictions h5ad file')
    parser.add_argument('--protein-predictions', type=str, required=True,
                        help='Path to protein predictions h5ad file')
    parser.add_argument('--control-rna', type=str, required=True,
                        help='Path to control RNA h5ad file')
    parser.add_argument('--control-protein', type=str, required=True,
                        help='Path to control protein h5ad file')
    parser.add_argument('--target-gene', type=str, required=True,
                        help='Target gene name')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--control-label', type=str, default='non-targeting',
                        help='Label for control cells (default: non-targeting)')
    
    args = parser.parse_args()
    
    # Run comprehensive analysis
    results = comprehensive_analysis(
        args.rna_predictions,
        args.protein_predictions,
        args.control_rna,
        args.control_protein,
        args.target_gene,
        args.output_dir,
        control_label=args.control_label
    )
    
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()

