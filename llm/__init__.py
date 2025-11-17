"""LLM module for processing user queries and interpreting Agent_Tools results."""

from .input import process_user_query, extract_perturbation_info
from .output import interpret_results, generate_summary, collect_results_from_pipeline
from .phenotype_kb import PhenotypeKB
from .futurehouse_client import search_literature, get_pmids, format_citation
from .plots import plot_pathway_enrichment, plot_phenotype_scores, plot_volcano, plot_rna_gsea, plot_protein_psea, plot_phenotype_enrichment
from .hypothesis_agent import generate_hypotheses
from .report import build_report, build_html_report
from .perturbation_orchestrator import (
    process_user_query as orchestrate_perturbation,
    load_valid_perturbation_names,
    validate_perturbation_name,
    detect_user_intent,
    run_state_gene_perturbation,
    run_state_drug_perturbation
)

__all__ = [
    'process_user_query',
    'extract_perturbation_info',
    'interpret_results',
    'generate_summary',
    'collect_results_from_pipeline',
    'PhenotypeKB',
    'search_literature',
    'get_pmids',
    'format_citation',
    'plot_pathway_enrichment',
    'plot_phenotype_scores',
    'plot_volcano',
    'plot_rna_gsea',
    'plot_protein_psea',
    'plot_phenotype_enrichment',
    'generate_hypotheses',
    'build_report',
    'build_html_report',
    'orchestrate_perturbation',
    'load_valid_perturbation_names',
    'validate_perturbation_name',
    'detect_user_intent',
    'run_state_gene_perturbation',
    'run_state_drug_perturbation'
]

