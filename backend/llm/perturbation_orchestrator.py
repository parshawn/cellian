"""
Perturbation Orchestrator: Validates perturbations, detects user intent, and orchestrates Agent_Tools execution.

This module:
- Loads valid perturbation names from STATE model pickle files
- Validates user-provided names (exact/close/none matching)
- Detects user intent (single perturbation vs comparison)
- Calls Agent_Tools functions after validation
- Generates outputs (DEGs, pathways, phenotypes, plots) for each perturbation
- Creates comparison summaries when both perturbations are run
"""

import os
import sys
import json
import pickle
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add Agent_Tools to path
agent_tools_path = Path(__file__).parent.parent / "Agent_Tools"
sys.path.insert(0, str(agent_tools_path))

AGENT_TOOLS_AVAILABLE = True  # Assume available, will fail gracefully if not

# Import LLM module functions
from .input import extract_perturbation_info
from .output import collect_results_from_pipeline
from .plots import (
    plot_volcano, plot_pathway_enrichment,
    plot_rna_gsea, plot_protein_psea
)
from .hypothesis_agent import generate_hypotheses
from .report import build_report, build_html_report


# Paths to STATE model perturbation lists
DRUG_PERTURBATION_LIST_PATH = "/home/nebius/ST-Tahoe/var_dims.pkl"
GENE_PERTURBATION_LIST_PATH = "/home/nebius/state/test_replogle/hepg2_holdout/var_dims.pkl"


def load_valid_perturbation_names() -> Tuple[List[str], List[str]]:
    """
    Load valid perturbation names from STATE model pickle files.
    
    Returns:
        Tuple of (drug_names, gene_names) lists
    """
    drug_names = []
    gene_names = []
    
    # Load drug names
    try:
        with open(DRUG_PERTURBATION_LIST_PATH, "rb") as f:
            var = pickle.load(f)
        drugs = var.get("pert_names", [])
        # Convert numpy strings to Python strings
        drug_names = [str(d) for d in drugs if d is not None]
    except Exception as e:
        print(f"Warning: Could not load drug perturbation names: {e}")
    
    # Load gene names
    try:
        with open(GENE_PERTURBATION_LIST_PATH, "rb") as f:
            var = pickle.load(f)
        genes = var.get("pert_names", [])
        # Convert numpy strings to Python strings
        gene_names = [str(g) for g in genes if g is not None]
    except Exception as e:
        print(f"Warning: Could not load gene perturbation names: {e}")
    
    return drug_names, gene_names


def normalize_name(name: str, is_gene: bool = True) -> str:
    """
    Normalize perturbation name for matching.
    
    Args:
        name: Input name
        is_gene: True for gene (uppercase), False for drug (case-insensitive)
    
    Returns:
        Normalized name
    """
    name = name.strip()
    if is_gene:
        return name.upper()
    else:
        return name.lower()


def extract_drug_name_from_tuple(drug_tuple_str: str) -> str:
    """
    Extract drug name from tuple string format like "[('(R)-Verapamil (hydrochloride)', 0.05, 'uM')]".
    
    Args:
        drug_tuple_str: Drug name in tuple format
    
    Returns:
        Extracted drug name (e.g., "(R)-Verapamil (hydrochloride)")
    """
    import ast
    try:
        # Parse the tuple string
        parsed = ast.literal_eval(drug_tuple_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            first_tuple = parsed[0]
            if isinstance(first_tuple, tuple) and len(first_tuple) > 0:
                return str(first_tuple[0])  # Drug name is first element
    except (ValueError, SyntaxError, IndexError):
        pass
    # Fallback: try regex to extract drug name from string
    import re
    match = re.search(r"\('([^']+)'", drug_tuple_str)
    if match:
        return match.group(1)
    return drug_tuple_str  # Return as-is if parsing fails


def find_best_match(requested_name: str, valid_names: List[str], is_gene: bool = True) -> Dict[str, Any]:
    """
    Find best match for requested name in valid names list.
    
    Args:
        requested_name: User-provided name
        valid_names: List of valid names from STATE model
        is_gene: True for gene, False for drug
    
    Returns:
        Dict with keys: requested_name, used_name, match_type, similarity_score
    """
    normalized_requested = normalize_name(requested_name, is_gene)
    
    # For drugs, extract drug names from tuple strings for matching
    if not is_gene:
        # Build a map of extracted drug names to original tuple strings
        drug_name_map = {}
        for v in valid_names:
            extracted = extract_drug_name_from_tuple(str(v))
            drug_name_map[normalize_name(extracted, is_gene=False)] = v
        normalized_valid = drug_name_map
    else:
        # For genes, use simple normalization
        normalized_valid = {normalize_name(v, is_gene): v for v in valid_names}
    
    # Exact match
    if normalized_requested in normalized_valid:
        return {
            "requested_name": requested_name,
            "used_name": normalized_valid[normalized_requested],
            "match_type": "exact",
            "similarity_score": 1.0
        }
    
    # Fuzzy match
    best_match = None
    best_score = 0.0
    
    for normalized_valid_name, original_valid_name in normalized_valid.items():
        # For genes, compare simple strings
        # For drugs, compare extracted drug names
        score = SequenceMatcher(None, normalized_requested, normalized_valid_name).ratio()
        if score > best_score:
            best_score = score
            best_match = original_valid_name
    
    if best_score >= 0.80:
        return {
            "requested_name": requested_name,
            "used_name": best_match,
            "match_type": "close",
            "similarity_score": best_score
        }
    else:
        return {
            "requested_name": requested_name,
            "used_name": None,
            "match_type": "none",
            "similarity_score": best_score if best_match else 0.0
        }


def detect_user_intent(query: str) -> Dict[str, Any]:
    """
    Intelligently detect user intent from query using LLM extraction.
    
    Extracts:
    - Gene names (single or multiple)
    - Drug names (single or multiple)
    - Perturbation types (KO, KD, OE for genes)
    - Comparison requirements
    - Pathway mentions
    - Phenotype filters
    
    Args:
        query: User query string
    
    Returns:
        Dict with keys: mode, gene_name, drug_name, has_comparison_phrase
    """
    query_lower = query.lower()
    
    # Check for comparison phrases (comprehensive list)
    comparison_phrases = [
        "which is stronger", "which has more effect", "which perturbation is stronger",
        "compare", "comparison", "difference between", "stronger activation", "stronger repression",
        "vs", "versus", "or", "which", "stronger", "weaker", "better", "more effective",
        "which one", "compare the effect", "what's the difference", "which works better"
    ]
    has_comparison_phrase = any(phrase in query_lower for phrase in comparison_phrases)
    
    # Use LLM to extract perturbation information (enhanced extraction)
    gene_name = None
    drug_name = None
    
    # First, try regex to extract gene from "KO of TP53" format (before LLM)
    ko_pattern = r'(?:ko|knockout)\s+of\s+([A-Z][A-Z0-9]{1,9})\b'
    ko_match = re.search(ko_pattern, query, re.IGNORECASE)
    if ko_match:
        gene_name = ko_match.group(1)
    
    # Also try standard gene pattern (uppercase words)
    if not gene_name:
        gene_pattern = r'\b([A-Z][A-Z0-9]{1,9})\b'
        gene_matches = re.findall(gene_pattern, query)
        if gene_matches:
            common_words = {"KO", "VS", "OR", "AND", "THE", "IS", "IN", "TO", "OF", "FOR", "RUN"}
            gene_candidates = [g for g in gene_matches if g not in common_words]
            if gene_candidates:
                gene_name = gene_candidates[0]
    
    # Use LLM to extract perturbation information (enhanced to extract multiple targets)
    try:
        # Use existing LLM extraction function (now enhanced to extract multiple targets)
        pert_info = extract_perturbation_info(query)
        extracted_target = pert_info.get("target", "")
        extracted_target2 = pert_info.get("target2", "")  # For comparisons
        pert_type = pert_info.get("type", "")
        pert_type2 = pert_info.get("type2", "")  # For comparisons
        
        # Check if LLM detected comparison
        if pert_info.get("is_comparison", False):
            has_comparison_phrase = True
        
        # Load valid names to determine if it's a gene or drug
        drug_names, gene_names = load_valid_perturbation_names()
        
        # Process first target
        if extracted_target and extracted_target != "UNKNOWN":
            # Check if LLM identified it as a drug
            if pert_type == "drug":
                drug_match = find_best_match(extracted_target, drug_names, is_gene=False)
                if drug_match["match_type"] != "none":
                    drug_name = extracted_target
            # Check if it's a gene (KO/KD/OE or generic gene)
            elif pert_type in ["KO", "KD", "OE"] or not gene_name:
                gene_match = find_best_match(extracted_target, gene_names, is_gene=True)
                if gene_match["match_type"] != "none":
                    gene_name = extracted_target
            # If LLM didn't specify type, try both
            elif pert_type not in ["drug", "KO", "KD", "OE"]:
                # Try gene first
                if not gene_name:
                    gene_match = find_best_match(extracted_target, gene_names, is_gene=True)
                    if gene_match["match_type"] != "none":
                        gene_name = extracted_target
                
                # Try drug if not a gene
                if not drug_name and not gene_name:
                    drug_match = find_best_match(extracted_target, drug_names, is_gene=False)
                    if drug_match["match_type"] != "none":
                        drug_name = extracted_target
        
        # Process second target (for comparisons)
        if extracted_target2 and extracted_target2 != "UNKNOWN":
            # Check if LLM identified it as a drug
            if pert_type2 == "drug":
                drug_match = find_best_match(extracted_target2, drug_names, is_gene=False)
                if drug_match["match_type"] != "none":
                    drug_name = extracted_target2
            # Check if it's a gene
            elif pert_type2 in ["KO", "KD", "OE"]:
                gene_match = find_best_match(extracted_target2, gene_names, is_gene=True)
                if gene_match["match_type"] != "none":
                    # If we already have a gene, this is a comparison
                    if gene_name and gene_name.upper() != extracted_target2.upper():
                        has_comparison_phrase = True
                    else:
                        gene_name = extracted_target2
            # Try both if type unknown
            else:
                if not gene_name or (gene_name and gene_name.upper() != extracted_target2.upper()):
                    gene_match = find_best_match(extracted_target2, gene_names, is_gene=True)
                    if gene_match["match_type"] != "none":
                        # If we have a different gene, this is a comparison
                        if gene_name and gene_name.upper() != extracted_target2.upper():
                            has_comparison_phrase = True
                        else:
                            gene_name = extracted_target2
                
                if not drug_name:
                    drug_match = find_best_match(extracted_target2, drug_names, is_gene=False)
                    if drug_match["match_type"] != "none":
                        drug_name = extracted_target2
                        
    except Exception as e:
        print(f"Warning: LLM extraction failed: {e}, using regex fallback")
    
    # Enhanced regex fallback for drug extraction
    # Pattern 1: Extract drug from "or DRUG", "vs DRUG", "versus DRUG" patterns
    if not drug_name:
        or_pattern = r'\b(?:or|vs|versus)\s+([a-z][a-z0-9_\-]+)\b'
        or_match = re.search(or_pattern, query_lower)
        if or_match:
            potential_drug = or_match.group(1)
            # Validate it's not a common word
            common_words = {"the", "is", "which", "stronger", "compare", "treatment", "perturbation", "with", "on", "for", "show", "simulate", "treating", "analysis", "effects", "results", "what", "happens", "if", "i", "knock", "down", "want", "to", "see", "effect", "can", "you", "analyze", "me", "run"}
            if potential_drug.lower() not in common_words and len(potential_drug) > 3:
                drug_name = potential_drug
    
    # Pattern 1b: Extract drug from "or treating with DRUG" or "or treatment with DRUG"
    if not drug_name:
        treating_pattern = r'\b(?:or|vs|versus)\s+(?:treating|treatment)\s+with\s+([a-z][a-z0-9_\-]+)\b'
        treating_match = re.search(treating_pattern, query_lower)
        if treating_match:
            potential_drug = treating_match.group(1)
            common_words = {"the", "is", "which", "stronger", "compare", "cells", "tissue"}
            if potential_drug.lower() not in common_words and len(potential_drug) > 3:
                drug_name = potential_drug
    
    # Pattern 2: Extract drug from "with DRUG", "on DRUG", "for DRUG" patterns (common in drug queries)
    if not drug_name:
        with_pattern = r'\b(?:with|on|for)\s+([a-z][a-z0-9_\-]+)\s+(?:drug|treatment|treatment|effects|results)?'
        with_match = re.search(with_pattern, query_lower)
        if with_match:
            potential_drug = with_match.group(1)
            common_words = {"the", "is", "which", "stronger", "compare", "treatment", "perturbation", "show", "simulate", "treating", "analysis", "effects", "results", "what", "happens", "if", "i", "knock", "down", "want", "to", "see", "effect", "can", "you", "analyze", "me", "run", "drug"}
            if potential_drug.lower() not in common_words and len(potential_drug) > 3:
                drug_name = potential_drug
    
    # Pattern 3: If no gene found and query has lowercase words, might be drug-only
    # Be more careful - look for drug-like patterns
    if not drug_name and not gene_name:
        # Look for patterns like "DRUG treatment", "DRUG drug", "DRUG effects"
        drug_context_pattern = r'\b([a-z][a-z0-9_\-]{3,})\s+(?:treatment|drug|effects|results|analysis)'
        drug_context_match = re.search(drug_context_pattern, query_lower)
        if drug_context_match:
            potential_drug = drug_context_match.group(1)
            common_words = {"show", "simulate", "treating", "analysis", "effects", "results", "what", "happens", "if", "i", "knock", "down", "want", "to", "see", "effect", "can", "you", "analyze", "me", "run", "the", "is", "which", "stronger", "compare"}
            if potential_drug.lower() not in common_words:
                drug_name = potential_drug
        
        # Last resort: find lowercase words that aren't common words
        if not drug_name:
            words = query.split()
            common_words_lower = {"of", "or", "vs", "versus", "the", "is", "which", "stronger", "compare", "run", "ko", "knockout", "treatment", "perturbation", "with", "on", "for", "show", "simulate", "treating", "analysis", "effects", "results", "drug", "what", "happens", "if", "i", "knock", "down", "want", "to", "see", "effect", "can", "you", "analyze", "me", "and", "a", "an"}
            potential_drugs = [w.strip('.,!?;:()[]{}') for w in words if w.lower().strip('.,!?;:()[]{}') not in common_words_lower and not w.isupper() and len(w.strip('.,!?;:()[]{}')) > 3]
            if potential_drugs:
                # Take the first potential drug (will be validated against drug list later)
                drug_name = potential_drugs[0]
    
    # Determine mode based on what was extracted
    if gene_name and drug_name:
        mode = "comparison"
    elif has_comparison_phrase and (gene_name or drug_name):
        mode = "comparison"
    elif gene_name:
        mode = "single_gene"
    elif drug_name:
        mode = "single_drug"
    else:
        mode = "unknown"
    
    return {
        "mode": mode,
        "gene_name": gene_name,
        "drug_name": drug_name,
        "has_comparison_phrase": has_comparison_phrase
    }


def validate_perturbation_name(name: str, is_gene: bool, valid_names: List[str]) -> Dict[str, Any]:
    """
    Validate perturbation name and return match info.
    
    Args:
        name: User-provided name
        is_gene: True for gene, False for drug
        valid_names: List of valid names
    
    Returns:
        Match info dict
    """
    if not name:
        return {
            "requested_name": name,
            "used_name": None,
            "match_type": "none",
            "similarity_score": 0.0
        }
    
    return find_best_match(name, valid_names, is_gene)


def run_state_gene_perturbation(pert_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run STATE gene perturbation via Agent_Tools.
    
    This function calls Agent_Tools perturbation_pipeline and collects results.
    Agent_Tools outputs CSV files in pathway_analysis/ directory that we parse.
    
    Args:
        pert_name: Validated gene name
        context: Context dict with cell_type, species, etc.
                 Can include 'condition' key: 'IFNγ', 'Control', or 'Co-Culture' (default: 'Control')
    
    Returns:
        Results dict with DEGs, pathways, phenotypes, metrics, etc.
    """
    # Get condition from context or default to "Control"
    condition = context.get("condition", "Control")
    
    # Map condition name to stdin input (Agent_Tools prompt: 1=IFNγ, 2=Control, 3=Co-Culture)
    condition_map = {
        "IFNγ": "1",
        "Control": "2",
        "Co-Culture": "3"
    }
    condition_input = condition_map.get(condition, "2")  # Default to Control if unknown
    
    print(f"\n{'='*70}")
    print(f"STEP 1: Running Gene Perturbation Pipeline")
    print(f"{'='*70}")
    print(f"Gene: {pert_name}")
    print(f"Condition: {condition}")
    print(f"Output directory: {Path(__file__).parent / 'perturbation_outputs' / f'gene_{pert_name}'}")
    print(f"Starting Agent_Tools pipeline...")
    print(f"{'='*70}\n")
    
    # Create output directory for this perturbation
    output_dir = Path(__file__).parent / "perturbation_outputs" / f"gene_{pert_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Call Agent_Tools perturbation_pipeline via subprocess
    try:
        import subprocess
        import time
        from datetime import datetime
        agent_tools_script = Path(__file__).parent.parent / "Agent_Tools" / "perturbation_pipeline.py"
        
        start_time = datetime.now()
        print(f"[{start_time.strftime('%H:%M:%S')}] Starting Agent_Tools perturbation_pipeline.py...")
        print(f"  Command: python {agent_tools_script} --target-gene {pert_name} --output-dir {output_dir}")
        print(f"  Condition: {condition} (will be selected via stdin)")
        print(f"  This will run:")
        print(f"    1. State inference (Perturbation → RNA)")
        print(f"    2. scTranslator inference (RNA → Protein)")
        print(f"    3. Pathway analysis (GSEA, KEGG, Reactome, GO)")
        print(f"  Estimated time: 20-40 minutes\n")
        
        # Pass condition via stdin to avoid interactive prompt
        # Agent_Tools prompts: 1=IFNγ, 2=Control, 3=Co-Culture
        # NOTE: Agent_Tools now handles genes not in experimental data automatically:
        #   - If gene IS in target_gene column: Uses normal protocol (validates against ground truth)
        #   - If gene IS NOT in target_gene column: Uses drug-like protocol (80/20 split, no validation)
        stdin_input = f"{condition_input}\n"
        
        # Stream output in real-time so it appears in logs immediately
        print(f"\n{'='*70}")
        print(f"AGENT_TOOLS OUTPUT (streaming in real-time):")
        print(f"{'='*70}\n")
        
        import subprocess as sp
        # Use Python's -u flag (unbuffered) to ensure all print statements appear immediately
        process = sp.Popen(
            [
                sys.executable, "-u",  # -u flag forces unbuffered stdout/stderr
                str(agent_tools_script),
                "--target-gene", pert_name,
                "--output-dir", str(output_dir)
            ],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=0,  # Unbuffered (0 = unbuffered, 1 = line buffered)
            universal_newlines=True
        )
        
        # Write stdin and read stdout/stderr in real-time
        stdout_lines = []
        stderr_lines = []
        condition_input_sent = False  # Track if we've sent the condition input
        
        # Read output line by line and print immediately
        # Wait for Agent_Tools to prompt for condition, then send input
        # This ensures ALL print statements from Agent_Tools appear in orchestrator logs
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Print immediately to log (preserve formatting, only strip trailing newline)
                # ALL Agent_Tools print statements will appear here
                print(output.rstrip(), flush=True)  # Explicit flush=True for immediate output
                stdout_lines.append(output)
                # Additional flush to ensure output appears in log file in real-time
                sys.stdout.flush()
                
                # Check if Agent_Tools is prompting for condition selection
                # The prompt appears as: "Enter choice (1, 2, or 3):" or "Enter choice (1 or 2):"
                if not condition_input_sent and (
                    "Enter choice (1, 2, or 3):" in output or 
                    "Enter choice (1 or 2):" in output or
                    ("CONDITION SELECTION" in output or "Enter choice" in output)
                ):
                    # Wait a tiny bit to ensure the prompt is fully printed
                    time.sleep(0.1)
                    # Now send the condition input
                    try:
                        process.stdin.write(stdin_input)
                        process.stdin.flush()
                        condition_input_sent = True
                        print(f"[ORCHESTRATOR] Sent condition selection: {condition_input.strip()}", flush=True)
                    except (BrokenPipeError, OSError):
                        # stdin might already be closed, try to continue
                        pass
        
        # If we never sent the condition input (e.g., condition was passed via --condition flag), close stdin
        if not condition_input_sent:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        
        # Wait for process to finish and get return code
        # Since we've already read all output and closed stdin, just wait for process to finish
        try:
            return_code = process.wait()  # wait() is safer than poll() after reading output
        except Exception:
            return_code = process.poll() or -1
        
        # Try to get any remaining output (may fail if already consumed, but that's okay)
        try:
            remaining_stdout, remaining_stderr = process.communicate(timeout=1)
            if remaining_stdout:
                # Print any remaining Agent_Tools output
                print(remaining_stdout.rstrip(), flush=True)
                stdout_lines.append(remaining_stdout)
                sys.stdout.flush()
            if remaining_stderr:
                # Print any remaining Agent_Tools stderr
                print(remaining_stderr.rstrip(), flush=True)
                stderr_lines.append(remaining_stderr)
                sys.stdout.flush()
        except (ValueError, OSError):
            # Expected: communicate() may fail if stdin/stdout already closed/consumed
            # This is fine since we've already read all the output in the loop above
            # All Agent_Tools print statements should have already been captured and printed
            pass
        
        # Combine all output for error messages
        full_output = ''.join(stdout_lines) + ''.join(stderr_lines)
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds() / 60
        
        print(f"\n{'='*70}")
        print(f"AGENT_TOOLS FINISHED (return code: {return_code}, elapsed: {elapsed:.1f} minutes)")
        print(f"{'='*70}\n")
        
        if return_code != 0:
            print(f"\n[{end_time.strftime('%H:%M:%S')}] ❌ Agent_Tools pipeline failed after {elapsed:.1f} minutes")
            error_msg = full_output if full_output else "No output captured"
            
            # Provide more detailed error information
            print(f"  Full error output:\n{error_msg[-2000:]}\n")  # Show last 2000 chars
            
            # Check if it's the old "No cells found" error (shouldn't happen with updated Agent_Tools)
            if "No cells found with target_gene" in error_msg:
                print(f"  ⚠️  WARNING: This error suggests Agent_Tools may not have the latest update.")
                print(f"     Agent_Tools should handle genes not in experimental data automatically.")
                print(f"     If the gene is not in experimental data, it should:")
                print(f"     1. Filter to control cells (non-targeting)")
                print(f"     2. Apply 80/20 split (80% perturbed, 20% control)")
                print(f"     3. Skip validation but still run analysis\n")
            
            return {"error": f"Agent_Tools failed: {error_msg}"}
        
        print(f"\n[{end_time.strftime('%H:%M:%S')}] ✅ Agent_Tools pipeline completed in {elapsed:.1f} minutes")
        print(f"  Collecting results from output directory...\n")
        
        # Collect results from output directory
        # This parses CSV files created by pathway_analysis
        # NOTE: Agent_Tools now handles genes not in experimental data - if gene not found,
        # it uses drug-like protocol (80/20 split) and skips validation, but still generates results
        results = collect_results_from_pipeline(
            output_dir=str(output_dir),
            target_gene=pert_name
        )
        
        # Add perturbation info
        results["perturbation_name"] = pert_name
        results["perturbation_type"] = "gene"
        
        print(f"  ✓ Results collected successfully")
        print(f"    - Differential expression: {'✓' if 'differential_rna' in results.get('pathway_analysis', {}) else '✗'}")
        print(f"    - GSEA results: {'✓' if 'gsea_rna' in results.get('pathway_analysis', {}) else '✗'}")
        protein_enrichments = sum(1 for db in ['kegg_enrichment', 'reactome_enrichment', 'go_enrichment'] if db in results.get('pathway_analysis', {}))
        rna_enrichments = sum(1 for db in ['kegg_enrichment_rna', 'reactome_enrichment_rna', 'go_enrichment_rna'] if db in results.get('pathway_analysis', {}))
        print(f"    - Protein pathway enrichments: {protein_enrichments} databases")
        print(f"    - RNA pathway enrichments: {rna_enrichments} databases (NEW)\n")
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"\n❌ Agent_Tools timed out after 1 hour")
        return {"error": "Agent_Tools timed out after 1 hour"}
    except Exception as e:
        print(f"\n❌ Error running gene perturbation: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error running gene perturbation: {e}"}


def run_state_drug_perturbation(pert_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run STATE drug perturbation via Agent_Tools.
    
    Args:
        pert_name: Validated drug name (in tuple format like "[('drugname', conc, 'uM')]")
        context: Context dict
                 Can include 'condition' key: 'IFNγ', 'Control', or 'Co-Culture' (default: 'Control')
    
    Returns:
        Results dict
    """
    # Get condition from context or default to "Control"
    condition = context.get("condition", "Control")
    
    # Map condition name to stdin input (Agent_Tools prompt: 1=IFNγ, 2=Control, 3=Co-Culture)
    condition_map = {
        "IFNγ": "1",
        "Control": "2",
        "Co-Culture": "3"
    }
    condition_input = condition_map.get(condition, "2")  # Default to Control if unknown
    
    print(f"\n{'='*70}")
    print(f"STEP 1: Running Drug Perturbation Pipeline")
    print(f"{'='*70}")
    print(f"Drug: {pert_name}")
    print(f"Condition: {condition}")
    
    # Create output directory (sanitize drug name for filesystem)
    safe_name = pert_name[:50].replace('/', '_').replace('(', '_').replace(')', '_').replace("'", '_')
    output_dir = Path(__file__).parent / "perturbation_outputs" / f"drug_{safe_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Starting Agent_Tools pipeline...")
    print(f"{'='*70}\n")
    
    try:
        import subprocess
        import time
        from datetime import datetime
        agent_tools_script = Path(__file__).parent.parent / "Agent_Tools" / "perturbation_pipeline.py"
        
        start_time = datetime.now()
        print(f"[{start_time.strftime('%H:%M:%S')}] Starting Agent_Tools perturbation_pipeline.py...")
        print(f"  Command: python {agent_tools_script} --perturbation-type drug --drug {pert_name[:80]}... --output-dir {output_dir}")
        print(f"  Condition: {condition} (will be selected via stdin)")
        print(f"  This will run:")
        print(f"    1. ST-Tahoe inference (Drug Perturbation → RNA)")
        print(f"    2. scTranslator inference (RNA → Protein)")
        print(f"    3. Pathway analysis (GSEA, KEGG, Reactome, GO)")
        print(f"  Estimated time: 20-40 minutes\n")
        
        # Pass condition via stdin to avoid interactive prompt
        # Agent_Tools prompts: 1=IFNγ, 2=Control, 3=Co-Culture
        stdin_input = f"{condition_input}\n"
        
        # Stream output in real-time so it appears in logs immediately
        print(f"\n{'='*70}")
        print(f"AGENT_TOOLS OUTPUT (streaming in real-time):")
        print(f"{'='*70}\n")
        
        import subprocess as sp
        # Use Python's -u flag (unbuffered) to ensure all print statements appear immediately
        process = sp.Popen(
            [
                sys.executable, "-u",  # -u flag forces unbuffered stdout/stderr
                str(agent_tools_script),
                "--perturbation-type", "drug",
                "--drug", pert_name,
                "--output-dir", str(output_dir)
            ],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=0,  # Unbuffered (0 = unbuffered, 1 = line buffered)
            universal_newlines=True
        )
        
        # Write stdin and read stdout/stderr in real-time
        stdout_lines = []
        stderr_lines = []
        condition_input_sent = False  # Track if we've sent the condition input
        
        # Read output line by line and print immediately
        # Wait for Agent_Tools to prompt for condition, then send input
        # This ensures ALL print statements from Agent_Tools appear in orchestrator logs
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Print immediately to log (preserve formatting, only strip trailing newline)
                # ALL Agent_Tools print statements will appear here
                print(output.rstrip(), flush=True)  # Explicit flush=True for immediate output
                stdout_lines.append(output)
                # Additional flush to ensure output appears in log file in real-time
                sys.stdout.flush()
                
                # Check if Agent_Tools is prompting for condition selection
                # The prompt appears as: "Enter choice (1, 2, or 3):" or "Enter choice (1 or 2):"
                if not condition_input_sent and (
                    "Enter choice (1, 2, or 3):" in output or 
                    "Enter choice (1 or 2):" in output or
                    ("CONDITION SELECTION" in output or "Enter choice" in output)
                ):
                    # Wait a tiny bit to ensure the prompt is fully printed
                    time.sleep(0.1)
                    # Now send the condition input
                    try:
                        process.stdin.write(stdin_input)
                        process.stdin.flush()
                        condition_input_sent = True
                        print(f"[ORCHESTRATOR] Sent condition selection: {condition_input.strip()}", flush=True)
                    except (BrokenPipeError, OSError):
                        # stdin might already be closed, try to continue
                        pass
        
        # If we never sent the condition input (e.g., condition was passed via --condition flag), close stdin
        if not condition_input_sent:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        
        # Wait for process to finish and get return code
        # Since we've already read all output and closed stdin, just wait for process to finish
        try:
            return_code = process.wait()  # wait() is safer than poll() after reading output
        except Exception:
            return_code = process.poll() or -1
        
        # Try to get any remaining output (may fail if already consumed, but that's okay)
        try:
            remaining_stdout, remaining_stderr = process.communicate(timeout=1)
            if remaining_stdout:
                # Print any remaining Agent_Tools output
                print(remaining_stdout.rstrip(), flush=True)
                stdout_lines.append(remaining_stdout)
                sys.stdout.flush()
            if remaining_stderr:
                # Print any remaining Agent_Tools stderr
                print(remaining_stderr.rstrip(), flush=True)
                stderr_lines.append(remaining_stderr)
                sys.stdout.flush()
        except (ValueError, OSError):
            # Expected: communicate() may fail if stdin/stdout already closed/consumed
            # This is fine since we've already read all the output in the loop above
            # All Agent_Tools print statements should have already been captured and printed
            pass
        
        # Combine all output for error messages
        full_output = ''.join(stdout_lines) + ''.join(stderr_lines)
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds() / 60
        
        print(f"\n{'='*70}")
        print(f"AGENT_TOOLS FINISHED (return code: {return_code}, elapsed: {elapsed:.1f} minutes)")
        print(f"{'='*70}\n")
        
        if return_code != 0:
            print(f"\n[{end_time.strftime('%H:%M:%S')}] ❌ Agent_Tools pipeline failed after {elapsed:.1f} minutes")
            error_msg = full_output if full_output else "No output captured"
            
            # Provide more detailed error information
            print(f"  Full error output:\n{error_msg[-2000:]}\n")  # Show last 2000 chars
            
            return {"error": f"Agent_Tools failed: {error_msg}"}
        
        print(f"\n[{end_time.strftime('%H:%M:%S')}] ✅ Agent_Tools pipeline completed in {elapsed:.1f} minutes")
        print(f"  Collecting results from output directory...\n")
        
        # Collect results (similar structure to gene perturbation)
        # For drugs, we use the drug name as identifier
        results = collect_results_from_pipeline(
            output_dir=str(output_dir),
            target_gene=None  # Drugs don't have target_gene, but we can use pert_name
        )
        
        # Add perturbation info
        results["perturbation_name"] = pert_name
        results["perturbation_type"] = "drug"
        
        print(f"  ✓ Results collected successfully")
        print(f"    - Differential expression: {'✓' if 'differential_rna' in results.get('pathway_analysis', {}) else '✗'}")
        print(f"    - GSEA results: {'✓' if 'gsea_rna' in results.get('pathway_analysis', {}) else '✗'}")
        protein_enrichments = sum(1 for db in ['kegg_enrichment', 'reactome_enrichment', 'go_enrichment'] if db in results.get('pathway_analysis', {}))
        rna_enrichments = sum(1 for db in ['kegg_enrichment_rna', 'reactome_enrichment_rna', 'go_enrichment_rna'] if db in results.get('pathway_analysis', {}))
        print(f"    - Protein pathway enrichments: {protein_enrichments} databases")
        print(f"    - RNA pathway enrichments: {rna_enrichments} databases (NEW)\n")
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"\n❌ Agent_Tools timed out after 1 hour")
        return {"error": "Agent_Tools timed out after 1 hour"}
    except Exception as e:
        print(f"\n❌ Error running drug perturbation: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error running drug perturbation: {e}"}


def process_pathway_batch_query(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a pathway-based batch query that requires discovering multiple perturbations.
    
    Examples:
        "compare the effect of top 5 relevant drugs and gene knockdowns 
         affecting the mTOR pathway that affects cell proliferation"
        "find top 3 genes in PI3K pathway"
    
    Args:
        query: User query string
        context: Optional context dict
    
    Returns:
        Results dict with batch perturbations
    """
    from .pathway_discovery import (
        parse_pathway_query, discover_pathway_perturbations, 
        rank_perturbations_by_relevance
    )
    from datetime import datetime
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print(f"\n{'='*70}")
    print(f"PATHWAY-BASED BATCH QUERY PROCESSING")
    print(f"{'='*70}")
    print(f"Query: {query}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    context = context or {}
    
    # Parse query to extract pathway, phenotype, number of items
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Parsing pathway query...")
    query_info = parse_pathway_query(query)
    pathway_name = query_info.get("pathway_name")
    phenotype_filter = query_info.get("phenotype_filter")
    num_items = query_info.get("num_items", 5)
    item_types = query_info.get("item_types", ["gene", "drug"])
    comparison_type = query_info.get("comparison_type", "batch")
    
    if not pathway_name:
        return {
            "error": "Could not extract pathway name from query",
            "query": query,
            "query_info": query_info
        }
    
    print(f"  Pathway: {pathway_name}")
    if phenotype_filter:
        print(f"  Phenotype filter: {phenotype_filter}")
    print(f"  Requested items: {num_items} of type(s): {item_types}")
    print(f"  Comparison type: {comparison_type}\n")
    
    # Discover perturbations
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Discovering perturbations...")
    discovered = discover_pathway_perturbations(
        pathway_name=pathway_name,
        phenotype_filter=phenotype_filter,
        num_items=num_items,
        item_types=item_types
    )
    
    discovered_genes = discovered.get("genes", [])
    discovered_drugs = discovered.get("drugs", [])
    
    print(f"  Found {len(discovered_genes)} genes: {discovered_genes[:5]}")
    print(f"  Found {len(discovered_drugs)} drugs: {[d[:50] for d in discovered_drugs[:3]]}\n")
    
    if not discovered_genes and not discovered_drugs:
        return {
            "error": f"No perturbations found for pathway '{pathway_name}'",
            "query": query,
            "query_info": query_info
        }
    
    # Validate and collect valid perturbations
    drug_names, gene_names = load_valid_perturbation_names()
    
    all_perturbations = []
    for gene in discovered_genes:
        match_info = validate_perturbation_name(gene, is_gene=True, valid_names=gene_names)
        if match_info["match_type"] != "none":
            all_perturbations.append({
                "type": "gene",
                "name": match_info["used_name"],
                "match_info": match_info
            })
    
    for drug in discovered_drugs:
        match_info = validate_perturbation_name(drug, is_gene=False, valid_names=drug_names)
        if match_info["match_type"] != "none":
            all_perturbations.append({
                "type": "drug",
                "name": match_info["used_name"],
                "match_info": match_info
            })
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running {len(all_perturbations)} perturbations...\n")
    
    # Run perturbations in parallel
    results = {
        "query": query,
        "query_info": query_info,
        "pathway_name": pathway_name,
        "perturbations": [],
        "batch_results": {}
    }
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for pert_info in all_perturbations:
            pert_type = pert_info["type"]
            pert_name = pert_info["name"]
            
            if pert_type == "gene":
                future = executor.submit(run_state_gene_perturbation, pert_name, context)
            else:
                future = executor.submit(run_state_drug_perturbation, pert_name, context)
            
            futures[future] = pert_info
        
        # Collect results as they complete
        for future in as_completed(futures):
            pert_info = futures[future]
            try:
                pert_results = future.result()
                if "error" not in pert_results:
                    # Extract query intent for adaptive output generation (use query from context if available)
                    # Use query from process_pathway_batch_query
                    query_intent = extract_perturbation_info(query) if query else None
                    pert_results["outputs"] = _generate_outputs(
                        pert_results, pert_info["type"], pert_info["name"], query_intent=query_intent
                    )
                    pert_info["results"] = pert_results
                    pert_info["outputs"] = pert_results["outputs"]
                    print(f"  ✓ {pert_info['type'].upper()}: {pert_info['name'][:50]} completed\n")
                else:
                    pert_info["error"] = pert_results["error"]
                    print(f"  ✗ {pert_info['type'].upper()}: {pert_info['name'][:50]} failed\n")
            except Exception as e:
                pert_info["error"] = str(e)
                print(f"  ✗ {pert_info['type'].upper()}: {pert_info['name'][:50]} error: {e}\n")
            
            results["perturbations"].append(pert_info)
    
    # Generate batch comparison summary
    if comparison_type == "multi" and len(results["perturbations"]) >= 2:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating batch comparison...")
        successful = [p for p in results["perturbations"] if "error" not in p and "results" in p]
        
        if len(successful) >= 2:
            # Compare all pairs or groups
            # For now, create a summary of all perturbations
            results["batch_summary"] = {
                "pathway": pathway_name,
                "total_perturbations": len(results["perturbations"]),
                "successful": len(successful),
                "perturbations": [
                    {
                        "type": p["type"],
                        "name": p["name"],
                        "has_results": "results" in p
                    }
                    for p in results["perturbations"]
                ]
            }
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETED")
    print(f"{'='*70}")
    print(f"Total perturbations: {len(results['perturbations'])}")
    print(f"Successful: {len([p for p in results['perturbations'] if 'error' not in p])}\n")
    
    return results


def process_user_query(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main entry point: Process user query, validate perturbations, run pipeline, generate outputs.
    
    Args:
        query: User query string
        context: Optional context dict (cell_type, species, etc.)
                 Can include 'condition' key: 'IFNγ', 'Control', or 'Co-Culture'
                 If not provided and running interactively, will prompt user for condition
    
    Returns:
        Complete results dict with perturbations, outputs, comparisons
    """
    from datetime import datetime
    
    print(f"\n{'='*70}")
    print(f"PERTURBATION ORCHESTRATOR")
    print(f"{'='*70}")
    print(f"Query: {query}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    context = context or {}
    
    # Intelligently detect query type using LLM-enhanced parsing
    # FIRST: Check if this is a direct comparison query (X vs Y format)
    # Direct comparisons should be handled as regular comparison, not pathway batch queries
    query_lower = query.lower()
    direct_comparison_pattern = r"(\b(?:[A-Z][A-Z0-9]{1,9}|[a-z][a-z0-9_\-]{3,})\b)\s+(?:vs|versus|or)\s+(\b(?:[A-Z][A-Z0-9]{1,9}|[a-z][a-z0-9_\-]{3,})\b)"
    is_direct_comparison = bool(re.search(direct_comparison_pattern, query, re.IGNORECASE))
    
    # Check if this is a pathway-based batch query or complex query
    from .pathway_discovery import parse_pathway_query
    
    # Try to parse as pathway query to see if it's a complex query
    query_info = parse_pathway_query(query)
    pathway_name = query_info.get("pathway_name")
    
    # Route to pathway batch processing ONLY if:
    # 1. Pathway name was explicitly extracted AND query mentions "pathway"/"pathways"/"signaling" explicitly, OR
    # 2. Query contains pathway-related keywords and explicitly requests batch discovery (e.g., "top N", "relevant", "find")
    # BUT NOT if it's a direct comparison query (X vs Y) - those should be handled as regular comparisons
    has_pathway_keywords = any(kw in query_lower for kw in [
        "pathway", "pathways", "signaling", "signalling"
    ])
    requests_batch_discovery = any(kw in query_lower for kw in [
        "top", "most", "relevant", "multiple", 
        "several", "list", "find", "discover", "search"
    ])
    
    # IMPORTANT: Only route to pathway batch if:
    # - It's NOT a direct comparison query (X vs Y format), AND
    # - Pathway is explicitly mentioned, OR
    # - Pathway name was extracted AND batch discovery keywords are present
    is_pathway_batch_query = (not is_direct_comparison) and \
                            ((pathway_name is not None and has_pathway_keywords) or \
                             (has_pathway_keywords and requests_batch_discovery))
    
    if is_pathway_batch_query:
        # Route to pathway batch processing
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Detected pathway-based batch query, routing to batch processor...\n")
        return process_pathway_batch_query(query, context)
    
    # Ask user for condition selection if not provided and running interactively
    if "condition" not in context:
        # Check if running interactively (stdin is a TTY)
        if sys.stdin.isatty():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Please select condition (cell type)...")
            print("\nAvailable conditions:")
            print("  1. IFNγ")
            print("  2. Control")
            print("  3. Co-Culture")
            
            while True:
                try:
                    choice = input("\nEnter choice (1, 2, or 3) [default: 2 (Control)]: ").strip()
                    if not choice:
                        choice = "2"  # Default to Control
                    
                    if choice == "1":
                        context["condition"] = "IFNγ"
                        print(f"  ✓ Selected: IFNγ\n")
                        break
                    elif choice == "2":
                        context["condition"] = "Control"
                        print(f"  ✓ Selected: Control\n")
                        break
                    elif choice == "3":
                        context["condition"] = "Co-Culture"
                        print(f"  ✓ Selected: Co-Culture\n")
                        break
                    else:
                        print("  Invalid choice. Please enter 1, 2, or 3.")
                except (EOFError, KeyboardInterrupt):
                    # Non-interactive or interrupted - default to Control
                    context["condition"] = "Control"
                    print(f"\n  Defaulting to: Control\n")
                    break
        else:
            # Non-interactive mode (e.g., SLURM) - default to Control
            context["condition"] = "Control"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Non-interactive mode: defaulting to 'Control' condition\n")
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Using condition: {context.get('condition', 'Control')}\n")
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading valid perturbation names...")
    # Load valid perturbation names
    drug_names, gene_names = load_valid_perturbation_names()
    print(f"  ✓ Loaded {len(drug_names)} drug names and {len(gene_names)} gene names\n")
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Detecting user intent...")
    # Detect user intent
    intent = detect_user_intent(query)
    print(f"  Mode: {intent['mode']}")
    if intent.get('gene_name'):
        print(f"  Gene detected: {intent['gene_name']}")
    if intent.get('drug_name'):
        print(f"  Drug detected: {intent['drug_name']}")
    if intent.get('has_comparison_phrase'):
        print(f"  Comparison phrase detected: Yes\n")
    else:
        print()
    
    results = {
        "query": query,
        "intent": intent,
        "perturbations": [],
        "comparison": None,
        "outputs": {}
    }
    
    # Validate and run perturbations
    if intent["mode"] == "single_gene":
        gene_name = intent["gene_name"]
        if not gene_name:
            # Try to extract from query using LLM
            pert_info = extract_perturbation_info(query)
            gene_name = pert_info.get("target", "")
        
        if gene_name:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Validating gene name: {gene_name}...")
            match_info = validate_perturbation_name(gene_name, is_gene=True, valid_names=gene_names)
            print(f"  Match type: {match_info['match_type']}")
            if match_info["match_type"] != "none":
                print(f"  Using name: {match_info['used_name']}\n")
            
            results["perturbations"].append({
                "type": "gene",
                "match_info": match_info
            })
            
            if match_info["match_type"] != "none":
                if match_info["match_type"] == "close":
                    # Inform user about close match
                    print(f"⚠️  Close match: Requested '{gene_name}' not found. Closest match: '{match_info['used_name']}' (similarity: {match_info['similarity_score']:.2f})")
                    print(f"   Proceeding with '{match_info['used_name']}'...\n")
                
                # Run perturbation
                pert_results = run_state_gene_perturbation(match_info["used_name"], context)
                if "error" not in pert_results:
                    results["perturbations"][0]["results"] = pert_results
                    # Extract query intent for adaptive output generation
                    query_intent = extract_perturbation_info(query) if query else None
                    results["perturbations"][0]["outputs"] = _generate_outputs(pert_results, "gene", match_info["used_name"], query_intent=query_intent)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Gene perturbation completed successfully\n")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Gene perturbation failed: {pert_results['error']}\n")
                    results["perturbations"][0]["error"] = pert_results["error"]
            else:
                # No match found - check if there was a closest match (even if below threshold)
                closest_name = match_info.get("used_name")
                closest_score = match_info.get("similarity_score", 0.0)
                if closest_name and closest_score > 0.0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ No match found for gene '{gene_name}'")
                    print(f"   Closest match was '{closest_name}' but similarity ({closest_score:.2f}) is too low (< 0.80)")
                    print(f"   There is nothing close to '{gene_name}' in the available gene list.\n")
                    results["perturbations"][0]["error"] = f"No match found for gene '{gene_name}'. Closest match was '{closest_name}' (similarity: {closest_score:.2f}) but too low."
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ No match found for gene '{gene_name}'")
                    print(f"   There is nothing close to '{gene_name}' in the available gene list.\n")
                    results["perturbations"][0]["error"] = f"No match found for gene '{gene_name}'. Nothing close to this name."
    
    elif intent["mode"] == "single_drug":
        drug_name = intent["drug_name"]
        if not drug_name:
            pert_info = extract_perturbation_info(query)
            drug_name = pert_info.get("target", "")
        
        if drug_name:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Validating drug name: {drug_name}...")
            match_info = validate_perturbation_name(drug_name, is_gene=False, valid_names=drug_names)
            print(f"  Match type: {match_info['match_type']}")
            if match_info["match_type"] != "none":
                print(f"  Using name: {match_info['used_name'][:80]}...\n")
            
            results["perturbations"].append({
                "type": "drug",
                "match_info": match_info
            })
            
            if match_info["match_type"] != "none":
                if match_info["match_type"] == "close":
                    print(f"⚠️  Close match: Requested '{drug_name}' not found. Closest match: '{match_info['used_name'][:80]}...' (similarity: {match_info['similarity_score']:.2f})")
                    print(f"   Proceeding with '{match_info['used_name'][:80]}...'...\n")
                
                pert_results = run_state_drug_perturbation(match_info["used_name"], context)
                if "error" not in pert_results:
                    results["perturbations"][0]["results"] = pert_results
                    # Extract query intent for adaptive output generation
                    query_intent = extract_perturbation_info(query) if query else None
                    results["perturbations"][0]["outputs"] = _generate_outputs(pert_results, "drug", match_info["used_name"], query_intent=query_intent)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Drug perturbation completed successfully\n")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Drug perturbation failed: {pert_results['error']}\n")
                    results["perturbations"][0]["error"] = pert_results["error"]
            else:
                # No match found - check if there was a closest match (even if below threshold)
                closest_name = match_info.get("used_name")
                closest_score = match_info.get("similarity_score", 0.0)
                if closest_name and closest_score > 0.0:
                    closest_display = str(closest_name)[:80] + '...' if len(str(closest_name)) > 80 else str(closest_name)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ No match found for drug '{drug_name}'")
                    print(f"   Closest match was '{closest_display}' but similarity ({closest_score:.2f}) is too low (< 0.80)")
                    print(f"   There is nothing close to '{drug_name}' in the available drug list.\n")
                    results["perturbations"][0]["error"] = f"No match found for drug '{drug_name}'. Closest match was '{closest_display}' (similarity: {closest_score:.2f}) but too low."
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ No match found for drug '{drug_name}'")
                    print(f"   There is nothing close to '{drug_name}' in the available drug list.\n")
                    results["perturbations"][0]["error"] = f"No match found for drug '{drug_name}'. Nothing close to this name."
    
    elif intent["mode"] == "comparison":
        # Validate both
        gene_name = intent.get("gene_name")
        drug_name = intent.get("drug_name")
        
        # Extract if not found using LLM as fallback
        if not gene_name or not drug_name:
            pert_info = extract_perturbation_info(query)
            if not gene_name:
                # Try to extract gene from LLM result
                extracted_target = pert_info.get("target", "")
                if extracted_target:
                    # Check if it's a gene (uppercase or matches gene pattern)
                    if extracted_target.isupper() or re.match(r'^[A-Z]{2,10}$', extracted_target):
                        gene_name = extracted_target
            if not drug_name:
                # Try to extract drug from query using word patterns
                # Look for lowercase words that might be drugs (after "or", "vs", etc.)
                words = query.split()
                for i, word in enumerate(words):
                    word_lower = word.lower()
                    # Check if previous word is "or", "vs", "versus"
                    if i > 0 and words[i-1].lower() in {"or", "vs", "versus"}:
                        if word_lower not in {"of", "the", "is", "which", "stronger", "compare", "run", "ko", "knockout", "treatment", "perturbation"} and not word.isupper() and len(word) > 3:
                            drug_name = word
                            break
                # If still not found, try LLM extraction
                if not drug_name:
                    extracted_target = pert_info.get("target", "")
                    if extracted_target and not gene_name:
                        # If no gene found, this might be a drug
                        drug_name = extracted_target
        
        # Validate both perturbations first (for comparison queries, we can run in parallel)
        gene_match = None
        drug_match = None
        
        if gene_name:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Validating gene name: {gene_name}...")
            gene_match = validate_perturbation_name(gene_name, is_gene=True, valid_names=gene_names)
            print(f"  Match type: {gene_match['match_type']}")
            if gene_match["match_type"] != "none":
                print(f"  Using name: {gene_match['used_name']}\n")
        
        if drug_name:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Validating drug name: {drug_name}...")
            drug_match = validate_perturbation_name(drug_name, is_gene=False, valid_names=drug_names)
            print(f"  Match type: {drug_match['match_type']}")
            if drug_match["match_type"] != "none":
                print(f"  Using name: {drug_match['used_name'][:80]}...\n")
        
        # Always run perturbations sequentially to avoid GPU OOM errors and I/O conflicts
        can_run_parallel = (
            gene_name and gene_match and gene_match["match_type"] != "none" and
            drug_name and drug_match and drug_match["match_type"] != "none"
        )
        
        # Sequential execution (always used to avoid GPU memory issues and I/O conflicts)
        if can_run_parallel:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Running perturbations SEQUENTIALLY (to avoid GPU OOM and I/O conflicts)...\n")
            if gene_match["match_type"] == "close":
                print(f"⚠️  Close match for gene: Requested '{gene_name}' not found. Using '{gene_match['used_name']}' (similarity: {gene_match['similarity_score']:.2f})\n")
            if drug_match["match_type"] == "close":
                print(f"⚠️  Close match for drug: Requested '{drug_name}' not found. Using '{drug_match['used_name'][:80]}...' (similarity: {drug_match['similarity_score']:.2f})\n")
            
            # Run gene first, then drug (with GPU cache clearing between)
            if gene_name and gene_match and gene_match["match_type"] != "none":
                gene_results = run_state_gene_perturbation(gene_match["used_name"], context)
                
                # Clear GPU cache after gene perturbation
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Cleared GPU cache after gene perturbation\n")
                except Exception:
                    pass
                
                if "error" not in gene_results:
                    # Extract query intent for adaptive output generation
                    query_intent = extract_perturbation_info(query) if query else None
                    results["perturbations"].append({
                        "type": "gene",
                        "match_info": gene_match,
                        "results": gene_results,
                        "outputs": _generate_outputs(gene_results, "gene", gene_match["used_name"], query_intent=query_intent)
                    })
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Gene perturbation completed successfully\n")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Gene perturbation failed: {gene_results['error']}\n")
                    results["perturbations"].append({
                        "type": "gene",
                        "match_info": gene_match,
                        "error": gene_results["error"]
                    })
            
            # Run drug perturbation
            if drug_name and drug_match and drug_match["match_type"] != "none":
                drug_results = run_state_drug_perturbation(drug_match["used_name"], context)
                
                # Clear GPU cache after drug perturbation
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Cleared GPU cache after drug perturbation\n")
                except Exception:
                    pass
                
                if "error" not in drug_results:
                    # Extract query intent for adaptive output generation
                    query_intent = extract_perturbation_info(query) if query else None
                    results["perturbations"].append({
                        "type": "drug",
                        "match_info": drug_match,
                        "results": drug_results,
                        "outputs": _generate_outputs(drug_results, "drug", drug_match["used_name"], query_intent=query_intent)
                    })
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Drug perturbation completed successfully\n")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Drug perturbation failed: {drug_results['error']}\n")
                    results["perturbations"].append({
                        "type": "drug",
                        "match_info": drug_match,
                        "error": drug_results["error"]
                    })
        
        # Single perturbation execution (for single gene or drug queries)
        else:
            if gene_name and gene_match and gene_match["match_type"] != "none":
                if gene_match["match_type"] == "close":
                    print(f"⚠️  Close match: Requested '{gene_name}' not found. Closest match: '{gene_match['used_name']}' (similarity: {gene_match['similarity_score']:.2f})")
                    print(f"   Proceeding with '{gene_match['used_name']}'...\n")
                
                gene_results = run_state_gene_perturbation(gene_match["used_name"], context)
                if "error" not in gene_results:
                    # Extract query intent for adaptive output generation
                    query_intent = extract_perturbation_info(query) if query else None
                    results["perturbations"].append({
                        "type": "gene",
                        "match_info": gene_match,
                        "results": gene_results,
                        "outputs": _generate_outputs(gene_results, "gene", gene_match["used_name"], query_intent=query_intent)
                    })
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Gene perturbation completed successfully\n")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Gene perturbation failed: {gene_results['error']}\n")
                    results["perturbations"].append({
                        "type": "gene",
                        "match_info": gene_match,
                        "error": gene_results["error"]
                    })
            elif gene_name:
                # No match found - check if there was a closest match (even if below threshold)
                closest_name = gene_match.get("used_name") if gene_match else None
                closest_score = gene_match.get("similarity_score", 0.0) if gene_match else 0.0
                if closest_name and closest_score > 0.0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ No match found for gene '{gene_name}'")
                    print(f"   Closest match was '{closest_name}' but similarity ({closest_score:.2f}) is too low (< 0.80)")
                    print(f"   There is nothing close to '{gene_name}' in the available gene list.\n")
                    results["perturbations"].append({
                        "type": "gene",
                        "match_info": gene_match,
                        "error": f"No match found for gene '{gene_name}'. Closest match was '{closest_name}' (similarity: {closest_score:.2f}) but too low."
                    })
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ No match found for gene '{gene_name}'")
                    print(f"   There is nothing close to '{gene_name}' in the available gene list.\n")
                    results["perturbations"].append({
                        "type": "gene",
                        "match_info": gene_match,
                        "error": f"No match found for gene '{gene_name}'. Nothing close to this name."
                    })
            
            if drug_name and drug_match and drug_match["match_type"] != "none":
                if drug_match["match_type"] == "close":
                    print(f"⚠️  Close match: Requested '{drug_name}' not found. Closest match: '{drug_match['used_name'][:80]}...' (similarity: {drug_match['similarity_score']:.2f})")
                    print(f"   Proceeding with '{drug_match['used_name'][:80]}...'...\n")
                
                drug_results = run_state_drug_perturbation(drug_match["used_name"], context)
                if "error" not in drug_results:
                    # Extract query intent for adaptive output generation
                    query_intent = extract_perturbation_info(query) if query else None
                    results["perturbations"].append({
                        "type": "drug",
                        "match_info": drug_match,
                        "results": drug_results,
                        "outputs": _generate_outputs(drug_results, "drug", drug_match["used_name"], query_intent=query_intent)
                    })
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Drug perturbation completed successfully\n")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Drug perturbation failed: {drug_results['error']}\n")
                    results["perturbations"].append({
                        "type": "drug",
                        "match_info": drug_match,
                        "error": drug_results["error"]
                    })
            elif drug_name:
                # No match found - check if there was a closest match (even if below threshold)
                closest_name = drug_match.get("used_name") if drug_match else None
                closest_score = drug_match.get("similarity_score", 0.0) if drug_match else 0.0
                if closest_name and closest_score > 0.0:
                    closest_display = str(closest_name)[:80] + '...' if len(str(closest_name)) > 80 else str(closest_name)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ No match found for drug '{drug_name}'")
                    print(f"   Closest match was '{closest_display}' but similarity ({closest_score:.2f}) is too low (< 0.80)")
                    print(f"   There is nothing close to '{drug_name}' in the available drug list.\n")
                    results["perturbations"].append({
                        "type": "drug",
                        "match_info": drug_match,
                        "error": f"No match found for drug '{drug_name}'. Closest match was '{closest_display}' (similarity: {closest_score:.2f}) but too low."
                    })
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ No match found for drug '{drug_name}'")
                    print(f"   There is nothing close to '{drug_name}' in the available drug list.\n")
                    results["perturbations"].append({
                        "type": "drug",
                        "match_info": drug_match,
                        "error": f"No match found for drug '{drug_name}'. Nothing close to this name."
                    })
        
        # Generate comparison (only if both perturbations succeeded)
        successful_perturbations = [p for p in results["perturbations"] if "error" not in p and "results" in p]
        if len(successful_perturbations) == 2:
            print(f"\n{'='*70}")
            print(f"STEP 2: Generating Comparison")
            print(f"{'='*70}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Comparing {successful_perturbations[0].get('match_info', {}).get('used_name', 'Perturbation 1')} vs {successful_perturbations[1].get('match_info', {}).get('used_name', 'Perturbation 2')}...\n")
            results["comparison"] = _generate_comparison(
                successful_perturbations[0],
                successful_perturbations[1]
            )
            if "error" not in results["comparison"]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Comparison completed successfully")
                print(f"  - Shared pathways: {len(results['comparison'].get('shared_pathways', []))}")
                print(f"  - Shared phenotypes: {len(results['comparison'].get('shared_phenotypes', []))}\n")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Comparison failed: {results['comparison'].get('error')}\n")
        elif len(successful_perturbations) == 1:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Comparison skipped: Only 1 perturbation succeeded (need 2 for comparison)\n")
            results["comparison"] = {"error": "Comparison requires both perturbations to succeed"}
    
    # Generate hypotheses at the END (after all perturbations complete)
    # Only for comparison queries (both drug and gene perturbations)
    if len(successful_perturbations) == 2:
        print(f"\n{'='*70}")
        print(f"STEP 3: Generating Hypotheses")
        print(f"{'='*70}\n")
        
        try:
            # Combine data from both perturbations for hypothesis generation
            pert1_outputs = successful_perturbations[0].get("outputs", {})
            pert2_outputs = successful_perturbations[1].get("outputs", {})
            
            pert1_name = successful_perturbations[0].get("match_info", {}).get("used_name", "Perturbation 1")
            pert2_name = successful_perturbations[1].get("match_info", {}).get("used_name", "Perturbation 2")
            
            # Combine DEGs, pathways from both perturbations
            combined_deg_list = pert1_outputs.get("_deg_list", []) + pert2_outputs.get("_deg_list", [])
            combined_pathways = pert1_outputs.get("_pathways", []) + pert2_outputs.get("_pathways", [])
            combined_validated_edges = pert1_outputs.get("_validated_edges", []) + pert2_outputs.get("_validated_edges", [])
            
            # Create context for hypothesis generation
            payload = {
                "context": {
                    "perturbation": f"{pert1_name} vs {pert2_name}",
                    "perturbation_type": "comparison",
                    "perturbation1": pert1_name,
                    "perturbation2": pert2_name
                },
                "deg_list": combined_deg_list,
                "pathways": combined_pathways,
                "phenotypes": [],  # Empty (PhenotypeKB removed)
                "validated_edges": combined_validated_edges
            }
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating hypotheses...")
            print(f"  This will:")
            print(f"    1. Use LLM to analyze all combined results")
            print(f"    2. Generate mechanistic hypotheses")
            print(f"    3. Save preliminary report (before literature search)")
            print(f"    4. Search literature via Edison Scientific (PaperQA)")
            print(f"    5. Add literature results to final report\n")
            sys.stdout.flush()
            
            # Step 1: Analyze combined results and generate LLM hypotheses
            print(f"[{datetime.now().strftime('%H:%M:%S')}] STEP 1: Analyzing combined results and generating LLM hypotheses...")
            print(f"  Note: This will generate hypotheses using LLM, then perform literature search")
            print(f"  Literature search may take 1-3 minutes per hypothesis...")
            sys.stdout.flush()
            
            # Generate hypotheses (this will do LLM generation first, then literature search)
            # Pass results/query/context so preliminary report can be saved IMMEDIATELY after LLM generation, before literature search
            try:
                hypotheses = generate_hypotheses(
                    payload,
                    save_preliminary_report={
                        "results": results,
                        "query": query,
                        "context": context
                    }
                )
            except Exception as e:
                print(f"  ✗ Error during hypothesis generation: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                # Try to extract any LLM hypotheses that were generated before the error
                hypotheses = {"hypotheses": [], "llm_generated_hypotheses_before_literature": []}
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] STEP 1: ✓ Completed hypothesis generation (LLM + literature search)")
            sys.stdout.flush()
            
            # Extract LLM-generated hypotheses (for reference, but preliminary report already saved)
            llm_generated_hypotheses = hypotheses.get("llm_generated_hypotheses_before_literature", [])
            
            # Check if preliminary report was saved (it should have been saved inside generate_hypotheses)
            if results.get("preliminary_report"):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] STEP 2: ✓ Preliminary report already saved (before literature search)")
                print(f"  Saved: {results.get('preliminary_report')}\n")
                sys.stdout.flush()
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] STEP 2: ⚠️  Preliminary report was not saved\n")
                sys.stdout.flush()
            
            # Store hypotheses (will be enriched with literature)
            results["hypotheses"] = hypotheses
            
            # Save hypotheses.json in main output directory
            main_output_dir = Path(__file__).parent / "perturbation_outputs"
            main_output_dir.mkdir(parents=True, exist_ok=True)
            hypotheses_path = main_output_dir / "hypotheses.json"
            import json
            with open(hypotheses_path, 'w') as f:
                json.dump(hypotheses, f, indent=2)
            results["hypotheses_file"] = str(hypotheses_path)
            
            num_hypotheses = len(hypotheses.get('hypotheses', []))
            if num_hypotheses > 0:
                print(f"  ✓ Generated {num_hypotheses} hypotheses")
            else:
                print(f"  ⚠️  Generated 0 hypotheses")
                explanation = hypotheses.get('no_hypotheses_explanation')
                if explanation:
                    print(f"    Explanation: {explanation}")
                lit_search_performed = hypotheses.get('literature_search_performed', False)
                if lit_search_performed:
                    num_papers = hypotheses.get('literature_papers_found', 0)
                    print(f"    Literature search: {num_papers} papers found")
            print(f"    Saved: {results['hypotheses_file']}\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            results["hypotheses"] = None
            results["hypotheses_file"] = None
    
    # Generate comprehensive final report after all perturbations, comparisons, and hypotheses
    print(f"\n{'='*70}")
    print(f"STEP 4: Generating Comprehensive Final Report")
    print(f"{'='*70}\n")
    
    try:
        final_report = _generate_comprehensive_report(results, query, context)
        if final_report:
            # Save in main output directory
            main_output_dir = Path(__file__).parent / "perturbation_outputs"
            main_output_dir.mkdir(parents=True, exist_ok=True)
            report_path = main_output_dir / "comprehensive_report.md"
            with open(report_path, 'w') as f:
                f.write(final_report)
            results["final_report"] = str(report_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Comprehensive report generated")
            print(f"  Saved: {report_path}")
            print(f"  Length: {len(final_report)} characters\n")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Could not generate comprehensive report\n")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed to generate comprehensive report: {e}\n")
        import traceback
        traceback.print_exc()
        results["final_report"] = None
    
    end_time = datetime.now()
    
    print(f"\n{'='*70}")
    print(f"ORCHESTRATOR COMPLETED")
    print(f"{'='*70}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total perturbations: {len(results['perturbations'])}")
    print(f"Successful: {sum(1 for p in results['perturbations'] if 'error' not in p)}")
    print(f"Comparison: {'✓ Generated' if results.get('comparison') and 'error' not in results['comparison'] else '✗ Not generated'}")
    print(f"Final Report: {'✓ Generated' if results.get('final_report') else '✗ Not generated'}")
    print(f"{'='*70}\n")
    
    return results


def _generate_outputs(pert_results: Dict[str, Any], pert_type: str, pert_name: str, query_intent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate all outputs (plots, hypotheses, reports) for a perturbation.
    
    Args:
        pert_results: Results from Agent_Tools (from collect_results_from_pipeline)
        pert_type: "gene" or "drug"
        pert_name: Perturbation name
        query_intent: Optional query intent dict from extract_perturbation_info with:
            - focus: "genes", "proteins", "pathways", "phenotypes", "both", or null
            - top_n_genes: integer or null
            - top_n_proteins: integer or null
            - top_n_pathways: integer or null
            - top_n_phenotypes: integer or null
            - protein_mentioned: boolean
            - output_types: list of desired outputs
    
    Returns:
        Dict with output file paths
    """
    from datetime import datetime
    
    print(f"\n{'='*70}")
    print(f"STEP 2: Generating Outputs for {pert_type.upper()}: {pert_name}")
    print(f"{'='*70}\n")
    
    # Sanitize pert_name for filesystem
    safe_name = str(pert_name)[:50].replace('/', '_').replace('(', '_').replace(')', '_').replace("'", '_')
    # Save all outputs directly in perturbation directory (like test_output/)
    output_dir = Path(__file__).parent / "perturbation_outputs" / f"{pert_type}_{safe_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    # Extract data from results
    # collect_results_from_pipeline returns:
    # - pathway_analysis: dict with differential_rna, differential_protein, gsea_rna, kegg_enrichment, etc.
    pathway_analysis = pert_results.get("pathway_analysis", {})
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading differential expression data...")
    sys.stdout.flush()  # Flush to ensure output appears in log
    
    # Convert differential expression DataFrames to deg_list format
    deg_list = []
    if "differential_rna" in pathway_analysis and pathway_analysis["differential_rna"] is not None:
        try:
            import pandas as pd
            de_rna = pathway_analysis["differential_rna"]
            if isinstance(de_rna, pd.DataFrame):
                for _, row in de_rna.iterrows():
                    # Preserve expression values (pred_mean, control_mean) in addition to stats
                    deg_dict = {
                        "gene": str(row.get("gene", "")),
                        "log2fc": float(row.get("log2fc", 0.0)),
                        "pval": float(row.get("pvalue", row.get("pval", 1.0))),
                        "pval_adj": float(row.get("pvalue_adj", row.get("pval_adj", 1.0)))
                    }
                    # Include expression values if available
                    if "pred_mean" in row:
                        deg_dict["pred_mean"] = float(row.get("pred_mean", 0.0))
                    if "control_mean" in row:
                        deg_dict["control_mean"] = float(row.get("control_mean", 0.0))
                    deg_list.append(deg_dict)
                print(f"  ✓ Loaded {len(deg_list)} DEGs with expression values")
            else:
                print(f"  ⚠️  Differential expression data is not a DataFrame")
        except Exception as e:
            print(f"  ✗ Failed to load DEGs: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ⚠️  No differential expression data available in pathway_analysis")
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading pathway enrichment data...")
    sys.stdout.flush()  # Flush to ensure output appears in log
    
    # Convert pathway enrichment to pathways format
    pathways = []
    # Collect pathways from both RNA and protein enrichments
    for db in ["gsea_rna", "kegg_enrichment", "reactome_enrichment", "go_enrichment", 
               "kegg_enrichment_rna", "reactome_enrichment_rna", "go_enrichment_rna"]:
        if db in pathway_analysis and pathway_analysis[db] is not None:
            try:
                import pandas as pd
                enrichment = pathway_analysis[db]
                if isinstance(enrichment, pd.DataFrame):
                    for _, row in enrichment.iterrows():
                        pathway_name = row.get("Term", "") or row.get("pathway", "") or str(row.name)
                        
                        # Extract NES (GSEA has it, Enrichr doesn't - use log10(p-value) as proxy)
                        nes = row.get("NES", row.get("Normalized Enrichment", None))
                        if nes is None or pd.isna(nes):
                            # For Enrichr results without NES, use -log10(p-value) as a proxy
                            pval = row.get("P-value", row.get("p-value", None))
                            if pval is not None and not pd.isna(pval):
                                try:
                                    pval_float = float(pval)
                                    if pval_float > 0:
                                        import math
                                        nes = -math.log10(pval_float)
                                    else:
                                        nes = 0.0
                                except (ValueError, TypeError):
                                    nes = 0.0
                            else:
                                nes = 0.0
                        nes = float(nes)
                        
                        # Extract p-value (regular, not adjusted)
                        pval = row.get("pval", row.get("P-value", row.get("p-value", row.get("NOM p-val", 1.0))))
                        if pval is None or pd.isna(pval):
                            pval = 1.0
                        pval = float(pval)
                        
                        # Extract FDR (Adjusted P-value for Enrichr, FDR q-val for GSEA) - keep for reference
                        fdr = row.get("FDR", row.get("FDR q-val", row.get("Adjusted P-value", 1.0)))
                        if fdr is None or pd.isna(fdr):
                            fdr = 1.0
                        fdr = float(fdr)
                        
                        # Extract member genes if available
                        member_genes = []
                        genes_str = row.get("Genes", row.get("Lead_genes", row.get("member_genes", "")))
                        if genes_str and not pd.isna(genes_str):
                            member_genes = [g.strip() for g in str(genes_str).split(';') if g.strip()]
                        
                        pathways.append({
                            "id": f"{db.upper()}_{pathway_name}",
                            "name": pathway_name,
                            "source": db,
                            "NES": nes,
                            "pval": pval,  # Regular p-value (not adjusted)
                            "FDR": fdr,    # Keep FDR for reference
                            "member_genes": member_genes
                        })
                print(f"  ✓ Loaded {len([p for p in pathways if p['source'] == db])} pathways from {db}")
            except Exception as e:
                print(f"  ✗ Failed to process {db} pathways: {e}")
                import traceback
                traceback.print_exc()
                pass
    
    print(f"  Total pathways loaded: {len(pathways)}\n")
    
    # PhenotypeKB scoring removed - no longer scoring phenotypes per perturbation
    phenotypes = []
    
    validated_edges = pert_results.get("validated_edges", [])
    
    # Extract protein differential expression if available
    protein_deg_list = []
    if "differential_protein" in pathway_analysis and pathway_analysis["differential_protein"] is not None:
        try:
            import pandas as pd
            de_protein = pathway_analysis["differential_protein"]
            if isinstance(de_protein, pd.DataFrame):
                for _, row in de_protein.iterrows():
                    protein_dict = {
                        "gene": str(row.get("gene", "")),
                        "log2fc": float(row.get("log2fc", 0.0)),
                        "pval": float(row.get("pvalue", row.get("pval", 1.0))),
                        "pval_adj": float(row.get("pvalue_adj", row.get("pval_adj", 1.0)))
                    }
                    if "pred_mean" in row:
                        protein_dict["pred_mean"] = float(row.get("pred_mean", 0.0))
                    if "control_mean" in row:
                        protein_dict["control_mean"] = float(row.get("control_mean", 0.0))
                    protein_deg_list.append(protein_dict)
                print(f"  ✓ Loaded {len(protein_deg_list)} differentially expressed proteins\n")
        except Exception as e:
            print(f"  ⚠️  Failed to load protein differential expression: {e}\n")
    
    # Apply query-based filtering (top N, focus)
    query_intent = query_intent or {}
    focus = query_intent.get("focus")  # "genes", "proteins", "pathways", "phenotypes", "both", null
    top_n_genes = query_intent.get("top_n_genes")
    top_n_proteins = query_intent.get("top_n_proteins")
    top_n_pathways = query_intent.get("top_n_pathways")
    top_n_phenotypes = query_intent.get("top_n_phenotypes")
    protein_mentioned = query_intent.get("protein_mentioned", False)
    output_types = query_intent.get("output_types")  # List of desired outputs or null for all
    
    # Filter DEGs if top_n_genes specified or focus is on genes
    filtered_deg_list = deg_list.copy()
    if top_n_genes or (focus in ["genes", "both"] and not top_n_proteins):
        n = top_n_genes or 10  # Default to 10 if focus is genes but no number specified
        # Sort by absolute log2FC and take top N
        filtered_deg_list = sorted(deg_list, key=lambda x: abs(x.get("log2fc", 0.0)), reverse=True)[:n]
        if len(deg_list) > len(filtered_deg_list):
            print(f"  📊 Filtered to top {len(filtered_deg_list)} genes (by |log2FC|)")
    
    # Filter proteins if top_n_proteins specified or focus is on proteins
    filtered_protein_list = protein_deg_list.copy()
    if top_n_proteins or (focus in ["proteins", "both"] or protein_mentioned):
        n = top_n_proteins or 10  # Default to 10 if focus is proteins but no number specified
        if protein_deg_list:
            filtered_protein_list = sorted(protein_deg_list, key=lambda x: abs(x.get("log2fc", 0.0)), reverse=True)[:n]
            if len(protein_deg_list) > len(filtered_protein_list):
                print(f"  📊 Filtered to top {len(filtered_protein_list)} proteins (by |log2FC|)")
    
    # Filter pathways if top_n_pathways specified
    filtered_pathways = pathways.copy()
    if top_n_pathways:
        # Sort by absolute NES and take top N
        filtered_pathways = sorted(pathways, key=lambda x: abs(x.get("NES", 0.0)), reverse=True)[:top_n_pathways]
        if len(pathways) > len(filtered_pathways):
            print(f"  📊 Filtered to top {len(filtered_pathways)} pathways (by |NES|)")
    
    # PhenotypeKB removed - no phenotype filtering needed
    filtered_phenotypes = []
    
    # Determine which plots to generate based on query intent
    # Always generate plots if we have data (unless user specifically excludes them)
    generate_all = output_types is None or "all" in output_types or len(output_types) == 0
    generate_plots = generate_all or "plots" in output_types
    
    # Generate volcano plot if we have DEGs (always generate if data available)
    generate_genes_plot = (generate_all or "genes" in output_types or focus in ["genes", "both", None]) and len(deg_list) > 0
    
    # Generate PSEA plot if we have protein data
    generate_proteins_plot = (generate_all or "proteins" in output_types or focus in ["proteins", "both"] or protein_mentioned) and len(protein_deg_list) > 0
    
    # Generate pathway plots if we have pathways (always generate if data available)
    generate_pathways_plot = (generate_all or "pathways" in output_types or focus in ["pathways", None]) and len(pathways) > 0
    
    # Phenotype plots removed (PhenotypeKB removed)
    generate_phenotypes_plot = False
    
    # Generate plots with proper labeling (separate for each analysis type)
    # Each plot gets a label indicating perturbation type and name
    pert_label = f"{pert_type.upper()}: {pert_name}"
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating plots...")
    print(f"  Output directory: {output_dir}")
    if query_intent:
        print(f"  Query focus: {focus or 'all'}")
        if top_n_genes or top_n_proteins or top_n_pathways:
            print(f"  Top N filters: genes={top_n_genes}, proteins={top_n_proteins}, pathways={top_n_pathways}")
    print()
    
    # Use filtered data for plots based on query intent
    plot_deg_list = filtered_deg_list if (focus in ["genes", "both", None] or not protein_mentioned) else deg_list
    plot_pathways = filtered_pathways if generate_pathways_plot else pathways
    
    # Determine top_n for plots based on query intent
    plot_top_n_pathways = top_n_pathways or 20
    
    plot_num = 0
    total_plots = sum([generate_genes_plot, generate_pathways_plot, generate_proteins_plot, generate_phenotypes_plot])
    
    # Use same file names as test_output/ (no prefix/suffix in filename)
    if generate_genes_plot and len(plot_deg_list) > 0:
        plot_num += 1
        try:
            print(f"  [{plot_num}/{total_plots}] Generating volcano plot (genes, log2FC vs -log10(p-value))...")
            plot_path = output_dir / "volcano.png"
            outputs["volcano"] = str(plot_volcano(plot_deg_list, str(plot_path), title_suffix=pert_label))
            print(f"    ✓ Saved: {outputs['volcano']}")
            sys.stdout.flush()
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            outputs["volcano"] = None
    elif generate_genes_plot and len(plot_deg_list) == 0:
        print(f"  ⚠️  Skipping volcano plot: No differential expression data available")
        outputs["volcano"] = None
    
    if generate_pathways_plot and len(plot_pathways) > 0:
        plot_num += 1
        try:
            print(f"  [{plot_num}/{total_plots}] Generating pathway enrichment plot...")
            plot_path = output_dir / "pathway_enrichment.png"
            outputs["pathway_enrichment"] = str(plot_pathway_enrichment(plot_pathways, str(plot_path), top_n=plot_top_n_pathways, title_suffix=pert_label))
            print(f"    ✓ Saved: {outputs['pathway_enrichment']}")
            sys.stdout.flush()
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            outputs["pathway_enrichment"] = None
        
        # Generate RNA GSEA plot if we have both DEGs and pathways (after pathway enrichment plot)
        if generate_genes_plot and len(plot_deg_list) > 0 and len(plot_pathways) > 0:
            plot_num += 1
            try:
                print(f"  [{plot_num}/{total_plots}] Generating RNA GSEA plot (horizontal bar chart)...")
                plot_path = output_dir / "rna_gsea.png"
                outputs["rna_gsea"] = str(plot_rna_gsea(plot_deg_list, plot_pathways, str(plot_path), top_pathways=plot_top_n_pathways, title_suffix=pert_label))
                print(f"    ✓ Saved: {outputs['rna_gsea']}")
                
                # Also create alias gsea.png for backward compatibility with test_output
                import shutil
                gsea_alias = output_dir / "gsea.png"
                shutil.copy(plot_path, gsea_alias)
                print(f"    ✓ Also saved as: {gsea_alias} (alias)")
                sys.stdout.flush()
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                outputs["rna_gsea"] = None
    elif generate_pathways_plot and len(plot_pathways) == 0:
        print(f"  ⚠️  Skipping pathway enrichment plot: No pathway data available")
        outputs["pathway_enrichment"] = None
        outputs["rna_gsea"] = None
    
    if generate_proteins_plot and len(protein_deg_list) > 0 and len(plot_pathways) > 0:
        plot_num += 1
        try:
            print(f"  [{plot_num}/{total_plots}] Generating protein PSEA plot (horizontal bar chart)...")
            plot_path = output_dir / "protein_psea.png"
            # Use protein_deg_list for protein PSEA
            outputs["protein_psea"] = str(plot_protein_psea(filtered_protein_list, validated_edges, plot_pathways, str(plot_path), top_pathways=plot_top_n_pathways, title_suffix=pert_label))
            print(f"    ✓ Saved: {outputs['protein_psea']}")
            
            # Also create alias psea.png for backward compatibility with test_output
            import shutil
            psea_alias = output_dir / "psea.png"
            shutil.copy(plot_path, psea_alias)
            print(f"    ✓ Also saved as: {psea_alias} (alias)")
            sys.stdout.flush()
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            outputs["protein_psea"] = None
    elif generate_proteins_plot:
        if len(protein_deg_list) == 0:
            print(f"  ⚠️  Skipping protein PSEA plot: No protein differential expression data available")
        elif len(plot_pathways) == 0:
            print(f"  ⚠️  Skipping protein PSEA plot: No pathway data available")
        outputs["protein_psea"] = None
    
    # Phenotype plots removed (PhenotypeKB removed)
    outputs["phenotype_enrichment"] = None
    outputs["phenotype_scores"] = None
    
    # Save filtered data to CSV files if requested
    if output_types and ("genes" in output_types or generate_all):
        try:
            import pandas as pd
            genes_csv = output_dir / "top_genes.csv"
            df_genes = pd.DataFrame(filtered_deg_list)
            df_genes.to_csv(genes_csv, index=False)
            outputs["genes_csv"] = str(genes_csv)
            print(f"  ✓ Saved top {len(filtered_deg_list)} genes to: {outputs['genes_csv']}")
        except Exception as e:
            print(f"  ⚠️  Failed to save genes CSV: {e}")
    
    if (output_types and "proteins" in output_types) or (generate_proteins_plot and protein_deg_list):
        try:
            import pandas as pd
            proteins_csv = output_dir / "top_proteins.csv"
            df_proteins = pd.DataFrame(filtered_protein_list)
            df_proteins.to_csv(proteins_csv, index=False)
            outputs["proteins_csv"] = str(proteins_csv)
            print(f"  ✓ Saved top {len(filtered_protein_list)} proteins to: {outputs['proteins_csv']}")
        except Exception as e:
            print(f"  ⚠️  Failed to save proteins CSV: {e}")
    
    if output_types and "pathways" in output_types:
        try:
            import pandas as pd
            pathways_csv = output_dir / "top_pathways.csv"
            df_pathways = pd.DataFrame(filtered_pathways)
            df_pathways.to_csv(pathways_csv, index=False)
            outputs["pathways_csv"] = str(pathways_csv)
            print(f"  ✓ Saved top {len(filtered_pathways)} pathways to: {outputs['pathways_csv']}")
        except Exception as e:
            print(f"  ⚠️  Failed to save pathways CSV: {e}")
    
    # Hypothesis generation moved to the end - after all perturbations complete
    # Store data needed for hypothesis generation at the end
    outputs["_hypotheses"] = None
    outputs["_hypotheses_file"] = None
    
    # Report generation moved to the end - after all perturbations and comparisons
    # Store data needed for final comprehensive report and hypothesis generation
    outputs["_pathways"] = pathways
    outputs["_phenotypes"] = phenotypes  # Empty list (PhenotypeKB removed)
    outputs["_deg_list"] = deg_list
    outputs["_validated_edges"] = validated_edges
    outputs["_report_data_ready"] = True
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ All outputs generated for {pert_type.upper()}: {pert_name}")
    print(f"{'='*70}\n")
    
    return outputs


def _generate_comparison(pert1: Dict[str, Any], pert2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comparison summary between two perturbations.
    
    Args:
        pert1: First perturbation results
        pert2: Second perturbation results
    
    Returns:
        Comparison dict
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting pathways and phenotypes for comparison...")
    
    pert1_results = pert1.get("results", {})
    pert2_results = pert2.get("results", {})
    
    # Extract pathways from pathway_analysis (same way as _generate_outputs)
    pert1_pathway_analysis = pert1_results.get("pathway_analysis", {})
    pert2_pathway_analysis = pert2_results.get("pathway_analysis", {})
    
    print(f"  Processing perturbation 1 pathways...")
    
    pert1_pathways = []
    for db in ["gsea_rna", "kegg_enrichment", "reactome_enrichment", "go_enrichment"]:
        if db in pert1_pathway_analysis and pert1_pathway_analysis[db] is not None:
            try:
                import pandas as pd
                enrichment = pert1_pathway_analysis[db]
                if isinstance(enrichment, pd.DataFrame):
                    for _, row in enrichment.iterrows():
                        pathway_name = row.get("Term", "") or row.get("pathway", "") or str(row.name)
                        pert1_pathways.append({
                            "id": f"{db.upper()}_{pathway_name}",
                            "name": pathway_name,
                            "source": db,
                            "NES": float(row.get("NES", row.get("Normalized Enrichment", 0.0))),
                            "FDR": float(row.get("FDR", row.get("FDR q-val", row.get("Adjusted P-value", 1.0)))),
                            "member_genes": []
                        })
            except Exception:
                pass
    
    pert2_pathways = []
    for db in ["gsea_rna", "kegg_enrichment", "reactome_enrichment", "go_enrichment"]:
        if db in pert2_pathway_analysis and pert2_pathway_analysis[db] is not None:
            try:
                import pandas as pd
                enrichment = pert2_pathway_analysis[db]
                if isinstance(enrichment, pd.DataFrame):
                    for _, row in enrichment.iterrows():
                        pathway_name = row.get("Term", "") or row.get("pathway", "") or str(row.name)
                        pert2_pathways.append({
                            "id": f"{db.upper()}_{pathway_name}",
                            "name": pathway_name,
                            "source": db,
                            "NES": float(row.get("NES", row.get("Normalized Enrichment", 0.0))),
                            "FDR": float(row.get("FDR", row.get("FDR q-val", row.get("Adjusted P-value", 1.0)))),
                            "member_genes": []
                        })
            except Exception:
                pass
    
    # Extract phenotypes: need to compute them from DEGs and pathways (same as _generate_outputs)
    pert1_deg_list = []
    if "differential_rna" in pert1_pathway_analysis and pert1_pathway_analysis["differential_rna"] is not None:
        try:
            import pandas as pd
            de_rna = pert1_pathway_analysis["differential_rna"]
            if isinstance(de_rna, pd.DataFrame):
                for _, row in de_rna.iterrows():
                    deg_dict = {
                        "gene": str(row.get("gene", "")),
                        "log2fc": float(row.get("log2fc", 0.0)),
                        "pval": float(row.get("pvalue", row.get("pval", 1.0))),
                        "pval_adj": float(row.get("pvalue_adj", row.get("pval_adj", 1.0)))
                    }
                    # Include expression values if available
                    if "pred_mean" in row:
                        deg_dict["pred_mean"] = float(row.get("pred_mean", 0.0))
                    if "control_mean" in row:
                        deg_dict["control_mean"] = float(row.get("control_mean", 0.0))
                    pert1_deg_list.append(deg_dict)
        except Exception:
            pass
    
    pert2_deg_list = []
    if "differential_rna" in pert2_pathway_analysis and pert2_pathway_analysis["differential_rna"] is not None:
        try:
            import pandas as pd
            de_rna = pert2_pathway_analysis["differential_rna"]
            if isinstance(de_rna, pd.DataFrame):
                for _, row in de_rna.iterrows():
                    deg_dict = {
                        "gene": str(row.get("gene", "")),
                        "log2fc": float(row.get("log2fc", 0.0)),
                        "pval": float(row.get("pvalue", row.get("pval", 1.0))),
                        "pval_adj": float(row.get("pvalue_adj", row.get("pval_adj", 1.0)))
                    }
                    # Include expression values if available
                    if "pred_mean" in row:
                        deg_dict["pred_mean"] = float(row.get("pred_mean", 0.0))
                    if "control_mean" in row:
                        deg_dict["control_mean"] = float(row.get("control_mean", 0.0))
                    pert2_deg_list.append(deg_dict)
        except Exception:
            pass
    
    # PhenotypeKB removed - no phenotype scoring
    pert1_phenotypes = []
    pert2_phenotypes = []
    
    print(f"  ✓ Perturbation 1: {len(pert1_pathways)} pathways, {len(pert1_phenotypes)} phenotypes")
    print(f"  ✓ Perturbation 2: {len(pert2_pathways)} pathways, {len(pert2_phenotypes)} phenotypes")
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Comparing shared pathways and phenotypes...\n")
    
    # Compare shared pathways
    pert1_pathway_dict = {p.get("id"): p for p in pert1_pathways}
    pert2_pathway_dict = {p.get("id"): p for p in pert2_pathways}
    
    shared_pathways = []
    for pathway_id in set(pert1_pathway_dict.keys()) & set(pert2_pathway_dict.keys()):
        p1 = pert1_pathway_dict[pathway_id]
        p2 = pert2_pathway_dict[pathway_id]
        
        p1_nes = abs(p1.get("NES", 0.0))
        p2_nes = abs(p2.get("NES", 0.0))
        
        shared_pathways.append({
            "pathway_id": pathway_id,
            "pathway_name": p1.get("name", ""),
            "pert1_nes": p1.get("NES", 0.0),
            "pert2_nes": p2.get("NES", 0.0),
            "pert1_strength": p1_nes,
            "pert2_strength": p2_nes,
            "stronger": pert1.get("match_info", {}).get("used_name", "Perturbation 1") if p1_nes > p2_nes else pert2.get("match_info", {}).get("used_name", "Perturbation 2"),
            "direction_pert1": "activation" if p1.get("NES", 0.0) > 0 else "repression",
            "direction_pert2": "activation" if p2.get("NES", 0.0) > 0 else "repression"
        })
    
    # Compare shared phenotypes
    pert1_pheno_dict = {p.get("phenotype_id"): p for p in pert1_phenotypes}
    pert2_pheno_dict = {p.get("phenotype_id"): p for p in pert2_phenotypes}
    
    shared_phenotypes = []
    for pheno_id in set(pert1_pheno_dict.keys()) & set(pert2_pheno_dict.keys()):
        p1 = pert1_pheno_dict[pheno_id]
        p2 = pert2_pheno_dict[pheno_id]
        
        p1_score = p1.get("score", 0.0)
        p2_score = p2.get("score", 0.0)
        
        shared_phenotypes.append({
            "phenotype_id": pheno_id,
            "phenotype_name": p1.get("name", ""),
            "pert1_score": p1_score,
            "pert2_score": p2_score,
            "pert1_direction": p1.get("direction", "mixed"),
            "pert2_direction": p2.get("direction", "mixed"),
            "stronger": pert1.get("match_info", {}).get("used_name", "Perturbation 1") if p1_score > p2_score else pert2.get("match_info", {}).get("used_name", "Perturbation 2")
        })
    
    # Generate comparison summary text
    summary_lines = []
    summary_lines.append("## Comparison Summary")
    summary_lines.append("")
    
    pert1_name = pert1.get("match_info", {}).get("used_name", "Perturbation 1")
    pert2_name = pert2.get("match_info", {}).get("used_name", "Perturbation 2")
    
    summary_lines.append(f"**Comparison: {pert1_name} vs {pert2_name}**")
    summary_lines.append("")
    
    # Pathway comparisons
    if shared_pathways:
        summary_lines.append("### Pathway Effects")
        summary_lines.append("")
        
        for pathway in shared_pathways[:10]:  # Top 10
            name = pathway["pathway_name"]
            p1_nes = pathway["pert1_nes"]
            p2_nes = pathway["pert2_nes"]
            stronger = pathway["stronger"]
            dir1 = pathway["direction_pert1"]
            dir2 = pathway["direction_pert2"]
            
            if dir1 == dir2:
                summary_lines.append(f"- **{name}**: Both {dir1}, but {stronger} is stronger (NES {p1_nes:.2f} vs {p2_nes:.2f})")
            else:
                summary_lines.append(f"- **{name}**: {pert1_name} {dir1} (NES {p1_nes:.2f}), {pert2_name} {dir2} (NES {p2_nes:.2f})")
    
    # Phenotype comparisons
    if shared_phenotypes:
        summary_lines.append("")
        summary_lines.append("### Phenotype Effects")
        summary_lines.append("")
        
        for pheno in shared_phenotypes[:10]:  # Top 10
            name = pheno["phenotype_name"]
            p1_score = pheno["pert1_score"]
            p2_score = pheno["pert2_score"]
            stronger = pheno["stronger"]
            dir1 = pheno["pert1_direction"]
            dir2 = pheno["pert2_direction"]
            
            if dir1 == dir2:
                summary_lines.append(f"- **{name}**: Both {dir1}, but {stronger} is stronger (score {p1_score:.2f} vs {p2_score:.2f})")
            else:
                summary_lines.append(f"- **{name}**: {pert1_name} {dir1} (score {p1_score:.2f}), {pert2_name} {dir2} (score {p2_score:.2f})")
    
    summary_text = "\n".join(summary_lines)
    
    print(f"  ✓ Found {len(shared_pathways)} shared pathways")
    print(f"  ✓ Found {len(shared_phenotypes)} shared phenotypes\n")
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving comparison results...")
    # Save comparison results in comparison/ directory
    comparison_dir = Path(__file__).parent / "perturbation_outputs" / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comparison report
    try:
        report_path = comparison_dir / "comparison_report.md"
        with open(report_path, 'w') as f:
            f.write("# Perturbation Comparison Report\n\n")
            f.write(summary_text)
        comparison_report_path = str(report_path)
        print(f"  ✓ Saved comparison report: {comparison_report_path}")
    except Exception as e:
        print(f"  ✗ Failed to save comparison report: {e}")
        comparison_report_path = None
    
    # Save comparison JSON
    try:
        import json
        comparison_json = {
            "perturbation1": pert1.get("match_info", {}).get("used_name", "Perturbation 1"),
            "perturbation2": pert2.get("match_info", {}).get("used_name", "Perturbation 2"),
            "shared_pathways": shared_pathways,
            "shared_phenotypes": shared_phenotypes,
            "summary": summary_text
        }
        json_path = comparison_dir / "comparison.json"
        with open(json_path, 'w') as f:
            json.dump(comparison_json, f, indent=2)
        comparison_json_path = str(json_path)
        print(f"  ✓ Saved comparison JSON: {comparison_json_path}\n")
    except Exception as e:
        print(f"  ✗ Failed to save comparison JSON: {e}\n")
        comparison_json_path = None
    
    return {
        "shared_pathways": shared_pathways,
        "shared_phenotypes": shared_phenotypes,
        "summary": summary_text,
        "comparison_report": comparison_report_path,
        "comparison_json": comparison_json_path
    }


def _generate_preliminary_report(
    results: Dict[str, Any], 
    query: str, 
    context: Dict[str, Any],
    llm_hypotheses: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Generate preliminary report with everything EXCEPT literature search results.
    
    This report is saved immediately after LLM hypothesis generation, before
    literature search runs. Contains all perturbations, comparison, and LLM-generated
    hypotheses with full details.
    
    Args:
        results: Results dict with perturbations and comparison
        query: User query
        context: Context dict
        llm_hypotheses: LLM-generated hypotheses (before literature enrichment)
    
    Returns:
        Markdown string with preliminary report
    """
    from .report import build_report
    
    lines = []
    lines.append("# Preliminary Perturbation Analysis Report")
    lines.append("")
    lines.append(f"**Query:** {query}")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("> **Note:** This is a preliminary report generated before literature search.")
    lines.append("> Literature search results will be added to the final comprehensive report.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Summary
    successful = [p for p in results.get("perturbations", []) if "error" not in p and "results" in p]
    lines.append(f"**Total Perturbations:** {len(results.get('perturbations', []))}")
    lines.append(f"**Successful:** {len(successful)}")
    if results.get("comparison") and "error" not in results.get("comparison", {}):
        comp = results["comparison"]
        lines.append(f"**Comparison:** ✓ Generated ({len(comp.get('shared_pathways', []))} shared pathways)")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Individual perturbation reports
    for i, pert in enumerate(successful, 1):
        pert_type = pert.get("type", "unknown")
        pert_name = pert.get("match_info", {}).get("used_name", pert.get("name", "Unknown"))
        outputs = pert.get("outputs", {})
        
        lines.append(f"## Perturbation {i}: {pert_name} ({pert_type})")
        lines.append("")
        
        # Get stored report data
        pathways = outputs.get("_pathways", [])
        phenotypes = []  # Empty (PhenotypeKB removed)
        hypotheses = {}  # No hypotheses in preliminary report (they're shown separately)
        deg_list = outputs.get("_deg_list", [])
        
        # Collect all plot paths
        plot_paths = {
            "volcano": outputs.get("volcano"),
            "pathway_enrichment": outputs.get("pathway_enrichment"),
            "rna_gsea": outputs.get("rna_gsea") or outputs.get("gsea"),
            "protein_psea": outputs.get("protein_psea") or outputs.get("psea"),
            "phenotype_scores": outputs.get("phenotype_scores"),
            "phenotype_enrichment": outputs.get("phenotype_enrichment")
        }
        
        # Build individual report section (without hypotheses section)
        context_dict = {
            "perturbation": pert_name,
            "perturbation_type": pert_type,
            "deg_list": deg_list
        }
        individual_report = build_report(context_dict, pathways, phenotypes, hypotheses, plot_paths)
        
        # Add individual report content (skip header and hypotheses section)
        individual_lines = individual_report.split("\n")
        skip_header = True
        in_hypotheses_section = False
        header_lines_skipped = 0
        
        for line in individual_lines:
            # Skip header
            if skip_header:
                if line.startswith("# Virtual Cell Analysis Report"):
                    skip_header = True
                    header_lines_skipped += 1
                    continue
                elif line.strip() and not line.startswith("**") and header_lines_skipped > 0:
                    if header_lines_skipped >= 3:
                        skip_header = False
                elif not line.strip() and header_lines_skipped > 0:
                    header_lines_skipped += 1
                    if header_lines_skipped >= 5:
                        skip_header = False
                    continue
                if skip_header:
                    continue
            
            # Skip hypotheses section (we'll add LLM hypotheses separately)
            if line.startswith("## Mechanistic Hypotheses"):
                in_hypotheses_section = True
                continue
            if in_hypotheses_section:
                if line.startswith("## ") and not line.startswith("## Mechanistic Hypotheses"):
                    in_hypotheses_section = False
                else:
                    continue
            
            # Add all other content
            lines.append(line)
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Comparison section
    if results.get("comparison") and "error" not in results.get("comparison", {}):
        comp = results["comparison"]
        lines.append("## Comparison Analysis")
        lines.append("")
        
        pert1_name = comp.get("perturbation1", "Perturbation 1")
        pert2_name = comp.get("perturbation2", "Perturbation 2")
        lines.append(f"**Comparison:** {pert1_name} vs {pert2_name}")
        lines.append("")
        
        shared_pathways = comp.get("shared_pathways", [])
        if shared_pathways:
            lines.append(f"### Shared Pathways ({len(shared_pathways)})")
            lines.append("")
            for pathway in shared_pathways[:20]:
                name = pathway.get("pathway_name", "Unknown")
                p1_nes = pathway.get("pert1_nes", 0.0)
                p2_nes = pathway.get("pert2_nes", 0.0)
                stronger = pathway.get("stronger", "Equal")
                lines.append(f"- **{name}**: {pert1_name} NES={p1_nes:.2f}, {pert2_name} NES={p2_nes:.2f} (stronger: {stronger})")
            lines.append("")
        
        summary = comp.get("summary", "")
        if summary:
            lines.append("### Summary")
            lines.append("")
            lines.append(summary)
            lines.append("")
    
    # LLM-Generated Hypotheses section (without literature search)
    if llm_hypotheses:
        lines.append("## LLM-Generated Mechanistic Hypotheses")
        lines.append("")
        lines.append("> **Status:** Hypotheses generated using LLM analysis of all combined results.")
        lines.append("> **Literature Search:** Running in background... Results will be added to final report.")
        lines.append("")
        
        for i, hyp in enumerate(llm_hypotheses, 1):
            statement = hyp.get("statement", "")
            mechanism = hyp.get("mechanism", [])
            key_pathways = hyp.get("key_pathways", [])
            key_genes = hyp.get("key_genes", [])
            
            lines.append(f"### Hypothesis {i}: {statement}")
            lines.append("")
            
            # Mechanism
            if mechanism:
                lines.append("**Mechanism:**")
                for step in mechanism:
                    lines.append(f"- {step}")
                lines.append("")
            
            # Key pathways
            if key_pathways:
                lines.append("**Key Pathways:**")
                for pathway in key_pathways:
                    lines.append(f"- {pathway}")
                lines.append("")
            
            # Key genes
            if key_genes:
                lines.append("**Key Genes:**")
                for gene in key_genes:
                    lines.append(f"- {gene}")
                lines.append("")
            
            # Literature status
            lines.append("**Literature Support:** *Pending... Literature search in progress.*")
            lines.append("")
            lines.append("---")
            lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    lines.append("1. Literature search is running for all generated hypotheses")
    lines.append("2. Final comprehensive report will include:")
    lines.append("   - Full paper citations with links")
    lines.append("   - Literature support classification")
    lines.append("   - Supporting evidence for each hypothesis")
    lines.append("")
    lines.append("*Preliminary report generated by Virtual Cell system*")
    lines.append("")
    
    return "\n".join(lines)


def _generate_comprehensive_report(results: Dict[str, Any], query: str, context: Dict[str, Any]) -> Optional[str]:
    """Generate one comprehensive report at the end with all perturbations and comparisons."""
    from .report import build_report
    
    lines = []
    lines.append("# Comprehensive Perturbation Analysis Report")
    lines.append("")
    lines.append(f"**Query:** {query}")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary
    successful = [p for p in results.get("perturbations", []) if "error" not in p and "results" in p]
    lines.append(f"**Total Perturbations:** {len(results.get('perturbations', []))}")
    lines.append(f"**Successful:** {len(successful)}")
    if results.get("comparison") and "error" not in results.get("comparison", {}):
        comp = results["comparison"]
        lines.append(f"**Comparison:** ✓ Generated ({len(comp.get('shared_pathways', []))} shared pathways, {len(comp.get('shared_phenotypes', []))} shared phenotypes)")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Individual perturbation reports
    for i, pert in enumerate(successful, 1):
        pert_type = pert.get("type", "unknown")
        pert_name = pert.get("match_info", {}).get("used_name", pert.get("name", "Unknown"))
        outputs = pert.get("outputs", {})
        
        lines.append(f"## Perturbation {i}: {pert_name} ({pert_type})")
        lines.append("")
        
        # Get stored report data
        pathways = outputs.get("_pathways", [])
        phenotypes = []  # Empty (PhenotypeKB removed)
        # Hypotheses are at the end for comparison queries, not per-perturbation
        hypotheses = {} if len(successful) >= 2 else outputs.get("_hypotheses", {})
        deg_list = outputs.get("_deg_list", [])
        
        # Collect all plot paths
        plot_paths = {
            "volcano": outputs.get("volcano"),
            "pathway_enrichment": outputs.get("pathway_enrichment"),
            "rna_gsea": outputs.get("rna_gsea") or outputs.get("gsea"),
            "protein_psea": outputs.get("protein_psea") or outputs.get("psea"),
            "phenotype_scores": outputs.get("phenotype_scores"),
            "phenotype_enrichment": outputs.get("phenotype_enrichment")
        }
        
        # Build individual report section
        context_dict = {
            "perturbation": pert_name,
            "perturbation_type": pert_type,
            "deg_list": deg_list
        }
        individual_report = build_report(context_dict, pathways, phenotypes, hypotheses, plot_paths)
        
        # Add individual report content (skip header as we already have section header)
        individual_lines = individual_report.split("\n")
        # Skip the first few lines (header) and add the rest
        skip_lines = 0
        for line in individual_lines:
            if line.startswith("# Virtual Cell Analysis Report"):
                skip_lines += 1
                continue
            elif skip_lines > 0 and not line.strip():
                skip_lines += 1
                if skip_lines >= 5:  # Skip header section
                    skip_lines = 0
                continue
            if skip_lines == 0:
                lines.append(line)
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Comparison section
    if results.get("comparison") and "error" not in results.get("comparison", {}):
        comp = results["comparison"]
        lines.append("## Comparison Analysis")
        lines.append("")
        
        pert1_name = comp.get("perturbation1", "Perturbation 1")
        pert2_name = comp.get("perturbation2", "Perturbation 2")
        lines.append(f"**Comparison:** {pert1_name} vs {pert2_name}")
        lines.append("")
        
        shared_pathways = comp.get("shared_pathways", [])
        shared_phenotypes = comp.get("shared_phenotypes", [])
        
        if shared_pathways:
            lines.append(f"### Shared Pathways ({len(shared_pathways)})")
            lines.append("")
            for pathway in shared_pathways[:20]:  # Top 20
                name = pathway.get("pathway_name", "Unknown")
                p1_nes = pathway.get("pert1_nes", 0.0)
                p2_nes = pathway.get("pert2_nes", 0.0)
                stronger = pathway.get("stronger", "Equal")
                lines.append(f"- **{name}**: {pert1_name} NES={p1_nes:.2f}, {pert2_name} NES={p2_nes:.2f} (stronger: {stronger})")
            lines.append("")
        
        if shared_phenotypes:
            lines.append(f"### Shared Phenotypes ({len(shared_phenotypes)})")
            lines.append("")
            for pheno in shared_phenotypes[:20]:  # Top 20
                name = pheno.get("phenotype_name", "Unknown")
                p1_score = pheno.get("pert1_score", 0.0)
                p2_score = pheno.get("pert2_score", 0.0)
                lines.append(f"- **{name}**: {pert1_name} score={p1_score:.2f}, {pert2_name} score={p2_score:.2f}")
            lines.append("")
        else:
            lines.append("No shared phenotypes found.")
            lines.append("")
        
        summary = comp.get("summary", "")
        if summary:
            lines.append("### Summary")
            lines.append("")
            lines.append(summary)
            lines.append("")
    
    # Hypotheses section (for comparison queries)
    if len(successful) >= 2 and results.get("hypotheses"):
        hypotheses = results["hypotheses"]
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
                
                # Literature support
                lit_support = hyp.get("literature_support", {})
                overall = lit_support.get("overall", "unknown")
                summary = lit_support.get("summary", "")
                supporting_papers_full = lit_support.get("supporting_papers_full", [])
                supporting_pmids = lit_support.get("supporting_papers", [])
                
                lines.append(f"**Literature Support:** {overall.upper()}")
                lines.append(f"{summary}")
                
                if supporting_papers_full:
                    lines.append("**Supporting Papers:**")
                    from .futurehouse_client import format_citation
                    for paper in supporting_papers_full[:5]:
                        citation = format_citation(paper)
                        lines.append(f"- {citation}")
                elif supporting_pmids:
                    lines.append("**Supporting Papers:**")
                    for pmid in supporting_pmids[:5]:
                        lines.append(f"- {pmid}")
                lines.append("")
                
                lines.append("---")
                lines.append("")
        else:
            # No hypotheses explanation
            no_hyp_explanation = hypotheses.get('no_hypotheses_explanation')
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
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Comprehensive report generated by Virtual Cell system*")
    lines.append("")
    
    return "\n".join(lines)
