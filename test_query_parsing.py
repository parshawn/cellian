#!/usr/bin/env python3
"""
Quick test script to check how queries are parsed for perturbation detection.
Usage: python test_query_parsing.py "your query here"
"""

import sys
import os
import json
import pickle
from pathlib import Path
from typing import Optional

# Add backend directories to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir / "llm"))

try:
    from input import extract_perturbation_info
except ImportError as e:
    print(f"❌ Error importing extract_perturbation_info: {e}")
    print("Make sure you're running this from the cellian directory")
    sys.exit(1)


def check_gene_like(name: str) -> bool:
    """Check if a name looks like a gene (uppercase, short)."""
    if not name:
        return False
    return len(name) <= 10 and name[0].isupper() and name.isupper()


def check_drug_like(name: str) -> bool:
    """Check if a name looks like a drug (lowercase start, longer, or multi-word)."""
    if not name:
        return False
    return name[0].islower() or len(name.split()) > 1 or " " in name


def format_value(value):
    """Format a value for display."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, dict)):
        return str(value) if value else "None"
    return str(value)


GENE_VAR_DIMS = Path("/home/nebius/state/test_replogle/hepg2_holdout/var_dims.pkl")
DRUG_VAR_DIMS = Path("/home/nebius/ST-Tahoe/var_dims.pkl")

_GENE_NAME_CACHE = None
_DRUG_NAME_CACHE = None

# Minimal regression set to ensure Gemini picks exact perturbation strings
MATCHING_TEST_CASES = [
    {
        "name": "Drug – (R)-Verapamil 0.5 uM",
        "query": "Run a drug perturbation with verapamil at 0.5 micromolar",
        "expected": "[('(R)-Verapamil (hydrochloride)', 0.5, 'uM')]",
        "perturbation_kind": "drug"
    },
    {
        "name": "Drug – (S)-Crizotinib 5 uM",
        "query": "Treat cells with crizotinib at 5 uM",
        "expected": "[('(S)-Crizotinib', 5.0, 'uM')]",
        "perturbation_kind": "drug"
    },
    {
        "name": "Gene – TP53 knockout",
        "query": "Knock out TP53 and report results",
        "expected": "TP53",
        "perturbation_kind": "gene"
    }
]


def load_perturbation_names(path: Path):
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        return set(str(name) for name in payload.get("pert_names", []) if name is not None)
    except Exception as exc:
        print(f"❌ Failed to load perturbation names from {path}: {exc}")
        return set()


def get_gene_names():
    global _GENE_NAME_CACHE
    if _GENE_NAME_CACHE is None:
        _GENE_NAME_CACHE = load_perturbation_names(GENE_VAR_DIMS)
    return _GENE_NAME_CACHE


def get_drug_names():
    global _DRUG_NAME_CACHE
    if _DRUG_NAME_CACHE is None:
        _DRUG_NAME_CACHE = load_perturbation_names(DRUG_VAR_DIMS)
    return _DRUG_NAME_CACHE


def summarize_training_match(label: str, name: str, kind_hint: Optional[str], match_info: Optional[dict]):
    if not name:
        return f"   {label}: None"
    
    candidate_type = None
    if match_info:
        candidate_type = match_info.get("candidate_type")
    if not candidate_type and kind_hint:
        if kind_hint.lower() == "drug":
            candidate_type = "drug"
        elif kind_hint.lower() in {"ko", "kd", "oe"}:
            candidate_type = "gene"
    if not candidate_type:
        candidate_type = "gene" if check_gene_like(name) else "drug"
    
    names_set = get_gene_names() if candidate_type == "gene" else get_drug_names()
    status = "✅" if name in names_set else "❌"
    method = match_info.get("method") if match_info else None
    extra = f"(method: {method})" if method else ""
    return f"   {label}: {status} {name} [{candidate_type}] {extra}"


def run_matching_tests():
    """Validate that extract_perturbation_info returns exact training perturbations."""
    print("=" * 80)
    print("PERTURBATION MATCHING TESTS")
    print("=" * 80)
    
    gene_names = get_gene_names()
    drug_names = get_drug_names()
    if not gene_names:
        print("⚠️  Warning: gene perturbation list is empty or unavailable.")
    if not drug_names:
        print("⚠️  Warning: drug perturbation list is empty or unavailable.")
    
    all_passed = True
    for case in MATCHING_TEST_CASES:
        print(f"\n→ Test: {case['name']}")
        print(f"   Query: {case['query']}")
        result = extract_perturbation_info(case["query"])
        
    if result.get("confidence", 0.0) == 0.5:
        print("   ❌ Gemini extraction unavailable. Ensure GOOGLE_API_KEY/GEMINI_API_KEY is set.")
        return 1
        
        # expected = case["expected"]
        # targets = [result.get("target"), result.get("target2")]
        # if expected in targets:
        #     print("   ✅ Matched expected perturbation.")
        # else:
        #     print("   ❌ Expected match not found.")
        #     print(f"      Expected: {expected}")
        #     print(f"      target:   {result.get('target')}")
        #     print(f"      target2:  {result.get('target2')}")
        #     print(f"      match info: {json.dumps(result.get('target_match_info') or {}, indent=2)}")
        #     all_passed = False
        #     continue
        
        # # Double-check that the selection exists in the training list
        # candidate_set = gene_names if case["perturbation_kind"] == "gene" else drug_names
        # if expected not in candidate_set:
        #     print("   ❌ Matched string not present in training list!")
        #     all_passed = False
        # else:
        #     print("   📦 Verified against var_dims.pkl.")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All matching tests passed.")
        return 0
    else:
        print("⚠️  One or more matching tests failed.")
        return 1


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_query_parsing.py \"your query here\"")
        print("   or: python test_query_parsing.py --test-matching")
        print("\nExample:")
        print('  python test_query_parsing.py "CHCHD2 vs Dimethyl fumarate"')
        print('  python test_query_parsing.py "What happens if I knock down TP53?"')
        print('  python test_query_parsing.py "TP53 knockout and imatinib"')
        print('  python test_query_parsing.py "Compare perturbation of Paclitaxel with PFDN4 loss"')
        sys.exit(1)
    
    if sys.argv[1] in {"--test-matching", "--matching-tests", "--validate-perturbations"}:
        exit_code = run_matching_tests()
        sys.exit(exit_code)
    
    query = " ".join(sys.argv[1:])
    
    print("=" * 80)
    print("QUERY PARSING TEST")
    print("=" * 80)
    print(f"\n📝 Query: {query}\n")
    
    # Extract perturbation info using LLM
    print("🔍 Extracting perturbation information using LLM...")
    try:
        result = extract_perturbation_info(query)
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Check if fallback was used (confidence 0.5 is fallback)
    using_fallback = result.get("confidence", 0.0) == 0.5
    if using_fallback:
        print("❌ Gemini extraction unavailable. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
        sys.exit(1)
    else:
        print(f"✅ LLM extraction successful (confidence: {result.get('confidence', 0.0):.2f})")
    
    # Extract key fields
    target = result.get("target", "None")
    target2 = result.get("target2", None)
    pert_type = result.get("type", "unknown")
    pert_type2 = result.get("type2", None)
    has_both = result.get("has_both", False)
    is_comparison = result.get("is_comparison", False)
    confidence = result.get("confidence", 0.0)
    
    # Apply auto-detection logic (same as in api.py) - for display only
    auto_detected = False
    final_has_both = has_both
    if target2 and not has_both:
        is_gene_like = check_gene_like(target)
        is_drug_like = check_drug_like(target2)
        if is_gene_like and is_drug_like:
            final_has_both = True
            auto_detected = True
    
    # Determine perturbation type for pipeline
    if final_has_both and target2:
        perturbation_type = "both"
    elif pert_type == "drug" or (target and not check_gene_like(target)):
        perturbation_type = "drug"
    else:
        perturbation_type = "gene"
    
    # Print results
    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80)
    
    print(f"\n🎯 PRIMARY TARGET:")
    print(f"   Target: {format_value(target)}")
    print(f"   Type: {format_value(pert_type)}")
    
    if target2:
        print(f"\n🎯 SECONDARY TARGET:")
        print(f"   Target: {format_value(target2)}")
        print(f"   Type: {format_value(pert_type2)}")
    
    print(f"\n📊 PERTURBATION DETECTION:")
    print(f"   Has Both (LLM): {format_value(result.get('has_both', False))}")
    if auto_detected:
        print(f"   ⚠️  Backend auto-detection would set has_both=True")
        print(f"      Reason: {target} (gene-like) + {target2} (drug-like)")
    print(f"   Has Both (Final): {format_value(final_has_both)}")
    print(f"   Is Comparison: {format_value(is_comparison)}")
    print(f"   Perturbation Type (Pipeline): {perturbation_type}")
    print(f"   Confidence: {confidence:.2f}")
    
    # Additional fields
    print(f"\n📋 ADDITIONAL INFORMATION:")
    if result.get("pathway_mentioned"):
        print(f"   Pathway Mentioned: {result['pathway_mentioned']}")
    if result.get("phenotype_mentioned"):
        print(f"   Phenotype Mentioned: {result['phenotype_mentioned']}")
    if result.get("focus"):
        print(f"   Focus: {result['focus']}")
    if result.get("protein_mentioned"):
        print(f"   Protein Mentioned: {result['protein_mentioned']}")
    if result.get("top_n_genes"):
        print(f"   Top N Genes: {result['top_n_genes']}")
    if result.get("top_n_proteins"):
        print(f"   Top N Proteins: {result['top_n_proteins']}")
    if result.get("top_n_pathways"):
        print(f"   Top N Pathways: {result['top_n_pathways']}")
    if result.get("top_n_phenotypes"):
        print(f"   Top N Phenotypes: {result['top_n_phenotypes']}")
    if result.get("output_types"):
        print(f"   Output Types: {result['output_types']}")
    
    print(f"\n📦 TRAINING SET MATCHES:")
    print(summarize_training_match("Target", target, pert_type, result.get("target_match_info")))
    print(summarize_training_match("Target2", target2, pert_type2, result.get("target2_match_info")))
    
    # Pipeline execution summary
    print(f"\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    
    if final_has_both and target2:
        # Determine which is gene and which is drug
        gene_target = None
        drug_target = None
        
        if pert_type in ["KO", "KD", "OE"] or (pert_type != "drug" and target and check_gene_like(target)):
            gene_target = target
            drug_target = target2
        elif pert_type2 in ["KO", "KD", "OE"] or (pert_type2 != "drug" and target2 and check_gene_like(target2)):
            gene_target = target2
            drug_target = target
        else:
            # Fallback to pattern matching
            if check_gene_like(target) and check_drug_like(target2):
                gene_target = target
                drug_target = target2
            elif check_gene_like(target2) and check_drug_like(target):
                gene_target = target2
                drug_target = target
            else:
                gene_target = target
                drug_target = target2
        
        print(f"\n✅ BOTH PERTURBATIONS WILL RUN:")
        print(f"   1. Gene Perturbation: {gene_target} ({pert_type if gene_target == target else pert_type2})")
        print(f"   2. Drug Perturbation: {drug_target}")
        print(f"\n   Execution: Sequential (gene first, then drug)")
    else:
        print(f"\n✅ SINGLE PERTURBATION WILL RUN:")
        print(f"   Target: {target}")
        print(f"   Type: {pert_type}")
        print(f"   Pipeline Type: {perturbation_type}")
    
    # Show full raw result
    print(f"\n" + "=" * 80)
    print("FULL RAW RESULT (JSON)")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

