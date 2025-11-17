"""Pathway-based discovery and automation for perturbation orchestrator."""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re


def parse_pathway_query(query: str) -> Dict[str, Any]:
    """
    Intelligently parse complex pathway-based queries using LLM extraction.
    
    Extracts:
    - Pathway name(s)
    - Phenotype filters
    - Number of items (top N)
    - Item types (genes, drugs, or both)
    - Comparison requirements
    - Perturbation types (KO, KD, OE for genes)
    
    Examples:
        "compare the effect of top 5 relevant drugs and gene knockdowns 
         affecting the mTOR pathway that affects cell proliferation"
        -> pathway: "mTOR", phenotype: "cell proliferation", num_items: 5, 
           item_types: ["gene", "drug"], comparison: "multi", gene_type: "KD"
        
        "find top 3 genes in PI3K pathway affecting apoptosis"
        -> pathway: "PI3K", phenotype: "apoptosis", num_items: 3, 
           item_types: ["gene"], comparison: "batch"
    
    Returns:
        Dict with extracted information
    """
    import os
    import json
    from typing import Optional
    
    # Try LLM extraction first (if available)
    try:
        from .input import process_user_query as extract_with_llm
        
        # Check if LLM is available
        try:
            import google.generativeai as genai
            # Try both GOOGLE_API_KEY and GEMINI_API_KEY (for compatibility)
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key and api_key.strip().startswith("AIza"):
                genai.configure(api_key=api_key.strip())
                model = genai.GenerativeModel("gemini-2.0-flash-exp")
                
                extraction_prompt = """Extract structured information from this pathway-based perturbation query.

Query: {query}

Return ONLY valid JSON with these fields:
- "pathway_name": pathway name as string (e.g., "mTOR", "PI3K", "PI3K pathway", "MAPK/ERK", "p53 signaling") OR null if NO pathway is explicitly mentioned. IMPORTANT: Do NOT extract gene names as pathways. If the query is a direct comparison like "TP53 vs Drug" or "MTOR knockout vs Rapamycin", set "pathway_name" to null because TP53/MTOR are GENE NAMES, not pathway names.
- "phenotype_filter": phenotype mentioned as string or null (e.g., "cell proliferation", "apoptosis", "cell cycle arrest", "differentiation"). Extract any phenotype, biological process, or cell state mentioned.
- "num_items": integer or null (e.g., 5, 3, 10). Extract "top N", "N most", "N relevant" numbers.
- "item_types": list of strings from ["gene", "drug"] (e.g., ["gene", "drug"], ["gene"], ["drug"]). Detect what types of perturbations are requested.
- "comparison_type": string - "multi" if explicitly comparing items, "batch" if processing multiple items, "single" if single item.
- "gene_perturbation_type": string from ["KO", "KD", "OE", null] - detect knockout, knockdown, or overexpression if specified.
- "perturbation_verbs": list of strings - extract verbs like ["knockout", "knockdown", "overexpress", "treat", "inhibit", "activate"]

Examples:
Query: "compare the effect of top 5 relevant drugs and gene knockdowns affecting the mTOR pathway that affects cell proliferation"
Result: {{"pathway_name": "mTOR", "phenotype_filter": "cell proliferation", "num_items": 5, "item_types": ["gene", "drug"], "comparison_type": "multi", "gene_perturbation_type": "KD", "perturbation_verbs": ["compare", "knockdowns", "affecting"]}}

Query: "find top 3 genes in PI3K pathway affecting apoptosis"
Result: {{"pathway_name": "PI3K", "phenotype_filter": "apoptosis", "num_items": 3, "item_types": ["gene"], "comparison_type": "batch", "gene_perturbation_type": null, "perturbation_verbs": ["find", "affecting"]}}

Query: "TP53 vs Verapamil" or "compare TP53 vs Drug"
Result: {{"pathway_name": null, "phenotype_filter": null, "num_items": null, "item_types": ["gene", "drug"], "comparison_type": "multi", "gene_perturbation_type": null, "perturbation_verbs": ["vs", "compare"]}}
Note: TP53 is a GENE NAME, not a pathway name, so pathway_name should be null.

Query: "MTOR knockout vs Rapamycin"
Result: {{"pathway_name": null, "phenotype_filter": null, "num_items": null, "item_types": ["gene", "drug"], "comparison_type": "multi", "gene_perturbation_type": "KO", "perturbation_verbs": ["knockout", "vs"]}}
Note: MTOR here is a GENE NAME being knocked out, not a pathway name, so pathway_name should be null.

IMPORTANT: 
- Only extract "pathway_name" if the query explicitly mentions a pathway (e.g., "mTOR pathway", "PI3K signaling", "p53 pathway")
- If a gene name like TP53, MTOR, PI3K, MAPK appears in a direct comparison format ("X vs Y"), treat it as a GENE NAME, not a pathway name
- Return ONLY valid JSON, no other text."""

                full_prompt = extraction_prompt.format(query=query)
                
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                
                content = response.text.strip()
                # Remove markdown code blocks if present
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                extracted = json.loads(content)
                
                # Validate and normalize
                result = {
                    "pathway_name": extracted.get("pathway_name"),
                    "phenotype_filter": extracted.get("phenotype_filter"),
                    "num_items": extracted.get("num_items") or (5 if extracted.get("num_items") is None else None),
                    "item_types": extracted.get("item_types", ["gene", "drug"]),
                    "comparison_type": extracted.get("comparison_type", "batch"),
                    "gene_perturbation_type": extracted.get("gene_perturbation_type"),
                    "perturbation_verbs": extracted.get("perturbation_verbs", [])
                }
                
                # Clean pathway name
                if result["pathway_name"]:
                    result["pathway_name"] = result["pathway_name"].replace(" pathway", "").strip()
                
                return result
                
        except Exception as e:
            # LLM failed, fall back to regex
            pass
    except Exception:
        pass
    
    # Regex-based fallback extraction
    query_lower = query.lower()
    pathway_name = None
    phenotype_filter = None
    num_items = None
    item_types = ["gene", "drug"]  # Default to both
    comparison_type = "batch"
    gene_perturbation_type = None
    perturbation_verbs = []
    
    # First check if this is a direct comparison query (X vs Y format)
    # If so, don't extract pathway names from gene names like TP53, MTOR, etc.
    direct_comparison_pattern = r"(\b(?:[A-Z][A-Z0-9]{1,9}|[a-z][a-z0-9_\-]{3,})\b)\s+(?:vs|versus|or)\s+(\b(?:[A-Z][A-Z0-9]{1,9}|[a-z][a-z0-9_\-]{3,})\b)"
    is_direct_comparison = bool(re.search(direct_comparison_pattern, query, re.IGNORECASE))
    
    # Extract pathway name (comprehensive patterns)
    # IMPORTANT: Only match if pathway is explicitly mentioned (not just a gene name)
    pathway_patterns = [
        # Specific pathways with explicit "pathway" or "signaling" keywords
        r"(?:mTOR|MTOR)\s+(?:pathway|signaling)",
        r"(?:PI3K|PI3K-AKT)\s+(?:pathway|signaling)",
        r"(?:MAPK|MAPK/ERK|ERK)\s+(?:pathway|signaling)",
        r"(?:p53|TP53)\s+(?:pathway|signaling)",
        r"(?:Wnt|Wnt.*pathway|Wnt.*signaling)",
        r"(?:Notch|Notch.*pathway)",
        r"(?:Hedgehog|Hedgehog.*pathway)",
        r"(?:JAK-STAT|JAK.*STAT)\s+(?:pathway|signaling)",
        r"(?:NF-?κB|NFkB|NF.*kappa.*B)\s+(?:pathway|signaling)",
        r"(?:TGF-?β|TGF.*beta)\s+(?:pathway|signaling)",
        r"(?:cell cycle|cell cycle.*pathway)",
        r"(?:apoptosis|apoptotic.*pathway)",
        # Generic pathway mention (must explicitly say "pathway")
        r"(\w+(?:\s+\w+)?)\s+pathway",
        r"pathway.*?(\w+(?:\s+\w+)?)",
    ]
    
    # Only extract pathway name if it's NOT a direct comparison query with gene names
    # OR if pathway is explicitly mentioned in the query
    has_explicit_pathway_keyword = any(kw in query_lower for kw in ["pathway", "pathways", "signaling", "signalling", "top", "relevant", "affecting"])
    
    if not is_direct_comparison or has_explicit_pathway_keyword:
        for pattern in pathway_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                pathway_name = match.group(1) if match.groups() else match.group(0)
                pathway_name = pathway_name.replace(" pathway", "").replace(" signaling", "").replace(" signalling", "").strip()
                # Additional check: if pathway_name looks like a single gene name in a direct comparison, don't use it
                if is_direct_comparison and not has_explicit_pathway_keyword:
                    # Gene names are typically uppercase 2-9 chars, or specific patterns
                    gene_like_pattern = r"^[A-Z][A-Z0-9]{1,9}$"
                    if re.match(gene_like_pattern, pathway_name):
                        pathway_name = None  # Treat as gene name, not pathway
                        continue
                if pathway_name:
                    break
    
    # Extract number (top N, N most, N relevant)
    num_patterns = [
        r"top\s+(\d+)",
        r"(\d+)\s+(?:most|relevant|top)",
        r"(\d+)\s+(?:drugs|genes|items|perturbations)",
        r"(\d+)\s+of",
    ]
    for pattern in num_patterns:
        match = re.search(pattern, query_lower)
        if match:
            num_items = int(match.group(1))
            break
    
    # Extract item types
    has_gene = any(word in query_lower for word in ["gene", "knockout", "knockdown", "overexpress", "knock", "overexpression"])
    has_drug = any(word in query_lower for word in ["drug", "compound", "treatment", "inhibit", "inhibitor", "therapeutic"])
    
    if has_gene and not has_drug:
        item_types = ["gene"]
    elif has_drug and not has_gene:
        item_types = ["drug"]
    elif has_gene and has_drug:
        item_types = ["gene", "drug"]
    
    # Extract gene perturbation type
    if "knockout" in query_lower or "ko" in query_lower or "knock out" in query_lower:
        gene_perturbation_type = "KO"
        perturbation_verbs.append("knockout")
    elif "knockdown" in query_lower or "kd" in query_lower or "knock down" in query_lower:
        gene_perturbation_type = "KD"
        perturbation_verbs.append("knockdown")
    elif "overexpress" in query_lower or "oe" in query_lower or "over expression" in query_lower:
        gene_perturbation_type = "OE"
        perturbation_verbs.append("overexpress")
    
    # Extract phenotype filter
    phenotype_patterns = [
        r"(?:affecting|that affects?|affects?)\s+([^.]*?)(?:\.|$|that|which)",
        r"(?:relating to|related to|involving)\s+([^.]*?)(?:\.|$|that|which)",
        r"cell\s+proliferation",
        r"apoptosis|apoptotic",
        r"cell\s+cycle",
        r"differentiation",
        r"senescence",
        r"migration",
        r"invasion",
        r"metastasis",
    ]
    
    for pattern in phenotype_patterns:
        match = re.search(pattern, query_lower)
        if match:
            phenotype_filter = match.group(1).strip() if match.groups() else match.group(0).strip()
            # Clean up common words
            phenotype_filter = re.sub(r"^(affecting|that affects?|affects?|relating to|related to|involving)\s+", "", phenotype_filter, flags=re.IGNORECASE).strip()
            if phenotype_filter:
                break
    
    # Determine comparison type
    if any(word in query_lower for word in ["compare", "comparison", "versus", "vs", "versus"]):
        comparison_type = "multi"
        perturbation_verbs.append("compare")
    elif num_items and num_items > 1:
        comparison_type = "batch"
    
    return {
        "pathway_name": pathway_name,
        "phenotype_filter": phenotype_filter,
        "num_items": num_items or 5,  # Default to 5
        "item_types": item_types,
        "comparison_type": comparison_type,
        "gene_perturbation_type": gene_perturbation_type,
        "perturbation_verbs": perturbation_verbs
    }


def find_genes_in_pathway(
    pathway_name: str,
    available_genes: List[str],
    pathway_enrichment_results: Optional[List[Dict[str, Any]]] = None
) -> List[str]:
    """
    Find genes in a pathway from available gene list using LLM-based discovery.
    
    This function uses LLM knowledge to discover genes related to a pathway,
    then validates them against the available genes list (.pkl file).
    
    Args:
        pathway_name: Pathway name (e.g., "mTOR", "PI3K pathway")
        available_genes: List of available gene names from STATE model (.pkl file)
        pathway_enrichment_results: Optional pathway enrichment results from previous analyses
    
    Returns:
        List of gene names in the pathway that are available for perturbation (from .pkl)
    """
    pathway_keywords = [
        pathway_name.lower().replace(" pathway", "").replace("pathway", "").strip(),
        "mTOR", "MTOR", "PI3K", "PI3K-AKT", "MAPK", "MAPK/ERK",
        "p53", "TP53", "cell cycle", "apoptosis"
    ]
    
    # Try to find pathway in enrichment results first
    if pathway_enrichment_results:
        matching_genes = []
        for pathway in pathway_enrichment_results:
            pathway_name_lower = pathway.get("name", "").lower()
            if any(keyword in pathway_name_lower for keyword in pathway_keywords):
                member_genes = pathway.get("member_genes", [])
                # Filter to available genes
                available = [g for g in member_genes if g.upper() in [ag.upper() for ag in available_genes]]
                matching_genes.extend(available)
        if matching_genes:
            return list(set(matching_genes))  # Remove duplicates
    
    # Fallback: Use PathwayLoader if available
    try:
        import sys
        from pathlib import Path
        reasoning_path = Path(__file__).parent.parent / "reasoning_layer" / "engine"
        if reasoning_path.exists():
            sys.path.insert(0, str(reasoning_path.parent))
            from reasoning_layer.engine.pathway_loader import PathwayLoader
            
            loader = PathwayLoader()
            # Search for pathway by name
            pathway_ids = [pid for pid, pdata in loader.pathways.items() 
                          if any(kw in pdata.get("name", "").lower() for kw in pathway_keywords)]
            
            if pathway_ids:
                all_genes = []
                for pid in pathway_ids:
                    genes = loader.get_genes_in_pathway(pid)
                    # Filter to available genes
                    available = [g for g in genes if g.upper() in [ag.upper() for ag in available_genes]]
                    all_genes.extend(available)
                if all_genes:
                    return list(set(all_genes))
    except Exception:
        pass
    
    # Use LLM to discover pathway genes based on knowledge
    # This is important: LLM knows pathway-gene associations, then we validate against .pkl
    try:
        import os
        import json
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key and api_key.strip().startswith("AIza"):
                genai.configure(api_key=api_key.strip())
                model = genai.GenerativeModel("gemini-2.0-flash-exp")
                
                discovery_prompt = f"""Based on your knowledge of pathway biology, list genes that are key components of the "{pathway_name}" pathway.

Return ONLY a JSON array of gene symbols (official gene names in uppercase, e.g., ["MTOR", "RPTOR", "AKT1"]).

Focus on well-established, core pathway components. For mTOR pathway, include: MTOR, RPTOR, RICTOR, AKT1, TSC1, TSC2, RHEB, RPS6KB1, EIF4EBP1, etc.
For PI3K pathway, include: PIK3CA, PIK3CB, PIK3R1, AKT1, PTEN, PDPK1, etc.
For MAPK pathway, include: MAPK1, MAPK3, MAP2K1, MAP2K2, RAF1, BRAF, KRAS, HRAS, etc.

Return ONLY the JSON array, no other text."""

                response = model.generate_content(
                    discovery_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                
                content = response.text.strip()
                # Remove markdown code blocks if present
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                llm_genes = json.loads(content)
                if isinstance(llm_genes, list):
                    # Filter to available genes (from .pkl)
                    available = [g.upper() for g in llm_genes if g.upper() in [ag.upper() for ag in available_genes]]
                    if available:
                        print(f"  ✓ LLM discovered {len(available)} genes in {pathway_name} pathway from available list")
                        return available
        except Exception as e:
            # LLM failed, fall back to hardcoded
            pass
    except Exception:
        pass
    
    # Fallback: Use hardcoded pathway-gene associations (then filter to available)
    pathway_gene_keywords = {
        "mtor": ["MTOR", "RPTOR", "RICTOR", "AKT1", "TSC1", "TSC2", "RHEB", "RPS6KB1", "EIF4EBP1", "MLST8", "DEPTOR", "AKT1S1"],
        "pi3k": ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1", "PIK3R2", "AKT1", "AKT2", "AKT3", "PTEN", "PDPK1", "FOXO1"],
        "mapk": ["MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "MAPKAPK2", "RAS", "RAF1", "BRAF", "KRAS", "HRAS", "NRAS", "MEK1", "MEK2"],
        "p53": ["TP53", "CDKN1A", "MDM2", "BAX", "BBC3", "P21", "GADD45A", "FAS", "CASP3"],
        "wnt": ["CTNNB1", "APC", "GSK3B", "AXIN1", "AXIN2", "TCF7", "LEF1", "DVL1", "DVL2"],
        "notch": ["NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "JAG1", "JAG2", "DLL1", "DLL3", "DLL4"],
        "jak-stat": ["JAK1", "JAK2", "JAK3", "TYK2", "STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6"]
    }
    
    pathway_normalized = pathway_name.lower().replace(" pathway", "").replace("pathway", "").strip()
    
    # Check if pathway matches any keyword
    for key, possible_genes in pathway_gene_keywords.items():
        if key in pathway_normalized or pathway_normalized in key:
            # Filter to available genes (from .pkl)
            available = [g for g in possible_genes if g.upper() in [ag.upper() for ag in available_genes]]
            if available:
                return available
    
    # If no match, try to find genes containing pathway name or keywords
    pathway_words = pathway_normalized.split()
    matching_genes = []
    for gene in available_genes:
        gene_upper = gene.upper()
        # Check if pathway keywords appear in gene name
        if any(word.upper() in gene_upper for word in pathway_words if len(word) > 2):
            matching_genes.append(gene)
    
    return matching_genes[:20]  # Limit to first 20 matches


def find_drugs_affecting_pathway(
    pathway_name: str,
    available_drugs: List[str],
    pathway_enrichment_results: Optional[List[Dict[str, Any]]] = None
) -> List[str]:
    """
    Find drugs that affect a pathway using LLM-based discovery.
    
    This function uses LLM knowledge to discover drugs known to target a pathway,
    then validates them against the available drugs list (.pkl file).
    
    Args:
        pathway_name: Pathway name (e.g., "mTOR", "PI3K pathway")
        available_drugs: List of available drug names from ST-Tahoe model (.pkl file)
        pathway_enrichment_results: Optional pathway enrichment results from previous analyses
    
    Returns:
        List of drug names (in tuple format from .pkl) that affect the pathway
    """
    from .perturbation_orchestrator import extract_drug_name_from_tuple
    
    # Use LLM to discover pathway-targeting drugs based on knowledge
    try:
        import os
        import json
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key and api_key.strip().startswith("AIza"):
                genai.configure(api_key=api_key.strip())
                model = genai.GenerativeModel("gemini-2.0-flash-exp")
                
                discovery_prompt = f"""Based on your knowledge of pharmacology, list drug names that target or affect the "{pathway_name}" pathway.

Return ONLY a JSON array of drug names (common names, e.g., ["Rapamycin", "Everolimus", "Temsirolimus"]).

Focus on well-known, FDA-approved or clinically tested drugs. For mTOR pathway, include: Rapamycin, Everolimus, Temsirolimus, Sirolimus, etc.
For PI3K pathway, include: Idelalisib, Copanlisib, Alpelisib, Duvelisib, etc.
For MAPK pathway, include: Trametinib, Cobimetinib, Selumetinib, Binimetinib, etc.
For p53/TP53 pathway, include: Nutlin-3, PRIMA-1, APR-246, RITA, CP-31398, MI-219, RG7112, Serdemetan, etc.

Return ONLY the JSON array, no other text."""

                response = model.generate_content(
                    discovery_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                
                content = response.text.strip()
                # Remove markdown code blocks if present
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                llm_drugs = json.loads(content)
                if isinstance(llm_drugs, list):
                    # Match against available drugs (from .pkl)
                    # Need to extract drug names from tuple format and match
                    available = []
                    for llm_drug in llm_drugs:
                        llm_drug_lower = str(llm_drug).lower()
                        for avail_drug in available_drugs:
                            if isinstance(avail_drug, str):
                                extracted_name = extract_drug_name_from_tuple(avail_drug)
                                if llm_drug_lower in extracted_name.lower() or extracted_name.lower() in llm_drug_lower:
                                    if avail_drug not in available:
                                        available.append(avail_drug)
                    
                    if available:
                        print(f"  ✓ LLM discovered {len(available)} drugs affecting {pathway_name} pathway from available list")
                        return available
                    else:
                        print(f"  ⚠️  LLM suggested {len(llm_drugs)} drugs for {pathway_name} pathway, but none matched available drug list")
                        print(f"     LLM suggestions: {llm_drugs[:10]}")  # Show first 10 for debugging
        except Exception as e:
            # LLM failed, fall back to hardcoded
            print(f"  ⚠️  LLM drug discovery failed for {pathway_name} pathway: {e}")
            pass
    except Exception:
        pass
    
    # Fallback: Use hardcoded drug-pathway associations (then match against available)
    pathway_drug_keywords = {
        "mtor": ["Rapamycin", "Everolimus", "Temsirolimus", "Sirolimus", "Ridaforolimus", "Deforolimus"],
        "pi3k": ["Idelalisib", "Copanlisib", "Alpelisib", "Duvelisib", "Pictilisib", "Buparlisib"],
        "mapk": ["Trametinib", "Cobimetinib", "Selumetinib", "Binimetinib", "Pimasertib", "Refametinib"],
        "jak-stat": ["Ruxolitinib", "Tofacitinib", "Baricitinib", "Fedratinib", "Pacritinib"],
        "wnt": ["LGK974", "PRI-724", "CWP232291", "ICG-001"],
        "notch": ["RO4929097", "MK-0752", "PF-03084014", "Demcizumab"],
        "p53": ["Nutlin", "PRIMA", "APR-246", "RITA", "CP-31398", "MI-219", "RG7112", "Serdemetan", "ReACp53"],
        "tp53": ["Nutlin", "PRIMA", "APR-246", "RITA", "CP-31398", "MI-219", "RG7112", "Serdemetan", "ReACp53"]
    }
    
    pathway_normalized = pathway_name.lower().replace(" pathway", "").replace("pathway", "").replace(" signaling", "").replace("signaling", "").strip()
    
    # Check if pathway matches any keyword (e.g., "p53", "tp53", "p53 pathway" all match "p53")
    for key, possible_drugs in pathway_drug_keywords.items():
        if key in pathway_normalized or pathway_normalized in key or any(keyword in pathway_normalized for keyword in [key, key.replace("-", ""), key.replace("_", "")]):
            # Match against available drugs (extract drug names from tuple format)
            available = []
            for drug in possible_drugs:
                drug_lower = drug.lower()
                for avail_drug in available_drugs:
                    if isinstance(avail_drug, str):
                        extracted_name = extract_drug_name_from_tuple(avail_drug)
                        extracted_lower = extracted_name.lower()
                        # More flexible matching: check if drug name contains keyword or vice versa
                        if (drug_lower in extracted_lower or 
                            extracted_lower in drug_lower or
                            any(word in extracted_lower for word in drug_lower.split("-") if len(word) > 3) or
                            any(word in extracted_lower for word in drug_lower.split(" ") if len(word) > 3)):
                            if avail_drug not in available:
                                available.append(avail_drug)
            if available:
                print(f"  ✓ Found {len(available)} drugs affecting {pathway_name} pathway from hardcoded associations")
                return available
            else:
                print(f"  ⚠️  Hardcoded drugs for {pathway_name} ({key}): {possible_drugs}, but none matched available drug list")
    
    # If no match, try fuzzy matching on available drug names
    pathway_words = pathway_normalized.split()
    matching_drugs = []
    for avail_drug in available_drugs:
        if isinstance(avail_drug, str):
            extracted_name = extract_drug_name_from_tuple(avail_drug).lower()
            # Check if pathway keywords appear in drug name (minimum 3 chars to avoid false positives)
            if any(word.lower() in extracted_name for word in pathway_words if len(word) > 2):
                if avail_drug not in matching_drugs:
                    matching_drugs.append(avail_drug)
    
    if matching_drugs:
        print(f"  ✓ Found {len(matching_drugs)} drugs via fuzzy matching for {pathway_name} pathway")
    else:
        print(f"  ⚠️  No drugs found for {pathway_name} pathway via any method (LLM, hardcoded, or fuzzy matching)")
    
    return matching_drugs[:20]  # Limit to first 20 matches


def rank_perturbations_by_relevance(
    perturbations: List[str],
    pathway_name: str,
    pert_type: str,  # "gene" or "drug"
    phenotype_filter: Optional[str] = None
) -> List[Tuple[str, float]]:
    """
    Rank perturbations by relevance to pathway and phenotype.
    
    Args:
        perturbations: List of perturbation names
        pathway_name: Pathway name
        pert_type: "gene" or "drug"
        phenotype_filter: Optional phenotype to filter by
    
    Returns:
        List of (perturbation_name, score) tuples, sorted by score (highest first)
    """
    # For now, simple scoring based on keyword matching
    # TODO: Use pathway enrichment results or database associations
    
    scores = []
    pathway_keywords = pathway_name.lower().split()
    
    for pert in perturbations:
        score = 0.0
        
        # Check if perturbation name matches pathway keywords
        pert_lower = pert.lower()
        for keyword in pathway_keywords:
            if keyword in pert_lower:
                score += 1.0
        
        # TODO: Add phenotype-based scoring if phenotype_filter is provided
        
        scores.append((pert, score))
    
    # Sort by score (highest first)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def discover_pathway_perturbations(
    pathway_name: str,
    phenotype_filter: Optional[str] = None,
    num_items: int = 5,
    item_types: List[str] = ["gene", "drug"]
) -> Dict[str, List[str]]:
    """
    Discover top N perturbations (genes/drugs) affecting a pathway.
    
    Args:
        pathway_name: Pathway name (e.g., "mTOR", "PI3K pathway")
        phenotype_filter: Optional phenotype filter
        num_items: Number of top items to return
        item_types: List of item types to search ["gene", "drug"], ["gene"], or ["drug"]
    
    Returns:
        Dict with keys "genes" and/or "drugs" containing lists of perturbation names
    """
    from .perturbation_orchestrator import load_valid_perturbation_names
    
    # Load available perturbations
    drug_names, gene_names = load_valid_perturbation_names()
    
    results = {}
    
    # Find genes in pathway
    if "gene" in item_types:
        pathway_genes = find_genes_in_pathway(pathway_name, gene_names)
        # Rank and select top N
        ranked_genes = rank_perturbations_by_relevance(
            pathway_genes, pathway_name, "gene", phenotype_filter
        )
        results["genes"] = [g for g, _ in ranked_genes[:num_items]]
    
    # Find drugs affecting pathway
    if "drug" in item_types:
        pathway_drugs = find_drugs_affecting_pathway(pathway_name, drug_names)
        # Rank and select top N
        ranked_drugs = rank_perturbations_by_relevance(
            pathway_drugs, pathway_name, "drug", phenotype_filter
        )
        results["drugs"] = [d for d, _ in ranked_drugs[:num_items]]
    
    return results

