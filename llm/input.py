"""LLM input processing: handles user questions and extracts perturbation information."""
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import lru_cache
from difflib import SequenceMatcher

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


MODEL_NAME = "gemini-2.0-flash-exp"  # Best Gemini model (falls back to gemini-1.5-pro if unavailable)
GENE_PERTURBATION_LIST_PATH = Path("/home/nebius/state/test_replogle/hepg2_holdout/var_dims.pkl")
DRUG_PERTURBATION_LIST_PATH = Path("/home/nebius/ST-Tahoe/var_dims.pkl")
_GEMINI_CONFIGURED = False
_GEMINI_MODEL_CACHE: Optional[Any] = None


def _configure_gemini():
    global _GEMINI_CONFIGURED
    if genai is None:
        raise RuntimeError("google-generativeai is not installed. Install it and set GOOGLE_API_KEY or GEMINI_API_KEY.")
    
    if _GEMINI_CONFIGURED:
        return
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini API key not set. Export GOOGLE_API_KEY or GEMINI_API_KEY.")
    
    api_key = api_key.strip()
    if len(api_key) < 20 or not api_key.startswith("AIza"):
        raise RuntimeError("Gemini API key appears invalid. Ensure it starts with 'AIza'.")
    
    genai.configure(api_key=api_key)
    _GEMINI_CONFIGURED = True


def _get_gemini_model():
    global _GEMINI_MODEL_CACHE
    _configure_gemini()
    
    if _GEMINI_MODEL_CACHE is not None:
        return _GEMINI_MODEL_CACHE
    
    try:
        _GEMINI_MODEL_CACHE = genai.GenerativeModel(MODEL_NAME)
    except Exception:
        try:
            _GEMINI_MODEL_CACHE = genai.GenerativeModel("gemini-1.5-pro")
        except Exception as exc:
            raise RuntimeError("Unable to initialize Gemini model (gemini-2.0-flash-exp or gemini-1.5-pro).") from exc
    
    return _GEMINI_MODEL_CACHE


def process_user_query(query: str) -> Dict[str, Any]:
    """
    Process user query and extract relevant information.
    
    Args:
        query: User's natural language question (e.g., "What happens if I knock down TP53?")
    
    Returns:
        Dictionary with extracted information:
        - "target": gene name
        - "type": perturbation type ("KO", "KD", "OE", or "unknown")
        - "confidence": confidence level (0.0 to 1.0)
        - "original_query": original query string
    """
    model = _get_gemini_model()

    extraction_prompt = """Extract perturbation information from the user's question.

Return a JSON object with:
- "target": PRIMARY gene name OR drug name (e.g., "TP53", "JAK1", "imatinib", "aspirin")
- "target2": SECOND gene or drug if BOTH are mentioned (e.g., if user says "TP53 and imatinib" or "both TP53 KO and aspirin")
- "type": perturbation type for target ("KO", "KD", "OE", "drug", or "unknown")
- "type2": perturbation type for target2 if both are mentioned ("KO", "KD", "OE", "drug", or null)
- "has_both": boolean - true if user mentions BOTH a gene AND a drug perturbation (not just comparison)
- "confidence": confidence level (0.0 to 1.0)
- "pathway_mentioned": pathway name if mentioned (e.g., "mTOR", "PI3K", null)
- "phenotype_mentioned": phenotype mentioned if any (e.g., "cell proliferation", "apoptosis", null)
- "is_comparison": boolean - true if query is asking to compare two perturbations
- "num_items": integer or null - number of items if "top N" or "N most" is mentioned
- "focus": string - what the query focuses on: "genes", "proteins", "pathways", "phenotypes", "both" (genes+proteins), or null
- "top_n_genes": integer or null - specific number if asking for "top N genes" (e.g., "top 10 genes" -> 10)
- "top_n_proteins": integer or null - specific number if asking for "top N proteins" (e.g., "top 5 proteins" -> 5)
- "top_n_pathways": integer or null - specific number if asking for "top N pathways" (e.g., "top 3 pathways" -> 3)
- "top_n_phenotypes": integer or null - specific number if asking for "top N phenotypes" (e.g., "top 7 phenotypes" -> 7)
- "protein_mentioned": boolean - true if query explicitly mentions "protein", "proteins", "protein changes", etc.
- "output_types": list of strings - what outputs user wants: ["plots", "genes", "proteins", "pathways", "phenotypes", "report", "all"] or null for all

IMPORTANT: 
- ORDER DOES NOT MATTER: The first mentioned item is not necessarily the gene. Identify genes vs drugs based on:
  * Gene indicators: Short uppercase acronyms (1-10 chars), mentions of "knockout", "knockdown", "loss", "KO", "KD", "OE", "overexpression"
  * Drug indicators: Longer names, multi-word names, lowercase or mixed case, common drug names (e.g., Paclitaxel, imatinib, aspirin)
  * Context clues: "X loss" or "X knockout" = gene, "X" alone could be either - use name patterns
- Extract BOTH genes and drugs if the user mentions BOTH a gene perturbation AND a drug (e.g., "TP53 KO and imatinib", "both JAK1 knockdown and aspirin")
- Set "has_both": true when user wants to analyze BOTH a gene AND a drug perturbation together
- For "X vs Y" or "compare X with Y" queries where one is a gene and one is a drug, set "has_both": true
- For "X vs Y" where both are the same type (both genes or both drugs), set "has_both": false and "is_comparison": true
- Extract pathway names if mentioned (e.g., "mTOR pathway", "PI3K signaling").
- Extract phenotypes if mentioned (e.g., "affecting cell proliferation", "that causes apoptosis").
- Detect if query focuses on proteins ("what proteins change?", "protein expression", "protein levels")
- Extract specific "top N" requests for genes, proteins, pathways, or phenotypes
- Detect what outputs the user wants (plots, genes list, proteins list, etc.)

Examples:
- "What happens if I knock down JAK1?" -> {"target": "JAK1", "type": "KD", "has_both": false, "confidence": 0.9, "is_comparison": false, "focus": null}
- "TP53 knockout and imatinib" -> {"target": "TP53", "target2": "imatinib", "type": "KO", "type2": "drug", "has_both": true, "confidence": 0.9}
- "Both JAK1 KD and aspirin" -> {"target": "JAK1", "target2": "aspirin", "type": "KD", "type2": "drug", "has_both": true, "confidence": 0.9}
- "Compare perturbation of Paclitaxel with PFDN4 loss" -> {"target": "PFDN4", "target2": "Paclitaxel", "type": "KO", "type2": "drug", "has_both": true, "is_comparison": true, "confidence": 0.9}
  NOTE: PFDN4 has "loss" = gene, Paclitaxel is a known drug name = drug. Order doesn't matter!
- "Paclitaxel vs PFDN4 knockout" -> {"target": "PFDN4", "target2": "Paclitaxel", "type": "KO", "type2": "drug", "has_both": true, "is_comparison": true, "confidence": 0.9}
  NOTE: PFDN4 has "knockout" = gene, Paclitaxel = drug. Order doesn't matter!
- "Show me top 10 proteins changed by TP53 knockout" -> {"target": "TP53", "type": "KO", "has_both": false, "top_n_proteins": 10, "focus": "proteins", "protein_mentioned": true, "output_types": ["proteins"], "confidence": 0.9}
- "Compare protein changes between TP53 KO and imatinib" -> {"target": "TP53", "target2": "imatinib", "type": "KO", "type2": "drug", "has_both": true, "is_comparison": true, "focus": "proteins", "protein_mentioned": true, "confidence": 0.9}
- "CHCHD2 vs Dimethyl fumarate" -> {"target": "CHCHD2", "target2": "Dimethyl fumarate", "type": "KO", "type2": "drug", "has_both": true, "is_comparison": true, "confidence": 0.9}
  NOTE: CHCHD2 is short uppercase = gene, Dimethyl fumarate is multi-word = drug
- "Find top 3 genes in PI3K pathway affecting apoptosis" -> {"pathway_mentioned": "PI3K", "phenotype_mentioned": "apoptosis", "top_n_genes": 3, "focus": "genes", "confidence": 0.85}
- "Show me the top 7 most changed proteins and their pathways" -> {"top_n_proteins": 7, "focus": "proteins", "protein_mentioned": true, "output_types": ["proteins", "pathways"], "confidence": 0.9}"""
    
    try:
        # Combine system prompt and user message for Gemini
        full_prompt = f"{extraction_prompt}\n\nUser query: {query}\n\nIMPORTANT: Return ONLY valid JSON, no other text."
        
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
        
        result = json.loads(content)
        result["original_query"] = query
        return _validate_perturbation_targets(result, query, llm_model=model)
    except Exception as exc:
        raise RuntimeError(f"Gemini extraction failed: {exc}") from exc


def extract_perturbation_info(query: str) -> Dict[str, Any]:
    """
    Alias for process_user_query for backward compatibility.
    
    Args:
        query: User's natural language question
    
    Returns:
        Dictionary with extracted perturbation information
    """
    return process_user_query(query)


def _validate_perturbation_targets(
    result: Dict[str, Any],
    original_query: str,
    llm_model: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Ensure extracted perturbations exist in training pickles before running downstream pipelines.
    """
    if result is None:
        return result
    
    result.setdefault("validation_messages", [])
    validated_targets = 0
    
    for field, type_field in (("target", "type"), ("target2", "type2")):
        requested_name = result.get(field)
        type_hint = result.get(type_field)
        
        result[f"{field}_requested"] = requested_name
        
        if not requested_name or requested_name == "UNKNOWN":
            result[field] = None
            result[f"{field}_validated"] = False
            if requested_name in (None, "", "UNKNOWN"):
                result[f"{field}_validation_error"] = "No perturbation specified."
            continue
        
        match_info = _match_with_training_set(
            requested_name,
            type_hint,
            original_query,
            llm_model=llm_model
        )
        result[f"{field}_match_info"] = match_info
        
        if match_info.get("match_type") != "none" and match_info.get("used_name"):
            result[field] = match_info["used_name"]
            result[f"{field}_validated"] = True
            result[f"{field}_match_type"] = match_info.get("match_type")
            result[f"{field}_match_method"] = match_info.get("method")
            result[f"{field}_match_similarity"] = match_info.get("similarity_score")
            result[f"{field}_perturbation_kind"] = match_info.get("candidate_type")
            validated_targets += 1
            
            # Update overall perturbation type if LLM guessed incorrectly
            if field == "target" and match_info.get("candidate_type") == "drug":
                result["type"] = "drug"
            elif field == "target" and match_info.get("candidate_type") == "gene" and result.get("type") == "unknown":
                # Keep original KO/KD/OE classification if provided, otherwise default to KO
                if type_hint not in {"KO", "KD", "OE"}:
                    result["type"] = "KO"
        else:
            result[field] = None
            result[f"{field}_validated"] = False
            error_msg = match_info.get("error") or f"Perturbation '{requested_name}' not found in training data."
            result[f"{field}_validation_error"] = error_msg
            result["validation_messages"].append(error_msg)
            if field == "target":
                result["type"] = "unknown"
    
    result["has_both"] = bool(result.get("target_validated") and result.get("target2_validated"))
    if validated_targets == 0:
        # Lower confidence if nothing could be validated
        result["confidence"] = min(result.get("confidence", 0.5), 0.2)
    elif validated_targets == 1:
        result["confidence"] = min(max(result.get("confidence", 0.5), 0.6), 0.95)
    
    return result


def _match_with_training_set(
    requested_name: str,
    type_hint: Optional[str],
    original_query: str,
    llm_model: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Validate a single perturbation against the available gene/drug lists.
    """
    match_info: Dict[str, Any] = {
        "requested_name": requested_name,
        "used_name": None,
        "match_type": "none",
        "similarity_score": 0.0,
        "candidate_type": None,
        "method": None,
        "error": None,
        "suggested_name": None
    }
    
    if not requested_name:
        match_info["error"] = "No perturbation specified."
        return match_info
    
    candidate_order = _infer_candidate_order(type_hint, requested_name)
    
    for candidate_type in candidate_order:
        candidates = _get_gene_names() if candidate_type == "gene" else _get_drug_names()
        if not candidates:
            continue
        
        # Direct exact match
        normalized_requested = _normalize_name(requested_name, is_gene=(candidate_type == "gene"))
        normalized_map = {
            _normalize_name(candidate, is_gene=(candidate_type == "gene")): candidate
            for candidate in candidates
        }
        if normalized_requested in normalized_map:
            match_info.update({
                "used_name": normalized_map[normalized_requested],
                "match_type": "exact",
                "similarity_score": 1.0,
                "candidate_type": candidate_type,
                "method": "exact"
            })
            return match_info
        
        # Ask Gemini to find closest formatted match using full list
        if llm_model:
            selection = _llm_select_candidate(
                original_query=original_query,
                requested_name=requested_name,
                candidate_type=candidate_type,
                candidates=candidates,
                llm_model=llm_model
            )
            if selection:
                match_info.update({
                    "used_name": selection,
                    "match_type": "close",
                    "similarity_score": 0.0,
                    "candidate_type": candidate_type,
                    "method": "llm"
                })
                return match_info
        
        # Fallback to fuzzy similarity if Gemini unavailable
        ranking = _rank_candidates_simple(requested_name, candidates, is_gene=(candidate_type == "gene"))
        if ranking["match_type"] != "none":
            match_info.update({
                "used_name": ranking["used_name"],
                "match_type": ranking["match_type"],
                "similarity_score": ranking.get("similarity_score", 0.0),
                "candidate_type": candidate_type,
                "method": "fuzzy"
            })
            return match_info
        if not match_info["suggested_name"] and ranking.get("used_name"):
            match_info["suggested_name"] = ranking["used_name"]
            match_info["candidate_type"] = candidate_type
    
    match_info["error"] = f"{requested_name} not found in {', '.join(candidate_order)} perturbation list."
    return match_info


def _rank_candidates_simple(
    requested_name: str,
    candidates: List[str],
    is_gene: bool
) -> Dict[str, Any]:
    if not candidates:
        return {
            "match_type": "none",
            "used_name": None,
            "similarity_score": 0.0
        }
    
    normalized_requested = _normalize_name(requested_name, is_gene=is_gene)
    best_score = 0.0
    best_name: Optional[str] = None
    
    for candidate in candidates:
        normalized_candidate = _normalize_name(candidate, is_gene=is_gene)
        if not normalized_candidate:
            continue
        score = SequenceMatcher(None, normalized_requested, normalized_candidate).ratio()
        if score > best_score:
            best_score = score
            best_name = candidate
    
    if best_score >= 0.80:
        return {
            "match_type": "close",
            "used_name": best_name,
            "similarity_score": best_score
        }
    
    return {
        "match_type": "none",
        "used_name": best_name,
        "similarity_score": best_score
    }


def _llm_select_candidate(
    original_query: str,
    requested_name: str,
    candidate_type: str,
    candidates: List[str],
    llm_model: Optional[Any]
) -> Optional[str]:
    """
    Use Gemini to select the closest candidate from the entire perturbation list.
    The LLM must return an exact string that already exists in var_dims['pert_names'].
    """
    if llm_model is None or not candidates:
        return None
    
    candidate_text = "\n".join(str(candidate) for candidate in candidates)
    
    prompt = f"""You must map a requested perturbation to an existing training perturbation.

User query: "{original_query}"
Requested perturbation: "{requested_name}"
Perturbation type: "{candidate_type.upper()}"

Available perturbations (exact strings you MUST choose from):
{candidate_text}

Return ONLY valid JSON in this format (no extra text):
{{"selection": "<exact string from the list above or NONE>"}}

Rules:
- If you find a match, the selection must be EXACTLY one of the strings above (including brackets, quotes, and formatting).
- If nothing is close, respond with "NONE".
"""
    
    try:
        response = llm_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        content = response.text.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        data = json.loads(content.strip())
        selection = data.get("selection") or data.get("match") or data.get("choice")
        if not selection:
            return None
        selection = str(selection).strip()
        if selection.upper() == "NONE":
            return None
        # Ensure the selection matches one of the available perturbations exactly
        for candidate in candidates:
            if str(candidate) == selection:
                return selection
        return None
    except Exception:
        return None


def _infer_candidate_order(type_hint: Optional[str], requested_name: str) -> List[str]:
    """
    Determine whether to search gene list, drug list, or both.
    """
    if isinstance(type_hint, str):
        normalized = type_hint.strip().lower()
        if normalized == "drug":
            return ["drug", "gene"]
        if normalized in {"ko", "kd", "oe"}:
            return ["gene", "drug"]
    
    if requested_name:
        stripped = requested_name.strip()
        if stripped.startswith("[(") or stripped.startswith("("):
            return ["drug", "gene"]
        if stripped.isupper() and any(char.isalpha() for char in stripped):
            return ["gene", "drug"]
        if stripped.lower() == stripped:
            return ["drug", "gene"]
    
    return ["gene", "drug"]


def _normalize_name(name: str, is_gene: bool) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    if is_gene:
        return "".join(text.replace("-", "").split()).upper()
    return " ".join(text.lower().split())


def _load_perturbation_names(path: Path) -> List[str]:
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        names = payload.get("pert_names", []) if isinstance(payload, dict) else []
        return [str(name) for name in names if name is not None]
    except Exception as exc:
        print(f"⚠️ Warning: Unable to load perturbation names from {path}: {exc}")
        return []


@lru_cache(maxsize=1)
def _get_gene_names() -> List[str]:
    return _load_perturbation_names(GENE_PERTURBATION_LIST_PATH)


@lru_cache(maxsize=1)
def _get_drug_names() -> List[str]:
    return _load_perturbation_names(DRUG_PERTURBATION_LIST_PATH)

