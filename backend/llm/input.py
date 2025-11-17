"""LLM input processing: handles user questions and extracts perturbation information."""
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

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
    if genai is None:
        return _extract_perturbation_fallback(query)
    
    # Try both GOOGLE_API_KEY and GEMINI_API_KEY (for compatibility)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _extract_perturbation_fallback(query)
    
    # Strip whitespace to avoid authentication issues
    api_key = api_key.strip()
    
    # Validate key format (Gemini keys start with "AIza")
    if not api_key.startswith("AIza"):
        return _extract_perturbation_fallback(query)
    
    # Debug: Log key format (first/last few chars only for security)
    if len(api_key) < 20:
        return _extract_perturbation_fallback(query)
    
    try:
        genai.configure(api_key=api_key)
    except Exception:
        return _extract_perturbation_fallback(query)
    
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
- Extract BOTH genes and drugs if the user mentions BOTH a gene perturbation AND a drug (e.g., "TP53 KO and imatinib", "both JAK1 knockdown and aspirin")
- Set "has_both": true when user wants to analyze BOTH a gene AND a drug perturbation together
- For "X vs Y" queries where one is a gene and one is a drug, set "has_both": true (e.g., "CHCHD2 vs Dimethyl fumarate" -> both should be analyzed)
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
- "Show me top 10 proteins changed by TP53 knockout" -> {"target": "TP53", "type": "KO", "has_both": false, "top_n_proteins": 10, "focus": "proteins", "protein_mentioned": true, "output_types": ["proteins"], "confidence": 0.9}
- "Compare protein changes between TP53 KO and imatinib" -> {"target": "TP53", "target2": "imatinib", "type": "KO", "type2": "drug", "has_both": true, "is_comparison": true, "focus": "proteins", "protein_mentioned": true, "confidence": 0.9}
- "CHCHD2 vs Dimethyl fumarate" -> {"target": "CHCHD2", "target2": "Dimethyl fumarate", "type": "KO", "type2": "drug", "has_both": true, "is_comparison": true, "confidence": 0.9}
- "Find top 3 genes in PI3K pathway affecting apoptosis" -> {"pathway_mentioned": "PI3K", "phenotype_mentioned": "apoptosis", "top_n_genes": 3, "focus": "genes", "confidence": 0.85}
- "Show me the top 7 most changed proteins and their pathways" -> {"top_n_proteins": 7, "focus": "proteins", "protein_mentioned": true, "output_types": ["proteins", "pathways"], "confidence": 0.9}"""
    
    try:
        # Try the best model first, fall back to gemini-1.5-pro if unavailable
        try:
            model = genai.GenerativeModel(MODEL_NAME)
        except Exception:
            model = genai.GenerativeModel("gemini-1.5-pro")
        
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
        return result
    except Exception as e:
        return _extract_perturbation_fallback(query)


def extract_perturbation_info(query: str) -> Dict[str, Any]:
    """
    Alias for process_user_query for backward compatibility.
    
    Args:
        query: User's natural language question
    
    Returns:
        Dictionary with extracted perturbation information
    """
    return process_user_query(query)


def _extract_perturbation_fallback(query: str) -> Dict[str, Any]:
    """Fallback perturbation extraction using simple heuristics."""
    query_lower = query.lower()
    
    # Try to find gene names (simple heuristic - look for common patterns)
    target = "UNKNOWN"
    pert_type = "unknown"
    
    # Common perturbation keywords
    if "knock down" in query_lower or "knockdown" in query_lower or " kd " in query_lower:
        pert_type = "KD"
    elif "knock out" in query_lower or "knockout" in query_lower or " ko " in query_lower:
        pert_type = "KO"
    elif "overexpress" in query_lower or "over express" in query_lower or " oe " in query_lower:
        pert_type = "OE"
    
    # Try to extract gene name (look for uppercase words or common gene name patterns)
    words = query.split()
    for word in words:
        # Remove punctuation
        clean_word = word.strip('.,!?;:()[]{}')
        # Check if it looks like a gene name (uppercase, 2-10 chars, mostly letters)
        if clean_word.isupper() and 2 <= len(clean_word) <= 10 and clean_word.isalpha():
            target = clean_word
            break
        # Also check capitalized words (like "JAK1" with numbers)
        elif clean_word and clean_word[0].isupper() and any(c.isdigit() for c in clean_word):
            target = clean_word.upper()
            break
    
    return {
        "target": target,
        "type": pert_type,
        "confidence": 0.5,
        "original_query": query
    }

