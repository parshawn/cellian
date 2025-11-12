"""LLM interpreter: interprets results and generates natural language summaries."""
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env file from reasoning_layer directory
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

try:
    import google.generativeai as genai
except ImportError:
    genai = None


SYSTEM_PROMPT = """You are a biological reasoning interpreter. You receive the results of a mechanistic reasoning pipeline and generate a comprehensive natural language summary.

Your task is to:
1. Interpret the hypothesis and predictions
2. Explain the validation scores and what they mean
3. Describe the error propagation through the chain
4. Provide biological insights based on the results
5. Summarize key findings in 2-4 sentences

Be clear, concise, and focus on biological significance."""

MODEL_NAME = "gemini-2.0-flash-exp"  # Best Gemini model (falls back to gemini-1.5-pro if unavailable)


def interpret_results(results: Dict[str, Any], query: Optional[str] = None) -> str:
    """
    Interpret results and generate natural language summary using LLM.
    
    Args:
        results: Results dictionary from HypothesisAgent
        query: Original user query (optional)
    
    Returns:
        Natural language interpretation of results
    """
    if genai is None:
        return _generate_fallback_summary(results)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _generate_fallback_summary(results)
    
    # Strip whitespace to avoid authentication issues
    api_key = api_key.strip()
    
    try:
        genai.configure(api_key=api_key)
    except Exception:
        return _generate_fallback_summary(results)
    
    # Prepare results summary for LLM
    results_summary = {
        "hypothesis": results.get("hypothesis", ""),
        "perturbation_name": results.get("perturbation_name", ""),
        "validation_scores": results.get("validation_scores", {}),
        "error_propagation": results.get("error_propagation", {}),
        "path": results.get("path", [])
    }
    
    # Get top predictions
    predictions = results.get("predictions", {})
    rna_pred = predictions.get("rna", {}).get("delta", {})
    prot_pred = predictions.get("protein", {}).get("delta", {})
    
    # Top 5 RNA changes
    top_rna = sorted(rna_pred.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_rna_summary = {gene: val for gene, val in top_rna}
    
    # Top 5 protein changes
    top_prot = sorted(prot_pred.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_prot_summary = {marker: val for marker, val in top_prot}
    
    user_message = f"""Original query: {query or "Not provided"}

Results Summary:
{json.dumps(results_summary, indent=2)}

Top RNA predictions: {top_rna_summary}
Top protein predictions: {top_prot_summary}

Generate a comprehensive natural language interpretation of these results, focusing on:
1. What the hypothesis predicts
2. How accurate the predictions are (based on validation scores)
3. What the error propagation tells us
4. Biological significance and insights"""
    
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
                max_output_tokens=500
            )
        )
        
        return response.text
    except Exception as e:
        return _generate_fallback_summary(results)


def _generate_fallback_summary(results: Dict[str, Any]) -> str:
    """Generate fallback summary without LLM."""
    perturbation_name = results.get("perturbation_name", "unknown")
    hypothesis = results.get("hypothesis", "")
    validation = results.get("validation_scores", {})
    
    rna_spearman = validation.get("rna", {}).get("spearman")
    prot_pearson = validation.get("protein", {}).get("pearson")
    
    summary = f"Perturbation Analysis: {perturbation_name}\n\n"
    summary += f"Hypothesis: {hypothesis}\n\n"
    
    if rna_spearman is not None:
        summary += f"RNA prediction accuracy (Spearman): {rna_spearman:.3f}\n"
    if prot_pearson is not None:
        summary += f"Protein prediction accuracy (Pearson): {prot_pearson:.3f}\n"
    
    return summary


def extract_perturbation_info(query: str) -> Dict[str, Any]:
    """
    Extract perturbation information from natural language query.
    
    Args:
        query: Natural language query (e.g., "What happens if I knock down JAK1?")
    
    Returns:
        Dictionary with perturbation info: {"target": str, "type": str}
    """
    if genai is None:
        return _extract_perturbation_fallback(query)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _extract_perturbation_fallback(query)
    
    try:
        genai.configure(api_key=api_key)
    except Exception:
        return _extract_perturbation_fallback(query)
    
    extraction_prompt = """Extract perturbation information from the user's question.

Return a JSON object with:
- "target": gene name (e.g., "JAK1")
- "type": perturbation type ("KO", "KD", "OE", or "unknown")
- "confidence": confidence level (0.0 to 1.0)

Examples:
- "What happens if I knock down JAK1?" -> {"target": "JAK1", "type": "KD", "confidence": 0.9}
- "Knock out STAT1" -> {"target": "STAT1", "type": "KO", "confidence": 0.95}
- "Overexpress HLA-A" -> {"target": "HLA-A", "type": "OE", "confidence": 0.9}"""
    
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
        
        return json.loads(content)
    except Exception as e:
        return _extract_perturbation_fallback(query)


def _extract_perturbation_fallback(query: str) -> Dict[str, Any]:
    """Fallback perturbation extraction using simple heuristics."""
    query_lower = query.lower()
    
    # Try to find gene names (simple heuristic - look for common patterns)
    # This is a fallback - LLM extraction is much better
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
        "confidence": 0.5
    }

