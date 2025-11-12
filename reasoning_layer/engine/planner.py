"""LLM planner: generates tool execution plans."""
import json
import os
from typing import Dict
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


SYSTEM_PROMPT = """You are a mechanistic reasoning planner. You receive a query, sample data, and available tools.

Available tools:
1. perturbation.get_embedding: Get pre-computed aligned perturbation embedding (Node 1)
   Args: {"perturbation_name": str}
   
2. kg.find_path: Find signed paths source→targets in the KG
   Args: {"source": str, "targets": [str], "max_hops": int}
   
3. state.predict: Predict ΔRNA for a perturbation in context (Node 2)
   Args: {"target": str, "context": {"cell_line": str, "condition": str}, "embedding": [float]}
   
4. captain.translate: Translate ΔRNA to Δprotein on a panel (Node 3)
   Args: {"delta_rna": {gene: float}, "panel": [str]}
   
5. validate.all: Compute RNA/Protein metrics and edge-sign accuracy
   Args: {"pred_rna": {gene: float}, "obs_rna": {gene: float}, "pred_prot": {marker: float}, "obs_prot": {marker: float}, "path_edges": [{"src": str, "dst": str, "sign": "+"|"-"}, ...]}

Policy: Execute tools in this order for the 3-node chain:
1. perturbation.get_embedding (Node 1: get perturbation embedding)
2. kg.find_path (optional: get knowledge graph paths)
3. state.predict (Node 2: predict RNA changes from embedding)
4. captain.translate (Node 3: translate RNA to protein)
5. validate.all (compute validation metrics)

Return a JSON object with:
- "plan": [ordered list of tool names to execute]
- "tool_calls": [list of {"name": str, "args": dict} for each tool]
- "rationale": "2-4 sentences explaining the plan and expected outcomes"
"""

MODEL_NAME = "gemini-2.0-flash-exp"  # Best Gemini model (falls back to gemini-1.5-pro if unavailable)


def plan(query: str, sample: dict, tools_spec: Dict[str, str]) -> dict:
    """
    Generate a plan using LLM.
    
    Args:
        query: User query string
        sample: Sample data dict
        tools_spec: Dict mapping tool names to descriptions
    
    Returns:
        Parsed JSON plan dict
    """
    if genai is None:
        raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    # Strip whitespace to avoid authentication issues
    api_key = api_key.strip()
    
    # Validate key format (Gemini keys start with "AIza")
    if not api_key.startswith("AIza"):
        raise ValueError(f"Invalid API key format: expected to start with 'AIza', got '{api_key[:10]}...'")
    
    # Debug: Log key format (first/last few chars only for security)
    if len(api_key) < 20:
        raise ValueError(f"API key appears too short: {len(api_key)} characters")
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        # Provide helpful error message
        raise ValueError(
            f"Failed to configure Gemini API. "
            f"API key format: {api_key[:10]}...{api_key[-5:] if len(api_key) > 15 else '...'} "
            f"(length: {len(api_key)}). "
            f"Please verify your API key is valid and active. "
            f"Original error: {str(e)}"
        ) from e
    
    # Build user message
    perturbation = sample.get("perturbation", {})
    context = sample.get("context", {})
    protein = sample.get("protein", {})
    rna = sample.get("rna", {})
    obs_delta_rna = rna.get("obs_delta", {})
    
    # Preview of obs_rna (first 5 entries)
    obs_rna_preview = dict(list(obs_delta_rna.items())[:5])
    
    user_message = f"""Query: {query}

Perturbation: {perturbation.get('target')} ({perturbation.get('type')})
Context: {context.get('cell_line')}, {context.get('condition')}
Protein panel: {protein.get('panel', [])}
Observed RNA delta (preview): {obs_rna_preview}

Generate a plan to execute the tools in order."""
    
    # Try the best model first, fall back to gemini-1.5-pro if unavailable
    try:
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception:
        # Fall back to gemini-1.5-pro if the experimental model is unavailable
        model = genai.GenerativeModel("gemini-1.5-pro")
    
    # Combine system prompt and user message for Gemini
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_message}\n\nIMPORTANT: Return ONLY valid JSON, no other text."
    
    try:
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
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response from Gemini: {e}\nResponse: {content[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to generate plan with Gemini API: {str(e)}")

