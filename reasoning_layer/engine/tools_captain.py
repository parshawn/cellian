"""CAPTAIN tool: translate RNA to protein."""
import random
from .registry import Tool


def captain_translate_fn(args: dict) -> dict:
    """
    Mock CAPTAIN translator.
    Returns deterministic Δprotein values (seed=13).
    """
    delta_rna = args.get("delta_rna", {})
    panel = args.get("panel", [])
    
    # Deterministic RNG (seed=13)
    rng = random.Random(13)
    delta_protein = {marker: rng.uniform(-1.0, 1.0) for marker in panel}
    
    return {"delta_protein": delta_protein}


captain_translate = Tool(
    "captain.translate",
    "Translate ΔRNA to Δprotein on a panel",
    captain_translate_fn
)

