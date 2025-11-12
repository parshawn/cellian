"""Knowledge graph tool: find paths between nodes."""
from .registry import Tool


def kg_find_path_fn(args: dict) -> dict:
    """
    Mock KG path finder.
    Always returns a fixed path: source -> STAT1 -> HLA-A
    """
    source = args.get("source", "")
    targets = args.get("targets", [])
    max_hops = args.get("max_hops", 3)
    
    # Fixed mock path
    path = [
        {"src": source, "dst": "STAT1", "rel": "activates", "sign": "+"},
        {"src": "STAT1", "dst": "HLA-A", "rel": "activates", "sign": "+"}
    ]
    
    return {"paths": [path]}


kg_find_path = Tool(
    "kg.find_path",
    "Find signed paths source→targets in the KG",
    kg_find_path_fn
)

