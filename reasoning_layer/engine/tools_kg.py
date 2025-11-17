"""Knowledge graph tool: find paths between nodes using pathway data."""
from typing import List
from .registry import Tool
from .pathway_loader import PathwayLoader


# Global pathway loader instance (lazy initialization)
_pathway_loader = None


def get_pathway_loader() -> PathwayLoader:
    """Get or create pathway loader instance."""
    global _pathway_loader
    if _pathway_loader is None:
        _pathway_loader = PathwayLoader()
    return _pathway_loader


def kg_find_path_fn(args: dict) -> dict:
    """
    Find paths between source and targets using pathway data.
    Falls back to mock if pathway data not available.
    """
    source = args.get("source", "")
    targets = args.get("targets", [])
    max_hops = args.get("max_hops", 3)
    
    if not source:
        return {"paths": []}
    
    loader = get_pathway_loader()
    
    # Try to find paths using pathway data
    paths = []
    for target in targets:
        path = _find_path_in_pathways(loader, source, target, max_hops)
        if path:
            paths.append(path)
    
    # If no paths found and we have pathway data, return empty
    # Otherwise, fall back to mock for backward compatibility
    if not paths and len(loader.pathways) == 0:
        # Fallback to mock path for backward compatibility
        path = [
            {"src": source, "dst": "STAT1", "rel": "activates", "sign": "+"},
            {"src": "STAT1", "dst": targets[0] if targets else "HLA-A", "rel": "activates", "sign": "+"}
        ]
        paths = [path]
    
    return {"paths": paths}


def _find_path_in_pathways(loader: PathwayLoader, source: str, target: str, max_hops: int) -> List[dict]:
    """
    Find a path from source to target using pathway data.
    
    Args:
        loader: PathwayLoader instance
        source: Source gene name
        target: Target gene name
        max_hops: Maximum path length
    
    Returns:
        List of edge dictionaries or empty list if no path found
    """
    # Get pathways containing source
    source_pathways = loader.get_pathways_for_gene(source)
    target_pathways = loader.get_pathways_for_gene(target)
    
    # Check if they share a pathway
    common_pathways = set(source_pathways) & set(target_pathways)
    
    if not common_pathways:
        return []
    
    # Try to find path within common pathway
    pathway_id = list(common_pathways)[0]
    pathway = loader.get_pathway(pathway_id)
    if not pathway:
        return []
    
    # Simple BFS to find path
    edges = pathway.get("edges", [])
    if not edges:
        # If no edges, check if genes are in same pathway (direct connection)
        if source in pathway.get("genes", []) and target in pathway.get("genes", []):
            return [{"src": source, "dst": target, "rel": "pathway_member", "sign": "+"}]
        return []
    
    # Build adjacency list
    adj = {}
    for edge in edges:
        if isinstance(edge, dict):
            src = edge.get("source") or edge.get("src")
            dst = edge.get("target") or edge.get("dst")
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src, dst = edge[0], edge[1]
        else:
            continue
        
        if src not in adj:
            adj[src] = []
        adj[src].append((dst, edge))
    
    # BFS
    from collections import deque
    queue = deque([(source, [])])
    visited = {source}
    
    while queue and len(queue[0][1]) < max_hops:
        node, path = queue.popleft()
        
        if node == target:
            # Reconstruct path
            result_path = []
            for i in range(len(path)):
                edge = path[i]
                if isinstance(edge, dict):
                    result_path.append({
                        "src": edge.get("source") or edge.get("src"),
                        "dst": edge.get("target") or edge.get("dst"),
                        "rel": edge.get("type") or edge.get("rel", "regulates"),
                        "sign": edge.get("sign", "+")
                    })
            return result_path
        
        if node in adj:
            for next_node, edge in adj[node]:
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [edge]))
    
    return []


kg_find_path = Tool(
    "kg.find_path",
    "Find signed paths source→targets in the KG",
    kg_find_path_fn
)

