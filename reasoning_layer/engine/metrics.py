"""Statistical metrics: Pearson, Spearman correlation, and cosine similarity."""
import math
from typing import List


def pearson(x: List[float], y: List[float]) -> float:
    """
    Compute Pearson correlation coefficient.
    Returns NaN if variance is zero or lists are empty.
    """
    if len(x) != len(y) or len(x) == 0:
        return float("nan")
    
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)
    sum_y2 = sum(yi * yi for yi in y)
    
    numerator = n * sum_xy - sum_x * sum_y
    denom_x = n * sum_x2 - sum_x * sum_x
    denom_y = n * sum_y2 - sum_y * sum_y
    
    if denom_x == 0 or denom_y == 0:
        return float("nan")
    
    denominator = math.sqrt(denom_x * denom_y)
    if denominator == 0:
        return float("nan")
    
    return numerator / denominator


def spearman(x: List[float], y: List[float]) -> float:
    """
    Compute Spearman rank correlation coefficient with tie-aware ranking.
    Returns NaN if all ranks are tied or lists are empty.
    """
    if len(x) != len(y) or len(x) == 0:
        return float("nan")
    
    # Create pairs with original indices
    pairs_x = [(val, i) for i, val in enumerate(x)]
    pairs_y = [(val, i) for i, val in enumerate(y)]
    
    # Sort by value
    pairs_x.sort(key=lambda p: p[0])
    pairs_y.sort(key=lambda p: p[0])
    
    # Assign ranks with tie handling (average rank for ties)
    def assign_ranks(pairs: List[tuple]) -> List[float]:
        ranks = [0.0] * len(pairs)
        i = 0
        while i < len(pairs):
            # Find all ties at this position
            tie_start = i
            tie_value = pairs[i][0]
            while i < len(pairs) and pairs[i][0] == tie_value:
                i += 1
            tie_end = i
            
            # Average rank for this group
            avg_rank = (tie_start + tie_end + 1) / 2.0
            
            # Assign to all in this tie group
            for j in range(tie_start, tie_end):
                ranks[pairs[j][1]] = avg_rank
        
        return ranks
    
    ranks_x = assign_ranks(pairs_x)
    ranks_y = assign_ranks(pairs_y)
    
    # Compute Pearson correlation on ranks
    return pearson(ranks_x, ranks_y)


def cosine_similarity(x: List[float], y: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns NaN if either vector has zero magnitude or lists are empty.
    
    Formula: cos(θ) = (x · y) / (||x|| * ||y||)
    
    Args:
        x: First vector
        y: Second vector
    
    Returns:
        Cosine similarity value between -1 and 1
    """
    if len(x) != len(y) or len(x) == 0:
        return float("nan")
    
    # Compute dot product
    dot_product = sum(xi * yi for xi, yi in zip(x, y))
    
    # Compute magnitudes
    norm_x = math.sqrt(sum(xi * xi for xi in x))
    norm_y = math.sqrt(sum(yi * yi for yi in y))
    
    # Check for zero magnitude
    if norm_x == 0 or norm_y == 0:
        return float("nan")
    
    # Cosine similarity
    return dot_product / (norm_x * norm_y)

