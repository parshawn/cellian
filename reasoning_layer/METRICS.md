# Metrics for Ground Truth Comparison

## Overview

The reasoning_layer uses multiple metrics to compare predicted profiles with ground truth data. These metrics are computed at each step of the 3-node chain and used for validation and error propagation analysis.

## Metrics Used

### 1. RNA Prediction Metrics (Node 2)

**Primary Metric: Spearman Correlation**
- **Purpose**: Measures rank correlation between predicted and real RNA changes
- **Range**: -1 to +1
- **Interpretation**: 
  - +1: Perfect positive rank correlation
  - 0: No rank correlation
  - -1: Perfect negative rank correlation
- **Why Spearman**: Rank-based metric is robust to outliers and non-linear relationships
- **Implementation**: Tie-aware ranking (handles duplicate values)

**Additional Metrics:**
- **Cosine Similarity**: Measures the angle between predicted and real RNA vectors
  - **Range**: -1 to +1
  - **Interpretation**: 
    - +1: Vectors point in the same direction (perfect similarity)
    - 0: Vectors are orthogonal (no similarity)
    - -1: Vectors point in opposite directions
  - **Why Cosine**: Useful for comparing high-dimensional expression profiles, measures directional similarity regardless of magnitude
- **MSE (Mean Squared Error)**: Average squared difference between predicted and real values
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and real values

### 2. Protein Prediction Metrics (Node 3)

**Primary Metric: Pearson Correlation**
- **Purpose**: Measures linear correlation between predicted and real protein changes
- **Range**: -1 to +1
- **Interpretation**:
  - +1: Perfect positive linear correlation
  - 0: No linear correlation
  - -1: Perfect negative linear correlation
- **Why Pearson**: Assumes linear relationship between RNA and protein (translation step)

**Additional Metrics:**
- **Cosine Similarity**: Measures the angle between predicted and real protein vectors
  - **Range**: -1 to +1
  - **Interpretation**: 
    - +1: Vectors point in the same direction (perfect similarity)
    - 0: Vectors are orthogonal (no similarity)
    - -1: Vectors point in opposite directions
  - **Why Cosine**: Useful for comparing high-dimensional protein profiles, measures directional similarity regardless of magnitude
- **MSE (Mean Squared Error)**: Average squared difference between predicted and real values
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and real values

### 3. Error Propagation Metrics

**Total Error**: 
- Formula: `sqrt(rna_error² + protein_error² + translation_error²)`
- Measures cumulative error through the 3-node chain

**Error Amplification**:
- Formula: `protein_error / rna_error`
- Measures how much error increases from RNA to protein prediction

**Error Contributions**:
- `rna_contribution`: Fraction of total error from RNA prediction step
- `protein_contribution`: Fraction of total error from protein prediction step
- `translation_contribution`: Fraction of total error from RNA→Protein translation

## Implementation Details

### Comparison Function

```python
def compare_profiles(pred, real, metric="pearson"):
    # Returns:
    # - metric: "pearson", "spearman", or "cosine"
    # - value: Correlation coefficient or cosine similarity
    # - n_features: Number of common features compared
    # - mse: Mean Squared Error
    # - mae: Mean Absolute Error
```

### Metric Calculations

**Pearson Correlation**:
- Measures linear relationship between two variables
- Sensitive to outliers
- Assumes normal distribution
- Formula: `cov(X,Y) / (std(X) * std(Y))`

**Spearman Correlation**:
- Measures rank correlation (monotonic relationship)
- Robust to outliers
- No distribution assumptions
- Formula: Pearson correlation on ranks

**Cosine Similarity**:
- Measures the cosine of the angle between two vectors
- Scale-invariant (magnitude-independent)
- Useful for high-dimensional embeddings and profiles
- Formula: `(x · y) / (||x|| * ||y||)`
- Returns NaN if either vector has zero magnitude

**MSE (Mean Squared Error)**:
- Formula: `mean((pred - real)²)`
- Penalizes large errors more than small errors
- Used for error propagation calculations

**MAE (Mean Absolute Error)**:
- Formula: `mean(|pred - real|)`
- Equal weight to all errors
- More interpretable than MSE

## Usage in HypothesisAgent

```python
# RNA comparison
rna_comparison = compare_profiles(
    pred_rna_delta, 
    real_rna_delta, 
    metric="spearman"  # Rank-based for RNA
)
rna_cosine = compare_profiles(
    pred_rna_delta, 
    real_rna_delta, 
    metric="cosine"  # Directional similarity for RNA
)

# Protein comparison
prot_comparison = compare_profiles(
    pred_prot_delta, 
    real_prot_delta, 
    metric="pearson"  # Linear correlation for protein
)
prot_cosine = compare_profiles(
    pred_prot_delta, 
    real_prot_delta, 
    metric="cosine"  # Directional similarity for protein
)

# Error propagation
error_prop = calculate_error_propagation(
    rna_error=rna_comparison["mse"],
    protein_error=prot_comparison["mse"],
    rna_to_protein_error=translation_error
)
```

## Output Format

The metrics are included in the final JSON output:

```json
{
  "validation_scores": {
    "rna": {
      "spearman": 0.85,
      "cosine": 0.82,
      "mse": 0.12,
      "mae": 0.08,
      "n_features": 100
    },
    "protein": {
      "pearson": 0.78,
      "cosine": 0.75,
      "mse": 0.15,
      "mae": 0.10,
      "n_features": 20
    },
    "edge_sign_accuracy": 0.80,
    "kg_edges_used": 5
  },
  "error_propagation": {
    "total_error": 0.19,
    "error_amplification": 1.25,
    "rna_error": 0.12,
    "protein_error": 0.15,
    "translation_error": 0.03,
    "rna_contribution": 0.63,
    "protein_contribution": 0.79,
    "translation_contribution": 0.16
  }
}
```

## Metric Selection Rationale

### Why Spearman for RNA?
- RNA expression changes can be non-linear
- Rank-based metrics are more robust to outliers
- Focuses on relative ordering of genes (which genes are up/down regulated)

### Why Pearson for Protein?
- Protein levels often have linear relationships with RNA
- Translation step (RNA→Protein) is typically modeled linearly
- Captures magnitude of changes, not just ranks

### Why MSE for Error Propagation?
- MSE penalizes large errors more (quadratic)
- Suitable for error propagation calculations
- Allows decomposition of total error into components

### Why Cosine Similarity?
- Scale-invariant: Focuses on directional similarity, not magnitude
- Useful for high-dimensional embeddings and expression profiles
- Complements correlation metrics by measuring angle between vectors
- Particularly useful when comparing profiles with different scales or normalization

### Why Edge-Sign Accuracy?
- Validates causal relationships: Checks if predicted changes match expected directions from knowledge graph
- Provides mechanistic validation beyond profile comparisons
- Measures biological plausibility of predictions
- Conditional metric: Only computed when KG path edges are available

## Current Implementation Status

### ✅ Implemented and Used
- **RNA Spearman Correlation**: Used in HypothesisAgent
- **RNA Cosine Similarity**: Used in HypothesisAgent
- **Protein Pearson Correlation**: Used in HypothesisAgent
- **Protein Cosine Similarity**: Used in HypothesisAgent
- **MSE and MAE**: Computed for both RNA and protein
- **Error Propagation**: Calculated using MSE values

### ✅ Now Integrated and Used
- **Edge-Sign Accuracy**: Computed by HypothesisAgent when KG path edges are available
  - Measures fraction of KG edges where predicted sign matches expected sign
  - Queries KG using `kg.find_path` with predicted genes as targets
  - Returns NaN if KG edges are not available or query fails
  - Validates causal relationships: checks if predicted RNA changes match expected directions from knowledge graph
  - Included in `validation_scores.edge_sign_accuracy` and `validation_scores.kg_edges_used`

## Limitations

1. **Missing Features**: Metrics only computed on intersection of predicted and real features
2. **No Weighting**: All features weighted equally (could weight by importance)
3. **No Significance Testing**: No p-values or confidence intervals
4. **Multiple Metrics**: Now uses Spearman/Pearson + Cosine for comprehensive evaluation
5. **Edge-Sign Accuracy Conditional**: Only computed when KG path edges are available (may be NaN if KG unavailable or query fails)

## Future Enhancements

- Add R² (coefficient of determination)
- Add concordance correlation coefficient (CCC)
- Add significance testing (p-values)
- Add feature-weighted metrics
- Add per-gene/per-protein metrics
- Add visualization of metric distributions

