# STATE Integration Updates

Based on the official [STATE documentation](https://pypi.org/project/arc-state/), I've updated the code to use the correct API.

## Key Changes

### 1. Command Updated
**Before**: `state emb --input input.h5ad --output output.h5ad`
**After**: `state emb transform --input input.h5ad --output output.h5ad`

### 2. Input Data Requirements

STATE requires specific AnnData format:

| Requirement | Description | How to Fix |
|------------|-------------|------------|
| **CSR Matrix** | `.X` must be scipy CSR sparse matrix | `from scipy.sparse import csr_matrix`<br>`X = csr_matrix(data)` |
| **gene_name** | `.var` must have `gene_name` column | `var = pd.DataFrame({'gene_name': gene_names}, index=gene_names)` |

### 3. Optional Parameters

```bash
# Specify model checkpoint
state emb transform \
  --model-folder /path/to/SE-600M \
  --checkpoint /path/to/checkpoint.ckpt \
  --input input.h5ad \
  --output output.h5ad

# Custom gene column name
state emb transform \
  --input input.h5ad \
  --output output.h5ad \
  --gene-column gene_symbols
```

## Updated Files

### ✓ test_embeddings.ipynb
- **Cell 0**: Added STATE requirements documentation
- **Cell 3**: Added Parquet support for faster loading
- **Cell 7**: Added CSR matrix conversion and `gene_name` column
- **Cell 9**: Updated command to `state emb transform` with error handling

### ✓ hypothesis_engine.py
- **Line 60**: Updated command from `state emb` → `state emb transform`
- **Line 46-70**: Added documentation about CSR and gene_name requirements

## Testing the Notebook

```bash
cd /home/nebius/cellian
jupyter notebook test_embeddings.ipynb
```

**Run the cells in order**. You should see:

```
✓ STATE embedding completed successfully
STATE Embedding Format:
  Shape: (1536,)  # or similar embedding dimension
  Type: <class 'numpy.ndarray'>
```

## Expected Output Location

STATE embeddings are typically stored in:
- `.obsm['X_state']` - Most common
- `.obsm['state']` - Alternative
- `.X` - Fallback if above don't exist

The notebook checks all three locations automatically.

## Troubleshooting

### Error: "STATE embedding failed"
1. Check STATE is installed: `state --help`
2. Verify input h5ad has CSR matrix: `print(type(adata.X))`
3. Verify gene_name exists: `print('gene_name' in adata.var.columns)`

### Error: Command not found
```bash
# Reinstall STATE
uv tool install arc-state

# Verify installation
which state
state --help
```

### Slow Loading (CSV taking too long)
```bash
# Convert to Parquet first
python convert_to_parquet.py
# or
sbatch convert_data.sh
```

## Next Steps

Once the notebook runs successfully:
1. Test with different perturbations
2. Run the full hypothesis engine: `python hypothesis_engine.py`
3. Submit batch jobs: `sbatch run_hypothesis_batch.sh`
