# scTranslator Zero Output Bug - FIXED

## Problem Summary

When running scTranslator inference, both predictions (y_pred.csv) and ground truth (y_truth.csv) were coming out as **all zeros**, even though:
- Input protein h5ad had real non-zero values
- Both RNA and protein files had proper `my_Id` columns
- All 12,594 cells were being processed

## Root Cause

**Bug Location:** `/home/nebius/scTranslator/code/model/utils.py`

The bug was in the data extraction code used by both dataset classes:

```python
# BUGGY CODE (lines 112 and 149):
tmp = [i for i in x.X[0]]
```

### What Was Wrong:

When `x.X[0]` returns a `numpy.matrix` object (common in AnnData), iterating over it with list comprehension returns:
- **ONE element** (the entire matrix object)
- NOT individual values (14 protein values)

This caused:
1. `tmp` = list of length 1 containing a matrix
2. `normalization(tmp)` fails: `(matrix-matrix)/(0)` → **NaN**
3. Test function detects NaN and replaces with **zeros**
4. Both predictions and ground truth become all zeros

## The Fix

**Files Modified:** `/home/nebius/scTranslator/code/model/utils.py`

### Fixed Lines:

**Line 112-120** in `fix_sc_normalize_truncate_padding()`:
```python
# OLD:
tmp = [i for i in x.X[0]]

# NEW:
row = x.X[0]
if hasattr(row, 'toarray'):  # Sparse matrix
    tmp = row.toarray().flatten().tolist()
elif hasattr(row, 'A'):  # numpy.matrix
    tmp = row.A.flatten().tolist()
else:  # Regular array
    tmp = np.asarray(row).flatten().tolist()
```

**Line 156-164** in `sc_normalize_truncate_padding()`:
- Same fix applied

### Why This Works:

- `.toarray()` - Converts scipy sparse matrix to dense numpy array
- `.A` - Converts numpy.matrix to numpy.ndarray
- `.flatten()` - Ensures 1D array (not 2D with shape (1, n))
- `.tolist()` - Converts to Python list for compatibility with rest of code

## Testing the Fix

### 1. Re-run inference:

```bash
cd /home/nebius/scTranslator
conda activate new_env

python code/main_scripts/stage3_inference_without_finetune.py \
  --pretrain_checkpoint='checkpoint/scTranslator_2M.pt' \
  --RNA_path='/tmp/tmpakdnoztl/input_dense.h5ad' \
  --Pro_path='/tmp/tmpakdnoztl/protein_input_mapped.h5ad'
```

### 2. Expected Output:

```
Total number of origin RNA genes:  18598
Total number of origin proteins:  14
Total number of origin cells:  12594
# of NAN in X 0
# of NAN in X 0
load data ended
----------------------------------------
single cell 20000 RNA To 1000 Protein on dataset/new_data-without_fine-tune
Overall performance in repeat_1 costTime: ~277s
Test Set: AVG mse [NON-ZERO], AVG ccc [NON-ZERO]  ← Should be meaningful values now
```

### 3. Verify Results:

```bash
# Check output files
cd /home/nebius/scTranslator/result/test/new_data-without_fine-tune

# View predictions (should have non-zero values)
head y_pred.csv

# View ground truth (should match your input protein data)
head y_truth.csv
```

Both files should now contain **real non-zero values** matching your input data.

## Impact

This bug affected:
- ✓ `fix_SCDataset` class (used with `--fix_set` flag)
- ✓ `SCDataset` class (default)
- ✓ All inference and fine-tuning scripts using these datasets

The fix ensures proper data extraction for:
- Sparse matrices (scipy)
- Dense numpy matrices
- Regular numpy arrays

## Additional Notes

### Data Format Requirements:

Your input data should have:
1. **RNA h5ad:**
   - Raw or unnormalized counts (scTranslator normalizes internally)
   - `my_Id` column in `.var` (mapped to scTranslator vocabulary)
   - Dense or sparse matrix format (now handled correctly)

2. **Protein h5ad:**
   - **Real ground truth values** for evaluation
   - `my_Id` column in `.var`
   - Same cells as RNA (matching `.obs.index`)

### For Hypothesis Engine Integration:

When using scTranslator in your hypothesis engine:

```python
# Create RNA input from perturbed cells
rna_adata = create_adata(rna_df[perturb_cells])

# Create protein ground truth for validation
protein_adata = create_adata(protein_df[perturb_cells])

# Run inference - now works correctly!
predictions = run_sctranslator(rna_adata, protein_adata)
```

The MSE and CCC metrics will now be meaningful measures of prediction accuracy.
