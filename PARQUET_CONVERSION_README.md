# Parquet Conversion Guide

## The Problem

Your RNA expression CSV file is **24GB** and takes **60+ seconds** to load with pandas. This slows down:
- Testing in Jupyter notebooks
- Running the hypothesis engine
- Batch processing jobs

## The Solution

Convert CSV → Parquet format for:
- **10-50x faster** loading (60s → 2-5s)
- **5-10x smaller** files (24GB → ~2-5GB)
- **Same data**, better format

## Quick Start

### Option 1: Run Locally (if you have enough RAM)
```bash
cd /home/nebius/cellian
python convert_to_parquet.py
```

**Requirements**: ~48GB RAM (2x file size)

### Option 2: Run on Cluster (Recommended for large files)
```bash
sbatch convert_data.sh
```

This runs the conversion with 64GB RAM allocation on the cluster.

## What Gets Converted

The script converts:
- ✓ `RNA_expression.csv` → `RNA_expression.parquet`
- ✓ `Protein_expression.csv` → `Protein_expression.parquet`

Original CSV files are **not deleted** automatically (you can delete them manually after verifying).

## After Conversion

The hypothesis engine will **automatically** use Parquet files when available:

```python
from hypothesis_engine import ModelGraph, HypothesisAgent

graph = ModelGraph()
agent = HypothesisAgent(graph)  # Auto-detects and uses .parquet files!
```

You'll see:
```
Loading ground truth data...
  Loading RNA from Parquet (fast)...
  ✓ Loaded in 2.31s
  Loading Protein from Parquet (fast)...
```

## Monitoring Conversion

```bash
# Check job status
squeue -u $USER

# View conversion progress
tail -f logs/convert_*.out
```

## File Comparison

| Format | Size | Load Time | Memory |
|--------|------|-----------|--------|
| CSV | 24GB | 60+ seconds | ~24GB |
| Parquet | ~3-5GB | 2-5 seconds | ~8-12GB |

## Troubleshooting

### Out of Memory Error
- Use the Slurm script instead of local: `sbatch convert_data.sh`
- Increase memory in Slurm script if needed: `#SBATCH --mem=128G`

### Permission Denied
- Ensure scripts are executable: `chmod +x convert_data.sh`

### Parquet Not Found After Conversion
- Check output location: `ls -lh /home/nebius/cellian/data/perturb-cite-seq/SCP1064/other/*.parquet`
- Verify job completed successfully: Check logs in `logs/convert_*.out`

## Manual Usage in Jupyter

If you want to manually load Parquet files in your notebooks:

```python
import pandas as pd

# Instead of:
# df = pd.read_csv("RNA_expression.csv", index_col=0)

# Use:
df = pd.read_parquet("RNA_expression.parquet")
```

**Note**: The updated test_embeddings.ipynb (cell 3) now automatically checks for Parquet files.

## Reverting to CSV

If you need to go back to CSV for any reason, just delete the `.parquet` files:
```bash
rm /home/nebius/cellian/data/perturb-cite-seq/SCP1064/other/*.parquet
rm /home/nebius/cellian/data/perturb-cite-seq/SCP1064/expression/*.parquet
```

The hypothesis engine will fall back to CSV automatically.
