# Multi-Omics Hypothesis Engine

Core framework for running reasoning chains across foundation models (STATE + CAPTAIN) validated against Perturb-CITE-seq ground truth.

## Quick Start

### Test Embeddings (Jupyter Notebook)
```bash
# Open the testing notebook to verify embeddings and formats
cd /home/nebius/cellian
jupyter notebook test_embeddings.ipynb
```

This notebook shows you:
- How STATE and CAPTAIN embeddings are generated
- The exact format and shape of each step
- How to test with a single perturbation
- How to save/load embeddings

### Local Run (CPU)
```bash
# Navigate to the cellian directory
cd /home/nebius/cellian

# Run the hypothesis engine
python hypothesis_engine.py
```

### Slurm Cluster Run (GPU)
```bash
# Run single test with GPU
sbatch run_hypothesis_engine.sh

# Run batch processing of multiple perturbations with GPU
sbatch run_hypothesis_batch.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/hypothesis_batch_<JOB_ID>.out
```

## What It Does

The engine runs a 3-node reasoning chain:

1. **Perturbation Node** → Creates AnnData with baseline control RNA profile
2. **RNA Node (STATE)** → Generates RNA embeddings via STATE CLI tool
3. **Protein Node (CAPTAIN)** → Predicts protein expression from RNA embeddings

Then validates predictions against real Perturb-CITE-seq data using cosine similarity.

## Requirements

Make sure you have the following installed:
```bash
# Python packages
pip install torch pandas numpy scipy anndata

# STATE tool (already installed via uv)
uv tool install arc-state
```

## Using in Your Code

```python
from hypothesis_engine import ModelGraph, HypothesisAgent

# Initialize models (optionally specify STATE model path)
graph = ModelGraph(state_model_path="/path/to/SE-600M")  # or None for default

# Create agent with ground truth data
agent = HypothesisAgent(graph)

# Generate hypothesis for any perturbation
result = agent.generate_hypothesis("KO of CD58")

print(f"RNA Score: {result['rna_score']}")
print(f"Protein Score: {result['protein_score']}")
print(f"Error Propagation: {result['error_propagation']}")
```

## How STATE Integration Works

The framework uses STATE as a command-line tool (not a Python API):
1. Creates temporary AnnData (.h5ad) file with control RNA profile
2. Runs `state emb --input input.h5ad --output output.h5ad`
3. Loads embeddings from output AnnData file
4. Passes embeddings to CAPTAIN for protein prediction

## Output Format

The `generate_hypothesis()` method returns:
```python
{
    'perturbation': str,           # The perturbation name
    'path': str,                   # The reasoning path taken
    'rna_score': float,            # Cosine similarity for RNA prediction
    'protein_score': float,        # Cosine similarity for protein prediction
    'error_propagation': float     # rna_score - protein_score
}
```

## Data Paths

The engine expects the following files:
- `/home/nebius/cellian/data/perturb-cite-seq/SCP1064/metadata/RNA_metadata.csv`
- `/home/nebius/cellian/data/perturb-cite-seq/SCP1064/other/RNA_expression.csv`
- `/home/nebius/cellian/data/perturb-cite-seq/SCP1064/expression/Protein_expression.csv`
- `/home/nebius/cellian/foundation_models/CAPTAIN_BASE/CAPTAIN_Base.pt`

The STATE model can be specified via `state_model_path` parameter or uses the default from your installation.

## Batch Processing

For processing multiple perturbations at scale, use the batch runner:

### Command Line Options

```bash
python run_hypothesis_batch.py \
    --output-dir /home/nebius/cellian/results \
    --max-perturbations 50  # Limit for testing

# Run specific perturbations
python run_hypothesis_batch.py \
    --perturbations CD58 HLA-B IFNGR1 JAK2 \
    --output-dir results

# Run all perturbations
python run_hypothesis_batch.py \
    --output-dir results
```

### Slurm Configuration

The provided Slurm scripts are configured for:
- **GPU**: 1x H100 (or available GPU)
- **Memory**: 128GB for batch processing, 64GB for single test
- **CPUs**: 16 cores for batch, 8 for single test
- **Time**: 48 hours for batch, 24 hours for single test

Edit the `#SBATCH` directives in the `.sh` files to match your cluster configuration:
- `--partition`: Change to your GPU partition name
- `--gres=gpu:1`: Adjust GPU type/count if needed
- `--mem`: Adjust memory based on dataset size

### Output

Results are saved in both JSON and CSV formats:
- `/home/nebius/cellian/results/hypothesis_results_YYYYMMDD_HHMMSS.json`
- `/home/nebius/cellian/results/hypothesis_results_YYYYMMDD_HHMMSS.csv`

Each result includes:
- Perturbation name
- RNA score (cosine similarity)
- Protein score (cosine similarity)
- Error propagation (RNA - Protein score)
- Any errors encountered
