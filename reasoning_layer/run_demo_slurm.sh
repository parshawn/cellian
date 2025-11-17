#!/bin/bash
#SBATCH --job-name=test_reasoning_demo
#SBATCH --output=logs/test_demo_%j.out
#SBATCH --error=logs/test_demo_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G 

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Load environment
source ~/.bashrc  # Adjust if needed

# Load modules if needed (uncomment and adjust based on your cluster setup)
# module load cuda/12.1
# module load python/3.10

# Navigate to reasoning_layer directory
cd /home/nebius/cellian/reasoning_layer || {
    echo "Error: Failed to change directory to /home/nebius/cellian/reasoning_layer"
    exit 1
}

# Create logs and results directories if they don't exist
mkdir -p logs results

# Activate conda environment
# Initialize conda (try common locations)
if [ -f "/home/nebius/miniconda/etc/profile.d/conda.sh" ]; then
    source /home/nebius/miniconda/etc/profile.d/conda.sh
elif [ -f "/home/nebius/miniconda3/etc/profile.d/conda.sh" ]; then
    source /home/nebius/miniconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi

# Activate new_env environment
if [ -d "/home/nebius/miniconda/envs/new_env" ] || [ -d "$HOME/miniconda/envs/new_env" ]; then
    conda activate new_env
    echo "Activated conda environment: new_env"
    echo "Python path: $(which python)"
else
    echo "Warning: new_env conda environment not found, using system Python"
fi
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not available"
    exit 1
fi

# Print Python version
echo "Python version: $(python --version)"
echo ""

# Print GPU info if GPU is requested
if [ -n "$CUDA_VISIBLE_DEVICES" ] || command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || echo "GPU info not available"
    echo ""
fi

# Run test mode with default settings
echo "Running reasoning engine in test mode..."
echo ""

# Run the script and capture exit code
python run_demo.py --test \
    --perturbation JAK1_KO \
    --genes "STAT1,IRF1,CXCL10,HLA-A,B2M" \
    --proteins "HLA-A,CD58" \
    --seed 42 \
    --dummy-graph \
    --output-file "results/test_demo_${SLURM_JOB_ID}.json" 

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
    echo "Results saved to: results/test_demo_${SLURM_JOB_ID}.json"
else
    echo "Job failed with exit code: $EXIT_CODE"
    echo "Check error logs for details"
fi
echo "=========================================="

exit $EXIT_CODE

