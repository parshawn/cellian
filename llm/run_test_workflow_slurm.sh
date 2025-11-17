#!/bin/bash
#SBATCH --job-name=llm_test_workflow
#SBATCH --output=llm/logs/test_workflow_%j.out
#SBATCH --error=llm/logs/test_workflow_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

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

# Navigate to project directory
cd /home/nebius/cellian || {
    echo "Error: Failed to change directory to /home/nebius/cellian"
    exit 1
}

# Create logs and output directories if they don't exist
mkdir -p llm/logs llm/test_output

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

# Activate conda environment if available
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

# Print memory info
echo "Memory Info:"
free -h
echo ""

# Run the test workflow
echo "Running LLM test workflow..."
echo ""

# Run the script and capture exit code
python llm/test_workflow.py

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
    echo "Results saved to: llm/test_output/"
    echo ""
    echo "Generated files:"
    ls -lh llm/test_output/ 2>/dev/null || echo "  (check output directory)"
else
    echo "Job failed with exit code: $EXIT_CODE"
    echo "Check error logs for details: llm/logs/test_workflow_${SLURM_JOB_ID}.err"
fi
echo "=========================================="

exit $EXIT_CODE

