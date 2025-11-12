#!/bin/bash
#SBATCH --job-name=hypothesis_batch
#SBATCH --output=/home/nebius/cellian/logs/hypothesis_batch_%j.out
#SBATCH --error=/home/nebius/cellian/logs/hypothesis_batch_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Load modules (adjust based on your cluster setup)
# module load cuda/12.1
# module load python/3.10

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory
cd /home/nebius/cellian

# Print GPU info
nvidia-smi
echo ""

# Run the batch hypothesis engine
# Options:
#   --state-model-path: Path to STATE model (optional, uses default if not specified)
#   --output-dir: Where to save results
#   --perturbations: Specific perturbations to test (optional, runs all if not specified)
#   --max-perturbations: Limit number of perturbations (optional, useful for testing)

echo "Starting batch hypothesis engine..."
python run_hypothesis_batch.py \
    --output-dir /home/nebius/cellian/results \
    --max-perturbations 50

# Example: Run specific perturbations
# python run_hypothesis_batch.py \
#     --output-dir /home/nebius/cellian/results \
#     --perturbations CD58 HLA-B IFNGR1 JAK2

# Example: Run all perturbations with custom STATE model
# python run_hypothesis_batch.py \
#     --state-model-path /path/to/SE-600M \
#     --output-dir /home/nebius/cellian/results

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
