#!/bin/bash
#SBATCH --job-name=hypothesis_engine
#SBATCH --output=/home/nebius/cellian/logs/hypothesis_%j.out
#SBATCH --error=/home/nebius/cellian/logs/hypothesis_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
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

# Navigate to project directory
cd /home/nebius/cellian

# Print GPU info
nvidia-smi
echo ""

# Run the hypothesis engine
echo "Starting hypothesis engine..."
python hypothesis_engine.py

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
