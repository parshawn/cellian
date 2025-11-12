#!/bin/bash
#SBATCH --job-name=convert_parquet
#SBATCH --output=/home/nebius/cellian/logs/convert_%j.out
#SBATCH --error=/home/nebius/cellian/logs/convert_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

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

# Show memory info
echo "Available memory:"
free -h
echo ""

# Run the conversion script
echo "Starting CSV to Parquet conversion..."
python convert_to_parquet.py

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
