#!/bin/bash
#SBATCH --job-name=cell_embed
#SBATCH --output=/home/nebius/cellian/logs/cell_embed_%j.out
#SBATCH --error=/home/nebius/cellian/logs/cell_embed_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00

conda run -n hackathon python generate_cell_embeddings.py
