#!/bin/bash
#SBATCH --job-name=perturbation_orchestrator
#SBATCH --output=llm/logs/orchestrator_%j.out
#SBATCH --error=llm/logs/orchestrator_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Load environment
source ~/.bashrc

# Navigate to project directory
cd /home/nebius/cellian || {
    echo "Error: Failed to change directory to /home/nebius/cellian"
    exit 1
}

# Create logs directory if it doesn't exist
mkdir -p llm/logs llm/perturbation_outputs

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

# Get query from command line argument
if [ -z "$1" ]; then
    echo "Error: Query argument is required"
    echo ""
    echo "Usage: sbatch llm/run_orchestrator_slurm.sh \"your query here\""
    echo ""
    echo "Examples:"
    echo "  sbatch llm/run_orchestrator_slurm.sh \"run KO TP53\""
    echo "  sbatch llm/run_orchestrator_slurm.sh \"imatinib\""
    echo "  sbatch llm/run_orchestrator_slurm.sh \"TP53 vs imatinib\""
    echo "  sbatch llm/run_orchestrator_slurm.sh \"which perturbation is stronger KO of TP53 or imatinib\""
    echo ""
    exit 1
fi

QUERY="$1"
echo "User Query: $QUERY"
echo ""

# Run the orchestrator
echo "Running perturbation orchestrator..."
echo ""

# Export query as environment variable to handle special characters properly
export SBATCH_QUERY="$QUERY"

# Run the orchestrator script
python -c "
import sys
import json
import os
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from llm.perturbation_orchestrator import process_user_query

# Process the query (passed via environment variable to handle special characters)
query = os.environ.get('SBATCH_QUERY')
if not query:
    print('Error: Query not provided')
    sys.exit(1)
print(f'Processing query: {query}')
print('')

results = process_user_query(query)

# Save results summary
output_dir = Path('llm/perturbation_outputs')
output_dir.mkdir(parents=True, exist_ok=True)

# Save JSON summary
summary_path = output_dir / 'orchestrator_results.json'
with open(summary_path, 'w') as f:
    json.dump({
        'query': query,
        'intent': results.get('intent', {}),
        'perturbations': [
            {
                'type': p.get('type'),
                'match_info': p.get('match_info', {}),
                'has_results': 'results' in p,
                'has_outputs': 'outputs' in p,
                'error': p.get('error')
            }
            for p in results.get('perturbations', [])
        ],
        'has_comparison': 'comparison' in results and results['comparison'] is not None
    }, f, indent=2)

print(f'Results summary saved to: {summary_path}')
print('')

# Print summary
print('='*70)
print('ORCHESTRATOR RESULTS SUMMARY')
print('='*70)
print(f'Query: {query}')
print(f'Intent mode: {results.get(\"intent\", {}).get(\"mode\", \"unknown\")}')
print(f'Number of perturbations: {len(results.get(\"perturbations\", []))}')
print('')

for i, pert in enumerate(results.get('perturbations', []), 1):
    print(f'Perturbation {i}:')
    print(f'  Type: {pert.get(\"type\")}')
    match_info = pert.get('match_info', {})
    print(f'  Requested: {match_info.get(\"requested_name\")}')
    print(f'  Used: {match_info.get(\"used_name\")}')
    print(f'  Match type: {match_info.get(\"match_type\")}')
    if 'error' in pert:
        print(f'  Error: {pert[\"error\"]}')
    elif 'outputs' in pert:
        outputs = pert['outputs']
        print(f'  Outputs generated:')
        for key, value in outputs.items():
            if value and key != 'hypotheses':
                print(f'    {key}: {value}')
    print('')

if 'comparison' in results and results['comparison']:
    comp = results['comparison']
    if 'error' not in comp:
        print('Comparison generated successfully')
        print(f'  Shared pathways: {len(comp.get(\"shared_pathways\", []))}')
        print(f'  Shared phenotypes: {len(comp.get(\"shared_phenotypes\", []))}')
    else:
        print(f'Comparison error: {comp[\"error\"]}')
    print('')

print('='*70)
print('Job completed successfully')
print('='*70)
"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
    echo "Results saved to: llm/perturbation_outputs/"
    echo ""
    echo "Generated directories:"
    ls -ld llm/perturbation_outputs/*/ 2>/dev/null | head -10 || echo "  (check output directory)"
else
    echo "Job failed with exit code: $EXIT_CODE"
    echo "Check error logs for details: llm/logs/orchestrator_${SLURM_JOB_ID}.err"
fi
echo "=========================================="

exit $EXIT_CODE

