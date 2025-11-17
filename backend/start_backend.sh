#!/bin/bash
# Start Cellian Backend Server

echo "=========================================="
echo "Starting Cellian Backend Server"
echo "=========================================="

# Activate conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate new_env

# Check if conda environment is activated
if [ "$CONDA_DEFAULT_ENV" != "new_env" ]; then
    echo "ERROR: Conda environment 'new_env' is not activated!"
    echo "Please run: conda activate new_env"
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"

# Navigate to backend directory
cd "$(dirname "$0")"

# Check if required directories exist
if [ ! -d "Agent_Tools" ]; then
    echo "ERROR: Agent_Tools directory not found!"
    exit 1
fi

if [ ! -d "llm" ]; then
    echo "ERROR: llm directory not found!"
    exit 1
fi

echo "✓ Agent_Tools directory found"
echo "✓ LLM directory found"

# Check if port 8000 is already in use
if lsof -i :8000 >/dev/null 2>&1; then
    echo ""
    echo "⚠️  WARNING: Port 8000 is already in use!"
    echo "   Attempting to find and kill the process..."
    OLD_PID=$(lsof -ti :8000)
    if [ ! -z "$OLD_PID" ]; then
        echo "   Found process PID: $OLD_PID"
        kill $OLD_PID 2>/dev/null
        sleep 2
        if lsof -i :8000 >/dev/null 2>&1; then
            echo "   ⚠️  Could not kill process. Please kill it manually:"
            echo "      kill $OLD_PID"
            echo "   Or use a different port by editing backend/api.py"
            exit 1
        else
            echo "   ✓ Killed old process"
        fi
    fi
fi

# Start FastAPI server
echo ""
echo "Starting FastAPI server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

python api.py

