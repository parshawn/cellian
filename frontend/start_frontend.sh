#!/bin/bash
# Start Cellian Frontend Development Server

echo "=========================================="
echo "Starting Cellian Frontend"
echo "=========================================="

# Navigate to frontend directory
cd "$(dirname "$0")"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start Vite dev server
echo ""
echo "Starting Vite dev server"
echo "Frontend will be available at http://localhost:5173"
echo "Press Ctrl+C to stop"
echo ""

npm run dev

