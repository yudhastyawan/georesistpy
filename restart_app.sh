#!/usr/bin/env bash

# Stop any existing georesistpy process
echo "🛑 Stopping existing application (if any)..."
lsof -ti:5006 | xargs kill -9 2>/dev/null

# Activate virtual environment
source .venv/bin/activate

# Start the application in the background
echo "🚀 Starting GeoResistPy..."
georesistpy --no-show &

# Wait a moment for the server to start
sleep 3

echo "✅ Application restarted and running!"
echo "➡️  Open your browser to: http://127.0.0.1:5006"
