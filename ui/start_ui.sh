#!/bin/bash
# Start the PaperPilot UI on the login node.
# From your local machine, port-forward with:
#   ssh -L 5050:localhost:5050 USER@cedar.alliancecan.ca
# Then open http://localhost:5050

set -e
cd "$(dirname "$0")/.."

if [ -f /scratch/$USER/agentic/bin/activate ]; then
    source /scratch/$USER/agentic/bin/activate
fi

python -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "Installing fastapi + uvicorn..."
    pip install --quiet fastapi uvicorn
}

echo "UI listening on http://localhost:5050"
exec python ui/app.py
