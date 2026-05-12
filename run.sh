#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=30G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

module load python/3.10
module load postgresql  

source /scratch/$USER/agentic/bin/activate

echo "HTTP_PROXY=$HTTP_PROXY HTTPS_PROXY=$HTTPS_PROXY http_proxy=$http_proxy https_proxy=$https_proxy"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# Models stored in scratch to avoid filling home quota
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_MODELS=/scratch/$USER/Agentic/.ollama/models
/scratch/$USER/ollama_extracted/bin/ollama serve &
OLLAMA_PID=$!

# Wait until the API is ready (timeout after 60s so the job doesn't hang forever)
WAIT=0
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
    WAIT=$((WAIT + 2))
    if [ $WAIT -ge 60 ]; then
        echo "ERROR: Ollama failed to start. Check that ollama_extracted is installed under /scratch/$USER."
        exit 1
    fi
done
echo "Ollama ready"

python paperpilot/main.py \
--input /scratch/$USER/Agentic/para.tex \
--output-dir /scratch/$USER/Agentic/output \
--section "${PAPERPILOT_SECTION:-method}" \
--topic "${PAPERPILOT_TOPIC:-Embodied co-design}"

kill $OLLAMA_PID