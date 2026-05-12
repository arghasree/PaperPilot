#!/bin/bash
# Submit run.sh, prefetch any missing references on the login node, then resubmit.
# Usage: ./run_pipeline.sh

set -e # exit immediately if non-zero exit code
cd "$(dirname "$0")"

echo "[1/3] Submitting compute job..."
JOB1=$(sbatch --parsable --wait run.sh)
LOG="slurm-${JOB1}.out"
echo "  Job $JOB1 finished. Log: $LOG"

echo "[2/3] Extracting uncached queries..."
grep "Network unavailable and query not cached" "$LOG" 2>/dev/null \
    | sed -E "s/.*not cached: '(.*)'.*/\1/" | sort -u > queries.txt
N=$(wc -l < queries.txt)

if [ "$N" -eq 0 ]; then
    echo "  No missing queries — cache was already complete."
    exit 0
fi
echo "  Found $N queries needing prefetch."

echo "[3/3] Prefetching on login node..."
python paperpilot/prefetch.py queries.txt

echo "Resubmitting compute job with populated cache..."
JOB2=$(sbatch --parsable --wait run.sh)
echo "Done. Final log: slurm-${JOB2}.out"