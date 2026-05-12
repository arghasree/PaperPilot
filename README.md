# PaperPilot

A multi-agent pipeline that refines a LaTeX paragraph: technical critic →
writing critic → rewriter → citation grounder (Semantic Scholar / ArXiv /
OpenAlex) → reviewer. A web UI wraps the whole flow — paste a paragraph
into your browser, get back a polished `.tex` + `.bib` and a rendered
preview.

The pipeline runs on Compute Canada (Alliance Canada). LLM inference is
local via Ollama on a GPU compute node; the UI server runs on the login
node and submits SLURM jobs.

---

## 1. Prerequisites

- A Compute Canada / Alliance Canada account with GPU allocation.
- SSH access to the cluster (this README uses `vulcan` — substitute your
  cluster, e.g. `cedar`, `narval`).
- Git on your local machine.

---

## 2. One-time setup on the cluster

Replace `USER` with your Compute Canada username below.

### 2.1 SSH in and clone

```bash
ssh USER@vulcan.alliancecan.ca
cd /scratch/USER
git clone <REPO_URL> Agentic
cd Agentic
```

### 2.2 Create the Python environment

```bash
module load python/3.10 postgresql
python -m venv /scratch/USER/agentic
source /scratch/USER/agentic/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn        # for the UI
```

### 2.3 Install Ollama (local LLM server)

Compute nodes have no internet, so Ollama and its models must be staged in
`/scratch` ahead of time.

```bash
cd /scratch/USER
mkdir ollama_extracted && cd ollama_extracted

# Download a Linux build of Ollama and unpack into this directory
# (see https://github.com/ollama/ollama/releases for the latest tarball)
curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz \
  | tar -xz
```

After this, `/scratch/USER/ollama_extracted/bin/ollama` should be
executable.

### 2.4 Pull the LLM models

The pipeline uses two models (see `paperpilot/config.py`):

| Agent                          | Model           |
| ------------------------------ | --------------- |
| technical_critic, reviewer     | `llama3.3:70b`  |
| writing_critic, rewriter, citation_grounder | `llama3.1:8b` |

Pull them once on the login node so they end up cached in `/scratch`:

```bash
export OLLAMA_MODELS=/scratch/USER/Agentic/.ollama/models
/scratch/USER/ollama_extracted/bin/ollama serve &
OLLAMA_PID=$!
sleep 5
/scratch/USER/ollama_extracted/bin/ollama pull llama3.1:8b
/scratch/USER/ollama_extracted/bin/ollama pull llama3.3:70b
kill $OLLAMA_PID
```

---

## 3. Running the UI

### 3.1 From your local machine — open an SSH tunnel

```bash
ssh -L 5050:localhost:5050 USER@vulcan.alliancecan.ca
```

The `-L 5050:localhost:5050` flag forwards `localhost:5050` on your laptop
to port 5050 on the login node, so your browser can reach the UI server.

### 3.2 On the login node — start the UI

```bash
cd /scratch/USER/Agentic
bash ui/start_ui.sh
```

Output should end with:

```
UI listening on http://localhost:5050
INFO:     Uvicorn running on http://0.0.0.0:5050
```

### 3.3 In your local browser

Open <http://localhost:5050>. You should see the PaperPilot UI.

---

## 4. Usage

1. Paste a LaTeX paragraph into the **Input** textarea. The pipeline auto-
   triggers ~400 ms after the paste event (or click **Run pipeline**).
2. Set **Section** and **Topic** before pasting — they propagate into the
   agent prompts.
3. The **Pipeline log** pane streams the script's stdout while the SLURM
   job runs. Expect two SLURM jobs in sequence on the first run: an
   initial job, then a prefetch step on the login node for any uncached
   citation queries, then a resubmit.
4. When the run finishes, **output.tex** and **output.bib** appear side-
   by-side, and the **Rendered preview** shows the paragraph with `\cite`
   commands converted to numbered superscripts plus a generated
   References list.

---

## 5. Running without the UI

The UI is just a wrapper around `./run_pipeline.sh`. To run the pipeline
directly:

```bash
# 1. Put your input in para.tex
$EDITOR /scratch/USER/Agentic/para.tex

# 2. Submit the compute job (uses the section/topic env vars below)
cd /scratch/USER/Agentic
PAPERPILOT_SECTION=method PAPERPILOT_TOPIC="your topic" ./run_pipeline.sh
```

Outputs land in `output/output.tex` and `output/output.bib`.

---

## 6. Notes & troubleshooting

- **First run is slow.** Compute nodes have no internet, so citation
  searches miss the cache; `run_pipeline.sh` scrapes the misses from the
  log, prefetches them on the login node (which has internet), and
  resubmits.
- **Checkpoint.** After Phase 1 (critic + rewriter) finishes, the pipeline
  writes `output/checkpoint.json`. On a resubmit it loads this and skips
  straight to citation grounding + reviewer. The checkpoint is deleted
  only when every citation query is satisfied; delete it manually if you
  want to force a full re-run.
- **Pre-existing `\cite{key}` not in output.bib.** These come from your
  own bibliography file. Pass `--bib path/to/your.bib` to
  `paperpilot/main.py` (edit `run.sh`) so the pipeline copies it into
  `output.bib` at the start — otherwise the rendered preview will show
  "missing entry" for those keys.
- **UI starts but pipeline never runs.** Check that `run_pipeline.sh` is
  executable (`chmod +x run_pipeline.sh`) and that the venv activated by
  `ui/start_ui.sh` has `uvicorn` installed.
