"""
PaperPilot web UI.

A thin FastAPI wrapper around run_pipeline.sh:
  - paste a LaTeX paragraph
  - it writes para.tex, runs ./run_pipeline.sh (which submits SLURM jobs)
  - streams the script log
  - shows output.tex, output.bib, and a rendered preview

Start with:
  cd /scratch/$USER/Agentic && python ui/app.py
Then port-forward and open http://localhost:5050 in your browser.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

PROJECT_DIR = Path(f"/scratch/{os.environ['USER']}/Agentic")
PARA_TEX = PROJECT_DIR / "para.tex"
OUTPUT_DIR = PROJECT_DIR / "output"
SCRIPT = PROJECT_DIR / "run_pipeline.sh"
UI_DIR = Path(__file__).resolve().parent

app = FastAPI(title="PaperPilot UI")
jobs: dict[str, dict] = {}


class RunRequest(BaseModel):
    latex: str
    section: str = "method"
    topic: str = "Embodied co-design"


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (UI_DIR / "index.html").read_text()


async def _run_pipeline(job_id: str, latex: str, section: str, topic: str) -> None:
    PARA_TEX.write_text(latex)

    jobs[job_id]["status"] = "running"

    env = os.environ.copy()
    env["PAPERPILOT_SECTION"] = section
    env["PAPERPILOT_TOPIC"] = topic

    proc = await asyncio.create_subprocess_exec(
        "bash", str(SCRIPT),
        cwd=str(PROJECT_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    jobs[job_id]["pid"] = proc.pid

    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        jobs[job_id]["log"] += line.decode("utf-8", errors="replace")

    rc = await proc.wait()
    jobs[job_id]["returncode"] = rc

    out_tex = OUTPUT_DIR / "output.tex"
    out_bib = OUTPUT_DIR / "output.bib"
    jobs[job_id]["output_tex"] = out_tex.read_text() if out_tex.exists() else ""
    jobs[job_id]["output_bib"] = out_bib.read_text() if out_bib.exists() else ""
    jobs[job_id]["status"] = "done" if rc == 0 else "failed"


@app.post("/run")
async def run(req: RunRequest) -> dict:
    if not req.latex.strip():
        raise HTTPException(400, "empty latex input")
    if not SCRIPT.exists():
        raise HTTPException(500, f"missing script: {SCRIPT}")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "starting",
        "log": "",
        "output_tex": "",
        "output_bib": "",
        "returncode": None,
    }
    asyncio.create_task(_run_pipeline(job_id, req.latex, req.section, req.topic))
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str) -> dict:
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    return jobs[job_id]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)

