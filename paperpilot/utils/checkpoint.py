"""
Checkpoint helpers for pipeline.py.

Saves state after the rewriter so a resubmitted job can skip the three
expensive LLM agents (technical critic, writing critic, rewriter) and
jump straight to citation grounder.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from agents.technical_critic import ClaimAssessment, TechnicalCritique
from agents.writing_critic import WritingAnnotation, WritingCritique

_STAGE = "post_rewriter"


def save(
    path: str | Path,
    preamble: str,
    postamble: str,
    paragraphs: list[dict],
) -> None:
    """
    Persist phase-1 results (technical critic + writing critic + rewriter).

    Each entry in *paragraphs* must have keys:
        original, rewritten, technical_critique (TechnicalCritique), writing_critique (WritingCritique)
    """
    data = {
        "stage": _STAGE,
        "preamble": preamble,
        "postamble": postamble,
        "paragraphs": [
            {
                "original": p["original"],
                "rewritten": p["rewritten"],
                "technical_critique": {
                    "assessments": [asdict(a) for a in p["technical_critique"].assessments]
                },
                "writing_critique": {
                    "annotations": [asdict(a) for a in p["writing_critique"].annotations]
                },
            }
            for p in paragraphs
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load(path: str | Path) -> tuple[str, str, list[dict]] | None:
    """
    Load a checkpoint written by save().

    Returns (preamble, postamble, paragraphs) where each paragraph dict has
    keys: original, rewritten, technical_critique, writing_critique.
    Returns None if no checkpoint exists or it has an unexpected stage.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("stage") != _STAGE:
        return None

    paragraphs = []
    for para in data["paragraphs"]:
        tech = TechnicalCritique(
            assessments=[ClaimAssessment(**a) for a in para["technical_critique"]["assessments"]]
        )
        writing = WritingCritique(
            annotations=[WritingAnnotation(**a) for a in para["writing_critique"]["annotations"]]
        )
        paragraphs.append({
            "original": para["original"],
            "rewritten": para["rewritten"],
            "technical_critique": tech,
            "writing_critique": writing,
        })

    return data["preamble"], data["postamble"], paragraphs
