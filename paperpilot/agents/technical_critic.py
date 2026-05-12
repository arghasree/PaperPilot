"""
Technical Critic — assess factual correctness and citation needs for each claim
in a LaTeX paragraph.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from utils.llm import chat

_SYSTEM = """\
You are a rigorous technical reviewer for ML research papers.
You will be given a paragraph (which may contain LaTeX markup) and the paper's topic.

Your job:
1. Identify every distinct factual claim in the paragraph.
2. For each claim output:
   - "claim": the claim as a short, plain-English phrase (strip LaTeX).
   - "verdict": one of "correct", "incorrect", or "uncertain".
   - "correction": if verdict is "incorrect", state the accurate fact; otherwise "".
   - "citation_needed": true if the claim requires a literature citation, false otherwise.
   - "suggested_search_query": a concise search query to find a supporting paper if
     citation_needed is true; otherwise "".

Rules:
- Report only genuine factual claims, not definitions or transitions.
- "uncertain" means the claim may be true but you are not confident — flag it.
- Correctness is judged relative to well-established ML knowledge up to your training cutoff.
- Respond ONLY with a JSON object of this exact shape, no prose before or after:

{
  "assessments": [
    {
      "claim": "...",
      "verdict": "correct|incorrect|uncertain",
      "correction": "...",
      "citation_needed": true,
      "suggested_search_query": "..."
    }
  ]
}
"""

_USER_TMPL = """\
Topic: {topic}

Paragraph:
{paragraph}
"""


@dataclass
class ClaimAssessment:
    claim: str
    verdict: str            # "correct" | "incorrect" | "uncertain"
    correction: str         # non-empty only when verdict == "incorrect"
    citation_needed: bool
    suggested_search_query: str


@dataclass
class TechnicalCritique:
    assessments: list[ClaimAssessment]

    def needs_correction(self) -> list[ClaimAssessment]:
        return [a for a in self.assessments if a.verdict != "correct"]

    def needs_citation(self) -> list[ClaimAssessment]:
        return [a for a in self.assessments if a.citation_needed]

    def format_for_prompt(self) -> str:
        """Human-readable summary suitable for passing to the Rewriter."""
        if not self.assessments:
            return "No factual issues found."
        lines = []
        for a in self.assessments:
            status = f"[{a.verdict.upper()}]"
            line = f"{status} {a.claim}"
            if a.correction:
                line += f"\n  → Correction: {a.correction}"
            if a.citation_needed:
                line += f"\n  → Needs citation (query: \"{a.suggested_search_query}\")"
            lines.append(line)
        return "\n".join(lines)


def _clean_json(text: str) -> str:
    text = re.sub(r'"\s*\([^)]*\)', '"', text)
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    text = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"',
                  lambda m: '"' + m.group(1).replace('\n', ' ') + '"',
                  text)
    return text


def _extract_json(text: str) -> dict:
    for candidate in (text, _clean_json(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        m = re.search(r'\{.*\}', candidate, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not parse JSON from model response:\n{text}")


def run(paragraph: str, topic: str) -> TechnicalCritique:
    """
    Assess every factual claim in *paragraph* given the paper *topic*.
    Returns a TechnicalCritique with one ClaimAssessment per claim.
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": _USER_TMPL.format(topic=topic, paragraph=paragraph)},
    ]
    raw = chat("technical_critic", messages, temperature=0.1)
    data = _extract_json(raw)

    assessments = [
        ClaimAssessment(
            claim=item.get("claim", ""),
            verdict=item.get("verdict", "uncertain"),
            correction=item.get("correction", ""),
            citation_needed=bool(item.get("citation_needed", False)),
            suggested_search_query=item.get("suggested_search_query", ""),
        )
        for item in data.get("assessments", [])
    ]
    return TechnicalCritique(assessments=assessments)