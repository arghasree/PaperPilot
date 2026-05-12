"""
Citation Grounder — for each claim that needs a citation, search Semantic Scholar,
select the best match, fetch real BibTeX, and inject \\cite{key} into the paragraph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from agents.technical_critic import ClaimAssessment
from utils.bibtex_io import append_entries
from utils.llm import chat
from utils.paper_search import PaperResult, fetch_bibtex, search

# ── prompts ──────────────────────────────────────────────────────────────────

_SELECT_SYSTEM = """\
You are helping select the best paper to cite for a specific claim in an ML paper.
You will be given the claim and a numbered list of candidate papers.
Reply with ONLY a single integer (the index of the best match) or the word "none" \
if no candidate adequately supports the claim.
No explanation. No prose.
"""

_SELECT_USER = """\
Claim: {claim}

Candidates:
{candidates}
"""

_INJECT_SYSTEM = """\
You are editing a LaTeX paragraph to add citation keys.
You will be given the paragraph and a list of (claim, cite_key) pairs.

Rules:
  - If \\cite{{existing_keys}} already appears immediately after the clause that makes \
the claim, append the new key: \\cite{{existing_keys,new_key}}.
  - If \\cite{{TODO}} appears near a claim, replace it with \\cite{{key}}.
  - Otherwise insert \\cite{{key}} immediately after the clause that makes the claim.
  - Do NOT change any other text — no rewording, no reformatting.
  - Return ONLY the updated paragraph, nothing else.
"""

_INJECT_USER = """\
Paragraph:
{paragraph}

Citations to inject:
{pairs}
"""


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class GroundedCitation:
    claim: str
    cite_key: str
    bibtex: str
    paper: PaperResult


@dataclass
class CitationGrounderResult:
    paragraph: str                          # updated with \\cite{key} injected
    grounded: list[GroundedCitation] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)   # claims with no good match


# ── internal helpers ──────────────────────────────────────────────────────────

def _format_candidates(papers: list[PaperResult]) -> str:
    lines = []
    for i, p in enumerate(papers):
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        lines.append(
            f"{i}. \"{p.title}\" — {authors} ({p.year}, {p.venue or 'n/a'}, "
            f"{p.citation_count} citations)"
        )
    return "\n".join(lines)


def _select_best(claim: str, papers: list[PaperResult]) -> PaperResult | None:
    """Ask the LLM to pick the best paper for the claim. Returns None if no good match."""
    messages = [
        {"role": "system", "content": _SELECT_SYSTEM},
        {"role": "user", "content": _SELECT_USER.format(
            claim=claim,
            candidates=_format_candidates(papers),
        )},
    ]
    raw = chat("citation_grounder", messages, temperature=0.0).strip().lower()
    if raw == "none":
        return None
    try:
        idx = int(raw)
        if 0 <= idx < len(papers):
            return papers[idx]
    except ValueError:
        pass
    return None


def _merge_adjacent_cites(text: str) -> str:
    """Collapse consecutive \\cite{a}\\cite{b} (with optional whitespace) into \\cite{a,b}."""
    pattern = re.compile(r'\\cite\{([^}]+)\}\s*\\cite\{([^}]+)\}')
    while pattern.search(text):
        text = pattern.sub(lambda m: r'\cite{' + m.group(1) + ',' + m.group(2) + '}', text)
    return text


def _inject_citations(paragraph: str, grounded: list[GroundedCitation]) -> str:
    """Ask the LLM to insert all \\cite{key} commands into the paragraph."""
    pairs = "\n".join(
        f'- Claim: "{g.claim}" → \\cite{{{g.cite_key}}}'
        for g in grounded
    )
    messages = [
        {"role": "system", "content": _INJECT_SYSTEM},
        {"role": "user", "content": _INJECT_USER.format(
            paragraph=paragraph,
            pairs=pairs,
        )},
    ]
    result = chat("citation_grounder", messages, temperature=0.0).strip()
    return _merge_adjacent_cites(result)


# ── public API ────────────────────────────────────────────────────────────────

def run(
    paragraph: str,
    claims: list[ClaimAssessment],
    bib_path: str | Path,
) -> CitationGrounderResult:
    """
    For each claim in *claims* (pre-filtered to citation_needed=True):
      1. Search Semantic Scholar using the suggested query.
      2. LLM selects the best match from the top results.
      3. Fetch real BibTeX via DOI (or construct from metadata).
      4. Append new entries to *bib_path* (deduplicates by key).
      5. Inject \\cite{key} into the paragraph via LLM.

    Returns CitationGrounderResult with the updated paragraph and grounded citations.
    """
    citation_claims = [c for c in claims if c.citation_needed]
    if not citation_claims:
        return CitationGrounderResult(paragraph=paragraph)

    grounded: list[GroundedCitation] = []
    not_found: list[str] = []

    for claim in citation_claims:
        query = claim.suggested_search_query or claim.claim
        try:
            results = search(query, limit=5)
        except RuntimeError:
            not_found.append(claim.claim)
            continue

        if not results:
            not_found.append(claim.claim)
            continue

        paper = _select_best(claim.claim, results)
        if paper is None:
            not_found.append(claim.claim)
            continue

        bibtex = fetch_bibtex(paper)
        grounded.append(GroundedCitation(
            claim=claim.claim,
            cite_key=paper.cite_key(),
            bibtex=bibtex,
            paper=paper,
        ))

    if not grounded:
        return CitationGrounderResult(paragraph=paragraph, not_found=not_found)

    append_entries(bib_path, [g.bibtex for g in grounded])
    updated = _inject_citations(paragraph, grounded)

    return CitationGrounderResult(
        paragraph=updated,
        grounded=grounded,
        not_found=not_found,
    )
