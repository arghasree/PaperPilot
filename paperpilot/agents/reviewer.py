"""
Reviewer — holistic final check on the refined .tex for technical correctness,
writing quality, citation placement, and narrative flow.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from utils.bibtex_io import read_bib
from utils.latex_io import read_tex
from utils.llm import chat

_SYSTEM = """\
You are a senior ML researcher doing a final review of a refined LaTeX section.
You will be given the full section text, the list of available BibTeX keys, \
the section type, and the paper topic.

Check for:
  - "technical"  — remaining factual errors or unsupported claims
  - "writing"    — grammar, style, or clarity issues that were not fixed
  - "citation"   — missing citations, \\cite{TODO} placeholders left in, or
                   citations placed at the wrong location
  - "flow"       — abrupt transitions, repetition, or incoherence across paragraphs

For each issue output:
  - "category":    one of the four above
  - "location":    the exact phrase or sentence containing the issue
  - "description": what is wrong
  - "suggestion":  how to fix it

Then give an overall verdict:
  - "approved"       — section is ready; no blocking issues
  - "needs_revision" — one or more issues must be addressed before submission

Finally, output the complete revised LaTeX text with ALL issues fixed applied directly.
Preserve all LaTeX commands and structure. If verdict is "approved", copy the input text unchanged.

CITATION RULES — read carefully. These are the ONLY allowed actions on \\cite{...}:

PRESERVE (immutable):
  - Every key in "Pre-existing keys" MUST appear in revised_text in its original
    position. These are valid even if absent from "Available BibTeX keys".
  - NEVER drop, substitute, or move a pre-existing key.

FIX (apply in revised_text, not just in "issues"):
  - \\cite{TODO...} placeholders: replace with a key from "Available BibTeX keys"
    that fits the claim, or REMOVE the entire \\cite{...} command if no key fits.
  - Citation keys that are NEITHER in "Pre-existing keys" NOR in "Available BibTeX
    keys" are hallucinated — REMOVE them or replace with a valid available key.

NEVER:
  - Invent new \\cite{...} keys that are not in "Available BibTeX keys".
  - Add new citations for claims that don't already have one.

OUTPUT CONTRACT — every issue you flag MUST be fixed in revised_text:
  - If you list a "citation" issue, the corresponding fix MUST appear in revised_text.
  - revised_text is the FINAL document, NOT a description of what to change.
  - A reviewer that reports issues without applying them in revised_text has failed.

Respond ONLY with a JSON object of this exact shape:
{
  "verdict": "approved|needs_revision",
  "summary": "one short paragraph overall assessment",
  "issues": [
    {
      "category": "...",
      "location": "...",
      "description": "...",
      "suggestion": "..."
    }
  ],
  "revised_text": "the complete revised LaTeX text"
}
"""

_USER_TMPL = """\
Topic: {topic}
Section type: {section_type}
Available BibTeX keys (newly fetched — do NOT replace pre-existing keys with these): {bib_keys}
Pre-existing citation keys (valid, from external .bib — preserve exactly as-is): {existing_keys}

Section text:
{text}
"""


@dataclass
class ReviewIssue:
    category: str    # technical | writing | citation | flow
    location: str
    description: str
    suggestion: str


@dataclass
class ReviewReport:
    verdict: str                              # "approved" | "needs_revision"
    summary: str
    revised_text: str = ""
    issues: list[ReviewIssue] = field(default_factory=list)

    def by_category(self, category: str) -> list[ReviewIssue]:
        return [i for i in self.issues if i.category == category]

    def approved(self) -> bool:
        return self.verdict == "approved"

    def format(self) -> str:
        lines = [
            f"Verdict : {self.verdict.upper()}",
            f"Summary : {self.summary}",
        ]
        if self.issues:
            lines.append(f"\nIssues ({len(self.issues)}):")
            for iss in self.issues:
                lines.append(f"\n  [{iss.category.upper()}] \"{iss.location}\"")
                lines.append(f"    Problem    : {iss.description}")
                lines.append(f"    Suggestion : {iss.suggestion}")
        else:
            lines.append("\nNo issues found.")
        return "\n".join(lines)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^```[a-z]*\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


def _clean_json(text: str) -> str:
    text = _strip_fences(text)
    # Character-level pass: fix unescaped backslashes and literal newlines
    # inside JSON string values without disturbing already-valid escapes.
    result = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if not in_string:
            result.append(c)
            if c == '"':
                in_string = True
        elif c == '\\':
            nc = text[i + 1] if i + 1 < len(text) else ''
            if nc in '"\\bfnrtu/':
                result.append(c)
                result.append(nc)
                i += 2
                continue
            else:
                result.append('\\\\')   # double invalid escape
        elif c == '"':
            result.append(c)
            in_string = False
        elif c == '\n':
            result.append('\\n')        # escape literal newline inside string
        elif c == '\r':
            pass                        # drop bare CR
        else:
            result.append(c)
        i += 1
    return ''.join(result)


def _extract_json(text: str) -> dict:
    candidates = [text, _strip_fences(text), _clean_json(text)]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        m = re.search(r'\{.*\}', candidate, re.DOTALL)
        if m:
            for attempt in (m.group(), _clean_json(m.group())):
                try:
                    return json.loads(attempt)
                except json.JSONDecodeError:
                    pass
    raise ValueError(f"Could not parse JSON from model response:\n{text}")


def run(
    tex_path: str | Path,
    bib_path: str | Path,
    section_type: str,
    topic: str,
    existing_keys: set[str] | None = None,
) -> ReviewReport:
    """
    Review the final .tex against the .bib for correctness, writing, citations,
    and flow. Returns a ReviewReport with verdict and per-issue detail.

    existing_keys: cite keys from the original input that must be preserved.
    """
    text = read_tex(tex_path)
    bib = read_bib(bib_path)
    bib_keys = ", ".join(bib.keys()) if bib else "(none)"
    existing_keys_str = ", ".join(sorted(existing_keys)) if existing_keys else "(none)"

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER_TMPL.format(
            topic=topic,
            section_type=section_type,
            bib_keys=bib_keys,
            existing_keys=existing_keys_str,
            text=text,
        )},
    ]
    raw = chat("reviewer", messages, temperature=0.1)
    data = _extract_json(raw)

    issues = [
        ReviewIssue(
            category=item.get("category", "writing"),
            location=item.get("location", ""),
            description=item.get("description", ""),
            suggestion=item.get("suggestion", ""),
        )
        for item in data.get("issues", [])
    ]
    return ReviewReport(
        verdict=data.get("verdict", "needs_revision"),
        summary=data.get("summary", ""),
        revised_text=data.get("revised_text", text),
        issues=issues,
    )