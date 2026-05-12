"""
Writing Critic — assess grammar, style, register, voice, and hedging in a LaTeX paragraph.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from utils.llm import chat

_SYSTEM = """\
You are an expert academic writing editor specialising in ML research papers.
You will be given a paragraph (which may contain LaTeX markup) and the section type \
(e.g. abstract, introduction, method, experiment, conclusion).

Your job: identify every writing-level issue in the paragraph.
For each issue output:
  - "issue_type": one of
      "grammar"       — incorrect grammar or punctuation
      "spelling"      — misspelled word
      "word_choice"   — imprecise, informal, or weak word choice
      "register"      — inappropriate formality level for academic writing
      "active_passive"— passive voice where active would be clearer (or vice versa)
      "hedging"       — missing hedge on uncertain claim, or over-hedged certain fact
      "clarity"       — sentence is ambiguous or hard to parse
      "conciseness"   — wordy phrasing that can be tightened
  - "location": the exact phrase or clause from the paragraph that contains the issue
  - "suggestion": a concise, actionable fix (rewrite the phrase if useful)

Rules:
- Ignore LaTeX commands (\\cite, \\ref, \\begin, etc.) — focus only on prose.
- Do not flag things that are correct. Only report genuine issues.
- Keep suggestions specific: quote the problematic phrase, then give the improved version.
- Respond ONLY with a JSON object of this exact shape, no prose before or after, no parenthetical notes inside string values:

{
  "annotations": [
    {
      "issue_type": "...",
      "location": "...",
      "suggestion": "..."
    }
  ]
}
"""

_USER_TMPL = """\
Section type: {section_type}

Paragraph:
{paragraph}
"""


@dataclass
class WritingAnnotation:
    issue_type: str   # grammar | spelling | word_choice | register | active_passive | hedging | clarity | conciseness
    location: str     # verbatim phrase from the paragraph
    suggestion: str   # actionable fix


@dataclass
class WritingCritique:
    annotations: list[WritingAnnotation]

    def by_type(self, issue_type: str) -> list[WritingAnnotation]:
        return [a for a in self.annotations if a.issue_type == issue_type]

    def format_for_prompt(self) -> str:
        """Human-readable summary suitable for passing to the Rewriter."""
        if not self.annotations:
            return "No writing issues found."
        lines = []
        for a in self.annotations:
            lines.append(f"[{a.issue_type.upper()}] \"{a.location}\"\n  → {a.suggestion}")
        return "\n".join(lines)


def _clean_json(text: str) -> str:
    # Strip parenthetical notes appended after string values, e.g. "foo" (note)
    text = re.sub(r'"\s*\([^)]*\)', '"', text)
    # Escape invalid backslash sequences (e.g. LaTeX \cite -> \\cite). JSON only
    # allows \" \\ \/ \b \f \n \r \t \uXXXX as valid escapes.
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    # Replace literal newlines inside JSON string values with a space
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


def run(paragraph: str, section_type: str) -> WritingCritique:
    """
    Assess writing quality of *paragraph* given its *section_type*.
    Returns a WritingCritique with one WritingAnnotation per issue found.
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": _USER_TMPL.format(section_type=section_type, paragraph=paragraph)},
    ]
    raw = chat("writing_critic", messages, temperature=0.1)
    data = _extract_json(raw)

    annotations = [
        WritingAnnotation(
            issue_type=item.get("issue_type", "clarity"),
            location=item.get("location", ""),
            suggestion=item.get("suggestion", ""),
        )
        for item in data.get("annotations", [])
    ]
    return WritingCritique(annotations=annotations)
