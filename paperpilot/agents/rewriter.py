"""
Rewriter — produce a single clean rewrite from technical and writing critiques.
"""

from __future__ import annotations

from agents.technical_critic import TechnicalCritique
from agents.writing_critic import WritingCritique
from utils.llm import chat

_SYSTEM = """\
You are an expert academic writer for ML research papers.
You will be given:
  1. An original LaTeX paragraph.
  2. A technical critique listing factual errors and claims that need citations.
  3. A writing critique listing grammar, style, and clarity issues.

Your job: produce a single rewritten paragraph that fixes every flagged issue.

Hard rules:
  - Preserve all existing LaTeX commands exactly as-is (\\section, \\cite, \\ref,
    \\begin, \\end, math environments, etc.).
  - CRITICAL: Every \\cite{key} command present in the original paragraph MUST appear
    in your rewrite. Do NOT drop, remove, or omit any existing citation keys.
  - Do NOT invent new \\cite{} keys — leave citation placeholders as \\cite{TODO}
    if a citation is needed but the key is not yet known.
  - Do NOT change the meaning or add new factual claims beyond what the critiques direct.
  - Return ONLY the rewritten paragraph, no explanation, no preamble.
"""

_USER_TMPL = """\
Original paragraph:
{paragraph}

Technical critique:
{technical}

Writing critique:
{writing}
"""


def run(
    paragraph: str,
    technical_critique: TechnicalCritique,
    writing_critique: WritingCritique,
) -> str:
    """
    Rewrite *paragraph* addressing all issues in both critiques.
    Returns the rewritten LaTeX paragraph as a plain string.
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": _USER_TMPL.format(
                paragraph=paragraph,
                technical=technical_critique.format_for_prompt(),
                writing=writing_critique.format_for_prompt(),
            ),
        },
    ]
    return chat("rewriter", messages, temperature=0.3).strip()