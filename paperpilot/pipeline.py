"""
Orchestrates all five agents in sequence.

Single-paragraph mode:  one paragraph, end-to-end.
Multi-paragraph mode:   each paragraph refined independently, then the full
                        section is passed to the Reviewer for flow/coherence.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import agents.citation_grounder as citation_grounder
import agents.reviewer as reviewer
import agents.rewriter as rewriter
import agents.technical_critic as technical_critic
import agents.writing_critic as writing_critic
from agents.citation_grounder import CitationGrounderResult
from agents.reviewer import ReviewReport
from agents.technical_critic import TechnicalCritique
from agents.writing_critic import WritingCritique
from utils import checkpoint as ckpt_io
from utils.bibtex_io import read_bib
from utils.latex_io import extract_body, join_paragraphs, read_tex, split_paragraphs, write_tex
from utils.llm import chat


@dataclass
class ParagraphResult:
    original: str
    technical_critique: TechnicalCritique
    writing_critique: WritingCritique
    rewritten: str
    citation_result: CitationGrounderResult

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "technical_issues": [
                {
                    "claim": a.claim,
                    "verdict": a.verdict,
                    "correction": a.correction,
                    "citation_needed": a.citation_needed,
                    "suggested_search_query": a.suggested_search_query,
                }
                for a in self.technical_critique.assessments
            ],
            "writing_issues": [
                {
                    "issue_type": a.issue_type,
                    "location": a.location,
                    "suggestion": a.suggestion,
                }
                for a in self.writing_critique.annotations
            ],
            "citations_grounded": [
                {"claim": g.claim, "cite_key": g.cite_key}
                for g in self.citation_result.grounded
            ],
            "citations_not_found": self.citation_result.not_found,
            "final": self.citation_result.paragraph,
        }


@dataclass
class PipelineResult:
    tex_path: Path
    bib_path: Path
    report: ReviewReport
    paragraphs: list[ParagraphResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tex_path": str(self.tex_path),
            "bib_path": str(self.bib_path),
            "verdict": self.report.verdict,
            "summary": self.report.summary,
            "review_issues": [
                {
                    "category": i.category,
                    "location": i.location,
                    "description": i.description,
                    "suggestion": i.suggestion,
                }
                for i in self.report.issues
            ],
            "paragraphs": [p.to_dict() for p in self.paragraphs],
        }


_RESTORE_SYSTEM = """\
You are editing a LaTeX paragraph. Some existing citation keys were accidentally dropped.
You will be given the paragraph and, for each missing key, the ORIGINAL sentence it appeared in.

Rules:
  - Re-insert \\cite{key} in the SAME sentence it appeared in originally.
  - Identify that sentence in the paragraph by matching its content (it may have been lightly reworded).
  - If a \\cite{} already exists in that sentence, append the key: \\cite{existing,missing}.
  - Otherwise insert \\cite{key} immediately before the sentence-ending period.
  - Do NOT change any other text. Return ONLY the updated paragraph.
"""

_RESTORE_USER = """\
Paragraph:
{paragraph}

Keys to restore with their original sentence context:
{items}
"""


def _extract_cite_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for m in re.finditer(r'\\cite\{([^}]+)\}', text):
        for key in m.group(1).split(','):
            keys.add(key.strip())
    return keys


def _build_key_context(text: str) -> dict[str, str]:
    """Map each cite key to the sentence it appeared in."""
    key_to_sent: dict[str, str] = {}
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        for key in _extract_cite_keys(sent):
            key_to_sent[key] = sent
    return key_to_sent


def _restore_dropped_cites(original: str, rewritten: str, agent: str = "rewriter") -> str:
    """Re-inject any \\cite{key} from *original* that are absent in *rewritten*."""
    missing = _extract_cite_keys(original) - _extract_cite_keys(rewritten)
    if not missing:
        return rewritten
    key_context = _build_key_context(original)
    items = "\n".join(
        f'- \\cite{{{k}}}: originally in sentence: "{key_context.get(k, "(unknown)")}"'
        for k in sorted(missing)
    )
    messages = [
        {"role": "system", "content": _RESTORE_SYSTEM},
        {"role": "user", "content": _RESTORE_USER.format(
            paragraph=rewritten,
            items=items,
        )},
    ]
    return chat(agent, messages, temperature=0.0).strip()


_TODO_KEY_RE = re.compile(r'^TODO\b', re.IGNORECASE)


def _scrub_cites(text: str, valid_keys: set[str]) -> tuple[str, list[str]]:
    """
    Deterministically remove TODO placeholders and unknown keys from \\cite{...}.

    - Drops any key matching /^TODO/i (e.g. "TODO", 'TODO: query "..."').
    - Drops any key not in *valid_keys*.
    - Empty \\cite{} commands (all keys dropped) are removed entirely along
      with one trailing space if present.

    Returns (cleaned_text, removed_keys).
    """
    removed: list[str] = []

    def replace(m: re.Match) -> str:
        raw_keys = [k.strip() for k in m.group(1).split(',')]
        kept = []
        for k in raw_keys:
            if _TODO_KEY_RE.match(k) or k not in valid_keys:
                removed.append(k)
            else:
                kept.append(k)
        if not kept:
            return ""
        return r'\cite{' + ', '.join(kept) + '}'

    cleaned = re.sub(r'\\cite\{([^}]+)\}', replace, text)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)         # collapse double spaces from removals
    cleaned = re.sub(r'[ \t]+([.,;:])', r'\1', cleaned)  # tighten orphan punctuation
    return cleaned, removed


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def run(
    input_path: str | Path,
    output_dir: str | Path,
    section_type: str,
    topic: str,
    bib_path: str | Path | None = None,
    multi: bool = False,
    verbose: bool = False,
) -> PipelineResult:
    """
    Run the full pipeline on a .tex file and write results to output_dir.

    Args:
        input_path:   Path to the input .tex file.
        output_dir:   Directory for output.tex and output.bib.
        section_type: Section context for agents (e.g. "method", "intro").
        topic:        Paper topic for agents (e.g. "GFlowNets for graph search").
        bib_path:     Optional existing .bib to append to.
        multi:        Split into paragraphs and refine each independently.
        verbose:      Print per-agent progress to stdout.

    Returns:
        PipelineResult with paths to output files and the reviewer report.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tex = out_dir / "output.tex"
    out_bib = out_dir / "output.bib"

    # Seed the output .bib from an existing one if provided, else create an empty file
    if bib_path and Path(bib_path).exists():
        shutil.copy(bib_path, out_bib)
    else:
        out_bib.touch()

    # Read and optionally strip document preamble
    text = read_tex(input_path)
    original_cite_keys = _extract_cite_keys(text)
    preamble, body, postamble = extract_body(text)
    working = body if preamble else text

    paragraphs = split_paragraphs(working) if multi else [working.strip()]

    checkpoint_path = out_dir / "checkpoint.json"
    ckpt = ckpt_io.load(checkpoint_path)

    # ── Phase 1: technical critic → writing critic → rewriter (skipped on resume) ──
    if ckpt:
        _log("Resuming from checkpoint — skipping technical critic, writing critic, rewriter.", verbose)
        preamble, postamble, phase1 = ckpt
    else:
        phase1: list[dict] = []
        total = len(paragraphs)
        for idx, para in enumerate(paragraphs):
            prefix = f"[{idx + 1}/{total}]"

            _log(f"{prefix} Technical critic …", verbose)
            tech = technical_critic.run(para, topic)
            if verbose and tech.assessments:
                for a in tech.needs_correction():
                    print(f"         {a.verdict.upper()}: {a.claim}")

            _log(f"{prefix} Writing critic …", verbose)
            writing = writing_critic.run(para, section_type)
            if verbose and writing.annotations:
                for a in writing.annotations:
                    print(f"         {a.issue_type}: \"{a.location[:60]}\"")

            _log(f"{prefix} Rewriter …", verbose)
            rewritten = rewriter.run(para, tech, writing)
            rewritten = _restore_dropped_cites(para, rewritten)

            phase1.append({
                "original": para,
                "rewritten": rewritten,
                "technical_critique": tech,
                "writing_critique": writing,
            })

        ckpt_io.save(checkpoint_path, preamble, postamble, phase1)
        _log("Checkpoint saved.", verbose)

    # ── Phase 2: citation grounder (always runs) ──
    para_results: list[ParagraphResult] = []
    for idx, p1 in enumerate(phase1):
        prefix = f"[{idx + 1}/{len(phase1)}]"

        _log(f"{prefix} Citation grounder …", verbose)
        cit = citation_grounder.run(p1["rewritten"], p1["technical_critique"].needs_citation(), out_bib)
        if verbose:
            for g in cit.grounded:
                print(f"         ✓ \\cite{{{g.cite_key}}}  ← {g.claim[:60]}")
            for c in cit.not_found:
                print(f"         ✗ not found: {c[:60]}")

        para_results.append(ParagraphResult(
            original=p1["original"],
            technical_critique=p1["technical_critique"],
            writing_critique=p1["writing_critique"],
            rewritten=p1["rewritten"],
            citation_result=cit,
        ))

    # Reassemble into a temporary tex for the reviewer to read
    refined = [r.citation_result.paragraph for r in para_results]
    final_body = join_paragraphs(refined)
    pre_review_text = (preamble + final_body + postamble) if preamble else final_body
    write_tex(out_tex, pre_review_text)

    # Final holistic review — produces revised_text with all fixes applied
    _log("Reviewer …", verbose)
    report = reviewer.run(out_tex, out_bib, section_type, topic, existing_keys=original_cite_keys)

    # Restore any pre-existing citations the reviewer dropped …
    report.revised_text = _restore_dropped_cites(pre_review_text, report.revised_text, "reviewer")

    # … then scrub leftover TODO placeholders and any hallucinated keys
    # (keys present in neither the original input nor the final .bib).
    bib_keys = set(read_bib(out_bib).keys())
    valid_keys = original_cite_keys | bib_keys
    report.revised_text, removed = _scrub_cites(report.revised_text, valid_keys)
    if removed and verbose:
        print(f"Scrubbed unfixed cite keys from final output: {removed}")

    write_tex(out_tex, report.revised_text)

    # Keep the checkpoint if any citation queries were uncached (network unavailable on
    # compute node) so the resubmitted job can skip Phase 1 and jump straight to
    # citation grounding with the now-populated cache.
    any_uncached = any(r.citation_result.not_found for r in para_results)
    if not any_uncached:
        checkpoint_path.unlink(missing_ok=True)

    return PipelineResult(
        tex_path=out_tex,
        bib_path=out_bib,
        report=report,
        paragraphs=para_results,
    )
