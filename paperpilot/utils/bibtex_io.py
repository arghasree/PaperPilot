"""
I/O helpers for BibTeX files: read, append new entries, deduplicate by key.
"""

import re
from pathlib import Path

# Matches the opening line of any BibTeX entry, e.g. @article{vaswani2017,
_ENTRY_START = re.compile(r'^(@\w+\s*\{)', re.MULTILINE)


def _parse_entries(text: str) -> dict[str, str]:
    """
    Return an ordered dict mapping citation key -> full entry string.
    Handles nested braces so multi-field entries are captured whole.
    """
    entries: dict[str, str] = {}
    for match in _ENTRY_START.finditer(text):
        start = match.start()
        # Walk forward counting braces to find the closing '}'
        depth = 0
        i = start
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        entry = text[start : i + 1]
        key = _extract_key(entry)
        if key and key not in entries:
            entries[key] = entry
    return entries


def _extract_key(entry: str) -> str | None:
    """Pull the citation key from the first line of an entry."""
    m = re.match(r'@\w+\s*\{\s*([^,\s]+)', entry)
    return m.group(1) if m else None


def read_bib(path: str | Path) -> dict[str, str]:
    """
    Read a .bib file and return {citation_key: entry_string}.
    Returns an empty dict if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        return {}
    return _parse_entries(p.read_text(encoding="utf-8"))


def append_entries(path: str | Path, new_entries: list[str]) -> dict[str, str]:
    """
    Append new BibTeX entry strings to a .bib file, skipping duplicates by key.

    Returns the full {key: entry} dict after merging.
    """
    existing = read_bib(path)

    to_add: list[str] = []
    for raw in new_entries:
        key = _extract_key(raw.strip())
        if key and key not in existing:
            existing[key] = raw.strip()
            to_add.append(raw.strip())

    if to_add:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            for entry in to_add:
                f.write("\n" + entry + "\n")

    return existing


def write_bib(path: str | Path, entries: dict[str, str]) -> None:
    """Write a complete {key: entry} dict to a .bib file, replacing its contents."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(entries.values()) + "\n", encoding="utf-8")


def deduplicate(path: str | Path) -> int:
    """
    Rewrite a .bib file keeping only the first occurrence of each key.
    Returns the number of duplicate entries removed.
    """
    p = Path(path)
    if not p.exists():
        return 0
    raw = p.read_text(encoding="utf-8")
    before = len(_ENTRY_START.findall(raw))
    entries = _parse_entries(raw)
    write_bib(p, entries)
    return before - len(entries)
