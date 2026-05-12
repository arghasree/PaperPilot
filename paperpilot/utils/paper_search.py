"""
Paper search with fallback chain: Semantic Scholar → ArXiv → OpenAlex.

Primary path:  search() -> PaperResult  ->  fetch_bibtex() via doi.org
BibTeX fallback: if no DOI, construct BibTeX from metadata directly.
Google Scholar scraping is intentionally omitted — it violates ToS.
"""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "title,authors,year,venue,externalIds,citationCount"
_S2_HEADERS = {"User-Agent": "paperpilot/0.1 (academic research tool)"}
_DOI_HEADERS = {"Accept": "application/x-bibtex", "User-Agent": "paperpilot/0.1"}
_ARXIV_BASE = "https://export.arxiv.org/api/query"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom",
             "arxiv": "http://arxiv.org/schemas/atom"}
_OA_BASE = "https://api.openalex.org/works"
_OA_HEADERS = {"User-Agent": "paperpilot/0.1 (mailto:paperpilot@research.invalid)"}
_OA_SELECT = "title,authorships,publication_year,primary_location,doi,cited_by_count,ids"

# Optional: set S2_API_KEY env var to get higher rate limits
_API_KEY = os.getenv("S2_API_KEY")
if _API_KEY:
    _S2_HEADERS["x-api-key"] = _API_KEY

# Disk cache for offline use (compute nodes have no internet).
# Populate on a login node by running the pipeline or prefetch.py with internet
# access; the cache file is then read on subsequent runs.
_CACHE_PATH = Path(os.getenv(
    "PAPER_SEARCH_CACHE",
    str(Path(__file__).resolve().parent.parent / "paper_search_cache.json")
))


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text())
        except Exception:
            return {"search": {}, "bibtex": {}}
    return {"search": {}, "bibtex": {}}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _search_key(query: str, limit: int) -> str:
    return f"{query.strip().lower()}|{limit}"


@dataclass
class PaperResult:
    title: str
    authors: list[str]      # ["Ashish Vaswani", "Noam Shazeer", ...]
    year: int | None
    venue: str
    doi: str | None
    s2_id: str
    arxiv_id: str | None
    citation_count: int

    def cite_key(self) -> str:
        """Generate a citation key: firstauthorlastname + year + firstcontentword."""
        last = self.authors[0].split()[-1].lower() if self.authors else "unknown"
        last = re.sub(r'[^a-z]', '', last)
        year = str(self.year) if self.year else "nd"
        # First meaningful word of the title
        stop = {"a", "an", "the", "of", "in", "on", "for", "and", "with", "is"}
        words = [w.lower() for w in re.split(r'\W+', self.title) if w]
        content = next((w for w in words if w not in stop), words[0] if words else "paper")
        content = re.sub(r'[^a-z]', '', content)
        return f"{last}{year}{content}"

    def _entry_type(self) -> str:
        v = self.venue.lower()
        if "arxiv" in v:
            return "misc"
        if any(k in v for k in ("journal", "transactions", "letters")):
            return "article"
        return "inproceedings"

    def to_bibtex(self) -> str:
        """Construct a BibTeX entry from metadata (used when DOI fetch fails)."""
        key = self.cite_key()
        etype = self._entry_type()
        author_str = " and ".join(self.authors)
        venue_field = "journal" if etype == "article" else "booktitle"

        lines = [f"@{etype}{{{key},"]
        lines.append(f"  title     = {{{self.title}}},")
        lines.append(f"  author    = {{{author_str}}},")
        if self.year:
            lines.append(f"  year      = {{{self.year}}},")
        if self.venue:
            lines.append(f"  {venue_field} = {{{self.venue}}},")
        if self.doi:
            lines.append(f"  doi       = {{{self.doi}}},")
        if self.arxiv_id:
            lines.append(f"  eprint    = {{{self.arxiv_id}}},")
            lines.append(f"  archivePrefix = {{arXiv}},")
        lines.append("}")
        return "\n".join(lines)


# ── per-source search helpers ─────────────────────────────────────────────────

def _search_s2(query: str, limit: int) -> list[PaperResult]:
    params = {"query": query, "limit": limit, "fields": _S2_FIELDS}
    resp = None
    try:
        for attempt in range(5):
            resp = requests.get(f"{_S2_BASE}/paper/search", params=params,
                                headers=_S2_HEADERS, timeout=15)
            if resp.status_code != 429:
                break
            wait = int(resp.headers.get("Retry-After", 0)) or (2 ** attempt)
            print(f"[paper_search] S2 429 rate-limited, sleeping {wait}s (attempt {attempt+1}/5)")
            time.sleep(wait)
        if resp is None or not resp.ok:
            code = resp.status_code if resp is not None else "?"
            print(f"[paper_search] S2 search failed ({code}).")
            return []
    except requests.RequestException:
        print(f"[paper_search] S2 network error for {query!r}.")
        return []

    results = []
    for p in resp.json().get("data", []):
        ext = p.get("externalIds") or {}
        results.append(PaperResult(
            title=p.get("title", ""),
            authors=[a["name"] for a in p.get("authors", [])],
            year=p.get("year"),
            venue=p.get("venue") or "",
            doi=ext.get("DOI"),
            s2_id=p.get("paperId", ""),
            arxiv_id=ext.get("ArXiv"),
            citation_count=p.get("citationCount") or 0,
        ))
    return results


def _search_arxiv(query: str, limit: int) -> list[PaperResult]:
    try:
        resp = requests.get(_ARXIV_BASE, params={
            "search_query": f"all:{query}",
            "max_results": limit,
            "sortBy": "relevance",
        }, headers=_S2_HEADERS, timeout=15)
        if not resp.ok:
            print(f"[paper_search] ArXiv search failed ({resp.status_code}).")
            return []
    except requests.RequestException:
        print(f"[paper_search] ArXiv network error for {query!r}.")
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    results = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        title = (entry.findtext("atom:title", "", _ARXIV_NS) or "").strip().replace("\n", " ")
        authors = [
            a.findtext("atom:name", "", _ARXIV_NS)
            for a in entry.findall("atom:author", _ARXIV_NS)
        ]
        published = entry.findtext("atom:published", "", _ARXIV_NS) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        abs_url = entry.findtext("atom:id", "", _ARXIV_NS) or ""
        # Strip version suffix: "http://arxiv.org/abs/1234.5678v2" → "1234.5678"
        arxiv_id = re.sub(r'v\d+$', '', abs_url.split("/abs/")[-1]) if "/abs/" in abs_url else None
        doi = entry.findtext("arxiv:doi", None, _ARXIV_NS)
        results.append(PaperResult(
            title=title,
            authors=authors,
            year=year,
            venue="arXiv",
            doi=doi,
            s2_id="",
            arxiv_id=arxiv_id,
            citation_count=0,
        ))
    return results


def _search_openalex(query: str, limit: int) -> list[PaperResult]:
    try:
        resp = requests.get(_OA_BASE, params={
            "search": query,
            "per-page": limit,
            "select": _OA_SELECT,
        }, headers=_OA_HEADERS, timeout=15)
        if not resp.ok:
            print(f"[paper_search] OpenAlex search failed ({resp.status_code}).")
            return []
    except requests.RequestException:
        print(f"[paper_search] OpenAlex network error for {query!r}.")
        return []

    results = []
    for item in resp.json().get("results", []):
        authors = [
            a["author"]["display_name"]
            for a in item.get("authorships", [])
            if a.get("author")
        ]
        loc = item.get("primary_location") or {}
        src = loc.get("source") or {}
        venue = src.get("display_name") or ""

        doi = item.get("doi") or None
        if doi and doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]

        ids = item.get("ids") or {}
        arxiv_url = ids.get("arxiv") or ""
        arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else None

        results.append(PaperResult(
            title=item.get("title") or "",
            authors=authors,
            year=item.get("publication_year"),
            venue=venue,
            doi=doi,
            s2_id="",
            arxiv_id=arxiv_id,
            citation_count=item.get("cited_by_count") or 0,
        ))
    return results


# ── public API ────────────────────────────────────────────────────────────────

def search(query: str, limit: int = 5) -> list[PaperResult]:
    """
    Search for *query* and return up to *limit* results.

    Falls back through Semantic Scholar → ArXiv → OpenAlex until results are found.
    Disk-cached: a cache hit skips all network calls. Empty results are NOT cached
    so that fallbacks are retried on the next run (useful after internet becomes
    available on compute nodes).
    """
    cache = _load_cache()
    key = _search_key(query, limit)
    if key in cache.get("search", {}):
        return [PaperResult(**p) for p in cache["search"][key]]

    results = _search_s2(query, limit)

    if not results:
        print(f"[paper_search] S2 returned no results for {query!r}, trying ArXiv …")
        results = _search_arxiv(query, limit)

    if not results:
        print(f"[paper_search] ArXiv returned no results for {query!r}, trying OpenAlex …")
        results = _search_openalex(query, limit)

    if not results:
        # This exact message is scraped by run_pipeline.sh to trigger prefetch + resubmit.
        print(f"[paper_search] Network unavailable and query not cached: {query!r}")
        return []

    cache.setdefault("search", {})[key] = [asdict(r) for r in results]
    _save_cache(cache)
    return results


def fetch_bibtex(paper: PaperResult) -> str:
    """
    Return a BibTeX string for *paper*.

    Disk-cached by S2 ID (or DOI/cite_key for ArXiv/OpenAlex results).
    Tries doi.org first (real publisher BibTeX); falls back to constructing
    an entry from metadata.
    """
    cache = _load_cache()
    key = paper.s2_id or paper.doi or paper.cite_key()
    if key in cache.get("bibtex", {}):
        return cache["bibtex"][key]

    bib = None
    if paper.doi:
        try:
            resp = requests.get(f"https://doi.org/{paper.doi}",
                                headers=_DOI_HEADERS, timeout=10,
                                allow_redirects=True)
            if resp.ok and resp.text.strip().startswith("@"):
                bib = resp.text.strip()
        except requests.RequestException:
            pass

    if bib is None:
        bib = paper.to_bibtex()

    cache.setdefault("bibtex", {})[key] = bib
    _save_cache(cache)
    return bib
