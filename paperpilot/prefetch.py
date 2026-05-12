"""
Prefetch paper search and BibTeX results into a disk cache so the pipeline
can run on compute nodes without internet access.

Usage (run on a login node):
  # 1. Run the pipeline once on compute — it will print
  #    "[paper_search] Network unavailable and query not cached: 'foo bar'" lines.
  # 2. Extract those queries into queries.txt (one per line), then:
  python paperpilot/prefetch.py queries.txt

  # Or pipe directly:
  grep "Network unavailable and query not cached" slurm-*.out \\
      | sed -E "s/.*not cached: '(.*)'.*/\\1/" | sort -u \\
      | python paperpilot/prefetch.py -
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.paper_search import search, fetch_bibtex


def main(source: str) -> None:
    if source == "-":
        queries = [line.strip() for line in sys.stdin if line.strip()]
    else:
        queries = [line.strip() for line in Path(source).read_text().splitlines() if line.strip()]

    # Without an API key, S2 limits unauthenticated traffic to ~1 req/sec.
    delay = 0.5 if os.getenv("S2_API_KEY") else 3.0

    print(f"Prefetching {len(queries)} queries (delay={delay}s)...")
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q}")
        results = search(q, limit=5)
        for paper in results:
            bib = fetch_bibtex(paper)
            print(bib)
            time.sleep(delay)
        time.sleep(delay)
    print("Done. Cache populated.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prefetch.py <queries.txt | ->")
        sys.exit(1)
    main(sys.argv[1])