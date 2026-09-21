"""
Phase A — OpenAlex collection for the technology-term extraction PoC.

Run modes (always run them in this order the first time):

    python collect_openalex.py --topics      # find the NLP topic id, then paste it below
    python collect_openalex.py --probe       # one call: verify filters, print credit cost
    python collect_openalex.py --year 2020   # one year end to end, inspect the output
    python collect_openalex.py --all         # full year-stratified pull

Design notes that matter downstream:

1. CANONICAL TEXT. `text` = display_name + " " + cleaned abstract, single-spaced.
   Every character offset in every annotation and every model prediction is
   defined against this exact string. It is built once, here, and never
   rebuilt anywhere else. If you change GAP_POLICY or the boilerplate regexes
   after annotation starts, every annotation is invalid.

2. GAP POLICY. OpenAlex inverted indexes occasionally skip positions. We DROP
   the gaps rather than emit empty tokens, so no double spaces ever appear.
   `n_gaps` is recorded per record so you can drop corrupt ones.

3. SAMPLING. One random sample per year, so the per-year proportions used in
   the trend analysis are unbiased. Relevance-ranked top-N is not a sample.

4. OUTPUT. Parquet, not CSV. Character offsets are load-bearing and CSV does
   not round-trip text reliably.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_URL = "https://api.openalex.org"

# Free key from https://openalex.org/settings/api
# The old `mailto` polite pool was retired in Feb 2026 and is now ignored.
# Keyless = $0.10/day, which is NOT enough for this pull. A key gives $1/day.
API_KEY = os.environ.get("OPENALEX_API_KEY", "")

# Run --topics to resolve this. Leave as None to collect without it, but the
# topic filter is the single most effective guard against power-electronics
# "transformer" papers.
PRIMARY_TOPIC_ID: str | None = None  # e.g. "T10028"

SEARCH_TERMS = (
    "transformer language model OR large language model OR "
    "pretrained language model OR BERT OR self-attention OR "
    "sentence embedding OR masked language modeling"
)

YEARS = list(range(2017, 2027))
PER_YEAR = 1200
PER_PAGE = 100          # current API maximum
SAMPLE_SEED = 42        # reproducible sampling

OUT_DIR = Path("openalex_corpus")
RAW_DIR = OUT_DIR / "by_year"
FINAL_PARQUET = OUT_DIR / "corpus.parquet"
COUNTS_JSON = OUT_DIR / "year_population_counts.json"

SELECT_FIELDS = ",".join([
    "id",
    "doi",
    "display_name",
    "publication_date",
    "publication_year",
    "cited_by_count",
    "type",
    "primary_location",
    "primary_topic",
    "abstract_inverted_index",
])

# --------------------------------------------------------------------------
# Text cleaning — frozen once annotation begins
# --------------------------------------------------------------------------

STRUCTURED_HEADER = re.compile(
    r"\b(BACKGROUND|OBJECTIVES?|METHODS?|MATERIALS AND METHODS|RESULTS?|"
    r"CONCLUSIONS?|PURPOSE|INTRODUCTION|DISCUSSION|AIMS?|FINDINGS|"
    r"SIGNIFICANCE|ABSTRACT)\s*:\s*",
    re.IGNORECASE,
)

COPYRIGHT_MARKER = re.compile(
    r"(©|\(C\)\s*\d{4}|Copyright\s+\d{4}|All rights reserved|"
    r"This article is protected by copyright)",
    re.IGNORECASE,
)

WHITESPACE = re.compile(r"\s+")


def reconstruct_abstract(inverted_index: dict | None) -> tuple[str | None, int]:
    """Inverted index -> plain text. Returns (text, n_gaps).

    Gaps are dropped, not preserved as empty tokens. See GAP POLICY above.
    """
    if not inverted_index:
        return None, 0

    positions = [
        (pos, word)
        for word, pos_list in inverted_index.items()
        for pos in pos_list
    ]
    if not positions:
        return None, 0

    positions.sort(key=lambda p: p[0])
    n_gaps = (positions[-1][0] + 1) - len(positions)
    return " ".join(word for _, word in positions), n_gaps


def strip_boilerplate(text: str) -> str:
    """Remove structured-abstract headers and trailing copyright notices.

    Conservative on purpose: the copyright cut only fires in the last quarter
    of the string, so a paper that discusses copyright in its abstract is not
    truncated mid-argument.
    """
    text = STRUCTURED_HEADER.sub(" ", text)

    match = None
    for m in COPYRIGHT_MARKER.finditer(text):
        match = m
    if match and match.start() > 0.75 * len(text):
        text = text[: match.start()]

    return text.strip()


def build_canonical_text(display_name: str, abstract: str) -> str:
    """THE canonical string. Nothing downstream may rebuild this differently."""
    title = WHITESPACE.sub(" ", unicodedata.normalize("NFC", display_name)).strip()
    body = WHITESPACE.sub(" ", unicodedata.normalize("NFC", abstract)).strip()
    return f"{title} {body}"


def normalize_title(title: str) -> str:
    """For duplicate detection only. Never used for offsets."""
    t = unicodedata.normalize("NFKD", title.lower())
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return WHITESPACE.sub(" ", t).strip()


# --------------------------------------------------------------------------
# HTTP with backoff and budget awareness
# --------------------------------------------------------------------------

class BudgetExhausted(RuntimeError):
    pass


def request_with_retry(session, path, params, max_attempts=6):
    """GET with exponential backoff. Distinguishes transient 429s from a
    drained daily budget, which backoff cannot fix (it resets at midnight UTC).
    """
    url = f"{BASE_URL}{path}"
    params = {**params, "api_key": API_KEY}

    for attempt in range(max_attempts):
        try:
            resp = session.get(url, params=params, timeout=90)
        except requests.RequestException as exc:
            wait = min(60, 2 ** attempt) + random.uniform(0, 1)
            print(f"  network error ({exc.__class__.__name__}); retry in {wait:.0f}s")
            time.sleep(wait)
            continue

        remaining = resp.headers.get("X-RateLimit-Remaining")

        if resp.status_code == 429:
            if remaining is not None and float(remaining) <= 0:
                raise BudgetExhausted(
                    "Daily API budget exhausted. Backoff will not help; the "
                    "budget resets at midnight UTC. Partial results are "
                    f"already saved in {RAW_DIR}/. Re-run tomorrow, or check "
                    "that API_KEY is actually set."
                )
            wait = min(60, 2 ** attempt) + random.uniform(0, 1)
            print(f"  429 (throttle); retry in {wait:.0f}s")
            time.sleep(wait)
            continue

        if resp.status_code >= 500:
            wait = min(60, 2 ** attempt) + random.uniform(0, 1)
            print(f"  {resp.status_code}; retry in {wait:.0f}s")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json(), resp.headers

    raise RuntimeError(f"Gave up after {max_attempts} attempts on {path}")


# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------

def build_filter(year: int) -> str:
    parts = [
        f"title_and_abstract.search:{SEARCH_TERMS}",
        f"publication_year:{year}",
        "language:en",
        "type:article|preprint",
        "has_abstract:true",
    ]
    if PRIMARY_TOPIC_ID:
        parts.append(f"primary_topic.id:{PRIMARY_TOPIC_ID}")
    return ",".join(parts)


def parse_work(work: dict) -> dict | None:
    abstract, n_gaps = reconstruct_abstract(work.get("abstract_inverted_index"))
    title = work.get("display_name")
    if not abstract or not title:
        return None

    abstract = strip_boilerplate(abstract)
    if len(abstract.split()) < 40:
        return None

    source = (work.get("primary_location") or {}).get("source") or {}
    topic = work.get("primary_topic") or {}

    return {
        "openalex_id": work["id"],
        "doi": work.get("doi"),
        "title": title,
        "abstract": abstract,
        "text": build_canonical_text(title, abstract),
        "publication_date": work.get("publication_date"),
        "publication_year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count", 0),
        "work_type": work.get("type"),
        "venue": source.get("display_name"),
        "primary_topic": topic.get("display_name"),
        "n_gaps": n_gaps,
    }


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------

def collect_year(session, year: int) -> tuple[list[dict], int]:
    """Random sample of PER_YEAR works from `year`.

    `sample` requires basic paging (page=1,2,...) with a fixed seed; it is not
    compatible with cursor paging or with `sort`. If your API version rejects
    sample+page, this raises immediately rather than silently degrading.
    """
    records: list[dict] = []
    n_pages = -(-PER_YEAR // PER_PAGE)
    population = None

    for page in range(1, n_pages + 1):
        params = {
            "filter": build_filter(year),
            "sample": PER_YEAR,
            "seed": SAMPLE_SEED,
            "per_page": PER_PAGE,
            "page": page,
            "select": SELECT_FIELDS,
        }
        payload, headers = request_with_retry(session, "/works", params)

        if population is None:
            population = payload.get("meta", {}).get("count")

        results = payload.get("results", [])
        if not results:
            break

        for work in results:
            parsed = parse_work(work)
            if parsed:
                records.append(parsed)

        print(
            f"  {year} page {page}/{n_pages}: "
            f"{len(records)} kept | credits left {headers.get('X-RateLimit-Remaining')}"
        )
        time.sleep(0.15)

    return records, (population or 0)


def save_year(records: list[dict], year: int) -> None:
    """Write immediately. A crash at year 2024 must not cost you 2017-2023."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(RAW_DIR / f"{year}.parquet", index=False)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Three passes. The title pass is the one that matters: an arXiv preprint
    and its conference version are distinct works with distinct OpenAlex ids
    and near-identical text. Leaving both in leaks train into test.
    """
    before = len(df)

    df = df.drop_duplicates(subset="openalex_id", keep="first")
    after_id = len(df)

    has_doi = df["doi"].notna()
    df = pd.concat([
        df[has_doi].drop_duplicates(subset="doi", keep="first"),
        df[~has_doi],
    ])
    after_doi = len(df)

    df = df.assign(_norm_title=df["title"].map(normalize_title))
    df = df.sort_values("cited_by_count", ascending=False)  # keep the canonical version
    df = df.drop_duplicates(subset="_norm_title", keep="first").drop(columns="_norm_title")
    after_title = len(df)

    print(
        f"\nDeduplication: {before} -> {after_id} (id) -> "
        f"{after_doi} (doi) -> {after_title} (title)"
    )
    print(f"  {before - after_title} duplicates removed "
          f"({100 * (before - after_title) / max(before, 1):.1f}%)")
    return df.reset_index(drop=True)


def finalize() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(RAW_DIR.glob("*.parquet"))]
    if not frames:
        sys.exit("No per-year files found. Run --year or --all first.")

    df = pd.concat(frames, ignore_index=True)
    df = deduplicate(df)

    # Offsets are meaningless if the text does not survive a round trip.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FINAL_PARQUET, index=False)
    reloaded = pd.read_parquet(FINAL_PARQUET)
    assert reloaded["text"].equals(df["text"]), "Text did not round-trip."

    print(f"\nSaved {len(df):,} records to {FINAL_PARQUET.resolve()}")
    print(f"Records with index gaps: {(df['n_gaps'] > 0).sum()} "
          f"(max {df['n_gaps'].max()})")
    print("\nPer year:")
    print(df["publication_year"].value_counts().sort_index().to_string())
    print("\nTop primary topics (check for power electronics):")
    print(df["primary_topic"].value_counts().head(12).to_string())
    return df


# --------------------------------------------------------------------------
# Helper modes
# --------------------------------------------------------------------------

def mode_topics(session) -> None:
    payload, _ = request_with_retry(
        session, "/topics",
        {"search": "natural language processing", "per_page": 10},
    )
    print("Paste the best-fitting id into PRIMARY_TOPIC_ID:\n")
    for t in payload.get("results", []):
        tid = t["id"].rsplit("/", 1)[-1]
        count = t.get("works_count")
        count_str = f"{count:,}" if isinstance(count, int) else "?"
        print(f"  {tid:<8} {t.get('display_name', ''):<55} works={count_str}")


def mode_probe(session) -> None:
    params = {
        "filter": build_filter(2023),
        "per_page": 5,
        "select": SELECT_FIELDS,
    }
    payload, headers = request_with_retry(session, "/works", params)
    meta = payload.get("meta", {})

    print(f"Matching works in 2023 : {meta.get('count'):,}")
    print(f"Credits used this call : {headers.get('X-RateLimit-Credits-Used')}")
    print(f"Credits remaining      : {headers.get('X-RateLimit-Remaining')} "
          f"/ {headers.get('X-RateLimit-Limit')}")
    print(f"Estimated full pull    : ~{len(YEARS) * -(-PER_YEAR // PER_PAGE)} calls\n")

    for work in payload.get("results", [])[:3]:
        parsed = parse_work(work)
        if parsed:
            print(f"- [{parsed['primary_topic']}] {parsed['title'][:90]}")
            print(f"  {parsed['text'][:180]}...\n")


def mode_collect(session, years: list[int]) -> None:
    populations = {}
    if COUNTS_JSON.exists():
        populations = json.loads(COUNTS_JSON.read_text())

    for year in years:
        print(f"\n=== {year} ===")
        try:
            records, population = collect_year(session, year)
        except BudgetExhausted as exc:
            print(f"\n{exc}")
            break
        save_year(records, year)
        populations[str(year)] = population
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        COUNTS_JSON.write_text(json.dumps(populations, indent=2))
        print(f"  {year}: {len(records)} kept of {population:,} matching works")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--year", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    if not API_KEY:
        sys.exit("Set OPENALEX_API_KEY first: https://openalex.org/settings/api")

    random.seed(SAMPLE_SEED)
    session = requests.Session()

    if args.topics:
        mode_topics(session)
    elif args.probe:
        mode_probe(session)
    elif args.year:
        mode_collect(session, [args.year])
        finalize()
    elif args.all:
        mode_collect(session, YEARS)
        finalize()
    elif args.finalize:
        finalize()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
