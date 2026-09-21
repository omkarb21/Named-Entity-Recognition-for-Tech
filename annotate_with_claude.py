"""
Phase B — annotate gold sets with Claude.

    python annotate_with_claude.py --input gold_val.jsonl  --output gold_val_annotated.jsonl
    python annotate_with_claude.py --input gold_test.jsonl --output gold_test_annotated.jsonl
    python annotate_with_claude.py --input gold_test.jsonl --output gold_test_annotated.jsonl --verify-sample 30

Design notes:

1. THE MODEL NEVER PRODUCES OFFSETS. LLMs cannot count characters reliably.
   It returns verbatim substrings; this script locates them with str.find.
   A span that is not verbatim in the text is DROPPED and logged, not guessed at.

2. RESUMABLE. Output is append-only JSONL keyed by openalex_id. Re-running
   skips what is already done, so a crash costs nothing.

3. NO TEMPERATURE. Claude Sonnet 5 rejects non-default sampling parameters.

Cost: ~240 abstracts is well under $1. The full 7,580-abstract corpus is
roughly $20 on the sync API, about half that via the Batch API.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

MODEL = "claude-haiku-4-5-20251001"
MAX_WORKERS = 8
MAX_TOKENS = 2000

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
write_lock = threading.Lock()


# --------------------------------------------------------------------------
# The annotation guideline. This IS the frozen policy — if you edit it after
# annotation starts, every previous annotation is on a different standard.
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are annotating scientific abstracts for a named-entity recognition dataset.
Mark every mention of a technology or a task.

LABELS

METHOD — a concrete model, architecture, algorithm, representation, training
technique, system, device, or material. The *how*. Test: could you in
principle implement or build it?
  Examples: BERT, Transformer encoder-decoder, self-attention, masked language
  modeling, wordpiece tokenization, retrieval-augmented generation, contrastive
  sentence embeddings, LoRA, parameter-efficient fine-tuning, knowledge graph
  embedding, neural machine translation, BiLSTM-CRF

TASK — a problem being solved or a capability being measured. The *what*.
Test: is it something a method is applied to?
  Examples: named entity recognition, question answering, machine translation,
  semantic parsing, text summarization, sentiment analysis, coreference resolution

Note the pair: "neural machine translation" is METHOD (it names an approach);
"machine translation" is TASK (it names the problem).

NEVER LABEL

- Fields and disciplines: natural language processing, machine learning,
  deep learning, artificial intelligence, computational linguistics
- Vague research language: novel method, proposed approach, our model,
  experimental setup, state-of-the-art results, high performance
- Bare generic nouns: model, method, framework, system, technique, task
- Datasets, benchmarks, metrics, institutions, or programming languages
- Vague capability phrases: semantic understanding, language understanding

GRANULARITY

Label the most specific named thing present. Do not label a category term when
a more specific term names the same thing.
- "a large language model such as GPT-4" -> label GPT-4 only
- "we evaluate large language models on X" -> label "large language models"
  (it is the subject of the claim, nothing more specific is present)
- "machine learning" -> never; it is a field

BOUNDARIES

Label the shortest span that uniquely names the thing.
1. Drop determiners, possessives, and evaluative adjectives: a, the, our,
   proposed, novel, state-of-the-art, efficient.
2. Drop empty head nouns (model, method, approach, framework, system,
   technique, architecture) UNLESS removing the word leaves something that is
   not a noun phrase, or the word is part of a conventional name.
3. Keep premodifiers only when removing them changes which thing is named.

  "a pretrained BERT model"        -> BERT
  "LoRA fine-tuning"               -> LoRA fine-tuning   (rule 3)
  "multi-head attention"           -> multi-head attention (rule 3)
  "encoder-decoder architecture"   -> encoder-decoder architecture (rule 2 exception)
  "a novel attention mechanism"    -> nothing (no identifying name)

REPEATED MENTIONS

List each distinct surface form once. The script labels every occurrence.
Include both an acronym and its expansion when both appear:
"retrieval-augmented generation (RAG)" -> two entries, "retrieval-augmented
generation" and "RAG", both METHOD.

OUTPUT

Return ONLY a JSON object, no prose, no markdown fences:

{"entities": [{"text": "<exact substring>", "label": "METHOD"}, ...]}

Every "text" value MUST appear verbatim in the input, character for character,
with the same capitalization and hyphenation. Do not normalize, expand, or
correct anything. If there are no entities, return {"entities": []}.
"""


# --------------------------------------------------------------------------
# Span location
# --------------------------------------------------------------------------

def locate_spans(text: str, entities: list[dict]) -> tuple[list[dict], list[str]]:
    """Map verbatim strings to character offsets.

    Longest first, so that "neural machine translation" claims its characters
    before "machine translation" can. Overlaps are rejected.
    """
    spans: list[dict] = []
    dropped: list[str] = []
    claimed: list[tuple[int, int]] = []

    ordered = sorted(
        entities,
        key=lambda e: len(e.get("text", "")),
        reverse=True,
    )

    for entity in ordered:
        surface = (entity.get("text") or "").strip()
        label = entity.get("label")
        if not surface or label not in ("METHOD", "TASK"):
            dropped.append(f"{surface!r} (bad label: {label!r})")
            continue

        found_any = False
        for match in re.finditer(re.escape(surface), text):
            start, end = match.start(), match.end()
            if any(start < c_end and end > c_start for c_start, c_end in claimed):
                continue
            spans.append({"start": start, "end": end, "label": label,
                          "text": surface})
            claimed.append((start, end))
            found_any = True

        if not found_any:
            # Not verbatim in the text. Do not guess — record and move on.
            dropped.append(f"{surface!r} (not found verbatim)")

    spans.sort(key=lambda s: s["start"])
    return spans, dropped


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

def extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in response: {raw[:200]}")
    return json.loads(raw[start:end + 1])


def annotate_one(record: dict, attempts: int = 3) -> dict:
    text = record["text"]
    last_error = None

    for _ in range(attempts):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            payload = extract_json(
                "".join(b.text for b in response.content if b.type == "text")
            )
            spans, dropped = locate_spans(text, payload.get("entities", []))
            return {
                "openalex_id": record["openalex_id"],
                "text": text,
                "publication_year": record.get("publication_year"),
                "entities": spans,
                "dropped": dropped,
                "annotator": MODEL,
            }
        except Exception as exc:          # noqa: BLE001 - PoC
            last_error = exc

    return {
        "openalex_id": record["openalex_id"],
        "text": text,
        "publication_year": record.get("publication_year"),
        "entities": [],
        "error": str(last_error),
        "annotator": MODEL,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["openalex_id"] for r in load_jsonl(path)}


def run(input_path: Path, output_path: Path) -> None:
    records = load_jsonl(input_path)
    done = already_done(output_path)
    todo = [r for r in records if r["openalex_id"] not in done]

    print(f"{len(records)} records, {len(done)} already annotated, "
          f"{len(todo)} to go")
    if not todo:
        return

    completed = 0
    with output_path.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(annotate_one, r): r for r in todo}
            for future in as_completed(futures):
                result = future.result()
                with write_lock:
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
                completed += 1
                if completed % 10 == 0:
                    print(f"  {completed}/{len(todo)}")

    report(output_path)


def report(output_path: Path) -> None:
    records = load_jsonl(output_path)
    n_entities = sum(len(r["entities"]) for r in records)
    n_method = sum(1 for r in records for e in r["entities"]
                   if e["label"] == "METHOD")
    n_task = n_entities - n_method
    errors = [r for r in records if r.get("error")]
    dropped = [d for r in records for d in r.get("dropped", [])]
    empty = sum(1 for r in records if not r["entities"])

    print(f"\n{'=' * 55}")
    print(f"Documents          : {len(records)}")
    print(f"Entities           : {n_entities}")
    print(f"  METHOD           : {n_method}")
    print(f"  TASK             : {n_task}")
    print(f"Entities/doc       : {n_entities / max(len(records), 1):.2f}")
    print(f"  METHOD/doc       : {n_method / max(len(records), 1):.2f}")
    print(f"Docs with 0 spans  : {empty}")
    print(f"Non-verbatim drops : {len(dropped)}")
    print(f"Failed documents   : {len(errors)}")

    if dropped:
        print("\nSample non-verbatim drops (check for systematic issues):")
        for item in dropped[:10]:
            print(f"  {item}")

    print(f"\nMETHOD entities: {n_method}. Below ~400 in your TEST set, "
          f"confidence intervals get too wide to compare systems.")


def verify_sample(output_path: Path, n: int, seed: int = 7) -> None:
    """Write a subset to a readable file for manual checking. Correct it by
    hand; agreement between your corrections and Claude is your ceiling."""
    import random

    records = load_jsonl(output_path)
    random.Random(seed).shuffle(records)
    chosen = records[:n]

    txt_path = output_path.with_name(output_path.stem + f"_verify{n}.txt")
    jsonl_path = output_path.with_name(output_path.stem + f"_verify{n}.jsonl")

    with txt_path.open("w", encoding="utf-8") as fh:
        for i, record in enumerate(chosen, 1):
            fh.write(f"{'=' * 70}\n[{i}] {record['openalex_id']}\n\n")
            fh.write(record["text"] + "\n\n")
            for entity in record["entities"]:
                fh.write(f"  [{entity['label']:<6}] {entity['text']}\n")
            fh.write("\n")

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in chosen:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {n} records for manual verification:")
    print(f"  read   : {txt_path}")
    print(f"  correct: {jsonl_path}")
    print("Fix the entities in the .jsonl by hand, then score Claude against "
          "your corrected version. That number is your ceiling.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-sample", type=int, default=0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("Set ANTHROPIC_API_KEY")

    if args.report_only:
        report(args.output)
    else:
        run(args.input, args.output)

    if args.verify_sample:
        verify_sample(args.output, args.verify_sample)
