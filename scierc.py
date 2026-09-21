"""
SciERC: an external, human-annotated yardstick for the Claude annotator.

Run from D:\\Annotation Script

  Step 1 - download and convert (once):
      uv run scierc.py --prepare

  Step 2 - annotate SciERC's abstracts with the same annotator you used for gold:
      uv run annotate_with_claude.py --input  data/scierc_input.jsonl \
                                     --output data/scierc_claude.jsonl

  Step 3 - score Claude against the human labels:
      uv run scierc.py --score --gold data/scierc_gold.jsonl \
                               --pred data/scierc_claude.jsonl

The number from step 3 is the ceiling you report alongside your model's F1.

CAVEAT worth writing into the paper: SciERC's Method/Task definitions are not
identical to your guideline (it has six types; Material, Metric,
OtherScientificTerm and Generic are dropped here). So this is a LOWER BOUND on
annotation quality, not an exact agreement figure. Say that plainly and it is
a strength rather than a weakness.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path

SCIERC_URL = "http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"

# SciERC type -> our label. Everything else is dropped.
TYPE_MAP = {"Method": "METHOD", "Task": "TASK"}

ENTITY_TYPES = ["METHOD", "TASK"]


# --------------------------------------------------------------------------
# Prepare
# --------------------------------------------------------------------------

def convert_doc(doc: dict) -> dict:
    """dygie format -> our {text, entities:[{start,end,label}]} format.

    SciERC stores token indices that are GLOBAL across the document, with an
    INCLUSIVE end index. Tokens are joined with single spaces; that is the
    canonical text for these records.
    """
    tokens = [tok for sentence in doc["sentences"] for tok in sentence]

    offsets, pos = [], 0
    for token in tokens:
        offsets.append((pos, pos + len(token)))
        pos += len(token) + 1
    text = " ".join(tokens)

    entities = []
    for sentence_ner in doc.get("ner", []):
        for start_tok, end_tok, label in sentence_ner:
            our_label = TYPE_MAP.get(label)
            if our_label is None:
                continue
            if start_tok >= len(offsets) or end_tok >= len(offsets):
                continue
            start = offsets[start_tok][0]
            end = offsets[end_tok][1]
            entities.append({"start": start, "end": end,
                             "label": our_label, "text": text[start:end]})

    entities.sort(key=lambda e: e["start"])
    return {"openalex_id": f"scierc:{doc['doc_key']}", "text": text,
            "publication_year": None, "entities": entities}


def prepare(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SCIERC_URL}")
    raw = urllib.request.urlopen(SCIERC_URL, timeout=120).read()

    docs = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name
            if "processed_data/json/" not in name or not name.endswith(".json"):
                continue
            split = Path(name).stem              # train / dev / test
            handle = tar.extractfile(member)
            if handle is None:
                continue
            for line in handle.read().decode("utf-8").splitlines():
                if line.strip():
                    docs.append((split, convert_doc(json.loads(line))))

    by_split: dict[str, list[dict]] = {}
    for split, doc in docs:
        by_split.setdefault(split, []).append(doc)

    all_docs = []
    for split, records in sorted(by_split.items()):
        path = out_dir / f"scierc_{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        n_ent = sum(len(r["entities"]) for r in records)
        print(f"  {path}  {len(records)} docs, {n_ent} entities "
              f"({n_ent / max(len(records), 1):.1f}/doc)")
        all_docs.extend(records)

    # The combined gold file, and the same documents stripped of labels for
    # feeding to the annotator.
    gold_path = out_dir / "scierc_gold.jsonl"
    input_path = out_dir / "scierc_input.jsonl"
    with gold_path.open("w", encoding="utf-8") as g, \
         input_path.open("w", encoding="utf-8") as i:
        for record in all_docs:
            g.write(json.dumps(record, ensure_ascii=False) + "\n")
            i.write(json.dumps({k: record[k] for k in
                                ("openalex_id", "text", "publication_year")},
                               ensure_ascii=False) + "\n")

    n_method = sum(1 for r in all_docs for e in r["entities"]
                   if e["label"] == "METHOD")
    n_task = sum(1 for r in all_docs for e in r["entities"]
                 if e["label"] == "TASK")
    print(f"\n  {gold_path}   {len(all_docs)} docs")
    print(f"  {input_path}  (feed this to annotate_with_claude.py)")
    print(f"  METHOD {n_method}   TASK {n_task}")


# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------

def load(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                records[record["openalex_id"]] = record
    return records


def spans_of(record: dict, etype: str | None) -> list[tuple[int, int, str]]:
    return [(e["start"], e["end"], e["label"])
            for e in record.get("entities", [])
            if e["label"] in ENTITY_TYPES and (etype is None or e["label"] == etype)]


def score(gold_records, pred_records, etype=None) -> dict:
    strict_tp = partial_tp = n_pred = n_gold = 0

    for key, gold_record in gold_records.items():
        pred_record = pred_records.get(key)
        if pred_record is None:
            n_gold += len(spans_of(gold_record, etype))
            continue

        gold = spans_of(gold_record, etype)
        pred = spans_of(pred_record, etype)
        n_gold += len(gold)
        n_pred += len(pred)
        strict_tp += len(set(gold) & set(pred))

        unmatched = list(gold)
        for p_start, p_end, p_type in pred:
            for idx, (g_start, g_end, g_type) in enumerate(unmatched):
                if g_type == p_type and p_start < g_end and p_end > g_start:
                    partial_tp += 1
                    unmatched.pop(idx)
                    break

    def prf(tp):
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gold if n_gold else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    sp, sr, sf = prf(strict_tp)
    pp, pr, pf = prf(partial_tp)
    return {"strict_p": sp, "strict_r": sr, "strict_f1": sf,
            "partial_p": pp, "partial_r": pr, "partial_f1": pf,
            "n_gold": n_gold, "n_pred": n_pred}


def run_score(gold_path: Path, pred_path: Path) -> None:
    gold_records = load(gold_path)
    pred_records = load(pred_path)
    overlap = set(gold_records) & set(pred_records)
    print(f"Gold {len(gold_records)} docs, predictions {len(pred_records)}, "
          f"scored on {len(overlap)}\n")

    print(f"{'type':<8} {'P':>7} {'R':>7} {'strict F1':>11} {'partial F1':>12} "
          f"{'gold':>7} {'pred':>7}")
    for etype in [None] + ENTITY_TYPES:
        res = score(gold_records, pred_records, etype)
        name = etype or "ALL"
        print(f"{name:<8} {res['strict_p']:>7.3f} {res['strict_r']:>7.3f} "
              f"{res['strict_f1']:>11.3f} {res['partial_f1']:>12.3f} "
              f"{res['n_gold']:>7} {res['n_pred']:>7}")

    method = score(gold_records, pred_records, "METHOD")
    print(f"\nCEILING for your write-up:")
    print(f"  Claude vs expert human annotation on SciERC (METHOD):")
    print(f"    strict F1  {method['strict_f1']:.3f}")
    print(f"    partial F1 {method['partial_f1']:.3f}")
    print("\n  Report your model's test F1 as a fraction of this. A large gap "
          "between strict and partial means boundary conventions differ, "
          "which is expected across annotation schemes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("data/scierc_gold.jsonl"))
    parser.add_argument("--pred", type=Path, default=Path("data/scierc_claude.jsonl"))
    args = parser.parse_args()

    if args.prepare:
        prepare(args.out_dir)
    elif args.score:
        run_score(args.gold, args.pred)
    else:
        parser.print_help()
