"""
Create the frozen gold splits. Run this ONCE.

    python make_splits.py --corpus "D:/Abstract data New/openalex_corpus/corpus.parquet"

Outputs (in the current directory):
    gold_rule_dev.jsonl       60   read only, no annotation
    gold_val.jsonl            60   checkpoint selection
    gold_test.jsonl          120   final evaluation, untouched until the end
    weak_train_pool.parquet ~7.5k  everything else

The split is fixed by SPLIT_SEED. Never regenerate it after annotation starts:
a document that moves from train to test invalidates your evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SPLIT_SEED = 7
N_RULE_DEV, N_VAL, N_TEST = 60, 60, 120
MAX_GAPS = 5          # one record in the corpus had 3368; it is corrupt


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    cols = ["openalex_id", "text", "publication_year"]
    with path.open("w", encoding="utf-8") as fh:
        for record in df[cols].to_dict(orient="records"):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  {path}  ({len(df)} records)")


def main(corpus_path: Path, out_dir: Path) -> None:
    df = pd.read_parquet(corpus_path)
    print(f"Loaded {len(df):,} records")

    df = df[df["n_gaps"] <= MAX_GAPS].reset_index(drop=True)
    print(f"After dropping corrupt records: {len(df):,}")

    n_gold = N_RULE_DEV + N_VAL + N_TEST
    gold = df.sample(n=n_gold, random_state=SPLIT_SEED).reset_index(drop=True)

    rule_dev = gold.iloc[:N_RULE_DEV]
    val = gold.iloc[N_RULE_DEV:N_RULE_DEV + N_VAL]
    test = gold.iloc[N_RULE_DEV + N_VAL:]

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nWrote:")
    write_jsonl(rule_dev, out_dir / "gold_rule_dev.jsonl")
    write_jsonl(val, out_dir / "gold_val.jsonl")
    write_jsonl(test, out_dir / "gold_test.jsonl")

    pool = df[~df["openalex_id"].isin(gold["openalex_id"])].reset_index(drop=True)
    pool_path = out_dir / "weak_train_pool.parquet"
    pool.to_parquet(pool_path, index=False)
    print(f"  {pool_path}  ({len(pool):,} records)")

    assert not set(pool["openalex_id"]) & set(gold["openalex_id"]), "leak!"

    print("\nGold set by year:")
    print(gold["publication_year"].value_counts().sort_index().to_string())
    print("\nTest set by year:")
    print(test["publication_year"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    main(args.corpus, args.out_dir)
