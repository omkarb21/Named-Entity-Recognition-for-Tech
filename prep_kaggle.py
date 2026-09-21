"""
Sample the reduced training pool and stage everything Kaggle needs.

Run from D:\\Annotation Script

    uv run prep_kaggle.py --subset 1500

Produces data/ (for local work) and kaggle_upload/ (zip this, upload once as a
Kaggle Dataset). Nothing large or regenerable goes in the upload: no venv, no
parquet corpus, no model checkpoints.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

SUBSET_SEED = 11

DATA_DIR = Path("data")
UPLOAD_DIR = Path("kaggle_upload")

# Files Kaggle needs. (path, required)
WANTED = [
    ("data/train_pool_subset.jsonl", True),
    ("data/gold_val_annotated.jsonl", True),
    ("data/gold_test_annotated.jsonl", True),
    ("data/scierc_train.jsonl", False),
    ("data/scierc_dev.jsonl", False),
    ("data/scierc_test.jsonl", False),
    ("data/scierc_gold.jsonl", False),
    ("data/scierc_claude.jsonl", False),
]


def make_subset(pool_path: Path, n: int, out_path: Path) -> None:
    df = pd.read_parquet(pool_path)
    n = min(n, len(df))
    subset = df.sample(n=n, random_state=SUBSET_SEED)
    cols = ["openalex_id", "text", "publication_year"]
    with out_path.open("w", encoding="utf-8") as fh:
        for record in subset[cols].to_dict(orient="records"):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  {out_path}  ({n} of {len(df)} pool records)")
    print(f"    -> annotate this with Haiku, output data/train_annotated.jsonl")


def stage() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    staged, missing = [], []

    for rel, required in WANTED:
        src = Path(rel)
        if src.exists():
            dst = UPLOAD_DIR / src.name
            shutil.copy2(src, dst)
            size_kb = dst.stat().st_size / 1024
            staged.append((dst.name, size_kb))
        elif required:
            missing.append(rel)

    # Also stage the annotated training file under its expected name.
    annotated = Path("data/train_annotated.jsonl")
    if annotated.exists():
        shutil.copy2(annotated, UPLOAD_DIR / annotated.name)
        staged.append((annotated.name,
                       (UPLOAD_DIR / annotated.name).stat().st_size / 1024))
    else:
        missing.append("data/train_annotated.jsonl (run the annotator first)")

    print("\nStaged in kaggle_upload/:")
    total = 0.0
    for name, size in sorted(staged):
        print(f"  {name:<34} {size:>9.0f} KB")
        total += size
    print(f"  {'TOTAL':<34} {total:>9.0f} KB")

    if missing:
        print("\nMissing (upload will be incomplete):")
        for item in missing:
            print(f"  {item}")

    manifest = {
        "files": [name for name, _ in staged],
        "note": "Upload this folder as a Kaggle Dataset. In a notebook the "
                "files appear at /kaggle/input/<dataset-slug>/",
    }
    (UPLOAD_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("weak_train_pool.parquet"))
    parser.add_argument("--subset", type=int, default=1500)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if not args.stage_only:
        make_subset(args.pool, args.subset, DATA_DIR / "train_pool_subset.jsonl")

    stage()
