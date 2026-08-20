#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(index_path: Path) -> list[dict]:
    rows = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_split(root: Path, name: str, rows: list[dict], source_features: Path) -> None:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    features_link = out / "features"
    if not features_link.exists():
        features_link.symlink_to(source_features)
    with (out / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            row = dict(row)
            row["split"] = name
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-cache", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    rows = read_rows(args.source_cache / "index.jsonl")
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cat[str(row.get("category", "unknown"))].append(row)

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}
    for category in sorted(by_cat):
        items = list(by_cat[category])
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * 0.1))
        n_val = max(1, round(n * 0.1))
        n_train = n - n_val - n_test
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    for name in splits:
        rng.shuffle(splits[name])

    args.output_root.mkdir(parents=True, exist_ok=True)
    source_features = (args.source_cache / "features").resolve()
    for name, split_rows in splits.items():
        write_split(args.output_root, name, split_rows, source_features)

    summary = {
        "source_cache": str(args.source_cache),
        "output_root": str(args.output_root),
        "total": len(rows),
        "counts": {k: len(v) for k, v in splits.items()},
        "by_split_category": {k: dict(Counter(r.get("category", "unknown") for r in v)) for k, v in splits.items()},
    }
    (args.output_root / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
