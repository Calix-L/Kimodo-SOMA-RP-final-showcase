#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


HOVER_CATEGORIES = {"true_hover", "ground_to_hover", "hover_to_ground"}


def eval_category(category: str) -> str:
    return "hover" if category in HOVER_CATEGORIES else category


def safe_name(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.@-]+", "_", value).strip("_")
    return value[:96] or fallback


def read_index(feature_cache: Path) -> list[dict]:
    rows = []
    with (feature_cache / "index.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-cache", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--count-per-category", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = read_index(args.feature_cache)
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[eval_category(row.get("category", ""))].append(row)

    rng = random.Random(args.seed)
    manifest = []
    for category in sorted(by_cat):
        candidates = list(by_cat[category])
        rng.shuffle(candidates)
        selected = candidates[: args.count_per_category]
        for i, row in enumerate(selected):
            sample_id = f"{i:04d}_{safe_name(row['motion_id'][:10], 'sample')}"
            out_dir = args.output_root / safe_name(category, "unknown") / "text2motion" / "overview" / sample_id
            out_dir.mkdir(parents=True, exist_ok=True)
            duration = float(row["num_frames"]) / 30.0
            meta = {
                "text": row.get("text", ""),
                "duration": duration,
                "seed": args.seed + len(manifest),
                "diffusion_steps": 100,
                "num_samples": 1,
            }
            source_meta = {
                "motion_id": row.get("motion_id", ""),
                "source_key": row.get("source_key", ""),
                "source_shard": row.get("source_shard", ""),
                "category": row.get("category", ""),
                "eval_category": category,
                "filename": row.get("filename", ""),
                "source_gltf_path": row.get("source_gltf_path", ""),
                "text": row.get("text", ""),
                "num_frames": row.get("num_frames", 0),
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "source_meta.json").write_text(
                json.dumps(source_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest.append(
                {
                    "sample_id": sample_id,
                    "category": row.get("category", ""),
                    "eval_category": category,
                    "motion_id": row.get("motion_id", ""),
                    "text": row.get("text", ""),
                    "duration": duration,
                    "sample_dir": str(out_dir),
                }
            )

    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(args.output_root), "samples": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
