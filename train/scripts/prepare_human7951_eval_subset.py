#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import shutil
import tarfile
from pathlib import Path


def safe_name(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.@-]+", "_", value).strip("_")
    return value[:96] or fallback


def read_json_from_tar(shard: Path, member: str) -> dict:
    with tarfile.open(shard) as tar:
        f = tar.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"{member} not found in {shard}")
        return json.load(f)


def load_metadata(csv_path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["key"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_csv)
    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(args.seed)
    selected = rng.sample(rows, args.count)

    manifest = []
    for i, row in enumerate(selected):
        key = row["source_key"]
        shard = Path(row["source_shard"])
        source_meta = read_json_from_tar(shard, f"{key}.meta.json")
        meta_row = metadata.get(key, {})

        text = row.get("text") or source_meta.get("descriptions", [""])[0]
        frames = int(source_meta.get("source_frames") or source_meta.get("conversion", {}).get("target_frames") or row["num_frames"])
        fps = float(source_meta.get("target_fps") or meta_row.get("target_fps") or 30)
        duration = frames / fps

        category = safe_name(row.get("motion_regime_category", "unknown"), "unknown")
        take = safe_name(source_meta.get("take_name") or meta_row.get("take_name") or row["filename"], "take")
        sample_id = f"{i:04d}_{key[:10]}"
        out_dir = args.output_root / category / "text2motion" / "overview" / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "text": text,
            "duration": duration,
            "seed": args.seed + i,
            "diffusion_steps": 100,
            "num_samples": 1,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "source_meta.json").write_text(
            json.dumps(source_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        manifest.append(
            {
                "sample_id": sample_id,
                "category": category,
                "take_name": take,
                "motion_id": key,
                "filename": row["filename"],
                "source_shard": str(shard),
                "source_gltf_path": source_meta.get("source_gltf_path"),
                "text": text,
                "duration": duration,
                "sample_dir": str(out_dir),
            }
        )

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
