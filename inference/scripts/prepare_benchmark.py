#!/usr/bin/env python3
"""Convert CSV/JSON/JSONL motion metadata into Kimodo benchmark layout."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--id-field", default="motion_id")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--frames-field", default="num_frames")
    parser.add_argument("--category-field", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise TypeError("JSON input must contain a list of objects")
        return value
    raise ValueError(f"Unsupported input format: {path.suffix}")


def safe_component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.@-]+", "_", value).strip("_")
    return cleaned[:120] or fallback


def require(row: dict[str, Any], field: str, index: int) -> Any:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Row {index} has no value for {field!r}")
    return value


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input.resolve())
    if not rows:
        raise ValueError("Input contains no samples")
    if args.fps <= 0 or args.max_frames <= 0 or args.diffusion_steps <= 0:
        raise ValueError("fps, max-frames and diffusion-steps must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing benchmark: {manifest_path}"
        )

    manifest: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_sample_ids: set[str] = set()
    capped = 0
    for index, row in enumerate(rows):
        motion_id = str(require(row, args.id_field, index)).strip()
        if motion_id in seen_ids:
            raise ValueError(f"Duplicate motion id: {motion_id}")
        seen_ids.add(motion_id)
        text = str(require(row, args.text_field, index)).strip()
        original_frames = int(require(row, args.frames_field, index))
        if original_frames <= 0:
            raise ValueError(f"Row {index} has invalid frame count: {original_frames}")
        inference_frames = min(original_frames, args.max_frames)
        capped += int(inference_frames != original_frames)

        category = args.split
        if args.category_field:
            category = str(require(row, args.category_field, index)).strip()
        category_dir = safe_component(category, args.split)
        sample_id = safe_component(motion_id, f"sample_{index:06d}")
        if sample_id in seen_sample_ids:
            sample_id = f"{index:06d}_{sample_id}"
        seen_sample_ids.add(sample_id)
        relative_dir = Path(category_dir) / "text2motion" / "overview" / sample_id
        sample_dir = output_dir / relative_dir
        sample_dir.mkdir(parents=True, exist_ok=False)

        meta = {
            "text": text,
            "duration": inference_frames / args.fps,
            "seed": args.seed + index,
            "diffusion_steps": args.diffusion_steps,
            "num_samples": 1,
        }
        source_meta = {
            **row,
            "motion_id": motion_id,
            "text": text,
            "category": category,
            "original_num_frames": original_frames,
            "inference_frames": inference_frames,
            "fps": args.fps,
            "duration_capped": inference_frames != original_frames,
        }
        (sample_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (sample_dir / "source_meta.json").write_text(
            json.dumps(source_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest.append(
            {
                "dataset_index": index,
                "sample_id": sample_id,
                "motion_id": motion_id,
                "category": category,
                "text": text,
                "original_num_frames": original_frames,
                "inference_frames": inference_frames,
                "fps": args.fps,
                "seed": args.seed + index,
                "diffusion_steps": args.diffusion_steps,
                "relative_sample_dir": relative_dir.as_posix(),
            }
        )

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "complete",
        "source": str(args.input.resolve()),
        "output_dir": str(output_dir),
        "num_samples": len(manifest),
        "fps": args.fps,
        "max_frames": args.max_frames,
        "num_capped_samples": capped,
        "diffusion_steps": args.diffusion_steps,
        "manifest": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
