#!/usr/bin/env python3
"""Build a canonical evaluation manifest from Kimodo overview outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "dataset_index",
    "key",
    "sample_id",
    "split",
    "prompt",
    "num_frames",
    "fps",
    "seed",
    "model",
    "base_model",
    "denoiser_checkpoint",
    "diffusion_steps",
    "postprocess",
    "prediction_npz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--base-model", default="Kimodo-SOMA-RP-v1.1")
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--expected-count", type=int)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError("Source manifest must contain a list")
    return value


def index_generated_outputs(root: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for path in root.rglob("motion.npz"):
        sample_id = path.parent.name
        if sample_id in outputs:
            raise ValueError(
                f"Duplicate generated sample directory {sample_id!r}: "
                f"{outputs[sample_id]} and {path}"
            )
        outputs[sample_id] = path.resolve()
    return outputs


def value(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        item = row.get(name)
        if item is not None and str(item).strip() != "":
            return item
    return default


def main() -> int:
    args = parse_args()
    rows = read_manifest(args.source_manifest.resolve())
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} source rows, found {len(rows)}"
        )
    generated = index_generated_outputs(args.generated_root.resolve())
    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    missing: list[str] = []

    for fallback_index, source in enumerate(rows):
        sample_id = str(
            value(source, "sample_id", default=Path(str(source.get("sample_dir", ""))).name)
        )
        if not sample_id:
            raise ValueError(f"Source row {fallback_index} has no sample_id/sample_dir")
        motion_path = generated.get(sample_id)
        if motion_path is None:
            missing.append(sample_id)
            continue
        key = str(value(source, "motion_id", "key", "sample_id", default=sample_id))
        if key in seen_keys:
            raise ValueError(f"Duplicate evaluation key: {key}")
        seen_keys.add(key)
        prompt = str(value(source, "text", "prompt", "description_active_en", default=""))
        if not prompt:
            raise ValueError(f"Source row {fallback_index} has no prompt text")
        num_frames = int(
            value(
                source,
                "inference_frames",
                "num_frames",
                "target_frames",
                default=0,
            )
        )
        if num_frames <= 0:
            raise ValueError(f"Source row {fallback_index} has invalid num_frames")
        records.append(
            {
                "dataset_index": int(value(source, "dataset_index", default=fallback_index)),
                "key": key,
                "sample_id": sample_id,
                "split": str(value(source, "category", "split", default=args.split)),
                "prompt": prompt,
                "num_frames": num_frames,
                "fps": float(value(source, "fps", "inference_fps", default=30.0)),
                "seed": int(value(source, "seed", default=20260818 + fallback_index)),
                "model": args.model,
                "base_model": args.base_model,
                "denoiser_checkpoint": args.checkpoint,
                "diffusion_steps": int(
                    value(source, "diffusion_steps", default=args.diffusion_steps)
                ),
                "postprocess": args.postprocess,
                "prediction_npz": str(motion_path),
            }
        )

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} generated motions; first IDs: {missing[:10]}"
        )
    if args.expected_count is not None and len(records) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} outputs, found {len(records)}")
    if not records:
        raise ValueError("No generated motions found")
    records.sort(key=lambda item: int(item["dataset_index"]))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "status": "complete",
        "num_samples": len(records),
        "generated_root": str(args.generated_root.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "model": args.model,
        "diffusion_steps": args.diffusion_steps,
        "postprocess": args.postprocess,
        "manifest": str(output_dir / "manifest.csv"),
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
