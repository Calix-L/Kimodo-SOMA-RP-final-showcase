#!/usr/bin/env python3
"""Validate a canonical Kimodo manifest and every referenced motion.npz file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = ("posed_joints", "foot_contacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def resolve_prediction(manifest: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else manifest.parent / path


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} rows, found {len(rows)}")
    if not rows:
        raise ValueError("Manifest contains no rows")

    keys: set[str] = set()
    indices: set[int] = set()
    frame_mismatches = 0
    frame_min: int | None = None
    frame_max = 0
    for row_number, row in enumerate(rows, start=2):
        key = row["key"]
        index = int(row["dataset_index"])
        if key in keys:
            raise ValueError(f"Duplicate key at CSV row {row_number}: {key}")
        if index in indices:
            raise ValueError(f"Duplicate dataset_index at CSV row {row_number}: {index}")
        keys.add(key)
        indices.add(index)
        path = resolve_prediction(manifest, row["prediction_npz"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction for {key}: {path}")
        with np.load(path, allow_pickle=False) as motion:
            missing = [name for name in REQUIRED_ARRAYS if name not in motion]
            if missing:
                raise KeyError(f"{key}: missing arrays {missing}")
            joints = np.asarray(motion["posed_joints"])
            contacts = np.asarray(motion["foot_contacts"])
            if joints.ndim != 3 or joints.shape[-1] != 3:
                raise ValueError(f"{key}: invalid posed_joints shape {joints.shape}")
            if contacts.ndim != 2 or contacts.shape[0] != joints.shape[0]:
                raise ValueError(
                    f"{key}: incompatible foot_contacts shape {contacts.shape}"
                )
            if not np.isfinite(joints).all():
                raise ValueError(f"{key}: posed_joints contains NaN/Inf")
            frames = int(joints.shape[0])
        expected_frames = int(row["num_frames"])
        frame_mismatches += int(frames != expected_frames)
        frame_min = frames if frame_min is None else min(frame_min, frames)
        frame_max = max(frame_max, frames)

    summary = {
        "status": "valid",
        "manifest": str(manifest),
        "num_samples": len(rows),
        "num_unique_keys": len(keys),
        "num_unique_indices": len(indices),
        "frame_min": frame_min,
        "frame_max": frame_max,
        "frame_count_mismatches": frame_mismatches,
        "required_arrays": list(REQUIRED_ARRAYS),
    }
    report = args.report.resolve() if args.report else manifest.parent / "validation.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
