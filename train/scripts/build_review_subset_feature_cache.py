#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch


def clean_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value)).strip("_") or "sample"


def read_json_member(tf: tarfile.TarFile, member: str) -> dict:
    f = tf.extractfile(member)
    if f is None:
        raise FileNotFoundError(member)
    return json.load(f)


def read_motion_member(tf: tarfile.TarFile, member: str) -> np.ndarray:
    f = tf.extractfile(member)
    if f is None:
        raise FileNotFoundError(member)
    arr = np.load(io.BytesIO(f.read()))
    if arr.ndim != 2 or arr.shape[1] != 369:
        raise ValueError(f"Expected [T, 369], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Motion contains non-finite values")
    return arr.astype("float32", copy=False)


def load_keep(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_audit(path: Path) -> dict[str, dict]:
    by_asset = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            asset_id = row.get("asset_id", "")
            if asset_id:
                by_asset[asset_id] = row
    return by_asset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--audit-csv", required=True, type=Path)
    parser.add_argument("--keep-jsonl", required=True, type=Path)
    parser.add_argument("--out-cache", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out_cache.exists() and args.overwrite:
        shutil.rmtree(args.out_cache)
    features_dir = args.out_cache / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    keep_rows = load_keep(args.keep_jsonl)
    audit_by_asset = load_audit(args.audit_csv)

    index_rows = []
    errors = []
    for keep in keep_rows:
        original_id = keep["original_id"]
        audit = audit_by_asset.get(original_id)
        if audit is None:
            errors.append({"original_id": original_id, "error": "missing audit row"})
            continue

        key = audit["sample_key"]
        shard = args.dataset_root / "dataset" / audit["source_tar"]
        motion_member = f"{key}.motion.npy"
        meta_member = f"{key}.meta.json"
        try:
            with tarfile.open(shard, "r") as tf:
                meta = read_json_member(tf, meta_member)
                arr = read_motion_member(tf, motion_member)
            rel = Path("features") / f"{len(index_rows):06d}_{clean_id(key)}.pt"
            torch.save({"features": torch.from_numpy(arr)}, args.out_cache / rel)
            index_rows.append(
                {
                    "motion_id": key,
                    "feature_file": str(rel),
                    "text": keep["text"],
                    "num_frames": int(arr.shape[0]),
                    "split": audit.get("split", meta.get("split", "")),
                    "filename": original_id,
                    "source_shard": str(shard),
                    "source_key": key,
                    "sequence_no": keep.get("sequence_no", ""),
                    "review_category": keep.get("category", ""),
                    "manual_decision": keep.get("manual_decision", ""),
                    "motion_regime_category": audit.get("motion_category", ""),
                    "sample_weight": 1.0,
                    "text_source": "csv:description_zh_edited",
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"original_id": original_id, "sample_key": key, "shard": str(shard), "error": repr(exc)})

    with (args.out_cache / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "dataset_root": str(args.dataset_root),
        "audit_csv": str(args.audit_csv),
        "keep_jsonl": str(args.keep_jsonl),
        "out_cache": str(args.out_cache),
        "filter_rule": "manual_decision != reject",
        "text_source": "CSV description_zh_edited",
        "requested_rows": len(keep_rows),
        "written_rows": len(index_rows),
        "error_count": len(errors),
        "frame_min": min((x["num_frames"] for x in index_rows), default=None),
        "frame_max": max((x["num_frames"] for x in index_rows), default=None),
    }
    (args.out_cache / "cache_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        with (args.out_cache / "cache_errors.jsonl").open("w", encoding="utf-8") as f:
            for row in errors:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
