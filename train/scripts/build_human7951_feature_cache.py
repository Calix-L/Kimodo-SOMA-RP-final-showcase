#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


def clean_id(value: str) -> str:
    chars = []
    for ch in str(value):
        chars.append(ch if ch.isalnum() or ch in "-_" else "_")
    return "".join(chars).strip("_") or "sample"


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    by_key = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = row.get("key") or row.get("\ufeffkey")
            if key:
                by_key[key] = row
    return by_key


def read_tar_records(tar_path: Path) -> list[tuple[str, str, str]]:
    records: dict[str, dict[str, str]] = {}
    with tarfile.open(tar_path, "r") as tf:
        for name in tf.getnames():
            if name.endswith(".motion.npy"):
                key = name[: -len(".motion.npy")]
                records.setdefault(key, {})["motion"] = name
            elif name.endswith(".meta.json"):
                key = name[: -len(".meta.json")]
                records.setdefault(key, {})["meta"] = name
    out = []
    for key, parts in sorted(records.items()):
        if "motion" in parts and "meta" in parts:
            out.append((key, parts["motion"], parts["meta"]))
    return out


def load_json_member(tf: tarfile.TarFile, member: str) -> dict[str, Any]:
    f = tf.extractfile(member)
    if f is None:
        raise FileNotFoundError(member)
    return json.load(f)


def load_motion_member(tf: tarfile.TarFile, member: str) -> np.ndarray:
    f = tf.extractfile(member)
    if f is None:
        raise FileNotFoundError(member)
    arr = np.load(io.BytesIO(f.read()))
    if arr.ndim != 2 or arr.shape[1] != 369:
        raise ValueError(f"Expected motion feature shape [T,369], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Motion contains non-finite values")
    return arr.astype("float32", copy=False)


def choose_text(meta: dict[str, Any], csv_row: dict[str, str] | None) -> str:
    if csv_row:
        for key in ("description_active_en", "description_original_en"):
            text = (csv_row.get(key) or "").strip()
            if text:
                return text
    descriptions = meta.get("descriptions") or []
    if descriptions:
        return str(descriptions[0]).strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--out-cache", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dataset_root = args.dataset_root
    shard_dir = dataset_root / "dataset"
    metadata_path = dataset_root / "metadata" / "retarget_v6_human7951_metadata.csv"
    if not shard_dir.is_dir():
        raise FileNotFoundError(shard_dir)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    if args.out_cache.exists() and args.overwrite:
        shutil.rmtree(args.out_cache)
    features_dir = args.out_cache / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    metadata_by_key = load_metadata(metadata_path)
    shards = sorted(shard_dir.glob("train-*.tar")) + sorted(shard_dir.glob("val-*.tar"))
    index_rows = []
    errors = []

    for shard in tqdm(shards, desc="shards"):
        records = read_tar_records(shard)
        with tarfile.open(shard, "r") as tf:
            for key, motion_member, meta_member in tqdm(records, desc=shard.name, leave=False):
                try:
                    meta = load_json_member(tf, meta_member)
                    arr = load_motion_member(tf, motion_member)
                    csv_row = metadata_by_key.get(key)
                    rel = Path("features") / f"{len(index_rows):06d}_{clean_id(key)}.pt"
                    torch.save({"features": torch.from_numpy(arr)}, args.out_cache / rel)
                    text = choose_text(meta, csv_row)
                    index_rows.append(
                        {
                            "motion_id": key,
                            "feature_file": str(rel),
                            "text": text,
                            "num_frames": int(arr.shape[0]),
                            "split": meta.get("split", ""),
                            "filename": meta.get("filename", ""),
                            "source_shard": str(shard),
                            "source_key": key,
                            "motion_regime_category": (csv_row or {}).get("motion_regime_category", ""),
                            "height_bin": (csv_row or {}).get("height_bin", ""),
                            "speed_bin": (csv_row or {}).get("speed_bin", ""),
                            "sample_weight": 1.0,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append({"shard": str(shard), "key": key, "error": repr(exc)})

    with (args.out_cache / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "dataset_root": str(dataset_root),
        "metadata": str(metadata_path),
        "shards": [str(x) for x in shards],
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


if __name__ == "__main__":
    main()

