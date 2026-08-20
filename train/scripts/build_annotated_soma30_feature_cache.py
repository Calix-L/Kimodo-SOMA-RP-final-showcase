#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimodo_seed_repro.kimodo_io import load_kimodo_model


def clean_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value)).strip("_") or "sample"


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
    return [
        (key, parts["motion"], parts["meta"])
        for key, parts in sorted(records.items())
        if "motion" in parts and "meta" in parts
    ]


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


def choose_text(meta: dict[str, Any]) -> str:
    for key in ("prompt", "text"):
        text = str(meta.get(key) or "").strip()
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
    ap.add_argument("--kimodo-root", required=True)
    ap.add_argument("--model", default="Kimodo-SOMA-RP-v1.1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.kimodo_root not in sys.path:
        sys.path.insert(0, args.kimodo_root)

    shard_dir = args.dataset_root / "soma30"
    if not shard_dir.is_dir():
        raise FileNotFoundError(shard_dir)

    if args.out_cache.exists() and args.overwrite:
        shutil.rmtree(args.out_cache)
    features_dir = args.out_cache / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_kimodo_model(
        args.model,
        str(device),
        kimodo_root=args.kimodo_root,
        skip_text_encoder=True,
        init_mode="pretrained",
    )
    motion_rep = model.motion_rep

    index_rows = []
    errors = []
    shards = sorted(shard_dir.glob("data-*.tar"))
    for shard in tqdm(shards, desc="shards"):
        records = read_tar_records(shard)
        with tarfile.open(shard, "r") as tf:
            for key, motion_member, meta_member in tqdm(records, desc=shard.name, leave=False):
                try:
                    meta = load_json_member(tf, meta_member)
                    arr = load_motion_member(tf, motion_member)
                    with torch.no_grad():
                        feats = motion_rep.normalize(torch.from_numpy(arr).to(device)).cpu().float()
                    rel = Path("features") / f"{len(index_rows):06d}_{clean_id(key)}.pt"
                    torch.save({"features": feats}, args.out_cache / rel)
                    motion_id = str(meta.get("key") or key)
                    index_rows.append(
                        {
                            "motion_id": motion_id,
                            "feature_file": str(rel),
                            "text": choose_text(meta),
                            "num_frames": int(arr.shape[0]),
                            "split": "train",
                            "filename": meta.get("asset_id", ""),
                            "source_shard": str(shard),
                            "source_key": key,
                            "category": meta.get("category", ""),
                            "category_zh": meta.get("category_zh", ""),
                            "source_gltf_path": meta.get("source_gltf_path", ""),
                            "sample_weight": 1.0,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append({"shard": str(shard), "key": key, "error": repr(exc)})

    with (args.out_cache / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "dataset_root": str(args.dataset_root),
        "shards": [str(x) for x in shards],
        "written_rows": len(index_rows),
        "error_count": len(errors),
        "frame_min": min((x["num_frames"] for x in index_rows), default=None),
        "frame_max": max((x["num_frames"] for x in index_rows), default=None),
        "normalized": True,
    }
    (args.out_cache / "cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if errors:
        with (args.out_cache / "cache_errors.jsonl").open("w", encoding="utf-8") as f:
            for row in errors:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
