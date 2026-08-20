#!/usr/bin/env python3
"""Evaluate Kimodo outputs with TMR retrieval and official physical metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kimodo-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tasks", choices=("all", "tmr", "physical"), default="all")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--text-encoders-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-batch-size", type=int, default=32)
    parser.add_argument("--motion-batch-size", type=int, default=1)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--tmr-max-frames", type=int, default=300)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_progress(path: Path, stage: str, processed: int, total: int, started: float) -> None:
    event = {
        "stage": stage,
        "processed": processed,
        "total": total,
        "elapsed_seconds": time.monotonic() - started,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps(event), flush=True)


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("Batch sizes must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    records.sort(key=lambda row: int(row["dataset_index"]))
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("No samples selected")
    return records


def prediction_path(manifest: Path, record: dict[str, Any]) -> Path:
    path = Path(record["prediction_npz"])
    return path if path.is_absolute() else manifest.parent / path


def unique_value(records: list[dict[str, Any]], field: str, cast=str) -> Any:
    values = {cast(record[field]) for record in records}
    if len(values) != 1:
        raise ValueError(f"Mixed {field} values: {sorted(values)}")
    return next(iter(values))


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def pad_motion_batch(
    manifest: Path,
    records: list[dict[str, Any]],
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    joint_arrays: list[np.ndarray] = []
    contact_arrays: list[np.ndarray] = []
    lengths: list[int] = []
    cropped = 0
    for record in records:
        with np.load(prediction_path(manifest, record), allow_pickle=False) as motion:
            joints = np.asarray(motion["posed_joints"], dtype=np.float32)
            contacts = np.asarray(motion["foot_contacts"], dtype=np.bool_)
        if max_frames is not None and len(joints) > max_frames:
            joints = joints[:max_frames]
            contacts = contacts[:max_frames]
            cropped += 1
        if joints.ndim != 3 or joints.shape[-1] != 3:
            raise ValueError(f"{record['key']}: bad posed_joints shape {joints.shape}")
        if contacts.ndim != 2 or len(contacts) != len(joints):
            raise ValueError(f"{record['key']}: bad foot_contacts shape {contacts.shape}")
        joint_arrays.append(joints)
        contact_arrays.append(contacts)
        lengths.append(len(joints))

    max_length = max(lengths)
    num_joints = joint_arrays[0].shape[1]
    num_contacts = contact_arrays[0].shape[1]
    joints_batch = np.zeros(
        (len(records), max_length, num_joints, 3), dtype=np.float32
    )
    contacts_batch = np.zeros(
        (len(records), max_length, num_contacts), dtype=np.bool_
    )
    for index, (joints, contacts, length) in enumerate(
        zip(joint_arrays, contact_arrays, lengths)
    ):
        if joints.shape[1] != num_joints or contacts.shape[1] != num_contacts:
            raise ValueError("Inconsistent joint/contact dimensions in one batch")
        joints_batch[index, :length] = joints
        contacts_batch[index, :length] = contacts
    return (
        joints_batch,
        contacts_batch,
        np.asarray(lengths, dtype=np.int64),
        cropped,
    )


def statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def evaluate_physical(
    manifest: Path,
    records: list[dict[str, Any]],
    output_dir: Path,
    device: str,
    batch_size: int,
    progress_path: Path,
) -> dict[str, Any]:
    from kimodo.metrics import FootContactConsistency, FootSkateFromHeight, FootSkateRatio
    from kimodo.skeleton import build_skeleton

    fps = unique_value(records, "fps", float)
    skeleton = build_skeleton(77).to(device)
    metric_objects = [
        FootContactConsistency(skeleton=skeleton, fps=fps),
        FootSkateRatio(skeleton=skeleton, fps=fps),
        FootSkateFromHeight(skeleton=skeleton, fps=fps),
    ]
    collected: dict[str, list[np.ndarray]] = {
        "foot_contact_consistency": [],
        "foot_skate_ratio": [],
        "foot_skate_from_height": [],
    }
    processed = 0
    started = time.monotonic()
    for batch_records in batched(records, batch_size):
        joints_np, contacts_np, lengths_np, _ = pad_motion_batch(manifest, batch_records)
        joints = torch.from_numpy(joints_np).to(device)
        contacts = torch.from_numpy(contacts_np).to(device)
        lengths = torch.from_numpy(lengths_np).to(device)
        with torch.inference_mode():
            for metric in metric_objects:
                result = metric(
                    posed_joints=joints, foot_contacts=contacts, lengths=lengths
                )
                for key, tensor in result.items():
                    collected[key].append(tensor.detach().cpu().numpy())
        processed += len(batch_records)
        if processed == len(records) or processed % (batch_size * 10) == 0:
            append_progress(progress_path, "physical", processed, len(records), started)

    arrays = {key: np.concatenate(parts) for key, parts in collected.items()}
    summary = {
        "num_samples": len(records),
        "fps": fps,
        "contact": statistics(arrays["foot_contact_consistency"]),
        "foot_skate_ratio": statistics(arrays["foot_skate_ratio"]),
        "foot_skate_height_m_per_s": statistics(arrays["foot_skate_from_height"]),
    }
    atomic_json(output_dir / "physical_summary.json", summary)
    with (output_dir / "physical_per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_index",
                "key",
                "contact",
                "foot_skate_ratio",
                "foot_skate_height_m_per_s",
            ],
        )
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "dataset_index": record["dataset_index"],
                    "key": record["key"],
                    "contact": float(arrays["foot_contact_consistency"][index]),
                    "foot_skate_ratio": float(arrays["foot_skate_ratio"][index]),
                    "foot_skate_height_m_per_s": float(
                        arrays["foot_skate_from_height"][index]
                    ),
                }
            )
    return summary


def evaluate_tmr(
    manifest: Path,
    records: list[dict[str, Any]],
    output_dir: Path,
    checkpoint_root: Path,
    text_encoders_dir: Path,
    device: str,
    motion_batch_size: int,
    text_batch_size: int,
    max_frames: int,
    progress_path: Path,
) -> dict[str, Any]:
    from kimodo.metrics import compute_tmr_retrieval_metrics
    from kimodo.model.load_model import load_model
    from kimodo.skeleton import build_skeleton

    os.environ.update(
        {
            "CHECKPOINT_DIR": str(checkpoint_root.resolve()),
            "TEXT_ENCODER_MODE": "local",
            "TEXT_ENCODERS_DIR": str(text_encoders_dir.resolve()),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    model = load_model(
        modelname="TMR-SOMA-RP-v1", device=device, default_family="TMR"
    )
    model.eval()
    skeleton = build_skeleton(77).to(device)

    motion_parts: list[np.ndarray] = []
    cropped_total = 0
    processed = 0
    started = time.monotonic()
    for batch_records in batched(records, motion_batch_size):
        joints_np, _, lengths_np, cropped = pad_motion_batch(
            manifest, batch_records, max_frames=max_frames
        )
        with torch.inference_mode():
            embeddings = model.encode_motion(
                torch.from_numpy(joints_np).to(device),
                original_skeleton=skeleton,
                lengths=torch.from_numpy(lengths_np).to(device),
                unit_vector=True,
            )
        motion_parts.append(embeddings.detach().cpu().float().numpy())
        cropped_total += cropped
        processed += len(batch_records)
        if processed == len(records) or processed % (motion_batch_size * 10) == 0:
            append_progress(progress_path, "tmr_motion", processed, len(records), started)
    motion_embeddings = np.concatenate(motion_parts)
    np.save(output_dir / "tmr_motion_embeddings.npy", motion_embeddings)

    text_parts: list[np.ndarray] = []
    processed = 0
    started = time.monotonic()
    for batch_records in batched(records, text_batch_size):
        prompts = [record["prompt"] for record in batch_records]
        with torch.inference_mode():
            embeddings = model.encode_raw_text(prompts, unit_vector=True)
        text_parts.append(embeddings.detach().cpu().float().numpy())
        processed += len(batch_records)
        if processed == len(records) or processed % (text_batch_size * 10) == 0:
            append_progress(progress_path, "tmr_text", processed, len(records), started)
    text_embeddings = np.concatenate(text_parts)
    np.save(output_dir / "tmr_text_embeddings.npy", text_embeddings)

    official = compute_tmr_retrieval_metrics(
        motion_embeddings, text_embeddings, rounding=4
    )
    per_sample_similarity = (
        np.einsum("bi,bi->b", motion_embeddings, text_embeddings) + 1.0
    ) / 2.0
    summary = {
        "num_samples": len(records),
        "model": "TMR-SOMA-RP-v1",
        "embedding_dim": int(motion_embeddings.shape[1]),
        "tmr_max_frames": max_frames,
        "num_cropped_motions": cropped_total,
        "r_at_1_percent": float(official["TMR/t2m_R/R01"]),
        "r_at_5_percent": float(official["TMR/t2m_R/R05"]),
        "t2m_sim": statistics(per_sample_similarity),
        "official_metrics": official,
    }
    atomic_json(output_dir / "tmr_summary.json", summary)
    with (output_dir / "tmr_per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset_index", "key", "t2m_sim"]
        )
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "dataset_index": record["dataset_index"],
                    "key": record["key"],
                    "t2m_sim": float(per_sample_similarity[index]),
                }
            )
    return summary


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.tasks in {"all", "tmr"} and (
        args.checkpoint_root is None or args.text_encoders_dir is None
    ):
        raise ValueError(
            "TMR evaluation requires --checkpoint-root and --text-encoders-dir"
        )
    sys.path.insert(0, str(args.kimodo_root.resolve()))
    manifest = args.manifest.resolve()
    records = load_manifest(manifest, args.limit)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"

    metadata = {
        "evaluated_model": unique_value(records, "model"),
        "num_samples": len(records),
        "diffusion_steps": unique_value(records, "diffusion_steps", int),
        "postprocess": unique_value(records, "postprocess", parse_bool),
        "fps": unique_value(records, "fps", float),
    }
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    atomic_json(output_dir / "run_config.json", {**metadata, "args": serialized_args})
    physical = None
    tmr = None
    if args.tasks in {"all", "physical"}:
        physical = evaluate_physical(
            manifest,
            records,
            output_dir,
            args.device,
            args.physical_batch_size,
            progress_path,
        )
    if args.tasks in {"all", "tmr"}:
        tmr = evaluate_tmr(
            manifest,
            records,
            output_dir,
            args.checkpoint_root,
            args.text_encoders_dir,
            args.device,
            args.motion_batch_size,
            args.text_batch_size,
            args.tmr_max_frames,
            progress_path,
        )

    summary: dict[str, Any] = {"status": "complete", **metadata}
    if tmr:
        summary.update(
            {
                "R@1_percent": tmr["r_at_1_percent"],
                "R@5_percent": tmr["r_at_5_percent"],
                "T2M_Sim_mean": tmr["t2m_sim"]["mean"],
            }
        )
    if physical:
        summary.update(
            {
                "Contact_mean": physical["contact"]["mean"],
                "Foot_Skate_Ratio_mean": physical["foot_skate_ratio"]["mean"],
                "Foot_Skate_Height_mean_m_per_s": physical[
                    "foot_skate_height_m_per_s"
                ]["mean"],
            }
        )
    atomic_json(output_dir / "metrics_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
