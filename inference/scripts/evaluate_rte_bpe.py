#!/usr/bin/env python3
"""Compute paired Root Trajectory Error and Body Pose Error against Game GT."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


METRICS = ("rte", "rte_xz", "rte_y", "bpe", "bpe_upper", "bpe_lower")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kimodo-root", required=True, type=Path)
    parser.add_argument("--repro-root", required=True, type=Path)
    parser.add_argument("--feature-index", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="LABEL=MANIFEST.csv",
        help="Repeat this option to compare several models in one run.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="Kimodo-SOMA-RP-v1.1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--posed-joints-from", choices=("rotations", "positions"), default="rotations"
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_predictions(spec: str) -> tuple[str, dict[str, dict[str, str]]]:
    if "=" not in spec:
        raise ValueError(f"Invalid prediction spec {spec!r}; expected LABEL=MANIFEST.csv")
    label, raw_path = spec.split("=", 1)
    manifest = Path(raw_path).resolve()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        motion_id = row.get("key") or row.get("motion_id")
        if not motion_id:
            raise KeyError(f"No key/motion_id in {manifest}")
        if motion_id in by_id:
            raise ValueError(f"Duplicate motion id {motion_id} in {manifest}")
        row["_manifest_dir"] = str(manifest.parent)
        by_id[motion_id] = row
    return label, by_id


def prediction_path(row: dict[str, str]) -> Path:
    raw_path = row.get("source_npz") or row.get("prediction_npz")
    if not raw_path:
        raise KeyError("Prediction row has no source_npz/prediction_npz")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(row["_manifest_dir"]) / path


def as_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def squeeze_batch(motion: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.squeeze(0)
        if torch.is_tensor(value) and value.ndim and value.shape[0] == 1
        else value
        for key, value in motion.items()
    }


def root_positions(motion: dict[str, Any], joints: torch.Tensor) -> torch.Tensor:
    if "root_positions" in motion:
        return as_tensor(motion["root_positions"], joints.device)
    return joints[:, 0]


def heading_angles(motion: dict[str, Any], joints: torch.Tensor, skeleton: Any) -> torch.Tensor:
    if "global_root_heading" in motion:
        heading = as_tensor(motion["global_root_heading"], joints.device)
        if heading.ndim == 2 and heading.shape[-1] == 2:
            cosine, sine = heading.unbind(-1)
            return torch.atan2(sine, cosine)
    from kimodo.motion_rep.feature_utils import compute_heading_angle

    return compute_heading_angle(joints.unsqueeze(0), skeleton).squeeze(0)


def body_coordinates(
    joints: torch.Tensor, root: torch.Tensor, heading: torch.Tensor
) -> torch.Tensor:
    relative = joints - root[:, None, :]
    cosine = torch.cos(-heading)[:, None]
    sine = torch.sin(-heading)[:, None]
    output = torch.empty_like(relative)
    output[..., 0] = cosine * relative[..., 0] + sine * relative[..., 2]
    output[..., 1] = relative[..., 1]
    output[..., 2] = -sine * relative[..., 0] + cosine * relative[..., 2]
    return output


def joint_groups(names: list[str]) -> tuple[list[int], list[int]]:
    lower_tokens = ("leg", "shin", "foot", "toe")
    lower = [
        index
        for index, name in enumerate(names)
        if any(token in name.lower() for token in lower_tokens)
    ]
    lower_set = set(lower)
    upper = [index for index in range(len(names)) if index not in lower_set]
    if not upper or not lower:
        raise ValueError("Could not construct SOMA77 upper/lower joint groups")
    return upper, lower


def category_of(row: dict[str, Any]) -> str:
    return str(row.get("category") or row.get("motion_regime_category") or "all")


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {"num_samples": len(rows)}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        result[f"{metric}_mean"] = float(values.mean()) if len(values) else math.nan
        result[f"{metric}_median"] = float(np.median(values)) if len(values) else math.nan
    result["truncated_sample_count"] = sum(
        int(row["gt_num_frames"] != row["pred_num_frames"]) for row in rows
    )
    return result


def write_result(
    label: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    joint_names: list[str],
    upper: list[int],
    lower: list[int],
) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    result = {
        "label": label,
        "units": "meters",
        "metric_direction": "lower_is_better",
        "alignment": "exact motion_id; frame t to frame t; min length; no DTW",
        "definitions": {
            "rte": "Mean framewise L2 root-position error in the common world frame.",
            "rte_xz": "Mean horizontal root-position L2 error.",
            "rte_y": "Mean absolute root-height error.",
            "bpe": "Mean joint L2 error after removing each motion's root position and yaw.",
            "bpe_upper": "BPE over SOMA77 upper-body joints.",
            "bpe_lower": "BPE over SOMA77 leg/shin/foot/toe joints.",
        },
        "joint_groups": {
            "skeleton": "SOMA77",
            "upper_names": [joint_names[index] for index in upper],
            "lower_names": [joint_names[index] for index in lower],
        },
        "overall": summarize(rows),
        "by_category": {
            name: summarize(items) for name, items in sorted(by_category.items())
        },
    }
    (output_dir / f"{label}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "motion_id",
        "category",
        "num_frames",
        "gt_num_frames",
        "pred_num_frames",
        *METRICS,
        "prediction_npz",
    ]
    with (output_dir / f"{label}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return result


def write_markdown(results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# Kimodo paired RTE/BPE evaluation",
        "",
        "单位均为米，数值越低越好。预测与 GT 按 motion_id 精确配对，逐帧直接比较，不使用 DTW。",
        "",
        "| 模型 | 样本数 | RTE | RTE-XZ | RTE-Y | BPE | BPE-Upper | BPE-Lower |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, result in results.items():
        summary = result["overall"]
        lines.append(
            f"| {label} | {summary['num_samples']} | {summary['rte_mean']:.6f} | "
            f"{summary['rte_xz_mean']:.6f} | {summary['rte_y_mean']:.6f} | "
            f"{summary['bpe_mean']:.6f} | {summary['bpe_upper_mean']:.6f} | "
            f"{summary['bpe_lower_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "RTE 衡量角色整体走到了哪里；BPE 在消除整体位置和朝向后衡量身体姿态是否正确。",
            "",
        ]
    )
    (output_dir / "rte_bpe_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    sys.path[:0] = [str(args.repro_root.resolve()), str(args.kimodo_root.resolve())]
    from kimodo.skeleton import SOMASkeleton30
    from kimodo_seed_repro.kimodo_io import load_kimodo_model

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model = load_kimodo_model(
        args.model,
        str(device),
        kimodo_root=str(args.kimodo_root.resolve()),
        skip_text_encoder=True,
        init_mode="pretrained",
    )
    model.eval()
    motion_rep = model.motion_rep
    skeleton30 = SOMASkeleton30().to(device)
    skeleton77 = skeleton30.somaskel77.to(device)
    joint_names = list(skeleton77.bone_order_names)
    if len(joint_names) != 77:
        raise ValueError(f"Expected SOMA77, found {len(joint_names)} joints")
    upper, lower = joint_groups(joint_names)

    index_rows = load_jsonl(args.feature_index.resolve())
    if args.limit is not None:
        index_rows = index_rows[: args.limit]
    if not index_rows:
        raise ValueError("Feature index contains no selected samples")
    motion_ids = [str(row["motion_id"]) for row in index_rows]
    if len(set(motion_ids)) != len(motion_ids):
        raise ValueError("Feature index contains duplicate motion_id values")
    predictions = dict(load_predictions(spec) for spec in args.prediction)
    expected = set(motion_ids)
    for label, rows in predictions.items():
        missing = expected - set(rows)
        if missing:
            raise ValueError(f"{label}: missing {len(missing)} predictions")

    per_model: dict[str, list[dict[str, Any]]] = {
        label: [] for label in predictions
    }
    started = time.monotonic()
    with torch.inference_mode():
        for sample_number, index_row in enumerate(index_rows, start=1):
            motion_id = str(index_row["motion_id"])
            feature_path = args.feature_cache.resolve() / index_row["feature_file"]
            payload = torch.load(feature_path, map_location="cpu", weights_only=True)
            features = payload["features"].to(device=device, dtype=torch.float32)[None]
            ground_truth = squeeze_batch(
                motion_rep.inverse(
                    features,
                    is_normalized=True,
                    posed_joints_from=args.posed_joints_from,
                    return_numpy=False,
                )
            )
            if ground_truth["posed_joints"].shape[1] == 30:
                ground_truth = skeleton30.output_to_SOMASkeleton77(ground_truth)
            gt_joints_full = ground_truth["posed_joints"].float()

            for label, rows in predictions.items():
                pred_path = prediction_path(rows[motion_id])
                with np.load(pred_path, allow_pickle=False) as archive:
                    prediction = {name: archive[name] for name in archive.files}
                pred_joints_full = as_tensor(prediction["posed_joints"], device)
                if gt_joints_full.shape[1] != 77 or pred_joints_full.shape[1] != 77:
                    raise ValueError(f"{motion_id}: RTE/BPE requires SOMA77 joints")
                gt_frames = len(gt_joints_full)
                pred_frames = len(pred_joints_full)
                frames = min(gt_frames, pred_frames)
                gt_joints = gt_joints_full[:frames]
                pred_joints = pred_joints_full[:frames]
                gt_root = root_positions(ground_truth, gt_joints_full)[:frames]
                pred_root = root_positions(prediction, pred_joints_full)[:frames]
                root_delta = pred_root - gt_root
                gt_heading = heading_angles(
                    ground_truth, gt_joints_full, skeleton77
                )[:frames]
                pred_heading = heading_angles(
                    prediction, pred_joints_full, skeleton77
                )[:frames]
                gt_body = body_coordinates(gt_joints, gt_root, gt_heading)
                pred_body = body_coordinates(pred_joints, pred_root, pred_heading)
                joint_error = torch.linalg.norm(pred_body - gt_body, dim=-1)
                per_model[label].append(
                    {
                        "motion_id": motion_id,
                        "category": category_of(index_row),
                        "num_frames": frames,
                        "gt_num_frames": gt_frames,
                        "pred_num_frames": pred_frames,
                        "rte": float(torch.linalg.norm(root_delta, dim=-1).mean()),
                        "rte_xz": float(
                            torch.linalg.norm(root_delta[:, [0, 2]], dim=-1).mean()
                        ),
                        "rte_y": float(root_delta[:, 1].abs().mean()),
                        "bpe": float(joint_error.mean()),
                        "bpe_upper": float(joint_error[:, upper].mean()),
                        "bpe_lower": float(joint_error[:, lower].mean()),
                        "prediction_npz": str(pred_path.resolve()),
                    }
                )
            if sample_number % args.progress_every == 0 or sample_number == len(index_rows):
                elapsed = time.monotonic() - started
                print(
                    f"progress={sample_number}/{len(index_rows)} "
                    f"rate={sample_number / max(elapsed, 1e-9):.2f} samples/s",
                    flush=True,
                )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        label: write_result(label, rows, output_dir, joint_names, upper, lower)
        for label, rows in per_model.items()
    }
    summary = {label: result["overall"] for label, result in results.items()}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(results, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
