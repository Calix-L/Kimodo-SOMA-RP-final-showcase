from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass
class FeatureItem:
    feature_path: Path
    text: str
    num_frames: int
    motion_id: str = ""
    text_feature_path: Path | None = None


def read_feature_index(cache_dir: str | Path, text_feature_dir: str | Path | None = None) -> list[FeatureItem]:
    index_path = Path(cache_dir) / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing feature cache index: {index_path}")

    items: list[FeatureItem] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            motion_id = str(row.get("motion_id", ""))
            text_feature_path = None
            if text_feature_dir is not None:
                text_feature_path = Path(text_feature_dir) / f"{motion_id}.pt"
            items.append(
                FeatureItem(
                    feature_path=Path(cache_dir) / row["feature_file"],
                    text=row["text"],
                    num_frames=int(row["num_frames"]),
                    motion_id=motion_id,
                    text_feature_path=text_feature_path,
                )
            )
    return items


class MotionFeatureDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        clip_frames: int,
        min_frames: int = 1,
        random_crop: bool = True,
        text_feature_dir: str | Path | None = None,
    ) -> None:
        self.items = [x for x in read_feature_index(cache_dir, text_feature_dir) if x.num_frames >= min_frames]
        if not self.items:
            raise ValueError(f"No cached clips with at least {min_frames} frames found in {cache_dir}")
        self.clip_frames = int(clip_frames)
        self.random_crop = bool(random_crop)
        self.text_feature_dir = Path(text_feature_dir) if text_feature_dir is not None else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        payload = torch.load(item.feature_path, map_location="cpu")
        feats = payload["features"].float()
        length = int(feats.shape[0])

        if length > self.clip_frames:
            if self.random_crop:
                start = random.randint(0, length - self.clip_frames)
            else:
                start = 0
            feats = feats[start : start + self.clip_frames]
            length = self.clip_frames

        return {
            "features": feats,
            "length": length,
            "text": item.text,
            "motion_id": item.motion_id,
            "text_features": torch.load(item.text_feature_path, map_location="cpu") if item.text_feature_path else None,
        }


def collate_motion_features(batch: list[dict[str, Any]]) -> dict[str, Any]:
    features = [x["features"] for x in batch]
    lengths = torch.tensor([x["length"] for x in batch], dtype=torch.long)
    padded = pad_sequence(features, batch_first=True)
    max_len = padded.shape[1]
    pad_mask = torch.arange(max_len).expand(len(batch), max_len) < lengths[:, None]
    return {
        "features": padded,
        "lengths": lengths,
        "pad_mask": pad_mask,
        "texts": [x["text"] for x in batch],
        "motion_ids": [x["motion_id"] for x in batch],
        "text_features": [x["text_features"] for x in batch] if batch and batch[0].get("text_features") is not None else None,
    }


def write_manifest(rows: list[dict[str, str]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["motion_id", "bvh_path", "text", "num_frames", "content_name"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
