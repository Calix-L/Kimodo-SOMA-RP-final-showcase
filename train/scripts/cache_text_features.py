#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimodo_seed_repro.kimodo_io import load_kimodo_model, text_to_features


def read_index(cache_dir: Path) -> list[dict]:
    rows = []
    with (cache_dir / "index.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kimodo-root", required=True)
    ap.add_argument("--feature-cache", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default="Kimodo-SOMA-RP-v1.1")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.kimodo_root not in sys.path:
        sys.path.insert(0, args.kimodo_root)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_index(args.feature_cache)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_kimodo_model(args.model, str(device), kimodo_root=args.kimodo_root, init_mode="pretrained")
    model.eval()

    written = 0
    for start in tqdm(range(0, len(rows), args.batch_size), desc="text-cache"):
        batch = rows[start : start + args.batch_size]
        missing = [row for row in batch if not (args.out_dir / f"{row['motion_id']}.pt").is_file()]
        if not missing:
            continue
        texts = [row.get("text", "") for row in missing]
        with torch.no_grad():
            text_feat, text_pad_mask = text_to_features(model, texts, device)
        lengths = text_pad_mask.long().sum(dim=1).cpu()
        for row, feat, length in zip(missing, text_feat.cpu(), lengths):
            torch.save({"text_feat": feat[: int(length)].contiguous(), "length": int(length)}, args.out_dir / f"{row['motion_id']}.pt")
            written += 1
    manifest = {"feature_cache": str(args.feature_cache), "out_dir": str(args.out_dir), "rows": len(rows), "written": written}
    (args.out_dir / "text_cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
