#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat


SOMA77_PARENTS = [
    -1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 3, 11, 12, 13, 14, 15, 16, 17,
    14, 19, 20, 21, 22, 14, 24, 25, 26, 27, 14, 29, 30, 31, 32, 14, 34,
    35, 36, 37, 3, 39, 40, 41, 42, 43, 44, 45, 42, 47, 48, 49, 50, 42,
    52, 53, 54, 55, 42, 57, 58, 59, 60, 42, 62, 63, 64, 65, 0, 67, 68,
    69, 70, 0, 72, 73, 74, 75,
]
SOMA77_EDGES = [(parent, child) for child, parent in enumerate(SOMA77_PARENTS) if parent >= 0]
assert len(SOMA77_EDGES) == 76


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.@-]+", "_", value).strip("_")[:100] or "sample"


def text_lines(draw: ImageDraw.ImageDraw, text: str, width: int, font: ImageFont.ImageFont) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        cand = f"{cur} {word}".strip()
        if draw.textbbox((0, 0), cand, font=font)[2] <= width:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines[:2]


def project(points: np.ndarray, box: tuple[int, int, int, int], global_min: np.ndarray, global_max: np.ndarray):
    x0, y0, x1, y1 = box
    center = (global_min + global_max) / 2
    scale = float(np.max(global_max - global_min))
    if scale <= 1e-6:
        scale = 1.0
    pts = points - center
    # Isometric-ish projection: horizontal uses x-z, vertical uses y with a little z depth.
    px = pts[:, 0] - 0.35 * pts[:, 2]
    py = pts[:, 1] + 0.18 * pts[:, 2]
    w, h = x1 - x0, y1 - y0
    sx = x0 + w * 0.5 + px / scale * w * 0.72
    sy = y0 + h * 0.55 - py / scale * h * 0.72
    return np.stack([sx, sy], axis=-1)


def draw_skeleton(draw: ImageDraw.ImageDraw, joints: np.ndarray, box: tuple[int, int, int, int], gmin, gmax):
    pts = project(joints, box, gmin, gmax)
    for parent, child in SOMA77_EDGES:
        a = tuple(pts[parent])
        b = tuple(pts[child])
        draw.line((a, b), fill=(30, 92, 170), width=3)
    r = 2
    for x, y in pts:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(12, 31, 68))


def load_meta(sample_dir: Path) -> tuple[dict, dict]:
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    source_meta_path = sample_dir / "source_meta.json"
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8")) if source_meta_path.is_file() else {}
    return meta, source_meta


def make_gif(sample_dir: Path, out_dir: Path, input_root: Path, output_root: Path) -> Path:
    rel = sample_dir.relative_to(output_root)
    src_dir = input_root / rel
    meta, source_meta = load_meta(src_dir)
    motion = np.load(sample_dir / "motion.npz")
    joints = np.asarray(motion["posed_joints"], dtype=np.float32)
    assert joints.ndim == 3 and joints.shape[1] == 77, joints.shape

    category = rel.parts[0] if len(rel.parts) > 0 else "unknown"
    sample_id = rel.name
    take = safe_name(str(source_meta.get("take_name") or sample_id))
    text = str(meta.get("text", ""))
    filename = f"{safe_name(category)}_{safe_name(sample_id)}_{take}_finetuned_soma77_only.gif"
    out_path = out_dir / filename

    width, height = 960, 540
    left_box = (24, 110, 468, 452)
    right_box = (492, 110, 936, 452)
    font = ImageFont.load_default()
    frames = []
    step = max(1, int(np.ceil(len(joints) / 32)))
    selected = list(range(0, len(joints), step))[:32]
    if selected[-1] != len(joints) - 1:
        selected.append(len(joints) - 1)
    gmin = joints.reshape(-1, 3).min(axis=0)
    gmax = joints.reshape(-1, 3).max(axis=0)

    for frame_idx in selected:
        im = Image.new("RGB", (width, height), (248, 250, 252))
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, width, 88), fill=(17, 24, 39))
        title = f"{category} / {sample_id} / {take} / frame {frame_idx}"
        draw.text((24, 14), title, fill=(255, 255, 255), font=font)
        for i, line in enumerate(text_lines(draw, text, 900, font)):
            draw.text((24, 38 + i * 18), line, fill=(229, 231, 235), font=font)

        draw.rectangle(left_box, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
        draw.text((left_box[0] + 28, left_box[1] + 132), "GT glTF preview not rendered", fill=(100, 116, 139), font=font)
        draw.text((left_box[0] + 28, left_box[1] + 154), "KS3_* env vars are missing", fill=(100, 116, 139), font=font)

        draw.rectangle(right_box, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
        draw_skeleton(draw, joints[frame_idx], right_box, gmin, gmax)
        draw.text((170, 476), "GT game preview", fill=(15, 23, 42), font=font)
        draw.text((642, 476), "finetuned SOMA77 skeleton, 76 official edges", fill=(15, 23, 42), font=font)
        frames.append(im)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=90, loop=0)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gif-dir", required=True, type=Path)
    args = parser.parse_args()
    args.gif_dir.mkdir(parents=True, exist_ok=True)

    made = []
    for motion_path in sorted(args.output_root.rglob("motion.npz")):
        made.append(make_gif(motion_path.parent, args.gif_dir, args.input_root, args.output_root))
    if len(made) != 10:
        raise RuntimeError(f"Expected 10 GIFs, made {len(made)}")
    for p in made:
        im = Image.open(p)
        assert getattr(im, "n_frames", 1) > 1
        first = im.convert("RGB")
        right = first.crop((480, 96, 960, 456))
        assert sum(ImageStat.Stat(right).stddev) > 1
        print(p)


if __name__ == "__main__":
    main()
