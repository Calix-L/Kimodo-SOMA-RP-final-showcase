#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import append_loss_row, load_config, train_step
from kimodo_seed_repro.data import MotionFeatureDataset, collate_motion_features
from kimodo_seed_repro.kimodo_io import load_kimodo_model, save_train_checkpoint
from kimodo_seed_repro.train_utils import (
    build_optimizer,
    cosine_lr,
    get_autocast,
    seed_everything,
    set_optimizer_lr,
    strip_to_plain_dict,
)


def make_loader(cache_dir: str | Path, cfg: dict, batch_size: int, random_crop: bool, shuffle: bool, drop_last: bool):
    data_cfg = cfg["data"]
    dataset = MotionFeatureDataset(
        cache_dir,
        clip_frames=int(data_cfg["clip_frames"]),
        min_frames=int(data_cfg.get("min_frames", 1)),
        random_crop=random_crop,
        text_feature_dir=data_cfg.get("text_feature_dir"),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_motion_features,
    )
    return dataset, loader


def write_csv(path: Path, row: dict) -> None:
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


@torch.no_grad()
def evaluate(model, loader, cfg: dict, device: torch.device, dtype_name: str, max_batches: int | None = None) -> dict:
    model.eval()
    denoiser = model.denoiser.model if hasattr(model.denoiser, "model") else model.denoiser
    denoiser.eval()
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["train"]["text_dropout"] = 0.0
    totals: dict[str, float] = {}
    total_loss = 0.0
    batches = 0
    for batch in loader:
        with get_autocast(device, dtype_name):
            loss, parts = train_step(model, batch, eval_cfg, device)
        total_loss += float(loss.detach().cpu())
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    denoiser.train()
    model.train(True)
    if batches == 0:
        raise RuntimeError("Validation loader produced no batches.")
    return {"val_loss": total_loss / batches, **{f"val_loss_{k}": v / batches for k, v in sorted(totals.items())}, "val_batches": batches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--kimodo-root", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.kimodo_root not in sys.path:
        sys.path.insert(0, args.kimodo_root)

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 7)))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    zero_text_condition = bool(cfg["train"].get("zero_text_condition", False))
    use_cached_text_features = bool(cfg.get("data", {}).get("text_feature_dir"))
    model = load_kimodo_model(
        cfg["model"],
        str(device),
        kimodo_root=args.kimodo_root,
        skip_text_encoder=zero_text_condition or use_cached_text_features,
        init_mode=cfg.get("init_mode", "pretrained"),
    )
    model.train(True)
    if not (zero_text_condition or use_cached_text_features):
        model.text_encoder.eval()
        for p in model.text_encoder.model.parameters() if hasattr(model.text_encoder, "model") else []:
            p.requires_grad = False

    denoiser = model.denoiser.model if hasattr(model.denoiser, "model") else model.denoiser
    denoiser.train()

    resume_from = cfg["train"].get("resume_from")
    resume_ckpt = None
    if resume_from:
        resume_ckpt = torch.load(resume_from, map_location="cpu")
        missing, unexpected = denoiser.load_state_dict(resume_ckpt["denoiser_state_dict"], strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
        print(f"Loaded denoiser checkpoint from {resume_from} step={resume_ckpt.get('step')}", flush=True)

    batch_size = int(cfg["train"]["batch_size"])
    train_dataset, train_loader = make_loader(
        args.train_cache,
        cfg,
        batch_size=batch_size,
        random_crop=bool(cfg["data"].get("random_crop", True)),
        shuffle=True,
        drop_last=True,
    )
    val_dataset, val_loader = make_loader(
        args.val_cache,
        cfg,
        batch_size=batch_size,
        random_crop=False,
        shuffle=False,
        drop_last=False,
    )

    optimizer = build_optimizer(denoiser.parameters(), cfg)
    if bool(cfg["train"].get("resume_optimizer", False)) and resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(strip_to_plain_dict(cfg)), encoding="utf-8")

    max_steps = int(cfg["train"]["max_steps"])
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    log_every = int(cfg["train"].get("log_every", 20))
    save_every = int(cfg["train"].get("save_every", 500))
    eval_every = int(cfg["train"].get("eval_every", save_every))
    base_lr = float(cfg["train"]["lr"])
    warmup = int(cfg["train"].get("warmup_steps", 0))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 0.0))
    dtype_name = cfg.get("dtype", "bf16")

    summary = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "batch_size": batch_size,
        "drop_last": True,
        "max_steps": max_steps,
        "planned_sample_exposures": max_steps * batch_size,
        "planned_exposures_per_train_sample": (max_steps * batch_size) / float(len(train_dataset)),
        "grad_accum_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "device": str(device),
        "dtype": dtype_name,
        "loss_type": "structured_motion" if bool(cfg["train"].get("structured_loss", {}).get("enabled", False)) else "masked_mse",
        "resume_from": resume_from,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    loss_csv = out_dir / "loss_history.csv"
    val_csv = out_dir / "val_loss_history.csv"
    step = 0
    running = 0.0
    running_parts: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    pbar = tqdm(total=max_steps, desc="train")

    while step < max_steps:
        for batch in train_loader:
            lr = cosine_lr(step, max_steps, base_lr, warmup)
            set_optimizer_lr(optimizer, lr)
            with get_autocast(device, dtype_name):
                loss, loss_parts = train_step(model, batch, cfg, device)
                loss = loss / grad_accum
            loss.backward()
            running += float(loss.detach().cpu()) * grad_accum
            for name, value in loss_parts.items():
                running_parts[name] = running_parts.get(name, 0.0) + float(value.detach().cpu())
            if (step + 1) % grad_accum == 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            pbar.update(1)
            if step % log_every == 0:
                avg = running / log_every
                pbar.set_postfix(loss=f"{avg:.5f}", lr=f"{lr:.2e}")
                append_loss_row(
                    loss_csv,
                    {
                        "step": step,
                        "loss": avg,
                        "lr": lr,
                        "elapsed_sec": time.time() - start_time,
                        "sample_exposures": step * batch_size,
                        "exposures_per_train_sample": (step * batch_size) / float(len(train_dataset)),
                        **{f"loss_{name}": value / log_every for name, value in sorted(running_parts.items())},
                    },
                )
                running = 0.0
                running_parts = {}
            if step % save_every == 0:
                save_train_checkpoint(out_dir / f"checkpoint_step_{step}.pt", model, optimizer, step, cfg)
            if step % eval_every == 0:
                metrics = evaluate(model, val_loader, cfg, device, dtype_name)
                write_csv(val_csv, {"step": step, "elapsed_sec": time.time() - start_time, **metrics})
                print(json.dumps({"step": step, **metrics}, indent=2), flush=True)
            if step >= max_steps:
                break

    save_train_checkpoint(out_dir / "checkpoint_last.pt", model, optimizer, step, cfg)
    if max_steps % eval_every != 0:
        metrics = evaluate(model, val_loader, cfg, device, dtype_name)
        write_csv(val_csv, {"step": step, "elapsed_sec": time.time() - start_time, **metrics})
    pbar.close()
    print(f"Training finished. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
