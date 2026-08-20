#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from kimodo_seed_repro.data import MotionFeatureDataset, collate_motion_features
from kimodo_seed_repro.kimodo_io import load_kimodo_model, save_train_checkpoint, text_to_features
from kimodo_seed_repro.train_utils import (
    build_keyframe_conditions,
    build_optimizer,
    cosine_lr,
    get_autocast,
    masked_mse,
    seed_everything,
    set_optimizer_lr,
    structured_motion_loss,
    strip_to_plain_dict,
)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def append_loss_row(path: Path, row: dict) -> None:
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def zero_text_features(model, batch_size: int, cfg: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    text_len = int(cfg["train"].get("zero_text_tokens", 1))
    llm_dim = None
    for module in model.modules():
        embed_text = getattr(module, "embed_text", None)
        if isinstance(embed_text, torch.nn.Linear):
            llm_dim = embed_text.in_features
            text_len = int(getattr(module, "num_text_tokens", text_len) or text_len)
            break
    if llm_dim is None:
        raise RuntimeError("Could not infer Kimodo text feature dimension from denoiser modules.")
    text_feat = torch.zeros(batch_size, text_len, llm_dim, device=device)
    text_pad_mask = torch.zeros(batch_size, text_len, dtype=torch.bool, device=device)
    return text_feat, text_pad_mask


def collate_cached_text_features(text_features: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    feats = [x["text_feat"].to(device) for x in text_features]
    lengths = torch.tensor([int(x["length"]) for x in text_features], device=device)
    text_feat = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    max_len = text_feat.shape[1]
    text_pad_mask = torch.arange(max_len, device=device).expand(len(feats), max_len) < lengths[:, None]
    return text_feat, text_pad_mask


def train_step(model, batch, cfg: dict, device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x0 = batch["features"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    texts = list(batch["texts"])

    text_dropout = float(cfg["train"].get("text_dropout", 0.0))
    if text_dropout > 0:
        texts = ["" if torch.rand(()) < text_dropout else t for t in texts]

    with torch.no_grad():
        if batch.get("text_features") is not None:
            text_feat, text_pad_mask = collate_cached_text_features(batch["text_features"], device)
        elif bool(cfg["train"].get("zero_text_condition", False)):
            text_feat, text_pad_mask = zero_text_features(model, x0.shape[0], cfg, device)
        else:
            text_feat, text_pad_mask = text_to_features(model, texts, device)

    bsz = x0.shape[0]
    num_base_steps = int(cfg["train"].get("num_base_steps") or model.diffusion.num_base_steps)
    t = torch.randint(0, num_base_steps, (bsz,), device=device)
    model.diffusion.calc_diffusion_vars(torch.arange(num_base_steps, device=device))
    noise = torch.randn_like(x0)
    xt = model.diffusion.q_sample(x0, t, noise=noise)
    observed_motion, motion_mask, condition_parts = build_keyframe_conditions(x0, pad_mask, model.motion_rep, cfg)

    denoiser = model.denoiser.model if hasattr(model.denoiser, "model") else model.denoiser
    pred_x0 = denoiser(
        xt,
        pad_mask,
        text_feat,
        text_pad_mask,
        t,
        first_heading_angle=torch.zeros(bsz, device=device),
        motion_mask=motion_mask,
        observed_motion=observed_motion,
    )
    if bool(cfg["train"].get("structured_loss", {}).get("enabled", False)):
        loss, loss_parts = structured_motion_loss(pred_x0, x0, pad_mask, model.motion_rep, cfg)
    else:
        loss, loss_parts = masked_mse(pred_x0, x0, pad_mask), {}
    loss_parts.update(condition_parts)
    return loss, loss_parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--kimodo-root", required=True)
    parser.add_argument("--feature-cache", required=True)
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
    resume_optimizer = bool(cfg["train"].get("resume_optimizer", False))
    resume_ckpt = None
    if resume_from:
        resume_ckpt = torch.load(resume_from, map_location="cpu")
        missing, unexpected = denoiser.load_state_dict(resume_ckpt["denoiser_state_dict"], strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
        print(f"Loaded denoiser checkpoint from {resume_from} step={resume_ckpt.get('step')}", flush=True)

    data_cfg = cfg["data"]
    dataset = MotionFeatureDataset(
        args.feature_cache,
        clip_frames=int(data_cfg["clip_frames"]),
        min_frames=int(data_cfg.get("min_frames", 1)),
        random_crop=bool(data_cfg.get("random_crop", True)),
        text_feature_dir=data_cfg.get("text_feature_dir"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_motion_features,
    )

    optimizer = build_optimizer(denoiser.parameters(), cfg)
    if resume_optimizer and resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        print("Loaded optimizer state from resume checkpoint.", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(strip_to_plain_dict(cfg)), encoding="utf-8")

    max_steps = int(cfg["train"]["max_steps"])
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    log_every = int(cfg["train"].get("log_every", 20))
    save_every = int(cfg["train"].get("save_every", 1000))
    base_lr = float(cfg["train"]["lr"])
    warmup = int(cfg["train"].get("warmup_steps", 0))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 0.0))
    dtype_name = cfg.get("dtype", "bf16")
    batch_size = int(cfg["train"]["batch_size"])
    loss_csv = out_dir / "loss_history.csv"

    step = 0
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    summary = {
        "dataset_size": len(dataset),
        "batch_size": batch_size,
        "drop_last": True,
        "max_steps": max_steps,
        "planned_sample_exposures": max_steps * batch_size,
        "planned_exposures_per_sample": (max_steps * batch_size) / float(len(dataset)),
        "grad_accum_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "device": str(device),
        "dtype": dtype_name,
        "loss_type": "structured_motion" if bool(cfg["train"].get("structured_loss", {}).get("enabled", False)) else "masked_mse",
        "resume_from": resume_from,
        "constraint_training": cfg["train"].get("constraint_training", {}),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    pbar = tqdm(total=max_steps, desc="train")
    while step < max_steps:
        for batch in loader:
            lr = cosine_lr(step, max_steps, base_lr, warmup)
            set_optimizer_lr(optimizer, lr)

            with get_autocast(device, dtype_name):
                loss, loss_parts = train_step(model, batch, cfg, device)
                loss = loss / grad_accum
            loss.backward()
            running += float(loss.detach().cpu()) * grad_accum
            if "running_parts" not in locals():
                running_parts = {}
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
                running = 0.0
                pbar.set_postfix(loss=f"{avg:.5f}", lr=f"{lr:.2e}")
                append_loss_row(
                    loss_csv,
                    {
                        "step": step,
                        "loss": avg,
                        "lr": lr,
                        "elapsed_sec": time.time() - start_time,
                        "sample_exposures": step * batch_size,
                        "exposures_per_sample": (step * batch_size) / float(len(dataset)),
                        **{f"loss_{name}": value / log_every for name, value in sorted(running_parts.items())},
                    },
                )
                running_parts = {}
            if step % save_every == 0:
                save_train_checkpoint(out_dir / f"checkpoint_step_{step}.pt", model, optimizer, step, cfg)
            if step >= max_steps:
                break

    save_train_checkpoint(out_dir / "checkpoint_last.pt", model, optimizer, step, cfg)
    pbar.close()
    print(f"Training finished. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
