#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import json

import torch


class CachedTextEncoder:
    def __init__(self, feature_index: Path, text_feature_dir: Path, device: str):
        from kimodo.sanitize import sanitize_text

        self.device = torch.device(device)
        self.by_text: dict[str, Path] = {}
        with feature_index.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                path = text_feature_dir / f"{row['motion_id']}.pt"
                text = str(row.get("text", ""))
                self.by_text.setdefault(text, path)
                self.by_text.setdefault(sanitize_text(text), path)

    def eval(self):
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self

    def __call__(self, texts: list[str]):
        feats = []
        lengths = []
        for text in texts:
            path = self.by_text.get(text)
            if path is None:
                raise KeyError(f"No cached text feature for prompt: {text!r}")
            payload = torch.load(path, map_location="cpu")
            feat = payload["text_feat"].float()
            feats.append(feat)
            lengths.append(int(payload["length"]))
        padded = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
        return padded.to(self.device), lengths


def parse_wrapper_args(argv: list[str]) -> tuple[Path, Path, Path, Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kimodo-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--feature-index", required=True, type=Path)
    parser.add_argument("--text-feature-dir", required=True, type=Path)
    args, rest = parser.parse_known_args(argv)
    return (
        args.kimodo_root.resolve(),
        args.checkpoint.resolve(),
        args.feature_index.resolve(),
        args.text_feature_dir.resolve(),
        rest,
    )


def main() -> None:
    kimodo_root, checkpoint_path, feature_index, text_feature_dir, rest = parse_wrapper_args(sys.argv[1:])
    sys.path.insert(0, str(kimodo_root))

    module_path = kimodo_root / "benchmark" / "generate_eval.py"
    spec = importlib.util.spec_from_file_location("kimodo_generate_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    generate_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate_eval)

    original_load_model = generate_eval.load_model

    def load_finetuned_model(*args, **kwargs):
        device = kwargs.get("device") or "cuda"
        kwargs["text_encoder"] = CachedTextEncoder(feature_index, text_feature_dir, device)
        model_result = original_load_model(*args, **kwargs)
        if isinstance(model_result, tuple):
            model = model_result[0]
        else:
            model = model_result
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        denoiser = model.denoiser.model if hasattr(model.denoiser, "model") else model.denoiser
        missing, unexpected = denoiser.load_state_dict(ckpt["denoiser_state_dict"], strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
        model.eval()
        print(f"Loaded finetuned denoiser checkpoint: {checkpoint_path} step={ckpt.get('step')}", flush=True)
        return model_result

    generate_eval.load_model = load_finetuned_model
    sys.argv = [str(module_path), *rest]
    generate_eval.main()


if __name__ == "__main__":
    main()
