#!/usr/bin/env python3
"""Run the official Kimodo benchmark runner with an optional finetuned denoiser."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


class CachedTextEncoder:
    """Serve precomputed text features with the interface expected by Kimodo."""

    def __init__(self, feature_index: Path, text_feature_dir: Path, device: str):
        from kimodo.sanitize import sanitize_text

        self.device = torch.device(device)
        self.by_text: dict[str, Path] = {}
        with feature_index.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                feature_path = text_feature_dir / f"{row['motion_id']}.pt"
                text = str(row.get("text", ""))
                self.by_text.setdefault(text, feature_path)
                self.by_text.setdefault(sanitize_text(text), feature_path)

    def eval(self) -> "CachedTextEncoder":
        return self

    def to(self, device: str) -> "CachedTextEncoder":
        self.device = torch.device(device)
        return self

    def __call__(self, texts: list[str]) -> tuple[torch.Tensor, list[int]]:
        features: list[torch.Tensor] = []
        lengths: list[int] = []
        for text in texts:
            feature_path = self.by_text.get(text)
            if feature_path is None:
                raise KeyError(f"No cached text feature for prompt: {text!r}")
            payload = torch.load(feature_path, map_location="cpu", weights_only=True)
            features.append(payload["text_feat"].float())
            lengths.append(int(payload["length"]))
        padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        return padded.to(self.device), lengths


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kimodo-root", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--feature-index", type=Path)
    parser.add_argument("--text-feature-dir", type=Path)
    args, official_args = parser.parse_known_args(argv)
    if bool(args.feature_index) != bool(args.text_feature_dir):
        parser.error("--feature-index and --text-feature-dir must be supplied together")
    return args, official_args


def load_official_runner(kimodo_root: Path) -> tuple[Any, Path]:
    module_path = kimodo_root / "benchmark" / "generate_eval.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Kimodo benchmark runner not found: {module_path}")
    sys.path.insert(0, str(kimodo_root))
    spec = importlib.util.spec_from_file_location("kimodo_generate_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import official runner: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module_path


def main() -> None:
    args, official_args = parse_wrapper_args(sys.argv[1:])
    kimodo_root = args.kimodo_root.resolve()
    runner, module_path = load_official_runner(kimodo_root)

    if args.checkpoint or args.feature_index:
        original_load_model = runner.load_model

        def load_model_override(*model_args, **model_kwargs):
            if args.feature_index:
                device = str(model_kwargs.get("device") or "cuda")
                model_kwargs["text_encoder"] = CachedTextEncoder(
                    args.feature_index.resolve(),
                    args.text_feature_dir.resolve(),
                    device,
                )
            model_result = original_load_model(*model_args, **model_kwargs)
            model = model_result[0] if isinstance(model_result, tuple) else model_result
            if args.checkpoint:
                checkpoint_path = args.checkpoint.resolve()
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
                if "denoiser_state_dict" not in checkpoint:
                    raise KeyError(
                        f"Checkpoint has no denoiser_state_dict: {checkpoint_path}"
                    )
                denoiser = (
                    model.denoiser.model
                    if hasattr(model.denoiser, "model")
                    else model.denoiser
                )
                incompatible = denoiser.load_state_dict(
                    checkpoint["denoiser_state_dict"], strict=True
                )
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise RuntimeError(
                        "Checkpoint key mismatch: "
                        f"missing={incompatible.missing_keys}, "
                        f"unexpected={incompatible.unexpected_keys}"
                    )
                print(
                    f"Loaded denoiser checkpoint {checkpoint_path}; "
                    f"step={checkpoint.get('step')}",
                    flush=True,
                )
            model.eval()
            return model_result

        runner.load_model = load_model_override

    sys.argv = [str(module_path), *official_args]
    runner.main()


if __name__ == "__main__":
    main()
