from __future__ import annotations

from pathlib import Path

import torch


class DummyTextEncoder:
    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, texts):
        raise RuntimeError("DummyTextEncoder cannot encode text. Load the real encoder for training.")


def load_kimodo_model(
    model_name: str,
    device: str,
    kimodo_root: str | None = None,
    skip_text_encoder: bool = False,
    init_mode: str = "pretrained",
):
    if kimodo_root:
        import sys

        if kimodo_root not in sys.path:
            sys.path.insert(0, kimodo_root)

    text_encoder = DummyTextEncoder() if skip_text_encoder else None
    if init_mode == "pretrained":
        from kimodo.model.load_model import load_model

        return load_model(model_name, device=device, eval_mode=False, text_encoder=text_encoder)
    if init_mode == "scratch":
        return load_kimodo_model_from_scratch(model_name, device=device, text_encoder=text_encoder)
    raise ValueError(f"Unsupported init_mode={init_mode!r}; expected 'pretrained' or 'scratch'.")


def load_kimodo_model_from_scratch(model_name: str, device: str, text_encoder=None):
    """Instantiate Kimodo architecture and motion stats, but do not load denoiser ckpt."""
    import importlib

    from omegaconf import OmegaConf

    loader = importlib.import_module("kimodo.model.load_model")
    loading = importlib.import_module("kimodo.model.loading")

    if model_name not in loading.AVAILABLE_MODELS:
        model_name = loader.resolve_model_name(model_name, "Kimodo")

    model_path = loader._resolve_hf_model_path(model_name)
    model_conf = OmegaConf.load(model_path / "config.yaml")
    if "denoiser" not in model_conf:
        raise ValueError(f"Kimodo config at {model_path} has no denoiser section.")
    model_conf.denoiser.ckpt_path = None

    if text_encoder is not None:
        runtime_conf = OmegaConf.create({"checkpoint_dir": str(model_path)})
    else:
        text_encoder_url = loading.get_env_var("TEXT_ENCODER_URL", loader.DEFAULT_TEXT_ENCODER_URL)
        runtime_conf = OmegaConf.create(
            {
                "checkpoint_dir": str(model_path),
                "text_encoder": loader._select_text_encoder_conf(text_encoder_url),
            }
        )

    model_cfg = OmegaConf.to_container(OmegaConf.merge(model_conf, runtime_conf), resolve=True)
    model_cfg.pop("checkpoint_dir", None)
    if text_encoder is not None:
        model_cfg["text_encoder"] = None

    model = loading.instantiate_from_dict(model_cfg, overrides={"device": device})
    if text_encoder is not None:
        model.text_encoder = text_encoder
    return model


def text_to_features(model, texts: list[str], device: torch.device):
    text_feat, text_lengths = model.text_encoder(texts)
    text_feat = text_feat.to(device)

    empty = torch.tensor([len(t.strip()) == 0 for t in texts], device=device)
    if empty.any():
        text_feat[empty] = 0

    max_len = text_feat.shape[1]
    lengths = torch.tensor(text_lengths, device=device)
    lengths[empty] = 0
    text_pad_mask = torch.arange(max_len, device=device).expand(len(texts), max_len) < lengths[:, None]
    return text_feat, text_pad_mask


def save_train_checkpoint(path: str | Path, model, optimizer, step: int, config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    denoiser = model.denoiser.model if hasattr(model.denoiser, "model") else model.denoiser
    torch.save(
        {
            "step": step,
            "denoiser_state_dict": denoiser.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )
