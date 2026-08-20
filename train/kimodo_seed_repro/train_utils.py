from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_autocast(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    if dtype_name == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype_name == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", enabled=False)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask[..., None].to(dtype=pred.dtype)
    sq = (pred - target).pow(2) * mask_f
    denom = mask_f.sum().clamp(min=1.0) * pred.shape[-1]
    return sq.sum() / denom


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    original_mask_ndim = mask.ndim
    while mask.ndim < pred.ndim:
        mask = mask[..., None]
    mask_f = mask.to(dtype=pred.dtype)
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta) * mask_f
    values_per_mask = math.prod(pred.shape[original_mask_ndim:])
    denom = mask_f.sum().clamp(min=1.0) * values_per_mask
    return loss.sum() / denom


def _masked_smooth_l1_zero(value: torch.Tensor, mask: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    return _masked_smooth_l1(value, torch.zeros_like(value), mask, beta=beta)


def _cont6d_to_matrix(cont6d: torch.Tensor) -> torch.Tensor:
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = F.normalize(x_raw, dim=-1, eps=1e-6)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1, eps=1e-6)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def _geodesic_angle(pred_6d: torch.Tensor, target_6d: torch.Tensor) -> torch.Tensor:
    pred_rot = _cont6d_to_matrix(pred_6d)
    target_rot = _cont6d_to_matrix(target_6d)
    rel = torch.matmul(pred_rot.transpose(-1, -2), target_rot)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-5, 1.0 - 1e-5)
    return torch.acos(cos)


def _masked_frame_select(mask: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gathered = mask.gather(1, indices)
    return gathered


def build_keyframe_conditions(
    x0: torch.Tensor,
    pad_mask: torch.Tensor,
    motion_rep,
    cfg: dict,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Sample sparse GT keyframe conditions for Kimodo stage-2 style training.

    The denoiser consumes normalized observed_motion plus a same-shaped mask.
    Keyframes are sampled over valid frames and expose kinematic slices that
    correspond to Kimodo's full-body constraints: root position, root heading,
    local joint positions, and joint rotations.
    """
    cond_cfg = cfg["train"].get("constraint_training", {})
    if not bool(cond_cfg.get("enabled", False)):
        return None, None, {}

    prob = float(cond_cfg.get("probability", 0.5))
    if prob <= 0.0:
        return None, None, {}

    bsz, _frames, dim = x0.shape
    device = x0.device
    motion_mask = torch.zeros_like(x0)
    observed_motion = torch.zeros_like(x0)

    slices = motion_rep.slice_dict
    slice_names = cond_cfg.get(
        "observed_slices",
        [
            "smooth_root_pos",
            "global_root_heading",
            "local_joints_positions",
            "global_rot_data",
        ],
    )
    observed_dims = torch.zeros(dim, dtype=torch.bool, device=device)
    for name in slice_names:
        sl = slices.get(name)
        if sl is None:
            raise KeyError(f"Unknown motion_rep slice in constraint_training.observed_slices: {name!r}")
        observed_dims[sl] = True

    min_keyframes = int(cond_cfg.get("min_keyframes", 1))
    max_keyframes = int(cond_cfg.get("max_keyframes", 20))
    min_keyframes = max(1, min_keyframes)
    max_keyframes = max(min_keyframes, max_keyframes)

    constrained = 0
    total_keyframes = 0
    for b in range(bsz):
        if torch.rand((), device=device) >= prob:
            continue
        valid_idx = torch.nonzero(pad_mask[b], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        n = int(torch.randint(min_keyframes, max_keyframes + 1, (), device=device).item())
        n = min(n, int(valid_idx.numel()))
        selected = valid_idx[torch.randperm(valid_idx.numel(), device=device)[:n]]
        motion_mask[b, selected[:, None], observed_dims] = 1.0
        constrained += 1
        total_keyframes += n

    if constrained == 0:
        return None, None, {
            "constraint_batch_fraction": x0.new_tensor(0.0),
            "constraint_keyframes_per_constrained_sample": x0.new_tensor(0.0),
            "constraint_mask_density": x0.new_tensor(0.0),
        }

    observed_motion = x0 * motion_mask
    stats = {
        "constraint_batch_fraction": x0.new_tensor(constrained / float(bsz)),
        "constraint_keyframes_per_constrained_sample": x0.new_tensor(total_keyframes / float(constrained)),
        "constraint_mask_density": motion_mask.mean().detach(),
    }
    return observed_motion, motion_mask, stats


def structured_motion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    motion_rep,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Kimodo paper-style structured x0 loss over unnormalized motion-rep blocks.

    Paper objective:
      L = gamma1 |r_p_hat-r_p| + gamma2 |r_a_hat-r_a|
        + gamma3 |j_p_hat-j_p| + gamma4 |j_v_hat-j_v|
        + gamma5 |j_a_hat-j_a| + gamma6 |e_hat-e|
        + gamma7 |FK(j_a_hat)-j_p|
    with masked Smooth-L1 terms.
    """
    loss_cfg = cfg["train"].get("structured_loss", {})
    weights = loss_cfg.get("weights", {})
    beta = float(loss_cfg.get("smooth_l1_beta", 0.1))

    pred_u = motion_rep.unnormalize(pred.float())
    target_u = motion_rep.unnormalize(target.float())
    mask = mask.bool()

    slices = motion_rep.slice_dict
    local_pos_slice = slices["local_joints_positions"]
    rot_slice = slices["global_rot_data"]
    vel_slice = slices["velocities"]
    root_slice = slices["smooth_root_pos"]
    heading_slice = slices["global_root_heading"]
    contact_slice = slices["foot_contacts"]

    pred_root = pred_u[..., root_slice]
    target_root = target_u[..., root_slice]
    pred_heading = pred_u[..., heading_slice]
    target_heading = target_u[..., heading_slice]
    pred_pos = pred_u[..., local_pos_slice].reshape(*pred_u.shape[:2], motion_rep.nbjoints, 3)
    target_pos = target_u[..., local_pos_slice].reshape(*target_u.shape[:2], motion_rep.nbjoints, 3)
    pred_rot = pred_u[..., rot_slice].reshape(*pred_u.shape[:2], motion_rep.nbjoints, 6)
    target_rot = target_u[..., rot_slice].reshape(*target_u.shape[:2], motion_rep.nbjoints, 6)
    pred_vel = pred_u[..., vel_slice].reshape(*pred_u.shape[:2], motion_rep.nbjoints, 3)
    target_vel = target_u[..., vel_slice].reshape(*target_u.shape[:2], motion_rep.nbjoints, 3)
    pred_contacts = pred_u[..., contact_slice]
    target_contacts = target_u[..., contact_slice]

    joint_weights_cfg = loss_cfg.get("joint_weights", {})
    joint_weights = torch.ones(motion_rep.nbjoints, device=pred.device, dtype=pred_u.dtype)
    for idx in joint_weights_cfg.get("high", []):
        if 0 <= int(idx) < motion_rep.nbjoints:
            joint_weights[int(idx)] = float(joint_weights_cfg.get("high_weight", 2.0))
    joint_weights = joint_weights.view(1, 1, motion_rep.nbjoints, 1)

    parts: dict[str, torch.Tensor] = {}
    parts["root_position"] = _masked_smooth_l1(pred_root, target_root, mask, beta)
    parts["root_angle"] = _masked_smooth_l1(pred_heading, target_heading, mask, beta)
    parts["joint_position"] = _masked_smooth_l1(pred_pos * joint_weights, target_pos * joint_weights, mask, beta)
    parts["joint_velocity"] = _masked_smooth_l1(pred_vel * joint_weights, target_vel * joint_weights, mask, beta)
    parts["joint_rotation"] = _masked_smooth_l1(pred_rot * joint_weights, target_rot * joint_weights, mask, beta)
    parts["foot_contact"] = _masked_smooth_l1(pred_contacts, target_contacts, mask, beta)

    # The paper's FK term compares positions induced by predicted rotations to
    # the target joint positions. Use Kimodo's own differentiable inverse path so
    # global 6D rotations are converted to local rotations consistently.
    pred_fk = motion_rep.inverse(pred_u, is_normalized=False, return_numpy=False)["posed_joints"]
    target_fk = motion_rep.inverse(target_u, is_normalized=False, return_numpy=False)["posed_joints"]
    parts["fk_position"] = _masked_smooth_l1(pred_fk * joint_weights, target_fk * joint_weights, mask, beta)

    total = pred.new_tensor(0.0, dtype=torch.float32)
    weighted_parts: dict[str, torch.Tensor] = {}
    for name, value in parts.items():
        weight = float(weights.get(name, 0.0))
        if weight:
            weighted_parts[name] = value.detach()
            total = total + weight * value

    if float(weights.get("feature_mse", 0.0)):
        feature = masked_mse(pred, target, mask)
        weighted_parts["feature_mse"] = feature.detach()
        total = total + float(weights["feature_mse"]) * feature

    return total, weighted_parts


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


class AdamAtan2(Optimizer):
    """Adam variant using atan2(m, sqrt(v)) as the normalized update."""

    def __init__(
        self,
        params,
        lr: float = 2e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamAtan2 does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                m_hat = exp_avg / bias_correction1
                v_hat_sqrt = (exp_avg_sq / bias_correction2).sqrt()

                if weight_decay:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(torch.atan2(m_hat, v_hat_sqrt), alpha=-lr)

        return loss


def build_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
    name = str(cfg["train"].get("optimizer", "adamw")).lower()
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    betas = tuple(cfg["train"].get("betas", (0.9, 0.999)))
    if name in {"adam_atan2", "adam-atan2", "adamatan2"}:
        return AdamAtan2(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer {name!r}.")


def strip_to_plain_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_to_plain_dict(v) for v in obj]
    return obj
