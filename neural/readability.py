"""Optional target-edge emphasis for pixel-flow training, without OCR.

V2 dataset ``caption_label_audit`` boxes cover captioned titles/buttons, not all
visible text (navigation, body copy, form labels and diagram nodes are missing).
This module therefore derives an edge map from the *clean training pixels*.
It emphasizes glyph strokes, thin lines and button borders together; it cannot
recognize letters, enforce spelling or establish readability by itself.

Inference does not use this module. No images, templates or lettering are added
to a generated frame. A change of training objective needs fresh quality checks.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch.nn import functional as F


def validate_edge_weight(value: float) -> float:
    """Accept a bounded finite emphasis; zero retains ordinary flow MSE."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("edge_weight must be a finite number in 0..16")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 16:
        raise ValueError("edge_weight must be a finite number in 0..16")
    return value


def _image_shape(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must be a Bx1xHxW tensor")
    if value.shape[0] < 1 or value.shape[1] != 1 or min(value.shape[2:]) < 1:
        raise ValueError(f"{name} must be a nonempty Bx1xHxW tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must have a floating dtype")


@torch.no_grad()
def edge_importance(clean_images: torch.Tensor, edge_weight: float = 0.0) -> torch.Tensor:
    """Return detached FP32 weights with mean 1 independently for each image.

    ``clean_images`` uses the existing dataset normalization [-1, 1]. Adjacent
    absolute differences mark both sides of each edge, then a 3x3 max filter
    includes nearby anti-aliased stroke pixels. Contrast is bounded to [0, 1].
    Raw weights are 1 + edge_weight * edge, then divided by their per-image
    mean. Constant images consequently have uniform weight 1, with no division
    by a vanishing edge count. Padding does not create artificial frame edges.
    """
    strength = validate_edge_weight(edge_weight)
    _image_shape(clean_images, "clean_images")
    clean = clean_images.detach().float()
    if strength == 0:
        return torch.ones_like(clean)
    edge = torch.zeros_like(clean)
    if clean.shape[-1] > 1:
        dx = (clean[..., 1:] - clean[..., :-1]).abs()
        edge = torch.maximum(edge, F.pad(dx, (0, 1, 0, 0)))
        edge = torch.maximum(edge, F.pad(dx, (1, 0, 0, 0)))
    if clean.shape[-2] > 1:
        dy = (clean[..., 1:, :] - clean[..., :-1, :]).abs()
        edge = torch.maximum(edge, F.pad(dy, (0, 0, 0, 1)))
        edge = torch.maximum(edge, F.pad(dy, (0, 0, 1, 0)))
    edge = F.max_pool2d(edge, kernel_size=3, stride=1, padding=1).mul(0.5).clamp(0, 1)
    weights = 1 + strength * edge
    return weights / weights.mean(dim=(1, 2, 3), keepdim=True)


def edge_weighted_flow_loss(
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    clean_images: torch.Tensor,
    edge_weight: float = 0.0,
) -> torch.Tensor:
    """FP32 flow-velocity MSE with optional normalized edge importance.

    Exact shapes and devices are required, so accidental batch/channel/size
    broadcasting cannot silently alter the objective. Importance weights are
    detached from the clean training pixels. The zero-strength branch is exactly
    the historical ``F.mse_loss(prediction.float(), target.float())``, including
    its operation order and gradients for both arguments.
    """
    strength = validate_edge_weight(edge_weight)
    for value, name in ((prediction, "prediction"), (target_velocity, "target_velocity"),
                        (clean_images, "clean_images")):
        _image_shape(value, name)
    if prediction.shape != target_velocity.shape or prediction.shape != clean_images.shape:
        raise ValueError("prediction, target_velocity and clean_images must have identical shapes")
    if prediction.device != target_velocity.device or prediction.device != clean_images.device:
        raise ValueError("prediction, target_velocity and clean_images must share a device")
    if strength == 0:
        return F.mse_loss(prediction.float(), target_velocity.float())
    weights = edge_importance(clean_images, strength)
    error = prediction.float() - target_velocity.float()
    return (error.square() * weights).mean()
