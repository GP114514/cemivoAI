"""Randomly initialized native-pixel neural head for experimental toggle edits.

This module contains a CNN and lossless crop padding only. It never draws a
track/thumb, thresholds a candidate, resizes input pixels, or falls back to a
procedural control. A caller must independently verify state and use the
original image for exact exterior compositing. Untrained output is not usable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FORMAT_VERSION = "cemivo-toggle-head-v1"
CANVAS_HEIGHT = 40
CANVAS_WIDTH = 64
MIN_HEIGHT, MAX_HEIGHT = 6, 36
MIN_WIDTH, MAX_WIDTH = 12, 64


@dataclass(frozen=True)
class ToggleHeadConfig:
    channels: int = 32
    dilations: tuple[int, ...] = (1, 2, 4, 8)

    def __post_init__(self):
        if type(self.channels) is not int or not 8 <= self.channels <= 64 or self.channels % 4:
            raise ValueError("channels must be a multiple of four in 8..64")
        if (not isinstance(self.dilations, tuple) or not 1 <= len(self.dilations) <= 6 or
                any(type(value) is not int or not 1 <= value <= 16 for value in self.dilations)):
            raise ValueError("dilations must contain 1..6 integers in 1..16")

    def to_dict(self):
        value = asdict(self)
        value["dilations"] = list(self.dilations)
        return value

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"channels", "dilations"}:
            raise ValueError("invalid toggle-head configuration")
        if not isinstance(value["dilations"], (tuple, list)):
            raise ValueError("invalid toggle-head dilations")
        return cls(channels=value["channels"], dilations=tuple(value["dilations"]))


def _state(value):
    if isinstance(value, str):
        if value not in ("off", "on"):
            raise ValueError("target state must be off/on or integer 0/1")
        return int(value == "on")
    if type(value) is not int or value not in (0, 1):
        raise ValueError("target state must be off/on or integer 0/1")
    return value


def _crop(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.dtype != torch.uint8:
            raise ValueError("native crop tensors must be CPU uint8")
        value = value.detach().numpy()
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise ValueError("native crops must be uint8 arrays")
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError("native crops must have one grayscale channel")
    height, width = value.shape
    if not MIN_HEIGHT <= height <= MAX_HEIGHT or not MIN_WIDTH <= width <= MAX_WIDTH:
        raise ValueError("native crop size must be height 6..36 and width 12..64")
    return value


def encode_batch(crops: Sequence[np.ndarray], target_states: Sequence[str | int], device=None):
    """Copy native uint8 crops to a 40x64 canvas without any resampling.

    Returns pixels in [-1,1], a boolean valid-pixel mask, int64 target states
    (off=0/on=1), and native (height,width) sizes. Padding is normalized zero;
    it is never part of a delivered image or a supervised target by itself.
    All returned tensors own their storage and cannot mutate caller arrays.
    """
    if not isinstance(crops, (list, tuple)) or not crops:
        raise ValueError("crops must be a nonempty list or tuple")
    if not isinstance(target_states, (list, tuple)) or len(crops) != len(target_states):
        raise ValueError("one target state is required for every crop")
    native = [_crop(value) for value in crops]
    states = [_state(value) for value in target_states]
    pixels = torch.zeros((len(native), 1, CANVAS_HEIGHT, CANVAS_WIDTH), dtype=torch.float32)
    valid = torch.zeros_like(pixels, dtype=torch.bool)
    sizes = []
    for index, value in enumerate(native):
        height, width = value.shape
        sizes.append((height, width))
        pixels[index, 0, :height, :width] = torch.from_numpy(value.copy()).float() / 127.5 - 1
        valid[index, 0, :height, :width] = True
    return {"pixels": pixels.to(device=device), "valid": valid.to(device=device),
        "target_states": torch.tensor(states, dtype=torch.long, device=device), "sizes": sizes}


class _Residual(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        groups = math.gcd(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.condition = nn.Linear(channels, 4 * channels)

    def forward(self, value, condition):
        scale1, shift1, scale2, shift2 = self.condition(condition).chunk(4, dim=1)
        hidden = self.norm1(value) * (1 + scale1[:, :, None, None]) + shift1[:, :, None, None]
        hidden = self.conv1(F.silu(hidden))
        hidden = self.norm2(hidden) * (1 + scale2[:, :, None, None]) + shift2[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return value + hidden / math.sqrt(2)


class ToggleHead(nn.Module):
    """Small conditional CNN; every weight starts from random initialization.

    The default dilated receptive field spans the full 64-pixel crop width.
    It predicts all native pixels jointly, so the target state can move a thumb
    across a track rather than relying on a local per-pixel transformation.
    """
    def __init__(self, config: ToggleHeadConfig | dict | None = None):
        super().__init__()
        if config is None:
            config = ToggleHeadConfig()
        elif isinstance(config, dict):
            config = ToggleHeadConfig.from_dict(config)
        if not isinstance(config, ToggleHeadConfig):
            raise ValueError("expected a ToggleHeadConfig")
        self.config = config
        self.input = nn.Conv2d(2, config.channels, 3, padding=1)
        self.state_embedding = nn.Embedding(2, config.channels)
        self.blocks = nn.ModuleList(_Residual(config.channels, dilation) for dilation in config.dilations)
        self.output_norm = nn.GroupNorm(math.gcd(8, config.channels), config.channels)
        self.output = nn.Conv2d(config.channels, 1, 3, padding=1)
        if self.parameter_count >= 1_000_000:
            raise ValueError("toggle-head must stay below one million parameters")

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, pixels, valid, target_states):
        if (not isinstance(pixels, torch.Tensor) or not pixels.is_floating_point() or
                pixels.ndim != 4 or pixels.shape[1:] != (1, CANVAS_HEIGHT, CANVAS_WIDTH) or
                pixels.shape[0] < 1):
            raise ValueError("pixels must be floating Bx1x40x64")
        if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.shape != pixels.shape or
                valid.device != pixels.device or not bool(valid.flatten(1).any(dim=1).all())):
            raise ValueError("valid must be a nonempty boolean mask matching pixels")
        if (not isinstance(target_states, torch.Tensor) or target_states.dtype != torch.long or
                target_states.shape != (pixels.shape[0],) or target_states.device != pixels.device or
                not bool(((target_states == 0) | (target_states == 1)).all())):
            raise ValueError("target_states must be B int64 states in 0..1")
        if not bool(torch.isfinite(pixels).all()) or bool((pixels.abs() > 1).any()):
            raise ValueError("pixels must be finite and in -1..1")
        hidden = self.input(torch.cat((pixels, valid.to(dtype=pixels.dtype)), dim=1))
        condition = self.state_embedding(target_states)
        for block in self.blocks:
            hidden = block(hidden, condition)
        # The image is predicted by the CNN. There is no mask-to-shape drawing,
        # thresholding or procedural reconstruction in the neural output path.
        return torch.tanh(self.output(F.silu(self.output_norm(hidden) + condition[:, :, None, None])))

    @torch.inference_mode()
    def predict_native(self, crop, target_state):
        """Return neural uint8 pixels at the exact original crop dimensions."""
        parameter = next(self.parameters())
        batch = encode_batch([crop], [target_state], device=parameter.device)
        was_training = self.training
        self.eval()
        try:
            prediction = self(batch["pixels"].to(dtype=parameter.dtype), batch["valid"], batch["target_states"])
            if not bool(torch.isfinite(prediction).all()):
                raise ValueError("toggle head produced nonfinite pixels")
            height, width = batch["sizes"][0]
            result = ((prediction[0, 0, :height, :width].float() + 1) * 127.5).round().clamp(0, 255)
            return result.to(dtype=torch.uint8).cpu().numpy().copy()
        finally:
            self.train(was_training)


def checkpoint_payload(model, step, *, heldout_validated=False, validation=None):
    """Build a self-describing checkpoint; validation defaults to false."""
    if not isinstance(model, ToggleHead) or type(step) is not int or step < 0:
        raise ValueError("checkpoint needs a ToggleHead and a nonnegative integer step")
    if type(heldout_validated) is not bool or (validation is not None and not isinstance(validation, dict)):
        raise ValueError("invalid validation metadata")
    if heldout_validated and not validation:
        raise ValueError("validated checkpoints require independent validation evidence")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if any(not value.is_floating_point() or not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("checkpoint model tensors must be finite floating point")
    return {"format_version": FORMAT_VERSION, "model_config": model.config.to_dict(),
        "state_dict": state,
        "step": step, "parameter_count": model.parameter_count,
        "native_canvas": [CANVAS_HEIGHT, CANVAS_WIDTH],
        "provenance": {"random_initialization": True, "pretrained_weights": False,
            "architecture": "native-pixel conditional dilated CNN"},
        "heldout_validated": heldout_validated, "validation": validation or {}}


def save_checkpoint(path, model, step, *, heldout_validated=False, validation=None):
    """Atomically save weights; does not change another model or training job."""
    payload = checkpoint_payload(model, step, heldout_validated=heldout_validated, validation=validation)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, device="cpu", *, require_validated=False):
    """Load only tensor/primitive data, check schema, and load weights strictly.

    Returns (model, metadata). Training code is responsible for its own optimizer
    and RNG checkpoint if exact continuation is required.
    """
    if type(require_validated) is not bool:
        raise ValueError("require_validated must be boolean")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"format_version", "model_config", "state_dict", "step", "parameter_count",
        "native_canvas", "provenance", "heldout_validated", "validation"}
    if not isinstance(payload, dict) or set(payload) != required or payload["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported or incomplete toggle-head checkpoint")
    if type(payload["step"]) is not int or payload["step"] < 0:
        raise ValueError("invalid training step")
    provenance = payload["provenance"]
    if (not isinstance(provenance, dict) or provenance.get("random_initialization") is not True or
            provenance.get("pretrained_weights") is not False or payload["native_canvas"] != [CANVAS_HEIGHT, CANVAS_WIDTH]):
        raise ValueError("invalid native/random-weight checkpoint provenance")
    if type(payload["heldout_validated"]) is not bool or not isinstance(payload["validation"], dict):
        raise ValueError("invalid validation metadata")
    if payload["heldout_validated"] and not payload["validation"]:
        raise ValueError("missing held-out validation evidence")
    if require_validated and not payload["heldout_validated"]:
        raise ValueError("toggle-head checkpoint has not passed held-out validation")
    # Loading a checkpoint must not advance the caller's CPU initialization RNG.
    with torch.random.fork_rng(devices=[]):
        model = ToggleHead(ToggleHeadConfig.from_dict(payload["model_config"]))
    if type(payload["parameter_count"]) is not int or payload["parameter_count"] != model.parameter_count:
        raise ValueError("toggle-head parameter count does not match its configuration")
    state = payload["state_dict"]
    if (not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) or not value.is_floating_point() or
            not bool(torch.isfinite(value).all()) for value in state.values())):
        raise ValueError("invalid or nonfinite model tensors")
    model.load_state_dict(state, strict=True)
    model.to(device=device)
    model.eval()
    metadata = {name: value for name, value in payload.items() if name != "state_dict"}
    return model, metadata
