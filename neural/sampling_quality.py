"""Optional FP32-state rectified-flow solvers for controlled quality comparisons.

The production sampler in ``model.py`` is not changed. Initial Gaussian noise
is drawn in the model parameter dtype with the historical local RNG/seed
convention, then promoted to FP32. Every integrator state and velocity update
remains FP32; only each model input is cast back to its parameter dtype.

Euler uses ``steps`` model forward calls. Midpoint and Heun each use
``2 * steps`` calls. CFG other than 1 batches conditional and unconditional
predictions, doubling evaluated images per call. Compare equal forward work,
not merely equal ``steps``. Heun evaluates its last predictor at t=1; whether
that helps trained image quality must be measured, not assumed.

All outputs are direct neural pixels. There are no renderers, target images,
OCR weights, sharpening, templates or text overlays in this module.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Integral, Real

import torch

from .model import tokenize


_SOLVERS = ("euler", "midpoint", "heun")
_SCHEDULES = ("uniform", "cosine")


def _options(steps, solver, time_schedule, guidance):
    if isinstance(steps, bool) or not isinstance(steps, Integral) or not 1 <= steps <= 128:
        raise ValueError("steps must be an integer in 1..128")
    if solver not in _SOLVERS:
        raise ValueError("solver must be euler, midpoint or heun")
    if time_schedule not in _SCHEDULES:
        raise ValueError("time_schedule must be uniform or cosine")
    if isinstance(guidance, bool) or not isinstance(guidance, Real):
        raise ValueError("guidance must be a finite number in 0..8")
    guidance = float(guidance)
    if not math.isfinite(guidance) or not 0 <= guidance <= 8:
        raise ValueError("guidance must be a finite number in 0..8")
    return int(steps), guidance


def quality_sampling_info(steps=8, solver="euler", time_schedule="uniform", guidance=1.0):
    """Return JSON-safe compute accounting; this does not score image quality."""
    steps, guidance = _options(steps, solver, time_schedule, guidance)
    calls = steps if solver == "euler" else 2 * steps
    multiplier = 1 if guidance == 1.0 else 2
    return {"solver": solver, "time_schedule": time_schedule, "integration_steps": steps,
        "state_dtype": "float32", "model_forward_calls": calls,
        "cfg_batch_multiplier": multiplier, "velocity_evaluations_per_image": calls * multiplier,
        "guidance": guidance, "final_clamp": [-1, 1]}


def _time_nodes(steps, schedule):
    if schedule == "uniform":
        return [index / steps for index in range(steps + 1)]
    nodes = [(1 - math.cos(math.pi * index / steps)) / 2 for index in range(steps + 1)]
    nodes[0], nodes[-1] = 0.0, 1.0
    return nodes


@torch.inference_mode()
def sample_flow_quality(model, prompts, seed=0, steps=8, size=256, device=None,
                        guidance=1.0, solver="euler", time_schedule="uniform"):
    """Sample using an optional solver, FP32 accumulation and trained velocities.

    ``model`` should already be in evaluation mode on its intended device.
    Model parameters, evaluation/training mode and global RNG are not changed.
    Prompt tokenization follows the project's byte tokenizer, but over-budget
    prompts are rejected rather than silently truncating a quality comparison.
    """
    steps, guidance = _options(steps, solver, time_schedule, guidance)
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(size, bool) or not isinstance(size, Integral) or not 32 <= size <= 1024 or size % 8:
        raise ValueError("size must be an integer multiple of 8 in 32..1024")
    if isinstance(prompts, str):
        prompts = [prompts]
    elif isinstance(prompts, Sequence) and not isinstance(prompts, (bytes, bytearray)):
        prompts = list(prompts)
    else:
        raise ValueError("prompts must be a string or a sequence of strings")
    if not prompts or any(not isinstance(value, str) or not value.strip() for value in prompts):
        raise ValueError("prompts must contain nonempty strings")
    max_length = model.config.text_max_length
    if type(max_length) is not int or max_length < 2:
        raise ValueError("model text_max_length must be an integer >=2")
    try:
        if any(len(value.encode("utf-8")) > max_length - 2 for value in prompts):
            raise ValueError("prompt exceeds model UTF-8 byte budget")
    except UnicodeEncodeError:
        raise ValueError("prompts must be valid UTF-8 text") from None
    parameter = next(model.parameters())
    if not parameter.is_floating_point():
        raise ValueError("model parameters must have a floating dtype")
    target_device = torch.device(device) if device is not None else parameter.device
    # NeuralGenerator stores torch.device('cuda') while its parameters report
    # cuda:0. Resolve an unspecified accelerator index to that actual device,
    # without consulting/initializing any other GPU.
    if target_device.type == parameter.device.type and (
            target_device.index is None or target_device.type == "cpu"):
        target_device = parameter.device
    if target_device != parameter.device:
        raise ValueError("device must match the model parameter device")
    dtype = parameter.dtype
    rng = torch.Generator(device=target_device).manual_seed(int(seed) % (2**63 - 1))
    # Preserve historical initial-noise bits even when the model is BF16.
    noise = torch.randn((len(prompts), 1, int(size), int(size)), device=target_device,
        dtype=dtype, generator=rng)
    state = noise.float()
    tokens = tokenize(prompts, max_length, target_device)
    if guidance != 1.0:
        empty = tokenize([""] * len(prompts), max_length, target_device)
        tokens = torch.cat((tokens, empty), dim=0)
    context = model.encode_text(tokens)

    def velocity(pixels, time):
        if guidance == 1.0:
            inputs = pixels.to(dtype=dtype)
        else:
            inputs = torch.cat((pixels, pixels), dim=0).to(dtype=dtype)
        times = torch.full((inputs.shape[0],), time, device=target_device, dtype=torch.float32)
        prediction = model(inputs, times, text_context=context)
        if prediction.shape != inputs.shape:
            raise ValueError("model velocity shape must match its pixel input")
        prediction = prediction.float()
        if guidance != 1.0:
            conditional, unconditional = prediction.chunk(2, dim=0)
            prediction = unconditional + guidance * (conditional - unconditional)
        return prediction

    nodes = _time_nodes(steps, time_schedule)
    for start, end in zip(nodes, nodes[1:]):
        delta = end - start
        first = velocity(state, start)
        if solver == "euler":
            state = state + delta * first
        elif solver == "midpoint":
            midpoint = state + (delta / 2) * first
            state = state + delta * velocity(midpoint, start + delta / 2)
        else:
            predictor = state + delta * first
            second = velocity(predictor, end)
            state = state + (delta / 2) * (first + second)
    if not bool(torch.isfinite(state).all()):
        raise FloatingPointError("non-finite neural sampling state")
    return state.clamp(-1, 1)
