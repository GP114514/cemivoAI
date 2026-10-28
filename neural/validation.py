"""Paired held-out flow diagnostics, without claiming an image quality score."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import tokenize


@torch.inference_mode()
def validate_conditioning(model, dataset, resolution, device, amp_enabled=True,
                          samples=15, seed=90625):
    """Compare correct/wrong/empty captions with identical held-out image/noise/t.

    A positive wrong-minus-correct MSE margin is a conditioning diagnostic. It
    does not establish readable lettering, correct layouts or semantic accuracy.
    All randomness is local; the training RNG/sampler state is left untouched.
    """
    device = torch.device(device)
    count = min(int(samples), len(dataset))
    if count < 2:
        raise ValueError("At least two validation examples are required")
    selection = np.random.default_rng(seed).permutation(len(dataset))
    indices, kinds = [], set()
    # Prefer coverage over drawing many near-identical examples of one family.
    for index in selection:
        row = dataset.rows[int(index)]
        if row["kind"] not in kinds:
            indices.append(int(index))
            kinds.add(row["kind"])
        if len(indices) == count:
            break
    for index in selection:
        if len(indices) == count:
            break
        if int(index) not in indices:
            indices.append(int(index))
    rows = [dataset[index] for index in indices]
    records = []
    was_training = model.training
    model.eval()
    try:
        for position, row in enumerate(rows):
            wrong = rows[(position + 1) % len(rows)]
            from .captions import augment_caption
            metadata = dataset.rows[indices[position]]
            natural = (row["prompt"] if metadata.get("caption_policy") == "verbatim-interaction-state" else
                augment_caption(metadata,np.random.default_rng(seed+3000+position)))
            captions = [row["prompt"], natural, wrong["prompt"], ""]
            tokens = tokenize(captions, model.config.text_max_length, device)
            image = torch.from_numpy(np.asarray(row["image"]).copy()).unsqueeze(0).to(device)
            if image.shape[-2:] != (resolution, resolution):
                image = F.interpolate(image, (resolution, resolution), mode="bilinear",
                                      align_corners=False, antialias=True)
            rng = torch.Generator(device=device).manual_seed(seed + position)
            noise = torch.randn(image.shape, generator=rng, device=device)
            target = image - noise
            for scalar_time in (0.1, 0.5, 0.9):
                state = ((1 - scalar_time) * noise + scalar_time * image).repeat(4, 1, 1, 1)
                times = torch.full((4,), scalar_time, device=device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=amp_enabled and device.type == "cuda"):
                    prediction = model(state, times, tokens)
                errors = (prediction.float() - target).square().mean(dim=(1, 2, 3)).cpu().tolist()
                records.append({"index": row["index"], "kind": row["kind"], "t": scalar_time,
                                "correct_mse": errors[0], "natural_mse":errors[1],
                                "natural_prompt":natural,"natural_caption_policy":metadata.get("caption_policy", "family-augmentation"),"wrong_mse": errors[2],
                                "blank_mse": errors[3]})
        def average(key, t=None):
            return float(np.mean([record[key] for record in records if t is None or record["t"] == t]))
        correct, wrong, blank = (average(key) for key in ("correct_mse", "wrong_mse", "blank_mse"))
        return {"samples": count, "families": sorted(kinds), "native_size": resolution,
                "correct_mse": correct, "wrong_mse": wrong, "blank_mse": blank,
                "natural_mse":average("natural_mse"),
                "natural_minus_correct_mse":average("natural_mse")-correct,
                "wrong_minus_correct_mse": wrong - correct,
                "blank_minus_correct_mse": blank - correct,
                "by_noise_time": {str(t): {key: average(key, t) for key in
                    ("correct_mse", "natural_mse", "wrong_mse", "blank_mse")} for t in (0.1, 0.5, 0.9)},
                "details": records, "seed": seed,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "interpretation": "Paired held-out velocity MSE only. These are not quality scores, OCR accuracy or proof of correct prompt following."}
    finally:
        model.train(was_training)


def write_validation(directory, report, step):
    directory = Path(directory) / "validation"
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(report, checkpoint_step=int(step))
    # Reuse the same Windows sharing-conflict handling as all other train files.
    from .train import atomic_json
    atomic_json(directory / f"step-{step:08}.json", report)
    atomic_json(directory / "latest.json", report)
    with (directory / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({key: value for key, value in report.items() if key != "details"},
                                ensure_ascii=False) + "\n")
