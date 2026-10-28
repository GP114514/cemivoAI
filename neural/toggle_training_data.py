"""Synthetic native-pixel pairs for training the separate toggle editing head.

TRAINING ONLY. Inference must never import or call this module. No fonts,
pretrained weights, real model outputs, resizing or external images are used.
Each uint8 [1,H,W] pair preserves track geometry and flips only its state.
"""
from io import BytesIO
import hashlib
from numbers import Integral

import numpy as np
from PIL import Image, ImageDraw

from stable_edit import measure_toggle_state

SPLIT_SEEDS = {"train": 2026102601, "val": 2026102602}
VALIDATION_CANVAS_SIZE = 256


def split_rng(split, seed=0):
    """Independent, repeatable seed namespaces; never touch global RNG state."""
    if split not in SPLIT_SEEDS:
        raise ValueError("split must be train or val")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return np.random.default_rng(np.random.SeedSequence([SPLIT_SEEDS[split], int(seed)]))


def _png(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def measurable_state(crop, bounds, background):
    """Measure a native crop without scaling or weakening the production gates.

    Production measure_toggle_state limits the switch to <=3% of a full frame.
    Embed at (96,96) in a plain 256px validation canvas so that restriction is
    applied legitimately; only the unchanged crop pixels determine its shape.
    This canvas is a diagnostic input and is never a training target.
    """
    pixels = np.asarray(crop)
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.dtype != np.uint8 or pixels.ndim != 2:
        raise ValueError("crop must be uint8 HxW or 1xHxW")
    canvas = Image.new("L", (VALIDATION_CANVAS_SIZE, VALIDATION_CANVAS_SIZE), int(background))
    canvas.paste(Image.fromarray(pixels), (96, 96))
    return measure_toggle_state(_png(canvas), [int(value)+96 for value in bounds])


def _draw(width, height, padding, state, background, foreground):
    image = Image.new("L", (width, height), background)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = padding, padding, width-padding, height-padding
    track_height = bottom-top
    draw.rounded_rectangle((left, top, right-1, bottom-1), radius=track_height//2,
        fill=foreground if state == "on" else background, outline=foreground, width=1)
    radius = max(1, (track_height-4)//2)
    center_x = right-1-track_height//2 if state == "on" else left+track_height//2
    center_y = top+track_height//2
    draw.ellipse((center_x-radius, center_y-radius, center_x+radius, center_y+radius),
        fill=background if state == "on" else foreground)
    return np.asarray(image, dtype=np.uint8).copy()[None], [left, top, right, bottom]


def make_example(rng, max_attempts=200):
    """Return one independently verified opposite-state pair.

    API: source/target uint8[1,H,W], bounds [x0,y0,x1,y1] relative to the
    crop, current_state/target_state strings, dark bool, padding int,
    background/foreground intensities, attempts. W=16..48, H=8..24.
    Source has mild per-pixel gray variation that never crosses threshold 128;
    target is a complete, crisp native drawing with the same geometry.
    Unmeasurable pairs are discarded, with at most max_attempts<=200 attempts.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, Integral) or not 1 <= max_attempts <= 200:
        raise ValueError("max_attempts must be an integer in 1..200")
    for attempt in range(1, int(max_attempts)+1):
        width, height, padding = int(rng.integers(16, 49)), int(rng.integers(8, 25)), int(rng.integers(1, 3))
        tw, th = width-2*padding, height-2*padding
        if th < 6 or not 1.7 <= tw/th <= 3.8:
            continue
        dark = bool(rng.integers(0, 2))
        high, low = int(rng.integers(240, 256)), int(rng.integers(0, 16))
        background, foreground = (low, high) if dark else (high, low)
        current = "on" if bool(rng.integers(0, 2)) else "off"
        target_state = "off" if current == "on" else "on"
        source, bounds = _draw(width, height, padding, current, background, foreground)
        target, target_bounds = _draw(width, height, padding, target_state, background, foreground)
        # Native gray jitter, not scaling/blur/repair: every binary component is
        # identical before and after this augmentation. Damaged tracks are not
        # accepted as labels for a definite source state.
        jitter = rng.integers(-5, 6, size=source.shape, dtype=np.int16)
        source = np.clip(source.astype(np.int16)+jitter, 0, 255).astype(np.uint8)
        if (bounds != target_bounds or measurable_state(source, bounds, background) != current
                or measurable_state(target, bounds, background) != target_state):
            continue
        return {"source": source, "target": target, "bounds": bounds,
            "current_state": current, "target_state": target_state, "dark": dark,
            "padding": padding, "background": background, "foreground": foreground,
            "attempts": attempt}
    raise RuntimeError(f"Could not draw a measurable native toggle pair in {max_attempts} attempts")


def make_splits(train_count, val_count, seed=0):
    """Build finite deterministic train/val sets with disjoint source/targets.

    Separate RNG namespaces establish independent sampling; exact pixel hashes
    additionally prevent accidental source/target overlap or duplicate rows.
    This accepts no image paths and cannot ingest the real neural heldout crop.
    """
    for count in (train_count, val_count):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("split counts must be nonnegative integers")
    seen, result = set(), {}
    for split, count in (("train", train_count), ("val", val_count)):
        rng, examples = split_rng(split, seed), []
        for _ in range(int(count)):
            for attempt in range(200):
                example = make_example(rng)
                fingerprints = {hashlib.sha256(str(example[key].shape).encode("ascii")+
                    example[key].tobytes()).digest() for key in ("source", "target")}
                if not fingerprints & seen:
                    seen.update(fingerprints); examples.append(example)
                    break
            else:
                raise RuntimeError("Could not obtain a unique synthetic pair in 200 attempts")
        result[split] = examples
    return result
