"""Native256 interaction-state training pairs, never an inference renderer.

This optional dataset addresses missing empty-chat/composer/toggle-state
supervision. It does not alter an existing dataset, checkpoint, training job or
serving model. Existing caption augmentation must be skipped for rows marked
``caption_policy=verbatim-interaction-state`` before using these examples.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

from .dataset import PixelCanvas, build_vocab


VERSION = "native256-interaction-pairs-v1"
CAPTION_POLICY = "verbatim-interaction-state"
NATIVE_SIZE = 256
MAX_PROMPT_BYTES = 190
SWITCH_LABELS = ("Notifications", "Sound", "Privacy")


@dataclass
class Sample:
    pixels: np.ndarray
    metadata: dict


@dataclass
class InteractionPair:
    pair_id: str
    bounds: tuple[int, int, int, int]
    before: Sample
    after: Sample


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError("layout seed must be an integer in 0..2**32-1")
    return value


def split_for_seed(seed, val_fraction=.05):
    """All families, states and pair members of one layout stay together."""
    _seed(seed)
    if isinstance(val_fraction, bool) or not isinstance(val_fraction, (int, float)) or not math.isfinite(val_fraction) or not 0 <= val_fraction <= .5:
        raise ValueError("val_fraction must be finite and in 0..0.5")
    digest = hashlib.sha256(f"{VERSION}:split:{seed}".encode("ascii")).digest()
    return "val" if int.from_bytes(digest[:8], "big") < int(val_fraction * 2**64) else "train"


def _canvas(seed, kind):
    rng = random.Random(seed)
    dark = rng.random() < .15
    canvas = PixelCanvas(rng, dark=dark, size=NATIVE_SIZE)
    canvas.padding = rng.choice((8, 10, 12))
    canvas.radius = rng.choice((2, 4, 6))
    canvas.linewidth = 1
    canvas.header = 30
    canvas.sidebar = 64
    title = f"CEMIVO {seed:08X}"
    p = canvas.padding
    canvas.rect(p, p, 256-2*p-1, 256-2*p-1, radius=5)
    canvas.line(p, p+canvas.header, 255-p, p+canvas.header)
    canvas.text(p+9, p+8, title, 13, True, maxw=232-2*p)
    if kind == "chat":
        canvas.line(p+64, p+30, p+64, 255-p)
        _button(canvas, p+7, p+43, 49, 24, "New")
        canvas.text(p+10, p+80, "Chat", 12, True)
    return canvas, title, dark


def _button(canvas, x, y, width, height, text, primary=False):
    canvas.rect(x, y, width-1, height-1, fill=canvas.fg if primary else canvas.bg, radius=3)
    visible = canvas.text(x+width/2, y+4, text, 12, True, center=True,
        fill=canvas.bg if primary else canvas.fg, maxw=width-6)
    if visible != text:
        raise ValueError("State dataset must not shorten a captioned button")


def _chat(seed, messages, composer):
    c, title, dark = _canvas(seed, "chat")
    p = c.padding
    x, right = p+72, 247-p
    y, bottom = p+39, 247-p
    width = right-x
    compose_y = bottom-53
    message_box = (x, y, right+1, compose_y-4)
    if messages:
        c.rect(x+7, y+19, width-16, 31, radius=5)
        c.text(x+14, y+26, "Hello", 14)
    else:
        c.text(x+width/2, y+48, "No messages", 12, center=True, maxw=width-12)
    c.rect(x, compose_y, width, 52, radius=4)
    c.text(x+6, compose_y+6, composer or "Message", 12, maxw=width-12)
    _button(c, x+5, compose_y+28, 68, 21, "Attach")
    _button(c, right-68, compose_y+28, 63, 21, "Send", primary=True)
    # Only the text line changes in the composer pair, not its buttons/border.
    composer_box = (x+3, compose_y+3, right-2, compose_y+25)
    return c, title, dark, message_box, composer_box


def _switch(canvas, bounds, state):
    x0, y0, x1, y1 = bounds
    canvas.d.rounded_rectangle((x0, y0, x1-1, y1-1), radius=8,
        fill=canvas.bg, outline=canvas.fg, width=1)
    center_x = x1-9 if state == "on" else x0+8
    canvas.d.ellipse((center_x-5, y0+3, center_x+5, y1-4), fill=canvas.fg)


def _settings(seed, states):
    c, title, dark = _canvas(seed, "settings")
    p = c.padding
    c.text(p+12, p+42, "Settings", 16, True)
    boxes = []
    for index, (label, state) in enumerate(zip(SWITCH_LABELS, states)):
        y = p+78+index*35
        c.text(p+13, y+1, label, 12)
        bounds = (208-p, y, 240-p, y+16)
        _switch(c, bounds, state)
        boxes.append(bounds)
    _button(c, p+13, 214-p, 66, 25, "Back")
    _button(c, 175-p, 214-p, 66, 25, "Save", primary=True)
    return c, title, dark, boxes


def _style(dark):
    return "black background white lines" if dark else "white background black lines"


def _metadata(seed, kind, title, dark, prompt, state, split, canvas):
    if not isinstance(prompt, str) or not prompt or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ValueError("Truthful state caption exceeded 190 UTF-8 bytes")
    if any(row["requested"] != row["visible"] or row["font_px"] < 12 for row in canvas.drawn_text):
        raise ValueError("State dataset requires complete labels and native fonts >=12px")
    return {"seed": seed, "layout_seed": seed, "kind": kind, "title": title,
        "language": "en", "dark": dark, "sidebar": kind == "chat",
        "buttons": ["New", "Attach", "Send"] if kind == "chat" else ["Back", "Save"],
        "prompt": prompt, "state": state, "split": split, "caption_policy": CAPTION_POLICY,
        "dataset_version": VERSION, "render_native_size": 256,
        "rendering_version": 4, "render_method": "native-vector-and-font-rasterization",
        "layout_coordinate_size": 256, "drawn_text": list(canvas.drawn_text)}


def _pair(seed, action, bounds, before, after, event):
    pair_id = hashlib.sha256(f"{VERSION}:{seed}:{action}".encode("ascii")).hexdigest()[:24]
    x0, y0, x1, y1 = bounds
    mask = np.zeros((256, 256), dtype=bool)
    mask[y0:y1, x0:x1] = True
    if not np.array_equal(before.pixels[~mask], after.pixels[~mask]):
        raise ValueError("A paired target changed pixels outside its declared region")
    changed = int(np.count_nonzero(before.pixels != after.pixels))
    if changed == 0:
        raise ValueError("Interaction pair has no real visible change")
    outside_hash = hashlib.sha256(before.pixels[~mask].tobytes()).hexdigest()
    for role, sample in (("before", before), ("after", after)):
        sample.metadata.update(pair_id=pair_id, pair_role=role, action=action,
            target_bounds=list(bounds), event=dict(event), preserve_outside=True,
            outside_changed_pixels=0, outside_sha256=outside_hash, changed_pixels=changed,
            content_sha256=hashlib.sha256(sample.pixels.tobytes()).hexdigest())
    return InteractionPair(pair_id, tuple(bounds), before, after)


def generate_pairs(seed, val_fraction=.05):
    """Return three two-image pairs for one native256 layout seed.

    Empty -> one Hello message changes only the message region. Empty composer
    -> Hi changes only its text line. One settings switch flips while all other
    switch states, labels and pixels remain equal. No images are upscaled.
    """
    _seed(seed)
    split = split_for_seed(seed, val_fraction)
    pairs = []
    for action, before_state, after_state in (
            ("chat_message", (False, ""), (True, "")),
            ("chat_composer", (False, ""), (False, "Hi"))):
        samples = []
        box = None
        for messages, composer in (before_state, after_state):
            c, title, dark, message_box, composer_box = _chat(seed, messages, composer)
            message_caption = "one Hello bubble" if messages else "empty messages, No messages label"
            input_caption = "Hi input" if composer else "empty input, Message label"
            change = "Only message area changes." if action == "chat_message" else "Only input text changes."
            prompt = f"Chat UI, {_style(dark)}, title {title}, {message_caption}, {input_caption}, New, Attach, Send. {change}"
            meta = _metadata(seed, "chat", title, dark, prompt,
                {"messages": ["Hello"] if messages else [], "composer": composer}, split, c)
            samples.append(Sample(np.asarray(c.image, dtype=np.uint8).copy(), meta))
            box = message_box if action == "chat_message" else composer_box
        event = {"type": "submit", "text": "Send Hello"} if action == "chat_message" else {
            "type": "click", "x": (box[0]+box[2])//2, "y": (box[1]+box[3])//2,
            "text": "Hi", "input_method": "contextmenu"}
        pairs.append(_pair(seed, action, box, *samples, event))
    rng = random.Random(seed ^ 0x6A09E667)
    states = [rng.choice(("on", "off")) for _ in SWITCH_LABELS]
    selected = rng.randrange(3)
    changed_states = list(states)
    changed_states[selected] = "off" if states[selected] == "on" else "on"
    samples = []
    for current in (states, changed_states):
        c, title, dark, boxes = _settings(seed, current)
        state_caption = ", ".join(f"{label} {value}" for label, value in zip(SWITCH_LABELS, current))
        prompt = f"Settings UI, {_style(dark)}, title {title}, {state_caption}, Back, Save. Only {SWITCH_LABELS[selected]} switch changes."
        meta = _metadata(seed, "settings", title, dark, prompt,
            dict(zip(SWITCH_LABELS, current)), split, c)
        samples.append(Sample(np.asarray(c.image, dtype=np.uint8).copy(), meta))
    box = boxes[selected]
    pairs.append(_pair(seed, "toggle", box, *samples,
        {"type": "click", "x": (box[0]+box[2])//2, "y": (box[1]+box[3])//2}))
    return pairs


def export_dataset(output, groups=1000, seed=20261026, val_fraction=.05):
    """Write a new, standalone Dataset-compatible shard; never append/overwrite."""
    _seed(seed)
    if isinstance(groups, bool) or not isinstance(groups, int) or groups < 1 or seed+groups > 2**32:
        raise ValueError("groups must be positive and remain within the layout seed range")
    split_for_seed(seed, val_fraction)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    count = groups*6
    pixels = np.lib.format.open_memmap(output/"images-00000.npy.partial", mode="w+", dtype=np.uint8, shape=(count,256,256))
    counts = {"train": 0, "val": 0}
    hashes = {"train": set(), "val": set()}
    try:
        with (output/"metadata-00000.jsonl.partial").open("x", encoding="utf-8", newline="\n") as handle:
            index = 0
            for layout_seed in range(seed, seed+groups):
                for pair in generate_pairs(layout_seed, val_fraction):
                    for sample in (pair.before, pair.after):
                        row = dict(sample.metadata)
                        row.update(index=index, offset=index, shard=0,
                            paired_sample_index=index+1 if row["pair_role"] == "before" else index-1)
                        opposite = "val" if row["split"] == "train" else "train"
                        if row["content_sha256"] in hashes[opposite]:
                            raise ValueError("Identical image would cross train/validation splits")
                        hashes[row["split"]].add(row["content_sha256"])
                        pixels[index] = sample.pixels
                        handle.write(json.dumps(row, ensure_ascii=False)+"\n")
                        counts[row["split"]] += 1
                        index += 1
        pixels.flush()
    finally:
        del pixels
    (output/"images-00000.npy.partial").replace(output/"images-00000.npy")
    (output/"metadata-00000.jsonl.partial").replace(output/"metadata-00000.jsonl")
    (output/"vocab.json").write_text(json.dumps(build_vocab(), indent=2), encoding="utf-8")
    manifest = {"version": 1, "dataset_version": VERSION, "format": "npy-uint8-grayscale-shards",
        "size": 256, "render_native_size": 256, "layout_coordinate_size": 256,
        "render_method": "native-vector-and-font-rasterization", "rendering_version": 4,
        "dtype": "uint8", "normalization": "image / 127.5 - 1", "base_seed": seed,
        "count": count, "groups": groups, "train_count": counts["train"], "val_count": counts["val"],
        "val_fraction": val_fraction, "max_length": 192, "vocab": "vocab.json",
        "caption_policy": CAPTION_POLICY,
        "training_integration_required": "Skip existing caption augmentation for verbatim-interaction-state rows; these examples are not loaded by the running job.",
        "split_policy": "SHA256 of layout seed; all families, states and paired members kept in one split; cross-split identical pixels rejected",
        "quality_claim": "Training targets only. Existing model is not changed or verified by generating this dataset.",
        "provenance": "Original procedural native256 UI state training pixels; no pretrained images or weights; not an inference renderer.",
        "shards": [{"images": "images-00000.npy", "metadata": "metadata-00000.jsonl", "count": count,
            "start_index": 0, "train_count": counts["train"], "val_count": counts["val"]}]}
    temporary = output/"manifest.json.partial"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output/"manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New standalone dataset directory; existing paths refused")
    parser.add_argument("--groups", type=int, default=1000, help="Each group contains three pairs, six samples")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--val-fraction", type=float, default=.05)
    args = parser.parse_args()
    manifest = export_dataset(args.output, args.groups, args.seed, args.val_fraction)
    print(json.dumps({"output": str(args.output.resolve()), "count": manifest["count"],
        "train": manifest["train_count"], "val": manifest["val_count"],
        "training_changed": False, "model_quality_claim": "none; training targets only"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
