"""Compare immutable copies of trained weights without changing the running app.

All image cells are direct neural pixels. Captions are in the JSON report only.
This is a visual review, not an OCR score or an automatic quality gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

from PIL import Image
import torch

from .model import FORMAT_VERSION
from .sample import NeuralGenerator

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = (
    ("chat", "黑白聊天窗口，联系人、聊天气泡、输入框和发送按钮"),
    ("settings", "黑白设置页面，通知开关与保存按钮"),
)


def freeze(source: Path, target: Path):
    target.mkdir()
    temporary = target / "inference.pt.copying"
    # Training publishes complete checkpoints with os.replace. Keeping a single
    # source handle open copies one publication without holding its model lock.
    with source.open("rb") as src, temporary.open("xb") as dst:
        shutil.copyfileobj(src, dst, 4 * 1024**2)
    saved = torch.load(temporary, map_location="cpu", weights_only=False)
    if saved.get("format_version") != FORMAT_VERSION or "model" not in saved:
        raise ValueError("Not a cemivoAI inference checkpoint")
    with temporary.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = {
        "step": int(saved["step"]), "native_size": int(saved.get("native_size", 256)),
        "sha256": digest, "source": str(source),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_initialization": saved.get("provenance", {}).get("random_initialization") is True,
        "pretrained_weights": saved.get("provenance", {}).get("pretrained_weights"),
        "frozen": True, "quality_verified": False,
    }
    del saved
    temporary.replace(target / "inference.pt")
    (target / "snapshot.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def compare(output, device="cpu"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        free, total = torch.cuda.mem_get_info()
        if free < 3 * 1024**3:
            raise RuntimeError("Keep training priority: less than 3 GiB free")
        torch.cuda.set_per_process_memory_fraction(0.10)
        torch.backends.cudnn.benchmark = False
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "device": device,
        "seed": 42, "guidance": 1.0, "sampling_steps": [8, 16, 32],
        "prompts": [{"kind": kind, "prompt": prompt} for kind, prompt in PROMPTS],
        "snapshots": [], "images": [], "review": "pending",
        "limitations": [
            "Direct pixels only; no text overlay, templates, sharpening or upscaling.",
            "One seed and two prompts cannot establish generation accuracy.",
            "No OCR accuracy is calculated; image readability needs explicit visual review.",
            "Training and production service are not restarted or reconfigured.",
            "Observed timings include GPU contention; speed is not a quality score.",
        ],
    }
    frozen = []
    for label, source in (
        ("app", ROOT / "neural" / "live-snapshot" / "inference.pt"),
        ("latest", ROOT / "neural" / "checkpoints" / "inference.pt"),
    ):
        folder = output / label
        metadata = freeze(source, folder)
        report["snapshots"].append(dict(metadata, label=label))
        frozen.append((label, folder, metadata))
    grid = Image.new("L", (256 * 3, 256 * 4), 255)
    for group, (label, folder, metadata) in enumerate(frozen):
        load_start = time.perf_counter()
        generator = NeuralGenerator(folder / "inference.pt", device=device)
        report["snapshots"][group]["load_ms"] = round((time.perf_counter() - load_start) * 1000, 2)
        for prompt_index, (kind, prompt) in enumerate(PROMPTS):
            for column, steps in enumerate((8, 16, 32)):
                start = time.perf_counter()
                image = generator.generate(prompt, seed=42, steps=steps, guidance=1.0)
                filename = f"{label}-{kind}-s{steps}.png"
                image.save(output / filename)
                grid.paste(image, (column * 256, (group * 2 + prompt_index) * 256))
                row = {"snapshot": label, "checkpoint_step": metadata["step"],
                    "kind": kind, "prompt": prompt, "steps": steps, "seed": 42,
                    "image": filename, "generate_ms": round((time.perf_counter() - start) * 1000, 2),
                    "neural_ms": round(generator.last_generation_seconds * 1000, 2),
                    "sha256_pixels": hashlib.sha256(image.tobytes()).hexdigest()}
                report["images"].append(row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
        del generator
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    grid.save(output / "comparison.png")
    report["grid"] = {"image": "comparison.png", "columns": [8, 16, 32],
        "rows": ["app/chat", "app/settings", "latest/chat", "latest/settings"]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output / "report.json"), "grid": str(output / "comparison.png")}, ensure_ascii=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New report directory; must not exist")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    compare(args.output, args.device)


if __name__ == "__main__":
    main()
