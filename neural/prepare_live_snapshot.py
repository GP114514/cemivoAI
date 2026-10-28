"""Copy and verify an immutable inference checkpoint for the image-only app."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone


def main():
    root = Path(__file__).resolve().parent
    target = root / "live-snapshot"
    target.mkdir(exist_ok=True)
    if (target / "snapshot.json").exists() or (target / "inference.pt").exists():
        raise SystemExit("A live snapshot already exists; it is deliberately not overwritten.")
    source = root / "checkpoints" / "inference.pt"
    temporary = target / "inference.pt.copying"
    # Training publishes inference.pt with atomic replacement, so an open source
    # handle reads one complete checkpoint. We never take the training lock.
    with source.open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=4 * 1024**2)
    import torch
    from .model import FORMAT_VERSION
    saved = torch.load(temporary, map_location="cpu", weights_only=False)
    if saved.get("format_version") != FORMAT_VERSION or "model" not in saved:
        raise RuntimeError("Not a cemivo pixel flow checkpoint")
    with temporary.open("rb") as source_copy:
        digest = hashlib.file_digest(source_copy, "sha256").hexdigest()
    metadata = {"step": int(saved["step"]), "native_size": int(saved.get("native_size", 256)),
        "sha256": digest,
        "source": str(source), "created_at": datetime.now(timezone.utc).isoformat(),
        "random_initialization": True, "pretrained_weights": False,
        "frozen": True, "quality_verified": False}
    del saved
    os.replace(temporary, target / "inference.pt")
    (target / "snapshot.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
