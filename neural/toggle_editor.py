"""Lazy CPU inference for a separately validated native-pixel toggle CNN.

Only the trained network produces edited pixels. The caller remains responsible
for measuring the source state, verifying the candidate transition, and applying
its independent exact-exterior compositing checks.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image, UnidentifiedImageError


class ToggleEditorError(ValueError):
    pass


class ToggleEditor:
    def __init__(self, directory_or_checkpoint):
        path = Path(directory_or_checkpoint).resolve()
        self.checkpoint = path if path.suffix == ".pt" else path / "inference.pt"
        self.report = self.checkpoint.parent / "report.json"
        self.lock = threading.RLock()
        self.model = None
        self.metadata = {}
        self.model_sha256 = None
        self.last_error = None

    def _report(self):
        try:
            if self.report.stat().st_size > 2 * 1024**2:
                return {}
            value = json.loads(self.report.read_text(encoding="utf-8-sig"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def status(self):
        """Read small readiness metadata only; never load or run torch here."""
        with self.lock:
            report = self._report()
            metadata = self.metadata
            provenance = metadata.get("provenance", {}) if self.model is not None else report
            validated = metadata.get("heldout_validated") if self.model is not None else report.get("heldout_validated")
            available = (self.checkpoint.is_file() and validated is True and
                provenance.get("random_initialization") is True and provenance.get("pretrained_weights") is False)
            return {"available": available, "loaded": self.model is not None, "device": "cpu",
                "heldout_validated": validated is True, "model_sha256": self.model_sha256,
                "parameters": metadata.get("parameter_count", report.get("parameters")),
                "step": metadata.get("step", report.get("steps")),
                "random_initialization": provenance.get("random_initialization") is True,
                "pretrained_weights": provenance.get("pretrained_weights"),
                "last_error": self.last_error}

    def public(self):
        return self.status()

    def _load(self):
        if self.model is not None:
            return 0.0
        started = time.perf_counter()
        try:
            from .toggle_head import load_checkpoint
            # Hash and loader refer to one immutable file publication. Model
            # directories must be frozen by the caller before use.
            digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
            model, metadata = load_checkpoint(self.checkpoint, device="cpu", require_validated=True)
        except Exception:
            self.last_error = "开关编辑模型未就绪或未通过独立验证。"
            raise ToggleEditorError(self.last_error) from None
        self.model, self.metadata, self.model_sha256 = model, metadata, digest
        self.last_error = None
        return round((time.perf_counter() - started) * 1000, 2)

    @staticmethod
    def _image(original_png):
        if not isinstance(original_png, bytes) or not original_png or len(original_png) > 8 * 1024**2:
            raise ToggleEditorError("原图必须是有效的原生灰度 PNG。")
        try:
            with Image.open(BytesIO(original_png)) as image:
                if image.format != "PNG" or image.mode != "L" or image.size not in ((256, 256), (512, 512)):
                    raise ValueError("invalid native image")
                image.load()
                return np.asarray(image, dtype=np.uint8).copy()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise ToggleEditorError("原图必须是 256 或 512 像素的原生灰度 PNG。") from None

    def render(self, original_png, bounds, target_state):
        """Return a native candidate PNG and timings; no drawing fallback."""
        started = time.perf_counter()
        original = self._image(original_png)
        height, width = original.shape
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 4 or
                any(type(value) is not int for value in bounds)):
            raise ToggleEditorError("开关范围必须是四个原生整数坐标。")
        x0, y0, x1, y1 = bounds
        if (not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height) or
                not 12 <= x1-x0 <= 64 or not 6 <= y1-y0 <= 36):
            raise ToggleEditorError("开关范围超出原图或编辑模型的原生尺寸限制。")
        if target_state not in ("on", "off"):
            raise ToggleEditorError("开关目标状态必须为 on 或 off。")
        with self.lock:
            load_ms = self._load()
            sampled = time.perf_counter()
            try:
                crop = original[y0:y1, x0:x1].copy()
                predicted = self.model.predict_native(crop, target_state)
                if (not isinstance(predicted, np.ndarray) or predicted.dtype != np.uint8 or predicted.shape != crop.shape):
                    raise ValueError("invalid native neural prediction")
            except Exception:
                self.last_error = "神经开关编辑未能生成有效的原生像素，原图已保留。"
                raise ToggleEditorError(self.last_error) from None
            sample_ms = round((time.perf_counter() - sampled) * 1000, 2)
            candidate = original.copy()
            candidate[y0:y1, x0:x1] = predicted
            encoded = time.perf_counter()
            output = BytesIO()
            Image.fromarray(candidate).save(output, format="PNG", compress_level=2, optimize=False)
            encode_ms = round((time.perf_counter() - encoded) * 1000, 2)
            self.last_error = None
            return output.getvalue(), {"sample_ms": sample_ms, "load_ms": load_ms,
                "encode_ms": encode_ms, "total_ms": round((time.perf_counter() - started) * 1000, 2),
                "model_sha256": self.model_sha256, "model_parameters": self.metadata["parameter_count"],
                "model_step": self.metadata["step"], "device": "cpu"}
