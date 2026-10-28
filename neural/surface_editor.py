"""CPU neural surface adapter; every displayed border and glyph is NN output."""
from __future__ import annotations
import hashlib
from io import BytesIO
from pathlib import Path
import pickle
import threading
import time
import numpy as np
from PIL import Image

LOAD_ERRORS = (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError,
    EOFError, pickle.UnpicklingError, IndexError, AttributeError, OverflowError)


class SurfaceEditor:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / 'neural' / 'surface-head-v1'
        self.model = self.text_model = None
        self.metadata = self.text_metadata = {}
        self.signature = None
        self.last_error = None
        self.lock = threading.RLock()

    def _refresh(self):
        try:
            self._load()
        except LOAD_ERRORS as error:
            self.model = self.text_model = None
            self.metadata = self.text_metadata = {}
            self.signature = None
            self.last_error = f'{type(error).__name__}: {error}'
            raise

    def _load(self):
        paths = (self.directory / 'inference.pt', self.root / 'neural' / 'text-head-v1' / 'inference.pt')
        signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        if self.signature == signature and self.model is not None:
            return
        self.model = self.text_model = None
        from .surface_head import load_checkpoint
        from .text_head import load_checkpoint as load_text
        model, metadata = load_checkpoint(paths[0])
        text_model, text_metadata = load_text(paths[1], require_validated=True)
        text_sha256 = hashlib.sha256(paths[1].read_bytes()).hexdigest()
        after = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        if signature != after:
            raise ValueError('surface or text weights changed while loading')
        self.model, self.text_model = model, text_model
        self.metadata, self.text_metadata = metadata, text_metadata
        self.text_sha256 = text_sha256
        self.signature = signature
        self.last_error = None

    def status(self):
        with self.lock:
            try:
                self._refresh()
            except LOAD_ERRORS as error:
                self.last_error = f'{type(error).__name__}: {error}'
            return {'available': self.model is not None and self.text_model is not None,
                'device': 'cpu', 'sha256': self.metadata.get('sha256'),
                'model_sha256': self.metadata.get('sha256'), 'checkpoint_step': self.metadata.get('step'),
                'parameters': self.metadata.get('parameters'), 'random_initialization': True,
                'pretrained_weights': False, 'font_or_bitmap_assets_at_inference': False,
                'last_error': self.last_error, 'scope': 'auxiliary neural desktop and navigation surfaces'}

    def render(self, seed=42, chrome=False):
        started = time.perf_counter()
        with self.lock:
            self._refresh()
            loaded = time.perf_counter()
            pixels, bounds = self.model.predict(seed, chrome)
            labels = {'title': 'CEMIVO', 'chat': '聊天', 'notes': '笔记', 'tasks': '待办',
                'settings': '设置', 'home': '桌面', 'back': '返回'}
            text_bounds = {}
            for role, box in bounds.items():
                label = labels[role]
                width = len(label)*(9 if label.isascii() else 16)
                x0, y0, x1, y1 = box
                left = x0+(x1-x0-width)//2
                top = y0+(y1-y0-18)//2
                if role != 'title' and (left <= x0 or top <= y0):
                    raise ValueError('neural box did not leave safe text padding')
                prediction = self.text_model.predict_text(label, width, 18)
                pixels[top:top+18, left:left+width] = prediction
                text_bounds[role] = [left, top, left+width, top+18]
            sampled = time.perf_counter()
            output = BytesIO()
            Image.fromarray(pixels).save(output, format='PNG', compress_level=2)
            png = output.getvalue()
            status = self.status()
            model = dict(status, name='cemivo-neural-surface', step=self.metadata.get('step'),
                text_model_sha256=self.text_sha256, text_parameters=self.text_metadata.get('parameter_count'),
                provenance={'random_initialization': True, 'pretrained_weights': False,
                    'font_or_bitmap_assets_at_inference': False,
                    'layout': 'learned conditional layout head', 'button_pixels': 'learned coordinate pixel MLP',
                    'text_pixels': 'trained neural glyph decoder', 'auxiliary_to_main_pixel_flow': True})
            return png, {'width': 256, 'height': 256, 'sha256': hashlib.sha256(png).hexdigest(),
                'seed': seed, 'steps': 1, 'prompt': 'Neural desktop' if not chrome else 'Neural navigation',
                'checkpoint_step': self.metadata.get('step'), 'model_sha256': self.metadata.get('sha256'),
                'parameters': self.metadata.get('parameters'),
                'model': model, 'surface': {'mode': 'chrome' if chrome else 'desktop',
                    'bounds': bounds, 'text_bounds': text_bounds, 'click_routing': 'AI screenshot interpretation only'},
                'timings': {'load_ms': round((loaded-started)*1000, 2),
                    'sample_ms': round((sampled-loaded)*1000, 2),
                    'total_ms': round((time.perf_counter()-started)*1000, 2)}}
