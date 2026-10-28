"""Lazy CPU adapter for trained neural text pixels; no font/bitmap rendering."""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import pickle
import threading
import time

import numpy as np
from PIL import Image


class TextEditor:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.model = None
        self.metadata = {}
        self.signature = None
        self.sha256 = None
        self.last_error = None
        self.lock = threading.RLock()

    def _refresh(self):
        path = self.directory / 'inference.pt'
        try:
            stat = path.stat()
        except FileNotFoundError:
            return
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self.signature:
            return
        self.signature = signature
        self.model = None
        self.metadata = {}
        self.sha256 = None
        try:
            from .text_head import load_checkpoint
            model, metadata = load_checkpoint(path, device='cpu', require_validated=True)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            after = path.stat()
            if (after.st_mtime_ns, after.st_size) != signature:
                raise ValueError('text checkpoint changed while loading; retry after publication')
            self.model, self.metadata, self.sha256 = model, metadata, digest
            self.last_error = None
        except (ImportError, ValueError, RuntimeError, OSError, KeyError, TypeError,
                EOFError, pickle.UnpicklingError, IndexError, AttributeError, OverflowError) as error:
            self.model, self.metadata, self.sha256 = None, {}, None
            self.last_error = f'{type(error).__name__}: {error}'

    def status(self):
        with self.lock:
            self._refresh()
            return {'available': self.model is not None, 'device': 'cpu',
                'model_sha256': self.sha256, 'random_initialization': True,
                'pretrained_weights': False, 'last_error': self.last_error,
                'step': self.metadata.get('step'), 'parameters': self.metadata.get('parameter_count'),
                'scope': 'trained characters and native text cells only'}

    @staticmethod
    def fit_bounds(text, bounds, outer):
        """Fit known native cell advances horizontally inside the AI's region.

        The accepted outer edit never grows. Use complete native text rows;
        do not move the text into buttons or other rows to make it fit.
        """
        from .text_head import character_advance
        x0,y0,x1,y1 = bounds
        if not (outer[0] <= x0 < x1 <= outer[2] and outer[1] <= y0 < y1 <= outer[3]):
            raise ValueError('文字内区超出 AI 已选择的编辑范围。')
        needed = max(sum(character_advance(c) for c in line) for line in text.split('\n'))
        if needed <= outer[2] - x0:
            x1 = max(x1, x0 + needed)
        rows=1;advance=0
        for character in text:
            if character=='\n':rows+=1;advance=0;continue
            width=character_advance(character)
            if advance+width>x1-x0:rows+=1;advance=0
            advance+=width
        # Do not erase unrelated controls below a short draft just because the
        # AI overestimated the vertical text rectangle.
        y1=y0+18*rows
        if y1 > outer[3]:
            raise ValueError('AI 选择的文字内区不足完整的 18 像素行，原图已保留。')
        return [x0,y0,x1,y1]

    def render(self, original_png, bounds, text):
        started = time.perf_counter()
        with self.lock:
            self._refresh()
            if self.model is None:
                raise ValueError('神经文字模型尚未通过独立字形与组合文字检查。')
            used_model = {'available':True,'device':'cpu','model_sha256':self.sha256,
                'random_initialization':True,'pretrained_weights':False,'last_error':None,
                'step':self.metadata.get('step'),'parameters':self.metadata.get('parameter_count'),
                'scope':'trained characters and native text cells only'}
            if (not isinstance(bounds, (tuple, list)) or len(bounds) != 4 or
                    any(type(value) is not int for value in bounds)):
                raise ValueError('文字区域须为原生整数坐标。')
            with Image.open(BytesIO(original_png)) as image:
                if image.format != 'PNG' or image.mode != 'L':
                    raise ValueError('文字编辑需要原生灰度 PNG。')
                image.load()
                pixels = np.array(image, dtype=np.uint8, copy=True)
            x0, y0, x1, y1 = bounds
            height, width = pixels.shape
            if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
                raise ValueError('文字区域超出画面。')
            background = float(np.median(pixels[y0:y1, x0:x1]))
            if 64 < background < 192:
                raise ValueError('文字区域的黑白背景不明确，保留主模型编辑路径。')
            sample_start = time.perf_counter()
            if text==' ' and y1-y0<18:
                prediction=self.model.predict_text(text,x1-x0,18,dark=background <=64)[:y1-y0].copy()
            else:
                prediction = self.model.predict_text(text, x1-x0, y1-y0, dark=background <= 64)
            if not isinstance(prediction, np.ndarray) or prediction.dtype != np.uint8 or prediction.shape != (y1-y0, x1-x0):
                raise ValueError('神经文字输出尺寸或像素类型不正确。')
            sample_ms = (time.perf_counter() - sample_start) * 1000
            candidate = pixels.copy()
            candidate[y0:y1, x0:x1] = prediction
            output = BytesIO()
            Image.fromarray(candidate).save(output, format='PNG', compress_level=2)
            return output.getvalue(), {'sample_ms': round(sample_ms, 2), 'load_ms': 0,
                'encode_ms': round((time.perf_counter()-started)*1000-sample_ms, 2), 'editor_model':used_model}
