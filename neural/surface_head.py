"""Learned layout and coordinate-to-pixel decoder for native monochrome surfaces.

Training target construction is deliberately kept outside this module. Inference
loads learned tensors only; it does not call a font, draw rectangles, or load
stored button images. This auxiliary head complements the main PixelFlow model.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn

FORMAT = 'cemivo-neural-surface-v1'
ROLES = ('title', 'chat', 'notes', 'tasks', 'settings', 'home', 'back')


class SurfaceHead(nn.Module):
    def __init__(self):
        super().__init__()
        # A learned linear layout head over crossed mode/variation features.
        # The products are generic condition features, never box coordinates.
        self.layout = nn.Linear(9, 28)
        self.pixel = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64),
            nn.ReLU(), nn.Linear(64, 1))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def layout_features(condition):
        return torch.cat((condition, condition[..., :1]*condition[..., 1:]), dim=-1)

    @staticmethod
    def conditioning(seed, chrome=False):
        if type(seed) is not int or not 0 <= seed < 2**31 or type(chrome) is not bool:
            raise ValueError('invalid surface seed or mode')
        rng = np.random.default_rng(seed)
        return np.concatenate(([float(chrome)], rng.uniform(-1, 1, 4))).astype(np.float32)

    @torch.inference_mode()
    def predict(self, seed=42, chrome=False):
        device = next(self.parameters()).device
        condition = torch.from_numpy(self.conditioning(seed, chrome)).to(device)
        boxes = self.layout(self.layout_features(condition)).reshape(7, 4).mul(256).round().to(torch.int64).cpu().tolist()
        selected = [0, 5, 6] if chrome else list(range(7))
        for index in selected:
            x0, y0, x1, y1 = boxes[index]
            if not 0 <= x0 < x1 <= 256 or not 0 <= y0 < y1 <= (32 if chrome else 256):
                raise ValueError('neural layout predicted invalid native bounds')
            if x1-x0 < (54 if index == 0 else 36) or y1-y0 < 18:
                raise ValueError('neural layout cannot contain its native text cells')
        yy, xx = torch.meshgrid(torch.arange(256, device=device), torch.arange(256, device=device), indexing='ij')
        # A learned exterior prediction supplies the canvas, including padding.
        blank = 1-self.pixel(torch.full((1, 4), -2., device=device)).sigmoid()
        image = blank.expand(256, 256).clone()
        for index in selected:
            if index == 0:
                continue
            x0, y0, x1, y1 = boxes[index]
            features = torch.stack((xx-x0, x1-1-xx, yy-y0, y1-1-yy), dim=-1).float().clamp(-2, 2)
            layer = (1-self.pixel(features.reshape(-1, 4)).sigmoid()).reshape(256, 256)
            image = torch.minimum(image, layer)
        pixels = image.mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        return pixels, {ROLES[index]: boxes[index] for index in selected}


def load_checkpoint(path, device='cpu'):
    path = Path(path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(saved, dict) or saved.get('format') != FORMAT or
            saved.get('random_initialization') is not True or saved.get('pretrained_weights') is not False or
            saved.get('heldout_validated') is not True or not isinstance(saved.get('validation'), dict)):
        raise ValueError('surface model has no valid provenance and heldout acceptance')
    with torch.random.fork_rng(devices=[]):
        model = SurfaceHead()
    state = saved.get('state_dict')
    if not isinstance(state, dict) or any(not isinstance(t, torch.Tensor) or not bool(torch.isfinite(t).all()) for t in state.values()):
        raise ValueError('invalid surface tensors')
    model.load_state_dict(state, strict=True)
    if saved.get('parameters') != model.parameter_count:
        raise ValueError('surface parameter count mismatch')
    model.to(device).eval()
    metadata = {k: v for k, v in saved.items() if k != 'state_dict'}
    metadata['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return model, metadata
