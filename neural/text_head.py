"""Random-weight neural native glyph decoder; no font/bitmap rendering at inference.

Characters select learned embedding vectors, shared nonlinear layers predict
every grayscale cell pixel, and cells are placed losslessly without scaling.
This auxiliary component is not the project's full 109M image generator. Its
character vocabulary is finite and validated strings do not prove arbitrary
language or UI quality. Training fonts and target bitmaps are never loaded here.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

FORMAT_VERSION = 'cemivo-neural-text-head-v1'
CELL_HEIGHT, CELL_WIDTH = 18, 16
ASCII_ADVANCE, CJK_ADVANCE = 9, 16


def character_advance(character):
    return ASCII_ADVANCE if ord(character) < 128 else CJK_ADVANCE


def text_layout(text, width, height, vocabulary):
    if not isinstance(text, str) or not text or len(text) > 500:
        raise ValueError('text must contain 1..500 characters')
    if type(width) is not int or type(height) is not int or not 1 <= width <= 1024 or not CELL_HEIGHT <= height <= 1024:
        raise ValueError('native text bounds need width 1..1024 and height 18..1024')
    positions = []
    x = y = 0
    for character in text:
        if character == '\n':
            x = 0; y += CELL_HEIGHT
            if y + CELL_HEIGHT > height:
                raise ValueError('complete text does not fit the native region')
            continue
        if character not in vocabulary:
            raise ValueError(f'character U+{ord(character):04X} is outside the trained vocabulary')
        advance = character_advance(character)
        if advance > width:
            raise ValueError('native region is narrower than one character')
        if x + advance > width:
            x = 0; y += CELL_HEIGHT
        if y + CELL_HEIGHT > height:
            raise ValueError('complete text does not fit the native region')
        positions.append((character, x, y, advance))
        x += advance
    return positions


class TextHead(nn.Module):
    def __init__(self, vocabulary, embedding_dim=128, hidden_dim=512):
        super().__init__()
        if (not isinstance(vocabulary, str) or not vocabulary or len(vocabulary) != len(set(vocabulary))
                or ' ' not in vocabulary or not 1 <= len(vocabulary) <= 20000):
            raise ValueError('vocabulary needs unique characters including blank space')
        if type(embedding_dim) is not int or not 16 <= embedding_dim <= 512 or type(hidden_dim) is not int or not 32 <= hidden_dim <= 1024:
            raise ValueError('invalid text network dimensions')
        self.vocabulary = vocabulary
        self.lookup = {character: index for index, character in enumerate(vocabulary)}
        self.embedding_dim, self.hidden_dim = embedding_dim, hidden_dim
        self.embedding = nn.Embedding(len(vocabulary), embedding_dim)
        self.decoder = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, CELL_HEIGHT * CELL_WIDTH))

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, character_ids):
        if (not isinstance(character_ids, torch.Tensor) or character_ids.dtype != torch.long
                or character_ids.ndim != 1 or character_ids.numel() < 1):
            raise ValueError('character ids must be a nonempty int64 vector')
        return self.decoder(self.embedding(character_ids)).reshape(-1, CELL_HEIGHT, CELL_WIDTH)

    @torch.inference_mode()
    def predict_text(self, text, width, height, dark=False):
        if type(dark) is not bool:
            raise ValueError('dark must be boolean')
        positions = text_layout(text, width, height, self.lookup)
        # Include the learned blank glyph: even all padding pixels are network
        # predictions. No font pixels or procedural white canvas is substituted.
        characters = [' '] + list(dict.fromkeys(character for character, *_ in positions))
        parameter = next(self.parameters())
        ids = torch.tensor([self.lookup[character] for character in characters], dtype=torch.long, device=parameter.device)
        was_training = self.training
        self.eval()
        try:
            prediction = self(ids).sigmoid()
            if not bool(torch.isfinite(prediction).all()):
                raise ValueError('text head produced nonfinite pixels')
            pixels = prediction.float().mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            by_character = dict(zip(characters, pixels))
            blank = by_character[' ']
            canvas = np.tile(blank, (math.ceil(height / CELL_HEIGHT), math.ceil(width / CELL_WIDTH)))[:height, :width].copy()
            for character, x, y, advance in positions:
                canvas[y:y + CELL_HEIGHT, x:x + advance] = by_character[character][:, :advance]
            return (255 - canvas if dark else canvas).copy()
        finally:
            self.train(was_training)


def save_checkpoint(path, model, step, *, heldout_validated=False, validation=None):
    if not isinstance(model, TextHead) or type(step) is not int or step < 0 or type(heldout_validated) is not bool:
        raise ValueError('invalid neural text checkpoint')
    if heldout_validated and not validation:
        raise ValueError('validation evidence required')
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('nonfinite text model weights')
    payload = {'format_version': FORMAT_VERSION, 'vocabulary': model.vocabulary,
        'embedding_dim': model.embedding_dim, 'hidden_dim': model.hidden_dim,
        'cell_size': [CELL_WIDTH, CELL_HEIGHT], 'advances': [ASCII_ADVANCE, CJK_ADVANCE],
        'parameter_count': model.parameter_count, 'step': step, 'state_dict': state,
        'heldout_validated': heldout_validated, 'validation': validation or {},
        'provenance': {'random_initialization': True, 'pretrained_weights': False,
            'architecture': 'learned character embedding and shared nonlinear pixel decoder',
            'font_or_bitmap_assets_at_inference': False}}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + '.', suffix='.tmp', dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name); torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, device='cpu', *, require_validated=False):
    if type(require_validated) is not bool:
        raise ValueError('require_validated must be boolean')
    saved = torch.load(Path(path), map_location='cpu', weights_only=True)
    if (not isinstance(saved, dict) or saved.get('format_version') != FORMAT_VERSION
            or saved.get('cell_size') != [CELL_WIDTH, CELL_HEIGHT]
            or saved.get('advances') != [ASCII_ADVANCE, CJK_ADVANCE]
            or type(saved.get('step')) is not int or saved['step'] < 0
            or type(saved.get('heldout_validated')) is not bool
            or not isinstance(saved.get('validation'), dict)
            or saved.get('provenance', {}).get('random_initialization') is not True
            or saved.get('provenance', {}).get('pretrained_weights') is not False
            or saved.get('provenance', {}).get('font_or_bitmap_assets_at_inference') is not False):
        raise ValueError('invalid neural text checkpoint schema/provenance')
    if require_validated and not saved['heldout_validated']:
        raise ValueError('neural text head has not passed its declared validation')
    if saved['heldout_validated'] and not saved['validation']:
        raise ValueError('missing neural text validation evidence')
    with torch.random.fork_rng(devices=[]):
        model = TextHead(saved['vocabulary'], saved['embedding_dim'], saved['hidden_dim'])
    if saved.get('parameter_count') != model.parameter_count:
        raise ValueError('text parameter count does not match architecture')
    state = saved.get('state_dict')
    if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('invalid neural text tensors')
    model.load_state_dict(state, strict=True)
    model.to(device=device).eval()
    return model, {key: value for key, value in saved.items() if key != 'state_dict'}
