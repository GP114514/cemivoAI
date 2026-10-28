"""TRAINING/EVALUATION ONLY: font-supervised targets for the random text head.

Inference must not import this module or load these target bitmaps/fonts.
"""
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .text_head import CELL_HEIGHT, CELL_WIDTH, character_advance, text_layout

ASCII_FONT = Path('C:/Windows/Fonts/consola.ttf')
CJK_FONT = Path('C:/Windows/Fonts/simsun.ttc')


def make_vocabulary():
    characters = [chr(value) for value in range(32, 127)]
    for first in range(0xA1, 0xF8):
        for second in range(0xA1, 0xFF):
            try:
                character = bytes([first, second]).decode('gb2312')
            except UnicodeDecodeError:
                continue
            if character not in characters:
                characters.append(character)
    return ''.join(characters)


def glyph_targets(vocabulary):
    latin, cjk = ImageFont.truetype(str(ASCII_FONT), 14), ImageFont.truetype(str(CJK_FONT), 14)
    result = []
    for character in vocabulary:
        image = Image.new('L', (CELL_WIDTH, CELL_HEIGHT), 255)
        draw = ImageDraw.Draw(image)
        draw.text((0 if ord(character) < 128 else 1, 14), character,
            font=latin if ord(character) < 128 else cjk, fill=0, anchor='ls')
        result.append(np.asarray(image, dtype=np.uint8).copy())
    return np.stack(result)


def target_text(text, width, height, vocabulary, targets):
    lookup = {character: index for index, character in enumerate(vocabulary)}
    positions = text_layout(text, width, height, lookup)
    canvas = np.full((height, width), 255, dtype=np.uint8)
    for character, x, y, advance in positions:
        canvas[y:y + CELL_HEIGHT, x:x + advance] = targets[lookup[character], :, :advance]
    return canvas


def target_audit(vocabulary, targets):
    return {'vocabulary_size': len(vocabulary), 'cell_size': [CELL_WIDTH, CELL_HEIGHT],
        'font_size': 14, 'ascii_font': str(ASCII_FONT), 'cjk_font': str(CJK_FONT),
        'ascii_font_sha256': hashlib.sha256(ASCII_FONT.read_bytes()).hexdigest(),
        'cjk_font_sha256': hashlib.sha256(CJK_FONT.read_bytes()).hexdigest(),
        'glyph_targets_sha256': hashlib.sha256(targets.tobytes()).hexdigest(),
        'vocabulary_sha256': hashlib.sha256(vocabulary.encode('utf-8')).hexdigest(),
        'target_dtype': str(targets.dtype), 'native_targets': True, 'resizing_used': False,
        'training_target_count': len(targets), 'string_training_examples': 0,
        'scope': 'Font glyph reconstruction for a finite vocabulary, not language understanding.'}
