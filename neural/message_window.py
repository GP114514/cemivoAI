"""Bounded text scrolling inside an already AI-confirmed message container.

This module chooses no screen controls and draws no pixels. It compiles only
the visible tail of previously committed neural text. The caller signs the
result against the exact current screenshot before the neural renderer uses it.
"""
from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image
from .text_head import character_advance


def contains(outer, inner):
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def compile_window(png, state, area, text):
    bounds = state.get('sent_message_bounds', [])
    visible = state.get('visible_sent_messages', state.get('sent_messages', []))
    if (not bounds or len(bounds) != len(visible)
            or any(not isinstance(value, str) or not value for value in visible)
            or any(not contains(area, box) for box in bounds)):
        raise ValueError('the visible message history is not aligned')
    left, top = bounds[0][:2]
    width = area[2] - left
    if width < 16:
        raise ValueError('message window is too narrow')

    def pack(messages):
        y = top
        items = []
        for message in messages:
            rows, advance = 1, 0
            for character in message:
                step = character_advance(character)
                if character == '\n' or step > width:
                    raise ValueError('unsupported message for cached scrolling')
                if advance + step > width:
                    rows += 1
                    advance = 0
                advance += step
            used_width = min(width, sum(character_advance(c) for c in message))
            box = [left, y, left + used_width, y + rows * 18]
            if not contains(area, box):
                return None
            items.append({'text': message, 'bounds': box})
            y = box[3] + 2
        return items

    messages = visible + [text]
    dropped = 0
    items = pack(messages)
    while items is None and len(messages) > 1:
        messages = messages[1:]
        dropped += 1
        items = pack(messages)
    if items is None or not dropped:
        raise ValueError('message does not fit, or no scrolling is needed')

    with Image.open(BytesIO(png)) as image:
        if image.format != 'PNG' or image.mode != 'L':
            raise ValueError('scrolling requires a native grayscale PNG')
        pixels = np.asarray(image)
        x0, y0, x1, y1 = area
        ink = pixels[y0:y1, x0:x1] < 248
        known = np.zeros(ink.shape, dtype=bool)
        for a, b, c, d in bounds:
            known[b-y0:d-y0, a-x0:c-x0] = True
        if np.any(ink & ~known):
            raise ValueError('message container has untracked pixels')
        ys, xs = np.nonzero(ink)
        if not len(xs):
            raise ValueError('the committed messages are not visible')
        # Clear exactly the old ink plus new text cells, rather than expanding
        # the effect to the entire panel or its border.
        clear = [min(x0 + int(xs.min()), min(i['bounds'][0] for i in items)),
                 min(y0 + int(ys.min()), min(i['bounds'][1] for i in items)),
                 max(x0 + int(xs.max()) + 1, max(i['bounds'][2] for i in items)),
                 max(y0 + int(ys.max()) + 1, max(i['bounds'][3] for i in items))]
    if not contains(area, clear):
        raise ValueError('scrolling escaped the confirmed message container')
    return {'items': deepcopy(items), 'clear_bounds': clear, 'hidden_count': dropped}
