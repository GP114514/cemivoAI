"""Synthetic *pixel training data* for the randomly initialized neural generator.

This module is used during dataset construction/training only. Neural inference
must not call its drawing functions. No external images or model weights are used.

Disk contract: manifest.json, vocab.json, images-00000.npy (uint8 N,H,W), and
metadata-00000.jsonl. Each shard is committed atomically when completely written.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import hashlib
import math
import os
from pathlib import Path
import random
import re
import string
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont

KINDS = ('chat', 'login', 'dashboard', 'settings', 'files', 'shop', 'music',
         'calendar', 'editor', 'mail', 'browser', 'terminal', 'diagram', 'shapes', 'text')
NAMES_ZH = ('聊天窗口', '登录页面', '数据面板', '设置页面', '文件管理器', '购物页面',
            '音乐播放器', '日历页面', '文档编辑器', '电子邮箱', '网页浏览器', '终端窗口',
            '流程图', '基础几何图形', '文字海报')
NAMES_EN = ('chat window', 'login page', 'dashboard', 'settings panel', 'file manager',
            'shopping page', 'music player', 'calendar', 'document editor', 'email inbox',
            'web browser', 'terminal window', 'flowchart', 'basic geometric shapes', 'text poster')
TITLES = ('CEMIVO', 'STUDIO', 'IDEA', 'MONDAY', 'LINE', 'SPACE', 'NOTES', 'OFFICE',
          '灵感', '创作', '工作台', '新项目', '小屋', '聊天', '笔记', '云端')
CN = {
    'send': ('发送', '提交', '发送消息'), 'new': ('新建', '新对话', '添加'),
    'save': ('保存', '确认', '应用'), 'cancel': ('取消', '返回', '关闭'),
    'login': ('登录', '继续', '进入'), 'signup': ('注册', '创建账户'),
    'attach': ('附件', '上传', '添加文件'), 'search': ('搜索', '查找'),
    'delete': ('删除', '移除'), 'play': ('播放', '开始', '播放全部'),
    'buy': ('购买', '加入购物袋', '查看详情'), 'share': ('分享', '发布', '导出'),
    'next': ('下一步', '继续', '完成'),
}
EN = {
    'send': ('Send', 'Submit', 'Send message'), 'new': ('New', 'New chat', 'Add'),
    'save': ('Save', 'Confirm', 'Apply'), 'cancel': ('Cancel', 'Back', 'Close'),
    'login': ('Log in', 'Continue', 'Enter'), 'signup': ('Sign up', 'Create account'),
    'attach': ('Attach', 'Upload', 'Add file'), 'search': ('Search', 'Find'),
    'delete': ('Delete', 'Remove'), 'play': ('Play', 'Start', 'Play all'),
    'buy': ('Buy', 'Add to bag', 'View'), 'share': ('Share', 'Publish', 'Export'),
    'next': ('Next', 'Continue', 'Done'),
}


def _dump(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def build_vocab():
    return {'version': 1, 'type': 'utf8-bytes', 'pad_id': 0, 'bos_id': 1,
            'eos_id': 2, 'byte_offset': 3, 'vocab_size': 259,
            'max_length': 192, 'description': 'UTF-8 byte value + 3. No unknown token.'}


def encode_prompt(text, max_length=192):
    if max_length < 2:
        raise ValueError('max_length must be at least 2')
    values = [1] + [value + 3 for value in str(text).encode('utf-8')[:max_length - 2]] + [2]
    result = np.zeros(max_length, dtype=np.int64)
    result[:len(values)] = values
    return result


@lru_cache(maxsize=48)
def _font(size, bold=False):
    paths = [r'C:\Windows\Fonts\msyhbd.ttc' if bold else r'C:\Windows\Fonts\msyh.ttc',
             '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
             '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for path in paths:
        try:
            return ImageFont.truetype(path, max(1, int(size)))
        except OSError:
            continue
    return ImageFont.load_default()


@lru_cache(maxsize=96)
def _native_font_variant(font, size):
    return font.font_variant(size=size)


class NativeDraw:
    """Draw logical 256-space geometry/fonts directly into the native raster.

    The proxy also covers direct ``c.d`` drawing calls used by charts, polygons
    and icons. It never creates a 256px intermediate image or resizes pixels.
    Text bounding boxes return logical coordinates for existing layout code.
    """
    def __init__(self, image, scale):
        self.draw = ImageDraw.Draw(image)
        self.scale = scale

    def _xy(self, points):
        if points and isinstance(points[0], (tuple, list)):
            return [tuple(round(value * self.scale) for value in point) for point in points]
        return tuple(round(value * self.scale) for value in points)

    def _width(self, width):
        return max(1, round(width * self.scale))

    def _font(self, font):
        # FreeType font_variant retains the TTC face index, font path, layout
        # engine and encoding while rasterizing glyphs at the actual native size.
        if not hasattr(font, 'font_variant') or not hasattr(font, 'size'):
            raise RuntimeError('Native-size dataset rendering requires a scalable FreeType font.')
        size = max(1, round(font.size * self.scale))
        return _native_font_variant(font, size)

    def rounded_rectangle(self, xy, radius=0, fill=None, outline=None, width=1):
        return self.draw.rounded_rectangle(self._xy(xy), radius=round(radius * self.scale),
            fill=fill, outline=outline, width=self._width(width))

    def line(self, xy, fill=None, width=1):
        return self.draw.line(self._xy(xy), fill=fill, width=self._width(width))

    def ellipse(self, xy, fill=None, outline=None, width=1):
        return self.draw.ellipse(self._xy(xy), fill=fill, outline=outline, width=self._width(width))

    def arc(self, xy, start, end, fill=None, width=1):
        return self.draw.arc(self._xy(xy), start, end, fill=fill, width=self._width(width))

    def polygon(self, xy, fill=None, outline=None, width=1):
        return self.draw.polygon(self._xy(xy), fill=fill, outline=outline, width=self._width(width))

    def textbbox(self, xy, text, font, anchor=None):
        bbox = self.draw.textbbox(self._xy(xy), text, font=self._font(font), anchor=anchor)
        return tuple(value / self.scale for value in bbox)

    def text(self, xy, text, font, fill=None, anchor=None):
        return self.draw.text(self._xy(xy), text, font=self._font(font), fill=fill, anchor=anchor)


class PixelCanvas:
    """256 logical layout units, rasterized at the requested native pixel size."""
    def __init__(self, rng, dark=False, size=256):
        self.rng = rng
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError('Native raster size must be a positive integer.')
        self.native_size = size
        self.scale = size / 256
        self.bg, self.fg = (0, 255) if dark else (255, 0)
        self.image = Image.new('L', (size, size), self.bg)
        # Retain the exact historical Pillow call path at 256px, so existing V2
        # pixels, captions and audits remain byte-for-byte reproducible.
        self.d = ImageDraw.Draw(self.image) if size == 256 else NativeDraw(self.image, self.scale)
        self.radius = rng.choice((0, 2, 4, 6))
        self.linewidth = rng.choice((1, 1, 1, 2))
        self.padding = rng.choice((6, 8, 10, 12))
        self.header = rng.choice((21, 24, 27))
        self.sidebar = rng.choice((48, 54, 60, 64))
        self.drawn_text = []
        self.button_labels = {}
        self.visible_title = ''

    def rect(self, x, y, w, h, fill=None, radius=None, width=None):
        self.d.rounded_rectangle((round(x), round(y), round(x + max(0, w)), round(y + max(0, h))),
            radius=self.radius if radius is None else max(0, radius),
            fill=self.bg if fill is None else fill, outline=self.fg,
            width=self.linewidth if width is None else width)

    def line(self, x, y, x2, y2, width=None):
        self.d.line((round(x), round(y), round(x2), round(y2)), fill=self.fg,
                    width=self.linewidth if width is None else width)

    def circle(self, x, y, radius, fill=None, width=None):
        self.d.ellipse((x - radius, y - radius, x + radius, y + radius),
            fill=self.bg if fill is None else fill, outline=self.fg,
            width=self.linewidth if width is None else width)

    def text(self, x, y, text, size=9, bold=False, center=False, fill=None, maxw=None):
        # Eight logical pixels is the minimum glyph size: at native 512px this
        # produces a real 16px font, rather than enlarging an 8px glyph bitmap.
        size = max(8, int(size))
        font = _font(size, bold)
        requested = str(text)
        text = requested
        if maxw is not None:
            # Prefer a complete label at a smaller legible size. Only when
            # eight pixels still cannot fit do we report a shortened label.
            while size > 8 and font.getlength(text) > maxw:
                size -= 1
                font = _font(size, bold)
            while text and font.getlength(text) > maxw:
                text = text[:-1]
        position = (round(x), round(y))
        anchor = 'mt' if center else 'lt'
        bbox = self.d.textbbox(position, text, font=font, anchor=anchor)
        self.d.text(position, text, font=font,
                    fill=self.fg if fill is None else fill, anchor=anchor)
        if self.native_size != 256:
            bbox = tuple(round(value * self.scale) for value in bbox)
        self.drawn_text.append({'requested': requested, 'visible': text,
            'font_px': size if self.native_size == 256 else max(1, round(size * self.scale)),
            'bbox': list(bbox)})
        return text

    def button(self, x, y, w, h, text, primary=False):
        self.rect(x, y, w, h, fill=self.fg if primary else self.bg)
        visible = self.text(x + w / 2, y + max(1, (h - 11) / 2), text,
                  8 if len(text) > 7 else 9, True, center=True,
                  fill=self.bg if primary else self.fg, maxw=w - 5)
        self.button_labels.setdefault(str(text), []).append(self.drawn_text[-1])
        return visible

    def label_width(self, text, minimum=40, padding=10):
        size = 8 if len(str(text)) > 7 else 9
        return max(minimum, math.ceil(_font(size, True).getlength(str(text))) + padding)

    def caption_label(self, requested, button=False):
        """Return exactly a visible drawn label and its rendering audit record."""
        requested = str(requested)
        candidates = self.button_labels.get(requested, []) if button else []
        if not candidates:
            candidates = [row for row in self.drawn_text if row['requested'] == requested]
        if not candidates:
            raise ValueError(f'Caption promised an undrawn label: {requested!r}')
        exact = [row for row in candidates if row['visible'] == requested]
        record = exact[0] if exact else max(candidates, key=lambda row: len(row['visible']))
        if not record['visible']:
            raise ValueError(f'Caption label has no visible characters: {requested!r}')
        return record['visible'], dict(record)

    def faux_text(self, x, y, w, lines=2):
        # Variable body-copy strokes are training data, never an inference path.
        for i in range(lines):
            self.line(x, y + i * 5, x + w * self.rng.uniform(.55, 1), y + i * 5, 1)

    def icon(self, x, y, name='plus', size=9):
        if name == 'plus':
            self.line(x + size / 2, y, x + size / 2, y + size)
            self.line(x, y + size / 2, x + size, y + size / 2)
        elif name == 'search':
            self.circle(x + size * .35, y + size * .35, size * .32)
            self.line(x + size * .6, y + size * .6, x + size, y + size)
        elif name == 'user':
            self.circle(x + size / 2, y + size * .23, size * .22)
            self.d.arc((x, y + size * .4, x + size, y + size * 1.35), 180, 360, self.fg, self.linewidth)
        elif name == 'folder':
            self.d.polygon([(x, y + 2), (x + size * .4, y + 2), (x + size * .5, y + 4), (x + size, y + 4), (x + size, y + size), (x, y + size)], fill=self.bg, outline=self.fg)
        elif name == 'image':
            self.rect(x, y, size, size, radius=0)
            self.line(x + 1, y + size - 1, x + size * .4, y + size * .4)
            self.line(x + size * .4, y + size * .4, x + size - 1, y + size - 1)
        elif name == 'play':
            self.d.polygon([(x + 1, y), (x + size, y + size / 2), (x + 1, y + size)], fill=self.fg)
        else:
            for k in range(3):
                self.circle(x + k * 4, y + size / 2, .8, self.fg)

    def frame(self, title, side=False):
        p, hh = self.padding, self.header
        self.rect(p, p, 256 - p * 2, 256 - p * 2, radius=max(2, self.radius))
        self.line(p, p + hh, 256 - p, p + hh)
        for k in range(3):
            self.circle(p + 8 + k * 6, p + hh / 2, 1.3, self.fg if k == 0 else self.bg, 1)
        self.visible_title = self.text(p + 30, p + (hh - 12) / 2, title, 10, True, maxw=145)
        self.icon(256 - p - 17, p + (hh - 8) / 2, 'search', 7)
        x, y = p + 10, p + hh + 10
        right, bottom = 246 - p, 246 - p
        if side:
            self.line(p + self.sidebar, p + hh, p + self.sidebar, 256 - p)
            x = p + self.sidebar + 9
        return x, y, right - x, bottom - y


def _draw_sample(seed, size=256, index=0):
    """Return real raster pixels and a caption derived from the sampled controls."""
    rng = random.Random(int(seed))
    kind_index = index % len(KINDS)
    kind = KINDS[kind_index]
    zh = rng.random() < .65
    dark = rng.random() < .12
    c = PixelCanvas(rng, dark, size=size)
    dictionary = CN if zh else EN
    label = lambda name: rng.choice(dictionary[name])
    title = rng.choice(TITLES) + (f' {rng.randint(1, 99):02}' if rng.random() < .5 else '')
    controls = []
    details = []
    has_sidebar = rng.random() < .65 and kind in ('chat', 'dashboard', 'settings', 'files', 'mail', 'music')
    if kind in ('diagram', 'shapes', 'text', 'login'):
        has_sidebar = False
    x, y, w, h = c.frame(title, has_sidebar)
    if has_sidebar:
        for j in range(rng.randint(3, 6)):
            yy = y + j * rng.choice((23, 25, 27))
            if yy > 211:
                break
            c.rect(c.padding + 6, yy, c.sidebar - 12, 18,
                   fill=c.fg if j == 0 else c.bg, width=1)
            name = rng.choice(('首页', '消息', '项目', '收藏', '设置') if zh else ('Home', 'Chat', 'Files', 'Saved', 'Setup'))
            c.text(c.padding + 10, yy + 4, name, 7, fill=c.bg if j == 0 else c.fg, maxw=c.sidebar - 19)
        details.append('左侧导航栏' if zh else 'left sidebar')

    if kind == 'chat':
        names = ('小明', '设计助手', '项目讨论', '产品团队') if zh else ('Alex', 'Design Bot', 'Project', 'Team')
        c.circle(x + 8, y + 7, 7)
        c.icon(x + 4, y + 2, 'user', 8)
        c.text(x + 22, y + 1, rng.choice(names), 9, True, maxw=w - 45)
        c.icon(x + w - 12, y + 2, 'more', 8)
        c.line(x, y + 22, x + w, y + 22)
        count = rng.randint(2, 5)
        usable = h - 88
        row = min(31, usable / count)
        messages = ('你好', '现在开始吧', '有一个新想法', '我收到了', '谢谢', '明天见', '一起讨论设计') if zh else ('Hello!', 'Let us start.', 'A new idea', 'Got it.', 'Thank you', 'See you', 'Design together')
        for j in range(count):
            bw = rng.randint(max(35, int(w * .46)), max(40, int(w * .86)))
            bx = x + (w - bw if j % 2 else 0)
            yy = y + 32 + j * row
            c.rect(bx, yy, bw, max(19, row - 6), fill=c.fg if j % 2 else c.bg)
            c.text(bx + 5, yy + 4, rng.choice(messages), 8,
                   fill=c.bg if j % 2 else c.fg, maxw=bw - 9)
        iy = y + h - 48
        c.rect(x, iy, w, 46)
        c.text(x + 5, iy + 5, '输入消息…' if zh else 'Message...', 8, maxw=w - 10)
        controls = [label('attach'), label('send')]
        c.icon(x + 6, iy + 29, 'plus', 8)
        send_width = min(c.label_width(controls[1]), w * .52)
        attach_width = min(c.label_width(controls[0], 45), max(20, w - send_width - 32))
        c.button(x + 21, iy + 25, attach_width, 17, controls[0])
        c.button(x + w - send_width - 5, iy + 25, send_width, 17, controls[1], True)
        details.append(f'{count}条聊天气泡' if zh else f'{count} message bubbles')

    elif kind == 'login':
        wide = rng.choice((132, 148, 160, 176))
        cx = (256 - wide) / 2
        top = y + rng.randint(0, 7)
        c.rect(117, top, 22, 22, fill=c.fg, radius=rng.randint(2, 6))
        c.text(128, top + 4, rng.choice(('C', 'M', 'S')), 12, True, True, c.bg)
        c.text(128, top + 31, '欢迎回来' if zh else 'Welcome back', 13, True, True)
        fields = rng.choice((2, 2, 3))
        if fields == 3:
            top -= 5
        for j in range(fields):
            yy = top + 56 + j * 38
            fields_zh = ('邮箱', '密码', '验证码')
            fields_en = ('Email', 'Password', 'Code')
            c.text(cx, yy, (fields_zh if zh else fields_en)[j], 8)
            c.rect(cx, yy + 12, wide, 21)
            value = ('name@mail.com', '••••••••', '123456')[j]
            c.text(cx + 5, yy + 17, value, 8)
        button_y = min(219, top + 61 + fields * 38)
        controls = [label('login')]
        c.button(cx, button_y, wide, 22, controls[0], True)
        if button_y + 35 < 245:
            controls.append(label('signup'))
            c.text(128, button_y + 29, controls[1], 8, center=True)
        details.append(f'{fields}个输入框' if zh else f'{fields} input fields')

    elif kind == 'dashboard':
        c.text(x, y, '数据概览' if zh else 'Overview', 13, True)
        columns = rng.choice((2, 3))
        cw = (w - 5 * (columns - 1)) / columns
        for j in range(columns):
            xx = x + j * (cw + 5)
            c.rect(xx, y + 23, cw, 44)
            c.text(xx + 5, y + 29, rng.choice(('访问', '项目', '用户') if zh else ('Views', 'Tasks', 'Users')), 7)
            c.text(xx + 5, y + 43, str(rng.randint(100, 9999)), 13, True, maxw=cw - 7)
        chart_y, chart_h = y + 78, rng.randint(72, 95)
        c.rect(x, chart_y, w, chart_h)
        c.text(x + 6, chart_y + 6, '增长趋势' if zh else 'Activity', 8, True)
        for j in range(3):
            c.line(x + 10, chart_y + 28 + j * 23, x + w - 9, chart_y + 28 + j * 23, 1)
        chart_type = rng.choice(('line', 'bar'))
        if chart_type == 'line':
            points = [(x + 11 + j * (w - 22) / 7, chart_y + chart_h - 10 - rng.randint(6, max(7, chart_h - 35))) for j in range(8)]
            c.d.line(points, fill=c.fg, width=rng.choice((1, 2)))
            for px, py in points:
                c.circle(px, py, 1.5, c.fg)
        else:
            for j in range(6):
                bh = rng.randint(10, max(11, chart_h - 27))
                c.rect(x + 12 + j * (w - 22) / 6, chart_y + chart_h - bh - 7, (w - 28) / 9, bh, fill=c.fg, radius=0)
        details.append(('折线图' if chart_type == 'line' else '柱状图') if zh else f'{chart_type} chart')
        controls = [label('share')]
        c.button(x + w - 50, y + h - 23, 50, 21, controls[0])

    elif kind == 'settings':
        c.text(x, y, '偏好设置' if zh else 'Preferences', 12, True)
        rows = rng.randint(3, 5)
        for j in range(rows):
            yy = y + 29 + j * min(31, (h - 58) / rows)
            c.text(x, yy, rng.choice(('通知', '自动保存', '隐私', '深色模式', '声音') if zh else ('Notify', 'Autosave', 'Privacy', 'Dark mode', 'Sound')), 9, True, maxw=w - 42)
            c.faux_text(x, yy + 14, w - 45, 1)
            enabled = rng.random() < .5
            c.rect(x + w - 28, yy + 1, 26, 12, fill=c.fg if enabled else c.bg, radius=6, width=1)
            c.circle(x + w - (8 if enabled else 21), yy + 7, 4, c.bg if enabled else c.fg, 1)
        controls = [label('cancel'), label('save')]
        c.button(x + max(0, w - 101), y + h - 22, 47, 20, controls[0])
        c.button(x + w - 49, y + h - 22, 49, 20, controls[1], True)
        details.append(f'{rows}个开关' if zh else f'{rows} toggle switches')

    elif kind == 'files':
        c.text(x, y, '我的文件' if zh else 'My files', 12, True)
        controls = [label('attach')]
        c.button(x + w - 47, y - 1, 47, 19, controls[0], True)
        c.rect(x, y + 28, w, 21)
        c.icon(x + 6, y + 34, 'search', 9)
        c.text(x + 21, y + 34, '搜索文件' if zh else 'Search files', 8)
        cols = rng.choice((2, 3))
        cw = (w - (cols - 1) * 5) / cols
        for j in range(cols):
            xx = x + j * (cw + 5)
            c.rect(xx, y + 60, cw, 50)
            c.icon(xx + 7, y + 67, 'folder', 17)
            c.text(xx + 6, y + 93, rng.choice(('设计', '文档', '图片') if zh else ('Design', 'Docs', 'Photos')), 8, maxw=cw - 8)
        for j in range(rng.randint(3, 5)):
            yy = y + 126 + j * 19
            if yy + 14 > y + h:
                break
            c.rect(x + 2, yy, 9, 12, radius=0)
            c.text(x + 18, yy + 1, rng.choice(('计划.pdf', '图片.png', '笔记.txt', '设计.svg') if zh else ('plan.pdf', 'photo.png', 'notes.txt', 'design.svg')), 8, maxw=w - 42)
            c.text(x + w - 20, yy + 1, f'{rng.randint(1, 99)}K', 7)

    elif kind == 'shop':
        c.text(x, y, '精选好物' if zh else 'Daily essentials', 14, True)
        cols, rows = rng.choice(((2, 2), (3, 2), (2, 1)))
        cw = (w - 7 * (cols - 1)) / cols
        ch = min(91, (h - 32 - (rows - 1) * 7) / rows)
        controls = [label('buy')]
        for j in range(cols * rows):
            xx, yy = x + j % cols * (cw + 7), y + 29 + j // cols * (ch + 7)
            c.rect(xx, yy, cw, ch)
            pw = rng.randint(17, max(18, int(cw * .5)))
            ph = max(15, ch - 47)
            c.rect(xx + (cw - pw) / 2, yy + 6, pw, ph, radius=rng.choice((0, 3, 8)))
            c.text(xx + 5, yy + ch - 36, f'¥{rng.randint(10, 499)}' if zh else f'${rng.randint(10, 99)}', 9, True)
            c.button(xx + 4, yy + ch - 21, cw - 8, 17, controls[0], j == 0)
        details.append(f'{cols * rows}个商品卡片' if zh else f'{cols * rows} product cards')

    elif kind == 'music':
        cover = rng.choice((57, 64, 71))
        c.rect(x, y, cover, cover)
        c.circle(x + cover / 2, y + cover / 2, cover * .32)
        c.circle(x + cover / 2, y + cover / 2, cover * .07, c.fg)
        c.text(x + cover + 10, y + 8, '专注时刻' if zh else 'Focus time', 10, True, maxw=w - cover - 10)
        c.text(x + cover + 10, y + 28, '精选歌单' if zh else 'My playlist', 8, maxw=w - cover - 10)
        controls = [label('play')]
        c.button(x + cover + 10, y + 45, max(40, w - cover - 13), 20, controls[0], True)
        count = rng.randint(3, 6)
        for j in range(count):
            yy = y + cover + 18 + j * 21
            if yy > y + h - 39:
                break
            c.text(x, yy, f'{j + 1:02}', 8)
            c.text(x + 19, yy, rng.choice(('晨光', '漫步', '远山', '白色房间') if zh else ('Morning', 'Walking', 'Mountains', 'White room')), 8, maxw=w - 51)
            c.text(x + w - 25, yy, f'3:{rng.randint(10, 59)}', 7)
            c.line(x, yy + 14, x + w, yy + 14, 1)
        c.line(x, y + h - 15, x + w, y + h - 15, 1)
        c.line(x, y + h - 15, x + w * rng.uniform(.15, .8), y + h - 15, 3)
        c.icon(x + w / 2 - 4, y + h - 35, 'play', 9)

    elif kind == 'calendar':
        month = rng.randint(1, 12)
        c.text(x, y, f'2026 / {month:02}', 13, True)
        controls = [label('new')]
        c.button(x + w - 43, y - 1, 43, 20, controls[0], True)
        gy, gh = y + 31, h - 33
        c.rect(x, gy, w, gh, radius=0)
        cellw, cellh = w / 7, (gh - 19) / 5
        for j, d in enumerate('一二三四五六日' if zh else 'MTWTFSS'):
            c.text(x + (j + .5) * cellw, gy + 4, d, 7, center=True)
            if j:
                c.line(x + j * cellw, gy, x + j * cellw, gy + gh, 1)
        for j in range(5):
            c.line(x, gy + 19 + j * cellh, x + w, gy + 19 + j * cellh, 1)
        start = rng.randint(0, 3)
        active = rng.randint(1, 28)
        for day in range(1, 31):
            idx = day + start - 1
            xx = x + idx % 7 * cellw
            yy = gy + 19 + idx // 7 * cellh
            if idx >= 35:
                continue
            if day == active:
                c.circle(xx + cellw / 2, yy + 9, 7, c.fg)
            c.text(xx + cellw / 2, yy + 4, day, 7, center=True, fill=c.bg if day == active else c.fg)
            if rng.random() < .13 and cellh > 24:
                c.line(xx + 4, yy + 22, xx + cellw - 4, yy + 22, 2)
        details.append(f'{month}月日历' if zh else f'month {month}')

    elif kind == 'editor':
        for j, labeltext in enumerate(('文件', '编辑', '插入', '视图') if zh else ('File', 'Edit', 'Insert', 'View')):
            c.text(x + j * 34, y, labeltext, 8)
        c.line(x, y + 16, x + w, y + 16)
        c.text(x + 4, y + 23, 'B    I    U    16    |    A    +', 9, True)
        c.line(x, y + 40, x + w, y + 40)
        px, py = x + rng.randint(5, 16), y + 51
        pw = w - (px - x) * 2
        c.rect(px, py, pw, h - 52, radius=0)
        heading = rng.choice(('项目计划', '设计说明', '我的笔记', '灵感记录') if zh else ('Project plan', 'Design brief', 'My notes', 'New ideas'))
        c.text(px + 12, py + 14, heading, 12, True, maxw=pw - 22)
        for j in range(rng.randint(3, 7)):
            yy = py + 42 + j * 12
            if yy > y + h - 16:
                break
            c.faux_text(px + 12, yy, pw - 25, rng.choice((1, 2)))
        controls = [label('share')]
        c.button(x + w - 43, y - 3, 43, 19, controls[0], True)

    elif kind == 'mail':
        c.text(x, y, '收件箱' if zh else 'Inbox', 13, True)
        controls = [label('new')]
        compose_width = min(c.label_width(controls[0], 41), w * .5)
        c.button(x + w - compose_width, y - 2, compose_width, 19, controls[0], True)
        for j in range(rng.randint(3, 5)):
            yy = y + 29 + j * 35
            if yy + 29 > y + h:
                break
            c.rect(x, yy + 3, 7, 7, radius=0, width=1)
            c.text(x + 14, yy, rng.choice(('产品团队', '设计小组', '日程助手', '每周灵感') if zh else ('Product Team', 'Design', 'Calendar', 'Inspiration')), 8, True, maxw=w - 36)
            c.text(x + w - 22, yy, f'{rng.randint(8, 19):02}:{rng.randint(0, 59):02}', 6)
            c.text(x + 14, yy + 12, rng.choice(('欢迎加入', '项目更新', '新的邀请', '会议提醒') if zh else ('Welcome aboard', 'Project update', 'New invitation', 'Meeting reminder')), 8, maxw=w - 20)
            c.line(x, yy + 29, x + w, yy + 29, 1)

    elif kind == 'browser':
        c.text(x, y, '<  >  o', 10)
        c.rect(x + 40, y - 2, w - 58, 18, radius=8)
        c.text(x + 47, y + 2, rng.choice(('cemivo.local', 'studio.local', 'idea.local', 'example.com')), 8)
        c.icon(x + w - 10, y + 2, 'plus', 8)
        c.line(x, y + 24, x + w, y + 24)
        c.text(x + 5, y + 39, title, 10, True, maxw=w - 20)
        heading = rng.choice(('让创作更简单', '探索新的可能', '从灵感开始', '设计你的世界') if zh else ('Create something', 'Explore more', 'Start with an idea', 'Design your world'))
        c.text(x + w / 2, y + 72, heading, 15 if zh else 13, True, True)
        c.faux_text(x + w * .22, y + 99, w * .56, 2)
        controls = [label('next')]
        c.button(x + w / 2 - 40, y + 119, 80, 24, controls[0], True)
        cols = rng.choice((2, 3))
        cw = (w - 6 * (cols - 1)) / cols
        for j in range(cols):
            xx = x + j * (cw + 6)
            c.rect(xx, y + 159, cw, max(23, h - 160))
            c.icon(xx + 7, y + 166, rng.choice(('image', 'folder', 'plus')), 13)
            if h > 195:
                c.faux_text(xx + 7, y + 188, cw - 14, 1)

    elif kind == 'terminal':
        c.text(x, y, rng.choice(('POWERSHELL', 'TERMINAL', 'CONSOLE')), 9, True)
        c.line(x, y + 17, x + w, y + 17)
        texts = ['Cemivo Terminal v1.0', '$ cd workspace', '$ cemivo init', '+ Ready to create', f'$ build project-{rng.randint(1, 99)}', 'Loading modules...', '+ Done', '$ _']
        for j, line in enumerate(texts):
            c.text(x + 3, y + 30 + j * rng.choice((17, 18, 19)), line, rng.choice((8, 9)), j in (1, 2, 4), maxw=w - 7)
        details.append('命令行文本' if zh else 'command line text')

    elif kind == 'diagram':
        count = rng.randint(3, 6)
        vertical = rng.random() < .7
        labels = ('开始', '输入', '处理', '检查', '保存', '完成') if zh else ('Start', 'Input', 'Process', 'Check', 'Save', 'End')
        controls = [labels[i] for i in range(count - 1)] + [labels[-1]]
        c.text(x, y, '工作流程' if zh else 'Workflow', 13, True)
        if vertical:
            bw = rng.randint(88, 145)
            bh = min(25, (h - 29) / (count * 1.45))
            gap = max(7, (h - 35 - count * bh) / (count - 1))
            for j, text in enumerate(controls):
                yy = y + 29 + j * (bh + gap)
                xx = x + (w - bw) / 2 + (rng.randint(-13, 13) if 0 < j < count - 1 else 0)
                c.button(xx, yy, bw, bh, text, j in (0, count - 1))
                if j < count - 1:
                    mid = xx + bw / 2
                    c.line(mid, yy + bh + 1, mid, yy + bh + gap - 2)
                    c.line(mid - 3, yy + bh + gap - 5, mid, yy + bh + gap - 2)
                    c.line(mid + 3, yy + bh + gap - 5, mid, yy + bh + gap - 2)
        else:
            count = min(count, 4)
            controls = controls[:count]
            bw = (w - (count - 1) * 10) / count
            yy = y + h * .45
            for j, text in enumerate(controls):
                xx = x + j * (bw + 10)
                c.button(xx, yy, bw, 28, text, j in (0, count - 1))
                if j < count - 1:
                    c.line(xx + bw + 1, yy + 14, xx + bw + 9, yy + 14)
                    c.line(xx + bw + 6, yy + 11, xx + bw + 9, yy + 14)
        details.append('竖向流程' if zh and vertical else '横向流程' if zh else 'vertical flow' if vertical else 'horizontal flow')

    elif kind == 'shapes':
        count = rng.randint(1, 6)
        shape_names = []
        cols = 1 if count == 1 else 2 if count <= 4 else 3
        rows = math.ceil(count / cols)
        cw, ch = w / cols, h / rows
        for j in range(count):
            xx, yy = x + j % cols * cw, y + j // cols * ch
            sx, sy = xx + cw / 2, yy + ch * .4
            radius = rng.randint(12, max(13, int(min(cw, ch) * .3)))
            shape = rng.choice(('circle', 'rectangle', 'triangle', 'line'))
            if shape == 'circle':
                c.circle(sx, sy, radius, c.fg if rng.random() < .2 else c.bg)
            elif shape == 'rectangle':
                c.rect(sx - radius, sy - radius * .7, radius * 2, radius * 1.4, radius=rng.choice((0, 0, 3)))
            elif shape == 'triangle':
                c.d.polygon([(sx, sy - radius), (sx - radius, sy + radius * .7), (sx + radius, sy + radius * .7)], outline=c.fg, fill=c.bg)
            else:
                c.line(sx - radius, sy + radius * .4, sx + radius, sy - radius * .4, rng.choice((1, 2, 3)))
            name = {'circle': '圆形', 'rectangle': '矩形', 'triangle': '三角形', 'line': '直线'}[shape] if zh else shape
            shape_names.append(name)
            c.text(sx, yy + ch * .8, name, 8, center=True)
        details.append(('、'.join(shape_names)) if zh else ', '.join(shape_names))

    elif kind == 'text':
        texts = ('让想法清晰可见', '保持热爱', '创造新的可能', '从这里开始', '专注当下', '简单的力量') if zh else ('MAKE IT SIMPLE', 'STAY CREATIVE', 'A NEW IDEA', 'START HERE', 'FOCUS TODAY', 'LESS IS MORE')
        line = rng.choice(texts)
        sizefont = rng.randint(17, 27) if zh else rng.randint(15, 23)
        while _font(sizefont, True).getlength(line) > w - 14 and sizefont > 10:
            sizefont -= 1
        c.text(x + w / 2, y + h * rng.uniform(.25, .45), line, sizefont, True, True)
        subtitle = rng.choice(('把灵感变成现实', '每一天都是新的开始', '记录你的创意') if zh else ('Make your ideas real.', 'Every day is a fresh start.', 'Write your own story.'))
        c.text(x + w / 2, y + h * .6, subtitle, 9, center=True)
        c.line(x + w * .3, y + h * .78, x + w * .7, y + h * .78)
        controls = [line, subtitle]

    # Captions must describe the characters actually drawn, not strings that
    # were silently shortened to fit a button. Preserve the requested string
    # in audit metadata when a genuine eight-pixel space limit remains.
    label_audit = []
    title, audit = c.caption_label(title)
    label_audit.append({'field': 'title', **audit})
    visible_controls = []
    for value in controls:
        visible, audit = c.caption_label(value, button=kind != 'text')
        visible_controls.append(visible)
        label_audit.append({'field': 'text' if kind == 'text' else 'button', **audit})
    controls = visible_controls
    for audit in label_audit:
        left, top, right, bottom = audit['bbox']
        if not (0 <= left <= right <= size and 0 <= top <= bottom <= size):
            raise ValueError(f'Caption label extends outside the image: {audit}')
    kind_name = (NAMES_ZH if zh else NAMES_EN)[kind_index]
    if zh:
        prefix = rng.choice(('生成一个', '画一个', '设计一个', '给我一个', '制作一个'))
        prompt = f'{prefix}黑白{kind_name}，标题“{title}”'
        if details:
            prompt += '，' + '，'.join(details)
        if controls:
            prompt += ('，文字' if kind == 'text' else '，按钮') + '、'.join(f'“{v}”' for v in controls)
        if dark:
            prompt += '，黑底白线'
    else:
        prefix = rng.choice(('Create a', 'Draw a', 'Design a', 'Generate a', 'Show a'))
        prompt = f'{prefix} black and white {kind_name}, title "{title}"'
        if details:
            prompt += ', ' + ', '.join(details)
        if controls:
            prompt += (', text ' if kind == 'text' else ', buttons ') + ', '.join(f'"{v}"' for v in controls)
        if dark:
            prompt += ', black background and white lines'
    metadata = {
        'index': int(index), 'seed': int(seed), 'kind': kind, 'prompt': prompt,
        'language': 'zh' if zh else 'en', 'title': title, 'buttons': controls,
        'sidebar': has_sidebar, 'dark': dark,
        'rendering_version': 2, 'caption_label_audit': label_audit,
    }
    if size != 256:
        metadata.update(rendering_version=3, render_native_size=size,
            render_method='native-vector-and-font-rasterization', layout_coordinate_size=256)
    return np.asarray(c.image, dtype=np.uint8), metadata


def _job(args):
    return _draw_sample(*args)


def generate_dataset(output, count=50000, size=256, workers=0, seed=20261025,
                     shard_size=10000, append=False, val_fraction=.05, reference=None):
    output = Path(output)
    if (isinstance(size, bool) or not isinstance(size, int) or
            count < 1 or not 64 <= size <= 1024 or not 0 <= val_fraction <= .5):
        raise ValueError('count > 0, integer 64 <= size <= 1024 and 0 <= val_fraction <= .5 are required.')
    reference_rows = {}
    reference_split_hashes = {'train': set(), 'val': set()}
    if reference:
        reference = Path(reference).resolve()
        if reference == output.resolve():
            raise ValueError('A revised dataset must use a different output directory from its reference.')
        reference_manifest = json.loads((reference / 'manifest.json').read_text(encoding='utf-8'))
        if reference_manifest['base_seed'] != seed or reference_manifest['size'] != size:
            raise ValueError('A paired revision must preserve its reference seed and native size.')
        for shard in reference_manifest['shards']:
            for line in (reference / shard['metadata']).read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                reference_rows[row['index']] = row
                if row.get('content_sha256'):
                    reference_split_hashes[row['split']].add(row['content_sha256'])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if not append:
            raise FileExistsError(f'{manifest_path} already exists; use --append or another output directory.')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['size'] != size or manifest['base_seed'] != seed:
            raise ValueError('Append must keep the same --size and --seed.')
        if size != 256 and (manifest.get('rendering_version') != 3 or
                manifest.get('render_native_size') != size or
                manifest.get('render_method') != 'native-vector-and-font-rasterization'):
            raise ValueError('Cannot append native-rendered pixels to a legacy resized dataset. Use a new output directory.')
        if size == 256:
            # Historical V2 at exactly 256px was already natively rasterized.
            # Adding these true fields does not relabel any enlarged raster.
            manifest.setdefault('render_native_size', 256)
            manifest.setdefault('render_method', 'native-vector-and-font-rasterization')
            manifest.setdefault('layout_coordinate_size', 256)
    else:
        manifest = {'version': 1, 'format': 'npy-uint8-grayscale-shards', 'size': size,
            'dtype': 'uint8', 'normalization': 'image / 127.5 - 1', 'base_seed': seed,
            'count': 0, 'train_count': 0, 'val_count': 0, 'val_fraction': val_fraction,
            'shards': [], 'vocab': 'vocab.json', 'max_length': 192,
            'provenance': 'Original procedural UI pixel training data. No pretrained images or weights.'}
        manifest['rendering_version'] = 2 if size == 256 else 3
        manifest['render_native_size'] = size
        manifest['render_method'] = 'native-vector-and-font-rasterization'
        manifest['layout_coordinate_size'] = 256
        manifest['caption_policy'] = 'Captioned text is checked against visible drawn labels; font >=8px; button widths adapt and any unavoidable shortening is recorded.'
        if reference:
            manifest['paired_reference'] = str(reference)
        _dump(output / 'vocab.json', build_vocab())
    # Train and validation are divided before random rendering. Their seed streams
    # occupy different parity classes, so not a single layout seed is shared.
    val_fraction = manifest['val_fraction']
    start = manifest['count']
    remaining = count
    # Content hashes guarantee that both splits contain genuinely different
    # raster samples, not merely different random seeds or metadata records.
    seen_images = set()
    for shard in manifest['shards']:
        old_rows = [json.loads(line) for line in (output / shard['metadata']).read_text(encoding='utf-8').splitlines()]
        if size != 256 and any(row.get('rendering_version') != 3 or
                row.get('render_native_size') != size or
                row.get('render_method') != 'native-vector-and-font-rasterization' for row in old_rows):
            raise ValueError('Native append refused: existing row metadata does not establish native rendering.')
        old_pixels = None
        for old in old_rows:
            digest = old.get('content_sha256')
            if not digest:
                if old_pixels is None:
                    old_pixels = np.load(output / shard['images'], mmap_mode='r')
                digest = hashlib.sha256(old_pixels[old['offset']].tobytes()).hexdigest()
            seen_images.add(digest)
        del old_pixels
    executor = ProcessPoolExecutor(max_workers=workers) if workers else None
    try:
        while remaining:
            n = min(shard_size, remaining)
            shard_index = len(manifest['shards'])
            image_name, meta_name = f'images-{shard_index:05}.npy', f'metadata-{shard_index:05}.jsonl'
            image_tmp = output / (image_name + '.tmp')
            meta_tmp = output / (meta_name + '.tmp')
            pixels = np.lib.format.open_memmap(image_tmp, mode='w+', dtype=np.uint8, shape=(n, size, size))
            jobs = []
            splits = []
            for local in range(n):
                index = start + local
                # The multiplicative hash gives reproducible class-balanced splits
                # without taking a contiguous region from any family.
                is_val = ((index * 2654435761 + seed) % 1000000) < int(val_fraction * 1000000)
                sample_seed = seed * 100000000 + index * 2 + int(is_val)
                if index in reference_rows:
                    # Carry forward even the rare seeds that were resampled
                    # for deduplication in the original dataset.
                    old = reference_rows[index]
                    sample_seed = old['seed']
                    is_val = old['split'] == 'val'
                jobs.append((sample_seed, size, index))
                splits.append('val' if is_val else 'train')
            iterator = executor.map(_job, jobs, chunksize=32) if executor else map(_job, jobs)
            train_n = val_n = 0
            with meta_tmp.open('w', encoding='utf-8', newline='\n') as metadata:
                for local, (image_array, row) in enumerate(iterator):
                    digest = hashlib.sha256(image_array.tobytes()).hexdigest()
                    attempt = 0
                    opposite = 'train' if splits[local] == 'val' else 'val'
                    while digest in seen_images or digest in reference_split_hashes[opposite]:
                        attempt += 1
                        # Even offsets preserve the train/validation seed parity.
                        next_seed = jobs[local][0] + attempt * 1000000000000000000
                        image_array, row = _draw_sample(next_seed, size, start + local)
                        digest = hashlib.sha256(image_array.tobytes()).hexdigest()
                    seen_images.add(digest)
                    if len(row['prompt'].encode('utf-8')) > 190:
                        raise ValueError(f'Caption exceeds 190 UTF-8 content bytes at index {row["index"]}')
                    pixels[local] = image_array
                    row['content_sha256'] = digest
                    row['split'] = splits[local]
                    row['shard'] = shard_index
                    row['offset'] = local
                    train_n += row['split'] == 'train'
                    val_n += row['split'] == 'val'
                    metadata.write(json.dumps(row, ensure_ascii=False) + '\n')
            pixels.flush()
            del pixels
            image_tmp.replace(output / image_name)
            meta_tmp.replace(output / meta_name)
            manifest['shards'].append({'images': image_name, 'metadata': meta_name,
                'count': n, 'start_index': start, 'train_count': train_n, 'val_count': val_n})
            manifest['count'] += n
            manifest['train_count'] += train_n
            manifest['val_count'] += val_n
            _dump(manifest_path, manifest)
            print(f"Committed {manifest['count']:,} samples ({train_n:,} train / {val_n:,} val in this shard)", flush=True)
            start += n
            remaining -= n
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    return manifest


class Dataset:
    """Numpy-only loader; can be passed directly to torch.utils.data.DataLoader.

    Default output: image float32[1,H,W] in [-1,1], tokens int64[max_length],
    prompt, kind, index. Set normalize=False to return image uint8[1,H,W].
    Metadata is loaded once; image shards stay memory mapped.
    """
    def __init__(self, root, split='train', max_length=192, normalize=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        self.vocab = json.loads((self.root / self.manifest['vocab']).read_text(encoding='utf-8'))
        self.max_length = max_length
        self.normalize = normalize
        if split not in ('train', 'val', 'all', None):
            raise ValueError('split must be train, val, or all')
        self.rows = []
        self._maps = {}
        for shard_index, shard in enumerate(self.manifest['shards']):
            with (self.root / shard['metadata']).open(encoding='utf-8') as source:
                for row in map(json.loads, source):
                    if split in ('all', None) or row['split'] == split:
                        row['shard'] = shard_index
                        self.rows.append(row)
        self._token_cache = np.zeros((len(self.rows), max_length), dtype=np.int64)
        for i, row in enumerate(self.rows):
            self._token_cache[i] = encode_prompt(row['prompt'], max_length)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        shard_index = row['shard']
        if shard_index not in self._maps:
            filename = self.manifest['shards'][shard_index]['images']
            self._maps[shard_index] = np.load(self.root / filename, mmap_mode='r')
        pixels = self._maps[shard_index][row['offset']]
        image = pixels.astype(np.float32) / 127.5 - 1 if self.normalize else np.array(pixels)
        return {'image': image[None], 'tokens': self._token_cache[index].copy(),
                'prompt': row['prompt'], 'kind': row['kind'], 'index': row['index']}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_maps'] = {}
        return state


UITrainingDataset = Dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='neural/data')
    parser.add_argument('--count', type=int, default=50000)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--shard-size', type=int, default=10000)
    parser.add_argument('--val-fraction', type=float, default=.05)
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--reference', help='Paired previous dataset: preserve per-index seed and split; forbid writing the reference')
    args = parser.parse_args()
    manifest = generate_dataset(**vars(args))
    print(json.dumps({k: manifest[k] for k in ('count', 'size', 'train_count', 'val_count')}, indent=2))


if __name__ == '__main__':
    main()
