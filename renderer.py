"""Cemivo's deterministic black/white scene renderer. No network or pretrained assets.

The learned model chooses a layout family; this module draws exact linework and
legible text. SVG is the immediate output. PNG export rasterizes the same scene.
"""
from __future__ import annotations

import html
import io
import math
import random
import re
import xml.etree.ElementTree as ET
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont


KINDS = ('chat', 'login', 'dashboard', 'settings', 'files', 'shop', 'music',
         'calendar', 'editor', 'mail', 'browser', 'terminal', 'diagram', 'shapes', 'text')
TITLES = dict(zip(KINDS, ('Cemivo Chat', '登录账户', '数据概览', '偏好设置', '我的文件',
                        '精选商店', '我的音乐', '日程安排', '文档编辑器', '收件箱',
                        '网页浏览器', 'Terminal', '工作流程', '基础图形', '文字画布')))


def _number(v):
    return str(round(float(v), 2)).rstrip('0').rstrip('.') if float(v) % 1 else str(int(v))


def _escape(v):
    return html.escape(str(v), quote=True)


def _fit(text, width, size=16):
    """Conservative character-width clipping without loading any font at inference."""
    used, out = 0.0, ''
    for ch in str(text):
        step = size * (1.0 if ord(ch) > 255 else .58)
        if used + step > max(0, width - size):
            return out + '…'
        used += step
        out += ch
    return out


class Scene:
    def __init__(self, width, height, prompt, style, rng):
        self.w, self.h = width, height
        self.dark = bool(re.search(r'深色|暗色|黑底|dark|black background', prompt, re.I))
        self.bg, self.fg = ('#000000', '#ffffff') if self.dark else ('#ffffff', '#000000')
        self.radius = 12 if style == 'rounded' else 3 if style == 'compact' else 7
        self.compact = style == 'compact'
        self.mobile = width / height < .85 or width < 660
        self.rng = rng
        self.parts = []
        self.count = 0
        self.rect(0, 0, width, height, fill=self.bg, stroke='none', radius=0)

    def add(self, tag, **attrs):
        self.parts.append('<' + tag + ' ' + ' '.join(f'{k.replace("_", "-")}="{_escape(_number(v) if isinstance(v, (int, float)) else v)}"' for k, v in attrs.items()) + '/>')
        self.count += 1

    def rect(self, x, y, w, h, fill=None, stroke=None, radius=None, sw=1.5):
        self.add('rect', x=x, y=y, width=max(0, w), height=max(0, h),
                 rx=self.radius if radius is None else radius,
                 fill=fill or self.bg, stroke=stroke or self.fg, stroke_width=sw)

    def line(self, x1, y1, x2, y2, sw=1.5, dash=None):
        attrs = dict(x1=x1, y1=y1, x2=x2, y2=y2, stroke=self.fg, stroke_width=sw, stroke_linecap='round')
        if dash:
            attrs['stroke_dasharray'] = dash
        self.add('line', **attrs)

    def circle(self, x, y, r, fill=None, sw=1.5):
        self.add('circle', cx=x, cy=y, r=r, fill=fill or self.bg, stroke=self.fg, stroke_width=sw)

    def poly(self, points, close=False, fill='none', sw=1.5):
        self.add('polygon' if close else 'polyline', points=' '.join(f'{_number(x)},{_number(y)}' for x, y in points), fill=fill, stroke=self.fg, stroke_width=sw, stroke_linecap='round', stroke_linejoin='round')

    def text(self, x, y, value, size=16, bold=False, fill=None, anchor='start', maxw=None):
        # These symbols are absent from several Windows CJK font versions. Keep
        # the same portable content in SVG and PNG rather than showing tofu.
        value = str(value).translate(str.maketrans({'▾': 'v', '✓': '+', '⌕': '', '↻': 'o', '↶': '<', '↷': '>', '▶': '>', '⋯': '…'}))
        value = _fit(value, maxw, size) if maxw is not None else str(value)
        self.parts.append(f'<text x="{_number(x)}" y="{_number(y)}" fill="{fill or self.fg}" font-size="{_number(size)}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{_escape(value)}</text>')
        self.count += 1

    def button(self, x, y, w, label, primary=False, h=40, icon=None):
        self.rect(x, y, w, h, fill=self.fg if primary else self.bg)
        if icon:
            self.icon(icon, x + 13, y + (h - 18) / 2, 18)
        self.text(x + w / 2 + (8 if icon else 0), y + h / 2 + 5, label, 14, primary, self.bg if primary else self.fg, 'middle', maxw=w - 18)

    def icon(self, name, x, y, s=20):
        def p(points, close=False, fill='none'):
            self.poly([(x + a * s, y + b * s) for a, b in points], close, fill)
        if name in ('plus', 'add'):
            self.line(x + s / 2, y + 2, x + s / 2, y + s - 2)
            self.line(x + 2, y + s / 2, x + s - 2, y + s / 2)
        elif name in ('search',):
            self.circle(x + s * .42, y + s * .42, s * .3)
            self.line(x + s * .65, y + s * .65, x + s * .95, y + s * .95)
        elif name == 'send':
            p([(.08, .1), (.95, .5), (.08, .9), (.28, .5), (.08, .1)], True)
            self.line(x + s * .28, y + s * .5, x + s * .72, y + s * .5)
        elif name == 'mic':
            self.rect(x + s * .35, y + s * .05, s * .3, s * .55, radius=s * .15)
            p([(.15, .42), (.15, .62), (.32, .78), (.68, .78), (.85, .62), (.85, .42)])
            self.line(x + s * .5, y + s * .78, x + s * .5, y + s)
            self.line(x + s * .3, y + s, x + s * .7, y + s)
        elif name == 'attach':
            p([(.3, .75), (.72, .32), (.62, .18), (.43, .2), (.1, .58), (.1, .78), (.3, .95), (.5, .95), (.95, .45)])
        elif name == 'menu':
            for k in (.2, .5, .8):
                self.line(x, y + s * k, x + s, y + s * k)
        elif name == 'folder':
            p([(.05, .25), (.4, .25), (.5, .4), (.95, .4), (.95, .9), (.05, .9)], True)
        elif name == 'image':
            self.rect(x, y, s, s, radius=2)
            self.circle(x + .7 * s, y + .28 * s, .1 * s)
            p([(.1, .88), (.38, .5), (.58, .73), (.73, .58), (.9, .88)])
        elif name == 'user':
            self.circle(x + s / 2, y + s * .28, s * .2)
            p([(.1, .95), (.18, .65), (.38, .54), (.62, .54), (.82, .65), (.9, .95)])
        elif name == 'bell':
            p([(.1, .75), (.25, .6), (.25, .28), (.4, .1), (.6, .1), (.75, .28), (.75, .6), (.9, .75), (.1, .75)], True)
            self.line(x + s * .4, y + s * .95, x + s * .6, y + s * .95)
        elif name == 'check':
            p([(.1, .5), (.38, .8), (.95, .14)])
        elif name == 'arrow':
            self.line(x, y + s / 2, x + s, y + s / 2)
            p([(.65, .15), (1, .5), (.65, .85)])
        elif name == 'play':
            p([(.23, .08), (.85, .5), (.23, .92)], True, self.fg)
        elif name == 'mail':
            self.rect(x, y + s * .15, s, s * .7, radius=2)
            p([(0, .15), (.5, .55), (1, .15)])
        elif name == 'gear':
            self.circle(x + s / 2, y + s / 2, s * .34)
            self.circle(x + s / 2, y + s / 2, s * .12)
            for angle in range(0, 360, 45):
                a = math.radians(angle)
                self.line(x + s * (.5 + .35 * math.cos(a)), y + s * (.5 + .35 * math.sin(a)), x + s * (.5 + .5 * math.cos(a)), y + s * (.5 + .5 * math.sin(a)))
        else:
            for k in (.2, .5, .8):
                self.circle(x + s * k, y + s / 2, 1.3, self.fg, .5)

    def svg(self, title):
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{_escape(title)}"><title>{_escape(title)}</title>' + ''.join(self.parts) + '</svg>'


def _quotes(prompt):
    return [a or b or c for a, b, c in re.findall(r'“([^”]{1,80})”|"([^"\n]{1,80})"|「([^」]{1,80})」', prompt)]


def _title(prompt, kind):
    match = re.search(r'(?:标题|名称|名字|title|name)\s*(?:为|叫|是|[:：=])?\s*[“"「]([^”"」]{1,64})[”"」]', prompt, re.I)
    return match.group(1) if match else TITLES[kind]


def _buttons(prompt):
    match = re.search(r'(?:按钮|buttons?)\s*(?:有|包括|包含|为|是|[:：=])?\s*(.{1,160})', prompt, re.I)
    if not match:
        return []
    text = re.split(r'[。\n；;]', match.group(1))[0]
    quoted = _quotes(text)
    if quoted:
        return quoted[:7]
    names = re.split(r'[,，、/]|以及|和', text)
    return [n.strip(' ：: ') for n in names if 0 < len(n.strip(' ：: ')) <= 12][:7]


def _frame(s, title, buttons=None, sidebar=True):
    w, h = s.w, s.h
    margin = 24 if not s.mobile else 16
    s.rect(margin, margin, w - 2 * margin, h - 2 * margin, radius=s.radius + 4, sw=2)
    s.line(margin, margin + 58, w - margin, margin + 58)
    for i in range(3):
        s.circle(margin + 20 + i * 16, margin + 29, 3.5, s.fg if i == 0 else None, 1)
    s.text(margin + (86 if w > 800 else 74), margin + 35, title, 17, True, maxw=w - 300 if w > 800 else w - 190)
    s.icon('search', w - margin - 74, margin + 20, 17)
    s.icon('more', w - margin - 36, margin + 20, 17)
    x, y = margin + 22, margin + 80
    innerw = w - margin * 2 - 44
    innerh = h - margin * 2 - 102
    if sidebar and not s.mobile:
        sw = max(155, min(230, w * .2))
        s.line(margin + sw, margin + 58, margin + sw, h - margin)
        x = margin + sw + 28
        innerw = w - margin - x - 28
    return x, y, innerw, innerh


def _nav(s, labels, active=0):
    left, top = (40, 112)
    if s.mobile:
        return
    width = max(155, min(230, s.w * .2)) - 32
    for i, label in enumerate(labels):
        yy = top + i * 52
        if yy + 42 > s.h - 96:
            break
        s.rect(left, yy, width, 40, fill=s.fg if i == active else s.bg, stroke=s.fg if i == active else 'none', radius=6)
        s.text(left + 15, yy + 26, label, 15, i == active, s.bg if i == active else s.fg, maxw=width - 28)
    s.text(left + 8, s.h - 52, 'CEMIVO  /  WORKSPACE', 10, maxw=width)


def _chat(s, title, prompt, custom):
    x, y, w, h = _frame(s, title)
    if not s.mobile:
        sw = max(155, min(230, s.w * .2))
        s.button(40, 105, sw - 32, '+  新建对话', h=42)
        s.text(48, 182, '最近会话', 12, True)
        for i, label in enumerate(('产品设计讨论', '今天的灵感', '界面原型评审', '项目协作群')):
            yy = 200 + i * 72
            if yy > s.h - 140:
                break
            s.rect(40, yy, sw - 32, 58, fill=s.fg if i == s.rng.randint(0, 0) else s.bg, stroke=s.fg if i == 0 else 'none')
            s.text(52, yy + 23, label, 14, i == 0, s.bg if i == 0 else s.fg, maxw=sw - 58)
            s.text(52, yy + 44, '开始新的想法…' if i else '好的，我们开始吧', 11, fill=s.bg if i == 0 else s.fg, maxw=sw - 58)
        s.icon('gear', 48, s.h - 72, 19)
        s.text(80, s.h - 57, '设置', 13)
    s.circle(x + 20, y + 20, 19)
    s.icon('user', x + 10, y + 8, 20)
    s.text(x + 52, y + 16, '设计助手', 17, True)
    s.text(x + 52, y + 39, s.rng.choice(['在线 · 随时与你讨论', '在线 · 一起探索新想法', '在线 · 开始今天的创作']), 11)
    if w > 400:
        s.button(x + w - 96, y + 2, 96, '语音通话', h=36)
    s.line(x, y + 60, x + w, y + 60)
    input_h = 118 if h > 430 else 90
    input_y = s.h - 46 - input_h
    msgs = [('你好！可以帮我设计一个聊天界面吗？', False), ('当然。你希望界面包含哪些功能？', True), ('需要会话列表、消息区，以及发送和附件按钮。', False), ('好的，所有交互入口都已为你准备好了。', True)]
    bubblew = min(w * (.70 + s.rng.random() * .06), 420 + s.rng.randrange(50))
    rowh = max(59, min(88, (input_y - y - 88) / 4))
    visible = 4 if input_y - y > 310 else 2
    for i, (msg, own) in enumerate(msgs[:visible]):
        yy = y + 82 + i * rowh
        bx = x + w - bubblew if own else x
        s.rect(bx, yy, bubblew, max(42, rowh - 16), fill=s.fg if own else s.bg, radius=s.radius)
        s.text(bx + 15, yy + max(28, (rowh - 16) / 2 + 5), msg, 13, fill=s.bg if own else s.fg, maxw=bubblew - 26)
    s.rect(x, input_y, w, input_h)
    s.text(x + 17, input_y + 30, '输入消息，与设计助手聊一聊…', 14, maxw=w - 32)
    base_y = input_y + input_h - 40
    for j, name in enumerate(('attach', 'image', 'mic')):
        s.icon(name, x + 17 + j * 38, base_y + 8, 18)
    labels = custom or ['发送']
    bw = min(100, (w - 145) / max(1, len(labels)))
    for i, label in enumerate(labels[-4:]):
        bx = x + w - 12 - (len(labels[-4:]) - i) * (bw + 7)
        s.button(bx, base_y, bw, label, i == len(labels[-4:]) - 1, h=32)


def _login(s, title, prompt, custom):
    _frame(s, title, sidebar=False)
    cw = min(410, s.w - 112)
    tight = s.h < 620
    cx, cy = (s.w - cw) / 2, max(98, s.h * .17)
    logo = 32 if tight else 44
    s.rect(cx + cw / 2 - logo / 2, cy, logo, logo, fill=s.fg, radius=10)
    s.text(cx + cw / 2, cy + logo * .73, 'C', 21 if tight else 25, True, s.bg, 'middle')
    s.text(s.w / 2, cy + (62 if tight else 84), title, 25 if tight else 27, True, anchor='middle', maxw=cw)
    s.text(s.w / 2, cy + (86 if tight else 111), '欢迎回来，继续你的创作', 13, anchor='middle')
    yy = cy + (117 if tight else 142)
    for label, placeholder in [('邮箱地址', 'name@example.com'), ('密码', '••••••••••')]:
        s.text(cx, yy, label, 13, True)
        s.rect(cx, yy + 11, cw, 40 if tight else 46)
        s.text(cx + 14, yy + (36 if tight else 40), placeholder, 14)
        yy += 76 if tight else 90
    if yy + 95 < s.h - 36:
        s.rect(cx, yy - 6, 14, 14, radius=2)
        s.text(cx + 24, yy + 6, '记住我', 12)
        s.text(cx + cw, yy + 6, '忘记密码？', 12, anchor='end')
        s.button(cx, yy + 28, cw, custom[0] if custom else '登录', True, h=45)
        s.text(s.w / 2, yy + 105, '还没有账户？  注册', 13, anchor='middle')


def _dashboard(s, title, prompt, custom):
    x, y, w, h = _frame(s, title)
    _nav(s, ['数据概览', '项目管理', '分析报告', '团队成员', '系统设置'])
    s.text(x, y + 23, '工作台', 24, True)
    s.text(x, y + 47, '一眼掌握项目的每一次进展', 13)
    cols = 2 if s.mobile else 3
    gap, staty = 14, y + 73
    cw = (w - gap * (cols - 1)) / cols
    for i, (label, val) in enumerate([('总访问量', '24,860'), ('活跃项目', '128'), ('转化率', '8.6%')][:cols]):
        xx = x + i * (cw + gap)
        s.rect(xx, staty, cw, 105)
        s.text(xx + 16, staty + 29, label, 12)
        s.text(xx + 16, staty + 67, val, 28, True)
        s.text(xx + 16, staty + 88, '↑ 12.8%  较上期', 10)
    chart_y, chart_h = staty + 128, min(230, max(100, h - 265))
    s.rect(x, chart_y, w, chart_h)
    s.text(x + 18, chart_y + 29, '访问趋势', 15, True)
    s.text(x + w - 18, chart_y + 29, '最近 7 天 ▾', 12, anchor='end')
    for j in range(3):
        yy = chart_y + 62 + j * max(10, (chart_h - 95) / 2)
        s.line(x + 35, yy, x + w - 25, yy, .8, '4 5')
    pts = [(x + 35 + k * (w - 70) / 8, chart_y + chart_h - 18 - max(20, chart_h - 80) * v) for k, v in enumerate([.1, .35, .22, .62, .44, .81, .68, .92, .78])]
    s.poly(pts, sw=2.5)
    for px, py in pts:
        s.circle(px, py, 3, s.fg)
    listy = chart_y + chart_h + 27
    if listy + 45 < s.h - 38:
        s.text(x, listy, '最近项目', 16, True)
        for i, label in enumerate(['界面设计系统', '移动端应用', '品牌升级计划']):
            yy = listy + 35 + i * 39
            if yy > s.h - 46:
                break
            s.text(x + 4, yy, label, 13)
            s.text(x + w - 10, yy, '进行中  →', 12, anchor='end')
            s.line(x, yy + 12, x + w, yy + 12, .8)


def _settings(s, title, prompt, custom):
    x, y, w, h = _frame(s, title)
    _nav(s, ['个人资料', '账户安全', '通知设置', '外观与显示', '帮助中心'], 3)
    s.text(x, y + 25, '外观与偏好', 24, True)
    s.text(x, y + 51, '按你的习惯，自定义工作空间', 13)
    rows = [('深色模式', '降低屏幕亮度，专注于内容', s.dark), ('桌面通知', '在收到新消息时提醒我', True), ('自动保存', '实时保存工作中的所有更改', True), ('紧凑布局', '使用更紧凑的面板和控件', s.compact)]
    for i, (label, sub, enabled) in enumerate(rows):
        yy = y + 83 + i * 87
        if yy + 64 > s.h - 115:
            break
        s.text(x, yy + 16, label, 16, True)
        s.text(x, yy + 42, sub, 12, maxw=w - 76)
        s.rect(x + w - 51, yy + 8, 48, 25, fill=s.fg if enabled else s.bg, radius=13)
        s.circle(x + w - (16 if enabled else 38), yy + 20.5, 8, s.bg if enabled else s.fg, .8)
        s.line(x, yy + 65, x + w, yy + 65, .8)
    s.button(x + max(0, w - 218), s.h - 98, 98, (custom + ['取消'])[0] if custom else '取消')
    s.button(x + w - 108, s.h - 98, 108, custom[-1] if custom else '保存更改', True)


def _files(s, title, prompt, custom):
    x, y, w, h = _frame(s, title)
    _nav(s, ['我的文件', '最近使用', '与我共享', '已收藏', '回收站'])
    s.text(x, y + 25, '我的文件', 24, True)
    s.button(x + w - 100, y - 1, 100, custom[0] if custom else '+  上传', True, h=36)
    s.rect(x, y + 54, w, 41)
    s.icon('search', x + 14, y + 66, 16)
    s.text(x + 42, y + 80, '搜索文件、文件夹…', 13)
    cols = 2 if s.mobile else 3
    cw = (w - 16 * (cols - 1)) / cols
    for i, label in enumerate(['设计资源', '项目文档', '图片素材'][:cols]):
        xx = x + i * (cw + 16)
        s.rect(xx, y + 118, cw, 102)
        s.icon('folder', xx + 16, y + 134, 27)
        s.text(xx + 16, y + 190, label, 14, True)
        s.text(xx + cw - 15, y + 190, '12', 11, anchor='end')
    ty = y + 257
    s.text(x, ty, '最近文件', 16, True)
    names = ['产品需求说明.pdf', '聊天界面设计.svg', '项目计划.docx', '品牌指南.pdf', '会议记录.txt']
    for i, name in enumerate(names):
        yy = ty + 42 + i * 48
        if yy > s.h - 51:
            break
        s.rect(x, yy - 18, 23, 29, radius=2)
        s.text(x + 36, yy + 1, name, 13, maxw=w - 140)
        s.text(x + w - 10, yy + 1, '2026/10/25  ⋯' if w > 420 else '⋯', 11, anchor='end')
        s.line(x, yy + 23, x + w, yy + 23, .8)


def _shop(s, title, prompt, custom):
    x, y, w, h = _frame(s, title, sidebar=False)
    s.text(x, y + 25, '日常好物', 25, True)
    s.text(x, y + 51, '简约设计 · 精选品质', 13)
    s.button(x + w - 105, y + 3, 105, '购物袋  02', h=36)
    tabs = ['全部', '办公', '生活', '配件']
    for i, label in enumerate(tabs):
        s.button(x + i * 76, y + 77, 65, label, i == 0, h=31)
    cols = 2 if s.mobile else 4 if w > 850 else 3
    cw, gap = (w - 18 * (cols - 1)) / cols, 18
    cardh = min(330, max(180, h - 154))
    for i in range(cols):
        xx, yy = x + i * (cw + gap), y + 131
        s.rect(xx, yy, cw, cardh)
        s.rect(xx + 12, yy + 12, cw - 24, max(70, cardh - 123))
        # A deliberately simple product silhouette.
        pw, ph = min(90, cw * .4), min(100, cardh * .32)
        s.rect(xx + (cw - pw) / 2, yy + 32, pw, ph, radius=8)
        for k in range(3):
            s.line(xx + (cw - pw) / 2 + 10, yy + 48 + k * 12, xx + (cw + pw) / 2 - 10, yy + 48 + k * 12, 1)
        s.text(xx + 14, yy + cardh - 84, ['灵感笔记本', '极简收纳盒', '桌面时钟', '便携水杯'][i], 15, True, maxw=cw - 28)
        s.text(xx + 14, yy + cardh - 57, ['¥ 49.00', '¥ 89.00', '¥ 129.00', '¥ 69.00'][i], 14)
        s.button(xx + 12, yy + cardh - 43, cw - 24, custom[0] if custom else '加入购物袋', i == 0, h=31)


def _music(s, title, prompt, custom):
    x, y, w, h = _frame(s, title)
    _nav(s, ['发现音乐', '我的收藏', '最近播放', '我的歌单', '+ 新建歌单'])
    art = min(190, w * .3) if not s.mobile else 130
    s.rect(x, y + 5, art, art)
    s.circle(x + art / 2, y + 5 + art / 2, art * .33)
    s.circle(x + art / 2, y + 5 + art / 2, art * .07, s.fg)
    tx = x + art + 25
    s.text(tx, y + 34, '精选歌单', 12)
    s.text(tx, y + 75, '专注时刻', 26 if w > 450 else 22, True, maxw=w - art - 25)
    s.text(tx, y + 105, '安静，让灵感慢慢发生。', 12, maxw=w - art - 25)
    s.button(tx, y + 127, min(125, w - art - 25), '▶  播放全部', True, h=36)
    starty = y + art + 51
    for i, (name, artist) in enumerate([('晨光', 'Morning Studio'), ('漫步', 'The Quiet Room'), ('白色房间', 'Cemivo Sessions'), ('远山', 'Soft Waves')]):
        yy = starty + i * 56
        if yy > s.h - 140:
            break
        s.text(x + 4, yy, f'{i + 1:02}', 12)
        s.text(x + 45, yy, name, 14, True)
        if w > 380:
            s.text(x + w * .5, yy, artist, 12)
        s.text(x + w - 5, yy, f'0{3 + i % 2}:2{i}', 12, anchor='end')
        s.line(x, yy + 18, x + w, yy + 18, .8)
    py = s.h - 104
    s.line(x, py, x + w, py)
    s.text(x, py + 32, '晨光  /  Morning Studio', 12, maxw=w - 120)
    s.icon('play', x + w - 70, py + 16, 23)
    s.line(x, py + 51, x + w, py + 51, 1)
    s.line(x, py + 51, x + w * .36, py + 51, 4)
    s.circle(x + w * .36, py + 51, 5, s.fg)


def _calendar(s, title, prompt, custom):
    x, y, w, h = _frame(s, title)
    _nav(s, ['日历视图', '所有日程', '我的任务', '共享日历', '设置'])
    s.text(x, y + 26, '2026 年 10 月', 24, True)
    s.button(x + w - 91, y + 1, 91, custom[0] if custom else '+  日程', True, h=35)
    gy = y + 63
    gh = max(210, s.h - gy - 56)
    cellw, cellh = w / 7, (gh - 35) / 5
    s.rect(x, gy, w, gh, radius=0)
    for j, day in enumerate('一二三四五六日'):
        s.text(x + (j + .5) * cellw, gy + 23, day, 12, anchor='middle')
        if j:
            s.line(x + j * cellw, gy, x + j * cellw, gy + gh, .8)
    s.line(x, gy + 35, x + w, gy + 35, .8)
    for k in range(1, 5):
        s.line(x, gy + 35 + k * cellh, x + w, gy + 35 + k * cellh, .8)
    for d in range(1, 32):
        idx = d + 2
        row, col = idx // 7, idx % 7
        xx, yy = x + col * cellw, gy + 35 + row * cellh
        if d == 25:
            s.circle(xx + 20, yy + 20, 13, s.fg)
        s.text(xx + 20, yy + 24, d, 12, d == 25, s.bg if d == 25 else s.fg, 'middle')
        if d in (7, 15, 20, 25) and cellh > 67:
            s.rect(xx + 7, yy + 38, cellw - 14, min(25, cellh - 43), fill=s.fg, radius=3, sw=.5)
            s.text(xx + 12, yy + 54, {7: '设计评审', 15: '团队会议', 20: '项目交付', 25: '创作时间'}[d], 10, fill=s.bg, maxw=cellw - 20)
        elif d in (7, 15, 20, 25) and cellh > 40:
            s.circle(xx + 20, yy + cellh - 10, 2.5, s.fg)


def _editor(s, title, prompt, custom):
    x, y, w, h = _frame(s, title, sidebar=False)
    tools = ['文件', '编辑', '插入', '格式', '视图']
    for i, label in enumerate(tools):
        s.text(x + i * 60, y + 16, label, 13)
    if w > 480:
        s.button(x + w - 88, y - 7, 88, custom[-1] if custom else '分享', True, h=32)
    s.line(x, y + 35, x + w, y + 35)
    s.text(x + 7, y + 65, '正文 ▾     16 ▾      B     I     U     ≡     • 列表     ↶     ↷', 14, maxw=w - 20)
    s.line(x, y + 83, x + w, y + 83)
    pw = min(680, w - 24)
    px, py = x + (w - pw) / 2, y + 109
    ph = max(150, s.h - py - 50)
    s.rect(px, py, pw, ph, radius=0)
    pad = 34 if pw > 500 else 22
    s.text(px + pad, py + 52, '让想法成为现实', 26, True, maxw=pw - pad * 2)
    s.text(px + pad, py + 80, '项目计划  /  2026.10.25', 11)
    content = ['每一个伟大的作品，都从一个简单的想法开始。', '在这里记录你的灵感，组织信息，连接新的可能。', '', '01  目标与愿景', '清晰的目标，让每一步都有方向。', '专注于真正重要的事情，并持续迭代。', '', '02  下一步计划', '定义问题，探索方案，验证想法。']
    yy = py + 124
    for line in content:
        if yy > py + ph - 25:
            break
        if line:
            s.text(px + pad, yy, line, 13, line.startswith(('01', '02')), maxw=pw - pad * 2)
        yy += 29


def _mail(s, title, prompt, custom):
    x, y, w, h = _frame(s, title)
    _nav(s, ['收件箱   12', '星标邮件', '已发送', '草稿箱', '垃圾箱'])
    s.text(x, y + 24, '收件箱', 24, True)
    s.button(x + w - 105, y, 105, custom[0] if custom else '+  写邮件', True, h=36)
    s.line(x, y + 54, x + w, y + 54)
    messages = [('Cemivo Team', '欢迎加入 Cemivo', '从现在开始，把你的想法变成现实。', '10:25'), ('设计协作组', '本周界面评审安排', '关于新版本聊天功能的设计细节…', '09:46'), ('产品团队', '你有一份新的项目邀请', '点击查看项目内容和协作成员。', '昨天'), ('每周灵感', '简单，也可以很有力量', '这一周值得收藏的设计灵感。', '昨天'), ('日程助手', '明天的会议提醒', '你的项目复盘会议将在明天开始。', '周五')]
    for i, (sender, subject, snippet, date) in enumerate(messages):
        yy = y + 77 + i * 99
        if yy + 69 > s.h - 45:
            break
        s.rect(x + 1, yy + 12, 15, 15, radius=2)
        s.text(x + 32, yy + 19, sender, 13, i < 2)
        s.text(x + w - 3, yy + 19, date, 11, anchor='end')
        s.text(x + 32, yy + 44, subject, 15, i < 2, maxw=w - 39)
        s.text(x + 32, yy + 67, snippet, 12, maxw=w - 39)
        s.line(x, yy + 86, x + w, yy + 86, .8)


def _browser(s, title, prompt, custom):
    x, y, w, h = _frame(s, title, sidebar=False)
    s.text(x + 1, y + 19, '←    →    ↻', 19)
    s.rect(x + 106, y - 6, max(50, w - 157), 38, radius=18)
    s.text(x + 126, y + 19, '⌕   cemivo.local / explore', 13, maxw=w - 194)
    s.icon('plus', x + w - 24, y + 2, 18)
    s.line(x, y + 50, x + w, y + 50)
    s.text(x + 18, y + 95, 'CEMIVO', 18, True)
    s.text(x + w - 18, y + 95, '探索    关于    联系', 12, anchor='end')
    midy = y + 174
    s.text(x + w / 2, midy, '让创作，简单一点。', min(41, (w - 32) / 10), True, anchor='middle', maxw=w - 32)
    s.text(x + w / 2, midy + 40, '从一个想法开始，探索无限可能。', 14, anchor='middle', maxw=w - 32)
    s.button(x + w / 2 - 72, midy + 66, 144, custom[0] if custom else '开始探索  →', True, h=44)
    cy = midy + 156
    if cy + 110 < s.h - 44:
        cols = 2 if s.mobile else 3
        cw = (w - 18 * (cols - 1)) / cols
        for i, label in enumerate(['专注创作', '快速协作', '持续探索'][:cols]):
            xx = x + i * (cw + 18)
            s.rect(xx, cy, cw, min(160, s.h - cy - 50))
            s.icon(['image', 'user', 'search'][i], xx + 20, cy + 20, 25)
            s.text(xx + 20, cy + 79, label, 16, True)
            s.text(xx + 20, cy + 105, '为每一个灵感，找到新可能。', 12, maxw=cw - 34)


def _terminal(s, title, prompt, custom):
    x, y, w, h = _frame(s, title, sidebar=False)
    s.text(x, y + 18, 'POWERSHELL   ▾          +       ⋯', 12, True)
    s.line(x, y + 36, x + w, y + 36)
    lines = ['Cemivo Terminal  v1.0', 'Copyright (c) Cemivo. All rights reserved.', '', 'PS C:\\Workspace> cemivo status', '  ✓ Local workspace connected', '  ✓ Rendering engine ready', '  ✓ No external services required', '', 'PS C:\\Workspace> generate --style monochrome', '  Creating lines and text...', '  Output saved: design.svg', '', 'PS C:\\Workspace> _']
    for i, line in enumerate(lines):
        yy = y + 69 + i * 29
        if yy > s.h - 58:
            break
        s.text(x + 7, yy, line, 13, line.startswith('PS'), maxw=w - 14)


def _diagram(s, title, prompt, custom):
    x, y, w, h = _frame(s, title, sidebar=False)
    s.text(x, y + 24, title, 24, True)
    s.text(x, y + 49, '从起点到结果，每一步都清晰可见', 13)
    labels = _quotes(prompt) or ['开始', '输入信息', '处理任务', '检查结果', '完成']
    labels = labels[:7]
    vertical = s.mobile or '竖' in prompt
    if vertical:
        bw = min(260, w * .7)
        bh = min(51, max(32, (h - 94) / (len(labels) * 1.65)))
        xx, yy = x + (w - bw) / 2, y + 80
        for i, label in enumerate(labels):
            s.rect(xx, yy, bw, bh, fill=s.fg if i in (0, len(labels) - 1) else s.bg, radius=bh / 2 if i in (0, len(labels) - 1) else s.radius)
            s.text(xx + bw / 2, yy + bh / 2 + 5, label, 14, True, s.bg if i in (0, len(labels) - 1) else s.fg, 'middle', maxw=bw - 25)
            if i < len(labels) - 1:
                s.line(xx + bw / 2, yy + bh + 2, xx + bw / 2, yy + bh + 24)
                s.poly([(xx + bw / 2 - 5, yy + bh + 18), (xx + bw / 2, yy + bh + 25), (xx + bw / 2 + 5, yy + bh + 18)])
            yy += bh + 33
    else:
        n = len(labels)
        gap = min(38, w / (n * 5))
        bw = (w - gap * (n - 1)) / n
        yy, bh = y + min(180, h * .35), 78
        for i, label in enumerate(labels):
            xx = x + i * (bw + gap)
            s.rect(xx, yy, bw, bh, fill=s.fg if i in (0, n - 1) else s.bg, radius=20 if i in (0, n - 1) else s.radius)
            s.text(xx + bw / 2, yy + 46, label, 14, True, s.bg if i in (0, n - 1) else s.fg, 'middle', maxw=bw - 18)
            if i < n - 1:
                s.icon('arrow', xx + bw + 5, yy + bh / 2 - 7, gap - 10)
        s.text(x + w / 2, yy + 137, '输入  →  处理  →  反馈', 13, anchor='middle')


def _shapes(s, title, prompt, custom):
    x, y, w, h = _frame(s, title, sidebar=False)
    s.text(x, y + 24, title, 24, True)
    s.text(x, y + 50, '基本线条与几何图形', 13)
    cols = 2 if s.mobile else 3
    cw, ch = w / cols, max(105, min(215, (h - 87) / 2))
    labels = ['直线', '矩形', '圆形', '三角形', '折线', '文字']
    single = None
    for key, index in [('三角形', 3), ('正方形', 1), ('圆形', 2), ('直线', 0)]:
        if key in prompt and not any(k in prompt for k in ('和', '、', '图形', '基础', '多个')):
            single = index
    indices = [single] if single is not None else list(range(6 if cols == 3 else 4))
    for k, idx in enumerate(indices):
        xx, yy = x + (k % cols) * cw, y + 80 + (k // cols) * ch
        if single is not None:
            xx, yy, cw, ch = x, y + 80, w, max(160, h - 80)
        cx, cy, sz = xx + cw / 2, yy + ch * .43, min(65, cw * .24, ch * .29)
        if idx == 0:
            s.line(cx - sz, cy + sz * .45, cx + sz, cy - sz * .45, 3)
        elif idx == 1:
            s.rect(cx - sz, cy - sz * .65, sz * 2, sz * 1.3, radius=0, sw=2)
        elif idx == 2:
            s.circle(cx, cy, sz, sw=2)
        elif idx == 3:
            s.poly([(cx, cy - sz), (cx - sz, cy + sz * .75), (cx + sz, cy + sz * .75)], True, sw=2)
        elif idx == 4:
            s.poly([(cx - sz, cy + sz * .5), (cx - sz * .3, cy - sz * .25), (cx + sz * .3, cy + sz * .3), (cx + sz, cy - sz * .7)], sw=2)
        else:
            s.text(cx, cy + 15, 'Aa 文字', 29, True, anchor='middle')
        s.text(cx, yy + ch - 17, labels[idx], 13, anchor='middle')


def _text(s, title, prompt, custom):
    x, y, w, h = _frame(s, title, sidebar=False)
    quotes = _quotes(prompt)
    lines = quotes or ['让每一个想法', '清晰可见。', 'Make every idea visible.']
    size = max(22, min(65, w / max(12, max(len(t) for t in lines)) * .87, (h - 80) / (len(lines) * 1.8)))
    total = len(lines) * size * 1.65
    yy = y + max(70, (h - total) / 2 + size)
    for i, label in enumerate(lines[:10]):
        s.text(x + w / 2, yy + i * size * 1.65, label, size, i == 0, anchor='middle', maxw=w - 28)
    s.line(x + w * .35, min(s.h - 65, yy + len(lines) * size * 1.65 + 20), x + w * .65, min(s.h - 65, yy + len(lines) * size * 1.65 + 20), 2)


_DRAW = dict(chat=_chat, login=_login, dashboard=_dashboard, settings=_settings,
             files=_files, shop=_shop, music=_music, calendar=_calendar,
             editor=_editor, mail=_mail, browser=_browser, terminal=_terminal,
             diagram=_diagram, shapes=_shapes, text=_text)


def generate_svg(prompt, kind='chat', width=1280, height=800, seed=1, variant=0, style='outline'):
    """Build an SVG scene in memory, usually below 2 ms on a desktop CPU."""
    kind = kind if kind in KINDS else 'chat'
    width, height = int(width), int(height)
    if not 320 <= width <= 4096 or not 320 <= height <= 4096:
        raise ValueError('Canvas dimensions must be between 320 and 4096.')
    prompt = str(prompt)[:4000]
    rng = random.Random(int(seed) + int(variant) * 1009)
    s = Scene(width, height, prompt, style, rng)
    title = _title(prompt, kind)
    _DRAW[kind](s, title, prompt, _buttons(prompt))
    return {'svg': s.svg(title), 'kind': kind, 'components': s.count, 'title': title}


@lru_cache(maxsize=96)
def _font(size, bold=False):
    for path in [r'C:\Windows\Fonts\msyhbd.ttc' if bold else r'C:\Windows\Fonts\msyh.ttc',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        try:
            return ImageFont.truetype(path, size=max(1, int(size)))
        except OSError:
            pass
    return ImageFont.load_default()


def render_png(svg):
    """Rasterize only Cemivo's restricted SVG vocabulary, without external resources.

    Text remains real Unicode until rendered by the installed local CJK font.
    Native-resolution text antialiasing keeps exports crisp without doubling the
    full canvas, keeping PNG export latency low. Unknown elements are ignored.
    """
    if len(svg) > 2_000_000 or '<!DOCTYPE' in svg or '<!ENTITY' in svg:
        raise ValueError('Unsupported SVG document.')
    root = ET.fromstring(svg)
    width, height = int(float(root.get('width', 1280))), int(float(root.get('height', 800)))
    if not 1 <= width <= 4096 or not 1 <= height <= 4096:
        raise ValueError('Unsupported canvas size.')
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    def color(value):
        return None if value in ('none', None) else value
    def num(el, name, default=0):
        return float(el.get(name, default))
    for el in root:
        tag = el.tag.split('}')[-1]
        fill, stroke = color(el.get('fill')), color(el.get('stroke'))
        sw = max(1, int(round(num(el, 'stroke-width', 1))))
        if tag == 'rect':
            x, y, w, h = (num(el, n) for n in ('x', 'y', 'width', 'height'))
            radius = num(el, 'rx')
            if w > 0 and h > 0:
                draw.rounded_rectangle((x, y, x + w, y + h), radius=max(0, min(radius, w / 2, h / 2)), fill=fill, outline=stroke, width=sw)
        elif tag == 'circle':
            cx, cy, r = (num(el, n) for n in ('cx', 'cy', 'r'))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=stroke, width=sw)
        elif tag == 'line':
            x1, y1, x2, y2 = (num(el, n) for n in ('x1', 'y1', 'x2', 'y2'))
            if el.get('stroke-dasharray'):
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                if length:
                    for d in range(0, int(length), 9):
                        end = min(d + 4, length)
                        draw.line((x1 + dx * d / length, y1 + dy * d / length, x1 + dx * end / length, y1 + dy * end / length), fill=stroke, width=sw)
            else:
                draw.line((x1, y1, x2, y2), fill=stroke, width=sw)
        elif tag in ('polygon', 'polyline'):
            pts = [tuple(map(float, pair.split(','))) for pair in el.get('points', '').split()]
            if len(pts) >= 2:
                if tag == 'polygon':
                    if fill:
                        draw.polygon(pts, fill=fill)
                    draw.line(pts + [pts[0]], fill=stroke, width=sw, joint='curve')
                else:
                    draw.line(pts, fill=stroke, width=sw, joint='curve')
        elif tag == 'text':
            value = ''.join(el.itertext())
            x, y = num(el, 'x'), num(el, 'y')
            font = _font(round(num(el, 'font-size', 16)), el.get('font-weight') in ('700', 'bold'))
            anchor = {'start': 'ls', 'middle': 'ms', 'end': 'rs'}.get(el.get('text-anchor', 'start'), 'ls')
            draw.text((x, y), value, font=font, fill=fill or 'black', anchor=anchor)
    buf = io.BytesIO()
    image.save(buf, format='PNG', compress_level=2, optimize=False)
    return buf.getvalue()
