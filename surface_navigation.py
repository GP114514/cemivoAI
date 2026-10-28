"""Navigation between real neural surfaces, without screen templates or hotspots.

The planner names a semantic transition from the full screenshot. This module
validates it, keeps a bounded page stack and reuses the last successful neural
pixels for Home/Back. It never invokes business tools, infers button labels,
draws controls, or rolls back committed documents/tasks.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
from io import BytesIO
from pathlib import Path
import re
import secrets
import threading
import time

import numpy as np
from PIL import Image, UnidentifiedImageError


_BUSINESS_REQUEST = re.compile(
    r'workspace_list|document_(?:read|save|export)|task_(?:add|update)|'
    r'(?:请|先|帮我|我要|需要|然后|并|再)?(?:保存|写入|新增|添加|更新|删除|导出|读取)(?:文档|笔记|待办|任务|文件|工作区|内容|这|一|名|到|为|：|:)|'
    r'\b(?:save|export|delete|update|read)\s+(?:the\s+|a\s+|my\s+)?(?:document|note|task|file|workspace)|'
    r'\b(?:add|create)\s+(?:the\s+|a\s+|my\s+)?(?:document|note|task)\b', re.I)


def _error(message, code='invalid_navigation', status=422):
    # Lazy import avoids a live_app -> helper -> live_app import cycle.
    from live_app import LiveError
    return LiveError(message, code, status)


def validate_navigation(value, plan, event=None):
    """Return a strict semantic navigation record or None when absent.

    No coordinates or label-to-route table exist here. A typed real-business
    request must not disappear into a navigation-only shortcut. The AI can
    plan navigation after those tools, but that belongs to the business path.
    """
    if not isinstance(value, dict) or value.get('navigation') is None:
        return None
    navigation = value['navigation']
    if not isinstance(plan, dict) or not isinstance(navigation, dict):
        raise _error('导航计划需要有效的 JSON 对象。')
    if set(navigation) - {'action', 'target_app'}:
        raise _error('导航计划含有不支持的字段。')
    action = navigation.get('action')
    if action not in ('home', 'back', 'open_app'):
        raise _error('导航只支持桌面、返回或打开应用视图。')
    if (plan.get('requires_tools') is not False or plan.get('should_update') is not True or
            value.get('update_mode',plan.get('update_mode')) != 'global' or value.get('target_type',plan.get('target_type')) != 'other' or
            plan.get('local_edit') is not None):
        raise _error('导航必须是独立的整屏视觉操作，不能替代真实业务工具或局部开关。')
    if isinstance(event, dict) and isinstance(event.get('text'), str) and _BUSINESS_REQUEST.search(event['text']):
        raise _error('这条文字同时要求业务处理，请先由 Agent 执行真实工具，再更新视图。', 'navigation_requires_business', 422)
    target = navigation.get('target_app')
    if action == 'open_app':
        if (not isinstance(target, str) or not target.strip() or len(target.strip()) > 80 or
                any(ord(character) < 32 or ord(character) == 127 for character in target)):
            raise _error('打开应用需要简短、明确的应用名称。')
        try:
            target.encode('utf-8')
        except UnicodeError:
            raise _error('应用名称含有无效文字。') from None
        return {'action': action, 'target_app': target.strip()}
    if target not in (None, ''):
        raise _error('返回桌面或上一页不需要另一个应用目标。')
    return {'action': action}


def _pixels(png):
    if not isinstance(png, bytes) or not 0 < len(png) <= 8 * 1024**2:
        raise _error('神经界面图片无效。', 'surface_invalid')
    try:
        with Image.open(BytesIO(png)) as image:
            if (image.format != 'PNG' or image.mode != 'L' or image.size != (256, 256) or
                    getattr(image, 'n_frames', 1) != 1):
                raise _error('神经桌面需要原生 256×256 单帧灰度 PNG。', 'surface_invalid')
            image.load()
            return np.array(image, dtype=np.uint8, copy=True)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise _error('神经界面图片无法读取。', 'surface_invalid') from None


def _encode(pixels):
    output = BytesIO()
    Image.fromarray(pixels).save(output, format='PNG', compress_level=2)
    return output.getvalue()


class NavigationRuntime:
    """Small page cache owned by the local LiveApp; caller retains journaling.

    Per-session stacks hold pages, not every click/input frame. Each page's
    latest successful pixels/state are refreshed before leaving it. Cache
    restoration never calls an Agent tool or pretends to reverse its receipts.
    """
    def __init__(self, live):
        self.live = live
        self.editor = None
        self.lock = threading.RLock()
        self.sessions = OrderedDict()
        self.pending = OrderedDict()
        self.max_sessions, self.max_pages = 8, 16

    def status(self):
        try:
            result = self._editor().status()
            return result if isinstance(result, dict) else {'available':False,'last_error':'Invalid surface model status'}
        except (ImportError, ValueError, RuntimeError, OSError) as error:
            return {'available':False,'last_error':str(error),'scope':'neural desktop and chrome only'}

    def _editor(self):
        existing = getattr(self.live, 'surface_editor', None)
        if existing is not None:
            return existing
        if self.editor is None:
            try:
                from neural.surface_editor import SurfaceEditor
                self.editor = SurfaceEditor(Path(self.live.root))
            except (ImportError, OSError, ValueError, RuntimeError) as error:
                raise _error('神经桌面模型尚未准备完成。', 'surface_unavailable', 503) from error
        return self.editor

    def _render_surface(self, seed, chrome):
        editor = self._editor()
        status = editor.status()
        if not isinstance(status, dict) or status.get('available') is not True:
            raise _error('神经桌面模型尚未通过验证，当前画面已保留。', 'surface_unavailable', 503)
        try:
            png, metadata = editor.render(seed=seed, chrome=chrome)
        except (ValueError, OSError, RuntimeError) as error:
            raise _error('神经桌面生成失败，当前画面已保留。', 'surface_generation_failed', 503) from error
        _pixels(png)
        if not isinstance(metadata, dict):
            raise _error('神经桌面缺少模型来源记录。', 'surface_invalid', 503)
        source = deepcopy(status)
        source.update(deepcopy(metadata))
        step = source.get('checkpoint_step', source.get('step'))
        digest = source.get('model_sha256')
        if type(step) is not int or step < 0 or not isinstance(digest, str) or re.fullmatch(r'[a-f0-9]{64}', digest, re.I) is None:
            raise _error('神经桌面来源步数或权重哈希不完整。', 'surface_invalid', 503)
        source.update(step=step, model_sha256=digest)
        return png, source

    @staticmethod
    def _clean(frame):
        result = deepcopy(frame)
        for name in ('agent', 'edit', 'scope', 'recognition', 'last_event', 'initial_plan', 'initial_review'):
            result.pop(name, None)
        # A cache restoration is visual history, not re-execution of an old
        # plan. New request intent/feedback is attached by the caller.
        result.pop('plan', None)
        return result

    @staticmethod
    def _image_identity(frame, png):
        frame = deepcopy(frame)
        frame.update(image_url='/api/live/images/' + secrets.token_hex(16) + '.png',
            sha256=hashlib.sha256(png).hexdigest(), width=256, height=256)
        return frame

    def draw_desktop(self, seed=42, steps=8):
        started = time.perf_counter()
        png, source = self._render_surface(seed, chrome=False)
        frame = self._image_identity({
            'prompt': 'Neural monochrome desktop with generated app controls',
            'seed': seed, 'steps': steps, 'checkpoint_step': source['step'],
            'surface_sampling_steps':source.get('steps',1),
            'model_sha256': source['model_sha256'], 'generator': 'native_neural_surface',
            'surface_model': source, 'experimental': True,'page_review':None,
            'application': {'id':secrets.token_hex(12),'name':'cemivoAI 桌面',
                'goal':'模型生成桌面与导航控件；依据完整实际截图打开对应应用视图、返回上一页或桌面。真实业务仅以 Agent 工具回执为准。'},
            'state': {'view': 'desktop', 'messages': [], 'notice': '桌面与按钮由神经模型生成；点击后结合完整画面判断。'},
            'surface_navigation': {'page_id': 'desktop', 'page_kind': 'desktop',
                'app_name': '', 'at_home': True, 'can_back': False,
                'action': 'home', 'source': 'neural_surface', 'restored': False},
            'timings': {'sample_ms': source.get('sample_ms', source.get('timings',{}).get('sample_ms',0)),
                'load_ms': source.get('load_ms', source.get('timings',{}).get('load_ms',0)),
                'total_ms': round((time.perf_counter()-started)*1000, 2)}}, png)
        return png, frame

    def add_chrome(self, png, frame):
        """Replace only top 32 rows with native neural chrome; no raster drawing."""
        original = _pixels(png)
        generated, source = self._render_surface(frame.get('seed', 42), chrome=True)
        chrome = _pixels(generated)
        output = original.copy()
        output[:32] = chrome[:32]
        delivered = _encode(output)
        actual = _pixels(delivered)
        if not np.array_equal(actual[32:], original[32:]):
            raise _error('导航条区域外像素核对失败。', 'surface_invalid')
        result = self._image_identity(frame, delivered)
        result.update(surface_model=source,
            surface_chrome={'bounds': [0,0,256,32], 'generator': 'native_neural_surface',
                'main_png_sha256': hashlib.sha256(png).hexdigest(),
                'outside_changed_pixels': 0,
                'outside_sha256_before': hashlib.sha256(original[32:].tobytes()).hexdigest(),
                'outside_sha256_after': hashlib.sha256(actual[32:].tobytes()).hexdigest()})
        return delivered, result

    def _remember(self, tracking, frame, png):
        _pixels(png)
        nav = frame.get('surface_navigation', {})
        page_id = nav.get('page_id') or tracking['stack'][-1]
        if page_id not in tracking['stack']:
            page_id = tracking['stack'][-1]
        stored = self._clean(frame)
        stored['surface_navigation'] = dict(nav, page_id=page_id,
            page_kind='desktop' if page_id == 'desktop' else 'app')
        tracking['pages'][page_id] = {'frame': stored, 'png': bytes(png)}
        tracking['pages'].move_to_end(page_id)
        while len(tracking['pages']) > self.max_pages:
            removable = next((key for key in tracking['pages'] if key not in tracking['stack'] and key != 'desktop'), None)
            if removable is None:
                break
            tracking['pages'].pop(removable)

    def ensure_tracking(self, previous):
        with self.lock:
            ident = previous.get('session_id')
            if not isinstance(ident, str) or not ident:
                raise _error('导航缺少当前会话。')
            if ident not in self.sessions:
                nav = previous.get('surface_navigation', {})
                page_id = 'desktop' if nav.get('page_kind') == 'desktop' else nav.get('page_id')
                if not isinstance(page_id, str) or not page_id or page_id == 'desktop' and nav.get('page_kind') != 'desktop':
                    page_id = 'page-' + secrets.token_hex(8)
                self.sessions[ident] = {'stack': [page_id], 'pages': OrderedDict()}
            self.sessions.move_to_end(ident)
            while len(self.sessions) > self.max_sessions:
                self.sessions.popitem(last=False)

    def observe(self, frame, png):
        """Confirm a caller-committed frame and refresh that page's last state.

        Call only after LiveApp stores its PNG/current frame successfully. A
        rendered navigation candidate alone never advances the page stack.
        """
        _pixels(png)
        with self.lock:
            self.ensure_tracking(frame)
            ident = frame['session_id']
            staged = self.pending.get(frame.get('image_url'))
            if staged is not None and staged['session_id'] == ident:
                tracking = deepcopy(staged['tracking'])
                self.pending.pop(frame['image_url'],None)
            else:
                tracking = deepcopy(self.sessions[ident])
                nav = frame.get('surface_navigation',{})
                page_id = nav.get('page_id')
                if isinstance(page_id,str) and page_id and page_id != tracking['stack'][-1]:
                    if page_id in tracking['stack']:
                        tracking['stack'] = tracking['stack'][:tracking['stack'].index(page_id)+1]
                    elif nav.get('page_kind') == 'desktop':
                        tracking['stack'] = ['desktop']
                    else:
                        tracking['stack'].append(page_id)
            self._remember(tracking,frame,png)
            self.sessions[ident] = tracking
            self.sessions.move_to_end(ident)

    def render(self, previous, png, plan, event, historyframes=None):
        """Return generated/restored pixels + complete metadata, without tools.

        historyframes is accepted for integration convenience but deliberately
        not used as a navigation stack: it also contains inputs/toggle edits.
        """
        navigation = validate_navigation(plan, plan, event)
        if navigation is None:
            raise _error('本次计划未声明有效导航。')
        with self.lock:
            self.ensure_tracking(previous)
            tracking = deepcopy(self.sessions[previous['session_id']])
            self._remember(tracking, previous, png)
            old_stack = list(tracking['stack'])
            action = navigation['action']
            restored = False
            if action == 'back':
                if len(old_stack) < 2:
                    raise _error('当前视图没有上一层页面，可返回桌面。', 'no_previous_page', 409)
                stack = old_stack[:-1]
                target = tracking['pages'].get(stack[-1])
                if target is None:
                    raise _error('上一页的神经画面已过期，请返回桌面。', 'page_expired', 409)
                output, frame = target['png'], self._clean(target['frame'])
                restored = True
            elif action == 'home':
                target = tracking['pages'].get('desktop')
                if target is None:
                    output, frame = self.draw_desktop(previous.get('seed',42), previous.get('steps',8))
                else:
                    output, frame = target['png'], self._clean(target['frame'])
                    restored = True
                stack = ['desktop']
            else:
                name = navigation['target_app']
                target_key = next((key for key, value in tracking['pages'].items()
                    if key != 'desktop' and value['frame'].get('surface_navigation',{}).get('app_name') == name), None)
                if target_key is not None:
                    target = tracking['pages'][target_key]
                    output, frame = target['png'], self._clean(target['frame'])
                    restored = True
                    stack = old_stack if old_stack[-1] == target_key else old_stack + [target_key]
                else:
                    output, frame = self.live._draw_page(plan['next_prompt'], previous.get('seed',42), previous.get('steps',8),
                        f'打开{name}应用。{plan.get("intent",name)}。页面要求：{plan["next_prompt"]}')
                    output, frame = self.add_chrome(output, frame)
                    target_key = 'page-' + secrets.token_hex(8)
                    frame['surface_navigation'] = {'page_id':target_key,'page_kind':'app','app_name':name}
                    frame['application'] = {'id':secrets.token_hex(12),'name':name,
                        'goal':f'打开并使用{name}应用视图。当前 AI 语义：{plan.get("intent",name)}。当前页面绘图描述：{plan["next_prompt"]}',
                        'parent_application_id':previous.get('application',{}).get('id')}
                    stack = old_stack + [target_key]
            # Stage navigation only after every model/decode check succeeded.
            # The caller's later observe confirms its actual frame commit.
            if len(stack) > self.max_pages:
                stack = stack[-self.max_pages:]
            tracking['stack'] = stack
            frame = self._image_identity(frame, output)
            nav = frame.get('surface_navigation', {})
            kind = 'desktop' if stack[-1] == 'desktop' else 'app'
            frame['surface_navigation'] = dict(nav, page_id=stack[-1], page_kind=kind,
                app_name='' if kind=='desktop' else nav.get('app_name', navigation.get('target_app','')),
                at_home=kind=='desktop', can_back=len(stack)>1, action=action,
                source='cached_neural_frame' if restored else 'neural_generation', restored=restored,
                history_does_not_undo_business=True)
            frame['state'] = deepcopy(frame.get('state', {'view':kind,'messages':[]}))
            frame['state']['view'] = kind
            frame['state']['notice'] = ('已返回模型生成的历史视图；没有撤销任何业务操作。' if restored else
                '已生成新的应用视图；真实业务是否完成仍以工具回执为准。')
            if restored:
                frame['timings'] = {'sample_ms':0,'load_ms':0,'cached':True,'total_ms':0}
            self._remember(tracking, frame, output)
            self.pending[frame['image_url']] = {'session_id':previous['session_id'],'tracking':tracking}
            while len(self.pending) > 32:
                self.pending.popitem(last=False)
            return output, frame
