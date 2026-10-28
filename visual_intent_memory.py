"""Conservative reuse of an AI-observed text policy; never draws UI pixels.

Only the server calls observe after committing a successful neural text edit.
Returned plans still pass the production validator. The opaque proof authorizes
only that exact compiled plan, source frame and event in this process.
"""
from copy import deepcopy
import hashlib
import hmac
from io import BytesIO
import json
from pathlib import Path
import re
import secrets

import numpy as np
from PIL import Image


PROOF = '_visual_intent_proof'
POLICY = {'full_image_hash_check': True, 'policy_source': 'AI-origin-policy',
          'pixel_generator': 'neural_text', 'fresh_vision': False}
_COMMAND = re.compile(
    r'打开|返回|主页|首页|设置|保存|导出|删除|关闭|切换|执行|运行|下载|上传|创建|生成|查询|搜索|'
    r'帮我|请帮|任务|文档|文件|发送|输入|填写|改成|替换|'
    r'\b(open|back|home|settings?|save|export|delete|close|switch|run|execute|download|upload|'
    r'create|generate|search|find|send|type|input|enter|write|replace|please|task|file|document)\b|'
    r'^[\s]*[/\\!$`]|https?://|[A-Za-z]:[\\/]', re.I)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _contains(a, b):
    return a[0] <= b[0] < b[2] <= a[2] and a[1] <= b[1] < b[3] <= a[3]


class Memory:
    def __init__(self):
        self.__secret = secrets.token_bytes(32)
        self.__pages = {}
        self.__vocab = None

    @staticmethod
    def _key(frame):
        app, state = frame.get('application', {}), frame.get('state', {})
        if not frame.get('session_id') or not app.get('id') or not isinstance(state, dict):
            raise ValueError('missing private page identity')
        return (frame['session_id'], app['id'], _json({k: state.get(k) for k in
                ('view', 'page', 'page_id', 'route')}))

    @staticmethod
    def _box(box, frame):
        return (isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)
                and 0 <= box[0] < box[2] <= frame['width'] and 0 <= box[1] < box[3] <= frame['height'])

    @staticmethod
    def _source(frame):
        return {k: deepcopy(frame.get(k)) for k in
                ('sha256', 'state', 'session_id', 'application', 'width', 'height', 'version')}

    def _record(self, png, frame):
        record = self.__pages.get(self._key(frame))
        if (not record or hashlib.sha256(png).hexdigest() != frame.get('sha256')
                or record['source'] != self._source(frame) or frame.get('edit', {}).get('preview') is True):
            return None
        return record

    def _signature(self, plan, source_sha, frame, event):
        body = {k: v for k, v in plan.items() if k != PROOF}
        data = _json([source_sha, self._source(frame), event, body]).encode('utf-8')
        return hmac.new(self.__secret, data, hashlib.sha256).hexdigest()

    def authorize(self, plan, png, previous, event):
        """True only for this process's unchanged, current compiled text plan."""
        try:
            proof = plan.get(PROOF)
            return (isinstance(proof, str) and self._record(png, previous) is not None
                    and plan.get('visual_intent_memory') == POLICY
                    and hmac.compare_digest(proof, self._signature(plan, hashlib.sha256(png).hexdigest(), previous, event)))
        except (ValueError, TypeError, KeyError, UnicodeError):
            return False

    def _literal(self, text, model_sha):
        if (not isinstance(text, str) or not text.strip() or len(text.encode('utf-8')) > 60
                or any(ord(c) < 32 or ord(c) == 127 for c in text) or _COMMAND.search(text)):
            return False
        # Read only the trained vocabulary, never fonts/targets or generated UI.
        try:
            path = Path(__file__).resolve().parent / 'neural/text-head-v1/inference.pt'
            stat = path.stat(); signature = (stat.st_mtime_ns, stat.st_size)
            if self.__vocab is None or self.__vocab[0] != signature:
                import torch
                data = path.read_bytes()
                saved = torch.load(BytesIO(data), map_location='cpu', weights_only=True)
                self.__vocab = (signature, hashlib.sha256(data).hexdigest(), frozenset(saved['vocabulary']))
        except Exception:
            # Missing/corrupt/publishing checkpoints cannot authorize a cache hit.
            self.__vocab = None
            return False
        return self.__vocab[1] == model_sha and all(c in self.__vocab[2] for c in text)

    @staticmethod
    def _rows(text, width):
        rows, advance = 1, 0
        for c in text:
            step = 9 if ord(c) < 128 else 16
            if step > width:
                raise ValueError('unknown or too narrow text area')
            if advance + step > width:
                rows += 1; advance = 0
            advance += step
        return rows

    def plan(self, planner, png, previous, event):
        """Compile a small literal input/send or return None for fresh AI work."""
        try:
            record = self._record(png, previous)
            if not record or event.get('type') != 'click':
                return None
            state, composer = record['source']['state'], record['composer']
            if any(_overlap(composer, old) for old in state.get('sent_message_bounds', [])):
                return None
            def clicked(box):
                return (box and type(event.get('x')) in (int, float) and type(event.get('y')) in (int, float)
                        and box[0] <= event['x'] < box[2] and box[1] <= event['y'] < box[3])
            if event.get('input_method') == 'contextmenu' and event.get('text') and clicked(record.get('input_trigger') or composer):
                action, text, trigger, box = 'input', event['text'], record.get('input_trigger') or composer, composer.copy()
                if not self._literal(text, record['text_model']) or self._rows(text, box[2]-box[0])*18 > box[3]-box[1]:
                    return None
                regions = [box.copy()]
                if state.get('input_bounds') and not _contains(box, state['input_bounds']):
                    return None
            elif (event.get('input_method') == 'pointer' and not event.get('text')
                    and clicked(record.get('send_trigger')) and record.get('message_area')
                    and state.get('input_text') and state.get('sent_messages') and state.get('sent_message_bounds')):
                action, text, trigger = 'send', state['input_text'], record['send_trigger']
                area, last, draft = record['message_area'], state['sent_message_bounds'][-1], state.get('input_bounds')
                if not self._literal(text, record['text_model']) or not self._box(draft, previous) or not _contains(composer, draft):
                    return None
                box = [last[0], last[3]+2, area[2], last[3]+2+self._rows(text, area[2]-last[0])*18]
                if (not _contains(area, box) or _overlap(box, composer)
                        or any(_overlap(box, old) for old in state['sent_message_bounds'])):
                    return None
                with Image.open(BytesIO(png)) as image:
                    if image.format != 'PNG' or image.mode != 'L' or image.size != (previous['width'], previous['height']):
                        return None
                    pixels = np.asarray(image)
                    if not bool(np.all(pixels[box[1]:box[3], box[0]:box[2]] >= 248)):
                        return None
                regions = [box.copy(), draft.copy()]
            else:
                return None
            value = {'intent': 'Display literal draft' if action == 'input' else 'Display the existing draft as a message',
                'expected_change': 'Only the selected neural text areas change', 'state_summary': 'Confirmed page text interaction',
                'should_update': True, 'next_prompt': ('Chat draft: ' if action == 'input' else 'Chat message: ')+text,
                'update_mode': 'local', 'target_type': 'other', 'scope_reason': 'Reuse the AI-observed text policy on the identical complete frame',
                'requires_tools': False, 'text_action': action, 'text_source': 'event' if action == 'input' else 'draft',
                'input_text': text, 'text_bounds': box, 'local_edit': {'kind': 'regions', 'trigger_bounds': trigger.copy(), 'regions': regions}}
            if action == 'send':
                value['clear_input_bounds'] = draft.copy()
            if len(value['next_prompt'].encode('utf-8')) > 120:
                return None
            try:
                result = planner.validate_auto_plan(value, previous, event)
            except Exception:
                # A production geometry/policy rejection must use fresh AI.
                return None
            result['visual_intent_memory'] = deepcopy(POLICY)
            result.update(provider='Local confirmed AI policy',model='neural_text_head',
                intent='输入文字' if action=='input' else '发送当前草稿',
                expected_change='已更新文字，其他区域保持不变。',
                recognition={'status':'recognized','source':'confirmed_full_frame_hash','attempts':0,
                    'full_image_used':False,'full_image_hash_checked':True,'focus_used':False,
                    'message':'复用本页已确认的操作，已核对完整画面与当前状态。'})
            result[PROOF] = self._signature(result, previous['sha256'], previous, event)
            return result
        except (ValueError, TypeError, KeyError, OSError, UnicodeError, IndexError):
            return None

    def observe(self, previous, frame, png, plan, event):
        """Learn/update only a committed, successful local neural text frame.

        message_area_bounds must originate in a full-image AI send plan. It is
        layout context only, never an instruction to erase that rectangle.
        """
        try:
            key = self._key(frame)
            prior = self.__pages.get(self._key(previous))
            if prior and prior['source'] != self._source(previous):
                prior = None
            edit = frame.get('edit', {})
            action = plan.get('text_action')
            valid = (action in ('input', 'send') and plan.get('update_mode') == 'local'
                and plan.get('target_type') == 'other' and plan.get('requires_tools') is False
                and not plan.get('navigation') and edit.get('text_applied') is True
                and edit.get('preview') is False and type(edit.get('outside_changed_pixels')) is int
                and edit['outside_changed_pixels'] == 0 and edit.get('generator') == 'native_neural_text_head'
                and hashlib.sha256(png).hexdigest() == frame.get('sha256'))
            internal = PROOF in plan
            if internal:
                # Its signed SHA was checked against actual bytes by plan().
                valid = valid and prior is not None and prior['source'] == self._source(previous)
                valid = (valid and plan.get('visual_intent_memory') == POLICY and isinstance(plan.get(PROOF), str)
                         and hmac.compare_digest(plan[PROOF], self._signature(plan, previous['sha256'], previous, event)))
            else:
                valid = valid and plan.get('recognition', {}).get('full_image_used') is True
            state = frame.get('state', {})
            composer = state.get('composer_bounds')
            if not valid or not self._box(composer, frame):
                raise ValueError('unconfirmed text edit')
            record = deepcopy(prior) if prior and self._key(previous) == key else {}
            record['composer'] = composer.copy()
            record['text_model'] = edit.get('editor_model', {}).get('model_sha256')
            layout = plan.get('local_edit', {})
            trigger = layout.get('trigger_bounds') or layout.get('bounds')
            if not self._box(trigger, frame):
                raise ValueError('missing AI trigger')
            if action == 'input':
                if not _contains(composer, edit.get('text_bounds', [])) or state.get('input_text') != plan.get('input_text'):
                    raise ValueError('uncommitted draft')
                if not internal:
                    record['input_trigger'] = trigger.copy()
            else:
                if (state.get('input_text') or not state.get('sent_messages') or state['sent_messages'][-1] != plan.get('input_text')
                        or not state.get('sent_message_bounds') or state['sent_message_bounds'][-1] != edit.get('text_bounds')):
                    raise ValueError('uncommitted send')
                if not internal:
                    record['send_trigger'] = trigger.copy()
                    area = plan.get('message_area_bounds')
                    if area is not None:
                        if (not self._box(area, frame) or (area[2]-area[0])*(area[3]-area[1])*2 > frame['width']*frame['height']
                                or not _contains(area, edit['text_bounds']) or _overlap(area, composer)
                                or any(not self._box(old, frame) or not _contains(area, old) for old in state['sent_message_bounds'])
                                or (previous.get('state', {}).get('input_bounds') and _overlap(area, previous['state']['input_bounds']))):
                            raise ValueError('invalid AI message container')
                        record['message_area'] = area.copy()
                if not record.get('message_area'):
                    record.pop('send_trigger', None)
            # Once learned, every actual effect stays inside the two AI areas.
            if prior and prior.get('message_area'):
                effects = edit.get('regions') or [edit.get('bounds')]
                if any(not self._box(box, frame) or not any(_contains(outer, box) for outer in
                       (record['composer'], record['message_area'])) for box in effects):
                    raise ValueError('edit escaped the learned page policy')
            record['source'] = self._source(frame)
            self.__pages[key] = record
            while len(self.__pages) > 8:
                self.__pages.pop(next(iter(self.__pages)))
            return True
        except (ValueError, TypeError, KeyError, IndexError, UnicodeError):
            session = frame.get('session_id') if isinstance(frame, dict) else None
            self.__pages = {key: value for key, value in self.__pages.items() if key[0] != session}
            return False
