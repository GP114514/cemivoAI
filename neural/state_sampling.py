"""Training-only state sampling and truthful compact captions.

These helpers never draw pixels or participate in model inference. Short aliases
retain the visible target text and the explicit interaction state; they do not
use the general family caption augmenter, which can omit state conditions.
"""
from __future__ import annotations

import math

CAPTION_POLICY = "metadata-state-versioned-zh-en-v2"
STATE_MARKER = "verbatim-interaction-state"
CURRICULUM_VERSION = "native256-interaction-curriculum-v2"
# These mixtures retain the original explicitly marked state pairs. Adding an
# ordinary app family does not reclassify its rows as interaction-state pairs.
APP_CURRICULUM_VERSION = "v3-plus-native256-app-curriculum-v4"
APP_FLOW_VERSION = 'native256-app-flows-v5'
APP_FLOW_COMBINED_VERSION = 'v4-plus-native256-app-flows-v5'
APP_FLOW_MARKER = 'verbatim-app-flow-state-v5'
INTERACTION_DATASET_VERSIONS = frozenset((
    "v2-plus-native256-interaction-v1",
    "v2-plus-native256-curriculum-v2",
    APP_CURRICULUM_VERSION,
    APP_FLOW_COMBINED_VERSION,
))


def resolve_app_flow_fraction(value, checkpoint=None, *, uniform_sampling=False):
    """V5 quota is opt-in; an enabled checkpoint retains its saved quota."""
    if type(uniform_sampling) is not bool:
        raise ValueError('uniform-sampling must be boolean')
    if uniform_sampling:
        if value is not None:
            raise ValueError('uniform-sampling cannot be combined with app-flow-fraction')
        return None
    if value is None and checkpoint is not None:
        value = checkpoint.get('training_args', {}).get('app_flow_fraction')
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            raise ValueError
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('app-flow-fraction must be a finite probability in 0..1') from None
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError('app-flow-fraction must be a finite probability in 0..1')
    return result


def validate_sampling_fractions(interaction_fraction, app_flow_fraction):
    """Independent legacy-state and V5 probabilities leave an ordinary remainder."""
    if app_flow_fraction is not None and (interaction_fraction or 0.0) + app_flow_fraction > 1:
        raise ValueError('interaction-fraction plus app-flow-fraction must be <=1')


def app_flow_caption_variants(metadata):
    """Keep exact bilingual V5 literals and every required empty/state condition."""
    if (not isinstance(metadata, dict) or metadata.get('dataset_version') != APP_FLOW_VERSION
            or metadata.get('caption_policy') != APP_FLOW_MARKER):
        raise ValueError('App flow captions require audited V5 state metadata')
    family, stage = metadata.get('flow_family'), metadata.get('flow_stage')
    conditions = {
        ('tasks', 'empty'): (('空任务列表', '空输入框'), ('empty list', 'empty input')),
        ('tasks', 'draft'): (('空列表', '输入“'), ('empty list', 'input "')),
        ('tasks', 'added'): (('未勾选', '空输入框'), ('one unchecked task', 'empty input')),
        ('tasks', 'completed'): (('已勾选', '空输入框'), ('one checked task', 'empty input')),
        ('tasks', 'undone'): (('未勾选', '空输入框'), ('one unchecked task', 'empty input')),
        ('notes', 'new'): (('笔记列表和正文为空',), ('empty note list and body',)),
        ('notes', 'typed'): (('未保存正文',), ('unsaved body',)),
        ('notes', 'saved'): (('已保存笔记',), ('saved note',)),
        ('notes', 'listselected'): (('已选择笔记',), ('selected note',)),
        ('notes', 'newagain'): (('一条已存笔记', '空白正文'), ('one saved note', 'blank body')),
    }
    if (family, stage) not in conditions:
        raise ValueError('Unsupported app flow family/state')
    if metadata.get('kind') != ('tasks' if family == 'tasks' else 'editor'):
        raise ValueError('App flow kind does not match its family')
    captions = metadata.get('short_captions')
    if not isinstance(captions, dict) or set(captions) != {'zh', 'en'}:
        raise ValueError('App flow state needs both exact short captions')
    variants = tuple(_text(captions[key]) for key in ('zh', 'en'))
    if any(len(value.encode('utf-8')) > 120 for value in variants):
        raise ValueError('App flow captions must fit 120 bytes without truncation')
    for caption, required in zip(variants, conditions[(family, stage)]):
        if any(term not in caption for term in required):
            raise ValueError('App flow caption lost an explicit empty/state condition')
    drawn = metadata.get('drawn_text')
    if not isinstance(drawn, list) or not drawn:
        raise ValueError('App flow needs actual visible labels')
    labels = []
    for label in drawn:
        if not isinstance(label, dict) or label.get('requested') != label.get('visible'):
            raise ValueError('App flow target label was truncated')
        labels.append(_text(label.get('visible')))
    title = _text(metadata.get('title'))
    literal = metadata.get('required_literal')
    required_literal = (family == 'tasks' and stage != 'empty') or (family == 'notes' and stage in ('typed', 'saved', 'listselected', 'newagain'))
    if required_literal != bool(literal):
        raise ValueError('App flow required literal does not match its state')
    if literal and (literal not in labels or any(literal not in caption for caption in variants)):
        raise ValueError('App flow caption lost exact visible literal')
    if title not in labels or any(title not in caption for caption in variants):
        raise ValueError('App flow caption lost exact visible title')
    state = metadata.get('state')
    if not isinstance(state, dict):
        raise ValueError('App flow requires structured state')
    if family == 'tasks':
        has_task = stage in ('added', 'completed', 'undone')
        tasks = state.get('tasks')
        if (not isinstance(tasks, list) or len(tasks) != int(has_task)
                or state.get('task_count') != int(has_task)
                or state.get('task_list_empty') is not (not has_task)
                or state.get('input_empty') is not (stage != 'draft')
                or bool(state.get('draft_text')) != (stage == 'draft')
                or (stage == 'draft' and state.get('draft_text') != literal)
                or (has_task and (tasks[0].get('text') != literal
                    or tasks[0].get('checked') is not (stage == 'completed')))):
            raise ValueError('App flow task caption conflicts with structured state')
    else:
        has_body = stage in ('typed', 'saved', 'listselected')
        has_note = stage in ('saved', 'listselected', 'newagain')
        if (state.get('document_body_empty') is not (not has_body)
                or state.get('note_list_empty') is not (not has_note)
                or state.get('note_count') != int(has_note)
                or bool(state.get('document_text')) != has_body
                or (has_body and state.get('document_text') != literal)
                or (stage == 'newagain' and (not isinstance(state.get('saved_notes'),list)
                    or len(state['saved_notes'])!=1 or state['saved_notes'][0].get('title')!=literal))
                or bool(state.get('selected_note')) != (stage == 'listselected')):
            raise ValueError('App flow note caption conflicts with structured state')
    return variants


def resolve_interaction_fraction(value, checkpoint=None, *, uniform_sampling=False):
    """Explicit value wins; resume inherits; None keeps historical sampling."""
    if type(uniform_sampling) is not bool:
        raise ValueError("uniform-sampling must be boolean")
    if uniform_sampling:
        if value is not None:
            raise ValueError("uniform-sampling cannot be combined with interaction-fraction")
        return None
    if value is None and checkpoint is not None:
        value=checkpoint.get("training_args",{}).get("interaction_fraction")
    if value is None:
        return None
    try:
        if isinstance(value,bool):
            raise ValueError
        result=float(value)
    except (ValueError,TypeError,OverflowError):
        raise ValueError("interaction-fraction must be a finite probability in 0..1") from None
    if not math.isfinite(result) or not 0<=result<=1:
        raise ValueError("interaction-fraction must be a finite probability in 0..1")
    return result


def _text(value):
    if not isinstance(value,str) or not value or any(ord(ch)<32 or ord(ch)==127 for ch in value):
        raise ValueError("State caption requires complete printable metadata text")
    value.encode("utf-8")
    return value


def state_caption_variants(metadata):
    """Return short (Chinese, English) aliases from actual state and labels.

    Unknown/malformed metadata is rejected instead of dropping state or slicing
    text. This first policy intentionally covers the existing state-pair corpus.
    """
    if not isinstance(metadata,dict) or metadata.get("caption_policy")!=STATE_MARKER:
        raise ValueError("State caption requires explicitly marked state metadata")
    if metadata.get("dataset_version")==CURRICULUM_VERSION:
        return _curriculum_caption_variants(metadata)
    if 'short_captions' in metadata:
        raise ValueError('Natural captions require the audited curriculum version')
    labels=[]
    drawn=metadata.get("drawn_text")
    if not isinstance(drawn,list) or not drawn:
        raise ValueError("State caption requires actual complete drawn labels")
    for item in drawn:
        if not isinstance(item,dict) or item.get("requested")!=item.get("visible"):
            raise ValueError("State caption cannot describe truncated or unknown labels")
        label=_text(item.get("visible"))
        if label not in labels:
            labels.append(label)
    title=_text(metadata.get("title"))
    if title not in labels:
        raise ValueError("State title must be visible in the training target")
    state=metadata.get("state")
    if not isinstance(state,dict):
        raise ValueError("State caption requires structured state")
    style="dark" if metadata.get("dark") is True else "light"
    kind=metadata.get("kind")
    if kind=="chat":
        if set(state)!={"messages","composer"} or not isinstance(state["messages"],list) or len(state["messages"])>1:
            raise ValueError("Unsupported chat state metadata")
        messages=[_text(item) for item in state["messages"]]
        composer=state["composer"]
        if not isinstance(composer,str):
            raise ValueError("Invalid composer state")
        if composer:
            _text(composer)
        if any(item not in labels for item in messages) or (composer and composer not in labels):
            raise ValueError("State content must be visible in the training target")
        en=f"Chat {style};messages={messages[0] if messages else 'empty'};input={composer or 'empty'};text="
        zh=f"聊天{style};消息={messages[0] if messages else '空'};输入={composer or '空'};字="
    elif kind=="settings":
        if set(state)!={"Notifications","Sound","Privacy"} or any(value not in ("on","off") for value in state.values()):
            raise ValueError("Unsupported switch state metadata")
        switches=";".join(f"{label}={state[label]}" for label in ("Notifications","Sound","Privacy"))
        if any(label not in labels for label in state):
            raise ValueError("Switch labels must be visible in the training target")
        en=f"Settings {style};{switches};text="
        zh=f"设置{style};{switches};字="
        # Each visible switch label is already complete in its state binding.
        labels=[label for label in labels if label not in state]
    else:
        raise ValueError("Unsupported state training family")
    variants=tuple(prefix+"|".join(labels) for prefix in (zh,en))
    if any(len(value.encode("utf-8"))>120 for value in variants):
        raise ValueError("Truthful state caption exceeds 120 bytes; metadata must be simplified before training")
    return variants


def _curriculum_caption_variants(metadata):
    """Validate already-audited natural captions without rewriting target text."""
    captions=metadata.get('short_captions')
    if not isinstance(captions,dict) or set(captions)!={'zh','en'}:
        raise ValueError('Natural state captions need exactly Chinese and English variants')
    variants=tuple(_text(captions[key]) for key in ('zh','en'))
    if any(len(caption.encode('utf-8'))>120 for caption in variants):
        raise ValueError('Natural state captions must fit 120 UTF-8 bytes without truncation')
    drawn=metadata.get('drawn_text')
    if not isinstance(drawn,list) or not drawn:
        raise ValueError('Natural state captions require actual drawn text evidence')
    labels=[]
    for item in drawn:
        if (not isinstance(item,dict) or item.get('requested')!=item.get('visible') or
                type(item.get('font_px')) is not int or item['font_px']<12):
            raise ValueError('Natural state captions require complete native text >=12px')
        labels.append(_text(item.get('visible')))
        bounds=item.get('bbox')
        if (not isinstance(bounds,(list,tuple)) or len(bounds)!=4 or
                any(type(value) is not int for value in bounds) or
                not 0<=bounds[0]<bounds[2]<=256 or not 0<=bounds[1]<bounds[3]<=256):
            raise ValueError('Text evidence requires valid native exclusive pixel bounds')
    title=_text(metadata.get('title'))
    if title not in labels or any(label not in caption for caption in variants for label in labels):
        raise ValueError('Natural captions must preserve every visible title and label')
    mapping=metadata.get('state_label_map')
    if not isinstance(mapping,dict):raise ValueError('Natural captions need visible state label mapping')
    state=metadata.get('state');kind=metadata.get('kind');zh,en=variants
    if not isinstance(state,dict):raise ValueError('Natural captions require structured state')
    if kind=='chat':
        if (set(state)!={'messages','composer'} or not isinstance(state['messages'],list) or
                len(state['messages'])>1 or not isinstance(state['composer'],str)):
            raise ValueError('Unsupported natural chat state')
        message=state['messages'][0] if state['messages'] else None
        if message is not None:
            _text(message)
            if message not in labels or '消息'+message not in zh or 'message '+message not in en:
                raise ValueError('Natural message caption must match the actual message text')
        elif '空消息区' not in zh or 'empty messages' not in en:
            raise ValueError('Natural empty message state must remain explicit')
        composer=state['composer']
        if composer:
            _text(composer)
            if composer not in labels or '输入'+composer not in zh or 'input '+composer not in en:
                raise ValueError('Natural composer caption must match the actual input text')
        elif '空输入框' not in zh or 'empty input' not in en:
            raise ValueError('Natural empty composer state must remain explicit')
        required=('new','attach','send')
    elif kind=='settings':
        if set(state)!={'Notifications','Sound','Privacy'} or any(value not in ('on','off') for value in state.values()):
            raise ValueError('Unsupported natural settings state')
        for key,value in state.items():
            label=_text(mapping.get(key))
            if label not in labels or label+('开' if value=='on' else '关') not in zh or label+' '+value not in en:
                raise ValueError('Natural switch caption must retain every exact visible label and state')
        if metadata.get('production_switch_gate_passed') is not True:
            raise ValueError('Natural switch targets require production pixel gate evidence')
        required=('back','save')
    else:
        raise ValueError('Unsupported natural state family')
    expected_buttons=[_text(mapping.get(key)) for key in required]
    if metadata.get('buttons')!=expected_buttons or any(label not in labels for label in expected_buttons):
        raise ValueError('Natural state captions must use the actual visible control labels')
    return variants
