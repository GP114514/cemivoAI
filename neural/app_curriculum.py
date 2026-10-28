"""Training-only, native app-family targets and truthful bilingual captions.

This module is a dataset constructor, never a runtime renderer. Fonts are used
only by PixelCanvas to construct supervised training pixels. New metadata needs
the explicit caption dispatcher below before enabling generic augmentation.
"""
from __future__ import annotations
import hashlib
import random
import numpy as np
from .dataset import PixelCanvas

VERSION = 'native256-app-curriculum-v4'
KINDS = ('tasks', 'desktop', 'editor')
ALIASES = {
    'tasks': {'zh': ('待办列表', '任务清单', '任务管理界面'),
        'en': ('task list', 'task manager', 'todo app')},
    'desktop': {'zh': ('桌面', '应用桌面', '应用启动页面'),
        'en': ('desktop', 'application desktop', 'app launcher')},
    'editor': {'zh': ('空白文档编辑器', '空白笔记页面', '新建文档界面'),
        'en': ('empty document editor', 'blank notes page', 'new document editor')},
}
DETAILS = {
    'tasks': {'zh': ('方形复选框和任务条目', '任务清单和添加按钮'),
        'en': ('square checkboxes and task rows', 'a task list and an Add button')},
    'desktop': {'zh': ('四个应用入口按钮', '聊天、笔记、待办、设置入口'),
        'en': ('four app launch buttons', 'chat, notes, tasks and settings entries')},
    'editor': {'zh': ('文件编辑工具栏和空白正文', '空白文档区域和保存按钮'),
        'en': ('a file editing toolbar and an empty document body', 'a blank document and Save button')},
}


def augment_app_caption(metadata, rng):
    """Preserve explicit empty-body/family constraints when sampling new rows.

Suggested integration: dispatch rows with dataset_version==VERSION here in
both train.BatchStream and validation.validate_conditioning. Keep historical
rows on captions.augment_caption. Do not add editor's historical body-line
DETAILS to this empty-editor family.
    """
    if metadata.get('dataset_version') != VERSION or not isinstance(rng, np.random.Generator):
        raise ValueError('app caption dispatcher requires audited V4 metadata')
    captions = metadata.get('short_captions')
    if not isinstance(captions, dict) or set(captions) != {'zh', 'en'}:
        raise ValueError('V4 needs both exact bilingual captions')
    result = captions['zh' if rng.random() < .5 else 'en']
    if not isinstance(result, str) or not 0 < len(result.encode('utf-8')) <= 120:
        raise ValueError('V4 caption outside native drawing budget')
    return result


def _button(c, x, y, w, h, label, primary=False, font=12):
    c.rect(x, y, w-1, h-1, radius=2, fill=c.fg if primary else c.bg, width=1)
    visible = c.text(x+w/2, y+(h-font)/2-1, label, font, True, center=True,
        fill=c.bg if primary else c.fg, maxw=w-8)
    if visible != label:
        raise ValueError(f'incomplete button label: {label}')
    return [x,y,x+w,y+h]


def render_app_target(kind, seed):
    if kind not in KINDS or type(seed) is not int:
        raise ValueError('invalid app target family or layout seed')
    rng = random.Random(seed)
    ui_language = 'zh' if kind == 'desktop' or seed % 2 else 'en'
    zh = ui_language == 'zh'
    dark = rng.random() < .08
    c = PixelCanvas(rng, dark=dark, size=256)
    c.radius = 2; c.linewidth = 1
    p = rng.randint(9, 15)
    title = 'CEMIVO' if seed % 5 < 3 else rng.choice({
        'tasks': ('TASKS', '待办'), 'desktop': ('CEMIVO', '桌面'), 'editor': ('NOTES', '笔记')}[kind])
    title_y = rng.randint(11, 16)
    c.text(p+3,title_y,title,rng.choice((13,14,15)),True,maxw=232-2*p)
    header_bottom = rng.randint(36,42)
    c.line(p,header_bottom,255-p,header_bottom,1)
    back = '返回' if zh else 'Back'
    home = '桌面' if zh else 'Home'
    nav_y = rng.randint(220,225)
    nav_height = 25
    controls = {}; states = {}; boxes = {}; body_bounds = None
    if kind == 'desktop':
        labels = ('聊天','笔记','待办','设置')
        gap = rng.randint(12,20)
        left = p+rng.randint(0,3)
        width = (256-2*left-gap)//2
        row_y = rng.randint(57,67)
        height = rng.randint(51,59)
        row_gap = rng.randint(18,26)
        for index,label in enumerate(labels):
            x=left+(index%2)*(width+gap)
            y=row_y+(index//2)*(height+row_gap)
            controls[label]=_button(c,x,y,width,height,label,font=14)
        states={'app_entry_count':4,'app_entries':list(labels),'switches':[], 'page':'desktop'}
        captions = {
            'zh':f'黑白桌面，{title}，四个应用按钮：聊天、笔记、待办、设置，桌面、返回',
            'en':f'Monochrome desktop, {title}, app buttons 聊天, 笔记, 待办, 设置, 桌面, 返回.'}
    elif kind == 'tasks':
        add = '添加' if zh else 'Add'
        count = rng.choice((3,4,5))
        row_y = header_bottom+rng.randint(17,22)
        row_step = rng.randint(25,27)
        texts = (('阅读计划','整理笔记','完成设计','检查任务','提交结果') if zh else
            ('Read notes','Organize files','Finish design','Review tasks','Send report'))
        visible = []
        checks = []
        for index in range(count):
            y=row_y+index*row_step
            x=p+4; size=rng.choice((12,13,14))
            c.rect(x,y,size-1,size-1,radius=0,width=1)
            checked=rng.random()<.25
            if checked:
                c.line(x+2,y+size//2,x+5,y+size-4,1)
                c.line(x+5,y+size-4,x+size-3,y+3,1)
            text=texts[index]
            c.text(x+size+9,y-1,text,12,maxw=230-(x+size+9))
            visible.append(text)
            checks.append({'bounds':[x,y,x+size,y+size],'checked':checked,'text':text})
            if index<count-1:c.line(p+4,y+22,251-p,y+22,1)
        add_y = min(nav_y-34,row_y+count*row_step+1)
        controls[add]=_button(c,256-p-68,add_y,68,26,add,True)
        states={'page':'tasks','task_count':count,'tasks':checks,'switches':[]}
        captions = {
            'zh':f'黑白待办列表，{title}，方形复选框和任务条目，{add}、{back}',
            'en':f'Black and white task list, {title}, square checkboxes, task rows, {add}, {back}.'}
    else:
        save = '保存' if zh else 'Save'
        menu = ('文件','编辑','视图') if zh else ('File','Edit','View')
        toolbar_y=header_bottom+rng.randint(10,14)
        for index,label in enumerate(menu):
            c.text(p+4+index*42,toolbar_y,label,11,True)
        controls[save]=_button(c,256-p-54,toolbar_y-4,54,24,save,True,font=11)
        format_y=toolbar_y+30
        for index,label in enumerate(('B','I','U')):
            x=p+4+index*29
            controls[label]=_button(c,x,format_y,23,23,label,font=12)
        body_y=format_y+34
        body_bottom=nav_y-rng.randint(9,15)
        body_left=p+rng.randint(0,4)
        body_right=255-body_left
        c.rect(body_left,body_y,body_right-body_left,body_bottom-body_y,radius=0,width=1)
        body_bounds=[body_left+2,body_y+2,body_right-1,body_bottom-1]
        # Deliberately no heading, placeholder, faux text, rows, checkboxes or
        # toggle controls in the document body.
        states={'page':'editor','document_body_empty':True,'document_body_bounds':body_bounds,'switches':[]}
        captions = {
            'zh':f'黑白文档编辑器，{title}，文件工具栏，空白正文，{save}、{back}',
            'en':f'Black and white document editor, {title}, file toolbar, empty body, {save}, {back}.'}
    controls[home]=_button(c,p,nav_y,64,nav_height,home,font=12)
    controls[back]=_button(c,256-p-64,nav_y,64,nav_height,back,font=12)
    for caption in captions.values():
        if len(caption.encode('utf-8'))>120:
            raise ValueError(f'caption exceeds 120 bytes: {caption}')
    for row in c.drawn_text:
        if row['requested'] != row['visible']:
            raise ValueError(f'caption target contains truncated text: {row}')
    pixels=np.asarray(c.image,dtype=np.uint8).copy()
    if body_bounds:
        x0,y0,x1,y1=body_bounds
        if not np.all(pixels[y0:y1,x0:x1]==c.bg):
            raise ValueError('empty editor contains non-background body pixels')
    metadata={'dataset_version':VERSION,'kind':kind,'seed':seed,'layout_seed':seed,
        'title':title,'sidebar':False,'dark':dark,'ui_language':ui_language,
        'buttons':list(controls),'controls':controls,'drawn_text':c.drawn_text,
        'state':states,'short_captions':captions,'caption_policy':'verbatim-app-family-v4',
        'render_native_size':256,'layout_coordinate_size':256,'rendering_version':6,
        'render_method':'native-vector-and-font-rasterization',
        'content_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
        'inference_renderer':False,'training_targets_only':True}
    return pixels,metadata
