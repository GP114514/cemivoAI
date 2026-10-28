"""Training targets for complete native task/note flows, never a UI runtime.

Pillow fonts and geometry below construct supervised target rasters only. This
module is not imported by image inference, Agent execution or the application.
The five related states share one fixed story layout. Semantic effect areas
are declared before raster comparison and independently checked by the export.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np

from .dataset import PixelCanvas

VERSION='native256-app-flows-v5'
CAPTION_POLICY='verbatim-app-flow-state-v5'
FAMILIES=('tasks','notes')
STATES={'tasks':('empty','draft','added','completed','undone'),
    'notes':('new','typed','saved','listselected','newagain')}


def _text(canvas,box,value,*,size=11,bold=False,center=False,fill=None,role='label'):
    x0,y0,x1,y1=box
    visible=canvas.text((x0+x1)/2 if center else x0,y0,value,size,bold,
        center=center,fill=fill,maxw=x1-x0)
    row=canvas.drawn_text[-1]
    if visible!=value:
        raise ValueError(f'flow label truncated: {value}')
    bx0,by0,bx1,by1=row['bbox']
    if not (0<=bx0<bx1<=256 and 0<=by0<by1<=256 and x0<=bx0 and bx1<=x1 and y0<=by0 and by1<=y1):
        raise ValueError(f'flow label exceeds its declared container: {value} {row["bbox"]} {box}')
    row.update(role=role,container=list(box))


def _button(canvas,bounds,label,primary=False):
    x0,y0,x1,y1=bounds
    canvas.rect(x0,y0,x1-x0-1,y1-y0-1,radius=1,width=1,
        fill=canvas.fg if primary else canvas.bg)
    _text(canvas,[x0+4,y0+6,x1-4,y1-4],label,size=11,bold=True,center=True,
        fill=canvas.bg if primary else canvas.fg,role='button')


def _captions(family,stage,title,literal,labels):
    back=labels['back']
    if family=='tasks':
        add=labels['add']
        if stage=='empty':
            state_zh='空任务列表，空输入框';state_en='empty list, empty input'
        elif stage=='draft':
            state_zh=f'空列表，输入“{literal}”';state_en=f'empty list, input "{literal}"'
        elif stage=='completed':
            state_zh=f'任务“{literal}”已勾选，空输入框';state_en=f'one checked task "{literal}", empty input'
        else:
            state_zh=f'任务“{literal}”未勾选，空输入框';state_en=f'one unchecked task "{literal}", empty input'
        result={'zh':f'黑白待办列表，{title}，{state_zh}，{add}、{back}',
            'en':f'Black/white task list, {title}, {state_en}, {add}, {back}.'}
    else:
        new,save=labels['new'],labels['save']
        if stage=='new':
            state_zh='笔记列表和正文为空';state_en='empty note list and body'
        elif stage=='newagain':
            state_zh=f'一条已存笔记“{literal}”，空白正文';state_en=f'one saved note "{literal}", blank body'
        elif stage=='typed':
            state_zh=f'未保存正文“{literal}”';state_en=f'unsaved body "{literal}"'
        elif stage=='saved':
            state_zh=f'已保存笔记“{literal}”';state_en=f'saved note "{literal}"'
        else:
            state_zh=f'已选择笔记“{literal}”';state_en=f'selected note "{literal}"'
        result={'zh':f'黑白笔记编辑器，{title}，{state_zh}，{new}、{save}、{back}',
            'en':f'Black/white notes editor, {title}, {state_en}, {new}, {save}, {back}.'}
    for caption in result.values():
        if not 0<len(caption.encode('utf-8'))<=120:
            raise ValueError(f'flow caption exceeds 120 UTF8 bytes: {caption}')
    return result


def render_story(family,seed):
    """Return five (uint8 target, metadata) pairs with fixed native geometry."""
    if family not in FAMILIES or type(seed) is not int or seed<0:
        raise ValueError('flow family and nonnegative integer story seed required')
    rng=random.Random(seed)
    dark=rng.random()<.08
    zh=seed%2==1
    p=rng.randint(10,16)
    title=rng.choice(('CEMIVO','待办' if family=='tasks' else '笔记','TASKS' if family=='tasks' else 'NOTES'))
    labels={'home':'桌面' if zh else 'Home','back':'返回' if zh else 'Back',
        'add':'添加' if zh else 'Add','new':'新建' if zh else 'New','save':'保存' if zh else 'Save'}
    suffix=str(rng.randint(1,99))
    literal=(('阅读' if family=='tasks' else '计划')+suffix if zh else
        ('Read ' if family=='tasks' else 'Plan ')+suffix)
    nav_y=rng.randint(222,225)
    controls={'home':[p,nav_y,p+62,nav_y+25],'back':[194-p,nav_y,256-p,nav_y+25]}
    if family=='tasks':
        row_y=rng.randint(66,76)
        check=[p+7,row_y,p+21,row_y+14]
        list_bounds=[p+3,54,253-p,157]
        row_bounds=[p+4,row_y-3,252-p,row_y+20]
        input_y=rng.randint(175,180)
        add_x=183+rng.randint(0,5)
        input_box=[p,input_y,add_x-7,input_y+27]
        input_inner=[p+5,input_y+5,add_x-12,input_y+22]
        controls.update(add=[add_x,input_y,256-p,input_y+27],input=input_box)
    else:
        left_width=rng.randint(67,76)
        top=rng.randint(85,91)
        editor_box=[left_width+11,top,256-p,210+rng.randint(0,3)]
        body=[editor_box[0]+5,top+6,editor_box[2]-5,editor_box[3]-5]
        list_bounds=[p+1,top+2,left_width,editor_box[3]-2]
        entry=[p+2,top+3,left_width-2,top+31]
        status_bounds=[118,13,245,31]
        controls.update(new=[p,45,p+64,71],save=[184-p,45,256-p,71],editor=editor_box)
    output=[]
    for position,stage in enumerate(STATES[family]):
        c=PixelCanvas(random.Random(seed),dark=dark,size=256)
        c.radius=1;c.linewidth=1
        _text(c,[p+2,12,113,34],title,size=14,bold=True,role='title')
        c.line(p,39,256-p,39,1)
        _button(c,controls['home'],labels['home'])
        _button(c,controls['back'],labels['back'])
        transition=None
        if family=='tasks':
            has_task=position>=2
            checked=stage=='completed'
            draft=literal if stage=='draft' else ''
            _button(c,controls['add'],labels['add'],primary=True)
            x0,y0,x1,y1=input_box;c.rect(x0,y0,x1-x0-1,y1-y0-1,radius=1,width=1)
            if draft:_text(c,input_inner,draft,size=11,role='task_draft')
            if has_task:
                x0,y0,x1,y1=check;c.rect(x0,y0,x1-x0-1,y1-y0-1,radius=0,width=1)
                if checked:
                    c.line(x0+3,y0+7,x0+6,y0+10,1);c.line(x0+6,y0+10,x0+11,y0+3,1)
                _text(c,[p+30,row_y-1,247-p,row_y+18],literal,size=12,role='task_text')
            state={'page':'tasks','task_count':int(has_task),'task_list_empty':not has_task,
                'tasks':[{'text':literal,'checked':checked,'checkbox_bounds':check}] if has_task else [],
                'draft_text':draft,'input_empty':not draft,'list_bounds':list_bounds,'composer_bounds':input_inner,
                'switches':[]}
            if position:
                effects=([input_inner] if stage=='draft' else [row_bounds,input_inner] if stage=='added' else [check])
                trigger=(input_box if stage=='draft' else controls['add'] if stage=='added' else check)
                transition={'action':{'draft':'type_first_task','added':'add_first_task','completed':'complete_task',
                    'undone':'undo_completion'}[stage],'trigger_bounds':trigger,'effect_regions':effects,
                    'event_text':literal if stage=='draft' else None}
            empty_regions=([list_bounds] if not has_task else [])+([input_inner] if not draft else [])
        else:
            document=literal if stage in ('typed','saved','listselected') else ''
            saved=stage in ('saved','listselected','newagain')
            selected=stage=='listselected'
            _button(c,controls['new'],labels['new'])
            _button(c,controls['save'],labels['save'],primary=True)
            c.line(left_width+3,top-3,left_width+3,editor_box[3],1)
            x0,y0,x1,y1=editor_box;c.rect(x0,y0,x1-x0-1,y1-y0-1,radius=0,width=1)
            status=({'new':'新建','typed':'未保存','saved':'已保存','listselected':'已选','newagain':'新建'}[stage] if zh else
                {'new':'New','typed':'Draft','saved':'Saved','listselected':'Selected','newagain':'New'}[stage])
            _text(c,status_bounds,status,size=11,role='document_status')
            if saved:
                x0,y0,x1,y1=entry
                c.rect(x0,y0,x1-x0-1,y1-y0-1,radius=1,width=1,fill=c.fg if selected else c.bg)
                _text(c,[x0+4,y0+7,x1-4,y1-4],literal,size=10,
                    fill=c.bg if selected else c.fg,role='note_list_entry')
            if document:_text(c,body,document,size=12,role='document_body')
            state={'page':'notes','document_body_empty':not document,'document_text':document,
                'document_body_bounds':body,'note_list_empty':not saved,'note_count':int(saved),
                'saved_notes':[{'title':literal,'body':literal}] if saved else [],'selected_note':literal if selected else None,
                'saved_state_training_target':stage=='saved','document_status':status,'switches':[]}
            if position:
                effects=([body,status_bounds] if stage=='typed' else [entry,status_bounds] if stage in ('saved','listselected') else [body,entry,status_bounds])
                trigger=(editor_box if stage=='typed' else controls['save'] if stage=='saved' else entry if stage=='listselected' else controls['new'])
                transition={'action':{'typed':'type_note','saved':'save_note_training_target','listselected':'select_saved_note',
                    'newagain':'new_document_preserving_list'}[stage],'trigger_bounds':trigger,'effect_regions':effects,
                    'event_text':literal if stage=='typed' else None}
            empty_regions=([body] if not document else [])+([list_bounds] if not saved else [])
        pixels=np.asarray(c.image,dtype=np.uint8).copy()
        for bounds in empty_regions:
            x0,y0,x1,y1=bounds
            if not np.all(pixels[y0:y1,x0:x1]==c.bg):
                raise ValueError(f'claimed empty state contains foreground: {family} {stage} {bounds}')
        captions=_captions(family,stage,title,literal,labels)
        required_literal=(literal if (family=='tasks' and stage!='empty') or (family=='notes' and stage in ('typed','saved','listselected','newagain')) else None)
        if required_literal and any(required_literal not in caption for caption in captions.values()):
            raise ValueError('flow caption lost exact requested content')
        metadata={'dataset_version':VERSION,'caption_policy':CAPTION_POLICY,'kind':'tasks' if family=='tasks' else 'editor',
            'flow_family':family,'flow_stage':stage,'flow_position':position,'story_id':f'{family}-{seed}',
            'seed':seed,'layout_seed':seed,'title':title,'ui_language':'zh' if zh else 'en','dark':dark,'sidebar':family=='notes',
            'buttons':[labels[k] for k in ('home','back','add') if k in controls] if family=='tasks' else [labels[k] for k in ('home','back','new','save')],
            'controls':{key:list(value) for key,value in controls.items()},'drawn_text':c.drawn_text,'state':state,
            'required_literal':required_literal,'short_captions':captions,'empty_regions':empty_regions,
            'transition_from_previous':transition,'training_targets_only':True,'inference_renderer':False,
            'real_business_executed':False,'native_size':256,'render_native_size':256,'render_method':'native-vector-and-font-rasterization',
            'rendering_version':7,'content_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()}
        output.append((pixels,metadata))
    return output


def validate_transition(before,after,transition):
    """Independently test predeclared semantic effect regions against rasters."""
    if before.shape!=(256,256) or after.shape!=(256,256) or before.dtype!=np.uint8 or after.dtype!=np.uint8:
        raise ValueError('flow transitions require native uint8 256x256 targets')
    regions=transition['effect_regions']
    mask=np.zeros((256,256),dtype=bool)
    for bounds in regions:
        if len(bounds)!=4 or any(type(v) is not int for v in bounds):raise ValueError('invalid effect coordinates')
        x0,y0,x1,y1=bounds
        if not 0<=x0<x1<=256 or not 0<=y0<y1<=256:raise ValueError('effect outside native canvas')
        mask[y0:y1,x0:x1]=True
    delta=before!=after
    outside=int(np.count_nonzero(delta&~mask));changed=int(delta.sum())
    if outside or not changed:raise ValueError(f'flow effects failed: outside={outside}, changed={changed}')
    x0,y0,x1,y1=transition['trigger_bounds']
    if not 0<=x0<x1<=256 or not 0<=y0<y1<=256:raise ValueError('invalid trigger rectangle')
    return {'verified_against_actual_pixels':True,'outside_changed_pixels':outside,'changed_pixels':changed,
        'selected_pixels':int(mask.sum()),'mask_area_percent':round(float(mask.mean()*100),4),
        'outside_sha256_before':hashlib.sha256(before[~mask].tobytes()).hexdigest(),
        'outside_sha256_after':hashlib.sha256(after[~mask].tobytes()).hexdigest(),
        'effect_region_source':'semantic layout declaration before raster comparison'}
