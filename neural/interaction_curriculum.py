"""Readable native256 state-pair training targets, never an inference renderer.

Version 2 removes seed IDs from visible titles, uses natural bilingual captions,
and validates every switch with the unchanged production pixel-state detector.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image

from .dataset import PixelCanvas, build_vocab
from .interaction_data import Sample, InteractionPair

VERSION='native256-interaction-curriculum-v2'
MIX_VERSION='v2-plus-native256-curriculum-v2'
CAPTION_POLICY='verbatim-interaction-state'
LABELS={
    'en':{'new':'New','sidebar':'Chat','empty_hint':'No messages','placeholder':'Message',
        'attach':'Attach','send':'Send','message':'Hello','typed':'Hi',
        'heading':'Settings','Notifications':'Notifications','Sound':'Sound','Privacy':'Privacy','back':'Back','save':'Save'},
    'zh':{'new':'新建','sidebar':'会话','empty_hint':'暂无消息','placeholder':'输入消息',
        'attach':'附件','send':'发送','message':'你好','typed':'好',
        'heading':'设置','Notifications':'通知','Sound':'声音','Privacy':'隐私','back':'返回','save':'保存'}}


@dataclass(frozen=True)
class Layout:
    padding:int
    header:int
    sidebar:int
    radius:int
    title_size:int
    label_size:int
    composer_height:int
    message_offset:int
    settings_top:int
    settings_gap:int
    switch_x:int
    switch_width:int
    buttons_y:int
    dark:bool
    title:str
    language:str


def layout_for_seed(seed):
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('layout seed must be an integer in 0..2**32-1')
    rng=random.Random(seed ^ 0x1B873593)
    return Layout(rng.choice((8,10,12)),rng.choice((29,31,33,35)),rng.choice((56,60,64,68)),
        rng.choice((3,4,5,6)),rng.choice((12,13,14)),rng.choice((12,13)),
        rng.choice((52,54,56)),rng.randrange(6),rng.randrange(5),rng.choice((31,33,35)),
        rng.choice((194,196,198,200,202)),rng.choice((32,34,36)),rng.choice((207,210,213)),
        rng.random()<.15,'CEMIVO' if seed%10<6 else rng.choice(('CHAT','NOTES','WORK','聊天','笔记')),
        'zh' if seed%2 else 'en')


def _canvas(seed,kind):
    layout=layout_for_seed(seed)
    c=PixelCanvas(random.Random(seed),dark=layout.dark,size=256)
    c.padding=layout.padding;c.radius=layout.radius;c.linewidth=1;c.header=layout.header;c.sidebar=layout.sidebar
    p=layout.padding
    c.rect(p,p,255-2*p,255-2*p,radius=layout.radius)
    c.line(p,p+layout.header,255-p,p+layout.header)
    c.text(p+9,p+7,layout.title,layout.title_size,True,maxw=232-2*p)
    if kind=='chat':
        labels=LABELS[layout.language]
        c.line(p+layout.sidebar,p+layout.header,p+layout.sidebar,255-p)
        _button(c,p+5,p+layout.header+11,layout.sidebar-10,25,labels['new'])
        c.text(p+9,p+layout.header+49,labels['sidebar'],layout.label_size,True)
    return c,layout


def _button(c,x,y,width,height,label,primary=False):
    c.rect(x,y,width-1,height-1,fill=c.fg if primary else c.bg,radius=3)
    visible=c.text(x+width/2,y+4,label,12,True,center=True,
        fill=c.bg if primary else c.fg,maxw=width-6)
    if visible!=label:raise ValueError('Curriculum button labels must be complete')


def _chat(seed,messages,composer):
    c,layout=_canvas(seed,'chat');labels=LABELS[layout.language];p=layout.padding
    x=p+layout.sidebar+8;right=247-p;y=p+layout.header+9
    compose_y=247-p-layout.composer_height
    message_box=(x,y,right+1,compose_y-4)
    if messages:
        top=y+15+layout.message_offset
        c.rect(x+5,top,right-x-12,34,radius=5)
        c.text(x+12,top+8,messages[0],14)
    else:
        c.text(x+(right-x)/2,y+46+layout.message_offset,labels['empty_hint'],12,center=True,maxw=right-x-12)
    c.rect(x,compose_y,right-x,layout.composer_height,radius=4)
    c.text(x+6,compose_y+6,composer or labels['placeholder'],12,maxw=right-x-12)
    # Controls have distinct boxes. The composer state pair changes only the
    # text line, never the outline or either button.
    button_y=compose_y+layout.composer_height-24
    _button(c,x+5,button_y,60,21,labels['attach'])
    _button(c,right-62,button_y,57,21,labels['send'],True)
    composer_box=(x+3,compose_y+3,right-2,button_y-2)
    return c,layout,message_box,composer_box


def _switch(c,bounds,state):
    x0,y0,x1,y1=bounds
    c.d.rounded_rectangle((x0,y0,x1-1,y1-1),radius=8,
        fill=c.fg if state=='on' else c.bg,outline=c.fg,width=1)
    cx=x1-9 if state=='on' else x0+8
    c.d.ellipse((cx-5,y0+3,cx+5,y1-4),fill=c.bg if state=='on' else c.fg)


def _settings(seed,states):
    c,layout=_canvas(seed,'settings');labels=LABELS[layout.language];p=layout.padding
    c.text(p+12,p+layout.header+11,labels['heading'],16,True)
    boxes={}
    for index,key in enumerate(('Notifications','Sound','Privacy')):
        y=p+layout.header+43+layout.settings_top+index*layout.settings_gap
        c.text(p+13,y+1,labels[key],layout.label_size)
        bounds=(layout.switch_x,y,layout.switch_x+layout.switch_width,y+16)
        _switch(c,bounds,states[key]);boxes[key]=bounds
    _button(c,p+13,layout.buttons_y,66,25,labels['back'])
    _button(c,175-p,layout.buttons_y,66,25,labels['save'],True)
    return c,layout,boxes


def natural_captions(kind,layout,state,drawn_text):
    """Keep state words natural and every visible label exact, without slicing."""
    labels=list(dict.fromkeys(row['visible'] for row in drawn_text))
    style_zh='黑底' if layout.dark else '白底';style_en='dark' if layout.dark else 'light'
    if kind=='chat':
        zh_message='空消息区' if not state['messages'] else '消息'+state['messages'][0]
        en_message='empty messages' if not state['messages'] else 'message '+state['messages'][0]
        zh_input='空输入框' if not state['composer'] else '输入'+state['composer']
        en_input='empty input' if not state['composer'] else 'input '+state['composer']
        zh=f'{layout.title}聊天{style_zh}，{zh_message}，{zh_input}'
        en=f'Chat {layout.title} {style_en}, {en_message}, {en_input}'
    else:
        mapping=LABELS[layout.language]
        zh_states='，'.join(mapping[key]+('开' if state[key]=='on' else '关') for key in ('Notifications','Sound','Privacy'))
        en_states=', '.join(mapping[key]+' '+state[key] for key in ('Notifications','Sound','Privacy'))
        zh=f'{layout.title}设置{style_zh}，{zh_states}'
        en=f'Settings {layout.title} {style_en}, {en_states}'
    result={}
    for language,caption in (('zh',zh),('en',en)):
        remaining=[label for label in labels if label not in caption]
        if remaining:caption+=('；字' if language=='zh' else '; ')+'/'.join(remaining)
        if len(caption.encode('utf-8'))>120:
            raise ValueError(f'Truthful natural {language} caption exceeds 120 bytes: {caption}')
        if any(label not in caption for label in labels):raise ValueError('Caption lost a visible label')
        result[language]=caption
    return result


def _sample(seed,kind,c,layout,state):
    if any(row['requested']!=row['visible'] or row['font_px']<12 for row in c.drawn_text):
        raise ValueError('Curriculum requires complete native labels with fonts >=12px')
    captions=natural_captions(kind,layout,state,c.drawn_text)
    language=layout.language;label_map=LABELS[language]
    metadata={'seed':seed,'layout_seed':seed,'kind':kind,'title':layout.title,'language':language,
        'dark':layout.dark,'sidebar':kind=='chat','prompt':captions[language],
        'short_captions':captions,'state':state,'state_label_map':dict(label_map),
        'buttons':[label_map[key] for key in (('new','attach','send') if kind=='chat' else ('back','save'))],
        'caption_policy':CAPTION_POLICY,'dataset_version':VERSION,'render_native_size':256,
        'render_method':'native-vector-and-font-rasterization','rendering_version':5,
        'layout_coordinate_size':256,'drawn_text':list(c.drawn_text)}
    return Sample(np.asarray(c.image,dtype=np.uint8).copy(),metadata)


def _pair(seed,action,bounds,before,after,event):
    x0,y0,x1,y1=bounds;mask=np.zeros((256,256),bool);mask[y0:y1,x0:x1]=True
    if not np.array_equal(before.pixels[~mask],after.pixels[~mask]):
        raise ValueError('Pair changes pixels outside its actual state region')
    changed=int(np.count_nonzero(before.pixels!=after.pixels))
    if not changed:raise ValueError('A state pair must visibly change')
    pair_id=hashlib.sha256(f'{VERSION}:{seed}:{action}'.encode()).hexdigest()[:24]
    outside=hashlib.sha256(before.pixels[~mask].tobytes()).hexdigest()
    for role,sample in (('before',before),('after',after)):
        sample.metadata.update(pair_id=pair_id,pair_role=role,action=action,target_bounds=list(bounds),
            event=dict(event),preserve_outside=True,outside_changed_pixels=0,outside_sha256=outside,
            changed_pixels=changed,content_sha256=hashlib.sha256(sample.pixels.tobytes()).hexdigest())
    return InteractionPair(pair_id,tuple(bounds),before,after)


def generate_pairs(seed):
    layout=layout_for_seed(seed);labels=LABELS[layout.language];pairs=[]
    for action,states in (('chat_message',(([],''),([labels['message']],''))),
            ('chat_composer',(([],''),([],labels['typed'])))):
        samples=[]
        for messages,composer in states:
            c,layout,message_bounds,composer_bounds=_chat(seed,messages,composer)
            samples.append(_sample(seed,'chat',c,layout,{'messages':list(messages),'composer':composer}))
        bounds=message_bounds if action=='chat_message' else composer_bounds
        event={'type':'submit','text':'Send '+labels['message']} if action=='chat_message' else {
            'type':'click','x':(bounds[0]+bounds[2])/2,'y':(bounds[1]+bounds[3])/2,
            'text':labels['typed'],'input_method':'contextmenu'}
        pairs.append(_pair(seed,action,bounds,*samples,event))
    rng=random.Random(seed^0x9E3779B9)
    states={key:rng.choice(('on','off')) for key in ('Notifications','Sound','Privacy')}
    selected=rng.choice(tuple(states));after_states=dict(states)
    after_states[selected]='off' if states[selected]=='on' else 'on'
    samples=[]
    from stable_edit import measure_toggle_state
    for values in (states,after_states):
        c,layout,boxes=_settings(seed,values)
        sample=_sample(seed,'settings',c,layout,values)
        output=BytesIO();Image.fromarray(sample.pixels).save(output,format='PNG')
        for key,bounds in boxes.items():
            if measure_toggle_state(output.getvalue(),list(bounds))!=values[key]:
                raise ValueError('Training switch target is incompatible with the production state detector')
        sample.metadata['switch_bounds']={key:list(value) for key,value in boxes.items()}
        sample.metadata['production_switch_gate_passed']=True
        samples.append(sample)
    bounds=boxes[selected]
    pairs.append(_pair(seed,'toggle',bounds,*samples,{'type':'click',
        'x':(bounds[0]+bounds[2])/2,'y':(bounds[1]+bounds[3])/2}))
    return pairs


def _component_splits(group_count,hash_groups,seeds,val_fraction):
    """Duplicate pixel identities join layout groups before assigning a split."""
    parent=list(range(group_count))
    def find(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]];index=parent[index]
        return index
    for first,second in hash_groups:
        left,right=find(first),find(second)
        if left!=right:parent[max(left,right)]=min(left,right)
    components={}
    for index in range(group_count):components.setdefault(find(index),[]).append(index)
    splits={}
    for root,members in components.items():
        digest=hashlib.sha256(f'{VERSION}:split:{min(seeds[index] for index in members)}'.encode()).digest()
        split='val' if int.from_bytes(digest[:8],'big')<int(val_fraction*2**64) else 'train'
        for index in members:splits[index]=(split,root)
    return splits,len(components)


def export_dataset(output,groups=2000,seed=20261027,val_fraction=.05,progress=None):
    if type(groups) is not int or groups<1 or type(seed) is not int or not 0<=seed<2**32 or seed+groups>2**32:
        raise ValueError('Invalid curriculum group count or seed')
    if isinstance(val_fraction,bool) or not isinstance(val_fraction,(int,float)) or not math.isfinite(val_fraction) or not 0<=val_fraction<=.5:
        raise ValueError('Invalid held-out fraction')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    count=groups*6;rows=[];owners={};joins=[];seeds=list(range(seed,seed+groups))
    target=output/'images-00000.npy'
    pixels=np.lib.format.open_memmap(target.with_suffix('.npy.partial'),mode='w+',dtype=np.uint8,shape=(count,256,256))
    try:
        for group,layout_seed in enumerate(seeds):
            for pair in generate_pairs(layout_seed):
                for sample in (pair.before,pair.after):
                    index=len(rows);row=dict(sample.metadata)
                    row.update(index=index,offset=index,shard=0,layout_group=group,
                        paired_sample_index=index+1 if row['pair_role']=='before' else index-1)
                    digest=row['content_sha256']
                    if digest in owners:joins.append((group,owners[digest]))
                    else:owners[digest]=group
                    pixels[index]=sample.pixels;rows.append(row)
            if progress and ((group+1)%100==0 or group+1==groups):progress(group+1,groups)
        pixels.flush()
    finally:del pixels
    target.with_suffix('.npy.partial').replace(target)
    assigned,components=_component_splits(groups,joins,seeds,val_fraction)
    hashes={'train':set(),'val':set()};counts=Counter()
    with (output/'metadata-00000.jsonl').open('x',encoding='utf-8',newline='\n') as handle:
        for row in rows:
            row['split'],row['split_component']=assigned[row['layout_group']]
            counts[row['split']]+=1;hashes[row['split']].add(row['content_sha256'])
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    if hashes['train']&hashes['val']:raise ValueError('A duplicate image crossed grouped splits')
    (output/'vocab.json').write_text(json.dumps(build_vocab(),indent=2),encoding='utf-8')
    manifest={'version':1,'dataset_version':VERSION,'format':'npy-uint8-grayscale-shards',
        'size':256,'render_native_size':256,'layout_coordinate_size':256,'rendering_version':5,
        'render_method':'native-vector-and-font-rasterization','dtype':'uint8','normalization':'image / 127.5 - 1',
        'base_seed':seed,'count':count,'groups':groups,'train_count':counts['train'],'val_count':counts['val'],
        'vocab':'vocab.json','max_length':192,'caption_policy':CAPTION_POLICY,
        'split_policy':'Union all layout groups connected by identical pixel hashes; assign each complete component to one deterministic split',
        'split_components':components,'cross_split_pixel_leaks':0,'selected_switch_state_gate_passes':groups*2,
        'all_switch_state_gate_passes':groups*6,'title_seed_ids_visible':False,
        'short_caption_languages':['zh','en'],'max_short_caption_utf8_bytes':120,
        'quality_claim':'Training targets only; generated model quality not established',
        'provenance':'Original native256 training raster targets. No pretrained images/weights; not an inference renderer.',
        'shards':[{'images':target.name,'metadata':'metadata-00000.jsonl','count':count,'start_index':0,
            'train_count':counts['train'],'val_count':counts['val']}]}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest
