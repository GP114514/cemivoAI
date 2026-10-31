"""Held-out task/note transition diagnostics; no training or UI rendering."""
from __future__ import annotations
import numpy as np
import torch
from .model import tokenize
from .region_objective import metadata_region_masks
from .state_sampling import APP_FLOW_MARKER, app_flow_caption_variants

TRANSITIONS=(('tasks','empty','draft'),('tasks','draft','added'),
    ('tasks','added','completed'),('tasks','completed','undone'),
    ('notes','new','typed'),('notes','typed','saved'),
    ('notes','saved','listselected'),('notes','listselected','newagain'))

def select_transitions(dataset, seed=20261101):
    groups={}
    for index,row in enumerate(dataset.rows):
        if row.get('split')!='val' or row.get('caption_policy')!=APP_FLOW_MARKER:
            continue
        key=(row.get('flow_family'),row.get('story_id'))
        if key[0] not in ('tasks','notes') or not isinstance(key[1],str):
            continue
        stages=groups.setdefault(key,{})
        stage=row.get('flow_stage')
        if stage in stages:
            previous=dataset.rows[stages[stage]]
            # V5 stores bilingual captions as two copies of the same target.
            # Treat these as one held-out state, never two independent examples.
            keys=('content_sha256','state','short_captions','drawn_text','empty_regions')
            if not row.get('content_sha256') or any(row.get(k)!=previous.get(k) for k in keys):
                raise ValueError('Conflicting held-out app story stage')
            def transition(meta):
                return {k:v for k,v in (meta.get('transition_from_previous') or {}).items()
                    if k not in ('previous_offset','current_offset')}
            if transition(row)!=transition(previous):
                raise ValueError('Conflicting held-out app transition geometry')
            continue
        stages[stage]=index
    rng=np.random.default_rng(seed)
    selected=[]
    for family,before,after in TRANSITIONS:
        eligible=[(story,stages[before],stages[after]) for (kind,story),stages in sorted(groups.items())
            if kind==family and before in stages and after in stages]
        if eligible:
            story,a,b=eligible[int(rng.integers(len(eligible)))]
            selected.append((family,before,after,story,a,b))
    return selected

@torch.inference_mode()
def validate_app_flows(model,dataset,resolution,device,amp_enabled=True,seed=20261101):
    selected=select_transitions(dataset,seed)
    if not selected:
        return None
    device=torch.device(device);records=[];was_training=model.training
    model.eval()
    try:
        for pair,(family,before,after,story,a,b) in enumerate(selected):
            left,right=dataset.rows[a],dataset.rows[b]
            # Validate captions against actual visible literals/state before any inference.
            ca,cb=app_flow_caption_variants(left),app_flow_caption_variants(right)
            if any(x==y for x,y in zip(ca,cb)):
                raise ValueError('App transition captions must distinguish their states')
            for direction,(index,other,correct,wrong) in enumerate(((a,b,ca,cb),(b,a,cb,ca))):
                metadata=dataset.rows[index]
                image=torch.from_numpy(np.asarray(dataset[index]['image']).copy()).unsqueeze(0).to(device)
                if image.shape!=(1,1,resolution,resolution):
                    raise ValueError('App flow validation requires native-size pixels')
                masks=metadata_region_masks(metadata,resolution,resolution)
                # Compare on the same union of the two declared state regions in both directions.
                other_masks=metadata_region_masks(dataset.rows[other],resolution,resolution)
                masks={name:torch.from_numpy(value|other_masks[name]).unsqueeze(0).to(device) for name,value in masks.items()}
                if not bool(masks['state'].any()):raise ValueError('Missing app state ROI')
                tokens=tokenize([correct[1],correct[0],wrong[1],wrong[0]],model.config.text_max_length,device)
                generator=torch.Generator(device=device).manual_seed(seed+pair*2+direction)
                noise=torch.randn(image.shape,generator=generator,device=device);target=image-noise
                for t in (.1,.5,.9):
                    inputs=((1-t)*noise+t*image).repeat(4,1,1,1)
                    times=torch.full((4,),t,device=device)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp_enabled and device.type=='cuda'):
                        prediction=model(inputs,times,tokens)
                    if prediction.shape!=inputs.shape or not bool(torch.isfinite(prediction).all()):
                        raise ValueError('Invalid app flow prediction')
                    error=(prediction.float()-target).square()
                    record={'family':family,'transition':before+'->'+after,'story':story,'direction':direction,
                        'index':index,'other_index':other,'t':t,'state':metadata['flow_stage']}
                    for name,mask in masks.items():
                        count=int(mask.sum().item())
                        if count:
                            values=(error*mask).sum(dim=(1,2,3)).div(count).cpu().tolist()
                            record[name+'_roi']={'correct_en_mse':values[0],'correct_zh_mse':values[1],
                                'wrong_en_mse':values[2],'wrong_zh_mse':values[3],
                                'en_margin':values[2]-values[0],'zh_margin':values[3]-values[1]}
                    records.append(record)
        def average(rows):
            values=[r['state_roi'] for r in rows]
            return {key:float(np.mean([v[key] for v in values])) for key in values[0]}
        return {'pairs':len(selected),'evaluations':len(records),'native_size':resolution,'seed':seed,
            'state_roi':average(records),'by_transition':{family+':'+before+'->'+after:
                average([r for r in records if r['family']==family and r['transition']==before+'->'+after])
                for family,before,after,story,a,b in selected},'details':records,'quality_accepted':False,
            'interpretation':'Held-out same-story opposite-state velocity MSE, English and Chinese. '
                'A positive margin is a conditioning diagnostic, not image readability or successful tool execution.'}
    finally:
        model.train(was_training)
