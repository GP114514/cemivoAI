"""Held-out same-family state conditioning diagnostics, never an OCR score."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch

from .model import tokenize
from .region_objective import metadata_region_masks
from .state_sampling import STATE_MARKER,state_caption_variants

_ACTIONS=('chat_message','chat_composer','toggle')
_CONDITIONS=('correct','natural','wrong','natural_wrong')
_TIMES=(.1,.5,.9)


def _select_pairs(dataset,resolution,maximum,seed):
    groups={}
    for index,row in enumerate(dataset.rows):
        if (row.get('caption_policy')!=STATE_MARKER or row.get('split')!='val' or
                row.get('action') not in _ACTIONS or row.get('pair_role') not in ('before','after') or
                row.get('render_native_size',resolution)!=resolution or not isinstance(row.get('pair_id'),str)):
            continue
        group=groups.setdefault(row['pair_id'],{'before':[],'after':[]})
        group[row['pair_role']].append(index)
    eligible=[]
    for pair_id,roles in sorted(groups.items()):
        if len(roles['before'])!=1 or len(roles['after'])!=1:
            continue
        before,after=roles['before'][0],roles['after'][0]
        a,b=dataset.rows[before],dataset.rows[after]
        if (a.get('kind')!=b.get('kind') or a.get('action')!=b.get('action') or
                a.get('state')==b.get('state') or a.get('target_bounds')!=b.get('target_bounds')):
            continue
        eligible.append((pair_id,a['action'],before,after))
    if not eligible:
        return []
    rng=np.random.default_rng(seed)
    order=[eligible[int(index)] for index in rng.permutation(len(eligible))]
    selected=[]
    # Cover each explicit state action before using additional pair slots.
    for action in _ACTIONS:
        match=next((item for item in order if item[1]==action),None)
        if match is not None and len(selected)<maximum:
            selected.append(match)
    for item in order:
        if len(selected)>=maximum:
            break
        if item not in selected:
            selected.append(item)
    return selected


def _scores(error,mask):
    pixels=int(mask.sum().item())
    if not pixels:
        return None
    values=(error*mask).sum(dim=(1,2,3)).div(pixels).cpu().tolist()
    result={name+'_mse':float(value) for name,value in zip(_CONDITIONS,values)}
    result['wrong_minus_correct_mse']=result['wrong_mse']-result['correct_mse']
    result['natural_wrong_minus_natural_mse']=result['natural_wrong_mse']-result['natural_mse']
    return result


@torch.inference_mode()
def validate_state_conditioning(model,dataset,resolution,device,amp_enabled=True,pairs=6,seed=190627):
    """Compare opposite states of the SAME held-out layout with fixed noise.

    At most six before/after pairs are evaluated in both directions. State and
    visible-text ROIs use original target metadata at native resolution. Every
    caption condition sees the same image, noise and time. All random generators
    are local, and the caller's train/eval mode is restored, including on error.
    """
    if type(resolution) is not int or resolution<1 or type(pairs) is not int or not 1<=pairs<=6 or type(seed) is not int:
        raise ValueError('State validation needs a native size, 1..6 pairs and integer seed')
    selected=_select_pairs(dataset,resolution,pairs,seed)
    if not selected:
        return None
    device=torch.device(device)
    records=[]
    was_training=model.training
    model.eval()
    try:
        for pair_position,(pair_id,action,before,after) in enumerate(selected):
            for direction,(index,other_index) in enumerate(((before,after),(after,before))):
                metadata,other=dataset.rows[index],dataset.rows[other_index]
                row,wrong=dataset[index],dataset[other_index]
                original,natural=row['prompt'],state_caption_variants(metadata)[0]
                wrong_original,wrong_natural=wrong['prompt'],state_caption_variants(other)[0]
                captions=[original,natural,wrong_original,wrong_natural]
                if original==wrong_original or natural==wrong_natural:
                    raise ValueError('Opposite-state captions must distinguish their actual targets')
                tokens=tokenize(captions,model.config.text_max_length,device)
                image=torch.from_numpy(np.asarray(row['image']).copy()).unsqueeze(0).to(device)
                if image.shape!=(1,1,resolution,resolution):
                    raise ValueError('State validation requires original native target pixels, without resizing')
                masks=metadata_region_masks(metadata,resolution,resolution)
                masks={name:torch.from_numpy(value).unsqueeze(0).to(device) for name,value in masks.items()}
                if not bool(masks['state'].any()):
                    raise ValueError('State validation needs a nonempty declared target region')
                noise_seed=seed+pair_position*2+direction
                generator=torch.Generator(device=device).manual_seed(noise_seed)
                noise=torch.randn(image.shape,generator=generator,device=device)
                target=image-noise
                for scalar_time in _TIMES:
                    inputs=((1-scalar_time)*noise+scalar_time*image).repeat(len(captions),1,1,1)
                    times=torch.full((len(captions),),scalar_time,device=device)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                                        enabled=amp_enabled and device.type=='cuda'):
                        prediction=model(inputs,times,tokens)
                    if prediction.shape!=inputs.shape or not bool(torch.isfinite(prediction).all()):
                        raise ValueError('Invalid model prediction in state validation')
                    error=(prediction.float()-target).square()
                    records.append({'pair_id':pair_id,'action':action,'kind':metadata['kind'],
                        'pair_role':metadata['pair_role'],'dataset_index':index,'wrong_dataset_index':other_index,
                        'state':deepcopy(metadata['state']),'wrong_state':deepcopy(other['state']),
                        'target_bounds':list(metadata['target_bounds']),'captions':dict(zip(_CONDITIONS,captions)),
                        't':scalar_time,'noise_seed':noise_seed,
                        'state_pixels':int(masks['state'].sum().item()),'text_pixels':int(masks['text'].sum().item()),
                        'state_roi':_scores(error,masks['state']),'text_roi':_scores(error,masks['text'])})
        def summary(region,time=None):
            rows=[record[region] for record in records if record[region] is not None and (time is None or record['t']==time)]
            return ({key:float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else None)
        return {'pairs':len(selected),'examples':len(selected)*2,'evaluations':len(records),
            'actions':sorted({item[1] for item in selected}),'native_size':resolution,
            'metric_kind':'same-family opposite-state conditioning diagnostic','quality_accepted':False,
            'caption_conditions':list(_CONDITIONS),'seed':seed,
            'state_roi':summary('state_roi'),'text_roi':summary('text_roi'),
            'by_noise_time':{str(t):{'state_roi':summary('state_roi',t),'text_roi':summary('text_roi',t)} for t in _TIMES},
            'details':records,'updated_at':datetime.now(timezone.utc).isoformat(),
            'interpretation':'Held-out same-family opposite-state velocity MSE on declared native ROIs. '
                'Positive margins diagnose caption conditioning only; these are not OCR accuracy, image quality, '
                'readable-text scores or proof of usable UI interaction.'}
    finally:
        model.train(was_training)


def write_state_validation(directory,report,step):
    if report is None:
        return
    from .train import atomic_json
    directory=Path(directory)/'validation'
    directory.mkdir(parents=True,exist_ok=True)
    report=dict(report,checkpoint_step=int(step))
    atomic_json(directory/f'state-step-{step:08}.json',report)
    atomic_json(directory/'state-latest.json',report)
