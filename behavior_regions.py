"""AI-selected effect areas separate from the clicked trigger; no UI drawing."""
from io import BytesIO
import hashlib
import math

import numpy as np
from PIL import Image
from stable_edit import StableEditError, _dimensions, _region_bounds, _decode_png


def contains(outer, inner):
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def validate_effect_regions(value, frame, event):
    width,height = _dimensions(frame)
    if not isinstance(value,dict) or value.get('kind') != 'regions':
        raise StableEditError('缺少 AI 选择的目标区域。')
    regions=value.get('regions')
    if not isinstance(regions,list) or not 1 <= len(regions) <= 4:
        raise StableEditError('一次局部操作需要 1–4 个明确目标区域。')
    regions=[_region_bounds(bounds,width,height) for bounds in regions]
    trigger=_region_bounds(value.get('trigger_bounds'),width,height)
    if event.get('type') != 'click' or any(type(event.get(k)) not in (int,float) or not math.isfinite(event[k]) for k in ('x','y')):
        raise StableEditError('目标区域需要原始点击依据。')
    if not trigger[0] <= event['x'] < trigger[2] or not trigger[1] <= event['y'] < trigger[3]:
        raise StableEditError('点击没有落在触发控件区域内。')
    mask=np.zeros((height,width),dtype=bool)
    for x0,y0,x1,y1 in regions:
        mask[y0:y1,x0:x1]=True
    if int(mask.sum())*100 > width*height*35:
        raise StableEditError('目标区域的总面积超过原图 35%。')
    return {'kind':'regions','regions':regions,'trigger_bounds':trigger}


def compose_effect_regions(original_png,candidate_png,edit):
    original,candidate=_decode_png(original_png),_decode_png(candidate_png)
    if original.shape != candidate.shape:
        raise StableEditError('局部候选与原图尺寸不同。')
    height,width=original.shape
    checked=validate_effect_regions(edit,{'width':width,'height':height},
        {'type':'click','x':edit['trigger_bounds'][0],'y':edit['trigger_bounds'][1]})
    mask=np.zeros(original.shape,dtype=bool)
    for x0,y0,x1,y1 in checked['regions']:
        mask[y0:y1,x0:x1]=True
    result=original.copy();result[mask]=candidate[mask]
    changed=int(np.count_nonzero(result[mask]!=original[mask]))
    if not changed:
        raise StableEditError('目标区域没有生成可见变化，原图与草稿已保留。')
    output=BytesIO();Image.fromarray(result).save(output,format='PNG',compress_level=2)
    delivered=_decode_png(output.getvalue())
    outside=int(np.count_nonzero(delivered[~mask]!=original[~mask]))
    if outside:
        raise StableEditError('目标区域外的像素发生变化。')
    regions=checked['regions']
    envelope=[min(r[0] for r in regions),min(r[1] for r in regions),max(r[2] for r in regions),max(r[3] for r in regions)]
    return output.getvalue(),{'kind':'regions','mode':'local_regions','regions':regions,'bounds':envelope,
        'trigger_bounds':checked['trigger_bounds'],'changed_pixels':changed,'outside_changed_pixels':outside,
        'outside_pixels':int((~mask).sum()),'edited_pixels':int(mask.sum()),'verified':True,
        'verification':'outside_pixels_exact','state_verified':False,
        'outside_sha256_before':hashlib.sha256(original[~mask].tobytes()).hexdigest(),
        'outside_sha256_after':hashlib.sha256(delivered[~mask].tobytes()).hexdigest()}
