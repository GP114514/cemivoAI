"""Read-only closed button geometry; no names, actions or preset locations."""
import numpy as np
from stable_edit import _decode_png, _components, _component_mask


def find_button_at(png,event,frame):
    pixels=_decode_png(png)
    x,y=event['x'],event['y']
    candidates=[]
    for component in _components(pixels < 128 if float(np.median(pixels)) >= 128 else pixels >= 128):
        x0,y0,x1,y1=(component[k] for k in ('x0','y0','x1','y1'))
        width,height=x1-x0,y1-y0
        if not (x0 <= x < x1 and y0 <= y < y1 and 14<=width<=160 and 10<=height<=64
                and .8<=width/height<=10 and width*height<=pixels.size*.2):continue
        mask=_component_mask(component)
        if min(mask[:2].any(axis=0).mean(),mask[-2:].any(axis=0).mean(),
                mask[:,:2].any(axis=1).mean(),mask[:,-2:].any(axis=1).mean()) < .50:continue
        # Text clusters, open brackets and lines do not form a four-sided box.
        if component['area'] < (width+height)*.9:continue
        candidates.append([x0,y0,x1,y1])
    if not candidates:return None
    bounds=min(candidates,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
    return {'bounds':bounds,'source':'closed_pixel_component','label':None,
        'note':'Pixel enclosure only; the AI must read its label and infer its purpose from the full screenshot.'}
