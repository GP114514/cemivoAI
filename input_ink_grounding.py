"""Read-only ink bounds in an AI-selected input line; no OCR or rendering."""
import numpy as np
from stable_edit import _decode_png,_components,_component_mask

def ground_input_text(png,requested,outer):
    pixels=_decode_png(png);height,width=pixels.shape
    def valid(box):
        return (isinstance(box,list) and len(box)==4 and all(type(v) is int for v in box)
            and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height)
    if not valid(requested) or not valid(outer):raise ValueError('Invalid input ink bounds')
    def contains(a,b):return a[0]<=b[0]<b[2]<=a[2] and a[1]<=b[1]<b[3]<=a[3]
    if not contains(outer,requested):raise ValueError('Input line must stay inside selected effects')
    # This narrow correction addresses clipped glyph edges, never a new row.
    if requested[3]-requested[1]>36:return requested.copy(),{'expanded':False,'components':[]}
    dark=float(np.median(pixels))<128
    components=_components(pixels>=128 if dark else pixels<128)
    enclosures=[]
    for item in components:
        box=[item[key] for key in ('x0','y0','x1','y1')];w,h=box[2]-box[0],box[3]-box[1]
        if w<14 or h<10:continue
        mask=_component_mask(item)
        coverage=min(mask[:2].any(axis=0).mean(),mask[-2:].any(axis=0).mean(),
            mask[:,:2].any(axis=1).mean(),mask[:,-2:].any(axis=1).mean())
        if coverage>=.6 and item['area']>=(w+h)*.9:enclosures.append(box)
    matched=[]
    for item in components:
        box=[item[key] for key in ('x0','y0','x1','y1')];w,h=box[2]-box[0],box[3]-box[1]
        if not contains(outer,box) or not(1<=w<=64 and 2<=h<=18):continue
        if box[0]<requested[0]-16 or box[2]>requested[2]+16:continue
        vertical=max(0,min(box[3],requested[3])-max(box[1],requested[1]))
        if vertical/h<.6 or box[1]<requested[1]-3 or box[3]>requested[3]+3:continue
        # A neighboring button and its label are protected together. The big
        # input enclosure is not a neighboring control if it contains this row.
        if any(contains(enclosure,box) and not contains(enclosure,requested) for enclosure in enclosures):continue
        matched.append(box)
    result=[min([requested[0]]+[b[0] for b in matched]),min([requested[1]]+[b[1] for b in matched]),
        max([requested[2]]+[b[2] for b in matched]),max([requested[3]]+[b[3] for b in matched])]
    if not contains(outer,result):raise ValueError('Ink correction escaped selected effects')
    return result,{'expanded':result!=requested,'components':matched,'original_bounds':requested.copy(),
        'corrected_bounds':result.copy(),'scope':'pixel geometry only; not OCR or input semantics'}
