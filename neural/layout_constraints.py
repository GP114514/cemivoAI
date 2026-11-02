"""Fit AI-chosen rectangles to native glyph sizes; never choose/draw controls."""
from copy import deepcopy

from .text_head import character_advance,text_layout


def fit_layout(value,vocabulary):
    """Keep every AI element/text, reflow only geometry, or refuse the page.

    Only a new whole-page candidate uses this helper. Existing screenshots and
    local editing masks are never resized/repacked. There are no app categories,
    button names, routes, image templates or pixel operations in this module.
    """
    if not isinstance(value,dict) or set(value)!={'screen_type','elements'}:
        raise ValueError('not an AI layout')
    if not isinstance(value['elements'],list) or not 2<=len(value['elements'])<=10:
        raise ValueError('invalid element count')
    items=[]
    for index,element in enumerate(value['elements']):
        if not isinstance(element,dict) or set(element)!={'kind','bounds','text'}:
            raise ValueError('invalid element schema')
        kind=element['kind'];text=element['text'];bounds=element['bounds']
        if kind not in ('text','button','input','panel') or not isinstance(text,str) or len(text)>80:
            raise ValueError('unsupported AI element')
        if (any(c not in vocabulary for c in text if c!='\n') or any(ord(c)<32 and c!='\n' for c in text)
                or (kind=='text' and not text.strip())):raise ValueError('untrained or empty text')
        if not isinstance(bounds,list) or len(bounds)!=4 or any(type(v) is not int for v in bounds):
            raise ValueError('invalid native bounds')
        x0,y0,x1,y1=bounds
        if not(0<=x0<x1<=256 and 0<=y0<y1<=256):raise ValueError('invalid source geometry')
        padding=0 if kind=='text' else 4
        required=max((sum(character_advance(c) for c in line) for line in text.split('\n')),default=0)
        width=min(244,max(x1-x0,required+padding*2,16 if padding else 1))
        height=max(y1-y0,26 if padding else 18)
        if text:
            # Work out the native wrapped height without changing literal text.
            for candidate in range(18,217,18):
                try:text_layout(text,width-padding*2,candidate,vocabulary);break
                except ValueError:continue
            else:raise ValueError('text cannot fit the canvas')
            height=max(height,candidate+padding*2)
        items.append({'index':index,'width':width,'height':height,'top':y0,'left':x0})
    ordered=sorted(items,key=lambda item:(item['top'],item['left']))
    groups=[]
    for item in ordered:
        if not groups or item['top']>groups[-1][0]['top']+8:groups.append([])
        groups[-1].append(item)

    def pack(compact):
        result=deepcopy(value);bottom=32
        for group in groups:
            top=bottom+4 if compact else max(bottom+4,36,group[0]['top'])
            left=6;row_height=0
            for item in group:
                if left+item['width']>250:
                    top+=row_height+4;left=6;row_height=0
                if top+item['height']>250:raise ValueError('AI controls cannot fit without removing content')
                result['elements'][item['index']]['bounds']=[left,top,left+item['width'],top+item['height']]
                left+=item['width']+4;row_height=max(row_height,item['height'])
            bottom=top+row_height
        from .layout_surface import validate_layout
        return validate_layout(result,vocabulary)
    try:return pack(False)
    except ValueError:return pack(True)
