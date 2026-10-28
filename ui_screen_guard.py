"""Detect measured settings switches leaking into a requested chat screen."""
import re
from stable_edit import _decode_png,_components,measure_toggle_state


def inspect_screen(png,prompt):
    chat=bool(re.search(r'\bchat\b|聊天',prompt[:80],re.I))
    explicit_switch=bool(re.search(r'\btoggles?\b|\bswitch(?:es)?\b|开关',prompt,re.I))
    if not chat or explicit_switch:return {'checked':False,'scope':'no chat-only constraint'}
    pixels=_decode_png(png);found=[]
    for component in _components(pixels<128):
        bounds=[component[k] for k in ('x0','y0','x1','y1')]
        w,h=bounds[2]-bounds[0],bounds[3]-bounds[1]
        if not (12<=w<=64 and 6<=h<=36 and 1.3<=w/h<=4.5):continue
        try:state=measure_toggle_state(png,bounds)
        except ValueError:continue
        if state is not None:found.append({'bounds':bounds,'state':state})
    return {'checked':True,'expected_screen':'chat','passed':not found,'unexpected_toggles':found,
        'scope':'measurable switches only; does not certify text or all layout details'}


def chat_repair_prompt(prompt):
    # Settings here is an entry to another page, never permission to insert its
    # switch controls into this chat. Keep the remaining visible state intact.
    return re.sub(r'\bSettings\b','Menu',prompt,flags=re.I)
