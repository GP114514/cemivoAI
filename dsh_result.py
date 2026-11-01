"""Render a receipt summary through the trained text head in an AI-chosen blank ROI."""
from io import BytesIO
import hashlib
import numpy as np
from PIL import Image
from live_app import LiveError

def blank_candidates(png,previous):
    pixels=np.asarray(Image.open(BytesIO(png)));h,w=pixels.shape
    if w<144 or h<26:return []
    boxes=[]
    reserved=previous.get('dsh_receipt',{})
    old=reserved.get('bounds')
    if isinstance(old,list) and len(old)==4 and all(type(v) is int for v in old):
        x0,y0,x1,y1=old
        if 0<=x0<x1<=w and 0<=y0<y1<=h and hashlib.sha256(pixels[y0:y1,x0:x1].tobytes()).hexdigest()==reserved.get('region_sha256'):
            boxes.append(old)
    for y in sorted(range(4,h-22,2),key=lambda y:abs((y+9)-h/2)):
        bounds=[8,y,w-8,y+18];region=pixels[y:y+18,8:w-8]
        background=int(np.median(region))
        if background not in (0,255) or np.any(np.abs(region.astype(int)-background)>12):continue
        if any(bounds[0]<b[2] and b[0]<bounds[2] and bounds[1]<b[3] and b[1]<bounds[3]
            for b in previous.get('state',{}).get('sent_message_bounds',[])):continue
        if any(abs(y-b[1])<18 for b in boxes):continue
        boxes.append(bounds)
        if len(boxes)==3:break
    return boxes

def draw_receipt(live,png,previous,plan,receipts):
    bounds=plan.get('dsh_result_bounds')
    if not receipts or any(r.get('backend')!='dsh' for r in receipts):
        raise LiveError('缺少 DSH 的实际工具回执。','dsh_receipt_missing',422)
    if any(r.get('status') not in ('pending','complete','failed') for r in receipts):
        raise LiveError('DSH 回执状态未确认。','dsh_receipt_missing',422)
    if not isinstance(bounds,list) or len(bounds)!=4 or any(type(x) is not int for x in bounds):
        raise LiveError('DSH 结果区域格式不正确。','dsh_result_bounds',422)
    before=np.asarray(Image.open(BytesIO(png)))
    h,w=before.shape;x0,y0,x1,y1=bounds
    if not(0<=x0<x1<=w and 0<=y0<y1<=h) or (x1-x0)*(y1-y0)>w*h*.1 or y1-y0<18 or x1-x0<64:
        raise LiveError('DSH 结果区域须为画面内的小范围文字行。','dsh_result_bounds',422)
    region=before[y0:y1,x0:x1];background=int(np.median(region))
    reserved=previous.get('dsh_receipt',{})
    reusable=(bounds==reserved.get('bounds') and hashlib.sha256(region.tobytes()).hexdigest()==reserved.get('region_sha256'))
    if not reusable and (background not in (0,255) or np.count_nonzero(np.abs(region.astype(int)-background)>12)>region.size*.01):
        raise LiveError('DSH 选择的结果区域不是空白，原控件已保留。','dsh_result_occupied',422)
    for old in previous.get('state',{}).get('sent_message_bounds',[]):
        if x0<old[2] and old[0]<x1 and y0<old[3] and old[1]<y1:
            raise LiveError('结果区域不能覆盖已有消息。','dsh_result_overlap',422)
    status=[r.get('status') for r in receipts]
    count=min(99999,max(0,int(previous.get('dsh_receipt',{}).get('count',0))))+1
    text=('待确认' if 'pending' in status else '部分失败' if 'failed' in status else '已执行')+' '+str(count)
    output,timings=live.text_editor.render(png,bounds,text)
    after=np.asarray(Image.open(BytesIO(output)));mask=np.zeros_like(before,dtype=bool);mask[y0:y1,x0:x1]=True
    outside=int(np.count_nonzero((before!=after)&~mask));changed=int(np.count_nonzero(before!=after))
    if outside or not changed:raise LiveError('DSH 结果区域像素核验未通过。','dsh_result_pixels',422)
    return output,{'sha256':hashlib.sha256(output).hexdigest(),'timings':timings,'prompt':previous['prompt'],
        'edit':{'kind':'dsh_receipt','bounds':bounds,'outside_changed_pixels':outside,'changed_pixels':changed,
            'result_text':text,'source':'neural_text_head_from_actual_dsh_receipts','preview':False},
        'dsh_receipt':{'text':text,'bounds':bounds,'tools':len(receipts),'count':count,
            'region_sha256':hashlib.sha256(after[y0:y1,x0:x1].tobytes()).hexdigest()}}
