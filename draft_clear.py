"""A visually interpreted Clear action edits only a tracked input draft."""
from copy import deepcopy
import hashlib,secrets,time


def validate_clear(plan,frame,event):
    from live_app import LiveError
    from behavior_regions import contains,validate_effect_regions
    from stable_edit import validate_region_plan
    from neural.layout_surface import text_interior
    try:
        state=frame.get('state',{})
        bounds=state.get('input_bounds')
        if (plan.get('text_action')!='clear' or plan.get('input_text')!='' or plan.get('text_source')!='draft'
                or plan.get('requires_tools') is not False or plan.get('update_mode')!='local'
                or plan.get('target_type')!='other' or not state.get('input_text')
                or not isinstance(bounds,list) or len(bounds)!=4 or any(type(v) is not int for v in bounds)):
            raise ValueError('clear requires an actual tracked draft and a visual-only local plan')
        if plan.get('clear_input_bounds') not in (None,bounds):
            raise ValueError('clear must retain the exact tracked draft bounds')
        raw=plan['local_edit']
        edit=(validate_effect_regions if raw.get('kind')=='regions' else validate_region_plan)(raw,frame,event)
        regions=edit.get('regions',[edit.get('bounds')])
        if not any(contains(region,bounds) for region in regions):raise ValueError('draft is outside selected effects')
        text_interior(frame,'input',bounds)
        a,b,c,d=bounds
        if any(a<x1 and x0<c and b<y1 and y0<d for x0,y0,x1,y1 in state.get('sent_message_bounds',[])):
            raise ValueError('draft overlaps prior messages')
        return bounds.copy()
    except (KeyError,TypeError,ValueError) as error:
        raise LiveError('清空操作必须只覆盖已跟踪的输入草稿，原图已保留。','invalid_clear_scope',422) from error


def render_clear(live,png,previous,plan,event):
    from live_app import LiveError
    from behavior_regions import compose_effect_regions
    from stable_edit import compose_region
    from neural.layout_surface import verify_borders
    bounds=validate_clear(plan,previous,event)
    try:candidate,timings=live.text_editor.render(png,bounds,' ')
    except ValueError as error:raise LiveError('神经模型未能清空当前草稿，原图已保留。','clear_unavailable',422) from error
    edit=plan['local_edit']
    delivered,evidence=(compose_effect_regions if edit['kind']=='regions' else compose_region)(png,candidate,edit)
    verify_borders(png,delivered,previous)
    started=time.perf_counter()
    if live.planner.verify_region(png,delivered,edit,plan['expected_change']) is not True:
        raise LiveError('清空草稿未通过视觉核验，原图与草稿已保留。','clear_unverified',422)
    used_model=timings.pop('editor_model')
    frame=deepcopy(previous)
    for key in ('agent','scope','edit'):frame.pop(key,None)
    frame.update(image_url='/api/live/images/'+secrets.token_hex(16)+'.png',
        sha256=hashlib.sha256(delivered).hexdigest(),
        edit=dict(evidence,generator='native_neural_text_head',draft_cleared=True,
            clear_input_bounds=bounds,vision_verified=True,preview=False,text_applied=True,
            editor_model=used_model,verification_source='fresh_vision+tracked_draft_bounds',semantics_guaranteed=False),
        timings=dict(timings,verify_ms=round((time.perf_counter()-started)*1000,2)))
    return delivered,frame
