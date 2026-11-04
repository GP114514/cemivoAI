"""Exact user-content transfer between AI-chosen neural layouts.

The AI chooses field bindings from the old screenshot. Values come from the
committed frame, never from an AI transcription. This module generates no pixels
and cannot execute tools or publish a frame. Any mismatch refuses the candidate.
"""
from copy import deepcopy
import hashlib
from neural.text_head import text_layout


def contains(outer,inner):
    return outer[0]<=inner[0]<inner[2]<=outer[2] and outer[1]<=inner[1]<inner[3]<=outer[3]


def interior(element):
    x0,y0,x1,y1=element['bounds'];return [x0+4,y0+4,x1-4,y1-4]


def capture(frame,png):
    if (frame.get('generator')!='dsh_layout_neural_surface' or frame.get('credential_request') or
            frame.get('system_state') or frame.get('system_page')):
        raise ValueError('content-preserving redesign requires an ordinary neural app page')
    if hashlib.sha256(png).hexdigest()!=frame.get('sha256'):
        raise ValueError('source screenshot identity changed')
    elements=frame['neural_layout']['elements'];state=deepcopy(frame.get('state',{}))
    fields=[dict(source=i,kind=e['kind'],bounds=e['bounds'].copy(),text=e['text'])
        for i,e in enumerate(elements) if e['kind'] in ('input','panel')]
    def locate(kind,boxes):
        found=[f for f in fields if f['kind']==kind and all(contains(interior(f),box) for box in boxes)]
        if not boxes or len(found)!=1:raise ValueError('committed content has no unique source field')
        return found[0]
    composer=state.get('composer_bounds') or state.get('input_bounds')
    input_source=None;message_source=None;visible=[]
    if composer:
        field=locate('input',[composer]);input_source=field['source']
        field['text']=state.get('input_text','')
    elif state.get('input_text'):
        raise ValueError('draft has no committed source geometry')
    if state.get('sent_message_bounds'):
        field=locate('panel',state['sent_message_bounds']);message_source=field['source']
        visible=state.get('visible_sent_messages',state.get('sent_messages',[]))
        if (not isinstance(visible,list) or len(visible)!=len(state['sent_message_bounds']) or
                any(not isinstance(t,str) or not t.strip() for t in visible)):
            raise ValueError('message state is not aligned with the committed image')
        field['text']='\n'.join(visible)
    elif state.get('sent_messages'):
        raise ValueError('messages have no committed source geometry')
    for field in fields:
        value=field['text']
        if (not isinstance(value,str) or len(value)>500 or
                any(ord(c)<32 and c!='\n' for c in value)):
            raise ValueError('source content cannot be transferred losslessly')
    return {'source_sha256':frame['sha256'],'png':bytes(png),'state':state,
        'elements':deepcopy(elements),'fields':fields,'input_source':input_source,
        'message_source':message_source,'visible_messages':deepcopy(visible),
        'application':deepcopy(frame.get('application',{}))}


def planning_context(transfer):
    return {'source_sha256':transfer['source_sha256'],'previous_application':transfer['application'],
        'rendered_elements':transfer['elements'],'retained_fields':deepcopy(transfer['fields']),
        'rule':'Bind every retained source field exactly once. The host injects its exact committed value. Never invent, clear, paraphrase, send or discard it.'}


def bind(proposed,transfer):
    value=deepcopy(proposed)
    if not isinstance(value,dict):raise ValueError('redesign must be an object')
    bindings=value.pop('state_bindings',None);elements=value.get('elements')
    if not isinstance(elements,list) or not isinstance(bindings,list) or len(bindings)!=len(transfer['fields']):
        raise ValueError('state_bindings must map EVERY retained input/panel once')
    source={f['source']:f for f in transfer['fields']};seen=set();targets=set()
    for binding in bindings:
        if (not isinstance(binding,dict) or set(binding)!={'source','target'} or
                any(type(v) is not int for v in binding.values())):
            raise ValueError('each state binding needs integer source and target indices')
        old,new=binding['source'],binding['target']
        if old not in source or old in seen or not 0<=new<len(elements) or new in targets:
            raise ValueError('state binding is missing, duplicated or out of range')
        if not isinstance(elements[new],dict) or elements[new].get('kind')!=source[old]['kind']:
            raise ValueError('a retained field must bind to the same control kind')
        elements[new]['text']=source[old]['text'];seen.add(old);targets.add(new)
    if any(e.get('kind') in ('input','panel') and e.get('text') for i,e in enumerate(elements) if i not in targets):
        raise ValueError('unbound fields cannot invent user content')
    return value,deepcopy(bindings)


def restore(transfer,layout,bindings,vocabulary):
    """Rebuild tracked geometry to match the new actual neural text placement."""
    from neural.layout_surface import text_placement
    state=deepcopy(transfer['state'])
    for key in ('input_bounds','composer_bounds','sent_message_bounds','visible_sent_messages'):
        state.pop(key,None)
    targets={b['source']:b['target'] for b in bindings}
    for field in transfer['fields']:
        element=layout['elements'][targets[field['source']]]
        if element['text']!=field['text'] or element['kind']!=field['kind']:
            raise ValueError('target layout changed retained content')
        if field['source']==transfer['input_source']:
            state['composer_bounds']=interior(element)
            if field['text']:state['input_bounds']=interior(element)
        if field['source']==transfer['message_source']:
            left,top,width,height=text_placement(element)
            positions=text_layout(element['text'],width,height,vocabulary)
            bounds=[];offset=0
            for message in transfer['visible_messages']:
                length=sum(c!='\n' for c in message);cells=positions[offset:offset+length];offset+=length
                if not cells:raise ValueError('retained message has no visible text cells')
                box=[left+min(p[1] for p in cells),top+min(p[2] for p in cells),
                    left+max(p[1]+p[3] for p in cells),top+max(p[2]+18 for p in cells)]
                if not contains(interior(element),box):raise ValueError('retained message escaped its new panel')
                bounds.append(box)
            state['sent_message_bounds']=bounds
            state['visible_sent_messages']=deepcopy(transfer['visible_messages'])
    return state,{'verified':True,'source_sha256':transfer['source_sha256'],'field_count':len(bindings),
        'bindings':deepcopy(bindings),'exact_values_from':'committed_ui_state',
        'scope':'content_and_new_geometry; not external business completion'}
