from copy import deepcopy
import hashlib
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock
from PIL import Image
from ui_state_transfer import capture,bind,restore,planning_context
from neural.layout_surface import validate_layout
from surface_navigation import validate_navigation,validate_redesign_state,NavigationRuntime
from live_app import LiveError


class StateTransferTests(unittest.TestCase):
    def setUp(self):
        self.png=b'identity fixture'
        self.elements=[{'kind':'text','bounds':[8,36,248,54],'text':'NOVA'},
            {'kind':'panel','bounds':[8,60,248,150],'text':''},
            {'kind':'input','bounds':[8,160,248,192],'text':''},
            {'kind':'button','bounds':[8,200,104,232],'text':'Send'}]
        self.frame={'generator':'dsh_layout_neural_surface','sha256':hashlib.sha256(self.png).hexdigest(),
            'neural_layout':{'elements':self.elements},'application':{'name':'NOVA'},
            'state':{'input_text':'Draft 42','input_bounds':[12,166,240,184],'composer_bounds':[12,166,240,184],
                'sent_messages':['Older hidden','Hello NOVA','Message 2'],'visible_sent_messages':['Hello NOVA','Message 2'],
                'sent_message_bounds':[[12,64,102,82],[12,84,93,102]],'view':'app','messages':[]}}
        self.layout={'screen_type':'chat','elements':deepcopy(self.elements)}
        # Move the target input/panel and reverse their element order.
        self.layout['elements'][1]={'kind':'input','bounds':[12,160,244,192],'text':'untrusted replacement'}
        self.layout['elements'][2]={'kind':'panel','bounds':[12,60,244,150],'text':'invented conversation'}
        self.layout['state_bindings']=[{'source':1,'target':2},{'source':2,'target':1}]
        self.vocabulary=set('NOVA Draft42HelloMessage2SendOlderhin')

    def test_exact_committed_values_replace_ai_transcriptions_and_rebuild_geometry(self):
        saved=deepcopy(self.frame);transfer=capture(self.frame,self.png)
        layout,bindings=bind(self.layout,transfer)
        self.assertEqual(layout['elements'][1]['text'],'Draft 42')
        self.assertEqual(layout['elements'][2]['text'],'Hello NOVA\nMessage 2')
        layout=validate_layout(layout,self.vocabulary)
        state,receipt=restore(transfer,layout,bindings,self.vocabulary)
        self.assertEqual(state['sent_messages'],self.frame['state']['sent_messages'])
        self.assertEqual(state['visible_sent_messages'],['Hello NOVA','Message 2'])
        self.assertEqual(state['input_bounds'],[16,164,240,188])
        self.assertEqual(state['sent_message_bounds'],[[16,64,106,82],[16,82,97,100]])
        self.assertTrue(receipt['verified']);self.assertEqual(self.frame,saved)
        self.assertNotIn('png',planning_context(transfer))

    def test_missing_duplicate_wrong_kind_boolean_and_invented_fields_are_refused(self):
        transfer=capture(self.frame,self.png)
        for bindings in ([],[{'source':1,'target':2}],[{'source':1,'target':2},{'source':1,'target':1}],
                [{'source':1,'target':1},{'source':2,'target':2}],
                [{'source':True,'target':2},{'source':2,'target':1}],
                [{'source':1,'target':2},{'source':2,'target':99}]):
            value=deepcopy(self.layout);value['state_bindings']=bindings
            with self.subTest(bindings=bindings),self.assertRaises(ValueError):bind(value,transfer)
        value=deepcopy(self.layout);value['elements'].append({'kind':'panel','bounds':[8,240,248,250],'text':'fake'})
        with self.assertRaises(ValueError):bind(value,transfer)

    def test_private_system_changed_source_and_untracked_content_cannot_transfer(self):
        for change in ({'credential_request':{'token':'never sent'}},{'system_page':'wifi'},
                {'system_state':{'wifi':{}}},{'sha256':'wrong'},
                {'state':{'input_text':'Draft without geometry'}},
                {'state':{'sent_messages':['Lost geometry']}}):
            with self.subTest(change=change),self.assertRaises(ValueError):capture(dict(self.frame,**change),self.png)

    def test_retained_text_still_requires_trained_glyphs_and_sufficient_space(self):
        transfer=capture(self.frame,self.png);layout,bindings=bind(self.layout,transfer)
        with self.assertRaises(ValueError):validate_layout(layout,set('NOVA Send'))
        layout['elements'][2]['bounds']=[12,60,64,86]
        with self.assertRaises(ValueError):validate_layout(layout,self.vocabulary)
        layout['elements'][1]['text']='Changed after binding'
        with self.assertRaises(ValueError):restore(transfer,layout,bindings,self.vocabulary)

    def test_navigation_can_only_preserve_during_explicit_redesign(self):
        plan={'requires_tools':False,'should_update':True,'update_mode':'global','target_type':'other',
            'navigation':{'action':'open_app','target_app':'NOVA','redraw':True,'preserve_state':True}}
        self.assertTrue(validate_navigation(plan,plan)['preserve_state'])
        ambiguous={'action':'open_app','target_app':'NOVA','redraw':True}
        with self.assertRaises(LiveError) as caught:validate_redesign_state(ambiguous,self.frame)
        self.assertEqual(caught.exception.code,'redesign_state_required')
        for decision in (True,False):validate_redesign_state(dict(ambiguous,preserve_state=decision),self.frame)
        for navigation in ({'action':'home','preserve_state':True},
                {'action':'open_app','target_app':'NOVA','preserve_state':True},
                {'action':'open_app','target_app':'NOVA','redraw':True,'preserve_state':'yes'}):
            value=dict(plan,navigation=navigation)
            with self.assertRaises(LiveError):validate_navigation(value,value)

    def test_failed_redesign_cannot_publish_state_or_advance_navigation(self):
        output=BytesIO();Image.new('L',(256,256),255).save(output,format='PNG');png=output.getvalue()
        previous=deepcopy(self.frame);previous.update(sha256=hashlib.sha256(png).hexdigest(),
            session_id='b'*32,version=3,seed=42,steps=8,
            surface_navigation={'page_id':'old-page','page_kind':'app','app_name':'NOVA'})
        live=SimpleNamespace(planner=SimpleNamespace(dsh_backend=None),
            _draw_page=Mock(side_effect=LiveError('Cannot fit every retained value','layout_invalid',422)))
        runtime=NavigationRuntime(live);runtime.observe(previous,png)
        before=deepcopy(runtime.sessions)
        plan={'requires_tools':False,'should_update':True,'update_mode':'global','target_type':'other',
            'next_prompt':'NOVA Send Reset','intent':'Redesign',
            'navigation':{'action':'open_app','target_app':'NOVA','redraw':True,'preserve_state':True}}
        with self.assertRaises(LiveError):runtime.render(previous,png,plan,{'type':'click','text':'Keep all content'})
        self.assertEqual(runtime.sessions,before);self.assertFalse(runtime.pending)
        transfer=live._draw_page.call_args.kwargs['state_transfer']
        self.assertEqual(transfer['state']['input_text'],'Draft 42')
        self.assertEqual(previous['state']['input_text'],'Draft 42')


if __name__=='__main__':unittest.main()
