from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
import numpy as np
from PIL import Image
from draft_clear import validate_clear,render_clear
from interaction_policy import visual_state,response_fields,response_needs_revision
from live_app import DeepSeekPlanner,LiveError


def encode(pixels):
    out=BytesIO();Image.fromarray(pixels).save(out,format='PNG');return out.getvalue()


class DraftClearTests(unittest.TestCase):
    def setUp(self):
        self.bounds=[16,180,106,198]
        self.state={'input_text':'Draft only','input_bounds':self.bounds.copy(),'composer_bounds':self.bounds.copy(),
            'sent_messages':['Hello NOVA'],'sent_message_bounds':[[16,80,106,98]]}
        self.frame={'width':256,'height':256,'state':deepcopy(self.state),'generator':'dsh_layout_neural_surface',
            'neural_layout':{'elements':[{'kind':'panel','bounds':[8,60,248,164],'text':''},
                {'kind':'input','bounds':[8,172,248,208],'text':''}]}}
        self.event={'type':'click','x':201,'y':231,'input_method':'pointer'}
        self.plan={'intent':'Clear draft','expected_change':'The input is empty; prior messages remain',
            'state_summary':'Empty draft','should_update':True,'next_prompt':'NOVA chat, empty input, keep messages',
            'update_mode':'local','target_type':'other','scope_reason':'Only draft changes','requires_tools':False,
            'text_action':'clear','input_text':'','text_source':'draft','clear_input_bounds':self.bounds.copy(),
            'local_edit':{'kind':'regions','trigger_bounds':[180,216,248,246],'regions':[[12,176,244,204]]}}

    def test_clear_is_a_valid_local_visual_action(self):
        checked=DeepSeekPlanner.validate_auto_plan(self.plan,self.frame,self.event)
        self.assertEqual(checked['clear_input_bounds'],self.bounds)
        self.assertFalse(response_needs_revision(checked,self.event,self.state))
        self.assertEqual(response_fields(self.plan)['input_text'],'')

    def test_cannot_clear_a_message_or_execute_business_work(self):
        for key,value in [('requires_tools',True),('clear_input_bounds',[16,80,106,98]),('input_text','changed')]:
            bad=deepcopy(self.plan);bad[key]=value
            with self.subTest(key=key),self.assertRaises(LiveError):validate_clear(bad,self.frame,self.event)

    def test_actual_render_erases_only_the_draft_and_preserves_borders(self):
        pixels=np.full((256,256),255,dtype=np.uint8);pixels[83:93,17:103]=0;pixels[183:193,17:103]=0
        original=encode(pixels)
        def blank(png,bounds,text):
            self.assertEqual(bounds,self.bounds);self.assertEqual(text,' ')
            out=pixels.copy();a,b,c,d=bounds;out[b:d,a:c]=255
            return encode(out),{'sample_ms':1,'editor_model':{'model_sha256':'fixture'}}
        live=SimpleNamespace(text_editor=SimpleNamespace(render=blank),planner=SimpleNamespace(verify_region=Mock(return_value=True)))
        result,frame=render_clear(live,original,self.frame,self.plan,self.event)
        after=np.array(Image.open(BytesIO(result)))
        mask=np.zeros((256,256),dtype=bool);a,b,c,d=self.bounds;mask[b:d,a:c]=True
        self.assertTrue(np.array_equal(pixels[~mask],after[~mask]))
        self.assertTrue(np.all(after[mask]==255))
        self.assertEqual(frame['edit']['outside_changed_pixels'],0)
        state=visual_state(self.state,self.plan,frame['edit'])
        self.assertEqual(state['input_text'],'');self.assertEqual(state['sent_messages'],['Hello NOVA'])
        self.assertEqual(state['composer_bounds'],self.bounds)

    def test_unverified_clear_does_not_discard_logical_draft(self):
        self.assertEqual(visual_state(self.state,self.plan,{'draft_cleared':True,'preview':True}),self.state)
        self.assertEqual(visual_state(self.state,self.plan,None),self.state)


if __name__=='__main__':unittest.main()
