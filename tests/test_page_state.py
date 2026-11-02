from collections import OrderedDict
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import unittest
from PIL import Image

from agent_runtime import AgentRuntime
from live_app import DeepSeekPlanner,LiveApp,Session
from visual_intent_memory import Memory


def blank_png():
    output=BytesIO();Image.new('L',(256,256),255).save(output,format='PNG');return output.getvalue()


class PageStateTests(unittest.TestCase):
    def test_returning_home_drops_old_layout_and_receipt(self):
        with TemporaryDirectory() as directory:
            png=blank_png();image_id='a'*32;session_id='b'*32
            previous={'session_id':session_id,'version':1,'width':256,'height':256,
                'sha256':hashlib.sha256(png).hexdigest(),'image_url':'/api/live/images/'+image_id+'.png',
                'prompt':'chat','seed':42,'steps':8,'last_event':None,
                'generator':'dsh_layout_neural_surface','neural_layout':{'screen_type':'chat'},
                'dsh_receipt':{'text':'old page result'},'state':{'view':'app','messages':[]}}
            plan={'intent':'Home','expected_change':'Return to desktop','state_summary':'desktop',
                'should_update':True,'next_prompt':'Desktop','requires_tools':False,
                'navigation':{'action':'home'},'update_mode':'global','target_type':'other','scope_reason':'Home navigation'}
            planner=SimpleNamespace(public=lambda:{'configured':True},infer_auto=lambda *args,**kwargs:deepcopy(plan))
            live=LiveApp(directory,renderer=object(),planner=planner)
            live.sessions[session_id]=Session(previous,[previous],[],OrderedDict());live._store_image(previous,png)
            desktop={'width':256,'height':256,'sha256':previous['sha256'],
                'image_url':'/api/live/images/'+'c'*32+'.png','prompt':'Desktop','generator':'native_neural_surface',
                'seed':42,'steps':8,'state':{'view':'desktop','messages':[]},'timings':{},
                'surface_navigation':{'page_id':'desktop','page_kind':'desktop'}}
            live.navigation=SimpleNamespace(render=lambda *args:(png,deepcopy(desktop)),observe=lambda *args:None)
            result=AgentRuntime(directory,live).event({'session_id':session_id,'version':1,'request_id':'home-test-001',
                'interaction_mode':'auto','event':{'type':'click','x':32,'y':16}})
            self.assertEqual(result['agent']['status'],'complete')
            self.assertNotIn('neural_layout',result);self.assertNotIn('dsh_receipt',result)
            self.assertEqual(result['state']['view'],'desktop')

    def test_visual_noop_preserves_desktop_view(self):
        with TemporaryDirectory() as directory:
            png=blank_png();session_id='b'*32
            frame={'session_id':session_id,'version':1,'width':256,'height':256,
                'sha256':hashlib.sha256(png).hexdigest(),'image_url':'/api/live/images/'+'a'*32+'.png',
                'prompt':'Desktop','seed':42,'steps':8,'last_event':None,'state':{'view':'desktop','messages':[]}}
            plan={'intent':'Already on desktop','should_update':False,'next_prompt':'','requires_tools':False,
                'update_mode':'none','target_type':'other','scope_reason':'At the requested view','no_change_reason':'at_target'}
            planner=SimpleNamespace(public=lambda:{'configured':True},infer_auto=lambda *args,**kwargs:plan)
            live=LiveApp(directory,renderer=object(),planner=planner)
            live.sessions[session_id]=Session(frame,[frame],[],OrderedDict());live._store_image(frame,png)
            result=AgentRuntime(directory,live).event({'session_id':session_id,'version':1,'request_id':'noop-test-001',
                'interaction_mode':'auto','event':{'type':'click','x':128,'y':128}})
            self.assertEqual(result['state']['view'],'desktop')

    def test_verified_message_area_is_clipped_to_neural_panel_interior(self):
        png=blank_png();digest=hashlib.sha256(png).hexdigest();memory=Memory()
        base={'session_id':'b'*32,'version':1,'application':{'id':'chat-app'},'width':256,'height':256,
            'sha256':digest,'generator':'dsh_layout_neural_surface',
            'neural_layout':{'elements':[{'kind':'panel','bounds':[8,82,248,168],'text':''},
                {'kind':'input','bounds':[8,176,176,212],'text':''}]},
            'state':{'view':'app','messages':[]}}
        composer=[12,184,120,202];trigger=[184,176,248,212]
        def edited(frame,version,state,bounds,regions):
            result=deepcopy(frame);result.update(version=version,state=state,
                edit={'text_applied':True,'preview':False,'outside_changed_pixels':0,
                    'generator':'native_neural_text_head','text_bounds':bounds,'regions':regions,
                    'editor_model':{'model_sha256':'model-hash'}})
            return result
        entered=edited(base,2,{'view':'app','messages':[],'input_text':'Hello CEMIVO',
            'input_bounds':composer,'composer_bounds':composer},composer,[composer])
        input_plan={'text_action':'input','input_text':'Hello CEMIVO','update_mode':'local',
            'target_type':'other','requires_tools':False,'recognition':{'full_image_used':True},
            'local_edit':{'kind':'region','bounds':[12,180,172,208]}}
        self.assertTrue(memory.observe(base,entered,png,input_plan,{}))
        message=[18,92,140,110]
        sent=edited(entered,3,{'view':'app','messages':[],'input_text':'','composer_bounds':composer,
            'sent_messages':['Hello CEMIVO'],'sent_message_bounds':[message]},message,[message,composer,[32,54,202,80]])
        send_plan={'text_action':'send','input_text':'Hello CEMIVO','update_mode':'local',
            'target_type':'other','requires_tools':False,'recognition':{'full_image_used':True},
            'local_edit':{'kind':'regions','trigger_bounds':trigger},'message_area_bounds':[8,82,248,168]}
        self.assertTrue(memory.observe(entered,sent,png,send_plan,{}))
        # A short second draft on the same confirmed frame may reuse the policy.
        planner=DeepSeekPlanner(reuse_ui_intents=True)
        event={'type':'click','x':90,'y':194,'input_method':'contextmenu','text':'Fast CEMIVO'}
        with patch.object(memory,'_literal',return_value=True):
            compiled=memory.plan(planner,png,sent,event)
            self.assertIsNotNone(compiled)
            second=edited(sent,4,dict(sent['state'],input_text='Fast CEMIVO',input_bounds=composer),composer,[composer])
            self.assertTrue(memory.observe(sent,second,png,compiled,event))
            result=memory.plan(planner,png,second,{'type':'click','x':216,'y':194,'input_method':'pointer'})
        self.assertIsNotNone(result)
        self.assertEqual(result['text_action'],'send')
        self.assertEqual(result['text_bounds'][2],244)
        self.assertTrue(memory.authorize(result,png,second,{'type':'click','x':216,'y':194,'input_method':'pointer'}))


if __name__=='__main__':unittest.main()
