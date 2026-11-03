from copy import deepcopy
from io import BytesIO
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch
import hashlib
import json
import unittest
from PIL import Image

from initial_planning import _validate,InitialPlanError,plan_application
from live_app import LiveApp,LiveError
from neural.layout_surface import validate_prompt_contract
from surface_navigation import NavigationRuntime,validate_navigation


def image():
    out=BytesIO();Image.new('L',(256,256),255).save(out,format='PNG');return out.getvalue()


def spec():
    return {'name':'NOVA','purpose':'A chat interface','screen_type':'chat','layout':['title and empty chat'],
        'controls':[{'label':'Send','purpose':'Send input'},{'label':'Clear','purpose':'Clear draft'}],
        'interaction_rules':['Keep messages empty initially'],'agent_tasks':[],
        'initial_prompt':'Monochrome NOVA chat, empty messages, empty input, Send, Clear',
        'initial_screen':{'title':'NOVA','required_labels':['Send','Clear'],'empty_inputs':1,'empty_panels':1}}


def frame(png):
    return {'image_url':'/api/live/images/'+'a'*32+'.png','sha256':hashlib.sha256(png).hexdigest(),
        'width':256,'height':256,'seed':42,'steps':8,'prompt':'NOVA',
        'surface_chrome':{'bounds':[0,0,256,32]},'generator':'dsh_layout_neural_surface',
        'page_review':{'accepted':True,'attempts':1,'review_ms':12,'checks':[],
            'scope':'screen_type_empty_state_and_main_control_shapes','readability':'unknown'},
        'timings':{'sample_ms':2,'page_review_ms':12,'total_ms':14}}


class PromptGenerationTests(unittest.TestCase):
    def test_default_dsh_create_uses_full_goal_and_neural_page_once(self):
        with TemporaryDirectory() as root:
            png=image();planner=SimpleNamespace(dsh_backend=object(),public=lambda:{'provider':'DSH','model':'fixture'})
            live=LiveApp(root,renderer=object(),planner=planner)
            live.navigation=SimpleNamespace(status=lambda:{'available':True},
                add_chrome=Mock(side_effect=AssertionError('navigation must not be drawn twice')),observe=Mock())
            live._draw=Mock(side_effect=AssertionError('old PixelFlow initial path must not run'))
            live._draw_page=Mock(return_value=(png,frame(png)))
            goal='Full original requirement. '*12
            with patch('initial_planning.plan_application',return_value=spec()) as planning:
                result=live.create({'prompt':goal,'seed':42,'steps':8})
            planning.assert_called_once_with(planner,goal)
            self.assertEqual(live._draw_page.call_args.args[3],goal)
            self.assertEqual(live._draw_page.call_args.kwargs['application_spec'],spec())
            self.assertEqual(result['application']['goal'],goal)
            self.assertEqual(result['application']['spec']['initial_screen'],spec()['initial_screen'])
            self.assertTrue(result['initial_review']['accepted'])
            self.assertEqual(result['initial_plan']['provider'],'DSH')

    def test_failed_dsh_page_never_replaces_the_active_frame(self):
        with TemporaryDirectory() as root:
            live=LiveApp(root,renderer=object(),planner=SimpleNamespace(dsh_backend=object()))
            live.active_session_id='old-session'
            live._draw_page=Mock(side_effect=LiveError('Required control missing','layout_invalid',422))
            with patch('initial_planning.plan_application',return_value=spec()),self.assertRaises(LiveError):
                live.create({'prompt':'Chat'})
            self.assertEqual(live.active_session_id,'old-session')
            self.assertFalse(live.sessions)
            self.assertFalse(live.generation_lock.locked())

    def test_missing_literal_and_nonempty_draft_are_rejected(self):
        layout={'elements':[{'kind':'text','text':'NOVA'}, {'kind':'button','text':'Send'},
            {'kind':'button','text':'Clear'},{'kind':'input','text':''},{'kind':'panel','text':''}]}
        self.assertTrue(validate_prompt_contract(layout,spec())['layout_contract_passed'])
        for changed in ('label','draft','title'):
            value=deepcopy(layout)
            if changed=='label':value['elements'][2]['text']='Reset'
            elif changed=='draft':value['elements'][3]['text']='Hi'
            else:value['elements'][0]['kind']='button'
            with self.subTest(changed=changed),self.assertRaises(ValueError):validate_prompt_contract(value,spec())

    def test_initial_contract_is_validated_and_legacy_plans_still_work(self):
        self.assertEqual(_validate(spec())['initial_screen'],spec()['initial_screen'])
        old=spec();old.pop('initial_screen');self.assertEqual(_validate(old),old)
        value=spec();value['initial_screen']['empty_inputs']=True
        with self.assertRaises(InitialPlanError):_validate(value)

    def test_new_dsh_planning_cannot_silently_omit_the_contract(self):
        old=spec();old.pop('initial_screen')
        planner=SimpleNamespace(dsh_backend=object(),complete=lambda *args,**kwargs:{'choices':[
            {'finish_reason':'stop','message':{'content':json.dumps(old)}}]})
        with self.assertRaises(InitialPlanError) as caught:plan_application(planner,'NOVA with Send and Clear')
        self.assertEqual(caught.exception.code,'initial_contract_missing')

    def test_explicit_redesign_bypasses_cache_and_keeps_complete_text(self):
        png=image();base=frame(png);base.update(session_id='b'*32,version=1,
            surface_navigation={'page_id':'page-old','page_kind':'app','app_name':'NOVA'},
            state={'view':'app','messages':[]},application={'id':'prior','goal':'Old design'})
        live=SimpleNamespace(planner=SimpleNamespace(dsh_backend=None),_draw_page=Mock(return_value=(png,frame(png))))
        navigation=NavigationRuntime(live);navigation.observe(base,png)
        plan={'navigation':{'action':'open_app','target_app':'NOVA','redraw':True},'requires_tools':False,
            'should_update':True,'update_mode':'global','target_type':'other','next_prompt':'New NOVA chat','intent':'Redesign'}
        original='Create NOVA with Send and Clear; preserve every word in this full description.'
        _,result=navigation.render(base,png,plan,{'type':'click','text':original})
        self.assertFalse(result['surface_navigation']['restored'])
        self.assertIn(original,live._draw_page.call_args.args[3])
        self.assertIn(original,result['application']['goal'])
        self.assertNotEqual(result['surface_navigation']['page_id'],'page-old')
        plan['navigation'].pop('redraw');live._draw_page.reset_mock()
        _,restored=navigation.render(base,png,plan,{'type':'click','text':''})
        self.assertTrue(restored['surface_navigation']['restored']);live._draw_page.assert_not_called()

    def test_navigation_redraw_is_ai_boolean_for_open_only(self):
        plan={'requires_tools':False,'should_update':True,'update_mode':'global','target_type':'other',
            'navigation':{'action':'open_app','target_app':'NOVA','redraw':True}}
        self.assertTrue(validate_navigation(plan,plan)['redraw'])
        plan['navigation']['redraw']='true'
        with self.assertRaises(LiveError):validate_navigation(plan,plan)
        plan['navigation']={'action':'home','redraw':True}
        with self.assertRaises(LiveError):validate_navigation(plan,plan)


if __name__=='__main__':unittest.main()
