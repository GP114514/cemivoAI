from copy import deepcopy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import unittest
from unittest.mock import Mock,patch

from PIL import Image
from agent_runtime import AgentRuntime
from computer_control import ComputerControl,validate_action
from live_app import DeepSeekPlanner,LiveApp,LiveError,Session
from windows_host import HostError,WindowsHost
from prompt_budget import compact_plan_prompt


PROFILE='a'*24

class FakeHost:
    def __init__(self):self.calls=[];self.opens=[];self.denied=False
    def info(self):return {'connected':True,'name':'LOCAL-PC','ram_gb':48}
    def wifi(self,scan=False):return {'available':True,'interfaces':[{'state':'connected','profile_id':PROFILE,'ssid':'MeasuredWiFi'}],
        'networks':[{'profile_id':PROFILE,'connectable':True}],
        'profiles':[{'id':PROFILE,'name':'MeasuredWiFi','connected':True}], 'errors':[]}
    def connect(self,profile):
        self.calls.append(profile)
        if self.denied:raise HostError('windows_access_denied','Denied',5)
        return {'ok':True,'status':'already_connected','message':'Actual connection checked'}
    def open_settings(self,page):self.opens.append(page);return {'ok':True,'status':'settings_requested','page':page}
    error=staticmethod(WindowsHost.error)


class ComputerTests(unittest.TestCase):
    def test_only_allowlisted_actions_and_fields_are_accepted(self):
        for value in ({'action':'shell','command':'anything'}, {'action':'settings','page':'ms-settings:other'},
                {'action':'wifi_connect','profile_id':'name & command'}, {'action':'inspect','page':'computer','password':'no'}):
            with self.subTest(value=value),self.assertRaises(ValueError):validate_action(value)

    def test_connect_target_must_be_pictured_or_named_and_saved(self):
        host=FakeHost();computer=ComputerControl(host)
        request={'action':'wifi_connect','profile_id':PROFILE}
        self.assertFalse(computer.invoke(request,{}, {'text':'Connect something'})['ok'])
        self.assertEqual(host.calls,[])
        self.assertTrue(computer.invoke(request,{}, {'text':'Connect MeasuredWiFi'})['ok'])
        page={'system_state':computer.snapshot()}
        self.assertTrue(computer.invoke(request,page, {'type':'click','text':''})['ok'])
        self.assertEqual(len(host.calls),2)
        self.assertFalse(computer.invoke(dict(request,profile_id='b'*24),page,{'text':''})['ok'])
        self.assertEqual(len(host.calls),2)

    def test_native_access_denial_is_a_failure_not_mock_success(self):
        host=FakeHost();host.denied=True;computer=ComputerControl(host)
        result=computer.invoke({'action':'wifi_connect','profile_id':PROFILE},{},{'text':'Connect MeasuredWiFi'})
        self.assertFalse(result['ok']);self.assertEqual(result['error']['native_code'],5)
        self.assertEqual(len(host.calls),1)

    def test_explicit_no_connect_is_enforced_even_if_ai_proposes_it(self):
        host=FakeHost();computer=ComputerControl(host)
        result=computer.invoke({'action':'wifi_connect','profile_id':PROFILE},{},{'text':'仅选择 MeasuredWiFi，不要连接'})
        self.assertFalse(result['ok']);self.assertEqual(host.calls,[])

    def test_real_wifi_cannot_silently_route_to_a_mock_application(self):
        plan={'navigation':{'action':'open_app','target_app':'Wi-Fi 状态'},'requires_tools':False}
        self.assertTrue(ComputerControl.needs_routing(plan,{}, {'text':'显示本机真实 Wi-Fi 状态'}))
        self.assertFalse(ComputerControl.needs_routing({'navigation':{'action':'home'}},{'system_state':{}},{'text':''}))
        self.assertFalse(ComputerControl.needs_routing({'navigation':{'action':'open_app','target_app':'Chat'}},
            {'system_state':{'computer':{}}},{'text':'Open Chat'}))
        self.assertTrue(ComputerControl.needs_routing({'system_action':{'action':'inspect','page':'computer'},
            'requires_tools':False,'should_update':True,'update_mode':'global'},{},{'text':'本机电脑'}))

    def test_cached_home_never_needs_a_network_caption_compressor(self):
        planner=SimpleNamespace(complete=Mock(side_effect=AssertionError('Home must not compress a sampled caption')))
        plan={'navigation':{'action':'home'},'should_update':True,'requires_tools':False,
            'update_mode':'global','target_type':'other','next_prompt':'Old system-state summary '*20}
        result=compact_plan_prompt(planner,plan)
        self.assertEqual(result['next_prompt'],'Neural desktop');planner.complete.assert_not_called()

    def test_settings_request_does_not_claim_device_or_network_connected(self):
        computer=ComputerControl(FakeHost())
        result=computer.invoke({'action':'settings','page':'bluetooth'},{},{'text':'Open Bluetooth'})
        self.assertEqual(result['status'],'settings_requested')
        self.assertNotIn('connection',result)

    def test_unchanged_native_state_gets_a_new_real_operation_label(self):
        computer=ComputerControl(FakeHost())
        one=computer.invoke({'action':'wifi_refresh'},{},{})
        two=computer.invoke({'action':'wifi_refresh'},{},{})
        self.assertEqual(one['system_state']['wifi'],two['system_state']['wifi'])
        self.assertEqual(one['receipt_label'],'Result 1')
        self.assertEqual(two['receipt_label'],'Result 2')

    def test_selecting_a_network_never_calls_connect(self):
        host=FakeHost();computer=ComputerControl(host)
        result=computer.invoke({'action':'wifi_select','profile_id':PROFILE},{},{'text':'MeasuredWiFi'})
        self.assertEqual(result['status'],'selected')
        self.assertEqual(result['system_state']['wifi']['selected_profile_id'],PROFILE)
        self.assertEqual(host.calls,[])

    def test_native_already_connected_never_submits_a_reconnect(self):
        host=WindowsHost();data=FakeHost().wifi();host.wifi=lambda:data
        host.handle=Mock(side_effect=AssertionError('A connected profile must not reconnect'))
        result=host.connect(PROFILE)
        self.assertEqual(result['status'],'already_connected')
        self.assertTrue(result['ok']);host.handle.assert_not_called()

    def test_native_admission_without_confirmed_connection_is_not_success(self):
        host=WindowsHost();data=FakeHost().wifi()
        data['interfaces']=[];data['profiles'][0].update(name='MeasuredWiFi',interface_id='12345678-1234-1234-1234-123456789abc')
        host.wifi=lambda:data
        native=SimpleNamespace(WlanConnect=Mock(return_value=0))
        @contextmanager
        def handle():yield native,object()
        host.handle=handle
        with patch('windows_host.time.monotonic',side_effect=[0,20]):result=host.connect(PROFILE)
        native.WlanConnect.assert_called_once()
        self.assertFalse(result['ok']);self.assertEqual(result['status'],'unconfirmed')

    def test_planner_receives_current_full_frame_and_host_facts(self):
        planner=DeepSeekPlanner();planner.computer=ComputerControl(FakeHost())
        frame={'width':256,'height':256,'version':1,'prompt':'Desktop','state':{}}
        image=BytesIO();Image.new('L',(256,256),255).save(image,format='PNG')
        def infer(instruction,context,images):
            self.assertEqual(images[0],image.getvalue())
            self.assertEqual(context['local_computer']['computer']['name'],'LOCAL-PC')
            self.assertIn('system_action',instruction)
            return {'intent':'Read local Wi-Fi','expected_change':'Read Windows','state_summary':'Reading',
                'requires_tools':True,'should_update':False,'update_mode':'none','target_type':'other',
                'scope_reason':'Real system request','next_prompt':'','system_action':{'action':'inspect','page':'wifi'}}
        planner._toggle_vision=infer
        plan=planner.infer_auto(image.getvalue(),frame,{'type':'click','x':80,'y':80,'text':'Wi-Fi'},[],True)
        self.assertEqual(plan['system_action'],{'action':'inspect','page':'wifi'})

    def test_mixed_visual_and_real_system_plan_is_corrected_without_a_fake_page(self):
        planner=DeepSeekPlanner(responsive_ui=True);planner.computer=ComputerControl(FakeHost())
        good={'intent':'Read PC','expected_change':'Host result','state_summary':'PC',
            'requires_tools':True,'should_update':False,'update_mode':'none','target_type':'other',
            'scope_reason':'Actual Windows action','next_prompt':'','system_action':{'action':'inspect','page':'computer'}}
        bad=dict(good,requires_tools=False,should_update=True,update_mode='global',next_prompt='Old fake PC page '*30,
            navigation={'action':'open_app','target_app':'Settings'})
        planner._toggle_vision=Mock(side_effect=[bad,good])
        planner.ensure_visible_response=Mock(side_effect=AssertionError('A real preflight must not become a simulated visual update'))
        image=BytesIO();Image.new('L',(256,256),255).save(image,format='PNG')
        result=planner.infer_auto(image.getvalue(),{'width':256,'height':256,'version':1,'prompt':'Desktop','state':{}},
            {'type':'click','x':90,'y':90,'text':'显示本机电脑信息'},[],True)
        self.assertEqual(result['system_action'],good['system_action'])
        self.assertEqual(planner._toggle_vision.call_count,2)
        planner.ensure_visible_response.assert_not_called()

    def test_connect_journal_replay_never_connects_twice_and_home_is_preserved(self):
        with TemporaryDirectory() as folder:
            buffer=BytesIO();Image.new('L',(256,256),255).save(buffer,format='PNG');png=buffer.getvalue()
            frame={'session_id':'b'*32,'version':1,'width':256,'height':256,'sha256':hashlib.sha256(png).hexdigest(),
                'image_url':'/api/live/images/'+'a'*32+'.png','prompt':'Desktop','seed':42,'steps':8,
                'last_event':None,'state':{'view':'desktop','messages':[]},
                'surface_navigation':{'page_id':'desktop','page_kind':'desktop'}}
            plan={'intent':'Connect measured Wi-Fi','expected_change':'Read actual result','state_summary':'Host',
                'requires_tools':True,'should_update':False,'update_mode':'none','target_type':'other',
                'scope_reason':'Real system request','next_prompt':'','system_action':{'action':'wifi_connect','profile_id':PROFILE}}
            planner=SimpleNamespace(public=lambda:{'configured':True},infer_auto=lambda *args,**kw:deepcopy(plan))
            live=LiveApp(folder,renderer=object(),planner=planner);host=FakeHost();live.computer=ComputerControl(host)
            live.sessions[frame['session_id']]=Session(frame,[frame],[],OrderedDict());live._store_image(frame,png)
            changed=Image.new('L',(256,256),255);changed.putpixel((70,70),0)
            output=BytesIO();changed.save(output,format='PNG');updated=output.getvalue()
            live._draw_page=lambda *args,**kwargs:(updated,{'width':256,'height':256,'sha256':hashlib.sha256(updated).hexdigest(),
                'image_url':'/api/live/images/'+'c'*32+'.png','system_state':kwargs['system_result']['system_state'],
                'system_page':'wifi','prompt':'Actual local Wi-Fi','seed':42,'steps':8,
                'state':{'view':'local_computer','messages':[]},'surface_chrome':{'bounds':[0,0,256,32]},'timings':{}})
            live.navigation=SimpleNamespace(observe=lambda *args:None)
            agent=AgentRuntime(folder,live)
            body={'session_id':frame['session_id'],'version':1,'request_id':'local-pc-replay-001',
                'interaction_mode':'auto','event':{'type':'click','x':90,'y':90,'text':'Connect MeasuredWiFi'}}
            result=agent.event(body);replay=agent.event(body)
            self.assertEqual(len(host.calls),1)
            self.assertEqual(result['agent']['business_status'],'complete')
            self.assertEqual(result['agent']['status'],'complete',result['agent'].get('warning'))
            self.assertEqual(replay['agent']['tools'],result['agent']['tools'])
            self.assertEqual(result['surface_navigation']['page_kind'],'app')
            self.assertEqual(result['state']['view'],'local_computer')


if __name__=='__main__':unittest.main()
