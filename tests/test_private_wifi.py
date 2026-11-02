from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch
from xml.etree import ElementTree as ET
import ctypes as C
import hashlib
import unittest

import console_events
from agent_runtime import AgentRuntime
from computer_control import ComputerControl
from live_app import LiveApp,Session,LiveError
from windows_host import WindowsHost,HostError,ConnectParameters,temporary_wifi_xml
from PIL import Image

NETWORK='b'*24
PROFILE='a'*24
SECRET='LOCAL_ONLY_TEST_8391'

class InlineHost:
    available=True
    def __init__(self):self.connections=0;self.password_received=False
    def info(self):return {'connected':True,'name':'LOCAL-PC'}
    def wifi(self):return {'available':True,'interfaces':[{'id':'12345678-1234-1234-1234-123456789abc','state':'connected','ssid':'Original','profile_id':PROFILE}],
        'profiles':[{'id':PROFILE,'name':'Original','connected':True}], 'errors':[],
        'networks':[{'id':NETWORK,'ssid':'InlineTest','ssid_hex':'496e6c696e6554657374','auth':7,'cipher':4,
            'secured':True,'connectable':True,'profile_id':None,'interface_id':'12345678-1234-1234-1234-123456789abc'}]}
    def connect_network(self,identity,password=''):
        self.connections+=1;self.password_received=password==SECRET
        return {'ok':True,'status':'connected','temporary':True,'connection':{'ssid':'InlineTest','state':'connected'}}
    error=staticmethod(WindowsHost.error)

def png(black=False):
    output=BytesIO();image=Image.new('L',(256,256),255)
    if black:image.putpixel((90,90),0)
    image.save(output,format='PNG');return output.getvalue()

class PrivateWifiTests(unittest.TestCase):
    def test_temporary_xml_escapes_credentials_without_changing_them(self):
        network=InlineHost().wifi()['networks'][0]
        secret='key&<tag>99'
        root=ET.fromstring(temporary_wifi_xml(network,secret));ns={'w':'http://www.microsoft.com/networking/WLAN/profile/v1'}
        self.assertEqual(root.find('.//w:keyMaterial',ns).text,secret)
        self.assertEqual(root.find('.//w:hex',ns).text,network['ssid_hex'])
        self.assertEqual(root.find('.//w:connectionMode',ns).text,'manual')
        self.assertEqual(root.find('.//w:authentication',ns).text,'WPA2PSK')

    def test_unsupported_authentication_never_falls_back_to_a_windows_dialog(self):
        network=InlineHost().wifi()['networks'][0];network['auth']=6
        with self.assertRaises(HostError) as caught:temporary_wifi_xml(network,SECRET)
        self.assertEqual(caught.exception.code,'unsupported_wifi_security')

    def test_host_has_no_external_settings_launcher(self):
        self.assertFalse(hasattr(WindowsHost(),'open_settings'))

    def test_temporary_native_connection_is_not_persisted_or_assumed_complete(self):
        host=WindowsHost();state=InlineHost().wifi();host.wifi=lambda:state;seen=[]
        def connect(handle,guid,parameters,reserved):
            values=C.cast(parameters,C.POINTER(ConnectParameters)).contents
            seen.append((values.mode,values.flags,SECRET in values.profile));return 0
        @contextmanager
        def handle():yield SimpleNamespace(WlanConnect=connect),object()
        host.handle=handle
        with patch('windows_host.time.monotonic',side_effect=[0,20]):result=host.connect_network(NETWORK,SECRET)
        self.assertEqual(seen,[(1,0,True)])
        self.assertFalse(result['ok']);self.assertEqual(result['status'],'unconfirmed')

    def test_same_ssid_with_different_security_is_not_reported_as_connected(self):
        host=WindowsHost();state=InlineHost().wifi();network=state['networks'][0]
        state['interfaces'][0].update(ssid=network['ssid'],ssid_hex=network['ssid_hex'],auth=1,cipher=0)
        host.wifi=lambda:state
        seen=Mock(return_value=0)
        @contextmanager
        def handle():yield SimpleNamespace(WlanConnect=seen),object()
        host.handle=handle
        with patch('windows_host.time.monotonic',side_effect=[0,20]):result=host.connect_network(NETWORK,SECRET)
        seen.assert_called_once();self.assertFalse(result['ok'])

    def test_matching_native_connection_does_not_reconnect(self):
        host=WindowsHost();state=InlineHost().wifi();network=state['networks'][0]
        state['interfaces'][0].update(ssid=network['ssid'],ssid_hex=network['ssid_hex'],auth=7,cipher=4)
        host.wifi=lambda:state;host.handle=Mock(side_effect=AssertionError('Do not reconnect an exact matching network'))
        result=host.connect_network(NETWORK)
        self.assertEqual(result['status'],'already_connected');host.handle.assert_not_called()

    def test_requesting_a_password_does_not_submit_a_connection(self):
        host=InlineHost();control=ComputerControl(host);state=control.snapshot()
        previous={'session_id':'b'*32,'system_state':state}
        previous['system_state']['wifi']['selected_network_id']=NETWORK
        result=control.invoke({'action':'wifi_network_connect','network_id':NETWORK},previous,{'text':''})
        self.assertEqual(result['status'],'credential_required');self.assertEqual(result['page'],'wifi_password')
        self.assertEqual(host.connections,0)
        self.assertEqual(len(result['credential_request']['token']),48)

    def test_expired_or_cancelled_password_page_cannot_submit_credentials(self):
        host=InlineHost();control=ComputerControl(host)
        previous={'session_id':'b'*32,'system_state':control.snapshot()}
        previous['system_state']['wifi']['selected_network_id']=NETWORK
        request=control.invoke({'action':'wifi_network_connect','network_id':NETWORK},previous,{'text':''})['credential_request']
        with self.assertRaises(LiveError):
            control.execute_private(None,'ignored',b'',previous,{'x':20,'y':20},{},
                {'token':request['token'],'password':SECRET})
        self.assertEqual(host.connections,0)

    def test_secret_bypasses_model_journal_console_and_replayed_connection(self):
        with TemporaryDirectory() as directory:
            host=InlineHost();control=ComputerControl(host);original=png()
            previous={'session_id':'b'*32,'version':1,'system_state':control.snapshot()}
            previous['system_state']['wifi']['selected_network_id']=NETWORK
            request=control.invoke({'action':'wifi_network_connect','network_id':NETWORK},previous,{'text':''})['credential_request']
            request.update(input_bounds=[12,112,240,134],submit_bounds=[12,150,112,180],status_bounds=[12,80,240,98])
            frame=dict(previous,width=256,height=256,sha256=hashlib.sha256(original).hexdigest(),
                image_url='/api/live/images/'+'a'*32+'.png',prompt='Private Wi-Fi password',seed=42,steps=8,
                last_event=None,state={'view':'local_computer','messages':[]},credential_request=request)
            planner=SimpleNamespace(public=lambda:{'configured':True},infer_auto=Mock(side_effect=AssertionError('No password in model planning')))
            live=LiveApp(directory,renderer=object(),planner=planner);live.computer=control
            live.sessions[frame['session_id']]=Session(frame,[frame],[],OrderedDict());live._store_image(frame,original)
            updated=png(True)
            def draw(local,pixels,source,result):
                self.assertNotIn(SECRET,str(result))
                return updated,{'width':256,'height':256,'sha256':hashlib.sha256(updated).hexdigest(),
                    'image_url':'/api/live/images/'+'c'*32+'.png','prompt':'Actual Wi-Fi result','seed':42,'steps':8,
                    'system_state':result['system_state'],'system_page':'wifi',
                    'state':{'view':'local_computer','messages':[]},'surface_chrome':{'bounds':[0,0,256,32]},'timings':{}}
            control.draw_private_result=draw
            live._draw_page=Mock(side_effect=AssertionError('A Wi-Fi result must work without the online model'))
            live.navigation=SimpleNamespace(observe=lambda *args:None)
            agent=AgentRuntime(directory,live)
            body={'session_id':frame['session_id'],'version':1,'request_id':'private-wifi-test-001',
                'token':request['token'],'password':SECRET,'x':60,'y':166}
            result=agent.private_wifi(body);replay=agent.private_wifi(body)
            self.assertEqual(result['agent']['status'],'complete',result['agent'].get('warning'))
            self.assertTrue(host.password_received);self.assertEqual(host.connections,1)
            self.assertEqual(replay['agent']['tools'],result['agent']['tools'])
            self.assertNotIn(SECRET,str(result)+str(agent.request_status(body['request_id']))+str(console_events.since(0)))
            for file in Path(directory).rglob('*.sqlite3*'):
                self.assertNotIn(SECRET.encode(),file.read_bytes())
            planner.infer_auto.assert_not_called()
            live._draw_page.assert_not_called()

if __name__=='__main__':unittest.main()
