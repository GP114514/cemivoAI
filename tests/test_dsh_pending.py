"""Timeout ownership and late receipts, without models, credentials or services."""
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import hashlib
import sys
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from agent_runtime import AgentRuntime
from dsh_backend import DshBackend
from dsh_bridge import DshBridge,DshPending,DshTimeout
from live_app import LiveApp,LiveError,Session


class DelayedBridge(DshBridge):
    def __init__(self,root,admission_timeout=False):
        super().__init__(root)
        self.process=SimpleNamespace(poll=lambda:None)
        self.release=Event();self.waiting=Event();self.prompts=0;self.waits=0
        self.admission_timeout=admission_timeout

    def start(self,*args,**kwargs):self.initialized=True

    def emit(self,kind,data,session_id=None):
        self._notification({'method':'session.event','params':{
            'sessionId':session_id or self.active_session,'event':{'type':kind,'data':data}}},self.process)

    def call(self,method,params=None,timeout=60):
        if method=='session/prompt':
            self.prompts+=1
            if self.admission_timeout:raise DshTimeout('admission timeout')
            self.emit('tool/call',{'callId':'write-one','name':'write','arguments':'{"path":"proof.txt"}'})
            return {'messageId':'accepted-once'}
        if method=='session/wait':
            self.waits+=1
            if self.waits==1:raise DshTimeout('observation timeout')
            self.waiting.set()
            if not self.release.wait(5):raise DshTimeout('test did not release observation')
            self.emit('tool/result',{'message':{'toolCallId':'write-one','isError':False,
                'content':[{'type':'text','text':'written'}]}})
            self.emit('assistant/message',{'message':{'content':[{'type':'text','text':'{"answer":"File written."}'}]}})
            self.emit('turn/end',{'reason':{'kind':'completed'}})
            return {}
        raise AssertionError(method)


class PendingTests(unittest.TestCase):
    def test_restart_does_not_poll_a_lost_observer_forever(self):
        with TemporaryDirectory() as directory:
            agent=AgentRuntime(directory,SimpleNamespace(planner=SimpleNamespace(dsh_backend=None)))
            agent.journal.reserve('restarted-test-001','fingerprint','screen')
            agent.journal.save('restarted-test-001','awaiting_result',{
                'dsh_pending':True,'dsh_session_id':'lost-session','tools':[],'response':None})
            result=agent.request_status('restarted-test-001')
            self.assertEqual(result['status'],'interrupted')
            self.assertEqual(result['dsh_session_id'],'lost-session')
            self.assertFalse(result['dsh_pending'])

    def test_reconfiguration_cannot_stop_an_unsettled_turn(self):
        with TemporaryDirectory() as directory:
            backend=DshBackend(directory)
            backend.bridge.unsettled_session='still-executing'
            with self.assertRaises(DshPending):backend.configure()
            self.assertEqual(backend.bridge.unsettled_session,'still-executing')

    def test_timeout_observes_original_session_without_resubmitting(self):
        bridge=DelayedBridge('.')
        settled=Event();results=[]
        def completed(result,error,confirmed):
            results.append((result,error,confirmed));settled.set()
        try:
            with self.assertRaises(DshPending):
                bridge.prompt([],session_id='original',on_settled=completed)
            self.assertTrue(bridge.waiting.wait(1))
            with self.assertRaises(DshPending):bridge.prompt([],session_id='duplicate')
            self.assertEqual(bridge.prompts,1)
            bridge.release.set();self.assertTrue(settled.wait(2))
            self.assertEqual(results[0][0]['session_id'],'original')
            self.assertIsNone(results[0][1]);self.assertTrue(results[0][2])
            for _ in range(100):
                if bridge.unsettled_session is None:break
                time.sleep(.01)
            self.assertIsNone(bridge.unsettled_session)
        finally:bridge.release.set()

    def test_unknown_admission_never_starts_a_replacement(self):
        bridge=DelayedBridge('.',admission_timeout=True)
        with self.assertRaises(DshPending):bridge.prompt([],session_id='unknown')
        with self.assertRaises(DshPending):bridge.prompt([],session_id='replacement')
        self.assertEqual(bridge.prompts,1)
        self.assertEqual(bridge.unsettled_session,'unknown')

    def test_late_or_foreign_session_cannot_write_current_receipts(self):
        bridge=DelayedBridge('.');bridge.active_session='current';observed=[];bridge.on_event=observed.append
        bridge.emit('tool/call',{'callId':'old','name':'write'},session_id='previous')
        self.assertEqual(observed,[])
        bridge.emit('tool/call',{'callId':'new','name':'read'},session_id='current')
        self.assertEqual(len(observed),1)
        bridge._notification({'method':'session.status','params':{'sessionId':'current'}},object())
        self.assertEqual(len(observed),1)

    def test_agent_keeps_frame_and_deduplicates_until_late_receipts_arrive(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);backend=DshBackend(root);bridge=DelayedBridge(root);backend.bridge=bridge
            planner=SimpleNamespace(public=lambda:{'configured':True},dsh_backend=backend)
            live=LiveApp(root,renderer=object(),planner=planner)
            buffer=BytesIO();Image.new('L',(256,256),255).save(buffer,format='PNG');png=buffer.getvalue()
            session_id='b'*32
            frame={'session_id':session_id,'version':1,'width':256,'height':256,
                'image_url':'/api/live/images/'+'a'*32+'.png','sha256':hashlib.sha256(png).hexdigest(),
                'prompt':'black and white desktop','seed':1,'steps':4,'state':{'messages':[]},'last_event':None}
            live.sessions[session_id]=Session(frame,[frame],[],OrderedDict());live._store_image(frame,png)
            agent=AgentRuntime(root,live)
            body={'session_id':session_id,'version':1,'request_id':'pending-test-001',
                'event':{'type':'click','x':128,'y':128,'text':'Write proof.txt'}}
            try:
                result=agent.event(body)
                self.assertEqual(result['agent']['status'],'awaiting_result')
                self.assertEqual(result['version'],frame['version'])
                self.assertEqual(result['sha256'],frame['sha256'])
                self.assertEqual(agent.request_status(body['request_id'])['status'],'awaiting_result')
                self.assertEqual(agent.event(body)['agent']['status'],'awaiting_result')
                with self.assertRaises(LiveError) as rejected:
                    agent.event(dict(body,request_id='pending-test-002'))
                self.assertEqual(rejected.exception.code,'dsh_busy')
                self.assertIsNone(agent.journal.get('pending-test-002'))
                self.assertEqual(bridge.prompts,1)
                bridge.release.set()
                for _ in range(200):
                    saved=agent.request_status(body['request_id'])
                    if saved['status']!='awaiting_result':break
                    time.sleep(.01)
                self.assertEqual(saved['status'],'render_failed')
                self.assertFalse(saved['dsh_pending'])
                self.assertEqual(saved['tools'][0]['status'],'complete')
                self.assertEqual(saved['business_status'],'complete')
                self.assertEqual(saved['answer'],'File written.')
                self.assertEqual(agent.event(body)['agent']['status'],'render_failed')
                self.assertEqual(bridge.prompts,1)
                self.assertEqual(live.current()['sha256'],frame['sha256'])
            finally:bridge.release.set()


if __name__=='__main__':unittest.main()
