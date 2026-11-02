from types import SimpleNamespace
import unittest

from dsh_bridge import DshBridge,DshError,DshPending,result_text


class ReusedBridge(DshBridge):
    def __init__(self):
        super().__init__('.')
        self.process=SimpleNamespace(poll=lambda:None)
        self.prompts=[];self.resumes=[];self.emit_response=True;self.resume_error=False

    def start(self,*args,**kwargs):self.initialized=True

    def emit(self,kind,data):
        self._notification({'method':'session.event','params':{'sessionId':self.active_session,
            'event':{'type':kind,'data':data}}},self.process)

    def call(self,method,params=None,timeout=60):
        if method=='session/resume':
            self.resumes.append(params['sessionId'])
            if self.resume_error:raise DshError('resume failed')
            return {}
        if method=='session/prompt':
            self.prompts.append(params['sessionId'])
            return {'messageId':'message-'+str(len(self.prompts))}
        if method=='session/wait':
            if self.emit_response:
                self.emit('assistant/message',{'message':{'content':[{'type':'text','text':'answer-'+str(len(self.prompts))}]}})
                self.emit('turn/end',{'reason':{'kind':'completed'}})
            return {}
        raise AssertionError(method)


class DshReuseTests(unittest.TestCase):
    def test_reuses_one_channel_but_returns_only_the_current_turn(self):
        bridge=ReusedBridge()
        first=bridge.prompt([]);second=bridge.prompt([])
        self.assertEqual(len(set(bridge.prompts)),1)
        self.assertEqual(result_text(first),'answer-1')
        self.assertEqual(result_text(second),'answer-2')
        self.assertEqual(len(second['events']),2)

    def test_old_success_cannot_complete_a_new_turn_without_events(self):
        bridge=ReusedBridge();result_text(bridge.prompt([]))
        bridge.emit_response=False
        with self.assertRaises(DshError):result_text(bridge.prompt([]))

    def test_process_restart_resumes_only_its_known_channel(self):
        bridge=ReusedBridge();bridge.prompt([]);bridge.open_sessions.clear()
        self.assertEqual(result_text(bridge.prompt([])),'answer-2')
        self.assertEqual(bridge.resumes,[bridge.session_id])
        self.assertEqual(len(set(bridge.prompts)),1)

    def test_failed_resume_never_opens_an_alternate_session_or_resends(self):
        bridge=ReusedBridge();bridge.prompt([]);bridge.open_sessions.clear();bridge.resume_error=True
        with self.assertRaises(DshError):bridge.prompt([])
        self.assertEqual(len(bridge.prompts),1)

    def test_failed_quiescence_check_blocks_reuse_of_the_channel(self):
        class FailedWait(ReusedBridge):
            def call(self,method,params=None,timeout=60):
                if method=='session/wait':raise DshError('quiescence is unknown')
                return super().call(method,params,timeout)
        bridge=FailedWait()
        with self.assertRaises(DshPending):bridge.prompt([])
        with self.assertRaises(DshPending):bridge.prompt([])
        self.assertEqual(len(bridge.prompts),1)
        self.assertEqual(bridge.unsettled_session,bridge.session_id)


if __name__=='__main__':unittest.main()
