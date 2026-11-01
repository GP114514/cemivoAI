import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dsh_bridge import public_notification,result_text,DshError,redact
from dsh_backend import json_answer

class DshContractTests(unittest.TestCase):
 def test_only_visible_answer_is_retained(self):
  message={'method':'session.event','params':{'sessionId':'owned','event':{'type':'assistant/message','seq':2,
   'data':{'message':{'content':[{'type':'reasoning','text':'hidden private reasoning'},{'type':'text','text':'visible answer'}],
    'source':{'credentials':'not retained'}}}}}}
  result=public_notification(message)
  self.assertEqual(result['params']['event']['data'],{'text':'visible answer'})
  self.assertIsNone(public_notification({'method':'session.event','params':{'event':{'type':'request/header','data':{'token':'not retained'}}}}))
 def test_only_completed_turn_can_be_reported_complete(self):
  answer={'method':'session.event','params':{'event':{'type':'assistant/message','data':{'text':'done'}}}}
  def terminal(kind):return {'method':'session.event','params':{'event':{'type':'turn/end','data':{'reason':{'kind':kind}}}}}
  self.assertEqual(result_text({'events':[answer,terminal('completed')]}),'done')
  for events in ([answer],[answer,terminal('max_tokens')],[answer,terminal('error')],[terminal('completed')]):
   with self.assertRaises(DshError):result_text({'events':events})
 def test_json_and_credential_redaction(self):
  self.assertEqual(json_answer('```json\n{"answer":"OK"}\n```'),{'answer':'OK'})
  self.assertNotIn('abcdefghi',redact('sk-abcdefghijklmno'))
  with self.assertRaises(ValueError):json_answer('[]')

 def test_capability_metadata_excludes_adapter_credentials(self):
  event={'method':'session.event','params':{'sessionId':'owned','event':{'type':'request/header',
   'data':{'header':{'config':{'apiKey':'private-value'},'adapterDefaults':{'secret':'private-value'},
    'tools':[{'name':'read','description':'not retained','parameters':{}}]}}}}}
  result=public_notification(event)
  self.assertEqual(result['params']['event']['data'],{'names':['read']})
  self.assertNotIn('private-value',str(result))

if __name__=='__main__':unittest.main()
