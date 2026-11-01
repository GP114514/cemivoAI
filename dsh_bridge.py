"""Local DSH stdio SDK transport. Never reads the Harness credential store."""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import uuid

def redact(value):
    return re.sub(r'(?:sk-|gh[pousr]_)[A-Za-z0-9_-]{8,}', '[REDACTED]', str(value))[:1200]

class DshError(RuntimeError):pass

class DshBridge:
    def __init__(self, root, executable=None, planning=False):
        self.root=Path(root).resolve()
        self.planning=planning
        self.executable=Path(executable or os.environ.get('CEMIVO_DSH_EXE','E:/DSH/DeepSeek Harness.exe'))
        self.process=None;self.pending={};self.events=[];self.lock=threading.RLock();self.write_lock=threading.Lock()
        self.stderr=[];self.initialized=False;self.active_session=None
        self.start_lock=threading.RLock();self.prompt_lock=threading.Lock()
        self.on_event=None

    def start(self, provider='deepseek-official', model='deepseek-flash'):
        with self.start_lock:return self._start(provider,model)

    def _start(self, provider, model):
        with self.lock:
            if self.initialized and self.process and self.process.poll() is None:return
            if not self.executable.is_file():raise DshError('DSH executable is missing. Set CEMIVO_DSH_EXE.')
            entry=self.executable.parent/'resources/app.asar/dsh/node_modules/@deepseek-ai/dsh/lib/bin.js'
            profile='cemivo-graphics-planner' if self.planning else 'cemivo-graphics'
            args=[str(self.executable),'--expose-internals',str(entry),'--profile',profile]
            home=Path(os.environ.get('DSH_HOME',str(Path.home()/'.dsh')))
            if not (home/'profiles'/profile/'package.json').exists():args+=['--from-default-profile','sdk']
            if self.planning:
                patch=Path(__file__).resolve().parent/'dsh-planner.patch.json'
                if not patch.is_file():raise DshError('Missing no-tools planning profile patch')
                args+=['--patch',str(patch)]
            env=dict(os.environ,ELECTRON_RUN_AS_NODE='1')
            self.process=subprocess.Popen(args,cwd=self.root,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace',bufsize=1,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            process=self.process
            def read():
                for line in process.stdout:
                    try:message=json.loads(line)
                    except ValueError:continue
                    if 'id' in message:
                        waiter=self.pending.get(message['id'])
                        if waiter:waiter.put(message)
                    elif 'method' in message:
                        message=public_notification(message)
                        if message is None:continue
                        self.events.append(message)
                        if self.on_event:
                            try:self.on_event(message)
                            except Exception:pass
                        if len(self.events)>2000:self.events=self.events[-2000:]
                for waiter in list(self.pending.values()):waiter.put({'error':{'message':'DSH process exited'}})
            def errors():
                for line in process.stderr:
                    self.stderr.append(redact(line.strip()));self.stderr=self.stderr[-20:]
            threading.Thread(target=read,daemon=True).start();threading.Thread(target=errors,daemon=True).start()
        try:
            result=self.call('initialize',{'cwd':str(self.root),'provider':provider,'model':model,'maxTokens':8192,'reasoningEffort':'low'},60)
            if result.get('serverInfo',{}).get('name')!='deepseek-harness-sdk-runtime':raise DshError('Unexpected DSH SDK server')
            self.initialized=True
        except Exception:
            self.close();raise

    def call(self, method, params=None, timeout=60):
        if not self.process or self.process.poll() is not None:raise DshError('DSH is not running')
        identity='cemivo_'+uuid.uuid4().hex;waiter=queue.Queue();self.pending[identity]=waiter
        try:
            with self.write_lock:
                self.process.stdin.write(json.dumps({'jsonrpc':'2.0','id':identity,'method':method,'params':params or {}},ensure_ascii=False)+'\n')
                self.process.stdin.flush()
            try:response=waiter.get(timeout=timeout)
            except queue.Empty:raise DshError('DSH request timed out; it was not automatically resent.') from None
            if 'error' in response:raise DshError(redact(response['error'].get('message','DSH SDK failure')))
            return response.get('result',{})
        finally:self.pending.pop(identity,None)

    def prompt(self, blocks, session_id=None, timeout=120, on_event=None):
        with self.prompt_lock:
            self.start();session_id=session_id or 'cemivo-graphics-'+uuid.uuid4().hex
            # Unique sessions isolate each accepted interaction; repeated UI
            # requests are deduplicated by the application's durable journal.
            self.active_session=session_id
            self.on_event=on_event
            try:
                admitted=self.call('session/prompt',{'sessionId':session_id,'contentBlocks':blocks},30)
                self.call('session/wait',{'sessionId':session_id},timeout)
                events=[m for m in self.events if m.get('params',{}).get('sessionId')==session_id]
                return {'session_id':session_id,'message_id':admitted.get('messageId'),'events':events}
            finally:self.on_event=None

    def close(self):
        process=self.process
        if process and process.poll() is None:
            try:self.call('shutdown',{},10)
            except Exception:
                process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        self.initialized=False;self.process=None

def public_notification(message):
    if message.get('method')=='session.status':return message
    if message.get('method')!='session.event':return None
    params=message.get('params',{});event=params.get('event',{});kind=event.get('type','')
    if kind=='request/header':
        header=event.get('data',{}).get('header')
        if not isinstance(header,dict):return None
        # Keep capability names only. Credentials, adapter config and headers
        # must never enter the bridge's retained/public event stream.
        names=[t['name'] for t in header.get('tools',[]) if isinstance(t,dict) and isinstance(t.get('name'),str)]
        event={'type':'capability/tools','data':{'names':names}}
        return {'method':'session.event','params':{'sessionId':params.get('sessionId'),'event':event}}
    if kind=='assistant/message':
        data=event.get('data',{});content=data.get('message',{}).get('content',[])
        text='\n'.join(b.get('text','') for b in content if b.get('type')=='text')
        if not text:return None
        event={'type':kind,'seq':event.get('seq'),'data':{'text':text}}
    elif not(kind.startswith('tool/') or kind=='turn/end'):
        return None
    return {'method':'session.event','params':{'sessionId':params.get('sessionId'),'event':event}}

def result_text(result):
    texts=[m['params']['event']['data']['text'] for m in result['events']
        if m.get('method')=='session.event' and m['params']['event'].get('type')=='assistant/message']
    terminals=[m['params']['event'] for m in result['events']
        if m.get('method')=='session.event' and m['params']['event'].get('type')=='turn/end']
    if not terminals or terminals[-1].get('data',{}).get('reason',{}).get('kind')!='completed':
        reason=terminals[-1].get('data',{}).get('reason',{}) if terminals else {'kind':'missing_terminal'}
        raise DshError('DSH turn did not complete ('+redact(json.dumps(reason,ensure_ascii=False))+'); no success was inferred.')
    if not texts:raise DshError('DSH completed without a visible answer')
    return texts[-1]
