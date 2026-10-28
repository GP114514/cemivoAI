"""Read-only console: no commands, API keys, or request replay."""
import argparse
import json
import sys
import time
from urllib.request import urlopen
from console_events import clean

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=7861)
    args=parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    base=f'http://127.0.0.1:{args.port}'
    cursor=0; stage=None; offline=False
    print('cemivoAI | INPUT / OUTPUT\nLive interaction summaries. API keys are hidden. Ctrl+C closes this console only.\n',flush=True)
    def get(route):
        with urlopen(base+route,timeout=3) as response:return json.load(response)
    try:
        while True:
            try:
                data=get('/api/console?after='+str(cursor))
                if offline:print('[CONNECTED]',flush=True)
                offline=False
                if cursor and data['oldest']>cursor+1:print('[INFO] Older entries expired from memory.',flush=True)
                for row in data['events']:
                    print('\n['+row['time']+'] '+clean(row['kind']),flush=True)
                    for key,value in row['fields'].items():print('  '+key+': '+clean(value),flush=True)
                cursor=data['cursor']
                status=get('/api/live/status').get('runtime',{})
                current=(status.get('busy'),status.get('stage'))
                if current!=stage and current[0]:print('[STAGE] '+clean(current[1]),flush=True)
                stage=current
            except Exception:
                if not offline:print('[WAIT] Local APP is unavailable; reconnecting…',flush=True)
                offline=True
            time.sleep(.4)
    except KeyboardInterrupt:pass

if __name__=='__main__':main()
