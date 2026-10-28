"""Bounded, process-memory-only interaction summaries for the local console."""
from collections import deque
import re
import threading
import time

_rows = deque(maxlen=200)
_lock = threading.Lock()
_sequence = 0

def clean(value):
    text = str(value)[:3000]
    text = re.sub(r'(?:sk-|gh[pousr]_)[A-Za-z0-9_-]{8,}', '[REDACTED]', text)
    text = re.sub(r'(?i)(Bearer\s+)\S+', r'\1[REDACTED]', text)
    return ''.join(c if c in '\n\t' or ord(c) >= 32 and not 127 <= ord(c) <= 159 else ' ' for c in text)

def emit(kind, **fields):
    global _sequence
    with _lock:
        _sequence += 1
        _rows.append(dict(id=_sequence, time=time.strftime('%H:%M:%S'), kind=kind,
            fields={k:clean(v) for k,v in fields.items() if v is not None}))

def since(cursor):
    with _lock:
        return {'events':[dict(row) for row in _rows if row['id'] > cursor],
                'cursor':_sequence, 'oldest':_rows[0]['id'] if _rows else _sequence}
