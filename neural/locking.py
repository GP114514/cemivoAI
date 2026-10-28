"""Non-blocking process/thread coordination for one checkpoint directory.

Training, sampling, and evaluation all acquire the first byte of training.lock.
Nested calls in the same process/thread/context reuse that OS lock. Other
threads and processes fail immediately instead of queueing GPU work.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
import threading

_HELD_PATHS = ContextVar('cemivo_model_access_paths', default=frozenset())
_THREAD_LOCKS = {}
_REGISTRY_LOCK = threading.Lock()
_BUSY = '模型正在训练或由其他请求使用，请稍后重试。'


@contextmanager
def model_access(directory):
    """Acquire checkpoint-directory access, or raise RuntimeError immediately.

    Lock ownership includes PID and thread identity, so a copied ContextVar or a
    fork cannot accidentally confer ownership on a different process/thread.
    The OS releases the byte/file lock when its file handle closes, including
    after exceptions or process termination.
    """
    directory = Path(directory).resolve()
    key = os.path.normcase(str(directory))
    process_key = (os.getpid(), key)
    with _REGISTRY_LOCK:
        thread_lock = _THREAD_LOCKS.setdefault(process_key, threading.RLock())
    if not thread_lock.acquire(blocking=False):
        raise RuntimeError(_BUSY)
    handle = None
    context_token = None
    owner = (os.getpid(), threading.get_ident(), key)
    try:
        held = _HELD_PATHS.get()
        if owner in held:
            yield
            return
        directory.mkdir(parents=True, exist_ok=True)
        handle = (directory / 'training.lock').open('a+b')
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(_BUSY) from error
        context_token = _HELD_PATHS.set(held | {owner})
        yield
    finally:
        try:
            if context_token is not None:
                _HELD_PATHS.reset(context_token)
        finally:
            try:
                if handle is not None:
                    handle.close()
            finally:
                thread_lock.release()
