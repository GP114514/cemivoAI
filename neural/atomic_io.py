"""Publish small JSON metadata only after its bytes reach the OS storage API."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time


def atomic_json(path, data):
    """Flush the new file before atomic replacement; never truncate the old one.

    This protects against a process interruption and reduces the power-loss
    window. Filesystem/controller durability still depends on the host. POSIX
    also permits syncing the directory entry; Windows does not expose that via
    the portable Python directory API.
    """
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Unique siblings keep independent publishers from truncating a shared
        # .tmp file. All handles close before Windows replacement.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                prefix=path.name + ".", suffix=".tmp", dir=path.parent,
                delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(.05)
        temporary = None
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
