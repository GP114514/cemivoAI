"""Publish an immutable snapshot of this project's latest atomic EMA weights."""
from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
import tempfile


class SnapshotError(ValueError):
    pass


def read_json(path):
    try:
        value=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):
        return {}


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def freeze_latest(root, native_size):
    """Read one published file handle; never lock, pause or mutate training."""
    import torch
    from .model import FORMAT_VERSION
    root=Path(root).resolve()
    neural=root/'neural'
    source=neural/'checkpoints/inference.pt'
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(prefix='live-model-',suffix='.copying',dir=neural,delete=False) as handle:
            temporary=Path(handle.name)
            # An open handle stays on one complete atomic publication even if
            # training publishes a newer inference.pt during this copy.
            with source.open('rb') as src:
                shutil.copyfileobj(src,handle,length=4*1024**2)
        saved=torch.load(temporary,map_location='cpu',weights_only=False)
        if (not isinstance(saved,dict) or saved.get('format_version')!=FORMAT_VERSION or
                not isinstance(saved.get('model'),dict) or not saved['model']):
            raise SnapshotError('最新权重不是完整的 cemivoAI 模型。')
        step,size=saved.get('step'),saved.get('native_size',256)
        if type(step) is not int or step<1 or type(size) is not int or size!=native_size:
            raise SnapshotError('最新权重的原生分辨率与当前 APP 不一致，已保留当前模型。')
        del saved
        digest=sha256(temporary)
        target=neural/f'live-snapshot-step-{step}-native{size}'
        if not target.resolve().is_relative_to(neural.resolve()):
            raise SnapshotError('模型快照必须保存在项目内。')
        metadata={'step':step,'native_size':size,'sha256':digest,
            'source':'neural/checkpoints/inference.pt','created_at':datetime.now(timezone.utc).isoformat(),
            'random_initialization':True,'pretrained_weights':False,'frozen':True,'quality_verified':False}
        if target.exists():
            existing=read_json(target/'snapshot.json')
            if (existing.get('step')!=step or existing.get('native_size')!=size or
                    existing.get('sha256')!=digest or sha256(target/'inference.pt')!=digest):
                raise SnapshotError('同名固定快照与最新权重不一致，未覆盖任何模型。')
            return target,existing
        target.mkdir()
        os.replace(temporary,target/'inference.pt')
        (target/'snapshot.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        return target,metadata
    except SnapshotError:
        raise
    except (OSError,ValueError,RuntimeError,EOFError,KeyError,TypeError,pickle.UnpicklingError):
        raise SnapshotError('最新模型未完整保存或无法读取，当前模型已保留。') from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def select_snapshot(root,directory):
    """Persist only the launcher selection, preserving unrelated settings."""
    root=Path(root).resolve()
    directory=Path(directory).resolve()
    if not directory.is_relative_to(root/'neural'):
        raise SnapshotError('模型快照必须保存在项目内。')
    path=root/'runtime.json'
    settings=read_json(path)
    settings['live_snapshot']=str(directory)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',prefix='runtime-',suffix='.next',dir=root,delete=False) as handle:
            temporary=Path(handle.name)
            json.dump(settings,handle,ensure_ascii=False,indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
