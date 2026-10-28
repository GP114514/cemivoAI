"""Random-initialized auxiliary surface training; target geometry lives here only."""
from __future__ import annotations
import itertools
import json
import os
from pathlib import Path
import sys
import time
import tempfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural.surface_head import SurfaceHead, FORMAT
from neural.atomic_io import atomic_json


def target_layout(condition):
    mode, a, b, c, d = condition.unbind(-1)
    batch = len(condition)
    desktop = torch.tensor([[16, 18, 240, 44], [16, 66, 118, 122], [138, 66, 240, 122],
        [16, 146, 118, 202], [138, 146, 240, 202], [16, 220, 118, 249],
        [138, 220, 240, 249]], dtype=condition.dtype, device=condition.device).expand(batch, -1, -1).clone()
    # Independently varied gaps, margins and row spacing are learned, rather
    # than loading one stored desktop image or coordinate table at inference.
    desktop[:, :, (0, 2)] += a[:, None, None]*3
    desktop[:, :5, (1, 3)] += b[:, None, None]*4
    desktop[:, 1:5, 2] += c[:, None]*4
    desktop[:, 1:5, 3] += d[:, None]*3
    chrome = desktop.clone()
    chrome[:, 0] = torch.tensor([136, 5, 251, 26], device=condition.device)
    chrome[:, 5] = torch.tensor([4, 3, 64, 29], device=condition.device)
    chrome[:, 6] = torch.tensor([70, 3, 130, 29], device=condition.device)
    chrome[:, [0, 5, 6], 0] += a[:, None]*2
    chrome[:, [0, 5, 6], 2] += a[:, None]*2
    chrome[:, [0, 5, 6], 1] += b[:, None]
    chrome[:, [0, 5, 6], 3] += b[:, None]
    return torch.lerp(desktop, chrome, mode[:, None, None]).reshape(batch, 28)/256


def main():
    output = ROOT/'neural'/'surface-head-v1'
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(271431)
    free, _ = torch.cuda.mem_get_info()
    if free < 3*1024**3:
        raise RuntimeError('preserve main training: need at least 3 GiB free')
    torch.cuda.set_per_process_memory_fraction(.04)
    device = 'cuda'
    model = SurfaceHead().to(device)
    layout_opt = torch.optim.Adam(model.layout.parameters(), lr=.01)
    pixel_opt = torch.optim.Adam(model.pixel.parameters(), lr=.004)
    features = torch.tensor(list(itertools.product((-2., -1., 0., 1., 2.), repeat=4)), device=device)
    targets = ((features >= 0).all(dim=1) & (features == 0).any(dim=1)).float()[:, None]
    start = time.perf_counter()
    print(json.dumps({'parameters': model.parameter_count, 'stage': 'training', 'main_training_modified': False}), flush=True)
    history = []
    for step in range(1, 3001):
        condition = torch.rand(512, 5, device=device)*2-1
        condition[:, 0] = torch.randint(0, 2, (512,), device=device).float()
        target = target_layout(condition)
        layout_loss = F.mse_loss(model.layout(model.layout_features(condition)), target)
        layout_opt.zero_grad(set_to_none=True)
        layout_loss.backward(); layout_opt.step()
        pixel_loss = F.binary_cross_entropy_with_logits(model.pixel(features), targets)
        pixel_opt.zero_grad(set_to_none=True)
        pixel_loss.backward(); pixel_opt.step()
        if step in (2000, 2700):
            for optimizer in (layout_opt, pixel_opt):
                for group in optimizer.param_groups:
                    group['lr'] *= .25
        if step % 500 == 0:
            item = {'step': step, 'layout_loss': float(layout_loss.detach()),
                'pixel_loss': float(pixel_loss.detach()), 'seconds': round(time.perf_counter()-start, 2)}
            history.append(item); print(json.dumps(item), flush=True)
    model.eval()
    with torch.inference_mode():
        test = torch.from_numpy(np.stack([model.conditioning(seed, chrome) for seed in range(8000, 8100) for chrome in (False, True)])).to(device)
        expected = target_layout(test)
        actual = model.layout(model.layout_features(test))
        error = (actual-expected).abs()*256
        prediction = model.pixel(features).sigmoid()
        pixel_wrong = int(((prediction >= .5) != (targets >= .5)).sum())
        pixel_max_error = float((prediction-targets).abs().max())
        layout_max_error = float(error.max())
    validation = {'heldout_layout_conditions': 200, 'layout_max_absolute_error_px': layout_max_error,
        'layout_mean_absolute_error_px': float(error.mean()),
        'coordinate_feature_states_checked': len(features), 'pixel_binary_errors': pixel_wrong,
        'coordinate_feature_states_seen_in_training': True,
        'pixel_max_absolute_error': pixel_max_error, 'layout_seed_range': [8000, 8099],
        'whole_main_model_quality_accepted': False, 'main_training_modified': False}
    validated = pixel_wrong == 0 and pixel_max_error < .03 and layout_max_error < 1.
    payload = {'format': FORMAT, 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'parameters': model.parameter_count, 'step': 3000, 'random_initialization': True,
        'pretrained_weights': False, 'heldout_validated': validated, 'validation': validation,
        'font_or_bitmap_assets_at_inference': False, 'architecture': 'conditional layout head and coordinate pixel MLP'}
    with tempfile.NamedTemporaryFile(prefix='surface.', suffix='.pt.tmp', dir=output, delete=False) as handle:
        temporary = Path(handle.name)
        torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, output/'inference.pt')
    atomic_json(output/'report.json', dict(payload, state_dict=None, history=history, elapsed_seconds=time.perf_counter()-start))
    print(json.dumps({'validated': validated, 'validation': validation}, ensure_ascii=False), flush=True)
    return 0 if validated else 2


if __name__ == '__main__':
    sys.exit(main())
