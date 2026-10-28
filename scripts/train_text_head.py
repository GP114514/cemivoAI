"""Train a finite native glyph decoder from random weights, separate from main training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural.atomic_io import atomic_json
from neural.text_head import TextHead, save_checkpoint
from neural.text_training_data import make_vocabulary, glyph_targets, target_text, target_audit

STRINGS = ['Hello42', '你好123', 'Hello world!', '你好世界', '测试输入文字', '聊天消息 ABC 456',
    '我的名字是小明', 'Input draft: cemivoAI', '保存文档：项目说明', 'AaBbYyZz019!?',
    '今天完成原型设计', '右键输入，回车发送。']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'neural/text-head-v1')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(2026102706)
    device = torch.device(args.device)
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info()
        if free < 3 * 1024**3:
            raise RuntimeError('Preserve main training: less than 3 GiB free')
        torch.cuda.set_per_process_memory_fraction(.04)
    started = time.perf_counter()
    vocabulary = make_vocabulary(); targets = glyph_targets(vocabulary)
    atomic_json(args.output / 'data-audit.json', target_audit(vocabulary, targets))
    model = TextHead(vocabulary).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.00001)
    all_targets = torch.from_numpy(targets.astype(np.float32) / 255).to(device)
    generator = torch.Generator(device=device).manual_seed(2026102706)
    # Frequent user-facing alphabet and words get explicit coverage, while half
    # the examples remain uniform across the whole finite character vocabulary.
    frequent = list(dict.fromkeys(''.join(STRINGS) + vocabulary[:95]))
    frequent_ids = torch.tensor([vocabulary.index(char) for char in frequent], device=device)
    history = []
    print(json.dumps({'stage': 'training', 'vocabulary': len(vocabulary), 'parameters': model.parameter_count,
        'steps': args.steps, 'main_training_modified': False}), flush=True)
    model.train()
    for step in range(1, args.steps + 1):
        ids = torch.cat([torch.randint(len(vocabulary), (256,), device=device, generator=generator),
            frequent_ids[torch.randint(len(frequent_ids), (128,), device=device, generator=generator)]])
        target = all_targets[ids]
        logits = model(ids)
        prediction = logits.sigmoid()
        weights = 1 + 3 * (target < .97).float()
        loss = ((prediction - target).square() * weights).mean() + .1 * F.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2); optimizer.step()
        if step in (args.steps // 2, args.steps * 3 // 4):
            for group in optimizer.param_groups:
                group['lr'] *= .3
        if step % 200 == 0 or step == args.steps:
            item = {'step': step, 'loss': round(float(loss.detach()), 6),
                'elapsed_seconds': round(time.perf_counter() - started, 2)}
            history.append(item); print(json.dumps(item), flush=True)
    model.eval(); details = []
    grid = Image.new('L', (640, len(STRINGS) * 24), 255)
    with torch.inference_mode():
        all_errors = []
        for start in range(0, len(vocabulary), 256):
            ids = torch.arange(start, min(len(vocabulary), start + 256), device=device)
            predicted = model(ids).sigmoid().mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            error = np.abs(predicted.astype(np.int16) - targets[start:start + len(ids)].astype(np.int16))
            all_errors.extend(error.mean(axis=(1, 2)).tolist())
    for index, text in enumerate(STRINGS):
        prediction = model.predict_text(text, 300, 18)
        target = target_text(text, 300, 18, vocabulary, targets)
        error = np.abs(prediction.astype(np.int16) - target.astype(np.int16))
        binary_disagreement = float(np.mean((prediction < 128) != (target < 128)))
        record = {'text': text, 'width': 300, 'height': 18, 'mae_0_255': float(error.mean()),
            'max_error': int(error.max()), 'binary_disagreement_fraction': binary_disagreement,
            'neural_pixels_sha256': hashlib.sha256(prediction.tobytes()).hexdigest(),
            'target_pixels_sha256': hashlib.sha256(target.tobytes()).hexdigest()}
        details.append(record)
        Image.fromarray(prediction).save(args.output / f'string-{index:02}.png')
        grid.paste(Image.fromarray(target), (0, index * 24)); grid.paste(Image.fromarray(prediction), (330, index * 24))
    grid.save(args.output / 'heldout-combinations.png')
    # These combinations were never training rasters. Their individual glyphs
    # WERE trained; this is compositional reconstruction, not unseen-character
    # generalization or independent OCR. Broad-vocabulary quality is separate.
    validated = all(item['mae_0_255'] < 3 and item['binary_disagreement_fraction'] < .004 for item in details)
    validation = {'heldout_string_combinations': details, 'string_images_used_for_training': False,
        'individual_characters_used_for_training': True, 'manual_visual_review_complete': False,
        'all_glyph_mean_mae': float(np.mean(all_errors)), 'all_glyph_p95_mae': float(np.quantile(all_errors, .95)),
        'all_glyph_max_mae': float(np.max(all_errors)),
        'finite_vocabulary_only': True, 'whole_app_quality_accepted': False}
    save_checkpoint(args.output / 'inference.pt', model, args.steps,
        heldout_validated=validated, validation=validation)
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'random_initialization': True,
        'pretrained_weights': False, 'parameters': model.parameter_count, 'vocabulary_size': len(vocabulary),
        'steps': args.steps, 'elapsed_seconds': round(time.perf_counter() - started, 2),
        'heldout_validated': validated, 'validation': validation, 'history': history,
        'main_training_modified': False, 'app_modified': False, 'runtime_modified': False,
        'inference_uses_font_or_bitmap_assets': False, 'quality_scope': 'finite glyph decoder only'}
    atomic_json(args.output / 'report.json', report)
    print(json.dumps({'output': str(args.output), 'validated': validated,
        'string_mae': [round(row['mae_0_255'], 3) for row in details],
        'all_glyph_p95_mae': validation['all_glyph_p95_mae'], 'elapsed': report['elapsed_seconds']}), flush=True)
    return 0 if validated else 2

if __name__ == '__main__':
    sys.exit(main())
