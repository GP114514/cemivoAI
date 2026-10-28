"""Compare genuine neural pixels at CFG scales 1, 2 and 4 after pausing training.

No scoring network, OCR, templates, target images or drawing fallback is used.
The grid contains unmodified neural image cells; row/column labels live in JSON.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .locking import model_access
from .sample import NeuralGenerator

ROOT=Path(__file__).resolve().parent.parent
PROMPTS=(
    ("chat","黑白聊天窗口，联系人、聊天气泡、输入框和发送按钮"),
    ("login","Black and white login page with username password and sign in button"),
    ("settings","黑白设置页面，通知开关与保存按钮"),
    ("diagram","Black and white flowchart with boxes and arrows"),
)
SCALES=(1.0,2.0,4.0)
SEED=42


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def compare(args):
    checkpoint=Path(args.checkpoint).resolve()
    # The same checkpoint/GPU lock spans loading, every condition and report
    # capture. Training cannot resume between cells of this comparison.
    with model_access(checkpoint.parent):
        output=Path(args.output).resolve()
        output.mkdir(parents=True,exist_ok=True)
        load_start=time.perf_counter()
        generator=NeuralGenerator(checkpoint,args.device)
        model_load_seconds=time.perf_counter()-load_start
        size=generator.native_size
        grid=Image.new('L',(len(PROMPTS)*size,len(SCALES)*size),255)
        reports=[]
        arrays={}
        started=time.perf_counter()
        for row,scale in enumerate(SCALES):
            for column,(name,prompt) in enumerate(PROMPTS):
                start=time.perf_counter()
                image=generator.generate(prompt,seed=SEED,steps=args.steps,size=size,guidance=scale)
                wall_ms=(time.perf_counter()-start)*1000
                filename=f'{name}-g{scale:g}-s{args.steps:03}-seed{SEED}.png'
                image.save(output/filename)
                grid.paste(image,(column*size,row*size))
                pixels=np.asarray(image,dtype=np.uint8)
                arrays[(scale,name)]=pixels.astype(np.float32)/255
                reports.append({'kind':name,'prompt':prompt,'guidance':scale,'seed':SEED,
                    'sampling_steps':args.steps,'image':filename,
                    'neural_sampling_ms':round(generator.last_generation_seconds*1000,3),
                    'generate_to_pil_ms':round(wall_ms,3),
                    'sha256_pixels':hashlib.sha256(pixels.tobytes()).hexdigest(),
                    'pixel_mean_0_255':round(float(pixels.mean()),4),
                    'pixel_std_0_255':round(float(pixels.std()),4)})
                print(f'{name}: guidance={scale:g} steps={args.steps} {wall_ms:.1f}ms',flush=True)
        grid.save(output/'grid.png')
        differences=[]
        for scale in SCALES[1:]:
            for name,_ in PROMPTS:
                difference=arrays[(scale,name)]-arrays[(1.0,name)]
                differences.append({'kind':name,'guidance':scale,'reference_guidance':1.0,
                    'mean_absolute_pixel_difference_0_1':float(np.mean(np.abs(difference))),
                    'root_mean_squared_pixel_difference_0_1':float(np.sqrt(np.mean(difference*difference)))})
        report={'report_version':1,'created_at':datetime.now(timezone.utc).isoformat(),
            'checkpoint':str(checkpoint),'checkpoint_sha256':file_hash(checkpoint),
            'checkpoint_step':generator.checkpoint_step,'parameters':generator.parameter_count,
            'native_size':[size,size],'device':str(generator.device),'dtype':str(generator.dtype),
            'torch_version':torch.__version__,'seed':SEED,'sampling_steps':args.steps,
            'grid':'grid.png','grid_rows_guidance':list(SCALES),
            'grid_columns':[{'kind':name,'prompt':prompt} for name,prompt in PROMPTS],
            'model_load_seconds':round(model_load_seconds,4),'images':reports,
            'pixel_differences_from_guidance_1':differences,
            'elapsed_sampling_and_saving_seconds':round(time.perf_counter()-started,4),
            'method':'Identical seed/noise and prompt per column; only CFG scale changes. v=unconditional+scale*(conditional-unconditional). Empty caption equals training caption-dropout tokens.',
            'limitations':['Pixel differences and timings are diagnostics, not accuracy, OCR correctness, prompt adherence or image-quality scores.',
                'Higher guidance does not guarantee better images or readable text.',
                'Guidance 1 uses a single conditional forward; other scales use one double-sized batched forward per sampling step.',
                'Timings are one observation per condition and can include first-use overhead; PNG saving and grid assembly are excluded from per-image timings.',
                'All grid cells are direct neural output; no training images, templates, line drawing or text drawing are substituted.']}
        temporary=output/'comparison.json.tmp'
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(output/'comparison.json')
        print(f'Saved {output / "grid.png"} and {output / "comparison.json"}',flush=True)
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default=str(ROOT/'neural'/'checkpoints'/'inference.pt'))
    parser.add_argument('--output',default=str(ROOT/'neural'/'guidance_comparison'))
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--steps',type=int,choices=(16,32),default=16)
    args=parser.parse_args()
    compare(args)


if __name__=='__main__':
    main()
