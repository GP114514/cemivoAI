"""Inspect a real neural checkpoint with pixel diagnostics and warm latency.

Run this as an independent process after pausing training. It reads weights and
validation data without changing them. The reported MSE, image variation and
latency are diagnostics, not accuracy, OCR scores, or a judgment of image quality.

Example:
  python -m neural.evaluate --samples 8 --steps 8,16,32 --output neural/evaluation
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .dataset import Dataset
from .model import flow_training_pair, tokenize
from .sample import NeuralGenerator
from .locking import model_access


ROOT=Path(__file__).resolve().parent.parent
PROMPTS=(
    ("chat","黑白聊天窗口，联系人、聊天气泡、输入框和发送按钮"),
    ("login","Black and white login page with username password and sign in button"),
    ("shapes","黑白基础几何图形，圆形、矩形、三角形和直线"),
    ("text","黑白文字海报，大标题写着你好世界，下面有副标题"),
)


def atomic_json(path,data):
    path=Path(path)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(temporary,path)


def synchronize(device):
    if device.type=="cuda":
        torch.cuda.synchronize(device)


def hash_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block=handle.read(1024*1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def summarize_latency(samples):
    values=np.array(samples,dtype=np.float64)
    return {"samples":len(samples),"p50_ms":round(float(np.median(values)),3),"p95_ms":round(float(np.percentile(values,95)),3),"min_ms":round(float(values.min()),3),"max_ms":round(float(values.max()),3),"all_ms":[round(float(value),3) for value in values]}


def image_diagnostics(image):
    pixels=np.array(image.convert("L"),dtype=np.uint8)
    return {"nonwhite_fraction_threshold_250":round(float(np.mean(pixels<250)),6),"near_black_fraction_threshold_5":round(float(np.mean(pixels<=5)),6),"pixel_mean_0_255":round(float(pixels.mean()),4),"pixel_std_0_255":round(float(pixels.std()),4),"pixel_std_normalized_0_1":round(float(pixels.std()/255),6),"sha256_pixels":hashlib.sha256(pixels.tobytes()).hexdigest()}


@torch.inference_mode()
def caption_ablation(generator,dataset,sample_count,seed,size):
    """Paired RF velocity MSE, with identical pixels/noise/time in all conditions."""
    rng=np.random.default_rng(seed)
    indices=rng.choice(len(dataset),size=min(sample_count,len(dataset)),replace=False)
    rows=[dataset[int(index)] for index in indices]
    if len(rows)<2:
        raise ValueError("Caption shuffling requires at least two validation examples")
    # A cyclic permutation has no fixed index. Prefer an offset with no equal
    # captions so the control is never accidentally identical to its target.
    offset=1
    for candidate in range(1,len(rows)):
        if all(rows[i]["prompt"]!=rows[(i+candidate)%len(rows)]["prompt"] for i in range(len(rows))):
            offset=candidate
            break
    per_sample=[]
    device=generator.device
    dtype=generator.dtype
    for position,row in enumerate(rows):
        paired=rows[(position+offset)%len(rows)]
        image=torch.from_numpy(np.array(row["image"],copy=True)).unsqueeze(0).to(device=device,dtype=torch.float32)
        if tuple(image.shape[-2:])!=(size,size):
            image=F.interpolate(image,size=(size,size),mode="bilinear",align_corners=False,antialias=True)
        local_seed=int(seed+1000+position)
        noise_rng=torch.Generator(device=device).manual_seed(local_seed)
        noise=torch.randn(image.shape,generator=noise_rng,device=device,dtype=torch.float32)
        # Spread the small evaluation set over the path without using or
        # advancing global torch RNG state.
        scalar_time=float(0.1+0.8*(position+0.5)/len(rows))
        t=torch.tensor([scalar_time],device=device,dtype=torch.float32)
        state,target,_=flow_training_pair(image,noise,t)
        conditions={"correct":row["prompt"],"shuffled":paired["prompt"],"blank":""}
        losses={}
        for condition,prompt in conditions.items():
            tokens=tokenize(prompt,generator.config.text_max_length,device=device)
            prediction=generator.model(state.to(dtype=dtype),t,tokens)
            loss=F.mse_loss(prediction.float(),target.float())
            losses[condition]=float(loss.cpu())
        per_sample.append({"validation_index":int(row["index"]),"dataset_position":int(indices[position]),"kind":row["kind"],"correct_caption":row["prompt"],"shuffled_caption":paired["prompt"],"shuffled_kind":paired["kind"],"noise_seed":local_seed,"time":round(scalar_time,6),"velocity_mse":losses,"shuffled_minus_correct_mse":losses["shuffled"]-losses["correct"],"blank_minus_correct_mse":losses["blank"]-losses["correct"]})
    means={condition:float(np.mean([row["velocity_mse"][condition] for row in per_sample])) for condition in ("correct","shuffled","blank")}
    return {"examples":len(per_sample),"split":"val","native_evaluation_size":size,"time_convention":"x_t=(1-t)*noise+t*image; target=image-noise","method":"One identical image/noise/time tuple per example; only the text tokens change. Unweighted mean pixel velocity MSE.","mean_velocity_mse":means,"mean_shuffled_minus_correct_mse":means["shuffled"]-means["correct"],"mean_blank_minus_correct_mse":means["blank"]-means["correct"],"interpretation":"Positive control-minus-correct differences are a limited conditioning diagnostic. They do not establish correct buttons, legible text, semantic understanding, or overall image quality.","per_sample":per_sample}


def generate_grid_and_benchmark(generator,step_counts,output,size,repeats):
    cells=[]
    pixel_reports=[]
    latency_reports=[]
    conditioning_reports=[]
    fixed_seed=42
    for sampling_steps in step_counts:
        print(f"Generating fixed prompts at {sampling_steps} steps...",flush=True)
        row_images=[]
        for name,prompt in PROMPTS:
            image=generator.generate(prompt,seed=fixed_seed,steps=sampling_steps,size=size)
            filename=f"{name}-s{sampling_steps:03}-seed{fixed_seed}.png"
            image.save(output/filename)
            row_images.append(image)
            pixel_reports.append({"name":name,"prompt":prompt,"seed":fixed_seed,"sampling_steps":sampling_steps,"image":filename,"size":[size,size],**image_diagnostics(image)})
        cells.extend(row_images)
        arrays=[np.array(image,dtype=np.float32)/255 for image in row_images]
        pairwise=[]
        for first in range(len(arrays)):
            for second in range(first+1,len(arrays)):
                pairwise.append({"first":PROMPTS[first][0],"second":PROMPTS[second][0],"mean_absolute_pixel_difference_0_1":float(np.mean(np.abs(arrays[first]-arrays[second])))})
        conditioning_reports.append({"sampling_steps":sampling_steps,"seed":fixed_seed,"mean_absolute_pixel_difference_across_prompt_pairs_0_1":float(np.mean([pair["mean_absolute_pixel_difference_0_1"] for pair in pairwise])),"pairwise":pairwise,"interpretation":"Same initial noise, different prompt text. A difference shows sensitivity, not correct adherence; no difference may indicate ignored conditioning or an early checkpoint."})
        # Warm each sampling configuration once before collecting timing. The
        # model has already loaded, and no disk write belongs to these timings.
        name,prompt=PROMPTS[0]
        generator.generate(prompt,seed=fixed_seed,steps=sampling_steps,size=size)
        neural_times=[]
        end_to_end_times=[]
        hashes=[]
        for repeat in range(repeats):
            synchronize(generator.device)
            start=time.perf_counter()
            image=generator.generate(prompt,seed=fixed_seed,steps=sampling_steps,size=size)
            synchronize(generator.device)
            elapsed_ms=(time.perf_counter()-start)*1000
            end_to_end_times.append(elapsed_ms)
            neural_times.append(float(generator.last_generation_seconds)*1000)
            hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
        latency_reports.append({"sampling_steps":sampling_steps,"size":[size,size],"batch_size":1,"seed":fixed_seed,"prompt":prompt,"warmup_generations":1,"warm_neural_generation":summarize_latency(neural_times),"warm_generate_to_pil":summarize_latency(end_to_end_times),"same_seed_pixel_hashes_equal":len(set(hashes))==1,"timing_definition":{"warm_neural_generation":"Includes text encoding and all neural sampling steps, with CUDA synchronized by NeuralGenerator; excludes model loading, GPU-to-CPU image transfer, PIL conversion, and PNG writes.","warm_generate_to_pil":"Includes text encoding, neural sampling, CUDA synchronization, GPU-to-CPU transfer, and PIL conversion; excludes model loading and PNG writes."}})
    grid=Image.new("L",(len(PROMPTS)*size,len(step_counts)*size),255)
    for index,image in enumerate(cells):
        grid.paste(image,(index%len(PROMPTS)*size,index//len(PROMPTS)*size))
    grid.save(output/"grid.png")
    return {"grid":"grid.png","grid_columns":[{"name":name,"prompt":prompt} for name,prompt in PROMPTS],"grid_rows_sampling_steps":step_counts,"images":pixel_reports,"warm_latency":latency_reports,"same_seed_different_prompt_pixel_differences":conditioning_reports}


def evaluate(args):
    # Keep the lock over loading, direct caption-ablation forwards, sampling,
    # timings and checkpoint hashing. Nested NeuralGenerator calls reenter it.
    with model_access(Path(args.checkpoint).resolve().parent):
        return _evaluate_locked(args)


def _evaluate_locked(args):
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    model_load_start=time.perf_counter()
    generator=NeuralGenerator(args.checkpoint,args.device)
    model_load_seconds=time.perf_counter()-model_load_start
    size=int(args.size or generator.native_size)
    dataset=Dataset(args.data,split="val",max_length=generator.config.text_max_length)
    if len(dataset)<2:
        raise ValueError("Need at least two held-out validation examples")
    if generator.device.type=="cuda":
        torch.cuda.reset_peak_memory_stats(generator.device)
    print(f"Checkpoint step {generator.checkpoint_step}; {generator.parameter_count:,} parameters; native size {generator.native_size}; evaluating {size}px",flush=True)
    ablation=caption_ablation(generator,dataset,args.samples,args.seed,size)
    generated=generate_grid_and_benchmark(generator,args.step_counts,output,size,args.repeats)
    report={"report_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"evaluation_process_id":os.getpid(),"checkpoint":str(generator.checkpoint_path.resolve()),"checkpoint_sha256":hash_file(generator.checkpoint_path),"checkpoint_step":generator.checkpoint_step,"parameters":generator.parameter_count,"native_training_size":[generator.native_size,generator.native_size],"evaluation_size":[size,size],"resized_or_non_native":size!=generator.native_size,"provenance":generator.provenance,"device":str(generator.device),"device_name":torch.cuda.get_device_name(generator.device) if generator.device.type=="cuda" else "CPU","model_dtype":str(generator.dtype),"torch_version":torch.__version__,"cuda_runtime":torch.version.cuda,"model_load_seconds":round(model_load_seconds,4),"validation_dataset":str(Path(args.data).resolve()),"validation_examples_available":len(dataset),"caption_ablation":ablation,**generated,"elapsed_seconds":round(time.perf_counter()-started,3),"limitations":["All values are diagnostics, not quality scores, accuracy rates, OCR correctness, or proof of usable generation.","A low RF velocity MSE can coexist with blank, blurry, malformed, or unreadable images.","Only a small locally synthesized validation subset is tested; it does not establish real-world generalization.","Validation images/captions come from the same synthetic generator family as training data, though the split differs.","Prompt sensitivity does not establish semantic adherence, correct geometry, or readable button text.","Nonwhite fraction and pixel standard deviation detect simple degeneracy but cannot judge layout or language.","Latency depends on device, resolution, sampling steps, warm state, software, and concurrent workloads.","No pretrained scoring network, OCR system, or human review is used by this script.","Grid cells are direct neural pixel output. Compositing the grid adds no UI templates, text, or line drawings.","This independent process does not access or modify the training process RNG, checkpoint, dataset, optimizer, or status files."]}
    if generator.device.type=="cuda":
        report["evaluation_peak_gpu_memory_mb"]=round(torch.cuda.max_memory_allocated(generator.device)/1024**2,2)
    atomic_json(output/"evaluation.json",report)
    print(json.dumps({"report":str(output/"evaluation.json"),"grid":str(output/"grid.png"),"checkpoint_step":generator.checkpoint_step,"parameters":generator.parameter_count,"caption_mean_velocity_mse":ablation["mean_velocity_mse"],"elapsed_seconds":report["elapsed_seconds"]},ensure_ascii=False,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",default=str(ROOT/"neural"/"checkpoints"/"inference.pt"))
    parser.add_argument("--data",default=str(ROOT/"neural"/"data"))
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--samples",type=int,default=8,help="Number of fixed validation examples for caption ablation")
    parser.add_argument("--steps",default="8,16,32",help="Comma-separated sampling step counts")
    parser.add_argument("--repeats",type=int,default=8,help="Warm single-image timing repetitions per sampling configuration")
    parser.add_argument("--seed",type=int,default=20261025,help="Seed for validation selection and noise; grids always use seed42")
    parser.add_argument("--size",type=int,help="Defaults to checkpoint native training size")
    parser.add_argument("--output",default=str(ROOT/"neural"/"evaluation"))
    args=parser.parse_args()
    try:
        args.step_counts=list(dict.fromkeys(int(value.strip()) for value in args.steps.split(",") if value.strip()))
    except ValueError:
        parser.error("--steps must be comma-separated integers")
    if not args.step_counts or any(step<1 or step>128 for step in args.step_counts):
        parser.error("Each sampling step count must be in 1..128")
    if args.samples<2 or args.repeats<2:
        parser.error("--samples and --repeats must be at least 2")
    if args.size is not None and (args.size<32 or args.size>1024 or args.size%8):
        parser.error("--size must be a multiple of 8 in 32..1024")
    evaluate(args)


if __name__=="__main__":
    main()
