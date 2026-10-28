"""Train a genuine text-conditioned pixel generator from random weights.

Example: python -m neural.train --data neural/data --steps 10000 --batch-size 4
    --accum 4 --save-every 500 --sample-every 500

No pretrained components are used. The first checkpoint is an early research
checkpoint, not evidence that usable text or image generation has converged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .model import FORMAT_VERSION, ModelConfig, PixelFlowUNet, flow_training_pair, sample_flow, tokenize
from .locking import model_access
from .readability import edge_weighted_flow_loss, validate_edge_weight
from .state_sampling import (resolve_interaction_fraction, state_caption_variants, STATE_MARKER,
    CAPTION_POLICY as STATE_CAPTION_POLICY, APP_FLOW_MARKER, app_flow_caption_variants,
    resolve_app_flow_fraction, validate_sampling_fractions)
from .region_objective import resolve_region_weight, metadata_region_masks, region_weighted_flow_loss
from .atomic_io import atomic_json

ROOT=Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINTS=ROOT/"neural"/"checkpoints"


def resolve_edge_weight(value, checkpoint=None):
    """Explicit CLI value wins; resume inherits; older/new runs default to zero."""
    if value is None:
        value=checkpoint.get("training_args",{}).get("edge_weight",0.0) if checkpoint is not None else 0.0
    return validate_edge_weight(value)


def resolve_caption_dropout(value, checkpoint=None):
    """Explicit CLI value wins; resume inherits; new/older runs use 0.1."""
    if value is None:
        value=checkpoint.get("training_args",{}).get("caption_dropout",0.1) if checkpoint is not None else 0.1
    try:
        if isinstance(value,bool):
            raise ValueError
        probability=float(value)
    except (TypeError,ValueError,OverflowError):
        raise ValueError("caption-dropout must be a finite probability in 0..1") from None
    if not math.isfinite(probability) or not 0<=probability<=1:
        raise ValueError("caption-dropout must be a finite probability in 0..1")
    return probability


def replace_file(source,target):
    # Windows readers may briefly hold the destination open while the workbench
    # polls it. Keep the update atomic and retry that transient sharing conflict.
    for attempt in range(20):
        try:
            os.replace(source,target)
            return
        except PermissionError:
            if attempt==19:
                raise
            time.sleep(0.05)


def atomic_torch_save(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("wb") as handle:
        torch.save(data,handle)
        handle.flush()
        os.fsync(handle.fileno())
    replace_file(temporary,path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive_training(directory):
    """Training and inference share one cross-process model-use lock."""
    with model_access(directory):
        yield


class BatchStream:
    """Random-access, memory-mapped batches with an exactly restorable sampler RNG."""
    def __init__(self,dataset,seed,caption_augment=0.0,interaction_fraction=None,app_flow_fraction=None):
        self.dataset=dataset
        self.rng=np.random.default_rng(seed)
        self.caption_augment=float(caption_augment)
        if not len(dataset):
            raise ValueError("The training dataset is empty")
        self.interaction_fraction=resolve_interaction_fraction(interaction_fraction)
        self.app_flow_fraction=resolve_app_flow_fraction(app_flow_fraction)
        validate_sampling_fractions(self.interaction_fraction,self.app_flow_fraction)
        self.state_indices=None
        self.app_flow_indices=None
        self.ordinary_indices=None
        self.state_captions={}
        self.last_batch_exposure=None
        self.last_batch_indices=()
        self.sampling_exposure={'counted_since_step':0,'batches':0,'total_samples':0,
            'category_counts':{},'family_counts':{},'app_flow_family_counts':{},'app_flow_state_counts':{}}
        if self.app_flow_fraction:
            self.state_indices=np.asarray([i for i,row in enumerate(dataset.rows) if row.get('caption_policy')==STATE_MARKER],dtype=np.int64)
            self.app_flow_indices=np.asarray([i for i,row in enumerate(dataset.rows) if row.get('caption_policy')==APP_FLOW_MARKER],dtype=np.int64)
            self.ordinary_indices=np.asarray([i for i,row in enumerate(dataset.rows) if row.get('caption_policy') not in (STATE_MARKER,APP_FLOW_MARKER)],dtype=np.int64)
            if not len(self.app_flow_indices):
                raise ValueError('app-flow-fraction requires audited V5 state-supervised rows')
            if (self.interaction_fraction or 0)>0 and not len(self.state_indices):
                raise ValueError('interaction-fraction requires state-supervised training rows')
            if self.app_flow_fraction+(self.interaction_fraction or 0)<1 and not len(self.ordinary_indices):
                raise ValueError('Sampling remainder requires ordinary training rows')
            self.state_captions={int(i):state_caption_variants(dataset.rows[int(i)]) for i in self.state_indices}
            self.state_captions.update({int(i):app_flow_caption_variants(dataset.rows[int(i)]) for i in self.app_flow_indices})
        elif self.interaction_fraction is not None:
            self.state_indices=np.asarray([i for i,row in enumerate(dataset.rows) if row.get("caption_policy")==STATE_MARKER],dtype=np.int64)
            self.ordinary_indices=np.asarray([i for i,row in enumerate(dataset.rows) if row.get("caption_policy")!=STATE_MARKER],dtype=np.int64)
            if not len(self.state_indices):
                raise ValueError("interaction-fraction requires state-supervised training rows")
            if self.interaction_fraction<1 and not len(self.ordinary_indices):
                raise ValueError("interaction-fraction below 1 requires ordinary training rows")
            self.state_captions={int(index):state_caption_variants(dataset.rows[int(index)]) for index in self.state_indices}

    def batch(self,batch_size,device,with_regions=False):
        if self.app_flow_fraction:
            choices=self.rng.random(batch_size)
            choose_flow=choices<self.app_flow_fraction
            choose_state=(choices>=self.app_flow_fraction)&(choices<self.app_flow_fraction+(self.interaction_fraction or 0))
            indices=np.empty(batch_size,dtype=np.int64)
            for selected,pool in ((choose_flow,self.app_flow_indices),(choose_state,self.state_indices),
                    (~(choose_flow|choose_state),self.ordinary_indices)):
                count=int(selected.sum())
                if count:
                    indices[selected]=pool[self.rng.integers(0,len(pool),size=count)]
        elif self.interaction_fraction is None:
            # Preserve every RNG draw in the historical uniform sampler.
            indices=self.rng.integers(0,len(self.dataset),size=batch_size)
        else:
            choose_state=self.rng.random(batch_size)<self.interaction_fraction
            indices=np.empty(batch_size,dtype=np.int64)
            count=int(choose_state.sum())
            if count:
                indices[choose_state]=self.state_indices[self.rng.integers(0,len(self.state_indices),size=count)]
            if count<batch_size:
                indices[~choose_state]=self.ordinary_indices[self.rng.integers(0,len(self.ordinary_indices),size=batch_size-count)]
        rows=[self.dataset[int(index)] for index in indices]
        if self.interaction_fraction is not None or self.app_flow_fraction:
            from .dataset import encode_prompt
            for position,index in enumerate(indices):
                variants=self.state_captions.get(int(index))
                if variants is not None:
                    variant=int(self.rng.integers(0,3))
                    if variant:
                        caption=variants[variant-1]
                        rows[position]=dict(rows[position],prompt=caption,tokens=encode_prompt(caption,self.dataset.max_length))
        if self.caption_augment:
            from .captions import augment_caption
            from .dataset import encode_prompt
            for position,index in enumerate(indices):
                # State-supervised pairs must keep exact empty/composer/toggle
                # conditions. Generic family augmentation can contradict them.
                if (self.dataset.rows[int(index)].get("caption_policy") == STATE_MARKER
                        or (self.app_flow_fraction and self.dataset.rows[int(index)].get('caption_policy') == APP_FLOW_MARKER)):
                    continue
                if self.rng.random()<self.caption_augment:
                    caption=augment_caption(self.dataset.rows[int(index)],self.rng)
                    rows[position]=dict(rows[position],prompt=caption,
                        tokens=encode_prompt(caption,self.dataset.max_length))
        images=torch.from_numpy(np.stack([row["image"] for row in rows]))
        tokens=torch.from_numpy(np.stack([row["tokens"] for row in rows]))
        if device.type=="cuda":
            images=images.pin_memory()
            tokens=tokens.pin_memory()
        images=images.to(device,non_blocking=True)
        tokens=tokens.to(device,non_blocking=True)
        if with_regions:
            height,width=rows[0]['image'].shape[-2:]
            masks=[metadata_region_masks(self.dataset.rows[int(index)],height,width) for index in indices]
            regions={name:torch.from_numpy(np.stack([row[name] for row in masks])).to(device,non_blocking=True)
                for name in ('text','state')}
            self._record_exposure(indices)
            return images,tokens,regions
        self._record_exposure(indices)
        return images,tokens

    def _record_exposure(self,indices):
        """Count emitted images without any extra sampler or global RNG draws."""
        categories=Counter();families=Counter();flow_families=Counter();flow_states=Counter()
        for index in indices:
            row=self.dataset.rows[int(index)];marker=row.get('caption_policy')
            category='app_flow' if marker==APP_FLOW_MARKER else 'interaction' if marker==STATE_MARKER else 'ordinary'
            categories[category]+=1;families[row.get('kind','unknown')]+=1
            if marker==APP_FLOW_MARKER:
                flow_families[row['flow_family']]+=1
                flow_states[row['flow_family']+':'+row['flow_stage']]+=1
        batch={'total_samples':len(indices),'category_counts':dict(categories),'family_counts':dict(families),
            'app_flow_family_counts':dict(flow_families),'app_flow_state_counts':dict(flow_states)}
        self.last_batch_indices=tuple(int(index) for index in indices)
        self.last_batch_exposure=batch
        self.sampling_exposure['batches']+=1
        self.sampling_exposure['total_samples']+=len(indices)
        for key in ('category_counts','family_counts','app_flow_family_counts','app_flow_state_counts'):
            total=Counter(self.sampling_exposure[key]);total.update(batch[key]);self.sampling_exposure[key]=dict(total)

    def exposure_snapshot(self):
        return json.loads(json.dumps(self.sampling_exposure))

    def restore_exposure(self,saved=None,*,counted_since_step=0):
        if saved is None:
            self.sampling_exposure['counted_since_step']=int(counted_since_step)
            return
        if (not isinstance(saved,dict) or type(saved.get('total_samples')) is not int
                or saved['total_samples']<0 or type(saved.get('batches')) is not int or saved['batches']<0
                or type(saved.get('counted_since_step')) is not int or saved['counted_since_step']<0):
            raise ValueError('Invalid saved actual-sampling exposure counters')
        for key in ('category_counts','family_counts','app_flow_family_counts','app_flow_state_counts'):
            counts=saved.get(key)
            if not isinstance(counts,dict) or any(not isinstance(k,str) or type(v) is not int or v<0 for k,v in counts.items()):
                raise ValueError('Invalid saved actual-sampling exposure counters')
        if (sum(saved['category_counts'].values())!=saved['total_samples']
                or sum(saved['family_counts'].values())!=saved['total_samples']
                or sum(saved['app_flow_family_counts'].values())!=saved['category_counts'].get('app_flow',0)
                or sum(saved['app_flow_state_counts'].values())!=saved['category_counts'].get('app_flow',0)):
            raise ValueError('Saved actual-sampling exposure counts disagree')
        self.sampling_exposure=json.loads(json.dumps(saved))


def capture_rng(stream):
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,"sampler":stream.rng.bit_generator.state}


def restore_rng(state,stream):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    stream.rng.bit_generator.state=state["sampler"]


@contextmanager
def use_ema(model,ema):
    """Temporarily swap tensor storage, without allocating another full GPU model."""
    original={name:parameter.data for name,parameter in model.named_parameters()}
    was_training=model.training
    try:
        for name,parameter in model.named_parameters():
            parameter.data=ema[name]
        model.eval()
        yield
    finally:
        for name,parameter in model.named_parameters():
            parameter.data=original[name]
        model.train(was_training)


def current_resolution(step,args):
    return args.warmup_resolution if step<args.warmup_steps else args.resolution


def save_checkpoints(directory,model,ema,optimizer,stream,step,samples_seen,elapsed,args,resolution):
    provenance={"random_initialization":True,"pretrained_weights":False,"generator":"pixel-space conditional rectified flow UNet","time_convention":"x_t=(1-t)*noise+t*image; v=image-noise; integrate t=0->1","dataset":str(Path(args.data).resolve()),"training_seed":args.seed,"caption_augmentation_probability":args.caption_augment,"caption_dropout_probability":args.caption_dropout,"edge_weight":args.edge_weight}
    provenance["interaction_fraction"]=getattr(args,"interaction_fraction",None)
    provenance["interaction_caption_policy"]=STATE_CAPTION_POLICY if getattr(args,"interaction_fraction",None) is not None else "verbatim-original"
    provenance['app_flow_fraction']=getattr(args,'app_flow_fraction',None)
    provenance['app_flow_caption_policy']=APP_FLOW_MARKER if getattr(args,'app_flow_fraction',None) else None
    exposure=stream.exposure_snapshot() if stream is not None and hasattr(stream,'exposure_snapshot') else None
    for name in ('text_roi_weight','state_roi_weight'):
        provenance[name]=getattr(args,name,0.0)
    common={"format_version":FORMAT_VERSION,"model_config":model.config.to_dict(),"step":step,"samples_seen":samples_seen,"elapsed_seconds":elapsed,"native_size":resolution,"provenance":provenance,"sampling_exposure":exposure}
    # Optimizer and all RNG states are retained only in the resumable file.
    full={**common,"checkpoint_kind":"resumable-training","model":model.state_dict(),"ema":ema,"optimizer":optimizer.state_dict(),"rng":capture_rng(stream),"training_args":vars(args)}
    atomic_torch_save(directory/"latest.pt",full)
    inference={**common,"checkpoint_kind":"inference-ema","model":{name:value.detach().to(device="cpu",dtype=torch.bfloat16) for name,value in ema.items()}}
    atomic_torch_save(directory/"inference.pt",inference)
    card={"name":"cemivoAI PixelFlow UNet (from scratch)","format_version":FORMAT_VERSION,"parameters":model.parameter_count,"native_width":resolution,"native_height":resolution,"checkpoint_step":step,"samples_seen":samples_seen,"random_initialization":True,"pretrained_weights":False,"text_encoder":"randomly initialized UTF-8 byte Transformer","tokenizer":"PAD0 BOS1 EOS2 bytes+3; max_length=192","architecture":model.config.to_dict(),"image_generation":"Neural pixel output only. No templates or renderer are called at inference.","training_data":"Locally synthesized monochrome UI and geometric image/caption pairs.","development_stage":"early-training" if step<10000 else "training-checkpoint; quality requires independent inspection","limitations":["Parameter count does not establish image quality or language understanding.","Early checkpoints may generate noise, empty images, malformed layouts or unreadable text.","Precise Chinese/English lettering and unseen interface concepts are not guaranteed.","Native resolution is the training resolution; resizing does not add learned detail.","Fast 1-4 step inference requires additional validation or training; default quality checks use more steps."],"updated_at":utc_now()}
    card["caption_augmentation_probability"]=args.caption_augment
    card["caption_dropout_probability"]=args.caption_dropout
    card["interaction_fraction"]=getattr(args,"interaction_fraction",None)
    card["interaction_caption_policy"]=provenance["interaction_caption_policy"]
    card['app_flow_fraction']=getattr(args,'app_flow_fraction',None)
    card['app_flow_caption_policy']=provenance['app_flow_caption_policy']
    card['sampling_exposure']=exposure
    card["edge_weight"]=args.edge_weight
    for name in ('text_roi_weight','state_roi_weight'):
        card[name]=getattr(args,name,0.0)
    atomic_json(directory/"model_card.json",card)


def save_preview(directory,model,ema,step,resolution,device,amp_enabled,steps=8):
    prompts=["黑白聊天窗口，联系人、聊天气泡、输入框和发送按钮", "Black and white login page with username password and sign in", "黑白设置页面，通知开关与保存按钮", "Black and white flowchart with boxes and arrows"]
    with use_ema(model,ema),torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp_enabled):
        images=sample_flow(model,prompts,seed=20261025,steps=steps,size=resolution,device=device)
    pixels=((images[:,0].cpu().numpy()+1)*127.5).round().clip(0,255).astype(np.uint8)
    canvas=Image.new("L",(resolution*2,resolution*2),255)
    for i,pixel in enumerate(pixels):
        canvas.paste(Image.fromarray(pixel),(i%2*resolution,i//2*resolution))
    output=directory/"previews"
    output.mkdir(parents=True,exist_ok=True)
    temporary=output/"latest.tmp.png"
    canvas.save(temporary,format="PNG")
    replace_file(temporary,output/"latest.png")
    canvas.save(output/f"step-{step:08}.png")
    atomic_json(output/"latest.json",{"step":step,"native_size":resolution,"sampling_steps":steps,"seed":20261025,"prompts":prompts,"source":"neural EMA weights; not training targets or procedural drawings"})


def smoke_test():
    """Small CPU test of real gradients, loss descent, checkpoint reload, and seeds."""
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config=ModelConfig(base_channels=32,channel_multipliers=(1,2),num_res_blocks=1,time_dim=128,text_dim=64,text_layers=1,text_heads=4,attention_heads=4,gradient_checkpointing=False)
    model=PixelFlowUNet(config)
    tokens=tokenize(["chat window send button","登录页面"])
    images=torch.ones(2,1,32,32)
    images[:,:,5:7,4:28]=-1
    images[:,:,10:25,5:7]=-1
    images[:,:,25:27,4:28]=-1
    noise=torch.randn_like(images)
    t=torch.full((2,),0.55)
    x,target,_=flow_training_pair(images,noise,t)
    assert torch.equal(flow_training_pair(images,noise,torch.zeros(2))[0],noise)
    assert torch.equal(flow_training_pair(images,noise,torch.ones(2))[0],images)
    assert torch.allclose(noise+target,images,atol=1e-6)
    opt=torch.optim.AdamW(model.parameters(),lr=0.001)
    losses=[]
    initial_output=model.output.weight.detach().clone()
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        prediction=model(x,t,tokens)
        loss=F.mse_loss(prediction,target)
        loss.backward()
        assert torch.isfinite(loss) and model.output.weight.grad is not None
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1]<losses[0],f"loss did not descend: {losses}"
    assert not torch.equal(initial_output,model.output.weight)
    model.eval()
    a=sample_flow(model,"chat window",seed=9,steps=2,size=32)
    b=sample_flow(model,"chat window",seed=9,steps=2,size=32)
    assert torch.equal(a,b),"same-seed generation must repeat on the same CPU"
    with tempfile.TemporaryDirectory(prefix="cemivo-flow-smoke-") as temporary:
        path=Path(temporary)/"test.pt"
        atomic_torch_save(path,{"format_version":FORMAT_VERSION,"model_config":config.to_dict(),"model":model.state_dict(),"step":10,"native_size":32})
        loaded=torch.load(path,map_location="cpu",weights_only=False)
        restored=PixelFlowUNet(ModelConfig.from_dict(loaded["model_config"]))
        restored.load_state_dict(loaded["model"])
        restored.eval()
        c=sample_flow(restored,"chat window",seed=9,steps=2,size=32)
        assert torch.equal(a,c),"checkpoint reload changed the output"
    # Count all default production parameters without running a GPU operation.
    production=PixelFlowUNet(ModelConfig())
    result={"passed":True,"test_device":"cpu","test_model_parameters":model.parameter_count,"production_parameters":production.parameter_count,"loss_before":losses[0],"loss_after":losses[-1],"checks":["genuine gradient update","fixed-batch loss descent","flow t=0 and t=1 endpoints","noise plus target velocity equals image","same seed repeatability","checkpoint load output equality","fixed UTF-8 tokenizer"]}
    del production
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def train(args):
    from .dataset import Dataset
    directory=Path(args.checkpoints).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    status_path=directory/"status.json"
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; install a supported PyTorch CUDA build")
    if device.type=="cpu" and not args.allow_cpu:
        raise RuntimeError("Production training requires the GPU. Use --allow-cpu only for small diagnostics or --smoke-test.")
    amp_enabled=device.type=="cuda" and not args.fp32
    if amp_enabled and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This run requests BF16 but the CUDA device reports no BF16 support")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type=="cuda":
        torch.cuda.manual_seed_all(args.seed)
        # At 512, autotuning tried temporary convolution workspaces that left
        # almost no room for the separate live app on this 16 GB GPU. Use the
        # bounded heuristic path; 256 training keeps its historical setting.
        torch.backends.cudnn.benchmark=args.resolution <= 256
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    torch.set_num_threads(args.cpu_threads)
    initial=time.perf_counter()
    step=0
    samples_seen=0
    previous_elapsed=0.0
    loss_value=None
    model=optimizer=ema=stream=None
    stop_requested=False

    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
        print("Stop requested; finishing the current optimizer step and saving.",flush=True)

    signal.signal(signal.SIGINT,request_stop)
    if hasattr(signal,"SIGTERM"):
        signal.signal(signal.SIGTERM,request_stop)

    def elapsed():
        return previous_elapsed+time.perf_counter()-initial

    def status(state,**extra):
        data={"state":state,"step":step,"total_steps":args.steps,"loss":loss_value,"samples_seen":samples_seen,"updated_at":utc_now(),"elapsed_seconds":round(elapsed(),3),"parameters":model.parameter_count if model is not None else None,"device":str(device),"native_resolution":current_resolution(max(step-1,0),args),"batch_size":args.batch_size,"gradient_accumulation":args.accum,"random_initialization":True,"pretrained_weights":False,"quality_verified":False,"pid":os.getpid(),"data_path":str(Path(args.data).resolve()),"caption_augmentation_probability":args.caption_augment}
        # None while preparing means a resume checkpoint has not been inspected.
        data["edge_weight"]=getattr(args,"edge_weight",None)
        data["caption_dropout_probability"]=getattr(args,"caption_dropout",None)
        data["interaction_fraction"]=getattr(args,"interaction_fraction",None)
        data["interaction_caption_policy"]=STATE_CAPTION_POLICY if getattr(args,"interaction_fraction",None) is not None else "verbatim-original"
        data['app_flow_fraction']=getattr(args,'app_flow_fraction',None)
        data['app_flow_caption_policy']=APP_FLOW_MARKER if getattr(args,'app_flow_fraction',None) else None
        data['sampling_exposure']=stream.exposure_snapshot() if stream is not None else None
        for name in ('text_roi_weight','state_roi_weight'):
            data[name]=getattr(args,name,None)
        if device.type=="cuda":
            data["gpu_memory_allocated_mb"]=round(torch.cuda.memory_allocated()/1024**2,1)
            data["gpu_memory_peak_mb"]=round(torch.cuda.max_memory_allocated()/1024**2,1)
        data.update(extra)
        atomic_json(status_path,data)

    try:
        status("preparing")
        loaded=None
        if args.resume:
            resume_path=Path(args.resume) if args.resume!="auto" else directory/"latest.pt"
            loaded=torch.load(resume_path,map_location="cpu",weights_only=False)
            if loaded.get("format_version")!=FORMAT_VERSION or "optimizer" not in loaded:
                raise ValueError("--resume requires a compatible resumable latest.pt checkpoint")
            config=ModelConfig.from_dict(loaded["model_config"])
        else:
            if args.config:
                config=ModelConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
                config.gradient_checkpointing=not args.no_checkpointing
            else:
                config=ModelConfig(base_channels=args.base_channels,time_dim=args.base_channels*4,text_dim=args.text_dim,text_layers=args.text_layers,text_heads=args.text_heads,gradient_checkpointing=not args.no_checkpointing)
        if args.caption_augment is None:
            args.caption_augment=float(loaded.get("training_args",{}).get("caption_augment",0.0) or 0.0) if loaded else 0.6
        if not 0<=args.caption_augment<=1 or not math.isfinite(args.caption_augment):
            raise ValueError("caption-augment must be a finite probability in 0..1")
        args.caption_dropout=resolve_caption_dropout(args.caption_dropout,loaded)
        args.edge_weight=resolve_edge_weight(getattr(args,"edge_weight",None),loaded)
        for name in ('text_roi_weight','state_roi_weight'):
            setattr(args,name,resolve_region_weight(getattr(args,name,None),loaded,name))
        args.interaction_fraction=resolve_interaction_fraction(getattr(args,"interaction_fraction",None),loaded,
            uniform_sampling=getattr(args,"uniform_sampling",False))
        args.app_flow_fraction=resolve_app_flow_fraction(getattr(args,'app_flow_fraction',None),loaded,
            uniform_sampling=getattr(args,'uniform_sampling',False))
        validate_sampling_fractions(args.interaction_fraction,args.app_flow_fraction)
        dataset=Dataset(args.data,split="train",max_length=config.text_max_length)
        validation_dataset=Dataset(args.data,split="val",max_length=config.text_max_length) if args.validate_every else None
        stream=BatchStream(dataset,args.seed,args.caption_augment,args.interaction_fraction,args.app_flow_fraction)
        model=PixelFlowUNet(config).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,fused=device.type=="cuda")
        if loaded is not None:
            model.load_state_dict(loaded["model"])
            optimizer.load_state_dict(loaded["optimizer"])
            ema={name:value.to(device) for name,value in loaded["ema"].items()}
            step=int(loaded["step"])
            samples_seen=int(loaded["samples_seen"])
            previous_elapsed=float(loaded.get("elapsed_seconds",0))
            restore_rng(loaded["rng"],stream)
            stream.restore_exposure(loaded.get('sampling_exposure'),counted_since_step=step)
            del loaded
        else:
            ema={name:parameter.detach().clone() for name,parameter in model.named_parameters()}
        parameters=model.parameter_count
        default_architecture=config.base_channels==128 and config.text_dim==384 and config.text_layers==4 and tuple(config.channel_multipliers)==(1,2,3,4)
        if default_architecture and not 100_000_000<=parameters<=150_000_000:
            raise RuntimeError(f"Production architecture has {parameters:,} active parameters; expected 100M..150M")
        print(json.dumps({"parameters":parameters,"device":str(device),"training_samples":len(dataset),"effective_batch_size":args.batch_size*args.accum,"start_step":step,"target_step":args.steps,"bf16":amp_enabled,"edge_weight":args.edge_weight,
            'interaction_fraction':args.interaction_fraction,'app_flow_fraction':args.app_flow_fraction,
            'sampling_exposure':stream.exposure_snapshot()},ensure_ascii=False),flush=True)
        model.train()
        status("running",training_samples=len(dataset))
        history_path=directory/"training.jsonl"
        while step<args.steps:
            if stop_requested or (directory/"STOP").exists() or (directory/"stop").exists():
                break
            resolution=current_resolution(step,args)
            start_step=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accumulated=0.0
            batch_exposure=[]
            # A short LR warmup is independent from the resolution warmup.
            warmup=min(1.0,(step+1)/max(args.lr_warmup_steps,1))
            progress=min(1.0,step/max(args.steps,1))
            lr=args.lr*warmup*(0.2+0.8*0.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:
                group["lr"]=lr
            for _ in range(args.accum):
                use_regions=args.text_roi_weight>0 or args.state_roi_weight>0
                if use_regions:
                    images,tokens,regions=stream.batch(args.batch_size,device,with_regions=True)
                else:
                    images,tokens=stream.batch(args.batch_size,device)
                batch_exposure.append(stream.last_batch_exposure)
                if images.shape[-1]!=resolution:
                    images=F.interpolate(images,size=(resolution,resolution),mode="bilinear",align_corners=False,antialias=True)
                    if use_regions:
                        regions={name:F.interpolate(mask.float(),size=(resolution,resolution),mode='nearest').bool()
                            for name,mask in regions.items()}
                if args.caption_dropout>0:
                    dropped=torch.rand(len(tokens),device=device)<args.caption_dropout
                    if bool(dropped.any()):
                        tokens[dropped]=tokenize([""]*int(dropped.sum()),config.text_max_length,device)
                state,target,t=flow_training_pair(images)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp_enabled):
                    prediction=model(state,t,tokens)
                    if use_regions:
                        loss=region_weighted_flow_loss(prediction,target,images,regions,edge_weight=args.edge_weight,
                            text_weight=args.text_roi_weight,state_weight=args.state_roi_weight)
                    elif args.edge_weight:
                        loss=edge_weighted_flow_loss(prediction,target,images,args.edge_weight)
                    else:
                        # Preserve the original loss path without edge-map work.
                        loss=F.mse_loss(prediction.float(),target.float())
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite loss at step {step+1}")
                (loss/args.accum).backward()
                accumulated+=float(loss.detach())/args.accum
                samples_seen+=len(images)
            gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(f"Non-finite gradient norm at step {step+1}")
            optimizer.step()
            step+=1
            loss_value=accumulated
            decay=min(args.ema_decay,(1+step)/(10+step))
            with torch.no_grad():
                for name,parameter in model.named_parameters():
                    ema[name].lerp_(parameter.detach(),1-decay)
            if device.type=="cuda":
                torch.cuda.synchronize()
            duration=time.perf_counter()-start_step
            entry={"step":step,"loss":round(loss_value,7),"learning_rate":lr,"grad_norm":float(gradient_norm),"samples_seen":samples_seen,"native_resolution":resolution,"step_seconds":round(duration,4),"elapsed_seconds":round(elapsed(),3),"updated_at":utc_now(),"caption_augment":args.caption_augment}
            entry["edge_weight"]=args.edge_weight
            entry["caption_dropout"]=args.caption_dropout
            entry["interaction_fraction"]=args.interaction_fraction
            entry['app_flow_fraction']=args.app_flow_fraction
            entry['batch_exposure']=batch_exposure
            entry['sampling_exposure']=stream.exposure_snapshot()
            entry['text_roi_weight']=args.text_roi_weight
            entry['state_roi_weight']=args.state_roi_weight
            with history_path.open("a",encoding="utf-8") as log:
                log.write(json.dumps(entry,ensure_ascii=False)+"\n")
            if step%10==0 or step==1:
                status("running",training_samples=len(dataset),step_seconds=round(duration,4),estimated_remaining_seconds=round(max(0,args.steps-step)*duration,1))
                print(json.dumps(entry,ensure_ascii=False),flush=True)
            if args.sample_every and step%args.sample_every==0:
                save_preview(directory,model,ema,step,resolution,device,amp_enabled,args.preview_steps)
            if args.validate_every and step%args.validate_every==0:
                from .validation import validate_conditioning,write_validation
                from .state_validation import validate_state_conditioning,write_state_validation
                with use_ema(model,ema):
                    validation=validate_conditioning(model,validation_dataset,resolution,device,amp_enabled,args.validation_samples)
                    state_validation=validate_state_conditioning(model,validation_dataset,resolution,device,amp_enabled)
                write_validation(directory,validation,step)
                write_state_validation(directory,state_validation,step)
                print(json.dumps({"validation_step":step,"correct_mse":validation["correct_mse"],
                    "wrong_minus_correct_mse":validation["wrong_minus_correct_mse"],
                    "blank_minus_correct_mse":validation["blank_minus_correct_mse"]}),flush=True)
                if state_validation is not None:
                    print(json.dumps({"state_validation_step":step,"pairs":state_validation["pairs"],
                        "state_wrong_minus_correct_mse":state_validation["state_roi"]["wrong_minus_correct_mse"],
                        "state_natural_wrong_minus_natural_mse":state_validation["state_roi"]["natural_wrong_minus_natural_mse"]}),flush=True)
            if args.save_every and step%args.save_every==0:
                save_checkpoints(directory,model,ema,optimizer,stream,step,samples_seen,elapsed(),args,resolution)
        resolution=current_resolution(max(step-1,0),args)
        if step>0:
            save_checkpoints(directory,model,ema,optimizer,stream,step,samples_seen,elapsed(),args,resolution)
            if not (directory/"previews"/"latest.png").exists() or step%max(args.sample_every,1):
                save_preview(directory,model,ema,step,resolution,device,amp_enabled,args.preview_steps)
        state="complete" if step>=args.steps else "paused"
        status(state,checkpoint_step=step,training_samples=len(dataset))
        print(f"{state}: checkpoint step {step}",flush=True)
        return state
    except BaseException as error:
        # Save the last fully completed optimizer step when possible. OOM may
        # prevent preview generation, so the recovery path does not sample.
        if model is not None and optimizer is not None and ema is not None and stream is not None and step>0:
            try:
                optimizer.zero_grad(set_to_none=True)
                if device.type=="cuda":
                    torch.cuda.empty_cache()
                save_checkpoints(directory,model,ema,optimizer,stream,step,samples_seen,elapsed(),args,current_resolution(max(step-1,0),args))
            except BaseException as save_error:
                print(f"Recovery checkpoint failed: {save_error}",flush=True)
        status("failed",error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",default=str(ROOT/"neural"/"data"))
    parser.add_argument("--checkpoints",default=str(DEFAULT_CHECKPOINTS))
    parser.add_argument("--steps",type=int,default=10000)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--accum",type=int,default=4)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--weight-decay",type=float,default=1e-4)
    parser.add_argument("--lr-warmup-steps",type=int,default=100)
    parser.add_argument("--resolution",type=int,default=256)
    parser.add_argument("--warmup-resolution",type=int,default=128)
    parser.add_argument("--warmup-steps",type=int,default=500)
    parser.add_argument("--save-every",type=int,default=500)
    parser.add_argument("--sample-every",type=int,default=500)
    parser.add_argument("--preview-steps",type=int,default=8)
    parser.add_argument("--validate-every",type=int,default=1000)
    parser.add_argument("--validation-samples",type=int,default=15)
    parser.add_argument("--caption-dropout",type=float,default=None,help="Empty-caption probability in 0..1. Resume inherits its saved setting unless overridden; new/older runs default to 0.1.")
    parser.add_argument("--caption-augment",type=float,default=None,help="Probability of grounded natural-language captions. Resume inherits the saved setting unless explicitly overridden.")
    sampling=parser.add_mutually_exclusive_group()
    sampling.add_argument("--interaction-fraction",type=float,default=None,help="Probability of drawing a state-supervised example (0..1). Enables truthful short Chinese/English state captions. Resume inherits; older/new runs retain uniform sampling when omitted.")
    sampling.add_argument("--uniform-sampling",action="store_true",help="Reset state-example fraction to uniform sampling, ignoring its saved checkpoint value when switching to a dataset without state supervision.")
    parser.add_argument('--app-flow-fraction',type=float,default=None,help='Independent probability of audited V5 task/note flow examples in 0..1; interaction plus app-flow must be <=1. Omitted resumes inherit; explicit 0 disables V5 quota and preserves the historical sampler path.')
    parser.add_argument("--edge-weight",type=float,default=None,help="Optional target-edge emphasis in 0..16; default 0 for new/older runs. Resume inherits its saved value unless overridden. This changes the loss, not an OCR score.")
    parser.add_argument('--text-roi-weight',type=float,default=None,help='Extra native glyph-box loss importance in 0..16; resume inherits, older runs default to zero.')
    parser.add_argument('--state-roi-weight',type=float,default=None,help='Extra state-change-region loss importance in 0..16; resume inherits, older runs default to zero.')
    parser.add_argument("--max-grad-norm",type=float,default=1.0)
    parser.add_argument("--ema-decay",type=float,default=0.999)
    parser.add_argument("--seed",type=int,default=20261025)
    parser.add_argument("--cpu-threads",type=int,default=4)
    parser.add_argument("--base-channels",type=int,default=128,help="Active UNet width; 192/256 expand the model and need more VRAM")
    parser.add_argument("--text-dim",type=int,default=384)
    parser.add_argument("--text-layers",type=int,default=4)
    parser.add_argument("--text-heads",type=int,default=6)
    parser.add_argument("--config",help="Optional JSON ModelConfig; resume always uses the checkpoint configuration")
    parser.add_argument("--resume",nargs="?",const="auto")
    parser.add_argument("--device")
    parser.add_argument("--fp32",action="store_true")
    parser.add_argument("--no-checkpointing",action="store_true")
    parser.add_argument("--allow-cpu",action="store_true")
    parser.add_argument("--smoke-test",action="store_true")
    args=parser.parse_args()
    if args.interaction_fraction is not None:
        try:
            args.interaction_fraction=resolve_interaction_fraction(args.interaction_fraction)
        except ValueError as error:
            parser.error(str(error))
    try:
        args.app_flow_fraction=resolve_app_flow_fraction(args.app_flow_fraction,uniform_sampling=args.uniform_sampling)
        validate_sampling_fractions(args.interaction_fraction,args.app_flow_fraction)
    except ValueError as error:
        parser.error(str(error))
    if args.caption_dropout is not None:
        try:
            args.caption_dropout=resolve_caption_dropout(args.caption_dropout)
        except ValueError as error:
            parser.error(str(error))
    if args.edge_weight is not None:
        try:
            args.edge_weight=validate_edge_weight(args.edge_weight)
        except ValueError as error:
            parser.error(str(error))
    for name in ('text_roi_weight','state_roi_weight'):
        value=getattr(args,name)
        if value is not None:
            try:setattr(args,name,resolve_region_weight(value,field=name))
            except ValueError as error:parser.error(str(error))
    if args.smoke_test:
        smoke_test()
        return
    if args.steps<1 or args.batch_size<1 or args.accum<1 or args.resolution%8 or args.warmup_resolution%8:
        parser.error("steps, batch-size and accum must be positive; resolutions must divide 8")
    with exclusive_training(args.checkpoints):
        train(args)


if __name__=="__main__":
    main()
