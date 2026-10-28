"""Text-conditioned pixel-space rectified-flow UNet.

Every learned tensor is initialized locally. There are no pretrained image,
language, autoencoder, perceptual-loss, or tokenizer weights in this module.
Training: x(t)=(1-t)*noise+t*image; target velocity=image-noise.
Sampling: integrate predicted velocity forward from t=0 to t=1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
BYTE_OFFSET = 3
VOCAB_SIZE = 259
FORMAT_VERSION = "cemivo-pixel-flow-v1"


@dataclass
class ModelConfig:
    in_channels: int = 1
    base_channels: int = 128
    channel_multipliers: tuple[int, ...] = (1, 2, 3, 4)
    num_res_blocks: int = 2
    time_dim: int = 512
    text_dim: int = 384
    text_layers: int = 4
    text_heads: int = 6
    text_max_length: int = 192
    attention_heads: int = 8
    dropout: float = 0.0
    gradient_checkpointing: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ModelConfig":
        value = dict(value)
        if "channel_multipliers" in value:
            value["channel_multipliers"] = tuple(value["channel_multipliers"])
        return cls(**value)


def tokenize(prompts: str | Sequence[str], max_length: int = 192, device=None) -> torch.Tensor:
    """Fixed UTF-8 byte tokenizer: PAD0 BOS1 EOS2 bytes+3; no fitted vocabulary."""
    if isinstance(prompts, str):
        prompts = [prompts]
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    tokens = torch.zeros((len(prompts), max_length), dtype=torch.long, device=device)
    for row, text in enumerate(prompts):
        raw = str(text).encode("utf-8")[:max_length-2]
        values = [BOS_TOKEN] + [byte+BYTE_OFFSET for byte in raw] + [EOS_TOKEN]
        tokens[row,:len(values)] = torch.tensor(values,dtype=torch.long,device=device)
    return tokens


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim//2
    frequencies = torch.exp(-math.log(10000)*torch.arange(half,device=t.device,dtype=torch.float32)/max(half-1,1))
    phases = t.float()[:,None]*1000.0*frequencies[None,:]
    result = torch.cat((torch.cos(phases),torch.sin(phases)),dim=-1)
    if dim%2:
        result=F.pad(result,(0,1))
    return result


def normalization(channels: int) -> nn.Module:
    return nn.GroupNorm(math.gcd(32,channels),channels,eps=1e-6)


class ByteTextEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding=nn.Embedding(VOCAB_SIZE,config.text_dim,padding_idx=PAD_TOKEN)
        self.position=nn.Parameter(torch.empty(1,config.text_max_length,config.text_dim))
        layer=nn.TransformerEncoderLayer(config.text_dim,config.text_heads,config.text_dim*4,dropout=0.0,activation="gelu",batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers=config.text_layers,norm=nn.LayerNorm(config.text_dim),enable_nested_tensor=False)
        nn.init.normal_(self.position,std=0.02)

    def forward(self, tokens: torch.Tensor):
        valid=tokens.ne(PAD_TOKEN)
        hidden=self.embedding(tokens)+self.position[:,:tokens.shape[1]]
        hidden=self.transformer(hidden,src_key_padding_mask=~valid)
        pooled=(hidden*valid[:,:,None]).sum(dim=1)/valid.sum(dim=1,keepdim=True).clamp_min(1)
        return hidden,pooled,valid


class ResidualBlock(nn.Module):
    def __init__(self,in_channels: int,out_channels: int,time_dim: int,dropout: float):
        super().__init__()
        self.norm1=normalization(in_channels)
        self.conv1=nn.Conv2d(in_channels,out_channels,3,padding=1)
        self.condition=nn.Sequential(nn.SiLU(),nn.Linear(time_dim,2*out_channels))
        self.norm2=normalization(out_channels)
        self.dropout=nn.Dropout(dropout)
        self.conv2=nn.Conv2d(out_channels,out_channels,3,padding=1)
        self.skip=nn.Conv2d(in_channels,out_channels,1) if in_channels!=out_channels else nn.Identity()

    def forward(self,x,condition):
        h=self.conv1(F.silu(self.norm1(x)))
        scale,shift=self.condition(condition).chunk(2,dim=1)
        h=self.norm2(h)*(1+scale[:,:,None,None])+shift[:,:,None,None]
        h=self.conv2(self.dropout(F.silu(h)))
        return (self.skip(x)+h)/math.sqrt(2)


class SpatialTextAttention(nn.Module):
    """Cross-attention at low spatial resolutions; SDPA needs no add-on package."""
    def __init__(self,channels: int,text_dim: int,heads: int,self_attention: bool=False):
        super().__init__()
        self.heads=heads
        self.head_dim=channels//heads
        if channels%heads:
            raise ValueError("attention channels must divide heads")
        self.norm=normalization(channels)
        self.query=nn.Linear(channels,channels)
        self.key=nn.Linear(text_dim,channels)
        self.value=nn.Linear(text_dim,channels)
        self.output=nn.Linear(channels,channels)
        self.ff_norm=nn.LayerNorm(channels)
        self.ff=nn.Sequential(nn.Linear(channels,channels*4),nn.GELU(),nn.Linear(channels*4,channels))
        self.has_self_attention=self_attention
        if self_attention:
            self.self_norm=nn.LayerNorm(channels)
            self.self_qkv=nn.Linear(channels,channels*3)
            self.self_output=nn.Linear(channels,channels)

    def _heads(self,x):
        b,n,_=x.shape
        return x.reshape(b,n,self.heads,self.head_dim).transpose(1,2)

    def forward(self,x,text,valid):
        b,c,h,w=x.shape
        hidden=self.norm(x).flatten(2).transpose(1,2)
        if self.has_self_attention:
            q,k,v=self.self_qkv(self.self_norm(hidden)).chunk(3,dim=-1)
            attended=F.scaled_dot_product_attention(self._heads(q),self._heads(k),self._heads(v),dropout_p=0.0)
            hidden=hidden+self.self_output(attended.transpose(1,2).reshape(b,h*w,c))
        attended=F.scaled_dot_product_attention(self._heads(self.query(hidden)),self._heads(self.key(text)),self._heads(self.value(text)),attn_mask=valid[:,None,None,:],dropout_p=0.0)
        hidden=hidden+self.output(attended.transpose(1,2).reshape(b,h*w,c))
        hidden=hidden+self.ff(self.ff_norm(hidden))
        return (x+hidden.transpose(1,2).reshape(b,c,h,w))/math.sqrt(2)


class ConditionedBlock(nn.Module):
    def __init__(self,in_channels,out_channels,config,attention=False,self_attention=False):
        super().__init__()
        self.residual=ResidualBlock(in_channels,out_channels,config.time_dim,config.dropout)
        self.attention=SpatialTextAttention(out_channels,config.text_dim,config.attention_heads,self_attention) if attention else None

    def forward(self,x,condition,text,valid):
        x=self.residual(x,condition)
        if self.attention is not None:
            x=self.attention(x,text,valid)
        return x


class PixelFlowUNet(nn.Module):
    def __init__(self,config: ModelConfig | None=None):
        super().__init__()
        self.config=config or ModelConfig()
        cfg=self.config
        self.text_encoder=ByteTextEncoder(cfg)
        self.time_embedding=nn.Sequential(nn.Linear(cfg.base_channels,cfg.time_dim),nn.SiLU(),nn.Linear(cfg.time_dim,cfg.time_dim))
        self.text_condition=nn.Linear(cfg.text_dim,cfg.time_dim)
        self.input=nn.Conv2d(cfg.in_channels,cfg.base_channels,3,padding=1)
        channels=[cfg.base_channels*m for m in cfg.channel_multipliers]
        self.down_blocks=nn.ModuleList()
        self.downsamples=nn.ModuleList()
        current=cfg.base_channels
        skip_channels=[current]
        for level,out_channels in enumerate(channels):
            blocks=nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                blocks.append(ConditionedBlock(current,out_channels,cfg,attention=level==len(channels)-1))
                current=out_channels
                skip_channels.append(current)
            self.down_blocks.append(blocks)
            if level<len(channels)-1:
                self.downsamples.append(nn.Conv2d(current,current,3,stride=2,padding=1))
                skip_channels.append(current)
        self.middle=ConditionedBlock(current,current,cfg,attention=True,self_attention=True)
        self.middle2=ConditionedBlock(current,current,cfg)
        self.up_blocks=nn.ModuleList()
        self.upsamples=nn.ModuleList()
        for level in reversed(range(len(channels))):
            out_channels=channels[level]
            blocks=nn.ModuleList()
            for _ in range(cfg.num_res_blocks+1):
                skip=skip_channels.pop()
                blocks.append(ConditionedBlock(current+skip,out_channels,cfg,attention=level==len(channels)-1))
                current=out_channels
            self.up_blocks.append(blocks)
            if level>0:
                self.upsamples.append(nn.Conv2d(current,current,3,padding=1))
        assert not skip_channels
        self.output_norm=normalization(current)
        self.output=nn.Conv2d(current,cfg.in_channels,3,padding=1)
        self.apply(self._initialize)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module,(nn.Linear,nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight,std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module,nn.MultiheadAttention):
            # TransformerEncoder clones a layer; give every clone independent
            # initial attention projections instead of retaining copied values.
            if module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_text(self,tokens):
        return self.text_encoder(tokens)

    def _block(self,block,x,condition,text,valid):
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block,x,condition,text,valid,use_reentrant=False)
        return block(x,condition,text,valid)

    def forward(self,x,t,tokens=None,text_context=None):
        if x.ndim!=4 or x.shape[1]!=self.config.in_channels:
            raise ValueError("Expected Bx1xHxW pixel tensor")
        divisor=2**(len(self.config.channel_multipliers)-1)
        if x.shape[-1]%divisor or x.shape[-2]%divisor:
            raise ValueError(f"Image dimensions must be divisible by {divisor}")
        if text_context is None:
            if tokens is None:
                raise ValueError("tokens or text_context are required")
            text_context=self.encode_text(tokens)
        text,pooled,valid=text_context
        condition=self.time_embedding(timestep_embedding(t,self.config.base_channels).to(pooled.dtype))+self.text_condition(pooled)
        h=self.input(x)
        skips=[h]
        for level,blocks in enumerate(self.down_blocks):
            for block in blocks:
                h=self._block(block,h,condition,text,valid)
                skips.append(h)
            if level<len(self.downsamples):
                h=self.downsamples[level](h)
                skips.append(h)
        h=self._block(self.middle,h,condition,text,valid)
        h=self._block(self.middle2,h,condition,text,valid)
        for level,blocks in enumerate(self.up_blocks):
            for block in blocks:
                h=torch.cat((h,skips.pop()),dim=1)
                h=self._block(block,h,condition,text,valid)
            if level<len(self.upsamples):
                h=F.interpolate(h,scale_factor=2,mode="nearest")
                h=self.upsamples[level](h)
        return self.output(F.silu(self.output_norm(h)))


def flow_training_pair(images: torch.Tensor, noise=None, t=None):
    """Create an exact straight noise->data pair with an explicit time convention."""
    if noise is None:
        noise=torch.randn_like(images)
    if t is None:
        t=torch.rand(images.shape[0],device=images.device)
    expanded=t[:,None,None,None]
    state=(1-expanded)*noise+expanded*images
    velocity=images-noise
    return state,velocity,t


@torch.inference_mode()
def sample_flow(model: PixelFlowUNet,prompts: str | Sequence[str],seed=0,steps=16,size=256,device=None,guidance=1.0):
    """Euler flow sampling with optional classifier-free guidance.

    guidance=1 preserves the original single-condition computation exactly.
    Other scales batch conditional and empty-caption predictions in one forward
    pass, using the same empty tokens as training caption dropout.
    """
    if steps<1 or steps>128:
        raise ValueError("steps must be in 1..128")
    if size<32 or size>1024 or size%8:
        raise ValueError("size must be a multiple of 8 in 32..1024")
    guidance=float(guidance)
    if not math.isfinite(guidance) or not 0<=guidance<=8:
        raise ValueError("guidance must be finite and in 0..8")
    if isinstance(prompts,str):
        prompts=[prompts]
    parameter=next(model.parameters())
    device=torch.device(device) if device is not None else parameter.device
    dtype=parameter.dtype
    generator=torch.Generator(device=device).manual_seed(int(seed)%(2**63-1))
    x=torch.randn((len(prompts),1,size,size),device=device,dtype=dtype,generator=generator)
    tokens=tokenize(prompts,model.config.text_max_length,device)
    if guidance!=1.0:
        empty_tokens=tokenize([""]*len(prompts),model.config.text_max_length,device)
        tokens=torch.cat((tokens,empty_tokens),dim=0)
    context=model.encode_text(tokens)
    for index in range(steps):
        if guidance==1.0:
            # Keep this branch byte-for-byte equivalent in operation order to
            # unguided historical sampling and avoid doubling its batch.
            t=torch.full((len(prompts),),index/steps,device=device,dtype=torch.float32)
            x=x+model(x,t,text_context=context)/steps
        else:
            t=torch.full((len(prompts)*2,),index/steps,device=device,dtype=torch.float32)
            conditional,unconditional=model(torch.cat((x,x),dim=0),t,text_context=context).chunk(2,dim=0)
            velocity=unconditional+guidance*(conditional-unconditional)
            x=x+velocity/steps
    return x.float().clamp(-1,1)


@torch.inference_mode()
def sample_flow_edit(model, prompt, source, bounds, seed=0, steps=8, device=None):
    """Experimental masked flow; known pixels follow the original noise path.

    This reuses trained velocity weights, not a separately trained edit model.
    Final uint8 compositing must still enforce the exact preservation boundary.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 128:
        raise ValueError("steps must be an integer in 1..128")
    if source.ndim != 4 or source.shape[0:2] != (1, 1):
        raise ValueError("source must be 1x1xHxW")
    height, width = source.shape[-2:]
    if (height != width or width < 32 or width > 1024 or width % 8 or
            len(bounds) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in bounds)):
        raise ValueError("invalid source dimensions or bounds")
    x0, y0, x1, y1 = bounds
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("edit bounds outside source")
    parameter = next(model.parameters())
    device = torch.device(device) if device is not None else parameter.device
    source = source.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(source).all()) or bool((source.abs() > 1).any()):
        raise ValueError("source pixels must be finite in -1..1")
    rng = torch.Generator(device=device).manual_seed(int(seed) % (2**63 - 1))
    noise = torch.randn(source.shape, device=device, dtype=parameter.dtype, generator=rng).float()
    mask = torch.zeros_like(source, dtype=torch.bool)
    mask[:, :, y0:y1, x0:x1] = True
    context = model.encode_text(tokenize(prompt, model.config.text_max_length, device))
    x = noise.clone()
    for index in range(steps):
        now = index / steps
        known = (1 - now) * noise + now * source
        x = torch.where(mask, x, known)
        t = torch.full((1,), now, device=device, dtype=torch.float32)
        candidate = x + model(x.to(dtype=parameter.dtype), t, text_context=context).float() / steps
        next_time = (index + 1) / steps
        known_next = (1 - next_time) * noise + next_time * source
        x = torch.where(mask, candidate, known_next)
    return torch.where(mask, x.float().clamp(-1, 1), source.float())
