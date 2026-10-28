"""Load only locally trained pixel weights and generate an image with the UNet.

There is deliberately no renderer, template, classifier, image retrieval,
network request, or text-drawing fallback in this inference module.
"""
from __future__ import annotations

import argparse
from numbers import Integral
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .model import FORMAT_VERSION, ModelConfig, PixelFlowUNet, sample_flow, sample_flow_edit
from .locking import model_access
from .sampling_quality import sample_flow_quality


class NeuralGenerator:
    def __init__(self,checkpoint=None,device="cuda"):
        checkpoint=Path(checkpoint) if checkpoint else Path(__file__).resolve().parent/"checkpoints"/"inference.pt"
        self.checkpoint_path=checkpoint.resolve()
        with model_access(self.checkpoint_path.parent):
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"No trained neural checkpoint: {self.checkpoint_path}")
            self.device=torch.device(device)
            if self.device.type=="cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; explicitly use device='cpu' for CPU inference")
            saved=torch.load(self.checkpoint_path,map_location="cpu",weights_only=False)
            if saved.get("format_version")!=FORMAT_VERSION:
                raise ValueError("Unsupported neural checkpoint format")
            self.config=ModelConfig.from_dict(saved["model_config"])
            self.config.gradient_checkpointing=False
            # Inference already has every trained tensor. Construct only the
            # architecture on meta, then attach those tensors directly; avoid
            # allocating and randomly filling a redundant CPU parameter set.
            # No mmap is retained, so Windows checkpoints remain replaceable.
            with torch.device("meta"):
                self.model=PixelFlowUNet(self.config)
            state=saved.get("ema",saved["model"])
            self.model.load_state_dict(state,strict=True,assign=True)
            self.checkpoint_step=int(saved.get("step",0))
            self.native_size=int(saved.get("native_size",256))
            self.size=self.native_size
            self.parameter_count=self.model.parameter_count
            self.provenance=saved.get("provenance",{})
            self.dtype=torch.bfloat16 if self.device.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float32
            self.model.to(device=self.device,dtype=self.dtype)
            self.model.eval()
            self.last_generation_seconds=None
            del saved,state

    def _prompt(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        try:
            encoded = prompt.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("prompt must be valid UTF-8 text") from None
        budget = self.model.config.text_max_length - 2
        if len(encoded) > budget:
            raise ValueError(f"prompt exceeds the model limit of {budget} UTF-8 bytes; it will not be truncated")
        return prompt

    @staticmethod
    def _integers(seed, steps, size):
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        if isinstance(steps, bool) or not isinstance(steps, Integral) or not 1 <= steps <= 128:
            raise ValueError("steps must be an integer in 1..128")
        if isinstance(size, bool) or not isinstance(size, Integral) or not 32 <= size <= 1024 or size % 8:
            raise ValueError("size must be an integer multiple of 8 in 32..1024")
        return int(seed), int(steps), int(size)

    @torch.inference_mode()
    def generate_tensor(self,prompt,seed=0,steps=16,size=None,guidance=1.0):
        prompt = self._prompt(prompt)
        seed, steps, size = self._integers(seed, steps, self.native_size if size is None else size)
        with model_access(self.checkpoint_path.parent):
            start=time.perf_counter()
            pixels=sample_flow_quality(self.model,prompt,seed=seed,steps=steps,size=size,device=self.device,guidance=guidance,solver="euler")
            if self.device.type=="cuda":
                torch.cuda.synchronize(self.device)
            self.last_generation_seconds=time.perf_counter()-start
            return pixels

    def generate(self,prompt,seed=0,steps=16,size=None,guidance=1.0):
        with model_access(self.checkpoint_path.parent):
            pixels=self.generate_tensor(prompt,seed,steps,size,guidance)
            array=((pixels[0,0].cpu().numpy()+1)*127.5).round().clip(0,255).astype(np.uint8)
            return Image.fromarray(array)

    @torch.inference_mode()
    def edit(self, source_image, prompt, bounds, seed=0, steps=8):
        """Generate pixels in one mask; callers enforce the final uint8 boundary."""
        prompt = self._prompt(prompt)
        seed, steps, _ = self._integers(seed, steps, self.native_size)
        if source_image.mode != "L" or source_image.size != (self.native_size, self.native_size):
            raise ValueError("Edit source must be a native-size grayscale image")
        with model_access(self.checkpoint_path.parent):
            start = time.perf_counter()
            array = np.asarray(source_image, dtype=np.float32).copy() / 127.5 - 1
            source = torch.from_numpy(array)[None, None]
            pixels = sample_flow_edit(self.model, prompt, source, bounds, seed, steps, self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.last_generation_seconds = time.perf_counter() - start
            result = ((pixels[0,0].cpu().numpy()+1)*127.5).round().clip(0,255).astype(np.uint8)
            return Image.fromarray(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output",default="neural/generated.png")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--steps",type=int,default=16)
    parser.add_argument("--size",type=int)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--guidance",type=float,default=1.0,help="Classifier-free guidance scale in 0..8; 1 preserves original sampling")
    args=parser.parse_args()
    checkpoint=Path(args.checkpoint) if args.checkpoint else Path(__file__).resolve().parent/"checkpoints"/"inference.pt"
    with model_access(checkpoint.resolve().parent):
        generator=NeuralGenerator(checkpoint,args.device)
        image=generator.generate(args.prompt,args.seed,args.steps,args.size,args.guidance)
        output=Path(args.output)
        output.parent.mkdir(parents=True,exist_ok=True)
        image.save(output)
        print(f"Saved {output.resolve()} | native trained size {generator.native_size} | checkpoint step {generator.checkpoint_step} | neural generation {generator.last_generation_seconds:.3f}s")


if __name__=="__main__":
    main()
