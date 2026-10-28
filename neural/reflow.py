"""Optional self-trained teacher -> reflow experiment; not a quality claim.

This module never runs automatically. `pairs` stores the EXACT same Gaussian
start x0 and its unclipped teacher ODE endpoint x1. `train` regresses x1-x0 on
(1-t)*x0+t*x1. No new noise is substituted for a saved pair at training time.
See REFLOW.md for provenance checks, locks, limitations, and commands.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from .locking import model_access
from .model import FORMAT_VERSION, ModelConfig, PixelFlowUNet, flow_training_pair, tokenize
from .train import atomic_json, atomic_torch_save, replace_file, restore_rng, utc_now

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "neural" / "checkpoints"
PAIRS_VERSION = "cemivo-self-trained-reflow-pairs-v1"
REFLOW_VERSION = "cemivo-self-trained-reflow-training-v1"
TIME_CONVENTION = "x_t=(1-t)*x0+t*x1; target=x1-x0; integrate t=0->1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def overlap(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return a == b or a in b.parents or b in a.parents


def separate_output(output, *protected):
    output = Path(output).resolve()
    for path in protected:
        if overlap(output, path):
            raise ValueError(f"Output must be separate from source/base paths: {output} / {path}")
    return output


@contextmanager
def access_all(*directories):
    # Also hold BASE when teacher/student is elsewhere: the base training lock
    # is the current project's GPU-use guard. All acquisitions are non-blocking.
    with ExitStack() as stack:
        paths = {os.path.normcase(str(Path(path).resolve())) for path in directories}
        for path in sorted(paths):
            stack.enter_context(model_access(path))
        yield


class StopFlag:
    def __init__(self, directory):
        self.directory, self.requested, self.previous = Path(directory), False, {}

    def stopped(self):
        return self.requested or any((self.directory / name).exists() for name in ("STOP", "stop"))

    def __enter__(self):
        def request(signum, frame):
            self.requested = True
        for name in ("SIGINT", "SIGTERM"):
            if hasattr(signal, name):
                signum = getattr(signal, name)
                self.previous[signum] = signal.signal(signum, request)
        return self

    def __exit__(self, *exc):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def select_device(args):
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only explicit cpu/cuda devices are supported")
    if device.type == "cpu" and not args.allow_cpu:
        raise ValueError("CPU production work requires --allow-cpu; use smoke-test for a tiny CPU check")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    amp = device.type == "cuda" and not args.fp32
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable; explicitly choose --fp32")
    torch.set_num_threads(args.cpu_threads)
    return device, amp


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def capture_state(stream, device):
    # train.capture_rng eagerly queries all CUDA generators when CUDA is
    # available, even for CPU work. Keep its schema, but do not initialize CUDA
    # in an explicitly CPU-only reflow/smoke process.
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "sampler": stream.rng.bit_generator.state,
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def read_teacher(path):
    """Load trusted local inference weights, checking declared origin and format."""
    path = Path(path).resolve()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format_version") != FORMAT_VERSION or saved.get("checkpoint_kind") != "inference-ema":
        raise ValueError("Teacher must be this project's inference-ema checkpoint, not latest.pt")
    provenance = saved.get("provenance", {})
    if provenance.get("random_initialization") is not True or provenance.get("pretrained_weights") is not False:
        raise ValueError("Teacher must declare random_initialization=True and pretrained_weights=False")
    if int(saved.get("step", 0)) < 1:
        raise ValueError("Teacher must have completed at least one optimizer step")
    config = ModelConfig.from_dict(saved["model_config"])
    size = int(saved["native_size"])
    if config.in_channels != 1 or size < 32 or size > 1024 or size % 8:
        raise ValueError("Expected a monochrome native resolution divisible by 8 in 32..1024")
    if size % (2 ** (len(config.channel_multipliers) - 1)):
        raise ValueError("Teacher native resolution is incompatible with its architecture")
    info = {"path": str(path), "sha256": sha256_file(path), "step": int(saved["step"]),
            "checkpoint_kind": saved["checkpoint_kind"], "provenance": provenance}
    return saved, config, size, info


def read_prompts(path, max_bytes):
    prompts = []
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            if Path(path).suffix.lower() == ".jsonl":
                row = json.loads(line)
                line = row.get("prompt") if isinstance(row, dict) else row
            if not isinstance(line, str) or not line.strip():
                raise ValueError(f"Expected a non-empty prompt at line {line_number}")
            text = line.strip()
            if len(text.encode("utf-8")) > max_bytes:
                raise ValueError(f"Prompt at line {line_number} exceeds {max_bytes} UTF-8 bytes; no truncation is allowed")
            prompts.append(text)
    if not prompts:
        raise ValueError("Prompt file is empty")
    return prompts


@torch.inference_mode()
def teacher_endpoint(model, x0, tokens, steps, amp=False):
    """Euler teacher rollout, preserving x0 and the RAW (not display-clamped) x1."""
    device = x0.device
    x = x0.clone().float()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        context = model.encode_text(tokens)
        for index in range(steps):
            t = torch.full((len(x),), index / steps, device=device, dtype=torch.float32)
            x = x + model(x, t, text_context=context).float() / steps
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError("Teacher produced a non-finite endpoint")
    return x


def atomic_npy(path, array):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    replace_file(temporary, path)


def write_pair_shard(directory, index, x0, x1, tokens, rows):
    prefix = f"pairs-{index:08}"
    arrays = {"x0": x0, "x1": x1, "tokens": tokens}
    shard = {"start": index, "count": len(rows), "sha256": {}}
    for key, array in arrays.items():
        name = f"{prefix}-{key}.npy"
        atomic_npy(directory / name, array)
        shard[key] = name
        shard["sha256"][name] = sha256_file(directory / name)
    name = f"{prefix}-captions.json"
    atomic_json(directory / name, rows)
    shard["captions"] = name
    shard["sha256"][name] = sha256_file(directory / name)
    return shard


def prepare_pairs(args, guard_directory=BASE):
    teacher = Path(args.teacher).resolve()
    output = separate_output(args.output, guard_directory, teacher.parent)
    with access_all(guard_directory, teacher.parent, output):
        device, amp = select_device(args)
        atomic_json(output / "status.json", {"state": "preparing", "phase": "teacher-pair-generation",
                    "pid": os.getpid(), "updated_at": utc_now(), "quality_verified": False})
        try:
            return _prepare_pairs(args, teacher, output, device, amp)
        except BaseException as error:
            record_failure(output, error, "teacher-pair-generation")
            raise
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()


def _prepare_pairs(args, teacher, output, device, amp):
    start, completed = time.perf_counter(), 0
    manifest_path, status_path = output / "manifest.json", output / "status.json"
    saved, config, size, teacher_info = read_teacher(teacher)
    prompts = read_prompts(args.prompts, config.text_max_length - 2)
    contract = {"teacher_sha256": teacher_info["sha256"], "teacher_steps": args.teacher_steps,
                "seed": args.seed, "target_count": args.count, "native_size": size,
                "batch_size": args.batch_size, "shard_size": args.shard_size,
                "prompts_sha256": canonical_hash(prompts), "prompt_order": "round-robin",
                "noise": "torch CPU Generator per index, seed=(seed+index)%(2**63-1), float32",
                "device_type": device.type, "bf16": amp, "time_convention": TIME_CONVENTION,
                "endpoint_clipping": False, "storage_dtype": "float32"}
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Pair manifest exists; choose --resume or a fresh --output")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != PAIRS_VERSION or manifest.get("generation_contract") != contract:
            raise ValueError("Pair generation settings/teacher/prompts changed; use a fresh output directory")
        # Validate every committed shard before appending. Uncommitted temporary
        # files are ignored and replaced at the next start index.
        if manifest.get("count") == 0 and not manifest.get("shards"):
            completed = 0
        else:
            dataset = PairDataset(output)
            completed = len(dataset)
            dataset.close()
    else:
        if args.resume:
            raise FileNotFoundError("--resume requires an existing pair manifest")
        manifest = {"format_version": PAIRS_VERSION, "generation_contract": contract,
                    "teacher": teacher_info, "model_config": saved["model_config"],
                    "prompts_file": str(Path(args.prompts).resolve()), "count": 0,
                    "shards": [], "created_at": utc_now(), "quality_verified": False}
        atomic_json(manifest_path, manifest)
    config.gradient_checkpointing = False
    model = PixelFlowUNet(config)
    model.load_state_dict(saved["model"], strict=True)
    del saved
    model.to(device=device, dtype=torch.bfloat16 if amp else torch.float32).eval()

    def status(state, **extra):
        atomic_json(status_path, {"state": state, "phase": "teacher-pair-generation",
                    "committed_pairs": completed, "target_pairs": args.count,
                    "teacher": teacher_info, "teacher_steps": args.teacher_steps,
                    "elapsed_seconds": round(time.perf_counter() - start, 3),
                    "quality_verified": False, "device": str(device), "pid": os.getpid(),
                    "updated_at": utc_now(), **extra})
    try:
        with StopFlag(output) as stop:
            status("running")
            while completed < args.count and not stop.stopped():
                shard_start = completed
                length = min(args.shard_size, args.count - completed)
                starts, ends, token_rows, metadata = [], [], [], []
                for offset in range(0, length, args.batch_size):
                    if stop.stopped():
                        break
                    batch_indices = list(range(shard_start + offset, shard_start + min(offset + args.batch_size, length)))
                    captions = [prompts[index % len(prompts)] for index in batch_indices]
                    seeds = [(args.seed + index) % (2**63 - 1) for index in batch_indices]
                    # An index's x0 is independent of shard/batch boundaries.
                    noise = torch.stack([torch.randn((1, size, size), generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32) for seed in seeds])
                    tokens = tokenize(captions, config.text_max_length)
                    endpoint = teacher_endpoint(model, noise.to(device), tokens.to(device), args.teacher_steps, amp)
                    starts.append(noise.numpy())
                    ends.append(endpoint.cpu().numpy())
                    token_rows.append(tokens.numpy())
                    metadata.extend({"index": index, "prompt": caption, "noise_seed": seed} for index, caption, seed in zip(batch_indices, captions, seeds))
                if metadata:
                    shard = write_pair_shard(output, shard_start, np.concatenate(starts),
                                             np.concatenate(ends), np.concatenate(token_rows), metadata)
                    manifest["shards"].append(shard)
                    completed += len(metadata)
                    manifest["count"], manifest["updated_at"] = completed, utc_now()
                    atomic_json(manifest_path, manifest)
                    status("running")
                    print(json.dumps({"committed_pairs": completed, "target_pairs": args.count}), flush=True)
            state = "complete" if completed == args.count else "paused"
            status(state)
            return {"state": state, "pairs": completed, "manifest": str(manifest_path)}
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise


class PairDataset:
    """Verified, memory-mapped x0/x1/token pairs. Never redraws x0."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != PAIRS_VERSION:
            raise ValueError("Unsupported reflow pair format")
        self.fingerprint = sha256_file(self.directory / "manifest.json")
        self.config = ModelConfig.from_dict(self.manifest["model_config"])
        self.size = int(self.manifest["generation_contract"]["native_size"])
        self.shards, self.ends, self.maps = self.manifest["shards"], [], OrderedDict()
        total = 0
        for shard in self.shards:
            if shard["start"] != total or shard["count"] < 1:
                raise ValueError("Pair shards are not contiguous")
            count = shard["count"]
            for key in ("x0", "x1", "tokens", "captions"):
                name = shard[key]
                if not isinstance(name, str) or any(part in name for part in ("/", "\\", "..")):
                    raise ValueError("Unsafe pair shard filename")
                path = self.directory / name
                if sha256_file(path) != shard["sha256"].get(name):
                    raise ValueError(f"Pair shard checksum mismatch: {name}")
                if key != "captions":
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    expected = (count, self.config.text_max_length) if key == "tokens" else (count, 1, self.size, self.size)
                    dtype = np.int64 if key == "tokens" else np.float32
                    if array.shape != expected or array.dtype != dtype:
                        raise ValueError(f"Invalid shape/dtype for {name}")
                    array._mmap.close()
            total += count
            self.ends.append(total)
        if total != self.manifest["count"] or not total:
            raise ValueError("Pair dataset is empty or manifest count is inconsistent")

    def __len__(self):
        return self.manifest["count"]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.ends, index)
        if shard_index not in self.maps:
            shard = self.shards[shard_index]
            self.maps[shard_index] = {key: np.load(self.directory / shard[key], mmap_mode="r", allow_pickle=False) for key in ("x0", "x1", "tokens")}
            if len(self.maps) > 8:
                _, previous = self.maps.popitem(last=False)
                for array in previous.values():
                    array._mmap.close()
        self.maps.move_to_end(shard_index)
        offset = index - self.shards[shard_index]["start"]
        return {key: np.array(array[offset], copy=True) for key, array in self.maps[shard_index].items()}

    def close(self):
        for maps in self.maps.values():
            for array in maps.values():
                array._mmap.close()
        self.maps.clear()


class PairStream:
    def __init__(self, dataset, seed):
        self.dataset, self.rng = dataset, np.random.default_rng(seed)

    def batch(self, count, device):
        rows = [self.dataset[int(index)] for index in self.rng.integers(0, len(self.dataset), size=count)]
        return tuple(torch.from_numpy(np.stack([row[key] for row in rows])).to(device) for key in ("x0", "x1", "tokens"))


def student_contract(args, dataset, teacher_info, config, device, amp):
    # Target total steps may be extended. The LR schedule depends only on the
    # completed step and fixed warmup, never on the requested total step count.
    return {"pairs_manifest_sha256": dataset.fingerprint, "teacher_sha256": teacher_info["sha256"],
            "model_config": config.to_dict(), "student_initialization": args.student_init,
            "seed": args.seed, "batch_size": args.batch_size, "accum": args.accum,
            "lr": args.lr, "lr_warmup_steps": args.lr_warmup_steps, "weight_decay": args.weight_decay,
            "ema_decay": args.ema_decay, "max_grad_norm": args.max_grad_norm,
            "device_type": device.type, "bf16": amp, "time_distribution": "uniform[0,1)",
            "time_convention": TIME_CONVENTION, "caption_dropout": 0.0}


def save_student(directory, model, ema, optimizer, stream, args, contract,
                 teacher_info, dataset, step, samples_seen, elapsed, device):
    provenance = {"random_initialization": True, "pretrained_weights": False,
                  "generator": "locally self-trained teacher -> conditional pixel reflow UNet",
                  "weight_origin": "all learned weights originate in local random initialization",
                  "student_initialization": args.student_init,
                  "student_fresh_random_initialization": args.student_init == "random",
                  "teacher": teacher_info, "dataset": str(dataset.directory),
                  "pairs_manifest_sha256": dataset.fingerprint, "training_seed": args.seed,
                  "time_convention": TIME_CONVENTION, "quality_verified": False}
    common = {"format_version": FORMAT_VERSION, "reflow_version": REFLOW_VERSION,
              "model_config": model.config.to_dict(), "step": step, "samples_seen": samples_seen,
              "elapsed_seconds": elapsed, "native_size": dataset.size, "provenance": provenance}
    atomic_torch_save(directory / "latest.pt", {**common, "checkpoint_kind": "resumable-reflow-training",
                      "model": model.state_dict(), "ema": ema, "optimizer": optimizer.state_dict(),
                      "rng": capture_state(stream, device), "resume_contract": contract,
                      "training_args": vars(args)})
    inference = {name: value.detach().to(device="cpu", dtype=torch.bfloat16) for name, value in ema.items()}
    atomic_torch_save(directory / "inference.pt", {**common, "checkpoint_kind": "inference-ema", "model": inference})
    atomic_json(directory / "model_card.json", {"name": "cemivoAI PixelFlow Reflow (experimental)",
                "format_version": FORMAT_VERSION, "reflow_version": REFLOW_VERSION,
                "parameters": model.parameter_count, "native_width": dataset.size, "native_height": dataset.size,
                "checkpoint_step": step, "teacher_checkpoint_step": teacher_info["step"],
                "random_initialization": True, "pretrained_weights": False,
                "student_initialization": args.student_init,
                "student_fresh_random_initialization": args.student_init == "random",
                "training_data": str(dataset.directory), "pairs_manifest_sha256": dataset.fingerprint,
                "teacher_sha256": teacher_info["sha256"], "architecture": model.config.to_dict(),
                "quality_verified": False, "one_to_four_step_quality_verified": False,
                "limitations": ["No project evidence yet that reflow improves quality or latency at equal quality.",
                                "A weak teacher can teach noise, blank images, or malformed UI.",
                                "1-4 step sampling is an experimental evaluation target, not a guaranteed capability.",
                                "All weights originate locally; teacher initialization is not fresh random student initialization."],
                "updated_at": utc_now()})


def train_reflow(args, guard_directory=BASE):
    pairs = Path(args.pairs).resolve()
    preliminary = json.loads((pairs / "manifest.json").read_text(encoding="utf-8"))
    teacher = Path(args.teacher or preliminary["teacher"]["path"]).resolve()
    output = separate_output(args.checkpoints, guard_directory, teacher.parent, pairs)
    with access_all(guard_directory, teacher.parent, pairs, output):
        device, amp = select_device(args)
        atomic_json(output / "status.json", {"state": "preparing", "phase": "reflow-training",
                    "pid": os.getpid(), "updated_at": utc_now(), "quality_verified": False})
        try:
            return _train_reflow(args, pairs, teacher, output, device, amp)
        except BaseException as error:
            record_failure(output, error, "reflow-training")
            raise
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()


def record_failure(directory, error, phase):
    path = Path(directory) / "status.json"
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    if previous.get("state") != "failed":
        atomic_json(path, {**previous, "state": "failed", "phase": phase,
                    "error": f"{type(error).__name__}: {error}", "updated_at": utc_now()})


def _train_reflow(args, pairs, teacher, output, device, amp):
    seed_all(args.seed, device)
    dataset = PairDataset(pairs)
    start, previous_elapsed, step, samples_seen, loss_value = time.perf_counter(), 0.0, 0, 0, None
    status_path = output / "status.json"
    try:
        saved, config, size, teacher_info = read_teacher(teacher)
        if teacher_info["sha256"] != dataset.manifest["teacher"]["sha256"]:
            raise ValueError("Teacher changed after pair generation; use the exact frozen teacher snapshot")
        if canonical_hash(saved["model_config"]) != canonical_hash(dataset.manifest["model_config"]) or size != dataset.size:
            raise ValueError("Teacher architecture/resolution differs from the pair manifest")
        config.gradient_checkpointing = not args.no_checkpointing
        contract = student_contract(args, dataset, teacher_info, config, device, amp)
        resume = None
        if args.resume:
            path = output / "latest.pt" if args.resume == "auto" else Path(args.resume).resolve()
            if path.parent != output:
                raise ValueError("Resume checkpoint must belong to the selected reflow output directory")
            resume = torch.load(path, map_location="cpu", weights_only=False)
            if resume.get("reflow_version") != REFLOW_VERSION or resume.get("checkpoint_kind") != "resumable-reflow-training":
                raise ValueError("--resume requires this tool's resumable latest.pt")
            if resume.get("resume_contract") != contract:
                raise ValueError("Resume configuration, dataset, teacher, or device changed; use identical settings (total --steps may increase)")
        elif (output / "latest.pt").exists() or (output / "inference.pt").exists():
            raise FileExistsError("Reflow weights already exist; use --resume or a new directory")
        model = PixelFlowUNet(config)
        if resume is None and args.student_init == "teacher":
            model.load_state_dict(saved["model"], strict=True)
        del saved
        model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay, fused=device.type == "cuda")
        stream = PairStream(dataset, args.seed)
        if resume is not None:
            model.load_state_dict(resume["model"], strict=True)
            optimizer.load_state_dict(resume["optimizer"])
            ema = {name: value.to(device) for name, value in resume["ema"].items()}
            step, samples_seen = int(resume["step"]), int(resume["samples_seen"])
            previous_elapsed = float(resume.get("elapsed_seconds", 0))
            restore_rng(resume["rng"], stream)
            del resume
        else:
            ema = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        if step > args.steps:
            raise ValueError("Target --steps is less than the loaded checkpoint step")

        def elapsed():
            return previous_elapsed + time.perf_counter() - start

        def status(state, **extra):
            atomic_json(status_path, {"state": state, "phase": "reflow-training", "step": step,
                        "total_steps": args.steps, "loss": loss_value, "samples_seen": samples_seen,
                        "elapsed_seconds": round(elapsed(), 3), "parameters": model.parameter_count,
                        "teacher_checkpoint_step": teacher_info["step"], "teacher_sha256": teacher_info["sha256"],
                        "student_initialization": args.student_init, "quality_verified": False,
                        "device": str(device), "pid": os.getpid(), "updated_at": utc_now(), **extra})

        def save():
            save_student(output, model, ema, optimizer, stream, args, contract,
                         teacher_info, dataset, step, samples_seen, elapsed(), device)

        rollback_rng, rollback_samples, phase = None, samples_seen, "idle"
        try:
            with StopFlag(output) as stop:
                status("running")
                while step < args.steps and not stop.stopped():
                    rollback_rng, rollback_samples = capture_state(stream, device), samples_seen
                    phase = "forward-backward"
                    optimizer.zero_grad(set_to_none=True)
                    lr = args.lr * min(1.0, (step + 1) / max(args.lr_warmup_steps, 1))
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    accumulated = 0.0
                    for _ in range(args.accum):
                        x0, x1, tokens = stream.batch(args.batch_size, device)
                        # Pass SAVED x0 explicitly: default flow_training_pair
                        # would draw independent noise and destroy reflow pairs.
                        state, target, t = flow_training_pair(x1, noise=x0)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                            loss = F.mse_loss(model(state, t, tokens).float(), target.float())
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(f"Non-finite loss before step {step + 1}")
                        (loss / args.accum).backward()
                        accumulated += float(loss.detach()) / args.accum
                        samples_seen += len(x0)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if not bool(torch.isfinite(norm)):
                        raise FloatingPointError("Non-finite gradient norm")
                    phase = "optimizer-started"
                    optimizer.step()
                    with torch.no_grad():
                        for name, parameter in model.named_parameters():
                            ema[name].lerp_(parameter.detach(), 1 - args.ema_decay)
                    step += 1
                    loss_value, phase = accumulated, "idle"
                    entry = {"step": step, "loss": loss_value, "lr": lr, "samples_seen": samples_seen,
                             "elapsed_seconds": round(elapsed(), 3), "updated_at": utc_now()}
                    with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    if step % args.log_every == 0 or step == 1:
                        status("running")
                        print(json.dumps(entry), flush=True)
                    if step % args.save_every == 0:
                        save()
                save()
                state = "complete" if step >= args.steps else "paused"
                status(state, checkpoint_step=step)
                return {"state": state, "step": step, "checkpoints": str(output)}
        except BaseException as error:
            recovery = "previous atomic checkpoint retained"
            # Forward/backward errors have not mutated the weights. Restore the
            # microbatch/time RNG and counters to the last completed step.
            # Never serialize a potentially half-updated Adam/EMA state.
            if phase != "optimizer-started":
                try:
                    optimizer.zero_grad(set_to_none=True)
                    if phase == "forward-backward" and rollback_rng is not None:
                        restore_rng(rollback_rng, stream)
                        samples_seen = rollback_samples
                    save()
                    recovery = "saved last fully completed optimizer step"
                except BaseException as save_error:
                    recovery = f"previous atomic checkpoint retained; recovery save failed: {save_error}"
            status("failed", error=f"{type(error).__name__}: {error}", recovery=recovery)
            raise
    finally:
        dataset.close()


def smoke_test(args):
    """Tiny local CPU only: no base locks, no existing weights, no CUDA calls."""
    torch.set_num_threads(2)
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    seed_all(71, torch.device("cpu"))
    with tempfile.TemporaryDirectory(prefix="cemivo-reflow-smoke-", dir=work) as temporary:
        temp = Path(temporary)
        teacher_dir = temp / "tiny-teacher"
        teacher_dir.mkdir()
        config = ModelConfig(base_channels=16, channel_multipliers=(1, 2), num_res_blocks=1,
                             time_dim=64, text_dim=32, text_layers=1, text_heads=4,
                             text_max_length=32, attention_heads=4, gradient_checkpointing=False)
        teacher = PixelFlowUNet(config).eval()
        tokens = tokenize(["chat", "line"], 32)
        x0 = torch.randn(2, 1, 32, 32)
        target_images = torch.ones_like(x0)
        target_images[:, :, 6:9, 4:28] = -1
        state, velocity, t = flow_training_pair(target_images, x0, torch.full((2,), .5))
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=.001)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(teacher(state, t, tokens), velocity)
            loss.backward()
            optimizer.step()
        teacher.eval()
        endpoint = teacher_endpoint(teacher, x0, tokens, 3)
        pair_state, pair_target, pair_t = flow_training_pair(endpoint, noise=x0, t=torch.full((2,), .5))
        assert torch.equal(flow_training_pair(endpoint, x0, torch.zeros(2))[0], x0)
        assert torch.equal(flow_training_pair(endpoint, x0, torch.ones(2))[0], endpoint)
        student = PixelFlowUNet(config)
        optimizer = torch.optim.AdamW(student.parameters(), lr=.001)
        losses = []
        for _ in range(14):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(student(pair_state, pair_t, tokens), pair_target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        assert losses[-1] < losses[0], f"Fixed-pair CPU loss did not descend: {losses}"
        teacher_path = teacher_dir / "inference.pt"
        atomic_torch_save(teacher_path, {"format_version": FORMAT_VERSION, "checkpoint_kind": "inference-ema",
                          "model_config": config.to_dict(), "model": teacher.state_dict(), "step": 3,
                          "native_size": 32, "provenance": {"random_initialization": True, "pretrained_weights": False}})
        prompts_path = temp / "prompts.txt"
        prompts_path.write_text("chat\nline\n", encoding="utf-8")
        parser = build_parser()
        pair_args = parser.parse_args(["pairs", "--teacher", str(teacher_path), "--prompts", str(prompts_path),
                    "--output", str(temp / "pairs"), "--count", "4", "--batch-size", "2", "--shard-size", "2",
                    "--teacher-steps", "3", "--device", "cpu", "--allow-cpu"])
        result = prepare_pairs(pair_args, guard_directory=teacher_dir)
        assert result["pairs"] == 4
        pair_args.resume = True
        assert prepare_pairs(pair_args, guard_directory=teacher_dir)["pairs"] == 4
        data = PairDataset(temp / "pairs")
        row = data[0]
        expected_noise = torch.randn((1, 32, 32), generator=torch.Generator(device="cpu").manual_seed(pair_args.seed))
        assert np.array_equal(row["x0"], expected_noise.numpy()), "Stored x0 differs from the teacher's exact starting noise"
        data.close()
        common = ["train", "--pairs", str(temp / "pairs"), "--teacher", str(teacher_path),
                  "--batch-size", "2", "--accum", "1", "--lr", ".0001", "--lr-warmup-steps", "1",
                  "--save-every", "2", "--log-every", "2", "--device", "cpu", "--allow-cpu", "--no-checkpointing"]
        full = parser.parse_args(common + ["--checkpoints", str(temp / "full"), "--steps", "4"])
        train_reflow(full, guard_directory=teacher_dir)
        split = parser.parse_args(common + ["--checkpoints", str(temp / "split"), "--steps", "2"])
        train_reflow(split, guard_directory=teacher_dir)
        (temp / "split" / "STOP").write_text("pause", encoding="utf-8")
        split.steps, split.resume = 4, "auto"
        assert train_reflow(split, guard_directory=teacher_dir)["state"] == "paused"
        (temp / "split" / "STOP").unlink()
        train_reflow(split, guard_directory=teacher_dir)
        a = torch.load(temp / "full" / "latest.pt", map_location="cpu", weights_only=False)
        b = torch.load(temp / "split" / "latest.pt", map_location="cpu", weights_only=False)
        assert all(torch.equal(value, b["model"][name]) for name, value in a["model"].items()), "CPU resume changed model weights"
        assert all(torch.equal(value, b["ema"][name]) for name, value in a["ema"].items()), "CPU resume changed EMA"
        reloaded = PixelFlowUNet(ModelConfig.from_dict(b["model_config"]))
        reloaded.load_state_dict(b["model"], strict=True)
        student_inference = torch.load(temp / "split" / "inference.pt", map_location="cpu", weights_only=False)
        reloaded.load_state_dict(student_inference["model"], strict=True)
        reloaded.eval()
        from .model import sample_flow
        for count in (1, 4):
            pixels = sample_flow(reloaded, "chat", seed=3, steps=count, size=32, device="cpu")
            assert pixels.shape == (1, 1, 32, 32) and bool(torch.isfinite(pixels).all())
        assert student_inference["provenance"]["dataset"] == str((temp / "pairs").resolve())
        random_args = parser.parse_args(common + ["--checkpoints", str(temp / "random"), "--steps", "1", "--student-init", "random"])
        train_reflow(random_args, guard_directory=teacher_dir)
        random_card = json.loads((temp / "random" / "model_card.json").read_text(encoding="utf-8"))
        assert random_card["student_fresh_random_initialization"] is True
        zero = parser.parse_args(["pairs", "--teacher", str(teacher_path), "--prompts", str(prompts_path),
                    "--output", str(temp / "zero-pairs"), "--count", "2", "--teacher-steps", "3",
                    "--device", "cpu", "--allow-cpu"])
        (temp / "zero-pairs").mkdir()
        (temp / "zero-pairs" / "STOP").write_text("pause", encoding="utf-8")
        assert prepare_pairs(zero, guard_directory=teacher_dir)["pairs"] == 0
        (temp / "zero-pairs" / "STOP").unlink()
        zero.resume = True
        assert prepare_pairs(zero, guard_directory=teacher_dir)["pairs"] == 2
        invalid_teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
        invalid_teacher["provenance"]["pretrained_weights"] = True
        invalid_path = temp / "invalid-teacher.pt"
        atomic_torch_save(invalid_path, invalid_teacher)
        try:
            read_teacher(invalid_path)
        except ValueError:
            pass
        else:
            raise AssertionError("Teacher provenance rejection failed")
        from concurrent.futures import ThreadPoolExecutor
        with model_access(teacher_dir), ThreadPoolExecutor(max_workers=1) as executor:
            try:
                executor.submit(prepare_pairs, pair_args, teacher_dir).result()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Another thread bypassed the shared model_access guard")
        try:
            separate_output(teacher_dir / "child", teacher_dir)
        except ValueError:
            pass
        else:
            raise AssertionError("Output/base overlap guard failed")
        result = {"passed": True, "device": "cpu", "test_parameters": teacher.parameter_count,
                  "fixed_pair_loss_before": losses[0], "fixed_pair_loss_after": losses[-1],
                  "checks": ["same-noise paired endpoints", "real gradients and fixed-pair loss descent",
                             "sharded pairs and checksums", "pair resume", "STOP pause",
                             "bitwise CPU full-vs-resumed model/EMA", "full and inference checkpoint loading",
                             "correct reflow dataset provenance", "finite 1/4-step inference",
                             "fresh random student option", "zero-pair STOP/resume",
                             "teacher provenance rejection", "shared-lock conflict rejection",
                             "output/base overlap rejection"],
                  "quality_or_speed_verified": False, "production_checkpoints_read": False}
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if args.report:
            atomic_json(Path(args.report).resolve(), result)
        return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pairs = commands.add_parser("pairs", help="Generate exact x0/teacher-x1 pairs, without training a student")
    pairs.add_argument("--teacher", default=str(BASE / "inference.pt"))
    pairs.add_argument("--prompts", required=True, help="UTF-8 text lines, or JSONL rows containing prompt")
    pairs.add_argument("--output", default=str(ROOT / "neural" / "reflow-data"))
    pairs.add_argument("--count", type=int, default=1024)
    pairs.add_argument("--batch-size", type=int, default=2)
    pairs.add_argument("--shard-size", type=int, default=32)
    pairs.add_argument("--teacher-steps", type=int, default=64)
    pairs.add_argument("--resume", action="store_true")
    train = commands.add_parser("train", help="Train a same-architecture reflow student; never modifies the base")
    train.add_argument("--pairs", default=str(ROOT / "neural" / "reflow-data"))
    train.add_argument("--teacher", help="Defaults to the frozen teacher recorded in the pairs manifest")
    train.add_argument("--checkpoints", default=str(ROOT / "neural" / "reflow-checkpoints"))
    train.add_argument("--student-init", choices=("teacher", "random"), default="teacher")
    train.add_argument("--resume", nargs="?", const="auto")
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--accum", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-5)
    train.add_argument("--lr-warmup-steps", type=int, default=100)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--ema-decay", type=float, default=.999)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--save-every", type=int, default=250)
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--no-checkpointing", action="store_true")
    for command in (pairs, train):
        command.add_argument("--device", default="cuda")
        command.add_argument("--allow-cpu", action="store_true")
        command.add_argument("--fp32", action="store_true")
        command.add_argument("--cpu-threads", type=int, default=4)
        command.add_argument("--seed", type=int, default=20261025)
    smoke = commands.add_parser("smoke-test", help="Temporary tiny CPU models only; does not read base weights")
    smoke.add_argument("--work", required=True, help="Existing task scratch directory for all temporary test files")
    smoke.add_argument("--report", help="Optional path to a JSON smoke report")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "smoke-test":
        smoke_test(args)
        return
    if args.batch_size < 1 or args.cpu_threads < 1 or not 0 <= args.seed < 2**63 - 1:
        parser.error("batch-size/cpu-threads must be positive and seed must be in 0..2**63-2")
    if args.command == "pairs":
        if args.count < 1 or args.shard_size < 1 or not 2 <= args.teacher_steps <= 1024:
            parser.error("count/shard-size must be positive; teacher-steps must be in 2..1024")
        result = prepare_pairs(args)
    else:
        if min(args.steps, args.accum, args.save_every, args.log_every) < 1 or args.lr_warmup_steps < 0:
            parser.error("steps/accum/save-every/log-every must be positive; warmup cannot be negative")
        if not all(math.isfinite(value) for value in (args.lr, args.weight_decay, args.ema_decay, args.max_grad_norm)) or args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.ema_decay < 1 or args.max_grad_norm <= 0:
            parser.error("Invalid lr/weight-decay/ema-decay/max-grad-norm")
        result = train_reflow(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
