"""Local training controls and read-only telemetry for the browser workbench."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from neural.locking import model_access
from neural.atomic_io import atomic_json
from neural.state_sampling import (APP_CURRICULUM_VERSION, INTERACTION_DATASET_VERSIONS,
    APP_FLOW_VERSION, APP_FLOW_COMBINED_VERSION)


def _is_native512_manifest(manifest):
    """Check the native rasterizer's manifest contract, not only its size."""
    return (type(manifest.get("size")) is int and manifest["size"] == 512
            and type(manifest.get("render_native_size")) is int
            and manifest["render_native_size"] == 512
            and manifest.get("rendering_version") == 3
            and manifest.get("render_method") == "native-vector-and-font-rasterization")


def _is_app_curriculum_manifest(manifest):
    """Require the audited native mixture contract before offering or launching it."""
    if not isinstance(manifest, dict):
        return False
    expected = {"dataset_version": APP_CURRICULUM_VERSION,
        "format": "npy-uint8-grayscale-shards", "size": 256,
        "render_native_size": 256, "layout_coordinate_size": 256,
        "rendering_version": 6, "dtype": "uint8",
        "render_method": "native-vector-and-font-rasterization",
        "normalization": "image / 127.5 - 1", "max_length": 192,
        "cross_split_pixel_leaks": 0, "source_cross_split_pixel_leaks": 0}
    if any(type(manifest.get(key)) is not type(value) or manifest[key] != value
            for key, value in expected.items()):
        return False
    counts = ("count", "train_count", "val_count")
    if (any(type(manifest.get(key)) is not int or manifest[key] <= 0 for key in counts)
            or manifest["train_count"] + manifest["val_count"] != manifest["count"]):
        return False
    sources = manifest.get("source_counts")
    kinds = manifest.get("kind_counts")
    if (not isinstance(sources, dict) or set(sources) != {"v3", "apps_v4"}
            or any(type(value) is not int or value <= 0 for value in sources.values())
            or sum(sources.values()) != manifest["count"]
            or not isinstance(kinds, dict)
            or any(type(value) is not int or value <= 0 for value in kinds.values())
            or sum(kinds.values()) != manifest["count"]
            or not all(kind in kinds for kind in ("tasks", "desktop", "editor"))
            or manifest.get("source_files_are_referenced") is not True):
        return False
    audit = manifest.get("audit")
    if not isinstance(audit, dict):
        return False
    audit_expected = {"passed": True, "native_size": 256,
        "train_val_pixel_leaks": 0, "source_new_split_pixel_leaks": 0,
        "empty_editor_body_checked": True, "caption_dispatch_checked": True,
        "complete_visible_labels_checked": True,
        "no_switch_controls_in_new_families": True,
        "bilingual_layout_pairs_same_split": True,
        "combined_global_indices_unique": True,
        "new_actual_pixel_hashes_checked": sources["apps_v4"],
        "source_actual_pixel_hashes_checked": sources["v3"]}
    if any(type(audit.get(key)) is not type(value) or audit[key] != value
            for key, value in audit_expected.items()):
        return False
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        return False
    totals = dict.fromkeys(counts, 0)
    for shard in shards:
        if (not isinstance(shard, dict)
                or any(not isinstance(shard.get(key), str) or not shard[key]
                    for key in ("images", "metadata"))
                or any(type(shard.get(key)) is not int or shard[key] < 0 for key in counts)
                or shard["count"] <= 0
                or shard["train_count"] + shard["val_count"] != shard["count"]
                or type(shard.get("start_index")) is not int
                or shard["start_index"] != totals["count"]):
            return False
        for key in counts:
            totals[key] += shard[key]
    return (all(totals[key] == manifest[key] for key in counts)
            and isinstance(manifest.get("vocab"), str) and bool(manifest["vocab"]))


def _app_curriculum_ready(dataset, manifest):
    if not _is_app_curriculum_manifest(manifest):
        return False
    paths = [manifest["vocab"]] + [shard[key] for shard in manifest["shards"]
        for key in ("images", "metadata")]
    try:
        return all((Path(dataset) / path).is_file() for path in paths)
    except (OSError, ValueError):
        return False


def _is_app_flows_manifest(manifest):
    """Require a reference-only V4+V5 mixture and completed independent target audit."""
    if not isinstance(manifest,dict):
        return False
    expected={'dataset_version':APP_FLOW_COMBINED_VERSION,'format':'npy-uint8-grayscale-shards',
        'size':256,'render_native_size':256,'layout_coordinate_size':256,'rendering_version':7,
        'render_method':'native-vector-and-font-rasterization','dtype':'uint8',
        'normalization':'image / 127.5 - 1','max_length':192,'source_files_are_referenced':True,
        'training_targets_only':True,'cross_split_pixel_leaks':0}
    if any(type(manifest.get(k)) is not type(v) or manifest[k]!=v for k,v in expected.items()):
        return False
    counts=('count','train_count','val_count')
    if (any(type(manifest.get(k)) is not int or manifest[k]<=0 for k in counts)
            or manifest['train_count']+manifest['val_count']!=manifest['count']):
        return False
    sources=manifest.get('source_counts');audit=manifest.get('audit')
    if (not isinstance(sources,dict) or set(sources)!={'v4','app_flows_v5'}
            or any(type(v) is not int or v<=0 for v in sources.values())
            or sum(sources.values())!=manifest['count'] or not isinstance(audit,dict)):
        return False
    checks={'passed':True,'native_size':256,'train_val_pixel_leaks':0,'source_new_split_pixel_leaks':0,
        'combined_global_indices_unique':True,'caption_dispatch_checked':True,
        'independent_audit_complete':True,'manual_target_review_complete':True,
        'new_actual_pixel_hashes_checked':sources['app_flows_v5'],'outside_effect_mask_changed_pixels':0}
    if any(type(audit.get(k)) is not type(v) or audit[k]!=v for k,v in checks.items()):
        return False
    if (not isinstance(manifest.get('sources'),dict) or set(manifest['sources'])!={'v4','app_flows_v5'}
            or any(not isinstance(v,str) or not v for v in manifest['sources'].values())
            or any(not isinstance(audit.get(k),str) or len(audit[k])!=64 for k in
                ('source_flow_metadata_sha256','combined_flow_metadata_sha256'))):
        return False
    shards=manifest.get('shards');totals=dict.fromkeys(counts,0)
    if not isinstance(shards,list) or len(shards)<2:
        return False
    for shard in shards:
        if (not isinstance(shard,dict) or any(not isinstance(shard.get(k),str) or not shard[k] for k in ('images','metadata'))
                or any(type(shard.get(k)) is not int or shard[k]<0 for k in counts)
                or shard['count']<=0 or shard['train_count']+shard['val_count']!=shard['count']
                or type(shard.get('start_index')) is not int or shard['start_index']!=totals['count']):
            return False
        for key in counts:totals[key]+=shard[key]
    return (all(totals[k]==manifest[k] for k in counts)
        and isinstance(manifest.get('vocab'),str) and bool(manifest['vocab']))


def _app_flows_ready(dataset,manifest):
    if not _is_app_flows_manifest(manifest):
        return False
    try:
        dataset=Path(dataset)
        if not all((dataset/name).is_file() for name in [manifest['vocab']]+[
                shard[key] for shard in manifest['shards'] for key in ('images','metadata')]):
            return False
        v4=dataset/manifest['sources']['v4'];flow=dataset/manifest['sources']['app_flows_v5']
        v4_manifest=read_json(v4/'manifest.json');flow_manifest=read_json(flow/'manifest.json')
        report=read_json(flow/'independent-audit.json')
        if (not _app_curriculum_ready(v4,v4_manifest)
                or flow_manifest.get('dataset_version')!=APP_FLOW_VERSION
                or flow_manifest.get('count')!=manifest['source_counts']['app_flows_v5']
                or v4_manifest.get('count')!=manifest['source_counts']['v4']):
            return False
        flow_expected={'format':'npy-uint8-grayscale-shards','size':256,'render_native_size':256,
            'layout_coordinate_size':256,'rendering_version':7,'dtype':'uint8','max_length':192,
            'render_method':'native-vector-and-font-rasterization','normalization':'image / 127.5 - 1',
            'training_targets_only':True,'real_business_executed':False,'active_training_changed':False}
        if any(type(flow_manifest.get(k)) is not type(v) or flow_manifest[k]!=v for k,v in flow_expected.items()):
            return False
        if (type(flow_manifest.get('train_count')) is not int or type(flow_manifest.get('val_count')) is not int
                or flow_manifest['train_count']<=0 or flow_manifest['val_count']<=0
                or flow_manifest['train_count']+flow_manifest['val_count']!=flow_manifest['count']
                or manifest['train_count']!=v4_manifest['train_count']+flow_manifest['train_count']
                or manifest['val_count']!=v4_manifest['val_count']+flow_manifest['val_count']
                or report.get('split_counts')!={'train':flow_manifest['train_count'],'val':flow_manifest['val_count']}
                or not isinstance(flow_manifest.get('shards'),list) or len(flow_manifest['shards'])!=1):
            return False
        required={'passed':True,'cpu_only':True,'torch_or_gpu_used':False,
            'native_size':256,'actual_rasters_hashed':flow_manifest['count'],'label_clipping_or_truncation_failures':0,
            'cross_split_actual_pixel_leaks':0,'cross_split_story_leaks':0,
            'source_v4_new_split_conflicts':0,'outside_effect_mask_changed_pixels':0,
            'manual_target_review_complete':True,'manual_target_review_images':20,
            'neural_generation_tested':False,'whole_app_quality_accepted':False}
        if any(type(report.get(k)) is not type(v) or report[k]!=v for k,v in required.items()):
            return False
        flow_audit=flow_manifest.get('audit',{})
        if any(flow_audit.get(k) is not True for k in ('passed','independent_audit_complete',
                'independent_actual_pixel_audit_passed','manual_target_review_complete')):
            return False
        if len(manifest['shards'])!=len(v4_manifest['shards'])+1:
            return False
        for actual,original in zip(manifest['shards'],v4_manifest['shards']):
            if (any((dataset/actual[key]).resolve()!=(v4/original[key]).resolve() for key in ('images','metadata'))
                    or any(actual.get(key)!=original.get(key) for key in ('count','train_count','val_count','start_index'))):
                return False
        last=manifest['shards'][-1]
        if ((dataset/last['images']).resolve()!=(flow/flow_manifest['shards'][0]['images']).resolve()
                or last['count']!=flow_manifest['count'] or last['start_index']!=v4_manifest['count']
                or last['train_count']!=flow_manifest['train_count'] or last['val_count']!=flow_manifest['val_count']):
            return False
        image_stat=(flow/flow_manifest['shards'][0]['images']).stat()
        if (report.get('stored_npy_bytes')!=image_stat.st_size
                or report.get('audited_images_mtime_ns')!=image_stat.st_mtime_ns
                or type(report.get('actual_label_rasters_replayed_and_equal')) is not int
                or report['actual_label_rasters_replayed_and_equal']<flow_manifest['count']
                or type(report.get('max_effect_area_percent')) not in (int,float)
                or not 0<report['max_effect_area_percent']<=35):
            return False
        for path,key in ((flow/flow_manifest['shards'][0]['metadata'],'source_flow_metadata_sha256'),
                (dataset/last['metadata'],'combined_flow_metadata_sha256')):
            with path.open('rb') as handle:
                if hashlib.file_digest(handle,'sha256').hexdigest()!=manifest['audit'][key]:
                    return False
        if report.get('audited_metadata_sha256')!=manifest['audit']['source_flow_metadata_sha256']:
            return False
        return True
    except (OSError,ValueError,TypeError,KeyError,IndexError):
        return False


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {} if default is None else default


def read_training_config(path):
    """Distinguish a missing launch record from one that cannot be trusted."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("launch record must be a JSON object")
        return value, None
    except FileNotFoundError:
        return {}, None
    except (OSError, ValueError) as error:
        return {}, type(error).__name__


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None if ctypes.get_last_error() == 5 else False
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return None
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


class TrainingControl:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.directory = self.root / "neural" / "checkpoints"
        self.work = self.root / "work"
        self.control_lock = threading.Lock()
        self.telemetry_lock = threading.Lock()
        self.gpu_cache = None
        self.gpu_updated = 0.0
        self.loss_position = 0
        self.loss_pending = ""
        self.loss_rows = {}
        self.process = None

    def training(self):
        state = read_json(self.directory / "status.json", {"state": "preparing", "step": 0})
        if self.process is not None and self.process.poll() is None and state.get("state") in ("paused", "complete", "completed", "stopped", "failed"):
            state = dict(state, state="preparing", pid=self.process.pid, error=None)
        if state.get("state") in ("running", "preparing") and state.get("pid"):
            alive = process_alive(state["pid"])
            if alive is False:
                state = dict(state, state="failed", error="训练进程已退出。可以从最近保存的检查点继续。")
        state["pause_requested"] = (self.directory / "STOP").exists() and state.get("state") in ("running", "preparing")
        return state

    def data_info(self, training=None):
        training = training or self.training()
        config = read_json(self.work / "training-control.json")
        current_path = Path(training.get("data_path") or config.get("data_path") or self.root / "neural" / "data")
        manifest = read_json(current_path / "manifest.json")
        available = [{"id": "current", "label": "当前训练数据", "count": manifest.get("count", 0)}]
        v2 = self.root / "neural" / "data-v2"
        v2_manifest = read_json(v2 / "manifest.json")
        if v2_manifest.get("count", 0) > 0:
            available.append({"id": "v2", "label": "文案校正版 V2", "count": v2_manifest["count"]})
        interaction = self.root / "neural" / "data-v2-interaction"
        interaction_manifest = read_json(interaction / "manifest.json")
        if (interaction_manifest.get("dataset_version") == "v2-plus-native256-interaction-v1" and
                interaction_manifest.get("size") == 256 and interaction_manifest.get("train_count", 0) > 0):
            available.append({"id": "interaction", "label": "V2 + 交互状态配对", "count": interaction_manifest["count"]})
        curriculum = self.root / "neural" / "data-v3-curriculum"
        curriculum_manifest = read_json(curriculum / "manifest.json")
        if (curriculum_manifest.get("dataset_version") == "v2-plus-native256-curriculum-v2" and
                curriculum_manifest.get("size") == 256 and curriculum_manifest.get("train_count", 0) > 0):
            available.append({"id": "curriculum", "label": "自然提示词与正确开关状态 V3", "count": curriculum_manifest["count"]})
        applications = self.root / "neural" / "data-app-curriculum-v4" / "combined"
        applications_manifest = read_json(applications / "manifest.json")
        applications_ready = _app_curriculum_ready(applications, applications_manifest)
        if applications_ready:
            available.append({"id": "applications", "label": "V4 · 桌面、待办与空白文档", "count": applications_manifest["count"]})
        app_flows=self.root/'neural/data-app-flows-v5-20261028/combined'
        app_flows_manifest=read_json(app_flows/'manifest.json')
        app_flows_ready=_app_flows_ready(app_flows,app_flows_manifest)
        if app_flows_ready:
            available.append({'id':'app_flows','label':'V5 · 待办与笔记完整状态流','count':app_flows_manifest['count']})
        native512 = self.root / "neural" / "data-native512"
        native512_manifest = read_json(native512 / "manifest.json")
        if native512_manifest.get("count", 0) > 0 and _is_native512_manifest(native512_manifest):
            available.append({"id": "native512", "label": "原生 512×512", "count": native512_manifest["count"]})
        return {"path": str(current_path), "count": manifest.get("count", 0),
                "native_size": manifest.get("size"), "native_resolution": manifest.get("size"),
                "native_rendering": _is_native512_manifest(manifest) if manifest.get("size") == 512 else manifest.get("render_method") == "native-vector-and-font-rasterization",
                "train_count": manifest.get("train_count", 0), "val_count": manifest.get("val_count", 0),
                "version": ('V5 · 待办与笔记完整状态流' if app_flows_ready else 'V5 数据待检查') if current_path.resolve()==app_flows.resolve() else ("原生 512×512" if _is_native512_manifest(manifest) else "512 数据待检查") if current_path.resolve() == native512.resolve() else ("V4 · 桌面、待办与空白文档" if applications_ready else "V4 数据待检查") if current_path.resolve() == applications.resolve() else "V3 · 自然提示词与正确状态" if current_path.resolve() == curriculum.resolve() else "V2 + 交互状态配对" if current_path.resolve() == interaction.resolve() else "V2 · 文案校正" if current_path.resolve() == v2.resolve() else "V1",
                "available": available}

    def gpu(self):
        with self.telemetry_lock:
            now = time.monotonic()
            if now - self.gpu_updated < 5:
                return self.gpu_cache
            self.gpu_updated = now
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
                def number(value):
                    try:
                        converted = float(value)
                        return converted if math.isfinite(converted) else None
                    except ValueError:
                        return None
                self.gpu_cache = dict(zip(("name", "used_mb", "total_mb", "utilization_percent", "temperature_c", "power_w"),
                    [fields[0]] + [number(field) for field in fields[1:6]]))
            except (OSError, subprocess.SubprocessError, IndexError):
                self.gpu_cache = None
            return self.gpu_cache

    def history(self):
        path = self.directory / "training.jsonl"
        with self.telemetry_lock:
            try:
                if path.stat().st_size < self.loss_position:
                    self.loss_rows.clear()
                    self.loss_position = 0
                    self.loss_pending = ""
                with path.open("rb") as source:
                    source.seek(self.loss_position)
                    data = source.read()
                    self.loss_position = source.tell()
                lines = (self.loss_pending + data.decode("utf-8", errors="replace")).split("\n")
                self.loss_pending = lines.pop()
                for line in lines:
                    try:
                        entry = json.loads(line)
                        if isinstance(entry.get("step"), int) and isinstance(entry.get("loss"), (int, float)):
                            self.loss_rows[entry["step"]] = {key: entry.get(key) for key in
                                ("step", "loss", "elapsed_seconds", "step_seconds")}
                    except ValueError:
                        continue
                points = sorted(self.loss_rows.values(), key=lambda row: row["step"])
                stride = max(1, math.ceil(len(points) / 800))
                sampled = points[::stride]
                if points and sampled[-1]["step"] != points[-1]["step"]:
                    sampled.append(points[-1])
                return sampled
            except OSError:
                return []

    def log_tail(self):
        config = read_json(self.work / "training-control.json")
        # The native512 migration used a separate stdout file. Honor the
        # recorded run log, but never treat arbitrary config paths as readable.
        recorded = config.get("log")
        allowed = (isinstance(recorded, str) and
            (recorded in ("training-process.log", "training-native512.log") or
             re.fullmatch(r"training-run-[0-9T-]+-[0-9]+\.log", recorded) is not None))
        path = self.work / recorded if allowed else None
        if path is None or not path.is_file():
            known = [self.work / "training-process.log", self.work / "training-native512.log",
                     self.directory / "training.jsonl"]
            existing = [candidate for candidate in known if candidate.is_file()]
            path = max(existing, key=lambda candidate: candidate.stat().st_mtime_ns) if existing else None
        if path is None:
            return []
        try:
            with path.open("rb") as source:
                source.seek(max(0, path.stat().st_size - 20000))
                return source.read().decode("utf-8", errors="replace").splitlines()[-30:]
        except OSError:
            return []

    def snapshot(self):
        training = self.training()
        _, config_error = read_training_config(self.work / "training-control.json")
        card = read_json(self.directory / "model_card.json")
        preview_info = read_json(self.directory / "previews" / "latest.json")
        preview_file = self.directory / "previews" / "latest.png"
        files = []
        for name, kind in (("latest.pt", "完整训练状态"), ("inference.pt", "EMA 推理权重")):
            path = self.directory / name
            if path.exists():
                stat = path.stat()
                files.append({"name": name, "type": kind, "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "step": card.get("checkpoint_step")})
        return {"training": training, "model": card, "gpu": self.gpu(),
                "configuration": {"launch_record_valid": config_error is None,
                    "warning": "训练启动记录损坏；继续训练时将检查状态记录或显式参数，避免使用默认值。" if config_error else None},
                "capabilities": {"native_training_resolution": True},
                "data": self.data_info(training),
                "preview": {**preview_info, "url": "/api/training-preview.png" if preview_file.exists() else None},
                "checkpoints": files, "loss_history": self.history(),
                "validation": read_json(self.directory / "validation" / "latest.json", None) or None,
                "state_validation": read_json(self.directory / "validation" / "state-latest.json", None) or None,
                "logs": self.log_tail()}

    def pause(self):
        with self.control_lock:
            training = self.training()
            if training.get("state") not in ("running", "preparing"):
                return {"ok": True, "message": "训练当前没有运行。"}
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / "STOP").write_text("pause requested from local workbench", encoding="ascii")
            return {"ok": True, "message": "已请求暂停，当前训练步结束后会保存检查点。"}

    def start(self, data):
        if not isinstance(data, dict):
            raise ValueError("请求需要 JSON 对象。")
        with self.control_lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("训练正在启动或运行，无需重复启动。")
            training = self.training()
            if training.get("state") in ("running", "preparing") and process_alive(training.get("pid")) is not False:
                raise RuntimeError("训练已经在运行，无需重复启动。")
            def integer(name, default, lower, upper):
                value = data.get(name, default)
                if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                    raise ValueError(f"{name} 需要是 {lower}–{upper} 之间的整数。")
                return value
            config, config_error = read_training_config(self.work / "training-control.json")
            card = read_json(self.directory / "model_card.json")
            def saved_integer(field, config_field, fallback, lower, upper):
                for value in (training.get(field), config.get(config_field)):
                    if type(value) is int and lower <= value <= upper:
                        return value
                if config_error and config_field not in data:
                    raise ValueError(f"训练启动记录损坏，且状态记录未保存 {config_field}；请填写该参数后继续，避免使用默认值。")
                return fallback
            steps = integer("steps", saved_integer("total_steps", "steps", 10000, 1, 1000000), 1, 1000000)
            saved_batch = saved_integer("batch_size", "batch_size", 4, 1, 16)
            saved_accum = saved_integer("gradient_accumulation", "accum", 2, 1, 32)
            caption_augment=data.get("caption_augment")
            if caption_augment is not None and (isinstance(caption_augment,bool) or not isinstance(caption_augment,(int,float)) or not math.isfinite(caption_augment) or not 0<=caption_augment<=1):
                raise ValueError("提示词增强比例需要是 0–1 之间的数值。")
            selected = data.get("data", "current")
            if selected not in ("current", "v2", "native512", "interaction", "curriculum", "applications",'app_flows'):
                raise ValueError("未知数据集。")
            if config_error and selected == "current" and not training.get("data_path"):
                raise ValueError("训练启动记录损坏，状态记录缺少数据路径；请明确选择训练数据后继续。")
            dataset = (self.root / "neural" / "data-native512" if selected == "native512" else
                self.root/'neural/data-app-flows-v5-20261028/combined' if selected=='app_flows' else
                self.root / "neural" / "data-app-curriculum-v4" / "combined" if selected == "applications" else
                self.root / "neural" / "data-v3-curriculum" if selected == "curriculum" else
                self.root / "neural" / "data-v2-interaction" if selected == "interaction" else
                self.root / "neural" / "data-v2" if selected == "v2" else Path(self.data_info(training)["path"]))
            manifest = read_json(dataset / "manifest.json")
            if not isinstance(manifest, dict):
                raise ValueError("所选训练数据清单需要是 JSON 对象。")
            applications = self.root / "neural" / "data-app-curriculum-v4" / "combined"
            app_flows=self.root/'neural/data-app-flows-v5-20261028/combined'
            using_app_flows=(selected=='app_flows' or dataset.resolve()==app_flows.resolve()
                or manifest.get('dataset_version')==APP_FLOW_COMBINED_VERSION)
            if using_app_flows and not _app_flows_ready(dataset,manifest):
                raise ValueError('V5 应用状态流尚未通过完整清单、独立靶图审计和引用文件检查。')
            if (selected == "applications" or dataset.resolve() == applications.resolve()
                    or manifest.get("dataset_version") == APP_CURRICULUM_VERSION):
                if not _app_curriculum_ready(dataset, manifest):
                    raise ValueError("V4 应用训练数据尚未通过原生 256 尺寸、数据版本和完整审计检查。")
            if manifest.get("train_count", 0) <= 0:
                raise ValueError("所选训练数据尚未准备好。")
            if selected == "interaction" and (manifest.get("dataset_version") != "v2-plus-native256-interaction-v1" or manifest.get("size") != 256):
                raise ValueError("交互状态配对数据尚未通过原生尺寸检查。")
            if selected == "curriculum" and (manifest.get("dataset_version") != "v2-plus-native256-curriculum-v2" or manifest.get("size") != 256):
                raise ValueError("V3 状态训练数据尚未通过原生尺寸检查。")
            interaction_fraction = data.get("interaction_fraction")
            if interaction_fraction is None:
                interaction_fraction = next((source.get("interaction_fraction") for source in (training, config)
                    if source.get("data_path") and Path(source["data_path"]).resolve() == dataset.resolve()
                    and source.get("interaction_fraction") is not None), None)
            if interaction_fraction is not None:
                if (isinstance(interaction_fraction, bool) or not isinstance(interaction_fraction, (int, float))
                        or not math.isfinite(interaction_fraction) or not 0 <= interaction_fraction <= 1):
                    raise ValueError("交互状态采样比例需要是 0–1 之间的有限数值。")
                if manifest.get("dataset_version") not in INTERACTION_DATASET_VERSIONS:
                    raise ValueError("交互状态采样比例需要使用已检查的交互配对数据。")
            app_flow_fraction=data.get('app_flow_fraction')
            if app_flow_fraction is None:
                app_flow_fraction=next((source.get('app_flow_fraction') for source in (training,config)
                    if source.get('data_path') and Path(source['data_path']).resolve()==dataset.resolve()
                    and source.get('app_flow_fraction') is not None),None)
            if using_app_flows and (app_flow_fraction is None or interaction_fraction is None):
                raise ValueError('V5 续训需明确保存或提供 app_flow_fraction 与 interaction_fraction，避免继承旧数据的隐藏配额。')
            if app_flow_fraction is not None:
                if (isinstance(app_flow_fraction,bool) or not isinstance(app_flow_fraction,(int,float))
                        or not math.isfinite(app_flow_fraction) or not 0<=app_flow_fraction<=1):
                    raise ValueError('应用状态流采样比例需要是 0–1 之间的有限数值。')
                if app_flow_fraction>0 and not using_app_flows:
                    raise ValueError('应用状态流采样比例需要已审计的 V5 组合数据。')
            if app_flow_fraction is not None and app_flow_fraction+(interaction_fraction or 0)>1:
                raise ValueError('应用状态流与旧交互状态采样比例之和不能超过 1。')
            if not using_app_flows:
                # Explicitly clear a prior V5 checkpoint's quota on V4/V3.
                # --uniform-sampling below clears both quotas for other data.
                app_flow_fraction=0.0
            region_weights = {}
            for name in ("text_roi_weight", "state_roi_weight"):
                value = data.get(name)
                if value is None:
                    value = next((source.get(name) for source in (training, config, card) if source.get(name) is not None), None)
                if value is not None:
                    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 16):
                        raise ValueError("文字和状态区域训练权重需要是 0–16 之间的有限数值。")
                    region_weights[name] = value
            resolution = manifest.get("size", 256)
            if type(resolution) is not int or resolution not in (256, 512):
                raise ValueError("训练数据的原生分辨率须为 256 或 512。")
            if resolution == 512 and not _is_native512_manifest(manifest):
                raise ValueError("512 数据必须来自原生绘制，不能使用放大图。")
            previous_resolution = training.get("native_resolution", config.get("native_resolution"))
            migrating_512 = resolution == 512 and previous_resolution != 512
            batch = integer("batch_size", 1 if migrating_512 else saved_batch, 1, 16)
            accum = integer("accum", min(32, saved_batch * saved_accum) if migrating_512 else saved_accum, 1, 32)
            edge_weight = data.get("edge_weight")
            if edge_weight is None:
                edge_weight = next((value for value in (training.get("edge_weight"), config.get("edge_weight"), card.get("edge_weight"))
                    if value is not None), None)
            if edge_weight is not None and (isinstance(edge_weight, bool) or not isinstance(edge_weight, (int, float))
                    or not math.isfinite(edge_weight) or not 0 <= edge_weight <= 16):
                raise ValueError("边缘训练权重需要是 0–16 之间的有限数值。")
            previous_step = int(card.get("checkpoint_step", 0))
            if steps <= previous_step:
                raise ValueError(f"目标是累计训练步数，需要大于已保存的 {previous_step} 步。")
            if (manifest.get('dataset_version') in (APP_CURRICULUM_VERSION,APP_FLOW_COMBINED_VERSION) and previous_step > 0
                    and not (self.directory / 'latest.pt').is_file()):
                raise ValueError('应用续训缺少完整 latest.pt 检查点，不能用随机权重替代已有训练。')
            interpreter = self.root / ".venv" / "Scripts" / "python.exe" if os.name == "nt" else self.root / ".venv" / "bin" / "python"
            if not interpreter.exists():
                raise RuntimeError("项目 Python 环境不存在，请检查 .venv。")
            with model_access(self.directory):
                # Probe and release before spawning: the child acquires this
                # same lock itself for the full lifetime of training.
                pass
            stop = self.directory / "STOP"
            args = [str(interpreter), "-u", "-m", "neural.train", "--data", str(dataset),
                "--steps", str(steps), "--batch-size", str(batch), "--accum", str(accum),
                "--warmup-steps", "0", "--resolution", str(resolution), "--save-every", "500",
                "--sample-every", "500", "--preview-steps", "16", "--validate-every", "1000"]
            if manifest.get("dataset_version") not in INTERACTION_DATASET_VERSIONS:
                args.insert(4, "--uniform-sampling")
            latest = self.directory / "latest.pt"
            if latest.exists():
                args += ["--resume", str(latest)]
            if caption_augment is not None:
                args += ["--caption-augment",str(caption_augment)]
            if edge_weight is not None:
                args += ["--edge-weight",str(edge_weight)]
            if interaction_fraction is not None:
                args += ["--interaction-fraction", str(interaction_fraction)]
            if app_flow_fraction is not None and manifest.get('dataset_version') in INTERACTION_DATASET_VERSIONS:
                args += ['--app-flow-fraction',str(app_flow_fraction)]
            for name, value in region_weights.items():
                args += ["--" + name.replace("_", "-"), str(value)]
            self.work.mkdir(exist_ok=True)
            env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            log_name = "training-run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "-" + str(os.getpid()) + ".log"
            record = {"data_path": str(dataset), "steps": steps,
                "batch_size": batch, "accum": accum, "native_resolution": resolution,
                "launcher_pid": None, "log": log_name}
            if edge_weight is not None:
                record["edge_weight"] = edge_weight
            if interaction_fraction is not None:
                record["interaction_fraction"] = interaction_fraction
            record['app_flow_fraction']=app_flow_fraction
            record.update(region_weights)
            effective_caption = caption_augment if caption_augment is not None else next((value for value in
                (training.get("caption_augmentation_probability"), config.get("caption_augment"), card.get("caption_augmentation_probability"))
                if value is not None), None)
            if effective_caption is not None:
                record["caption_augment"] = effective_caption
            if config_error:
                record["recovered_launch_record_error"] = config_error
            # Persist every effective launch parameter before starting a child.
            # A disk/flush failure must not start an unrecorded training run.
            atomic_json(self.work / "training-control.json", record)
            if stop.exists():
                stop.unlink()
            with (self.work / log_name).open("ab", buffering=0) as log:
                log.write(("\n[training launch " + datetime.now(timezone.utc).isoformat() + "]\n").encode())
                self.process = subprocess.Popen(args, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, creationflags=flags, start_new_session=os.name != "nt")
            record["launcher_pid"] = self.process.pid
            launch_warning = None
            try:
                atomic_json(self.work / "training-control.json", record)
            except OSError:
                # The child is already running. Retain its live handle and the
                # durable intent; do not report a failed launch and invite retry.
                launch_warning = "训练已启动；启动进程编号未能写回记录，训练状态将由进程自身更新。"
            # Do not overwrite the child's status: its own first write may
            # already have happened. Keep a short launch record separately.
            return {"ok": True, "message": f"已启动训练，将从保存的 {previous_step} 步继续至累计 {steps} 步。" +
                    (" 原启动记录损坏，已从状态记录和本次参数恢复。" if config_error else ""),
                    "pid": self.process.pid, "launch_record_recovered": bool(config_error),
                    "warning": launch_warning}
