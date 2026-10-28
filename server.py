"""Local workbench for cemivoAI's randomly initialized pixel generative model.

There is intentionally no classification/template image fallback in this service.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
from contextlib import contextmanager
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import secrets
import threading
import time
import traceback
from urllib.parse import urlsplit, parse_qs, quote
from neural.locking import model_access
from training_control import TrainingControl
from live_app import LiveApp, LiveError
from agent_runtime import AgentRuntime

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "neural" / "checkpoints"
VERSION = "2.11.4-experimental"


@contextmanager
def immediate_sample_access(lock):
    if not lock.acquire(blocking=False):
        raise RuntimeError("上一张图片仍在生成，请完成后再提交。")
    try:
        yield
    finally:
        lock.release()


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


class Studio:
    def __init__(self, snapshot_directory=None):
        self.cache_lock = threading.Lock()
        self.sample_lock = threading.Lock()
        self.images = OrderedDict()
        self.generator = None
        self.loaded_mtime = None
        self.last_error = None
        self.training_control = TrainingControl(ROOT)
        self.live = LiveApp(ROOT, snapshot_directory=snapshot_directory)
        self.live.generation_lock = self.sample_lock
        self.agent = AgentRuntime(ROOT, self.live)

    def status(self):
        card = read_json(CHECKPOINTS / "model_card.json", {})
        training = {"state": "preparing", "step": 0,
            "total_steps": 0, "loss": None, "samples_seen": 0, "updated_at": None,
            "error": None, **self.training_control.training()}
        model = {"name": "cemivo-pixel-flow", "parameters": None, "native_width": 256,
                 "native_height": 256, "checkpoint_step": 0, "random_initialization": True}
        model.update(card)
        if not model.get("parameters") and training.get("parameters"):
            model["parameters"] = training["parameters"]
        if not card and training.get("native_resolution"):
            model["native_width"] = model["native_height"] = training["native_resolution"]
        preview_info = read_json(CHECKPOINTS / "previews" / "latest.json", {})
        preview = CHECKPOINTS / "previews" / "latest.png"
        checkpoint = CHECKPOINTS / "inference.pt"
        ready = checkpoint.exists() and training.get("state") in ("paused", "complete", "completed", "stopped")
        if self.last_error:
            training = dict(training, error=self.last_error)
        return {"app_id": "cemivoAI.neural", "version": VERSION, "ready": ready,
                "model": model, "training": training, "preview_step": preview_info.get("step"),
                "capabilities": {"sampling_steps": [4, 8, 16, 32],
                    "resolutions": [int(model.get("native_width", 256))], "formats": ["png"]},
                "preview_url": "/api/training-preview.png" if preview.exists() else None}

    def generate(self, data):
        if not isinstance(data, dict):
            raise ValueError("请求需要一个 JSON 对象。")
        prompt = data.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
            raise ValueError("提示词需要 1–2000 个字符。")
        if len(prompt.strip().encode("utf-8")) > 190:
            raise ValueError("当前模型适合短提示词，请精简到约 60 个汉字以内。")
        seed = data.get("seed", secrets.randbelow(2**31))
        steps = data.get("steps", 16)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("种子需要是 0–2147483647 之间的整数。")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps not in (4, 8, 16, 32):
            raise ValueError("采样步数可选 4、8、16、32。")
        state = self.status()
        if not state["ready"]:
            raise RuntimeError("模型仍在准备或训练中，请等待保存并暂停训练后再生成。")
        size = int(state["model"].get("native_width", 256))
        if data.get("width", size) != size or data.get("height", size) != size:
            raise ValueError(f"当前模型的原生输出尺寸是 {size}×{size}。")
        with immediate_sample_access(self.sample_lock), model_access(CHECKPOINTS):
            state = self.status()
            if not state["ready"]:
                raise RuntimeError("模型正在训练，请待保存并暂停后再采样。")
            size = int(state["model"].get("native_width", 256))
            if data.get("width", size) != size or data.get("height", size) != size:
                raise ValueError(f"当前模型的原生输出尺寸已更新为 {size}×{size}，请刷新后重试。")
            checkpoint = CHECKPOINTS / "inference.pt"
            stamp = checkpoint.stat().st_mtime_ns
            load_ms = 0.0
            if self.generator is None or stamp != self.loaded_mtime:
                from neural.sample import NeuralGenerator
                started_load = time.perf_counter()
                self.generator = NeuralGenerator(checkpoint, device="cuda")
                self.loaded_mtime = stamp
                load_ms = (time.perf_counter()-started_load)*1000
            start = time.perf_counter()
            image = self.generator.generate(prompt.strip(), seed=seed, steps=steps, size=size)
            sampled = time.perf_counter()
            output = BytesIO()
            image.convert("L").save(output, format="PNG", compress_level=2, optimize=False)
            encoded = time.perf_counter()
        image_id = secrets.token_hex(10)
        with self.cache_lock:
            self.images[image_id] = output.getvalue()
            while len(self.images) > 96:
                self.images.popitem(last=False)
        self.last_error = None
        return {"id": image_id, "png_url": f"/api/images/{image_id}.png", "width": size,
                "height": size, "seed": seed, "steps": steps, "prompt": prompt.strip(),
                "model": {"name": state["model"]["name"], "checkpoint_step": state["model"].get("checkpoint_step", 0)},
                "timings": {"sample_ms": round((sampled-start)*1000, 2),
                    "encode_ms": round((encoded-sampled)*1000, 2),
                    "total_ms": round((encoded-start)*1000, 2), "load_ms": round(load_ms, 2)}}


class Handler(BaseHTTPRequestHandler):
    server_version = "cemivoAI/2.11"

    def log_message(self, fmt, *args):
        if not any(route in str(args) for route in ("/api/status", "/api/training/status", "/api/live/status")):
            print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}", flush=True)

    def send_bytes(self, status, body, content_type, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        if filename:
            self.send_header("Content-Disposition", "attachment; filename=\"cemivo-download\"; filename*=UTF-8''" + quote(str(filename), safe=""))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionError, OSError):
            pass

    def json(self, code, value):
        return self.send_bytes(code, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/status":
            return self.json(200, self.server.studio.status())
        if path == "/api/training/status":
            return self.json(200, self.server.studio.training_control.snapshot())
        if path == "/api/live/status":
            return self.json(200, self.server.studio.live.status())
        if path == "/api/agent/workspace":
            return self.json(200, self.server.studio.agent.workspace())
        if path == "/api/agent/request":
            try:
                query = parse_qs(urlsplit(self.path).query)
                return self.json(200, self.server.studio.agent.request_status(query.get("id", [None])[0]))
            except LiveError as error:
                return self.json(error.status, {"error": str(error), "code": error.code})
        if path.startswith("/api/agent/exports/"):
            token = path.removeprefix("/api/agent/exports/")
            result = self.server.studio.agent.tools.get_export(token)
            if result is None:
                return self.json(404, {"error": "未找到此导出文档。"})
            body, filename = result
            return self.send_bytes(200, body, "text/plain; charset=utf-8", filename)
        if path == "/api/live/session":
            try:
                query = parse_qs(urlsplit(self.path).query)
                return self.json(200, self.server.studio.live.get(query.get("id", [None])[0]))
            except LiveError as error:
                return self.json(error.status, {"error": str(error), "code": error.code})
        if path == '/api/live/current':
            try:
                return self.json(200,self.server.studio.live.current())
            except LiveError as error:
                return self.json(error.status,{'error':str(error),'code':error.code})
        if path.startswith("/api/live/images/"):
            image_id = path.removeprefix("/api/live/images/").removesuffix(".png")
            body = self.server.studio.live.image(image_id)
            return self.send_bytes(200, body, "image/png") if body is not None else self.json(404, {"error": "此帧已不在本次缓存中。"})
        if path == "/api/training-preview.png":
            preview = CHECKPOINTS / "previews" / "latest.png"
            return self.send_bytes(200, preview.read_bytes(), "image/png") if preview.exists() else self.json(404, {"error": "训练预览尚未保存。"})
        if path.startswith("/api/images/"):
            image_id = path.removeprefix("/api/images/").removesuffix(".png")
            with self.server.studio.cache_lock:
                body = self.server.studio.images.get(image_id)
            if body is None:
                return self.json(404, {"error": "图片已不在本次缓存中。"})
            return self.send_bytes(200, body, "image/png", f"cemivo-{image_id}.png")
        static = {"/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
            "/train": ("train.html", "text/html; charset=utf-8"),
            "/train.html": ("train.html", "text/html; charset=utf-8"),
            "/train.js": ("train.js", "text/javascript; charset=utf-8"),
            "/train.css": ("train.css", "text/css; charset=utf-8"),
            "/live": ("live.html", "text/html; charset=utf-8"),
            "/live.html": ("live.html", "text/html; charset=utf-8"),
            "/live.js": ("live.js", "text/javascript; charset=utf-8"),
            "/live.css": ("live.css", "text/css; charset=utf-8"),
            '/screen':('screen.html','text/html; charset=utf-8'),
            '/screen.js':('screen.js','text/javascript; charset=utf-8'),
            '/screen.css':('screen.css','text/css; charset=utf-8')}
        if path not in static:
            return self.json(404, {"error": "找不到此页面。"})
        filename, content_type = static[path]
        return self.send_bytes(200, (ROOT / "web" / filename).read_bytes(), content_type)

    def do_POST(self):
        route = urlsplit(self.path).path
        if route not in ("/api/generate", "/api/training/start", "/api/training/pause",
                         "/api/live/config", "/api/live/model", "/api/live/session", "/api/live/desktop", "/api/live/event", "/api/agent/event"):
            return self.json(404, {"error": "未知接口。"})
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
            return self.json(403, {"error": "此接口仅供本机工作台使用。"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16000:
                raise ValueError("请求长度不正确。")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if route == "/api/training/start":
                result = self.server.studio.training_control.start(body)
            elif route == "/api/training/pause":
                result = self.server.studio.training_control.pause()
            elif route == "/api/live/config":
                result = self.server.studio.live.planner.configure(body)
            elif route == "/api/live/model":
                result = self.server.studio.live.update_model(body)
            elif route == "/api/live/session":
                result = self.server.studio.live.create(body)
            elif route == "/api/live/desktop":
                result = self.server.studio.live.desktop(body)
            elif route == "/api/live/event":
                result = self.server.studio.live.event(body)
            elif route == "/api/agent/event":
                result = self.server.studio.agent.event(body)
            else:
                result = self.server.studio.generate(body)
            return self.json(200, result)
        except LiveError as error:
            return self.json(error.status, {"error": str(error), "code": error.code})
        except (ValueError, UnicodeDecodeError) as error:
            return self.json(422, {"error": str(error)})
        except RuntimeError as error:
            return self.json(409, {"error": str(error)})
        except Exception as error:
            traceback.print_exc()
            self.server.studio.last_error = str(error)
            return self.json(500, {"error": "神经模型生成失败，详情已显示在状态面板与本地日志中。"})


def main():
    parser = argparse.ArgumentParser(description="cemivoAI 从零训练图像大模型工作台")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--snapshot", type=Path, help="Immutable inference snapshot under this project's neural directory")
    args = parser.parse_args()
    if args.snapshot is not None:
        args.snapshot = args.snapshot.resolve()
        if (not args.snapshot.is_relative_to((ROOT / "neural").resolve()) or
                not (args.snapshot / "inference.pt").is_file() or
                not (args.snapshot / "snapshot.json").is_file()):
            parser.error("--snapshot must name a prepared project-local inference snapshot")
    service = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    service.studio = Studio(snapshot_directory=args.snapshot)
    service.daemon_threads = True
    print(f"cemivoAI Neural ready: http://127.0.0.1:{args.port}", flush=True)
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.server_close()


if __name__ == "__main__":
    main()
