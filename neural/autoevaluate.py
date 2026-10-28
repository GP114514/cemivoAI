"""Wait without torch/GPU use, then run the existing independent evaluator.

Run explicitly, from the project root:
  .venv\\Scripts\\python.exe -m neural.autoevaluate --target-step 20000

Waiting reads training/model-card JSON and queries the recorded PID. It never
loads weights or holds the checkpoint training.lock. A separate work-directory
OS lock prevents duplicate watchers. The eventual neural.evaluate subprocess
acquires the existing checkpoint model_access lock; a known busy-lock response
is retried, never bypassed. No training, distillation or external notification
is started. --self-test uses only temporary CPU mocks, not a real evaluator.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
BUSY_MESSAGE = "模型正在训练或由其他请求使用，请稍后重试。"
SAMPLING_STEPS = (8, 16, 32)
SAMPLES, REPEATS = 15, 4


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(.05)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def integer(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def same_path(first, second):
    return os.path.normcase(str(Path(first).resolve())) == os.path.normcase(str(Path(second).resolve()))


def local_path(value, project):
    """Accept paths, not URLs, UNC shares, control characters or shell fragments."""
    text = os.fspath(value)
    if not text or any(character in text for character in ("\x00", "\n", "\r", ";", "|", "&", "<", ">", "`")) or "$(" in text:
        raise ValueError("A path must be a plain local filesystem path, without shell fragments")
    if text.startswith(("\\\\", "//")) or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        raise ValueError("Network/URL paths are not accepted")
    path = Path(text)
    if not path.is_absolute():
        path = Path(project) / path
    return path.resolve()


def process_is_running(pid):
    """Read-only PID check. Never uses os.kill on Windows."""
    if integer(pid) is None or pid <= 0:
        raise ValueError("Training status has no valid PID; process exit cannot be established")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            code = ctypes.get_last_error()
            if code in (87, 1168):  # Invalid parameter / not found: exited PID.
                return False
            # Access denied and unknown errors are conservative: do not sample.
            return True
        try:
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    proc = Path("/proc") / str(pid) / "stat"
    try:
        if proc.exists() and proc.read_text().rsplit(")", 1)[-1].strip().split()[0] == "Z":
            return False
        os.kill(pid, 0)  # POSIX only: signal 0 checks existence without signaling.
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@dataclass
class WatchConfig:
    project: Path
    checkpoints: Path
    work: Path
    output_root: Path
    allowed_data_roots: tuple[Path, ...]
    target_step: int = 20000
    max_wait_seconds: float = 4 * 3600
    poll_seconds: float = 5
    evaluation_timeout_seconds: float = 3600
    device: str = "cuda"

    @property
    def state_path(self):
        return self.work / "autoevaluation.json"

    @property
    def guard_directory(self):
        # The guard stays project-wide even when --work changes the JSON/log
        # location. Two watchers cannot evade deduplication with different work.
        return self.project / "work" / "autoeval-guard"

    @property
    def checkpoint(self):
        return self.checkpoints / "inference.pt"


def validate_config(config):
    for value in (config.project, config.checkpoints, config.work, config.output_root, *config.allowed_data_roots):
        local_path(value, config.project)
    if within(config.work, config.checkpoints) or within(config.checkpoints, config.work):
        raise ValueError("Watcher work/guard directory must be separate from the checkpoint directory")
    if within(config.guard_directory, config.checkpoints) or within(config.checkpoints, config.guard_directory):
        raise ValueError("The independent watch guard must not overlap the checkpoint directory")
    if integer(config.target_step) is None or config.target_step < 1:
        raise ValueError("target-step must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in (config.max_wait_seconds, config.poll_seconds, config.evaluation_timeout_seconds)):
        raise ValueError("Wait, poll and evaluation timeout values must be finite and positive")
    if config.device not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")


def data_path_from_status(status, config):
    value = status.get("data_path")
    if not isinstance(value, str) or not value:
        raise ValueError("Terminal training status has no data_path; no dataset fallback is permitted")
    path = local_path(value, config.project)
    roots = (config.project, *config.allowed_data_roots)
    if not any(within(path, root) for root in roots):
        raise ValueError("status.data_path is outside the project; explicitly add --allow-data-root for that local dataset")
    if not path.is_dir():
        raise FileNotFoundError(f"Training dataset directory is missing: {path}")
    read_json(path / "manifest.json")
    return path


def card_signature(card):
    return hashlib.sha256(json.dumps(card, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def build_command(config, data, stage):
    # Every path is a separate argument. No command-string interpolation, no shell.
    return [sys.executable, "-m", "neural.evaluate", "--checkpoint", str(config.checkpoint),
            "--data", str(data), "--samples", str(SAMPLES), "--steps", ",".join(map(str, SAMPLING_STEPS)),
            "--repeats", str(REPEATS), "--device", config.device, "--output", str(stage)]


def run_evaluator(command, *, cwd, timeout):
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    return subprocess.run(list(command), cwd=str(cwd), env=environment, shell=False,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def validate_report(report_path, config, data):
    report_path = Path(report_path)
    report = read_json(report_path)
    actual = integer(report.get("checkpoint_step"))
    if actual is None or actual < config.target_step:
        raise ValueError("Evaluation report checkpoint step is missing or below the requested target")
    if not isinstance(report.get("checkpoint"), str) or not same_path(report["checkpoint"], config.checkpoint):
        raise ValueError("Evaluation report refers to a different checkpoint")
    if not isinstance(report.get("validation_dataset"), str) or not same_path(report["validation_dataset"], data):
        raise ValueError("Evaluation report used a different dataset from terminal status.data_path")
    if report.get("device") != config.device:
        raise ValueError("Evaluation report used a different device from the requested evaluator")
    if report.get("grid_rows_sampling_steps") != list(SAMPLING_STEPS):
        raise ValueError("Evaluation report has unexpected sampling step counts")
    available = integer(report.get("validation_examples_available"))
    ablation = report.get("caption_ablation", {})
    if available is None or available < 2 or ablation.get("examples") != min(SAMPLES, available):
        raise ValueError("Evaluation report has an incomplete caption-control sample count")
    latencies = report.get("warm_latency", [])
    if len(latencies) != len(SAMPLING_STEPS) or sorted(row.get("sampling_steps", -1) for row in latencies) != list(SAMPLING_STEPS):
        raise ValueError("Evaluation report lacks one or more latency configurations")
    for row in latencies:
        if row.get("warm_neural_generation", {}).get("samples") != REPEATS or row.get("warm_generate_to_pil", {}).get("samples") != REPEATS:
            raise ValueError("Evaluation report lacks the requested latency repeats")
    names = [report.get("grid")]
    images = report.get("images", [])
    columns = report.get("grid_columns", [])
    if not columns or len(images) != len(columns) * len(SAMPLING_STEPS):
        raise ValueError("Evaluation report has an incomplete generated image grid")
    names.extend(row.get("image") for row in images)
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or any(character in name for character in ("/", "\\")) or not name.endswith(".png"):
            raise ValueError("Evaluation report has an unsafe or missing PNG filename")
        if not (report_path.parent / name).is_file():
            raise FileNotFoundError(f"Evaluation PNG is missing: {name}")
    return report


def watch(config, *, clock=time.monotonic, sleeper=time.sleep, runner=run_evaluator,
          process_probe=process_is_running, timestamp=utc_now):
    """Inject clock/sleeper/runner/probe for fast mocks; no torch imports here."""
    validate_config(config)
    start, deadline = clock(), clock() + config.max_wait_seconds
    try:
        previous_state = read_json(config.state_path)
    except (OSError, ValueError):
        previous_state = {}
    state = {"version": 1, "state": "waiting", "watcher_pid": os.getpid(),
             "target_step": config.target_step, "actual_step": None, "report_path": None,
             "checkpoint_directory": str(config.checkpoints), "checkpoint": str(config.checkpoint),
             "guard_directory": str(config.guard_directory),
             "started_at": timestamp(), "updated_at": timestamp(), "errors": [],
             "evaluation_attempts": 0, "busy_retries": 0, "max_wait_seconds": config.max_wait_seconds,
             "device": config.device,
             "sampling_steps": list(SAMPLING_STEPS), "samples": SAMPLES, "repeats": REPEATS,
             "weights_modified": False, "notifications_sent": False}

    def write(status="waiting", **extra):
        state.update(extra)
        state["state"], state["updated_at"] = status, timestamp()
        state["elapsed_seconds"] = round(max(0, clock() - start), 3)
        atomic_json(config.state_path, state)
        return state

    def error(kind, message):
        record = {"type": kind, "message": str(message)[-8000:], "updated_at": timestamp()}
        if not state["errors"] or state["errors"][-1]["type"] != kind or state["errors"][-1]["message"] != record["message"]:
            state["errors"].append(record)
            state["errors"] = state["errors"][-50:]

    def wait(reason):
        write("waiting", wait_reason=reason)
        remaining = deadline - clock()
        if remaining > 0:
            sleeper(min(config.poll_seconds, remaining))

    write(wait_reason="Waiting for training status JSON")
    while True:
        if clock() >= deadline:
            error("expired", "Maximum waiting time elapsed before a complete evaluation could start")
            return write("expired", wait_reason="Maximum waiting time reached")
        try:
            training = read_json(config.checkpoints / "status.json")
        except (OSError, ValueError) as exc:
            error("status-read", exc)
            wait("Training status JSON is not available yet")
            continue
        phase, training_step = training.get("state"), integer(training.get("step"))
        state.update(training_state=phase, training_step=training_step, training_pid=training.get("pid"))
        if phase == "failed":
            error("training-failed", training.get("error") or "Training reported failed without an error message")
            return write("failed", failure_reason="Training failed; evaluation was not started")
        if phase not in ("complete", "paused"):
            wait("Training is not in complete/paused state")
            continue
        if training_step is None or training_step < config.target_step:
            wait("Terminal training step is still below the target; waiting for a later resume")
            continue
        try:
            if integer(training.get("pid")) is None or training["pid"] <= 0:
                raise ValueError("Training PID is missing; cannot establish that the process exited")
            running = process_probe(training["pid"])
            if not isinstance(running, bool):
                raise ValueError("Process exit query returned an unknown result")
        except Exception as exc:
            error("process-query", exc)
            wait("Cannot confirm that the training process exited")
            continue
        if running:
            wait("Training PID is still running, even though terminal status was written")
            continue
        try:
            card = read_json(config.checkpoints / "model_card.json")
            checkpoint_step = integer(card.get("checkpoint_step"))
        except (OSError, ValueError) as exc:
            error("model-card-read", exc)
            wait("Waiting for committed checkpoint model-card JSON")
            continue
        state["actual_step"] = checkpoint_step
        terminal_checkpoint_step = integer(training.get("checkpoint_step"))
        terminal_checkpoint_step = training_step if terminal_checkpoint_step is None else terminal_checkpoint_step
        if checkpoint_step is None or checkpoint_step < config.target_step or checkpoint_step != terminal_checkpoint_step:
            wait("Checkpoint model-card step is below target or not yet consistent with terminal training status")
            continue
        try:
            data = data_path_from_status(training, config)
            state["data_path"] = str(data)
            signature = card_signature(card)
            destination = config.output_root / f"evaluation-step-{checkpoint_step}"
            if within(destination, config.checkpoints) or within(config.checkpoints, destination):
                raise ValueError("Evaluation output would overlap the checkpoint directory")
            # Reuse only our own previously verified report for this exact
            # unchanged JSON model-card, dataset, checkpoint and evaluation spec.
            if destination.exists():
                report_path = destination / "evaluation.json"
                if previous_state.get("state") == "complete" and previous_state.get("source_card_signature") == signature and previous_state.get("actual_step") == checkpoint_step and previous_state.get("report_path") == str(report_path) and previous_state.get("sampling_steps") == list(SAMPLING_STEPS) and previous_state.get("samples") == SAMPLES and previous_state.get("repeats") == REPEATS and previous_state.get("device") == config.device:
                    report = validate_report(report_path, config, data)
                    return write("complete", actual_step=report["checkpoint_step"], report_path=str(report_path),
                                 output_directory=str(destination), reused_existing_report=True,
                                 source_card_signature=signature, wait_reason=None)
                raise FileExistsError(f"Preserving an existing evaluation directory; choose a different --output-root: {destination}")
            stage = config.output_root / f".autoevaluation-stage-{os.getpid()}-{uuid.uuid4().hex}"
            if within(stage, config.checkpoints) or within(config.checkpoints, stage):
                raise ValueError("Staging output would overlap checkpoint directory")
            command = build_command(config, data, stage)
            state["evaluation_attempts"] += 1
            write("evaluating", command=command, staging_directory=str(stage), report_path=str(stage / "evaluation.json"),
                  source_card_signature=signature, wait_reason=None)
            result = runner(command, cwd=config.project, timeout=config.evaluation_timeout_seconds)
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            if result.returncode != 0:
                if BUSY_MESSAGE in output:
                    state["busy_retries"] += 1
                    error("model-busy", output)
                    state["report_path"] = None
                    wait("Evaluator respected model_access; model is busy, retrying after the next poll")
                    continue
                error("evaluation-process", f"Exit code {result.returncode}: {output}")
                return write("failed", failure_reason="Independent evaluator failed", exit_code=result.returncode)
            report = validate_report(stage / "evaluation.json", config, data)
            actual_step = report["checkpoint_step"]  # The evaluator's loaded weights are authoritative.
            destination = config.output_root / f"evaluation-step-{actual_step}"
            if destination.exists():
                raise FileExistsError(f"Actual checkpoint output already exists; staging results preserved: {destination}")
            # Stage is on the same output-root volume. Publishing does not copy,
            # touch or rename a checkpoint, dataset, training config or weights.
            stage.rename(destination)
            return write("complete", actual_step=actual_step, report_path=str(destination / "evaluation.json"),
                         output_directory=str(destination), staging_directory=None,
                         source_card_signature=signature, wait_reason=None,
                         report_summary={"checkpoint_step": actual_step,
                           "caption_mean_velocity_mse": report["caption_ablation"].get("mean_velocity_mse"),
                           "sampling_steps": report["grid_rows_sampling_steps"]})
        except (Exception, KeyboardInterrupt) as exc:
            error(type(exc).__name__, exc)
            return write("failed", failure_reason="Evaluation workflow stopped; source weights were not changed")


def run_guarded(config, **dependencies):
    # This import is deliberately the torch-free locking module, not train.py.
    from .locking import model_access
    validate_config(config)
    with model_access(config.guard_directory):
        return watch(config, **dependencies)


def self_test(work, report_path):
    """Fast deterministic state-machine mocks; no real subprocess or weights."""
    from types import SimpleNamespace
    from concurrent.futures import ThreadPoolExecutor
    from .locking import model_access
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cases = []
    if "torch" in sys.modules:
        raise AssertionError("self-test must run in a fresh process without torch")

    class Clock:
        def __init__(self):
            self.value, self.on_sleep = 0.0, None
        def now(self):
            return self.value
        def sleep(self, seconds):
            self.value += seconds
            if self.on_sleep:
                self.on_sleep(self.value)

    with tempfile.TemporaryDirectory(prefix="autoeval-mock-", dir=work) as temporary:
        base = Path(temporary)

        def fixture(name, step=20000, phase="complete", max_wait=5):
            project = base / name
            checkpoint = project / "neural" / "checkpoints"
            data = project / "neural" / "data-v2"
            checkpoint.mkdir(parents=True)
            data.mkdir()
            atomic_json(data / "manifest.json", {"count": 100})
            config = WatchConfig(project, checkpoint, project / "work", project / "neural", (),
                                 max_wait_seconds=max_wait, poll_seconds=1, device="cpu")
            training = {"state": phase, "step": step, "checkpoint_step": step, "pid": 123,
                        "data_path": str(data)}
            atomic_json(checkpoint / "status.json", training)
            atomic_json(checkpoint / "model_card.json", {"checkpoint_step": step, "updated_at": "mock"})
            return config, training, data

        def mock_runner(config, data, actual=None, busy_first=False, fail=False, calls=None):
            calls = [] if calls is None else calls
            def run(command, *, cwd, timeout):
                calls.append(list(command))
                assert isinstance(command, list) and cwd == config.project and timeout > 0
                assert command[command.index("--data") + 1] == str(data)
                assert command[command.index("--samples") + 1] == "15"
                assert command[command.index("--steps") + 1] == "8,16,32"
                assert command[command.index("--repeats") + 1] == "4"
                if busy_first and len(calls) == 1:
                    return SimpleNamespace(returncode=1, stdout="", stderr="RuntimeError: " + BUSY_MESSAGE)
                if fail:
                    return SimpleNamespace(returncode=2, stdout="", stderr="mock evaluator failure")
                output = Path(command[command.index("--output") + 1])
                output.mkdir(parents=True)
                step = actual if actual is not None else read_json(config.checkpoints / "model_card.json")["checkpoint_step"]
                rows = [{"sampling_steps": count, "warm_neural_generation": {"samples": 4},
                         "warm_generate_to_pil": {"samples": 4}} for count in SAMPLING_STEPS]
                images = []
                for index in range(12):
                    name = f"mock-{index}.png"
                    (output / name).write_bytes(b"CPU MOCK ONLY; not a neural image")
                    images.append({"image": name})
                (output / "grid.png").write_bytes(b"CPU MOCK ONLY")
                atomic_json(output / "evaluation.json", {"mock": True, "checkpoint_step": step,
                            "checkpoint": str(config.checkpoint), "validation_dataset": str(data),
                            "device": config.device,
                            "validation_examples_available": 100, "caption_ablation": {"examples": 15},
                            "grid_rows_sampling_steps": list(SAMPLING_STEPS), "warm_latency": rows,
                            "grid_columns": [{"name": str(i)} for i in range(4)], "images": images, "grid": "grid.png"})
                return SimpleNamespace(returncode=0, stdout="mock complete", stderr="")
            return run

        config, training, data = fixture("wait-and-resume", step=19999, phase="paused")
        clock, calls, alive = Clock(), [], {"value": True}
        def advance(value):
            training.update(step=20010, checkpoint_step=20010, state="running" if value < 2 else "complete")
            alive["value"] = value < 3
            atomic_json(config.checkpoints / "status.json", training)
            atomic_json(config.checkpoints / "model_card.json", {"checkpoint_step": 20010})
        clock.on_sleep = advance
        result = run_guarded(config, clock=clock.now, sleeper=clock.sleep,
                             runner=mock_runner(config, data, calls=calls), process_probe=lambda pid: alive["value"])
        assert result["state"] == "complete" and result["actual_step"] == 20010 and clock.value == 3
        assert len(calls) == 1 and Path(result["report_path"]).parent.name == "evaluation-step-20010"
        assert not (config.checkpoints / "training.lock").exists(), "Watcher touched the real training lock"
        again = run_guarded(config, clock=clock.now, sleeper=clock.sleep,
                            runner=lambda *a, **k: (_ for _ in ()).throw(AssertionError("Duplicate evaluation")),
                            process_probe=lambda pid: False)
        assert again["state"] == "complete" and again.get("reused_existing_report") is True
        cases.extend(["paused below target waits for resumed training", "terminal PID must exit", "actual higher checkpoint step", "no training.lock touched", "verified report reuse"])

        config, training, data = fixture("busy-retry")
        clock, calls = Clock(), []
        result = run_guarded(config, clock=clock.now, sleeper=clock.sleep,
                             runner=mock_runner(config, data, busy_first=True, calls=calls), process_probe=lambda pid: False)
        assert result["state"] == "complete" and result["busy_retries"] == 1 and len(calls) == 2
        cases.append("shared model lock busy retries without bypass")

        config, training, data = fixture("checkpoint-race")
        clock = Clock()
        result = run_guarded(config, clock=clock.now, sleeper=clock.sleep,
                             runner=mock_runner(config, data, actual=20020), process_probe=lambda pid: False)
        assert result["actual_step"] == 20020 and Path(result["report_path"]).parent.name == "evaluation-step-20020"
        cases.append("evaluator actual step determines published directory")

        config, training, data = fixture("training-failed", phase="failed")
        training["error"] = "mock training OOM retained"
        atomic_json(config.checkpoints / "status.json", training)
        clock = Clock()
        result = run_guarded(config, clock=clock.now, sleeper=clock.sleep,
                             runner=lambda *a, **k: (_ for _ in ()).throw(AssertionError("Must not evaluate")), process_probe=lambda pid: True)
        assert result["state"] == "failed" and "mock training OOM retained" in result["errors"][-1]["message"]
        cases.append("training failure exits with original cause")

        config, training, data = fixture("expired", step=1, phase="paused", max_wait=2)
        clock = Clock()
        result = run_guarded(config, clock=clock.now, sleeper=clock.sleep,
                             runner=lambda *a, **k: (_ for _ in ()).throw(AssertionError("Must not evaluate")), process_probe=lambda pid: False)
        assert result["state"] == "expired" and clock.value == 2
        cases.append("maximum wait expires without a subprocess")

        config, training, data = fixture("evaluator-failed")
        clock = Clock()
        result = run_guarded(config, clock=clock.now, sleeper=clock.sleep,
                             runner=mock_runner(config, data, fail=True), process_probe=lambda pid: False)
        assert result["state"] == "failed" and "mock evaluator failure" in result["errors"][-1]["message"]
        cases.append("non-busy evaluator failure is preserved")

        config, training, data = fixture("bad-path")
        training["data_path"] = str(data) + "; unwanted-command"
        atomic_json(config.checkpoints / "status.json", training)
        clock = Clock()
        result = run_guarded(config, clock=clock.now, sleeper=clock.sleep,
                             runner=lambda *a, **k: (_ for _ in ()).throw(AssertionError("Must reject path")), process_probe=lambda pid: False)
        assert result["state"] == "failed" and "shell fragments" in result["errors"][-1]["message"]
        cases.append("plain local path validation and list arguments")

        config, training, data = fixture("stale-card", max_wait=2)
        atomic_json(config.checkpoints / "model_card.json", {"checkpoint_step": 19999})
        clock = Clock()
        result = run_guarded(config, clock=clock.now, sleeper=clock.sleep,
                             runner=lambda *a, **k: (_ for _ in ()).throw(AssertionError("Stale checkpoint must not evaluate")), process_probe=lambda pid: False)
        assert result["state"] == "expired" and result["evaluation_attempts"] == 0
        cases.append("stale/below-target model card never triggers evaluation")

        config, training, data = fixture("bad-report")
        clock = Clock()
        result = run_guarded(config, clock=clock.now, sleeper=clock.sleep,
                             runner=mock_runner(config, data, actual=19999), process_probe=lambda pid: False)
        assert result["state"] == "failed" and not (config.output_root / "evaluation-step-19999").exists()
        cases.append("below-target actual evaluation report is not published")

        original_run, captured = subprocess.run, {}
        def record_run(command, **kwargs):
            captured.update(command=command, **kwargs)
            return SimpleNamespace(returncode=0, stdout="CPU mock", stderr="")
        try:
            subprocess.run = record_run
            run_evaluator(build_command(config, data, config.output_root / "mock-stage"), cwd=config.project, timeout=10)
        finally:
            subprocess.run = original_run
        assert isinstance(captured["command"], list) and captured["shell"] is False
        assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
        cases.append("default subprocess runner uses argument list and shell=False")

        config, training, data = fixture("duplicate-watch")
        with model_access(config.guard_directory), ThreadPoolExecutor(max_workers=1) as executor:
            try:
                alternate_work = replace(config, work=config.project / "other-watch-work")
                executor.submit(run_guarded, alternate_work).result()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Duplicate watcher bypassed its independent OS lock")
        cases.append("duplicate watcher independent OS lock rejection even with another work path")

    report = {"passed": True, "test_kind": "CPU state-machine mocks only", "torch_imported": "torch" in sys.modules,
              "real_evaluations_started": 0, "production_checkpoints_read": False,
              "gpu_used": False, "cases": cases, "updated_at": utc_now()}
    assert report["torch_imported"] is False
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default=str(ROOT / "neural" / "checkpoints"))
    parser.add_argument("--target-step", type=int, default=20000)
    parser.add_argument("--max-wait-seconds", type=float, default=4 * 3600,
                        help="Maximum waiting/busy-retry time; a started evaluator has its own timeout")
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--evaluation-timeout-seconds", type=float, default=3600)
    parser.add_argument("--work", default=str(ROOT / "work"))
    parser.add_argument("--output-root", default=str(ROOT / "neural"))
    parser.add_argument("--allow-data-root", action="append", default=[],
                        help="Explicit local dataset root outside this project; may be repeated")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--self-test", action="store_true", help="Run only temporary CPU mocks; never start a real evaluator")
    parser.add_argument("--test-report", help="CPU mock report path; defaults to work/autoevaluation-tests.json")
    args = parser.parse_args()
    try:
        work = local_path(args.work, ROOT)
        if args.self_test:
            report = local_path(args.test_report, ROOT) if args.test_report else work / "autoevaluation-tests.json"
            self_test(work, report)
            return 0
        config = WatchConfig(ROOT, local_path(args.checkpoints, ROOT), work,
                             local_path(args.output_root, ROOT),
                             tuple(local_path(path, ROOT) for path in args.allow_data_root),
                             args.target_step, args.max_wait_seconds, args.poll_seconds,
                             args.evaluation_timeout_seconds, args.device)
        result = run_guarded(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["state"] == "complete" else 2
    except (Exception, KeyboardInterrupt) as exc:
        # In particular, a duplicate watch does not overwrite the active
        # watcher's JSON. An OS-held guard is released automatically on exit.
        print(f"autoevaluate: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
