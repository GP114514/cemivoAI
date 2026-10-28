# Setup, training and model artifacts

Run all commands from the repository root unless a command explicitly changes directory. The source release contains no trained checkpoints, optimizer state, training arrays or local Agent database. The film can be played and re-rendered from its included recording assets without any of those APP artifacts.

## 1. Python environment

Development used Python 3.12, PyTorch 2.9.1 with CUDA 12.8, NumPy and Pillow. The committed requirements select the CUDA build used on the development machine.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use a PyTorch build appropriate to your operating system and GPU when the pinned CUDA wheel is unavailable. For a CPU-only development environment, install a CPU PyTorch wheel plus the NumPy/Pillow versions specified in `requirements-neural.txt`, rather than using its CUDA-specific Torch pin. CPU training requires `--allow-cpu` and will be substantially slower. The training CLI exposes model-width and batch controls; the full default architecture needs significant GPU memory. No minimum VRAM guarantee is established by this source release.

The Python model and server commands are portable, but the packaged PowerShell/CMD launchers and several auxiliary font-training paths target Windows. The main dataset generator searches Microsoft YaHei, Noto Sans CJK and DejaVu fonts in order. Supply appropriate installed fonts if multilingual supervision is needed; missing CJK glyphs change the training targets.

## 2. Generate a main-model training dataset

The dataset builder creates local synthetic training targets. They are supervision, not model-generated quality samples.

```sh
python -m neural.dataset --output neural/data --count 50000 --size 256 --seed 20261025
```

For PowerShell without activating the environment, replace `python` with `.\.venv\Scripts\python.exe`. The output includes `manifest.json`, vocabulary metadata, sharded grayscale NumPy arrays and per-image metadata. The default validation fraction is 0.05. `--count` can be lowered for a pipeline experiment, but a smaller dataset is not equivalent to the trained demo. Additional curriculum generators are present under `neural/`; using the base command above does not reconstruct all historical curriculum stages or their training schedule.

## 3. Train and inspect the main visual model

```sh
python -m neural.train --data neural/data --steps 10000 --resolution 256 --warmup-steps 0 --batch-size 4 --accum 4 --save-every 500 --sample-every 500
```

This starts from random weights and writes under `neural/checkpoints/` by default. `--steps` is the total target, not an additional-step count. The example is a runnable training configuration, not a promise of usable UI quality at 10,000 steps. Inspect genuinely generated images, exact labels and interaction behavior before choosing a checkpoint.

Continue an existing run:

```sh
python -m neural.train --data neural/data --resume neural/checkpoints/latest.pt --steps 20000 --batch-size 4 --accum 4 --warmup-steps 0
```

`latest.pt` contains resumable training state; `inference.pt` contains the published inference weights. Never use the presence of a file, decreasing loss or a step count as proof that prompt following has passed. Do not delete checkpoints while a process is writing them. Only load trusted checkpoints: some main-model checkpoint loading paths use PyTorch deserialization with `weights_only=False`.

Sample the main generator directly:

```sh
python -m neural.sample "Black and white chat UI with Send and Attach buttons" --checkpoint neural/checkpoints/inference.pt --steps 8 --seed 42 --output samples/chat.png
```

The sampling CLI defaults to CUDA. Add `--device cpu` for CPU sampling. Direct prompts are bounded by the byte-level text encoder; long application specifications are handled separately by the interaction planner.

## 4. Prepare an immutable inference snapshot

Once a complete main inference checkpoint exists:

```sh
python -m neural.prepare_live_snapshot
```

This copies `neural/checkpoints/inference.pt` to `neural/live-snapshot/inference.pt`, verifies it and writes `snapshot.json` with its step, native size and SHA-256. It deliberately refuses to overwrite an existing snapshot. It does not certify image quality.

To select a different prepared snapshot at launch, use the server's explicit option:

```sh
python server.py --port 7861 --snapshot neural/live-snapshot
```

Snapshots must reside within this project's `neural/` directory and contain both `inference.pt` and `snapshot.json`. The Python command avoids machine-specific launcher configuration. PowerShell launchers additionally read a local `runtime.json`; do not copy another machine's absolute paths into it.

## 5. Auxiliary checkpoints and interaction

The interaction runtime also uses separately trained neural editors. These files are not created by `neural.train`:

| Expected path | Role |
| --- | --- |
| `neural/text-head-v1/inference.pt` | Finite-vocabulary neural text pixels |
| `neural/toggle-head-experiment-v1/inference.pt` | Local switch edits |
| `neural/surface-head-v1/inference.pt` | Navigation surface and chrome geometry/pixels |

Their loaders validate checkpoint schema, model provenance and declared validation metadata; placing arbitrary tensors in these paths is insufficient. A missing auxiliary checkpoint disables the relevant feature. These narrow heads do not establish that the main generator can draw arbitrary applications.

The text-training helper uses installed Consolas and SimSun paths in `neural/text_training_data.py`; it does not bundle those fonts. Adjust training-only font paths for a different environment, and revalidate the resulting model. Inference uses the learned head, not the font rasterizer.

Separate research recipes are included under `scripts/`:

```sh
python scripts/train_text_head.py --output neural/text-head-v1 --steps 4000 --device cuda
python scripts/train_surface_head.py
```

The text recipe can use `--device cpu`; it refuses an already-existing output directory. The surface recipe currently fixes CUDA, 3,000 steps and output `neural/surface-head-v1`; it requires at least 3 GiB of free GPU memory and can replace files in that directory. Check the source before reusing an existing experiment directory. Both save declared held-out validation results; completion of a loop is not sufficient if validation fails.

`scripts/train_toggle_head.py` is the separate toggle recipe. It requires your own separately generated grayscale PNG containing a measurable **off** switch, plus its native half-open pixel bounds. The image is used only for held-out validation. For example, after locating the switch in your own image:

```sh
python scripts/train_toggle_head.py --output neural/toggle-head-experiment-v1 --steps 1800 --device cuda --validation-image samples/switch-off.png --validation-bounds 211 106 240 121
```

The coordinates above are illustrative; replace them with the actual bounds in your image. The script checks the image and bounds before training begins. Synthetic-only training is not enough to satisfy that checkpoint's declared acceptance. These auxiliary recipes and the base main-model commands do not reconstruct every historical curriculum or guarantee the complete recorded APP behavior.

After preparing matching checkpoints, copy `runtime.example.json` to a local `runtime.json` and adjust its project-relative paths if you want the optional toggle editor or PowerShell launchers. Run from the project root. The Python server defaults to `neural/live-snapshot`; `--snapshot` explicitly selects its main model. The toggle editor is enabled only when `runtime.json` configures `toggle_head`.

## 6. Start the local application

```sh
python server.py --port 7861
```

The service binds to `127.0.0.1`. It can start its workbench before weights exist, but generation and interaction will report missing models until those artifacts are configured.

| Address | Purpose |
| --- | --- |
| `http://127.0.0.1:7861/` | Image sampling workbench |
| `http://127.0.0.1:7861/train` | Training controls and telemetry |
| `http://127.0.0.1:7861/live` | Planner configuration and full interaction console |
| `http://127.0.0.1:7861/screen` | Minimal image interaction window |

Enter your own DeepSeek key in the local configuration UI. The default key setting is process-local; it is not part of this repository. The `DEEPSEEK_API_KEY` environment variable is also supported. Restarting clears a key entered only in the UI. A planner request transmits the current screen and interaction context to the configured external provider. Provider/model capabilities and your account determine whether that planning request is accepted.

In the image surface, left-click supplies the click position; right-click opens text input. Enter submits and Shift+Enter creates a line break. Actual business work is limited to the tools implemented in `agent_tools.py`; generating the appearance of another application does not connect that application's services. Documents and tasks are stored locally in the runtime-created `agent-data/` directory, which must remain out of Git.

## 7. Render the English film

```sh
cd film
npm install
node render.cjs --preview
node render.cjs
node audit-film.cjs
```

Install Node.js and FFmpeg first. The Canvas dependency is `@napi-rs/canvas` 0.1.100. `render.cjs` registers Windows Segoe UI and Consolas; change those installed-font paths to render elsewhere. Watch the included MP4 without installing any development tools.

The retained WAV files let `node make-audio.cjs` rebuild the mix without a speech-service request. Regenerating narration with `synthesize-voice.py` requires Python packages `edge-tts==7.3.1` and `srt==3.5.3`, FFmpeg and network access. It uses a generic online neural voice, not a cloned person. The film replays the saved original model PNGs; rendering it does not start training or call the Agent.
