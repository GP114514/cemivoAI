# CEMIVO — AI as a Software Rendering Engine

[CEMIVO software](https://github.com/GP114514/cemivoAI) · [40-second English film](renders/CEMIVO-AI-Rendering-Engine-EN.mp4) · [JS project ZIP](CEMIVO-AI-Rendering-Engine-JS.zip)

This film focuses on the project's central experiment: **using AI as the rendering layer of software**. The current interface image, interaction and state enter an intent planner. AI chooses a layout or update region. Trained local neural decoders generate UI pixels. Agent results feed the next rendering request.

The idea is expressed directly in the opening: **“AI as a software rendering engine.”** The animation follows the render request, layout planning, neural surface/text decoding, local and global updates, and the feedback loop from Agent execution to interface state.

## What you are seeing

This is an explicitly labelled **principle film with actual retained model PNGs**, not native APP screen recording. The simulated pointer, pixel reveal, wireframe, tensor symbols and data pulses are JS explanatory graphics. Their timing is illustrative. Tensor patterns are not captured activations, and the film makes no inference-speed claim. The Agent scene explains architecture; no new tool execution is claimed in that scene.

The chat before/after PNGs are from the recorded 2.11.4 test. Their exact edit mask is retained; an independent pixel comparison reports **0 changed pixels outside the selected regions**. The Settings PNG and layout example are from the 2.14.0 DSH-layout test. The film does not depict a continuous run across those versions. All source PNGs remain unchanged and hashed in `provenance.json`.

### Current rendering implementation

- DeepSeek/DSH selects intent, elements, text and bounds using whole-frame context.
- Trained local SurfaceHead/TextHead decoders supply UI pixels in the DSH page path.
- Verified same-page interactions can reuse bounded neural edit rules; new or uncertain interactions return to planning.
- The Agent and native allowlisted tools execute tasks separately. Success, failure and pending results feed visual feedback.
- The main 109M PixelFlow model trains separately. **The shown DSH Settings page is auxiliary-decoder output, not main PixelFlow output.**

The host shell, diagrams, pointer and captions use ordinary software rendering. Arbitrary application replacement, perfect general UI generation and instant fresh AI responses remain research goals. This film demonstrates the rendering concept with bounded existing outputs; it does not certify a universal software engine.

## Storyboard

| Time | Focus |
| --- | --- |
| 00–08.4 | AI as a software rendering engine; interface as generated output |
| 08.4–14.3 | The whole image, input and state form a render request |
| 14.3–20.2 | AI layout → learned neural surface/text decoding → pixels |
| 20.2–27 | Local region update, followed by a new-view example |
| 27–33.4 | Agent action → verified state → next rendered interface |
| 33.4–40 | AI as the rendering layer; open-source project link |

## Re-render

Use Node.js, FFmpeg/FFprobe, and `@napi-rs/canvas` 0.1.100. `npm ci` installs the declared Canvas package. Windows Segoe UI/Consolas fonts are registered when available. Set `FFMPEG`, `FFPROBE` or `CEMIVO_CANVAS_MODULE` if using existing installations outside PATH.

```text
node render.cjs --preview
node make-audio.cjs
node render.cjs --principle
node audit.cjs
```

The retained WAV clips and mix allow offline rendering. Regenerating speech uses `python synthesize-voice.py`, `edge-tts==7.3.1` and `srt==3.5.3`. Voice is generic Microsoft `en-US-AndrewNeural`, with no API key or voice cloning. The procedural score is original JS synthesis. Rendering makes no APP requests, runs no Agent tasks, and does not interrupt training.

`motion.cjs` contains the original vector animation. `assets/layout-example.json` retains actual Settings element geometry. `audit.json` checks video metadata, narration timing, original image hashes and outside-region pixel preservation. Encoded frames are retained in `renders/encoded-storyboard.png` for review.

## Real screen recording

The user's native APP recording requirement remains unfinished: Local browser use is disabled in Work settings, and Windows Computer Use previously could not verify the browser URL. No alternative browser or desktop recorder bypassed these restrictions. `recordings/README.md` explains genuine-source requirements. The optional `--film` compositor rejects absent or unreviewed native recording input; it is separate from this principle-film export.

## Share copy

🧠 AI as a software rendering engine.
🖱️ CEMIVO turns screen context and intent into the next interface. AI plans the layout; trained neural decoders draw the pixels.
🎬 40s principle film · open-source research prototype
🔗 https://github.com/GP114514/cemivoAI
#AI #GenerativeUI

No X post was sent for this revision. Source is MIT; external models, speech services and fonts retain their own terms.
