# CEMIVO — What if software drew itself?

[CEMIVO software repository](https://github.com/GP114514/cemivoAI) · [Video](renders/CEMIVO-Principle-Animation-EN.mp4)

Original 30-second English principle animation, made with JavaScript Canvas. 1080 × 1350, 30 fps, H.264 / AAC. It includes animated pointers and click rings, whole-image perception, intent routing, Agent tools, neural layout/surface/text heads, edit masks, subtitles, a generic neural voice, and an original procedural JS score.

**Status: principle animation complete; requested real-screen promotional film unfinished.**

`renders/CEMIVO-Principle-Animation-EN.mp4` is the animation preview. It is explicitly labelled **PRINCIPLE ANIMATION**. The displayed 256 × 256 model PNGs are retained unchanged with source hashes in `provenance.json`. They are API outputs, not native screen recording. The old chat PNGs are from version 2.11.4; the six-category settings PNG is from 2.14.0. They illustrate the architecture and are not a claim that all current arbitrary prompts produce perfect UI.

The project does **not** claim instantaneous fresh AI decisions. DeepSeek/DSH selects structured layouts and intent; trained local SurfaceHead/TextHead models draw UI pixels, and allowlisted native/Agent tools execute actual actions. The larger random-initialized image model continues training separately. All-software replacement remains a research goal.

## Deliverables

- `renders/CEMIVO-Principle-Animation-EN.mp4` — 30-second supporting animation, not a completed live demo.
- `renders/poster.jpg` — cover.
- `renders/encoded-storyboard.png` — frames extracted from the actual encoded MP4 for visual review.
- `audio/narration-captions.srt` — sentence captions.
- `audio/narration-words.srt` — word timings.
- `motion.cjs` — deterministic vector animation and simulated pointer renderer.
- `screen-film.cjs` — normal-speed recording compositor, waiting cuts, optional pointer overlay. Requires actual input; not live-validated yet.
- `recordings/README.md` — genuine capture requirements and recording input format.

## Re-render

Requires Node.js, FFmpeg/FFprobe and `@napi-rs/canvas` 0.1.100. Install with `npm ci`. Windows Segoe UI and Consolas are registered when available; substitute local fonts on other operating systems. Set `FFMPEG` and `FFPROBE` to explicit binaries if they are not on PATH. `CEMIVO_CANVAS_MODULE` can point to an existing Canvas package when working locally.

```text
node render.cjs --preview
node make-audio.cjs
node render.cjs --principle
node audit.cjs
```

Retained WAVs allow offline re-rendering. To regenerate the generic Microsoft `en-US-AndrewNeural` speech, install `edge-tts==7.3.1` and `srt==3.5.3`, then run `python synthesize-voice.py`. Speech generation is online; it uses no API key or voice cloning. Rendering makes no APP requests and runs no Agent tasks.

After restoring the approved browser channel and recording the actual APP, supply the reviewed `recordings/recording.json` and run `node render.cjs --film`. This export **refuses to substitute the PNG replay for real recording**. See `recordings/README.md`. The current blocker is Local browser use disabled in Work settings; Windows Computer Use also could not verify the current browser URL. Neither restriction was bypassed.

The software/training service was not modified or restarted to produce this animation. No network connection, volume, credential, or system setting was changed.

## Suggested title and X copy for the eventual real-screen cut

**What if software drew its own interface?**

🖱️ Building CEMIVO: software whose interface is driven by AI.
🧠 Click the screen. AI reads the context, the agent acts, and the interface redraws.
👀 What would you turn it into?
🔗 https://github.com/GP114514/cemivoAI
#AI #GenerativeUI

Do not post the preview as a real-screen recording. Nothing was uploaded or posted by this edition.
