# Software With an Entirely AI-Driven Interface

42-second English X edition of the cemivoAI film. 1080×1350 (4:5), 30 fps, H.264/AAC MP4. The first frame shows the actual generated UI; there is no introductory logo sequence. Seven new English neural-voice clips replace the original Windows SAPI narration. Captions are burned into the video for silent playback.

## Publish

- `renders/cemivoAI-X-EN.mp4`: upload this video.
- `X-post.txt`: title, a 250-character English main post, and a longer introduction.
- `audio/narration-captions.srt`: separate sentence subtitles, if needed.
- `renders/poster.jpg`: matching cover.

The MP4 is 42 seconds and under 2 MB. X's current web-upload guidance lists a 140-second / 512-MB limit for accounts without a paid plan: https://help.x.com/en/using-x/x-videos . The film remains a local deliverable; upload and publishing are done separately.

## Evidence

The original localhost API trace and seven native 256×256 model PNGs are retained unchanged from the English recording. The opening replays the source image sequence at normal speed; it does not simulate button output or speed up inference. All six recorded input/send operations passed, with zero changed pixels outside their approved regions. The final Agent receipt is for a real local document, saved and read back to verify its exact content.

The 37–45 ms figure refers to the recorded confirmed same-page short-text operations only. The first input/send requests took 3.6 and 4.6 seconds. General UI quality remains in training; the film identifies the APP as a research prototype. Source UI labels, including Chinese navigation labels, are preserved. Surrounding flow diagrams and edit-mask marks explain the system and are distinct from the source pixels.

This is an API-driven recording, not desktop screen capture. The earlier Computer Use tool stopped because it could not reliably verify the browser URL; no alternate browser/window recorder was used.

## Sound

Voice: generic Microsoft `en-US-AndrewNeural`, −2% rate, generated with `edge-tts` 7.3.1. This was online synthesis, without an API key or voice cloning. All seven original MP3s, 48-kHz stereo WAVs, word timing JSONL and SRT files are retained. All narration clips are checked for overlap and truncation before mixing. The original JavaScript score ducks under narration. Final encoded audio measures −15.73 LUFS integrated and −1.20 dBTP true peak.

## Re-render

Use Node.js, FFmpeg and `@napi-rs/canvas` 0.1.100. Run `npm install` in this directory to install the declared Canvas dependency. Windows Segoe UI and Consolas fonts are registered by `render.cjs`; replace their paths with suitable installed fonts when rendering on another platform.

1. `node make-audio.cjs` mixes the retained voice WAVs and original procedural music.
2. `node render.cjs --preview` renders the layout contact sheet.
3. `node render.cjs` renders the MP4.
4. `node audit-film.cjs` verifies source hashes, the recorded results, audio timing and actual encoded video, then extracts the encoded keyframes for visual review.

The retained audio files let you re-render without calling a speech service. `trace.json` remains the original recording evidence; the renderer loads this edition's subtitle cues from `audio/cues.json` in memory. Re-rendering makes no APP requests and executes no Agent tasks.

The theme revision states the central idea directly: software whose entire interface is driven by AI. Its opening headline, first and last narration clips, cover and X copy are updated together. To regenerate speech, use Python with edge-tts==7.3.1 and srt==3.5.3, then run synthesize-voice.py; ordinary re-rendering uses the retained WAVs without a speech-service call.
