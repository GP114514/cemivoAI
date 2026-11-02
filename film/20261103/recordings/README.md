# Native recording input — still pending

The original request requires real screen recording. **There is no native recording in this edition yet.** Existing API PNGs are used only in the explicitly labelled principle animation. They are not accepted as the live demonstration.

On 2026-11-03, Windows Computer Use could not confidently determine the browser URL. The dedicated browser inventory also returned `Local browser use is disabled in Work settings`. No alternate desktop/browser capture mechanism was used to bypass either restriction.

Enable Local browser use in Work settings before resuming automated interaction. Alternatively the user can supply an authentic APP screen recording. Do not film API keys, Wi-Fi passwords, unrelated desktop windows, private conversation history, or the DSH credential store. Use a fresh harmless demo such as `One screen. One intent.`. Avoid promising instant fresh AI requests: measured new pages still take seconds.

Suggested actual shots:

1. Real APP desktop, visible pointer enters Chat.
2. Right click the pictured input; type `One screen. One intent.`; submit and click Send. Keep the actual reply visible.
3. Enter Settings → Sound. If changing volume is authorized for that recording, show the actual result and restore the prior value. Read-only Display is an alternative.
4. A workspace Agent task can create a new test document and read it back. Retain genuine tool receipts; do not simulate success.

Copy the reviewed raw recording here as `app-screen.mp4`. Keep the original file unchanged. Create `recording.json` after review:

```json
{
  "file": "recordings/app-screen.mp4",
  "sha256": "SHA256_OF_THE_REAL_UNCHANGED_VIDEO",
  "capture_kind": "native_screen_recording",
  "reviewed_for_authenticity": true,
  "reviewed_for_secrets": true,
  "recorded_at": "ISO_8601_ACTUAL_RECORDING_TIME",
  "source_cursor_visible": true,
  "crop": { "x": 0, "y": 0, "width": 800, "height": 800 },
  "shots": [
    { "start_s": 0, "duration_s": 12, "caption": "Click the image. The running APP responds." }
  ],
  "pointer": []
}
```

The crop must match the real viewport and lie inside the video. Pick 8–30 seconds total of source ranges. Every segment plays at 1×. Waiting cuts are labelled. The 30-second principle animation follows. A simulated pointer can be added using source-time samples `{ "time_s": 2.0, "x": 0.5, "y": 0.8, "click": true }`. Coordinates are normalized to the crop. Review these points against the **actual** input moments; do not invent events or UI outcomes. The film labels this overlay. The original pointer can remain visible instead.

`node render.cjs --film` refuses to export without the real input, reviews and matching source hash. This guard prevents an animation-only export being mistaken for the requested completed film.
