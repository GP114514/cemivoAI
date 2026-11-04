# Generate an interface from a description

Latest: [`2.16.0-stateful-ui`](STATEFUL-UI.md) adds content-preserving layout revisions, nested input-action compatibility and bounded native Send effects, verified with actual generated PNGs and follow-up interactions.

`2.15.1-prompt-ui` connects first-screen creation and explicit interface redesign to the same DSH layout + locally trained neural pixel-decoder path used by DSH navigation.

In the minimal APP, right-click the image, describe the interface, and submit. For example:

> 生成黑白待办界面，标题必须为“待办”，按钮文字必须为“新增”和“完成”。初始显示一个空任务列表区域和一个空输入框。不添加设置、开关或示例任务。

Or:

> Design a new monochrome chat UI titled ORBIT. Exact buttons Send and Reset. One empty message panel and one empty input. Generate this design now.

The full original text remains in application context; it is not replaced by the short drawing caption. The AI can request `navigation.redraw: true` when the user explicitly wants a new design. Ordinary navigation continues to restore the last app state. Home and Back remain pictured neural controls.

## Rendering and checking

The first-screen plan retains `initial_screen`: an exact requested title, required visible labels, and separate counts for explicitly empty inputs and content panels. The layout planner receives this plan plus the full original goal. Missing required literals or empty controls cause a bounded correction; they are not silently substituted. Unknown required glyphs produce an explicit unavailable-character error.

All delivered image pixels still come from trained local SurfaceHead/TextHead decoders. The DSH path does not use the main 109M PixelFlow model for page output. No new pretrained image model, per-app page template, DOM control renderer or font rasterizer was added. Geometry fitting preserves AI-selected controls and exact text. The complete image, including neural navigation, receives the existing structural review before publication; navigation is not drawn twice.

`POST /api/live/session` automatically plans a DSH initial screen when `plan_initial` is omitted. It accepts a full goal up to the existing 2,000-character / 6,000-UTF-8-byte planning limit. Explicit legacy options remain supported. The standalone DeepSeek/PixelFlow path retains its previous behavior.

```json
{
  "prompt": "Create a monochrome notes UI. Exact title Paper, exact buttons New and Save. Show one empty editor panel. No sample text.",
  "seed": 42,
  "steps": 8
}
```

This route only creates an interface. It does not execute a workspace task or authenticate to an external service. Required-label checks concern layout text supplied to the neural decoder, not independent OCR certification or business completion.

## Local input clear

A visually interpreted draft-clear button uses `text_action: clear` with a local effect mask. It must refer to the exact already-tracked input bounds, cannot overlap prior messages, and cannot execute tools. The trained text head generates the blank draft pixels. Exact outside-mask and border checks plus fresh visual verification gate the result. The logical draft is cleared only after accepted drawing. Empty-draft clicks are no-ops; request replay does not repeat the action. This action does not erase conversation history or files.

## Verification on 2026-11-03

The previous first-screen API reproduced the chat-with-switches failure in **17.63 s**, returning `screen_mismatch`. After the change, actual HTTP generation of these five pages passed literal-title/button, empty-state and full-image structural checks:

| Goal | Retained output | Observed total |
| --- | --- | --- |
| NOVA chat, Send + Clear, empty messages/input | [NOVA](assets/prompt-ui/chat.png) | 6.34 s |
| Chinese 待办, 新增 + 完成, empty list/input | [Tasks](assets/prompt-ui/tasks.png) | 5.97 s |
| Paper notes, New + Save, empty editor | [Notes](assets/prompt-ui/notes.png) | 5.70 s |
| SIGN IN, User/Password fields, Login + Cancel | [Login mockup](assets/prompt-ui/login.png) | 8.64 s |
| Calc, empty display, exactly 7 / 8 / + / = | [Calculator mockup](assets/prompt-ui/calc.png) | 6.11 s |

The login/calculator cases establish visual generation only, not real authentication or arithmetic behavior. None of these generation calls executed business tools. These are single-run timings, not general latency guarantees.

Actual right-click-style HTTP events created ORBIT, then redesigned the same app from Clear to Reset. Both preserved the complete original text and generated new frames instead of restoring old cached pixels; observed round trips were 10.78 and 9.07 s.

The final deployed input/Send/clear sequence completed in 3.90 / 6.47 / 4.41 s, with a confirmed subsequent input taking 58 ms. Every modified frame had 0 changed pixels outside its effect regions. The Clear test independently compared all pixels outside the actual draft bounds, retained `Keep message` in the message area, confirmed an already-empty second click did not alter pixels, and checked identical request replay.

**72 Python tests and the Node window-controller suite passed.** Native browser mouse/keyboard acceptance was unavailable because Local browser use was disabled; the actual backend interaction chain and controller fixtures were tested. Training resumed from the saved 245,000-step checkpoint toward 1,000,000 and remained running during APP-only deployments.

## Scope

The current path renders native 256×256 monochrome text, buttons, inputs and panels. It has a finite trained glyph vocabulary and a 10-element body limit. Missing glyphs, overly dense layouts and unsupported controls are reported; arbitrary complete apps, real external services and instant fresh AI responses are not guaranteed. General image-model training continues separately and was not promoted merely because these auxiliary drawing tests passed.
