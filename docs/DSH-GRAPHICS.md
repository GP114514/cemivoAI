# DSH graphical interaction backend

The `2.12.1-dsh-experimental` source adds a DeepSeek Harness backend. cemivoAI is the image-based interaction surface; DSH interprets the full frame, executes workspace tasks, and supplies authoritative tool receipts. The local cemivoAI models still produce the image pixels. This integration does not replace the DSH desktop application or copy its credential store.

## Setup

1. Install and configure DeepSeek Harness. The integration was tested against its local SDK version `0.2.1-rc.3`, using the `deepseek-official / deepseek-flash` route.
2. Set `CEMIVO_DSH_EXE` to the full path of the installed `DeepSeek Harness.exe` if it differs from the development installation at `E:/DSH/DeepSeek Harness.exe`.
3. Install the cemivoAI runtime and trusted inference weights as described in SETUP.md.
4. Start a fresh cemivoAI APP service after updating its source. In the small window, select **DSH · 图形交互**, then **连接 DSH 并进入**. A DeepSeek key does not need to be copied into cemivoAI for this mode. Standalone DeepSeek API mode remains available separately.

On Windows, `Restart DSH Graphics.cmd` is a manual restart entry point for an existing local installation. It checks port ownership, refuses a busy APP or an unsettled DSH turn, preserves the training process, starts the small window and console, and verifies the new graphics protocol and DSH connection. It refuses to stop an unrelated service using port 7861. It is not an installer and requires the configured runtime and model files.

The bridge creates its own `cemivo-graphics` execution profile and `cemivo-graphics-planner` planning profile from DSH's standard SDK template. It does not edit the existing Desktop profile. The planning profile applies the checked-in `dsh-planner.patch.json` to disable tool producers. An actual model-request capability handshake must report an empty tool roster before configuration succeeds. The execution profile keeps the normal DSH workspace sandbox and approval policy. Missing approval support fails closed; the bridge does not grant unrestricted filesystem access.

Default task workspace: `agent-data/dsh-workspace` under the cemivoAI project. Model weights, training source and the DSH credential store are outside this workspace. Do not commit the runtime-created workspace or DSH session logs.

## Execution and rendering

1. Screenshot, click coordinates and typed input enter DSH's **no-tool planner**.
2. UI navigation and local edits use the existing neural drawing/verification path.
3. Actual workspace tasks enter a separate DSH SDK execution session. Tool call/result events are journaled as they arrive; a prompt enqueue receipt is never treated as task completion.
4. DSH selects a pixel-verified blank strip for an execution-status summary. The trained neural text head draws that summary, preserving all pixels outside the selected region. The reserved strip can be reused for later results, with a visible sequence counter.
5. The APP keeps its durable request deduplication. A repeated request returns its recorded result, rather than executing DSH tools again. A timed-out request is never silently resent. DSH session identity is retained for diagnosis of interrupted work.

If an admitted execution exceeds the foreground wait, the APP returns `awaiting_result`, keeps the original frame, blocks further execution prompts, and continues observing the **same** SDK session. Late tool receipts update the original journal entry. The window polls that request rather than treating the click as failed and sending another. If the turn later finishes, the actual answer and receipts become available; its status remains `render_failed` because no background screen redraw was performed. The user can inspect the result and continue. If the APP restarts and loses its observer, the request is reported as `interrupted`, with its DSH session identity retained, rather than polled forever or silently re-executed. An uncertain admission acknowledgement or an observation that cannot confirm quiescence remains blocked for manual DSH inspection. The SDK provides no per-prompt cancel operation.

Execution instructions direct text-file tasks to native glob/read/write/edit tools, with an existence check and read-before-overwrite. They prohibit permission repair, escalation attempts and alternate access routes after native access denial. These are model instructions; the DSH sandbox remains the enforcement boundary. Mixed failed or pending tool receipts are reported as partial failure rather than blanket success.

### Coherent page drawing in DSH mode

New whole pages in DSH mode use a structured layout planned by DSH and the project's existing random-trained neural border and glyph decoders. DSH chooses every element, label and bound from the goal; there is no per-application page template or button routing map. Geometry, spacing, trained-character coverage and complete text fit are checked before drawing. The top 32 rows remain reserved for the existing neural Home/Back controls. A full-image structural review still checks the delivered page. Click interpretation still uses the actual image and event.

This path supports disjoint text, button, input and panel elements. All canvas, border and glyph pixel values come from the trained auxiliary neural decoders; inference uses no font files, stored control images, HTML, SVG or raster drawing commands. It is **not an output of the main 109M PixelFlow model**, and does not add a newly trained model in this update. Frames explicitly report `generator: dsh_layout_neural_surface` and `neural_layout.main_pixel_flow_used: false`. The main randomly initialized image model continues training independently to the configured target. Standalone API mode retains its existing PixelFlow page path. The selected PixelFlow sampling-step count is not a latency guarantee for this auxiliary single-pass drawing mode.

The preceding direct PixelFlow navigation test correctly identified a Chat click but failed its screen guard because the generated page contained switches, even after one redraw. The new DSH layout path passed the same desktop-to-Chat component test in 16.44 seconds with an empty conversation panel, empty input and Send control; its pixel decoding measured 16.88 ms, while DSH planning and review dominated the total. These are single measured component runs, not a real-time response or arbitrary-application guarantee.

For local input/send, the planner receives the actual rendered element bounds as geometry evidence alongside the full screenshot. Text must fit the corresponding input/panel interior with 4px padding. A pixel comparison rejects any edit that changes a rendered control's outline. This addresses a discovered case where a visually accepted draft erased the input's bottom border.

The corrected actual chat flow displayed `Hello CEMIVO` in the input, then moved it into the message panel and cleared the input, with zero changed pixels outside the chosen regions. Its first input and Send measured 15.31 and 27.52 seconds respectively: these first-time provider round trips are still slow. The existing confirmed-interaction cache can now retain the DSH-generated panel as the message area after an actual full-image Send verification. Reusing those real verified artifacts in a component run with network calls forbidden completed a second literal input in 95.31 ms and Send in 32.56 ms. All outlines and outside pixels remained unchanged. Explicit file/task commands are excluded from the short-literal cache and still require fresh AI task interpretation. Cached timings do not apply to new pages, unconfirmed interactions, arbitrary text or business tasks.

A neural label such as “已执行 1” summarizes actual completed tool receipts; it is not a claim that an external service or every user goal has been fulfilled. Failed or pending receipts use a different label. A response without tool receipts is not accepted as business-task success. Detailed answers and tool results remain in the console and journal.

The bridge retains visible assistant text, bounded tool facts, terminal state and capability names. It strips reasoning blocks, system messages, raw request headers and adapter configuration from its retained/public notification stream. DSH itself retains sessions according to its own settings.

## Verification performed

- Real SDK handshake and a model-backed response succeeded.
- Full-image identification returned the six actual desktop button labels.
- A simulated click on the pictured Chat control opened a neural chat page without business tools. The measured component round trip was 11.68 seconds.
- Real DSH tools created a test file and read it back; a separate filesystem check verified the exact content.
- Component end-to-end tests reached both business and rendering `complete`, with actual neural result pixels and **zero changed pixels outside the chosen result strip**. Completed runs measured 18.53, 21.94 and 34.26 seconds across development iterations. These are bounded test measurements, not a general latency guarantee.
- The updated text-file instruction completed a real existing-file write/read-back and neural feedback round trip in 17.02 seconds, using only native glob/read/write/read tools. A preceding run took 54.30 seconds and was correctly marked partial failure: shell permission/ACL attempts were denied before the native file work eventually succeeded. The instruction was tightened after that run; no ACL repair or sandbox escalation was performed by the integration.
- Request replay returned the same image without repeating execution.
- Neural receipt tests cover reserved-strip reuse, visible counter changes, failure labels, missing receipts and invalid regions. Controller tests cover DSH/API selection, setup gating, failure retention and absence of API-key browser storage.
- Deterministic transport/journal tests cover timeout ownership, late receipts, repeated-click rejection, uncertain admission, foreign-session events, refusal to reconfigure an unsettled turn, and recovery after losing an observer. The window-controller test covers awaiting-result polling without a second execution request. These timeout cases are simulated SDK responses, not a claim of provider-wide fault tolerance.

Early mixed planning/execution attempts timed out or hit the model token cap. These attempts were recorded as failures and did not establish successful rendering. Separating the no-tool planner, executor and receipt placement addresses that cause; arbitrary application quality and rare-provider failure handling remain experimental.

**Deployment status in the development session:** updated components were tested directly without launching another HTTP service. Automatic approval blocked both restarting the existing APP service and starting a parallel test service. The already-running old service was not bypassed or hot-patched. A fresh APP process is required to activate these source changes; browser/native-window end-to-end acceptance of that deployed version remains pending.
