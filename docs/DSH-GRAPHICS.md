# DSH graphical interaction backend

The `2.12.0-dsh-experimental` source adds a DeepSeek Harness backend. cemivoAI is the image-based interaction surface; DSH interprets the full frame, executes workspace tasks, and supplies authoritative tool receipts. The local cemivoAI models still produce the image pixels. This integration does not replace the DSH desktop application or copy its credential store.

## Setup

1. Install and configure DeepSeek Harness. The integration was tested against its local SDK version `0.2.1-rc.3`, using the `deepseek-official / deepseek-flash` route.
2. Set `CEMIVO_DSH_EXE` to the full path of the installed `DeepSeek Harness.exe` if it differs from the development installation at `E:/DSH/DeepSeek Harness.exe`.
3. Install the cemivoAI runtime and trusted inference weights as described in SETUP.md.
4. Start a fresh cemivoAI APP service after updating its source. In the small window, select **DSH · 图形交互**, then **连接 DSH 并进入**. A DeepSeek key does not need to be copied into cemivoAI for this mode. Standalone DeepSeek API mode remains available separately.

The bridge creates its own `cemivo-graphics` execution profile and `cemivo-graphics-planner` planning profile from DSH's standard SDK template. It does not edit the existing Desktop profile. The planning profile applies the checked-in `dsh-planner.patch.json` to disable tool producers. An actual model-request capability handshake must report an empty tool roster before configuration succeeds. The execution profile keeps the normal DSH workspace sandbox and approval policy. Missing approval support fails closed; the bridge does not grant unrestricted filesystem access.

Default task workspace: `agent-data/dsh-workspace` under the cemivoAI project. Model weights, training source and the DSH credential store are outside this workspace. Do not commit the runtime-created workspace or DSH session logs.

## Execution and rendering

1. Screenshot, click coordinates and typed input enter DSH's **no-tool planner**.
2. UI navigation and local edits use the existing neural drawing/verification path.
3. Actual workspace tasks enter a separate DSH SDK execution session. Tool call/result events are journaled as they arrive; a prompt enqueue receipt is never treated as task completion.
4. DSH selects a pixel-verified blank strip for an execution-status summary. The trained neural text head draws that summary, preserving all pixels outside the selected region. The reserved strip can be reused for later results, with a visible sequence counter.
5. The APP keeps its durable request deduplication. A repeated request returns its recorded result, rather than executing DSH tools again. A timed-out request is never silently resent. DSH session identity is retained for diagnosis of interrupted work.

A neural label such as “已执行 1” summarizes actual completed tool receipts; it is not a claim that an external service or every user goal has been fulfilled. Failed or pending receipts use a different label. A response without tool receipts is not accepted as business-task success. Detailed answers and tool results remain in the console and journal.

The bridge retains visible assistant text, bounded tool facts, terminal state and capability names. It strips reasoning blocks, system messages, raw request headers and adapter configuration from its retained/public notification stream. DSH itself retains sessions according to its own settings.

## Verification performed

- Real SDK handshake and a model-backed response succeeded.
- Full-image identification returned the six actual desktop button labels.
- A simulated click on the pictured Chat control opened a neural chat page without business tools. The measured component round trip was 11.68 seconds.
- Real DSH tools created a test file and read it back; a separate filesystem check verified the exact content.
- Component end-to-end tests reached both business and rendering `complete`, with actual neural result pixels and **zero changed pixels outside the chosen result strip**. Completed runs measured 18.53, 21.94 and 34.26 seconds across development iterations. These are bounded test measurements, not a general latency guarantee.
- Request replay returned the same image without repeating execution.
- Neural receipt tests cover reserved-strip reuse, visible counter changes, failure labels, missing receipts and invalid regions. Controller tests cover DSH/API selection, setup gating, failure retention and absence of API-key browser storage.

Early mixed planning/execution attempts timed out or hit the model token cap. These attempts were recorded as failures and did not establish successful rendering. Separating the no-tool planner, executor and receipt placement addresses that cause; arbitrary application quality and rare-provider failure handling remain experimental.

**Deployment status in the development session:** updated components were tested directly without launching another HTTP service. Automatic approval blocked both restarting the existing APP service and starting a parallel test service. The already-running old service was not bypassed or hot-patched. A fresh APP process is required to activate these source changes; browser/native-window end-to-end acceptance of that deployed version remains pending.
