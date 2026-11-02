# Local Windows computer integration

`2.13.0-local-computer` connects the generated APP to the computer on which its
Python service runs. DSH interprets the full image and the user's click/text;
the host executes a narrow Windows operation and returns actual system facts.
The neural border/text models then generate a page from those facts. This is
separate from the main PixelFlow model's training and from DSH's file workspace.

## Use

Connect DSH in the small window, then right-click and enter, for example:

- `显示这台本地电脑的信息：电脑名称、内存、CPU线程和Windows版本。`
- `显示本机真实 Wi-Fi 连接状态和已保存网络，给我刷新和连接按钮。`
- `选择 Network 2，不要连接。`
- `连接 Wi-Fi <the exact saved network name>。`
- `打开 Windows Wi-Fi 设置。` or `打开 Windows 蓝牙设置。`

A generated network-name control selects a saved profile first. **Connect**
acts on that selected identity; selection itself does not change the connection.
**Refresh** reads the state currently reported by Windows. A visible `Result N`
is the actual operation sequence, so an unchanged state still has feedback.
Home/Back remain generated-page navigation and do not undo Windows changes.

Computer information includes the local name, Windows build/architecture, CPU
name/thread count, RAM and system-disk capacity/free space. Wi-Fi information
includes interface state, current SSID, signal/rates, visible networks and saved
profile names. MAC addresses, BSSIDs, Wi-Fi passwords and profile XML/keys are
not exported by this integration. Names that the current text model cannot
reliably render can use an ASCII display alias such as `Network 2`; the actual
name and measured identity remain in the host facts and console.

## Real operations and limits

- `inspect`: read the local PC/Wi-Fi state.
- `wifi_refresh`: retrieve visible networks and saved profiles. This reads the
  list maintained by Windows, rather than certifying a new active RF scan.
- `wifi_select`: change only this page's selected saved-profile identity.
- `wifi_connect`: use one actually observed saved profile with native
  `WlanConnect`, then query the connection for up to 12 seconds. Admission is
  not success. An unconfirmed result is reported as unconfirmed, never silently
  retried. If Windows already reports that target connected, no reconnect is
  submitted.
- `settings`: request a fixed `ms-settings:` page for Wi-Fi, network, Bluetooth,
  devices, about, display or sound. Launch acceptance is not proof of pairing
  or connecting. Password entry and first-time network configuration belong in
  Windows Settings; the APP exposes no password argument.

The host bridge does not expose arbitrary shell commands, programs, registry
writes, profile installation/deletion, remote control, shutdown, credential
extraction or permission repair. Existing DSH workspace tasks remain separate.
Windows access restrictions are respected and returned as errors. In particular,
WLAN details may be unavailable when the desktop process lacks the required
Wi-Fi/location permission. No permission is changed automatically.

Only a server-validated `system_action` can execute one of these operations.
Actual local-computer requests cannot fall through to an imagined settings app.
A proposed action mixed with visual-only flags is corrected before execution;
the final plan is checked again. Connect requires a currently saved profile and
either the exact explicitly named target or the selected identity on this
session's measured system page. Explicit “do not connect / only select” wording
is enforced locally even if the model proposes Connect. Request journaling
reserves the operation before execution and deduplicates replay. An interrupted
Windows receipt is not replayed by the workspace-tool recovery path.

The host endpoint `/api/computer` is read-only and served by the loopback APP.
There is no unauthenticated generic system-command endpoint. `WINDOWS CALL` and
`WINDOWS RESULT` records appear in the existing read-only command window.

## Drawing and verification

DSH chooses the screen category, text and controls. For a new whole-page
candidate, a generic geometry constraint pass can enlarge/reflow those exact
rectangles to fit the native trained glyph cells. It preserves every AI-chosen
control and literal label, chooses no app template, and draws no pixels. It is
never applied to an existing screenshot or a local edit mask. Unknown glyphs or
unfit content still fail. The final full PNG, including neural navigation, is
reviewed against the actual Windows receipt before publication. A failed redraw
does not undo or repeat a completed system operation; its receipt remains saved.

## Measured development acceptance

The deployed Windows build read the actual CPU, memory/disk information, Wi-Fi
interface, current saved-network connection and saved/visible network names. The final
HTTP interaction test passed:

- local-computer information and Wi-Fi pages backed by Windows receipts;
- pictured Refresh and Connect clicks;
- current-network verification returning `already_connected` without submitting
  a reconnect or changing networks;
- exact replay of the Connect receipt without a second operation;
- Home restoring the identical desktop with no old system-page metadata.

A further real AI interaction selected a different saved profile without
connecting it, verified that the original Windows connection was unchanged,
and restored the UI selection. The two selection requests measured 7.59/7.90 s.

These page/refresh/current-connection operations measured **6.08–6.70 seconds**;
Home measured **2.28 seconds** in the final run. These are individual development
HTTP observations, not native-window latency guarantees. Forty-seven Python
regression tests and the Node window-controller suite passed. Earlier rejected
layout/routing/refresh attempts remain in the workspace evidence rather than
being presented as successful tests.

A transition to a different real network, first-time password entry, native
Settings-window appearance and Bluetooth pairing were **not live-tested**. The
connection-admission/timeout, access-denial, unknown-target, no-connect wording,
selection-only and replay cases have deterministic tests. This does not certify
general control of every Windows application.

Implementation references: [WlanConnect](https://learn.microsoft.com/en-us/windows/win32/api/wlanapi/nf-wlanapi-wlanconnect),
[WLAN connection state](https://learn.microsoft.com/en-us/windows/win32/api/wlanapi/ns-wlanapi-wlan_connection_attributes),
and [Windows Settings URIs](https://learn.microsoft.com/en-us/windows/apps/develop/launch/launch-settings).
