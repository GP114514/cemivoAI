# Local Windows computer integration

`2.14.0-system-settings` connects the generated APP to the computer on which its
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
- `打开设置。` opens the six-category in-app hub.
- `打开声音设置。` / `打开显示设置。` / `打开蓝牙设置。`

A generated network-name control selects a saved or visible network first. **Connect**
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
- `wifi_network_select` / `wifi_network_connect`: select an observed visible
  network and connect it, requesting a private in-app password when required.
- `settings`: open the internal category hub or a specific category, with no
  external Windows Settings launch.
- `audio_volume` / `audio_mute`: apply and read back default-output sound state.
- `bluetooth_scan`: enumerate/inquire for classic Bluetooth devices.

The host bridge does not expose arbitrary shell commands, programs, registry
writes, persistent profile installation/deletion, remote control, shutdown, credential
extraction or permission repair. Existing DSH workspace tasks remain separate.
Windows access restrictions are respected and returned as errors. In particular,
WLAN details may be unavailable when the desktop process lacks the required
Wi-Fi/location permission. No permission is changed automatically.

Only a server-validated `system_action` can execute one of these operations.
Actual local-computer requests cannot fall through to an imagined settings app.
A proposed action mixed with visual-only flags is corrected before execution;
the final plan is checked again. Connect requires a measured saved/visible network and
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

## Initial 2.13 development acceptance

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

A transition to a different real network using a real password and Bluetooth
pairing were **not live-tested**. The
connection-admission/timeout, access-denial, unknown-target, no-connect wording,
selection-only and replay cases have deterministic tests. This does not certify
general control of every Windows application.

Implementation references: [WlanConnect](https://learn.microsoft.com/en-us/windows/win32/api/wlanapi/nf-wlanapi-wlanconnect),
[WLAN connection state](https://learn.microsoft.com/en-us/windows/win32/api/wlanapi/ns-wlanapi-wlan_connection_attributes),
and [Core Audio endpoint volume](https://learn.microsoft.com/en-us/windows/win32/api/endpointvolume/nn-endpointvolume-iaudioendpointvolume).

## All settings stay inside cemivo

The former native Windows Settings launch path has been removed. **Settings**
now means the generated category hub: **Wi-Fi, Sound, Display, Bluetooth, Devices,
About**. The page validator requires all six entry buttons together. Each
category has its own page identity and returns to the same generated hub.

- **Wi-Fi:** saved and visible network selection, connection and refresh. An
  unsaved WPA/WPA2 personal or WPA3-SAE/AES network can request a password in an
  AI-chosen empty input on the generated image. A masked local input layer
  receives keystrokes in that rectangle. Its dedicated loopback submission
  bypasses AI planning, ordinary event text and draft/session storage. The APP
  supplies an in-memory **temporary** WLAN profile; it does not save a credential
  file or automatically persist the new profile. Enterprise/EAP and unsupported
  security modes remain unavailable inside the APP.
- **Sound:** actual default-output volume/mute, with native Core Audio setters
  and read-back. There is no invented output-device switching function.
- **Display:** actual current resolution, refresh rate and color depth. This
  release displays those facts and refreshes them; changing resolution or
  brightness is not implemented.
- **Bluetooth:** actual classic-radio/remembered/connected device enumeration
  and inquiry. An absent or unready radio is reported honestly. Pairing and
  radio power control are not implemented; no Windows pairing window is opened.
- **Devices:** present-device inventory/class counts from Windows SetupAPI.
  Installing/removing drivers and devices is not implemented.
- **About:** local computer/CPU/memory/disk/Windows information.

The password field and its submit button are bound to indices and rectangles
chosen by the AI layout. A short-lived, session-bound token prevents a stale
page from submitting credentials. Passwords are excluded from model prompts,
screenshots, the durable request journal, console fields and browser storage.
They necessarily exist transiently in local process/request memory while passed
to Windows. The APP does not claim to erase all OS/runtime memory copies.
Replay returns the recorded native result instead of submitting the key again.

Connection feedback uses the local neural text head in the existing AI-chosen
input/receipt regions. It does not require an online model after the native
connection attempt. A failed/unconfirmed result can renew the private input
inside the same page for retry. A controlled failure fixture using the actual
text weights verified zero outside-region pixel changes and no model/network
connection calls during that feedback. A real invalid-password join was not run.

Latest deployed verification passed the internal Wi-Fi-settings page, selection
of a real visible unsaved network, the generated private password page, in-app
cancel/Home, stale-private-submission rejection and unchanged existing connection.
No real Wi-Fi password was submitted in this live test. The private submission,
temporary-profile admission, replay, journal/model/console exclusion and offline
feedback ownership have deterministic tests.

The six-category acceptance clicked every actual generated entry. It exercised
the Sound volume-down button and restored the original volume, read Display and
Devices, reported the current unready Bluetooth radio, visited About/Wi-Fi, and
returned to the identical hub. New pages took roughly 6–8 seconds and cached
Back took roughly 2 seconds in that run. Native-window mouse/keyboard acceptance
is still not established by these HTTP/controller tests. Sixty-one Python tests
and the Node controller suite passed. No external Settings launch API remains.
