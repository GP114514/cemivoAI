"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const elements = Object.fromEntries([
    "connectionDot", "connectionText", "sessionForm", "initialPrompt", "promptBytes", "samplingSteps", "sessionSeed", "createSession", "createLabel", "syncSession", "backButton", "plannerForm", "plannerKey", "plannerEndpoint", "plannerModel", "savePlanner", "plannerBadge", "plannerEvidence", "frameVersion", "activityState", "activityText", "notice", "frameStage", "nativeDimensions", "displayDimensions", "imageHost", "emptyState", "pixelSurface", "liveImage", "clickMarker", "frameCheckpoint", "plannerTiming", "sampleTiming", "serverTiming", "roundtripTiming", "eventForm", "eventText", "submitText", "planEvidence", "planIntent", "planExpected", "planPrompt", "sessionIdentity", "factVersion", "factSteps", "frameHash", "messageCount", "loadTiming", "presentTiming", "sessionNotice", "contextMessages", "trainingState", "trainingSteps", "snapshotStep", "runtimeDevice", "runtimeLoaded", "runtimeError", "eventCount", "eventList", "emptyEvents", "agentToggle", "agentModeBadge", "agentAvailability", "queryAgentRequest", "agentResultTitle", "agentResultStatus", "agentPending", "agentResultEvidence", "agentAnswer", "agentWarning", "agentRequestId", "agentRounds", "agentTiming", "agentRenderTiming", "agentTotalTiming", "agentToolCount", "agentToolList", "refreshWorkspace", "workspaceEvidence", "documentCount", "taskCount", "workspaceDocuments", "workspaceTasks", "agentCapabilities", "interactionMode", "interactionModeBadge", "interactionModeNote", "scopeDecision", "localEditEvidence"
  ].map((id) => [id, $(id)]));
  Object.assign(elements, Object.fromEntries(["visionFeedback", "visionFeedbackTitle", "visionFeedbackElapsed", "visionFeedbackCoordinates", "visionFeedbackMessage", "visionFeedbackEvidence", "clarifyFeedback"].map(id => [id, $(id)])));
  Object.assign(elements, Object.fromEntries(["contextInput", "contextInputClose", "contextInputForm", "contextInputText", "contextInputSend", "contextInputPosition", "contextInputStatus", "promptNote", "sampling32Option"].map(id => [id, $(id)])));
  elements.modelVersionNotice = $("modelVersionNotice");
  elements.loadLatestModel = $("loadLatestModel");
  const encoder = new TextEncoder();
  $("openScreenWindow")?.addEventListener("click",()=>{window.open('/screen','cemivoAI-screen','popup,width=540,height=580,resizable=yes');});
  const state = { frame: null, status: null, online: false, busy: false, polling: false, plannerDirty: false, events: 0, statusTimer: null, destroyed: false, agentEnabled: true, agentAvailable: false, workspaceBusy: false, workspaceChecked: false, agentQueryBusy: false, latestAgentRequestId: null, agentUncertain: false, confirmedAgent: null, interactionMode: "auto" };
  const knownNumber = (value) => typeof value === "number" && Number.isFinite(value);
  const plainText = (value) => typeof value === "string" && value.trim() ? value : "—";
  const numberText = (value) => knownNumber(value) ? new Intl.NumberFormat("zh-CN").format(value) : "—";
  const msText = (value) => knownNumber(value) && value >= 0 ? `${new Intl.NumberFormat("zh-CN", { maximumFractionDigits: value < 100 ? 1 : 0 }).format(value)} ms` : "—";
  const put = (node, value) => { node.textContent = value; };
  Object.assign(state, { feedbackActive: false, feedbackTimer: null, feedbackStarted: null, feedbackStage: "", feedbackAgent: false });
  Object.assign(state, { contextAnchor: null, contextComposing: false, contextSending: false, contextDismissTarget: null });
  Object.assign(state, { frameHighlightTimer: null, presentation: null });

  class ApiError extends Error {
    constructor(message, status = 0, code = "") { super(message); this.name = "ApiError"; this.status = status; this.code = code; }
  }

  async function requestJSON(url, { method = "GET", body, timeout = 120000 } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    const started = performance.now();
    try {
      const response = await fetch(url, {
        method, headers: body === undefined ? { Accept: "application/json" } : { Accept: "application/json", "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal, cache: "no-store", credentials: "same-origin"
      });
      let data;
      try { data = await response.json(); } catch { throw new ApiError(`本地服务返回了无法读取的响应（${response.status}）。`, response.status); }
      if (!response.ok) throw new ApiError(typeof data?.error === "string" ? data.error : `请求未完成（${response.status}）。`, response.status, typeof data?.code === "string" ? data.code : "");
      return { data, rtt: performance.now() - started };
    } catch (error) {
      if (error instanceof ApiError) throw error;
      if (error?.name === "AbortError") throw new ApiError("请求超时，当前画面已保留。可先同步会话，再决定下一步。", 0, "timeout");
      throw new ApiError("无法连接本地服务，当前画面已保留。", 0, "network");
    } finally { clearTimeout(timer); }
  }

  function showNotice(message, isError = false) {
    put(elements.notice, message);
    elements.notice.classList.toggle("error", isError);
    elements.notice.hidden = !message;
  }

  const busyInteractionNotice = "上一项操作仍在处理中，这次点击或说明没有发送，也不会排队或自动重放。结果返回后，请在新画面再次操作。";
  function explainBusyInteraction() {
    showNotice(busyInteractionNotice);
    if (!state.feedbackActive) showFeedback("正在处理上一次操作", busyInteractionNotice, { kind: "working" });
  }

  function feedbackText(value, fallback) {
    const text = typeof value === "string" ? value.trim() : "";
    if (!text) return fallback;
    return /[\u3400-\u9fff]/u.test(text) ? text : `${fallback}\nAI 返回：${text}`;
  }

  function previewLabel(agent) {
    return agent?.business_status === "failed" ? "业务未完成 · 待确认预览" : agent?.business_status === "partial" ? "业务未全部完成 · 待确认预览" : "待确认预览";
  }

  function feedbackContext(event) {
    put(elements.visionFeedbackCoordinates, event.type === "create" ? "已收到初始应用说明" : event.type === "click" ? `原图点击 (${numberText(event.x)}, ${numberText(event.y)}) · 第 ${numberText(state.frame?.version)} 帧` : event.type === "back" ? "读取历史画面，不重新执行任务" : "已收到文字操作说明");
  }

  function showFeedback(title, message, { kind = "working", clarify = false } = {}) {
    elements.visionFeedback.hidden = false;
    elements.visionFeedback.dataset.state = kind;
    put(elements.visionFeedbackTitle, title);
    put(elements.visionFeedbackMessage, message);
    elements.clarifyFeedback.hidden = !clarify;
    elements.clarifyFeedback.disabled = state.busy;
  }

  function stopFeedbackTimer() {
    clearTimeout(state.feedbackTimer);
    state.feedbackTimer = null;
    state.feedbackActive = false;
  }

  function feedbackTick() {
    if (!state.feedbackActive || state.destroyed) return;
    const elapsed = Math.max(0, Math.floor((performance.now() - state.feedbackStarted) / 1000));
    put(elements.visionFeedbackElapsed, `已等待 ${elapsed} 秒`);
    if (elapsed >= 5) {
      const detail = state.feedbackCreate
        ? state.feedbackStage === "refining" ? "正在检查实际生成的首屏，必要时只修复一次；请勿重复创建。" : state.feedbackStage === "drawing" ? "本地模型正在绘制首屏，已有画面会保留到新图就绪；请勿重复创建。" : "正在根据完整应用说明规划首屏；请求仍在处理中，请勿重复创建。"
        : state.feedbackStage === "refining" ? "正在结合完整页面复查点击细节；仍在识图/执行，请稍候。" : state.feedbackStage === "drawing" ? "正在生成并核验新画面；仍在识图/执行，请稍候。" : "仍在识图/执行，请稍候。请勿重复提交同一次操作。";
      put(elements.visionFeedbackMessage, detail);
    }
    state.feedbackTimer = setTimeout(feedbackTick, 1000);
  }

  function feedbackStage(stage) {
    if (!state.feedbackActive) return;
    state.feedbackStage = stage;
    const stages = state.feedbackCreate
      ? { understanding: ["正在规划首屏", "正在根据完整应用说明整理页面与控件。"], refining: ["正在检查首屏画面", "正在结合完整需求检查实际生成结果，必要时只修复一次。"], drawing: ["正在生成首屏画面", "本地模型正在绘制首屏，检查结果以实际返回内容为准。"] }
      : { understanding: ["正在识别点击目标", "已记录点击位置，正在结合当前画面理解你的意图。"], refining: ["正在结合完整页面复查点击细节", "首次识别不够明确，正在把完整页面与点击附近的细节一起复查。"], acting: ["Agent 正在执行任务", "正在调用实际能力，执行结果会显示在本次回执中。"], drawing: ["正在生成并核验画面", "识别计划已返回，正在绘制结果并检查允许变化的范围。"], history: ["正在读取上一帧", "只读取历史画面，不会重复执行任务。"] };
    const [title, message] = stages[stage] || stages.understanding;
    showFeedback(title, message);
  }

  function beginFeedback(event, agentMode) {
    stopFeedbackTimer();
    feedbackContext(event);
    state.feedbackActive = true; state.feedbackStarted = performance.now(); state.feedbackAgent = agentMode; state.feedbackCreate = event.type === "create";
    put(elements.visionFeedbackEvidence, state.feedbackCreate || state.status?.capabilities?.vision_grounding === true ? "" : "当前后台未加载识图增强，本次使用已有识图能力。"); put(elements.visionFeedbackElapsed, "已等待 0 秒");
    feedbackStage(event.type === "back" ? "history" : event.type === "click" ? "understanding" : agentMode ? "acting" : "understanding");
    feedbackTick();
  }

  function finishFrameFeedback(frame) {
    if (frame.last_event?.type === "create" && !frame.agent) {
      const rejected = frame.initial_review && frame.initial_review.accepted !== true;
      const detail = feedbackText(frame.state?.notice, rejected ? "首屏尚未通过检查，实际图片已保留。" : "首屏已生成，文字与完整交互仍需结合实际画面判断。");
      if (knownNumber(state.feedbackStarted)) put(elements.visionFeedbackElapsed, `本次等待 ${Math.max(0, Math.floor((performance.now() - state.feedbackStarted) / 1000))} 秒`);
      put(elements.visionFeedbackEvidence, frame.initial_review ? "首屏检查回执已返回，文字可读性与完整功能仍需实际判断。" : "本次仅生成首屏，没有视觉检查回执。");
      showFeedback(rejected ? "首屏尚未通过检查" : "首屏已生成", detail, { kind: rejected ? "error" : "complete" });
      if (rejected) showNotice(detail, true);
      stopFeedbackTimer();
      return;
    }
    const recognition = frame.recognition || frame.plan?.recognition;
    const status = frame.agent?.status;
    const failed = ["render_failed", "partial_failure", "interrupted"].includes(status);
    const preview = verifiedRegionPreview(frame);
    const noRedraw = frame.plan?.should_update === false && frame.last_event?.type !== "back";
    const needsInput = recognition?.status === "needs_input" || status === "needs_input";
    const evidence = [];
    if (recognition?.full_image_used === true) evidence.push("已结合完整页面");
    if (Number.isInteger(recognition?.attempts) && recognition.attempts >= 1 && recognition.attempts <= 2) evidence.push(`实际识别 ${recognition.attempts} 次`);
    if (recognition?.focus_used === true) evidence.push("已检查点击区域放大图");
    else if (recognition?.focus_used === false) evidence.push("本次未使用放大复查");
    if (noRedraw && frame.plan?.no_change_reason === "user_requested") evidence.push("已按你的要求保留原图");
    else if (noRedraw && frame.plan?.no_change_reason === "at_target") evidence.push("AI 判断已处于目标状态");
    if (!failed && !noRedraw && frame.plan?.text_action === "input") evidence.push("已请求重绘输入内容，文字以实际图片为准");
    else if (!failed && !noRedraw && frame.plan?.text_action === "send") evidence.push("已请求显示发送的消息，文字以实际图片为准");
    if (preview) evidence.push("区域外像素未变；区域内内容尚未通过视觉核验");
    if (!recognition && state.status?.capabilities?.vision_grounding !== true) evidence.push("当前后台未加载识图增强，结果来自已有识图能力。");
    put(elements.visionFeedbackEvidence, evidence.join(" · "));
    const fallback = failed ? "本次执行或画面更新未全部完成，画面保持。任务情况请检查本次回执。" : needsInput ? "暂时无法确定这次点击的目标。原图保持，请补充你希望执行的动作。" : noRedraw ? "本次返回了结果并保留原图，没有重新绘图。" : "本次操作已返回结果，请结合实际画面检查。";
    const detail = feedbackText(failed ? frame.agent?.warning || recognition?.message : needsInput ? recognition?.message || frame.agent?.answer || frame.plan?.intent : frame.agent?.answer || frame.state?.notice || recognition?.message || (noRedraw ? frame.plan?.expected_change || frame.plan?.intent : null), fallback);
    if (knownNumber(state.feedbackStarted)) put(elements.visionFeedbackElapsed, `本次等待 ${Math.max(0, Math.floor((performance.now() - state.feedbackStarted) / 1000))} 秒`);
    if (failed) {
      showFeedback("本次操作未全部完成", detail, { kind: "error", clarify: true });
      showNotice(detail, true);
    } else if (preview) {
      const message = feedbackText(frame.agent?.warning || frame.state?.notice,
        "已显示模型生成的待确认预览，内容和文字尚未通过核验。原文字已保留，可继续修改或返回上一帧。");
      showFeedback(`${previewLabel(frame.agent)} · 内容尚未通过核验`, message, { kind: "preview" });
      showNotice(message);
    } else if (needsInput) {
      showFeedback("需要补充说明 · 原图保持", detail, { kind: "needs-input", clarify: true });
      showNotice(detail);
    } else {
      const mode = frame.scope?.mode;
      const samePixels = state.presentation?.sessionId === frame.session_id && state.presentation?.version === frame.version && state.presentation?.changed === false;
      showFeedback(frame.last_event?.type === "back" ? "已读取历史画面" : noRedraw ? "结果已返回 · 画面保持不变" : samePixels ? "绘图已返回 · 像素保持相同" : mode === "local" || frame.edit ? "局部更新已返回" : "画面更新已返回", detail, { kind: "complete" });
      if (noRedraw || samePixels) showNotice(detail);
    }
    stopFeedbackTimer();
  }

  function failFeedback(error) {
    const unresolvedAgent = state.feedbackAgent && state.agentUncertain;
    const fallback = state.feedbackCreate ? "首屏生成尚未完成，已有画面和应用说明已保留。" : unresolvedAgent ? "任务执行结果尚未确认。请先查询任务回执，不要重复提交；当前画面保持。" : "本次没有更新画面。原图保持，请补充操作意图后再决定下一步。";
    const detail = feedbackText(error?.message, fallback);
    showFeedback(state.feedbackCreate ? "首屏生成未完成 · 已有画面保留" : unresolvedAgent ? "任务结果待确认" : "本次操作未完成 · 原图保持", detail.includes(fallback) ? detail : `${detail}\n${fallback}`, { kind: "error", clarify: !unresolvedAgent && !state.feedbackCreate });
    stopFeedbackTimer();
  }

  function addEvent(name, detail = "", isError = false) {
    elements.emptyEvents?.remove();
    const row = document.createElement("li");
    if (isError) row.classList.add("error");
    const title = document.createElement("span"); title.className = "event-name"; title.textContent = name;
    row.append(title);
    if (detail) { const text = document.createElement("span"); text.className = "event-detail"; text.textContent = detail; row.append(text); }
    const time = document.createElement("time"); const now = new Date(); time.dateTime = now.toISOString(); time.textContent = now.toLocaleTimeString("zh-CN", { hour12: false }); row.append(time);
    elements.eventList.prepend(row);
    while (elements.eventList.children.length > 30) elements.eventList.lastElementChild.remove();
    state.events += 1; put(elements.eventCount, String(state.events));
  }

  function updatePromptCount() {
    const text = elements.initialPrompt.value;
    const bytes = encoder.encode(text).length;
    const planned = state.status?.capabilities?.initial_planning === true;
    const limit = planned ? 6000 : 190;
    put(elements.promptBytes, `${bytes} / ${limit} 字节${planned ? ` · ${Array.from(text).length} / 2000 字` : ""}`);
    elements.promptBytes.classList.toggle("over-limit", bytes > limit || Array.from(text).length > 2000);
    put(elements.promptNote, planned ? `${state.status?.planner?.configured === true ? "" : "请先保存视觉 AI 配置。"}完整说明会保留，由视觉 AI 规划首屏；最多 2000 字、6000 UTF-8 字节。生成文字尚不可靠。` : bytes > 190 ? "当前后台尚未支持完整应用说明，请等待更新。现有说明已保留，不会截断；短描述仍可按 190 字节生成。" : "当前后台使用短描述直接绘图，最多 190 个 UTF-8 字节；生成文字尚不可靠。");
    return bytes;
  }

  function updateControls() {
    const available = state.online && state.status?.available === true;
    const promptBytes = updatePromptCount();
    const planned = state.status?.capabilities?.initial_planning === true;
    const validPrompt = elements.initialPrompt.value.trim().length > 0 && promptBytes <= (planned ? 6000 : 190) && Array.from(elements.initialPrompt.value).length <= 2000;
    const supports32 = Array.isArray(state.status?.capabilities?.sampling_steps) && state.status.capabilities.sampling_steps.includes(32);
    elements.sampling32Option.disabled = !supports32;
    put(elements.sampling32Option, supports32 ? "32 步" : "32 步 · 需后台支持");
    elements.createSession.disabled = state.busy || !available || !validPrompt || (planned && state.status?.planner?.configured !== true);
    put(elements.createLabel, state.frame ? "重建会话" : "创建会话");
    elements.syncSession.disabled = state.busy || !state.online || !state.frame;
    elements.backButton.disabled = state.busy || !state.online || !state.frame;
    elements.submitText.disabled = state.busy || !available || !state.frame || !elements.eventText.value.trim();
    elements.savePlanner.disabled = state.busy || !state.online;
    elements.agentToggle.disabled = state.busy || !state.online || !state.agentAvailable || state.status?.planner?.configured !== true;
    elements.agentToggle.checked = state.agentEnabled;
    put(elements.agentModeBadge, state.agentEnabled ? "开启" : "关闭");
    elements.agentModeBadge.classList.toggle("configured", state.agentEnabled);
    const modeAvailable = editingAvailable();
    elements.interactionMode.value = state.interactionMode;
    elements.interactionMode.disabled = state.busy || !state.online;
    const modeNames = { auto: "AI 自动选择", toggle_only: "只改开关", full: "整屏重绘" };
    put(elements.interactionModeBadge, modeAvailable ? modeNames[state.interactionMode] : "更新待加载");
    elements.interactionModeBadge.classList.toggle("configured", modeAvailable && state.interactionMode !== "full");
    put(elements.interactionModeNote, !modeAvailable ? "当前后台尚未加载所选模式，画面点击暂不发送。可以主动选择整屏重绘。" : state.interactionMode === "auto" ? "页面操作优先可见更新，开关锁定其他像素；未通过验证保留原图。开启 Agent 后，需要实际执行的任务交给工具完成。" : state.interactionMode === "toggle_only" ? "只允许开关局部编辑，其余像素锁定；识别不到开关则保持原图。画面点击不执行 Agent 工具。" : "整屏重绘可能改变整个画面；开启 Agent 时，点击也可以执行工作区工具。");
    elements.refreshWorkspace.disabled = !state.online || state.workspaceBusy;
    elements.loadLatestModel.hidden = state.status?.capabilities?.model_update !== true;
    elements.loadLatestModel.disabled = state.busy || !state.online || state.status?.capabilities?.model_update !== true || state.status?.model_update?.available !== true;
    elements.queryAgentRequest.disabled = !state.online || !state.latestAgentRequestId || state.agentQueryBusy || state.busy;
    put(elements.submitText, state.agentEnabled ? "发送任务给 Agent ↗" : "发送文字给视觉 AI ↗");
    if (state.agentEnabled && state.agentUncertain) elements.submitText.disabled = true;
    for (const element of [elements.initialPrompt, elements.samplingSteps, elements.sessionSeed, elements.plannerEndpoint, elements.plannerModel, elements.plannerKey, elements.eventText]) element.disabled = state.busy;
    elements.liveImage.setAttribute("aria-disabled", state.busy || !state.online || !modeAvailable ? "true" : "false");
    elements.frameStage.setAttribute("aria-busy", String(state.busy));
    elements.clarifyFeedback.disabled = state.busy;
    updateContextInputControls();
    elements.activityState.classList.toggle("busy", state.busy);
    elements.activityState.classList.toggle("ready", Boolean(state.frame) && !state.busy);
    if (!state.busy) put(elements.activityText, state.frame ? "等待下一次操作" : available ? "可创建首屏" : "等待绘图模型");
  }

  function neuralSurfaceSource(frame) {
    const standalone = frame?.generator === "native_neural_surface";
    const navigation = frame?.surface_chrome?.generator === "native_neural_surface";
    // An application frame can also carry surface_model for its navigation.
    // That metadata alone must never change the identity of its main model.
    if (!standalone && !navigation) return null;
    const source = frame.surface_model;
    const recordedNumber = (...values) => values.find(value => Number.isInteger(value) && value >= 0);
    return {
      standalone,
      step: recordedNumber(source?.checkpoint_step, source?.model?.checkpoint_step, source?.model?.step, source?.step),
      parameters: recordedNumber(source?.parameters, source?.model?.parameters),
      textParameters: recordedNumber(source?.model?.text_parameters, source?.text_parameters)
    };
  }

  function surfaceModelDescription(surface) {
    const step = knownNumber(surface.step) ? `${numberText(surface.step)} 步` : "步数未提供";
    const parameters = [];
    if (knownNumber(surface.parameters)) parameters.push(`布局与按钮 ${numberText(surface.parameters)} 参数`);
    if (knownNumber(surface.textParameters)) parameters.push(`文字头 ${numberText(surface.textParameters)} 参数`);
    return `神经桌面/导航辅助模型 ${step}${parameters.length ? `（${parameters.join("；")}）` : ""}`;
  }

  function renderModelVersion() {
    const snapshot = state.status?.snapshot;
    const frame = state.frame;
    const surface = neuralSurfaceSource(frame);
    const snapshotKnown = knownNumber(snapshot?.step);
    if (!snapshotKnown && !surface) {
      elements.modelVersionNotice.hidden = true;
      put(elements.modelVersionNotice, "");
      return;
    }
    const parts = [snapshotKnown ? `APP 当前使用 ${numberText(snapshot.step)} 步模型（主图像模型）` : "主图像模型状态尚未读取，辅助模型来源不代表主模型状态。"];
    if (snapshotKnown && knownNumber(snapshot.native_size)) parts[0] += ` · 原生 ${numberText(snapshot.native_size)} × ${numberText(snapshot.native_size)}`;
    if (surface) parts.push(`${surface.standalone ? "当前图片由" : "当前画面的导航由"}${surfaceModelDescription(surface)}生成；主模型状态仍以服务快照为准。`);
    const mainFrame = !surface?.standalone;
    const differentStep = snapshotKnown && mainFrame && knownNumber(frame?.checkpoint_step) && frame.checkpoint_step !== snapshot.step;
    const differentWeights = snapshotKnown && mainFrame && typeof frame?.model_sha256 === "string" && frame.model_sha256 && typeof snapshot.sha256 === "string" && snapshot.sha256 && frame.model_sha256 !== snapshot.sha256;
    if (differentStep) parts.push(`当前图片来自 ${numberText(frame.checkpoint_step)} 步模型；重建会话后才能使用当前 APP 模型。`);
    else if (differentWeights) parts.push("当前图片与 APP 使用的权重不同，请重建会话后继续。");
    const training = state.status?.training;
    const trainingAhead = snapshotKnown && knownNumber(training?.step) && training.step > snapshot.step;
    const update = state.status?.model_update;
    const checkpointReady = state.status?.capabilities?.model_update === true && update?.available === true && knownNumber(update.checkpoint_step);
    if (checkpointReady) parts.push(`已保存 ${numberText(update.checkpoint_step)} 步新模型，可点击“加载最新模型”；当前画面会保留，新会话使用更新后的模型。`);
    else if (trainingAhead) parts.push(`训练进度已到 ${numberText(training.step)} 步，尚未切换到 APP；训练进度不会自动更新推理模型。`);
    if (!state.online) parts.unshift("服务未连接，以下是上次读取的模型状态。");
    put(elements.modelVersionNotice, parts.join("\n"));
    elements.modelVersionNotice.hidden = false;
    elements.modelVersionNotice.classList.toggle("attention", Boolean(differentStep || differentWeights || trainingAhead || checkpointReady || !state.online));
  }

  function renderStatus(data) {
    state.status = data; state.online = true;
    elements.connectionDot.className = "connected"; put(elements.connectionText, "本地服务在线");
    const planner = data?.planner;
    const configured = planner?.configured === true;
    const verified = configured && planner?.verified === true;
    elements.plannerBadge.classList.toggle("configured", configured);
    put(elements.plannerBadge, planner && typeof planner.configured === "boolean" ? verified ? "已有真实响应" : configured ? "已配置" : "待配置" : "—");
    put(elements.plannerEvidence, verified ? `${plainText(planner.model)} 已返回过有效视觉计划；后续请求仍以实际响应为准。` : configured ? `本次配置：${plainText(planner.model)}。保存配置不代表远端连接已验证。` : "填写密钥后保存。首次交互时才会实际请求 DeepSeek。");
    if (!state.plannerDirty) {
      if (typeof planner?.base_url === "string" && planner.base_url) elements.plannerEndpoint.value = planner.base_url;
      if (typeof planner?.model === "string" && planner.model) elements.plannerModel.value = planner.model;
    }
    const training = data?.training;
    const trainingNames = { running: "训练中", paused: "已暂停", complete: "阶段完成", failed: "发生错误", preparing: "准备中" };
    put(elements.trainingState, typeof training?.state === "string" ? trainingNames[training.state] || training.state : "—");
    put(elements.trainingSteps, knownNumber(training?.step) && knownNumber(training?.total_steps) ? `${numberText(training.step)} / ${numberText(training.total_steps)}` : "—");
    put(elements.snapshotStep, numberText(data?.snapshot?.step));
    renderModelVersion();
    put(elements.runtimeDevice, plainText(data?.runtime?.device));
    put(elements.runtimeLoaded, typeof data?.runtime?.loaded === "boolean" ? data.runtime.loaded ? "是" : "否" : "—");
    const runtimeError = typeof data?.runtime?.last_error === "string" ? data.runtime.last_error : "";
    put(elements.runtimeError, runtimeError); elements.runtimeError.hidden = !runtimeError;
    updateControls();
    if (!state.workspaceChecked && !state.workspaceBusy) void refreshWorkspace();
    if (state.busy && data?.runtime?.busy === true) {
      if (data.runtime.stage === "understanding") put(elements.activityText, state.feedbackCreate ? "DeepSeek 正在规划首屏" : "DeepSeek 正在理解画面");
      else if (data.runtime.stage === "refining") put(elements.activityText, state.feedbackCreate ? "正在检查实际生成的首屏" : "正在结合完整页面复查点击细节");
      else if (data.runtime.stage === "drawing") put(elements.activityText, state.feedbackCreate ? "本地图像模型正在生成首屏" : "本地图像模型正在绘图");
      else if (data.runtime.stage === "acting") put(elements.activityText, "Agent 正在执行任务");
      if (["understanding", "refining", "drawing", "acting"].includes(data.runtime.stage)) feedbackStage(data.runtime.stage);
    }
  }

  async function pollStatus() {
    if (state.polling || state.destroyed) return;
    state.polling = true;
    clearTimeout(state.statusTimer);
    try { const result = await requestJSON("/api/live/status", { timeout: 10000 }); renderStatus(result.data); }
    catch {
      state.online = false; elements.connectionDot.className = "offline"; put(elements.connectionText, "本地服务未连接");
      renderModelVersion();
      if (!state.busy) showNotice("本地服务暂时未连接，已有图片会保留。连接恢复后可同步当前会话。", true);
      updateControls();
    } finally { state.polling = false; if (!state.destroyed) state.statusTimer = setTimeout(pollStatus, 5000); }
  }

  function safeImageURL(value) {
    if (typeof value !== "string" || !value) throw new ApiError("本帧没有可读取的真实图片，保留当前画面。");
    if (value.startsWith("data:image/png;base64,")) return value;
    const url = new URL(value, window.location.href);
    if (url.origin !== window.location.origin || !["http:", "https:"].includes(url.protocol)) throw new ApiError("本帧图片地址不属于本地服务，保留当前画面。");
    return url.href;
  }

  function validateFrame(frame) {
    if (!frame || typeof frame.session_id !== "string" || !frame.session_id || !Number.isInteger(frame.version) || frame.version < 0 || !Number.isInteger(frame.width) || !Number.isInteger(frame.height) || frame.width < 1 || frame.height < 1) throw new ApiError("本地服务返回了不完整的帧信息，保留当前画面。");
  }

  function updateDisplayDimensions() {
    if (!state.frame || elements.liveImage.hidden || !elements.liveImage.getAttribute("src")) { put(elements.displayDimensions, "显示 —"); return; }
    const bounds = elements.liveImage.getBoundingClientRect();
    put(elements.displayDimensions, bounds.width && bounds.height ? `显示 ${Math.round(bounds.width)} × ${Math.round(bounds.height)}` : "显示 —");
  }

  function renderContext(messages) {
    elements.contextMessages.replaceChildren();
    if (!Array.isArray(messages) || !messages.length) { const empty = document.createElement("li"); empty.className = "list-empty"; empty.textContent = Array.isArray(messages) ? "本会话暂无消息。" : "—"; elements.contextMessages.append(empty); return; }
    for (const message of messages.slice(-8)) {
      const row = document.createElement("li"); const role = document.createElement("strong"); role.textContent = plainText(message?.role); const content = document.createElement("span"); content.textContent = plainText(message?.text); row.append(role, content); elements.contextMessages.append(row);
    }
  }

  function safeJSON(value) {
    try { const result = JSON.stringify(value, null, 2); return typeof result === "string" ? result : "—"; }
    catch { return "无法显示此结果。"; }
  }

  function safeExportURL(value) {
    if (typeof value !== "string") return null;
    try { const url = new URL(value, window.location.href); return url.origin === window.location.origin && /^\/api\/agent\/exports\/[A-Za-z0-9_-]+$/.test(url.pathname) && !url.search && !url.hash ? url.href : null; }
    catch { return null; }
  }

  function renderAgentResult(agent, { requestId = null, fromFrame = false } = {}) {
    if (!agent || typeof agent !== "object") return;
    const id = typeof agent.request_id === "string" ? agent.request_id : requestId;
    if (fromFrame && state.latestAgentRequestId && id && id !== state.latestAgentRequestId) return;
    state.confirmedAgent = agent;
    if (id) state.latestAgentRequestId = id;
    const status = typeof agent.status === "string" ? agent.status : "";
    const labels = { complete: "结果已返回", preview: previewLabel(agent), render_failed: "绘图未完成", needs_input: "需要补充信息", partial_failure: "部分步骤失败", interrupted: "执行已中断", running: "执行中", pending: "等待确认" };
    put(elements.agentResultTitle, "Agent 任务回执");
    put(elements.agentResultStatus, labels[status] || plainText(status));
    put(elements.agentAnswer, plainText(agent.answer));
    put(elements.agentRequestId, id ? `${id.slice(0, 12)}…` : "—"); elements.agentRequestId.title = id || "";
    put(elements.agentRounds, numberText(agent.rounds)); put(elements.agentTiming, msText(agent.timings?.agent_ms)); put(elements.agentRenderTiming, msText(agent.timings?.render_ms)); put(elements.agentTotalTiming, msText(agent.timings?.total_ms));
    const warning = typeof agent.warning === "string" ? agent.warning : "";
    put(elements.agentWarning, warning); elements.agentWarning.hidden = !warning;
    const hadTools = Array.isArray(agent.tools) && agent.tools.length > 0;
    let evidence = status === "render_failed" ? hadTools ? "画面生成失败，工具执行记录仍保留，请逐项查看实际结果。" : "本次没有执行业务工具，画面生成失败，原图已保留。" : status === "partial_failure" || status === "interrupted" ? "执行未全部完成。请逐条查看真实工具状态；已成功的写入仍然保存。" : status === "needs_input" ? "Agent 需要补充信息。以下为本次已返回的实际结果。" : "以下仅展示本次返回的工具记录；工具完成与画面生成是不同结果。";
    if (status === "preview") evidence = "模型生成的画面已作为待确认预览显示；内容与文字未通过视觉核验。可继续操作或返回上一帧；业务结果以逐条工具记录为准。";
    if (agent.has_saved_frame === false) evidence += " 此请求的历史画面已不可恢复，工具回执仍保留；可新建会话继续。";
    put(elements.agentResultEvidence, evidence);
    const tools = Array.isArray(agent.tools) ? agent.tools : null;
    put(elements.agentToolCount, tools ? String(tools.length) : "—"); elements.agentToolList.replaceChildren();
    if (!tools?.length) { const empty = document.createElement("li"); empty.className = "list-empty"; empty.textContent = tools ? "本次未返回工具执行记录。" : "—"; elements.agentToolList.append(empty); }
    for (const tool of tools || []) {
      const row = document.createElement("li"); const detail = document.createElement("details"); const summary = document.createElement("summary");
      const name = document.createElement("span"); name.textContent = plainText(tool.name);
      const badge = document.createElement("span"); badge.className = "tool-state";
      const writes = ["document_save", "task_add", "task_update"].includes(tool.name);
      const read = /(?:read|list|search|get)$/.test(String(tool.name));
      badge.textContent = tool.status === "complete" ? writes ? "已保存" : read ? "已读取" : "已完成" : tool.status === "failed" ? "失败" : tool.status === "pending" ? "待确认" : plainText(tool.status);
      badge.classList.toggle("tool-failed", tool.status === "failed"); summary.append(name, badge); detail.append(summary);
      const operation = document.createElement("p"); operation.className = "tool-operation"; operation.textContent = typeof tool.operation_id === "string" ? `操作 ${tool.operation_id}` : "操作标识 —"; detail.append(operation);
      for (const [label, value] of [["实际参数", tool.arguments], ["实际结果", tool.result]]) { const title = document.createElement("span"); title.className = "audit-label"; title.textContent = label; const content = document.createElement("pre"); content.textContent = safeJSON(value); detail.append(title, content); }
      if (tool.result && typeof tool.result === "object") {
        const candidates = [tool.result.export?.download_url, tool.result.export_url, tool.result.download_url, tool.result.url];
        const url = candidates.map(safeExportURL).find(Boolean);
        if (url) { const link = document.createElement("a"); link.href = url; link.className = "export-link"; link.textContent = "下载本次实际导出的文档 ↓"; link.setAttribute("download", ""); detail.append(link); }
      }
      row.append(detail); elements.agentToolList.append(row);
    }
    const terminal = ["complete", "preview", "render_failed", "partial_failure", "interrupted", "needs_input"].includes(status);
    if (terminal && (!state.latestAgentRequestId || id === state.latestAgentRequestId)) { state.agentUncertain = false; elements.agentPending.hidden = true; }
    updateControls();
  }

  function renderWorkspace(data) {
    if (!Array.isArray(data?.documents) || !Array.isArray(data?.tasks)) throw new ApiError("工作区返回的数据格式不完整。");
    const renderItems = (list, items, type) => {
      list.replaceChildren();
      if (!items.length) { const empty = document.createElement("li"); empty.className = "list-empty"; empty.textContent = type === "document" ? "工作区暂无文档或笔记。" : "工作区暂无待办。"; list.append(empty); return; }
      for (const item of items) {
        const row = document.createElement("li"); const title = document.createElement("strong"); title.textContent = plainText(item.title);
        const meta = document.createElement("span"); const statusNames = { open: "待办", pending: "待办", todo: "待办", in_progress: "进行中", done: "完成", completed: "完成" };
        const revision = knownNumber(item.revision) ? `版本 ${item.revision}` : "版本 —";
        meta.textContent = type === "task" ? `${typeof item.status === "string" ? statusNames[item.status] || item.status : "—"} · ${revision}` : `${revision} · ${plainText(item.updated_at)}`;
        const id = document.createElement("small"); id.textContent = typeof item.id === "string" ? `ID ${item.id}` : "ID —";
        row.append(title, meta, id); list.append(row);
      }
    };
    put(elements.documentCount, String(data.documents.length)); put(elements.taskCount, String(data.tasks.length));
    renderItems(elements.workspaceDocuments, data.documents, "document"); renderItems(elements.workspaceTasks, data.tasks, "task");
    elements.agentCapabilities.replaceChildren();
    const capabilities = Array.isArray(data.capabilities) ? data.capabilities : [];
    if (!capabilities.length) { const row = document.createElement("li"); row.className = "list-empty"; row.textContent = "—"; elements.agentCapabilities.append(row); }
    for (const capability of capabilities) { const row = document.createElement("li"); row.textContent = typeof capability === "string" ? capability : plainText(capability?.label || capability?.name || capability?.description); elements.agentCapabilities.append(row); }
    put(elements.workspaceEvidence, `来自本机工作区的只读快照 · ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`);
  }

  async function refreshWorkspace() {
    if (state.workspaceBusy || !state.online) return;
    state.workspaceBusy = true; state.workspaceChecked = true; updateControls();
    try {
      const result = await requestJSON("/api/agent/workspace", { timeout: 15000 }); renderWorkspace(result.data); state.agentAvailable = true;
      put(elements.agentAvailability, state.status?.planner?.configured === true ? "工作区工具已加载。开启开关后才会执行 Agent 任务。" : "工作区工具已加载；先保存 DeepSeek 配置，再开启 Agent。");
    } catch (error) {
      state.agentAvailable = false; state.agentEnabled = false;
      put(elements.agentAvailability, error.status === 404 ? "Agent 更新待加载。可稍后只读刷新；现有画面功能不受影响。" : "工作区暂时未读取，已有结果保留。可只读刷新后重试。");
      put(elements.workspaceEvidence, error.status === 404 ? "Agent 更新待加载，尚未取得工作区数据。" : "读取失败，保留上次已确认的工作区内容。");
    } finally { state.workspaceBusy = false; updateControls(); }
  }

  async function queryAgentResult() {
    if (!state.latestAgentRequestId || state.agentQueryBusy || state.busy || !state.online) return;
    const queriedRequestId = state.latestAgentRequestId;
    state.agentQueryBusy = true; updateControls();
    try {
      const result = await requestJSON(`/api/agent/request?id=${encodeURIComponent(queriedRequestId)}`, { timeout: 20000 });
      // A new Agent task can start while this read-only lookup is in flight.
      // Its uncertainty marker and recovery ID must not be replaced by A's
      // late receipt after the user has already submitted task B.
      if (state.latestAgentRequestId !== queriedRequestId) return;
      if (typeof result.data.request_id === "string" && result.data.request_id !== queriedRequestId) throw new ApiError("任务查询返回了其他请求的回执，已保留当前任务状态。");
      renderAgentResult(result.data, { requestId: queriedRequestId });
      await refreshWorkspace();
      if (state.latestAgentRequestId !== queriedRequestId) return;
      const noSavedFrame = result.data.has_saved_frame === false;
      if (!state.agentUncertain && state.frame && !noSavedFrame) { try { await synchronizeSession({ quiet: true }); } catch { /* Tool results remain visible even when a frame cannot be read. */ } }
      showNotice(state.agentUncertain ? "任务仍待确认。本次只查询结果，没有重复执行。" : noSavedFrame ? "已读取真实任务回执与工作区。历史画面缓存已不可恢复，已保存的业务结果仍保留；可创建新会话继续，没有重复执行任务。" : "已读取任务结果与工作区，没有重复执行任务。", noSavedFrame);
      addEvent("已查询 Agent 任务结果", noSavedFrame ? "真实工具结果已展示；历史画面不可恢复，可新建会话。未请求旧会话或重复执行。" : "只读查询，没有重新发送执行请求。");
    } catch (error) { showNotice(`尚未确认这次 Agent 结果。${error.message || "请稍后再查询，不要重复提交。"}`, true); }
    finally { state.agentQueryBusy = false; updateControls(); }
  }

  function markPresentedFrame(changed) {
    clearTimeout(state.frameHighlightTimer);
    state.frameHighlightTimer = null;
    elements.pixelSurface.classList.remove("frame-updated");
    if (!changed || state.destroyed) return;
    // This outline belongs to the viewer, outside the model PNG. It appears
    // only after a different image is decoded; no generated UI is overlaid.
    elements.pixelSurface.classList.add("frame-updated");
    state.frameHighlightTimer = setTimeout(() => {
      elements.pixelSurface.classList.remove("frame-updated");
      state.frameHighlightTimer = null;
    }, 900);
  }

  async function applyFrame(frame, rtt, started, { expectedSessionId = null } = {}) {
    validateFrame(frame);
    const isOlder = () => state.frame && frame.session_id === state.frame.session_id && frame.version < state.frame.version;
    const wrongSession = () => expectedSessionId !== null && (frame.session_id !== expectedSessionId || state.frame?.session_id !== expectedSessionId);
    if (wrongSession()) throw new ApiError("会话已更换，已忽略先前会话的图片响应。", 0, "old_frame_response");
    if (isOlder()) throw new ApiError("收到较旧的帧响应，已保留当前较新的画面。", 0, "old_frame_response");
    if (frame.agent) renderAgentResult(frame.agent, { fromFrame: true });
    const url = safeImageURL(frame.image_url);
    const nextImage = new Image(); nextImage.decoding = "async"; nextImage.src = url;
    try { await nextImage.decode(); } catch { throw new ApiError("新的真实图片未能载入，当前图片已保留。请同步会话重试读取。"); }
    if (nextImage.naturalWidth !== frame.width || nextImage.naturalHeight !== frame.height) throw new ApiError("图片实际尺寸与帧信息不一致，当前图片已保留。");
    if (wrongSession()) throw new ApiError("会话已更换，已忽略先前会话的图片响应。", 0, "old_frame_response");
    if (isOlder()) throw new ApiError("收到较旧的帧响应，已保留当前较新的画面。", 0, "old_frame_response");
    const changed = !state.frame ? true : typeof frame.sha256 === "string" && frame.sha256 && typeof state.frame.sha256 === "string" && state.frame.sha256 ? frame.sha256 !== state.frame.sha256 : null;
    elements.liveImage.src = nextImage.src;
    elements.liveImage.alt = `cemivoAI 实际像素输出，第 ${frame.version} 帧，原生 ${frame.width} × ${frame.height}`;
    elements.pixelSurface.hidden = false; elements.emptyState.hidden = true; elements.clickMarker.hidden = true;
    state.frame = frame;
    state.presentation = { sessionId: frame.session_id, version: frame.version, changed };
    markPresentedFrame(changed === true);
    renderModelVersion();
    put(elements.frameVersion, `v ${String(frame.version).padStart(3, "0")}`);
    put(elements.nativeDimensions, `原生 ${frame.width} × ${frame.height}`);
    const surface = neuralSurfaceSource(frame);
    put(elements.frameCheckpoint, surface?.standalone
      ? `桌面辅助检查点 ${numberText(surface.step)}`
      : `${surface ? "主" : ""}检查点 ${numberText(frame.checkpoint_step)}${surface ? ` · 导航辅助 ${numberText(surface.step)} 步` : ""}`);
    put(elements.factVersion, numberText(frame.version)); put(elements.factSteps, knownNumber(frame.steps) ? `${frame.steps} 步` : "—");
    put(elements.sessionIdentity, frame.session_id.length > 14 ? `${frame.session_id.slice(0, 12)}…` : frame.session_id); elements.sessionIdentity.title = frame.session_id;
    put(elements.frameHash, typeof frame.sha256 === "string" && frame.sha256 ? `${frame.sha256.slice(0, 12)}…` : "—"); elements.frameHash.title = typeof frame.sha256 === "string" ? frame.sha256 : "";
    put(elements.plannerTiming, msText(frame.timings?.planner_ms)); put(elements.sampleTiming, msText(frame.timings?.sample_ms)); put(elements.serverTiming, msText(frame.timings?.total_ms)); put(elements.roundtripTiming, msText(rtt)); put(elements.loadTiming, msText(frame.timings?.load_ms)); put(elements.presentTiming, msText(performance.now() - started));
    put(elements.messageCount, Array.isArray(frame.state?.messages) ? numberText(frame.state.messages.length) : "—");
    put(elements.sessionNotice, plainText(frame.state?.notice)); renderContext(frame.state?.messages);
    const returnedPlan = frame.plan && typeof frame.plan === "object" ? frame.plan : null;
    // Initial frames also carry a local creation record. Only provider-tagged
    // plans are evidence of an actual vision-planner response.
    const plan = typeof returnedPlan?.provider === "string" && returnedPlan.provider ? returnedPlan : null;
    const initialPlanned = !frame.agent && frame.initial_plan?.provider === "DeepSeek" && frame.initial_plan.planned === true;
    put(elements.planIntent, plainText(plan?.intent || (initialPlanned ? returnedPlan?.intent : null))); put(elements.planExpected, plainText(plan?.expected_change || (initialPlanned ? returnedPlan?.expected_change : null))); put(elements.planPrompt, plainText(plan?.next_prompt));
    const needsInput = (frame.recognition || plan?.recognition)?.status === "needs_input" || frame.agent?.status === "needs_input";
    put(elements.planEvidence, plan ? plan.should_update === false ? needsInput ? "视觉 AI 明确请求补充说明，本次保留原图，没有重新采样。" : "本次计划选择保留原图，没有重新采样。操作结果与原因见下方说明及实际任务回执。" : "本帧返回的视觉 AI 推断；结果仍需以实际画面判断。" : initialPlanned ? "DeepSeek 已根据完整应用说明规划首屏，下方保留实际交给图像模型的绘图描述；外观请以真实像素检查。" : frame.agent ? "本次 Agent 未得到有效界面计划。工具执行情况以任务回执为准。" : "这张首屏由本地图像模型直接生成，尚无视觉 AI 推断。");
    if (frame.last_event?.type === "create" && frame.initial_review && !frame.agent) {
      const review = frame.initial_review;
      const explanation = review.accepted === true
        ? "首屏主要类型与控件形状已由视觉 AI 检查；文字可读性与完整功能仍需实际判断。"
        : "首屏尚未通过结构检查，请查看实际画面和说明。";
      put(elements.planEvidence, explanation);
      if (review.accepted !== true) showNotice(plainText(frame.state?.notice), true);
    }
    if (!frame.agent && state.confirmedAgent) {
      put(elements.agentResultTitle, "先前 Agent 任务回执");
      put(elements.agentResultEvidence, "以下保留先前任务的真实工具回执，与当前画面操作无关；已保存的结果仍然有效。");
    }
    const edit = verifiedToggleEdit(frame) || verifiedRegionEdit(frame);
    const preview = verifiedRegionPreview(frame);
    if (preview) put(elements.planEvidence, "本帧是模型生成的待确认预览。区域外像素已锁定，区域内内容和文字尚未通过视觉核验；原文字已保留，可继续修改或返回。");
    put(elements.localEditEvidence, preview ? `待确认预览 · 外部像素变化 0\n区域内内容与文字未通过视觉核验\n区域内变化 ${numberText(preview.changed_pixels)} 个像素\n区域 [${preview.bounds.join(", ")}]，仅区域外像素一致性已核验。` : edit ? `局部编辑 · 外部像素变化 ${numberText(edit.outside_changed_pixels)}\n${edit.mode === "toggle_only" ? `轨道与滑块核验通过 · 目标状态：${edit.target_state === "on" ? "开启" : "关闭"}` : "局部区域视觉核验通过"}\n区域内变化 ${numberText(edit.changed_pixels)} 个像素\n区域 [${edit.bounds.join(", ")}]，以后台逐像素与状态核验回执为准。` : frame.edit ? "本帧局部编辑未提供完整的通过校验回执。" : "本帧没有局部编辑回执。");
    const scope = frame.scope;
    const scopeValid = scope?.decided_by === "DeepSeek" && ["local", "global", "none"].includes(scope.mode) && typeof scope.reason === "string" && scope.reason.trim();
    put(elements.scopeDecision, scopeValid ? `AI 选择：${{local:"局部更新",global:"全局重绘",none:"保持原图"}[scope.mode]}\n${scope.reason}` : "本帧没有新的 AI 自动范围选择回执。");
    if (!plan) put(elements.planPrompt, frame.agent ? "本次未生成绘图描述。" : plainText(frame.generation_prompt || frame.prompt));
    updateDisplayDimensions(); updateControls();
  }

  async function synchronizeSession({ quiet = false } = {}) {
    if (!state.frame) return;
    const sessionId = state.frame.session_id;
    const started = performance.now();
    const result = await requestJSON(`/api/live/session?id=${encodeURIComponent(sessionId)}`, { timeout: 30000 });
    await applyFrame(result.data, result.rtt, started, { expectedSessionId: sessionId });
    if (!quiet) addEvent("已同步当前帧", `当前版本 ${result.data.version}，未重复发送操作。`);
  }

  async function runOperation(label, operation) {
    if (state.busy) return false;
    state.busy = true; put(elements.activityText, label); showNotice(""); updateControls();
    try { return (await operation()) !== false; }
    catch (error) {
      if (state.feedbackActive) failFeedback(error);
      if (error?.status === 409 && String(error.code).toLowerCase() === "stale_frame" && state.frame) {
        showNotice("请求未完成，正在读取当前会话。此次操作不会自动重发。", true);
        addEvent("请求冲突，读取当前帧", "只同步，不自动重发点击或文字。", true);
        try { await synchronizeSession({ quiet: true }); showNotice("已同步当前会话。请根据最新画面决定下一次操作。"); }
        catch (syncError) { showNotice(syncError.message || "同步失败，已有画面已保留。", true); }
      } else { showNotice(error?.message || "请求未完成，已有画面已保留。", true); addEvent("请求未完成", error?.code ? `代码：${error.code}` : "当前画面已保留。", true); }
      return false;
    } finally { stopFeedbackTimer(); state.busy = false; if (elements.notice.textContent === busyInteractionNotice) showNotice(""); updateControls(); }
  }

  function ensurePlanner() {
    if (state.status?.planner?.configured === true) return true;
    showNotice("先在运行台填写 DeepSeek API Key，并选择“保存到本次运行”。首屏可以独立生成，点击后的视觉推断需要这份配置。", true);
    addEvent("等待视觉 AI 配置", "尚未发送画面或点击坐标。", true);
    elements.plannerKey.focus();
    showFeedback("尚未开始识图", "视觉 AI 配置尚未就绪，点击没有发送。请先保存本次运行的密钥。", { kind: "needs-input" });
    return false;
  }

  function editingAvailable() {
    return state.interactionMode === "full" || (state.interactionMode === "auto" ? state.status?.capabilities?.adaptive_edit === true && (!state.agentEnabled || state.status?.capabilities?.adaptive_agent === true) : state.status?.capabilities?.toggle_edit === true);
  }

  function ensureEditingCapability() {
    if (editingAvailable()) return true;
    showNotice("所选画面更新模式尚未加载，点击没有发送。请等待 APP 加载更新；也可主动选择整屏重绘。", true);
    showFeedback("识图更新尚未加载", "当前服务不支持所选模式，点击没有发送。请等待后台加载更新，或主动改用整屏重绘。", { kind: "needs-input" });
    return false;
  }

  function verifiedToggleEdit(frame) {
    const edit = frame?.edit;
    const bounds = edit?.bounds;
    if (!Number.isInteger(frame?.width) || !Number.isInteger(frame?.height) || frame.width < 1 || frame.height < 1) return null;
    if (edit?.mode !== "toggle_only" || edit.verified !== true || edit.outside_changed_pixels !== 0 || !Number.isInteger(edit.changed_pixels) || edit.changed_pixels < 1 || !["on", "off"].includes(edit.target_state) || !Array.isArray(bounds) || bounds.length !== 4 || !bounds.every(Number.isInteger)) return null;
    if (edit.pixel_state_verified !== true || edit.vision_verified !== true || !["on", "off"].includes(edit.before_state) || !["on", "off"].includes(edit.after_state) || edit.before_state === edit.after_state || edit.after_state !== edit.target_state || edit.current_state !== edit.before_state) return null;
    const [x0, y0, x1, y1] = bounds;
    if (x0 < 0 || y0 < 0 || x1 > frame.width || y1 > frame.height || x1 <= x0 || y1 <= y0 || edit.changed_pixels > (x1 - x0) * (y1 - y0)) return null;
    return edit;
  }

  function regionPixelEvidence(frame) {
    const edit = frame?.edit; const bounds = edit?.bounds;
    if (!Number.isInteger(frame?.width) || !Number.isInteger(frame?.height) || frame.width < 1 || frame.height < 1 || frame.width > 1024 || frame.height > 1024) return null;
    if (edit?.mode === "local_regions") {
      if (state.status?.capabilities?.behavior_regions !== true || edit.kind !== "regions" || edit.verified !== true || edit.outside_changed_pixels !== 0 || !Number.isInteger(edit.changed_pixels) || edit.changed_pixels < 1 || !Array.isArray(edit.regions) || edit.regions.length < 1 || edit.regions.length > 4) return null;
      const mask = new Uint8Array(frame.width * frame.height); let area = 0;
      const validBox = box => Array.isArray(box) && box.length === 4 && box.every(Number.isInteger) && box[0] >= 0 && box[1] >= 0 && box[2] <= frame.width && box[3] <= frame.height && box[2]-box[0] >= 2 && box[3]-box[1] >= 2 && (box[2]-box[0])*(box[3]-box[1])*100 <= frame.width*frame.height*35;
      if (!validBox(edit.trigger_bounds) || !edit.regions.every(validBox)) return null;
      for (const [x0,y0,x1,y1] of edit.regions) for (let y=y0;y<y1;y++) for (let x=x0;x<x1;x++) { const i=y*frame.width+x; if (!mask[i]) { mask[i]=1; area++; } }
      if (area*100 > frame.width*frame.height*35 || edit.edited_pixels !== area || edit.outside_pixels !== frame.width*frame.height-area || edit.changed_pixels > area || edit.verification !== "outside_pixels_exact" || !/^[a-f0-9]{64}$/i.test(edit.outside_sha256_before || "") || edit.outside_sha256_before !== edit.outside_sha256_after) return null;
      const envelope = [Math.min(...edit.regions.map(b=>b[0])),Math.min(...edit.regions.map(b=>b[1])),Math.max(...edit.regions.map(b=>b[2])),Math.max(...edit.regions.map(b=>b[3]))];
      if (!Array.isArray(bounds) || bounds.length !== 4 || !bounds.every((value,i)=>value===envelope[i])) return null;
      return edit;
    }
    if (edit?.mode !== "local_region" || edit.kind !== "region" || edit.verified !== true || edit.outside_changed_pixels !== 0 || !Number.isInteger(edit.changed_pixels) || edit.changed_pixels < 1 || !Array.isArray(bounds) || bounds.length !== 4 || !bounds.every(Number.isInteger)) return null;
    const [x0, y0, x1, y1] = bounds; const area = (x1-x0)*(y1-y0);
    if (x0 < 0 || y0 < 0 || x1 > frame.width || y1 > frame.height || x1-x0 < 2 || y1-y0 < 2 || area*100 > frame.width*frame.height*35 || edit.changed_pixels > area) return null;
    return edit;
  }

  function verifiedRegionEdit(frame) {
    const edit = regionPixelEvidence(frame);
    const policy=frame?.plan?.visual_intent_memory;
    const memory=state.status?.capabilities?.confirmed_local_text_memory===true&&policy?.full_image_hash_check===true
      &&policy.policy_source==='AI-origin-policy'&&policy.pixel_generator==='neural_text'&&policy.fresh_vision===false
      &&edit?.local_memory_verified===true&&edit.vision_verified===false&&edit.generator==='native_neural_text_head'
      &&edit.verification_source==='AI-origin-policy+full_image_hash+neural_text'
      &&['input','send'].includes(frame.plan.text_action)&&frame.plan.requires_tools===false
      &&edit.verification==='outside_pixels_exact'&&/^[a-f0-9]{64}$/i.test(edit.outside_sha256_before||'')
      &&edit.outside_sha256_before===edit.outside_sha256_after;
    return (edit?.vision_verified === true||memory) && edit.preview !== true ? edit : null;
  }

  function verifiedRegionPreview(frame) {
    if (state.status?.capabilities?.region_preview !== true) return null;
    const edit = regionPixelEvidence(frame), scope = frame?.scope;
    if (!edit || edit.preview !== true || edit.vision_verified !== false || frame.plan?.should_update !== true || scope?.mode !== "local" || scope.target_type !== "other" || scope.decided_by !== "DeepSeek" || typeof scope.reason !== "string" || !scope.reason.trim() || scope.reason.length > 300) return null;
    const [x0, y0, x1, y1] = edit.bounds;
    const outsidePixels = frame.width * frame.height - (edit.mode === "local_regions" ? edit.edited_pixels : (x1 - x0) * (y1 - y0));
    if (edit.verification !== "outside_pixels_exact" || edit.outside_pixels !== outsidePixels || !/^[a-f0-9]{64}$/i.test(edit.outside_sha256_before || "") || edit.outside_sha256_before !== edit.outside_sha256_after) return null;
    return edit;
  }

  function verifiedAutoFrame(frame, previous, event) {
    const scope = frame?.scope;
    const localPolicy=scope?.decided_by==='Local confirmed policy'&&scope.mode==='local'&&scope.target_type==='other'&&verifiedRegionEdit(frame)?.local_memory_verified===true;
    if ((scope?.decided_by !== "DeepSeek"&&!localPolicy) || typeof scope.reason !== "string" || !scope.reason.trim() || scope.reason.length > 300 || !["local", "global", "none"].includes(scope.mode) || !["toggle", "other", "uncertain"].includes(scope.target_type)) return false;
    if (frame.agent?.status === "preview" && !verifiedRegionPreview(frame)) return false;
    if (scope.mode === "none") return scope.target_type !== "toggle" && frame.plan?.should_update === false && !frame.edit && typeof previous.sha256 === "string" && frame.sha256 === previous.sha256;
    if (frame.plan?.should_update !== true || scope.target_type === "uncertain") return false;
    if (scope.mode === "global") return scope.target_type === "other" && !frame.edit;
    const edit = scope.target_type === "toggle" ? verifiedToggleEdit(frame) : verifiedRegionEdit(frame) || verifiedRegionPreview(frame);
    if (!edit) return false;
    const [x0, y0, x1, y1] = edit.mode === "local_regions" ? edit.trigger_bounds : edit.bounds;
    return event.x >= x0 && event.x < x1 && event.y >= y0 && event.y < y1;
  }

  async function sendEvent(event, detail) {
    if (state.busy) { explainBusyInteraction(); return false; }
    if (!state.frame) return false;
    feedbackContext(event);
    if (!state.online) {
      showNotice("本地服务未连接，点击没有发送。当前画面保持，连接恢复后再操作。", true);
      showFeedback("本地服务未连接", "点击没有发送，原图保持。连接恢复后可同步当前会话。", { kind: "error" });
      return;
    }
    // Text submitted without coordinates can still use adaptive global/none
    // planning. Toggle-only editing requires a clicked control, so standalone
    // text uses full mode when that particular selector is active.
    const clickMode = event.type === "back" || (event.type === "submit" && state.interactionMode === "toggle_only") ? "full" : state.interactionMode;
    const toggleMode = clickMode === "toggle_only";
    const autoMode = clickMode === "auto";
    if (event.type !== "back" && clickMode !== "full" && !ensureEditingCapability()) return;
    if (event.type !== "back" && !ensurePlanner()) return;
    const agentMode = state.agentEnabled && event.type !== "back" && !toggleMode;
    if (agentMode && !state.agentAvailable) { showNotice("Agent 工作区尚未就绪。请先只读刷新工作区，现有画面功能仍可使用。", true); showFeedback("Agent 工作区尚未就绪", "点击没有发送。请先只读刷新工作区，再决定下一步操作。", { kind: "needs-input" }); return; }
    if (agentMode && state.agentUncertain) { showNotice("上一次 Agent 结果仍待确认。请先选择“查询这次任务结果”，不要重复提交任务。", true); showFeedback("上次任务结果待确认", "本次点击没有重复发送。请先查询原任务的执行回执。", { kind: "needs-input" }); elements.queryAgentRequest.focus(); return; }
    if (typeof window.crypto?.randomUUID !== "function") { showNotice("当前浏览器无法生成请求标识，请使用支持本地安全上下文的浏览器。", true); return; }
    const frame = state.frame;
    const payload = { session_id: frame.session_id, version: frame.version, request_id: window.crypto.randomUUID(), event };
    if (event.type !== "back") payload.interaction_mode = clickMode;
    if (agentMode) {
      state.latestAgentRequestId = payload.request_id; state.agentUncertain = true;
      put(elements.agentPending, `正在等待请求 ${payload.request_id.slice(0, 12)}…；下方保留上次已确认的结果。`); elements.agentPending.hidden = false;
    }
    return await runOperation(event.type === "back" ? "正在返回上一帧" : agentMode ? "Agent 正在理解与执行任务" : "视觉 AI 推断 / 绘图中", async () => {
      const started = performance.now();
      beginFeedback(event, agentMode);
      addEvent(agentMode ? "已提交 Agent 任务" : event.type === "click" ? "已提交画面点击" : event.type === "submit" ? "已提交运行台文字" : "请求返回上一帧", detail);
      let result; let receivedAgent = false;
      try {
        result = await requestJSON(agentMode ? "/api/agent/event" : "/api/live/event", { method: "POST", body: payload, timeout: agentMode ? 240000 : 120000 });
        if (result.data.edit?.preview === true || result.data.agent?.status === "preview") {
          const preview = verifiedRegionPreview(result.data);
          const trigger = preview?.mode === "local_regions" ? preview.trigger_bounds : preview?.bounds;
          if (toggleMode || !preview || event.type !== "click" || event.x < trigger[0] || event.x >= trigger[2] || event.y < trigger[1] || event.y >= trigger[3]) throw new ApiError("待确认预览缺少完整的区域外像素核验，当前图片已保留。", 0, "scope_unverified");
        }
        const adaptiveFailed = autoMode && agentMode && ["render_failed", "partial_failure", "interrupted"].includes(result.data.agent?.status);
        if (adaptiveFailed && (result.data.sha256 !== frame.sha256 || result.data.edit)) throw new ApiError("任务未完整完成且画面回执不一致，原图已保留。请查询任务执行记录。", 0, "scope_unverified");
        if (autoMode && !adaptiveFailed && !verifiedAutoFrame(result.data, frame, event)) throw new ApiError("后台没有返回一致的 AI 范围选择及局部校验回执，当前画面已保留。", 0, "scope_unverified");
        if (toggleMode && result.data.plan?.should_update === false && (typeof frame.sha256 !== "string" || result.data.sha256 !== frame.sha256)) throw new ApiError("后台声明保留原图，但画面校验值不一致。当前画面已保留，请同步会话检查状态。", 0, "toggle_unverified");
        if (toggleMode && result.data.plan?.should_update !== false && !verifiedToggleEdit(result.data)) throw new ApiError("后台没有返回完整的局部编辑校验结果，当前画面已保留。请同步会话检查状态。", 0, "toggle_unverified");
        if (agentMode && result.data.agent) {
          const older = state.frame && result.data.session_id === state.frame.session_id && result.data.version < state.frame.version;
          if (!older) renderAgentResult(result.data.agent, { requestId: payload.request_id });
          receivedAgent = true; void refreshWorkspace();
        }
        await applyFrame(result.data, result.rtt, started, { expectedSessionId: frame.session_id });
        finishFrameFeedback(result.data);
      } catch (error) {
        if (agentMode && receivedAgent) {
          error.message = "Agent 回执已返回并保留，但画面未能读取。请查看实际执行记录与工作区，不要重复执行原任务。";
        } else if (agentMode && String(error.code).toLowerCase() === "agent_result_available") {
          // The business receipt exists even when its old session/image cache
          // does not. Retain this exact request ID for a read-only lookup.
          state.latestAgentRequestId = payload.request_id; state.agentUncertain = true;
          put(elements.agentPending, `请求 ${payload.request_id.slice(0, 12)}… 已有执行回执。请选择“查询这次任务结果”查看真实工具结果，不要重复执行。`); elements.agentPending.hidden = false;
          error.message = "服务器已有这次 Agent 的执行回执，但原画面可能不可恢复。请查询这次任务结果，先查看工具保存状态；不会自动重发任务。";
        } else if (agentMode) {
          const notExecuted = ["busy", "stale_frame", "frame_expired", "planner_not_configured", "invalid_event", "invalid_request", "invalid_session", "session_expired"].includes(String(error.code).toLowerCase());
          if (notExecuted) { state.agentUncertain = false; elements.agentPending.hidden = true; state.latestAgentRequestId = state.confirmedAgent?.request_id || null; }
          else { put(elements.agentPending, `请求 ${payload.request_id.slice(0, 12)}… 的结果尚未确认。先查询本次任务、刷新工作区，不要重复提交。`); error.message = `Agent 结果尚未确认；工具可能已经保存。请先查询这次任务结果并刷新工作区，不要重复提交。${error.message || ""}`; }
        }
        throw error;
      }
      if (agentMode) {
        const status = result.data.agent?.status;
        const message = status === "preview" ? `${previewLabel(result.data.agent)}已显示，内容未通过核验` : status === "render_failed" ? "画面更新未完成，请查看本次回执" : status === "partial_failure" || status === "interrupted" ? "Agent 返回部分执行结果" : status === "needs_input" ? "Agent 需要补充信息" : "Agent 结果已返回";
        addEvent(message, `请求 ${payload.request_id.slice(0, 12)}… · 以逐条工具结果和真实工作区为准。`, status === "render_failed" || status === "partial_failure" || status === "interrupted");
        if (status === "render_failed") showNotice(result.data.agent?.warning || "本次画面更新失败，原图已保留；任务情况请查看实际工具记录。", true);
        return !["render_failed", "partial_failure", "interrupted", "needs_input"].includes(status);
      }
      const noRedraw = result.data.plan?.should_update === false && event.type !== "back";
      const edit = verifiedToggleEdit(result.data) || verifiedRegionEdit(result.data);
      const preview = verifiedRegionPreview(result.data);
      const needsInput = (result.data.recognition || result.data.plan?.recognition)?.status === "needs_input";
      const samePixels = state.presentation?.changed === false;
      addEvent(preview ? "待确认预览已显示，内容未通过核验" : noRedraw ? needsInput ? "AI 请求补充说明，原图保留" : "操作结果已返回，原图保留" : (toggleMode || autoMode) && edit ? "局部编辑校验已通过" : event.type === "back" ? "已读取上一帧画面" : samePixels ? "绘图已完成，像素保持相同" : "真实画面已更新", `帧 ${result.data.version} · ${(toggleMode || autoMode) && (edit || preview) ? `外部像素变化 ${numberText((edit || preview).outside_changed_pixels)} · ` : ""}API 往返 ${msText(result.rtt)}`);
      return !needsInput;
    });
  }

  function contextInputFeedback(title, message, kind = "needs-input") {
    put(elements.contextInputStatus, message);
    showNotice(message, kind === "error");
    if (!state.feedbackActive) showFeedback(title, message, { kind });
  }

  function contextInputAvailable() {
    if (state.busy || state.contextSending) { contextInputFeedback("正在处理上一次操作", "正在识图或执行任务，本次说明没有发送；文字已保留，请等待结果。", "working"); return false; }
    if (!state.frame || !elements.liveImage.getAttribute("src")) { contextInputFeedback("请先创建画面", "当前没有可操作的图片，本次说明没有发送。", "error"); return false; }
    if (!state.online) { contextInputFeedback("本地服务未连接", "本地服务未连接，本次说明没有发送；文字已保留，连接恢复后再操作。", "error"); return false; }
    return true;
  }

  function nativeImagePoint(event) {
    const rect = elements.liveImage.getBoundingClientRect();
    if (!state.frame || !rect.width || !rect.height) return null;
    const clientX = knownNumber(event.clientX) ? event.clientX : rect.left + rect.width / 2;
    const clientY = knownNumber(event.clientY) ? event.clientY : rect.top + rect.height / 2;
    return {
      x: Math.max(0, Math.min(state.frame.width - 1, Math.floor((clientX - rect.left) / rect.width * state.frame.width))),
      y: Math.max(0, Math.min(state.frame.height - 1, Math.floor((clientY - rect.top) / rect.height * state.frame.height))), clientX, clientY
    };
  }

  function contextAnchorCurrent(anchor) {
    const frame = state.frame;
    return Boolean(anchor && frame && anchor.sessionId === frame.session_id && anchor.version === frame.version && anchor.width === frame.width && anchor.height === frame.height && anchor.sha256 === frame.sha256 && anchor.imageURL === frame.image_url && anchor.renderedSource === elements.liveImage.getAttribute("src"));
  }

  function updateContextInputControls() {
    const waiting = state.busy || state.contextSending;
    elements.contextInputText.disabled = waiting;
    elements.contextInputSend.disabled = waiting || !elements.contextInputText.value.trim();
    elements.contextInput.setAttribute("aria-busy", String(state.contextSending));
  }

  function positionContextInput() {
    if (elements.contextInput.hidden || !state.contextAnchor) return;
    // The temporary panel uses viewport coordinates, so page/container scrolling
    // cannot move it out of view or change the recorded native image coordinates.
    const margin = 12;
    const width = Math.max(1, window.innerWidth || 1024);
    const height = Math.max(1, window.innerHeight || 768);
    elements.contextInput.style.maxWidth = `${Math.max(1, width - margin * 2)}px`;
    elements.contextInput.style.maxHeight = `${Math.max(1, height - margin * 2)}px`;
    const rect = elements.contextInput.getBoundingClientRect();
    const panelWidth = Math.min(rect.width || 360, Math.max(1, width - margin * 2));
    const panelHeight = Math.min(rect.height || 250, Math.max(1, height - margin * 2));
    const left = Math.min(state.contextAnchor.clientX + 10, width - panelWidth - margin);
    const top = Math.min(state.contextAnchor.clientY + 10, height - panelHeight - margin);
    elements.contextInput.style.left = `${Math.max(Math.min(margin, width / 2), left)}px`;
    elements.contextInput.style.top = `${Math.max(Math.min(margin, height / 2), top)}px`;
  }

  function closeContextInput({ restoreFocus = false } = {}) {
    elements.contextInput.hidden = true;
    state.contextAnchor = null;
    state.contextComposing = false;
    if (restoreFocus) elements.liveImage.focus({ preventScroll: true });
  }

  function openContextInput(event) {
    event.preventDefault();
    if (!contextInputAvailable()) return false;
    const point = nativeImagePoint(event);
    if (!point) { contextInputFeedback("图片尚未显示", "图片显示尺寸尚未就绪，本次说明没有发送。", "error"); return false; }
    const frame = state.frame;
    state.contextAnchor = { ...point, sessionId: frame.session_id, version: frame.version, width: frame.width, height: frame.height, sha256: frame.sha256, imageURL: frame.image_url, renderedSource: elements.liveImage.getAttribute("src") };
    state.contextComposing = false;
    put(elements.contextInputPosition, `原图 (${point.x}, ${point.y}) · 第 ${frame.version} 帧`);
    put(elements.contextInputStatus, elements.contextInputText.value.trim() ? "上次文字已保留，可修改后回车发送；不会自动重复执行。" : "输入文字或操作说明后回车发送；关闭不会发送，文字会保留。");
    elements.contextInput.hidden = false;
    updateContextInputControls(); positionContextInput(); markClick(point.x, point.y);
    elements.contextInputText.focus({ preventScroll: true });
    return true;
  }

  async function submitContextInput() {
    if (state.contextComposing) return false;
    if (!contextInputAvailable()) return false;
    const anchor = state.contextAnchor;
    if (!contextAnchorCurrent(anchor)) { contextInputFeedback("点击位置已过期", "画面已变化，旧位置没有发送；文字已保留，请在最新画面重新右键选择位置。", "error"); return false; }
    const text = elements.contextInputText.value.trim();
    if (!text || text.length > 500) { contextInputFeedback("请填写操作说明", "请输入 1 到 500 字的操作说明，当前没有发送。"); return false; }
    state.contextSending = true; updateContextInputControls();
    put(elements.contextInputStatus, "正在结合完整页面与该位置理解说明，请等待结果；不会自动重发。");
    let success = false;
    try {
      markClick(anchor.x, anchor.y);
      success = await sendEvent({ type: "click", x: anchor.x, y: anchor.y, text, input_method: "contextmenu" }, `右键输入 · 原图 (${anchor.x}, ${anchor.y}) · 帧 ${anchor.version} · ${text.length > 80 ? `${text.slice(0, 80)}…` : text}`) === true;
    } finally {
      state.contextSending = false; updateContextInputControls();
    }
    if (state.contextAnchor === anchor) {
      if (success) { closeContextInput({ restoreFocus: true }); }
      else { put(elements.contextInputStatus, "本次未全部完成，文字已保留，未自动重试。请先查看识图结果或 Agent 回执；若画面已变，请重新右键选择位置。"); positionContextInput(); }
    }
    return success;
  }

  function markClick(x, y) {
    if (!state.frame) return;
    elements.clickMarker.style.left = `${(x + .5) / state.frame.width * 100}%`; elements.clickMarker.style.top = `${(y + .5) / state.frame.height * 100}%`; elements.clickMarker.hidden = false;
  }

  function clickPixels(x, y) {
    if (state.busy) { explainBusyInteraction(); return; }
    if (!state.frame) return;
    markClick(x, y);
    // An unsent/previous toolbar draft is not the instruction for every future
    // button click. Only explicit text submit and right-click entry attach text.
    void sendEvent({ type: "click", x, y, input_method: "pointer" }, `原图坐标 (${x}, ${y}) · 帧 ${state.frame.version}`);
  }

  elements.sessionForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.busy || elements.createSession.disabled) return;
    const prompt = elements.initialPrompt.value.trim(); const steps = Number(elements.samplingSteps.value); const seed = Number(elements.sessionSeed.value);
    const planned = state.status?.capabilities?.initial_planning === true;
    if (!prompt || encoder.encode(prompt).length > (planned ? 6000 : 190) || Array.from(prompt).length > 2000) { showNotice(planned ? "应用说明需要在 2000 字、6000 UTF-8 字节以内，内容没有截断。" : "当前后台只支持 190 个 UTF-8 字节的短描述，完整说明已保留；请等待后台更新。", true); return; }
    const supportedSteps = [4, 8, 16, ...(Array.isArray(state.status?.capabilities?.sampling_steps) && state.status.capabilities.sampling_steps.includes(32) ? [32] : [])];
    if (!supportedSteps.includes(steps) || !Number.isInteger(seed) || seed < 0 || seed > 2147483647) { showNotice("请选择后台支持的采样步数，并填写 0 到 2147483647 的整数种子。", true); return; }
    if (planned && !ensurePlanner()) return;
    void runOperation(planned ? "AI 正在规划应用并生成首屏" : "本地图像模型正在生成首屏", async () => {
      const started = performance.now(); const body = { steps, seed, prompt }; if (planned) { body.plan_initial = true; if (state.status?.capabilities?.initial_verification === true) body.verify_initial = true; }
      beginFeedback({ type: "create" }, false);
      // Planning, one caption compression, two bounded reviews and their
      // compressions can consume 120 seconds before local drawing/loading.
      const result = await requestJSON("/api/live/session", { method: "POST", body, timeout: planned ? 240000 : 120000 });
      await applyFrame(result.data, result.rtt, started); finishFrameFeedback(result.data); addEvent("已创建真实像素会话", `原生 ${result.data.width} × ${result.data.height} · ${steps} 步`);
    });
  });

  elements.plannerForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.busy || elements.savePlanner.disabled) return;
    let endpoint;
    try {
      endpoint = new URL(elements.plannerEndpoint.value.trim());
      if (endpoint.protocol !== "https:" || endpoint.hostname !== "api.deepseek.com" || endpoint.port || endpoint.username || endpoint.password || endpoint.search || endpoint.hash || !["", "/", "/v1", "/v1/"].includes(endpoint.pathname)) throw new Error();
    } catch { showNotice("此版本仅接受 https://api.deepseek.com 或其 /v1 官方地址。密钥尚未发送。", true); return; }
    const model = elements.plannerModel.value.trim();
    if (model !== "deepseek-flash") { showNotice("此版本使用支持图片输入的 deepseek-flash，请检查模型名称。", true); elements.plannerModel.focus(); return; }
    const apiKey = elements.plannerKey.value.trim();
    if (!apiKey) { showNotice("请在密码框中填写 DeepSeek API Key。", true); elements.plannerKey.focus(); return; }
    void runOperation("保存本次运行配置", async () => {
      let configured;
      try { configured = await requestJSON("/api/live/config", { method: "POST", body: { base_url: endpoint.href.replace(/\/$/, ""), model, api_key: apiKey }, timeout: 15000 }); }
      catch (error) { if (typeof error.message === "string") error.message = error.message.split(apiKey).join("[已隐藏]"); throw error; }
      elements.plannerKey.value = ""; state.plannerDirty = false;
      if (configured.data && typeof configured.data.configured === "boolean") renderStatus({ ...(state.status || {}), planner: configured.data });
      showNotice("配置已保存到本次服务运行。下一次点击或文字提交时，才会实际请求 DeepSeek。");
      addEvent("已保存视觉 AI 配置", "仅保存本次运行配置，尚未验证远端连接。");
      await pollStatus();
    });
  });

  elements.eventForm.addEventListener("submit", (event) => { event.preventDefault(); if (state.busy) { explainBusyInteraction(); return; } const text = elements.eventText.value.trim(); if (!text || elements.submitText.disabled) return; void sendEvent({ type: "submit", text, input_method: "keyboard" }, text.length > 80 ? `${text.slice(0, 80)}…` : text); });
  elements.liveImage.addEventListener("click", (event) => {
    if (state.busy) { explainBusyInteraction(); return; }
    if (!state.frame) return;
    const point = nativeImagePoint(event);
    if (!point) { showNotice("图片显示尺寸尚未就绪，点击没有发送。请等待图片显示后再操作。", true); return; }
    clickPixels(point.x, point.y);
  });
  elements.liveImage.addEventListener("contextmenu", openContextInput);
  elements.contextInputForm.addEventListener("submit", (event) => { event.preventDefault(); return submitContextInput(); });
  elements.contextInputText.addEventListener("input", updateContextInputControls);
  elements.contextInputText.addEventListener("compositionstart", () => { state.contextComposing = true; });
  elements.contextInputText.addEventListener("compositionend", () => { state.contextComposing = false; });
  elements.contextInputText.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229 && !state.contextComposing) { event.preventDefault(); if (!event.repeat) return submitContextInput(); }
  });
  elements.contextInputClose.addEventListener("click", () => closeContextInput({ restoreFocus: true }));
  window.addEventListener("keydown", (event) => { if (event.key === "Escape" && !elements.contextInput.hidden) { event.preventDefault(); closeContextInput({ restoreFocus: true }); } });
  window.addEventListener("pointerdown", (event) => {
    state.contextDismissTarget = null;
    if (!elements.contextInput.hidden && !elements.contextInput.contains(event.target)) { state.contextDismissTarget = event.target; closeContextInput(); }
  });
  window.addEventListener("click", (event) => {
    const dismissedTarget = state.contextDismissTarget; state.contextDismissTarget = null;
    // Consume the click that dismissed the panel, including clicks on the image;
    // closing the explanation must never double as another application action.
    if (dismissedTarget && (dismissedTarget === event.target || dismissedTarget.contains?.(event.target))) {
      event.preventDefault(); event.stopPropagation();
      contextInputFeedback("输入说明已关闭", "已关闭输入说明，文字已保留。这次只关闭面板，没有执行画面操作；请再点击要操作的按钮。", "complete");
    }
  }, true);
  window.addEventListener("scroll", positionContextInput, true);
  window.addEventListener("resize", positionContextInput);
  elements.samplingSteps.addEventListener("change", updateControls);
  elements.liveImage.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); if (!event.repeat && state.frame) clickPixels(Math.floor(state.frame.width / 2), Math.floor(state.frame.height / 2)); } });
  elements.syncSession.addEventListener("click", () => { if (!elements.syncSession.disabled) void runOperation("读取当前会话", () => synchronizeSession()); });
  elements.loadLatestModel.addEventListener("click", () => {
    if (elements.loadLatestModel.disabled || elements.loadLatestModel.hidden) return;
    void runOperation("正在加载最新已保存模型", async () => {
      const result = await requestJSON("/api/live/model", { method: "POST", body: {}, timeout: 120000 });
      if (result.data?.available !== true || !knownNumber(result.data?.snapshot?.step) || typeof result.data?.snapshot?.sha256 !== "string" || !result.data.snapshot.sha256) throw new ApiError("模型更新回执不完整，当前图片和输入已保留。请等待模型状态刷新后确认结果。");
      renderStatus(result.data);
      showNotice(`APP 当前使用 ${numberText(result.data?.snapshot?.step)} 步模型。当前图片和输入已保留；创建新会话即可使用该模型。`);
      addEvent("推理模型已更新", `当前模型 ${numberText(result.data?.snapshot?.step)} 步，没有重新生成画面或执行业务任务。`);
    });
  });
  elements.backButton.addEventListener("click", () => { if (!elements.backButton.disabled) void sendEvent({ type: "back" }, "运行台操作：请求读取上一帧。"); });
  elements.agentToggle.addEventListener("change", () => {
    if (elements.agentToggle.disabled) { elements.agentToggle.checked = state.agentEnabled; return; }
    state.agentEnabled = elements.agentToggle.checked;
    elements.eventText.placeholder = state.agentEnabled ? "例如：创建一份项目笔记，再添加两个待办。" : "例如：我想打开设置；或在这里写要发送的消息。";
    addEvent(state.agentEnabled ? "已开启 Agent 任务模式" : "已关闭 Agent 任务模式", state.agentEnabled ? "仅操作本 APP 的文档、笔记与待办；相关工作区内容会发送给 DeepSeek。" : "后续操作仅走画面理解与绘图，不执行 Agent 工具。");
    updateControls();
  });
  elements.interactionMode.addEventListener("change", () => {
    if (state.busy || elements.interactionMode.disabled) { elements.interactionMode.value = state.interactionMode; return; }
    const selected = elements.interactionMode.value;
    if (!["auto", "toggle_only", "full"].includes(selected)) { elements.interactionMode.value = state.interactionMode; return; }
    state.interactionMode = selected;
    addEvent(selected === "auto" ? "已选择 AI 自动判断范围" : selected === "toggle_only" ? "已选择只改开关" : "已选择整屏重绘", selected === "auto" ? "每次点击由 AI 选择局部、全局或保留原图；界面展示实际选择回执。" : selected === "toggle_only" ? "只允许开关局部修改，区域外像素锁定。" : "后续点击可能改变整个画面；Agent 模式可执行工作区工具。");
    updateControls();
  });
  elements.refreshWorkspace.addEventListener("click", () => { if (!elements.refreshWorkspace.disabled) void refreshWorkspace(); });
  elements.clarifyFeedback.addEventListener("click", () => {
    if (state.busy || elements.clarifyFeedback.disabled) return;
    elements.eventText.focus();
    showNotice(state.agentUncertain ? "先查询原任务回执确认执行情况。此处只补充说明，不会自动重放原任务。" : "请在下方写明要操作的控件或希望发生的变化。补充说明不会自动发送，确认后再提交。");
  });
  elements.queryAgentRequest.addEventListener("click", () => { if (!elements.queryAgentRequest.disabled) void queryAgentResult(); });
  elements.initialPrompt.addEventListener("input", updateControls); elements.eventText.addEventListener("input", updateControls);
  for (const element of [elements.plannerEndpoint, elements.plannerModel, elements.plannerKey]) element.addEventListener("input", () => { state.plannerDirty = true; });
  window.addEventListener("resize", updateDisplayDimensions);
  window.addEventListener("beforeunload", () => { state.destroyed = true; clearTimeout(state.statusTimer); stopFeedbackTimer(); clearTimeout(state.frameHighlightTimer); elements.pixelSurface.classList.remove("frame-updated"); elements.plannerKey.value = ""; });
  updateControls(); void pollStatus();
})();
