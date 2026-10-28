'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const numberFormat = new Intl.NumberFormat('zh-CN');
  const timeFormat = new Intl.DateTimeFormat('zh-CN', {timeZone: 'Asia/Shanghai', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false});
  const activeStates = new Set(['running', 'preparing', 'starting', 'initializing', 'pausing', 'stopping']);
  const stoppedStates = new Set(['paused', 'complete', 'completed', 'stopped', 'failed', 'error']);
  const allowedTrainingData = new Set(['current', 'v2', 'native512', 'interaction', 'curriculum','applications','app_flows']);
  const stateNames = {running: '训练进行中', preparing: '正在准备训练', starting: '正在启动训练', initializing: '正在初始化模型', paused: '训练已暂停', complete: '训练阶段完成', completed: '训练阶段完成', stopped: '训练已停止', failed: '训练中断', error: '训练中断', idle: '等待开始训练', pausing: '正在保存并暂停', stopping: '正在保存并暂停'};
  let snapshot = null;
  let connected = false;
  let polling = false;
  let actionBusy = false;
  let pausePending = false;
  let startPendingAt = 0;
  let targetTouched = false;
  let batchTouched = false;
  let accumTouched = false;
  let networkErrorShown = false;
  let previewKey = '';
  let lastLogText = '';
  let history = [];

  const numeric = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = value => numeric(value) ? numberFormat.format(value) : '—';
  const setText = (id, value) => { $(id).textContent = String(value); };
  function captionTrainingLabel(value) {
    if (!numeric(value) || value <= 0 || value > 1) return '原始描述';
    const natural = Math.round(value * 1000) / 10;
    const original = Math.round((1 - value) * 1000) / 10;
    return `${fmt(natural)}% 自然表达 / ${fmt(original)}% 原始描述`;
  }
  function duration(value) {
    if (!numeric(value) || value < 0) return '—';
    const seconds = Math.floor(value);
    if (seconds < 60) return `${seconds} 秒`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes} 分 ${seconds % 60} 秒`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours} 小时 ${minutes % 60} 分`;
    return `${Math.floor(hours / 24)} 天 ${hours % 24} 小时`;
  }
  function bytes(value) {
    if (!numeric(value)) return '—';
    if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`;
    if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`;
    return `${(value / 1024).toFixed(1)} KB`;
  }
  function date(value) {
    if (value === null || value === undefined || value === '') return '—';
    let candidate = value;
    if (typeof candidate === 'number' && candidate < 1e12) candidate *= 1000;
    const result = new Date(candidate);
    return Number.isNaN(result.getTime()) ? '—' : timeFormat.format(result);
  }
  function toast(message, error = false) {
    const node = document.createElement('div');
    node.className = `toast${error ? ' error' : ''}`;
    node.textContent = String(message);
    $('toast-region').append(node);
    while ($('toast-region').children.length > 3) $('toast-region').firstElementChild.remove();
    window.setTimeout(() => node.remove(), error ? 8500 : 5000);
  }
  async function request(path, options = {}) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), options.method ? 20000 : 8000);
    try {
      const response = await fetch(path, {cache: 'no-store', ...options, signal: controller.signal, headers: {'Accept': 'application/json', ...(options.headers || {})}});
      let data;
      try { data = await response.json(); } catch (_) { throw new Error(`服务返回了无法读取的响应（${response.status}）。`); }
      if (!response.ok || data.ok === false) throw new Error(data.error || data.message || `请求失败（${response.status}）。`);
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('请求超时，请查看实时状态确认操作结果后再重试。');
      throw error;
    } finally { window.clearTimeout(timer); }
  }

  function updateButtons() {
    const training = snapshot?.training || {};
    const active = activeStates.has(training.state);
    const nativeControlPending = training.native_resolution === 512 && snapshot?.capabilities?.native_training_resolution !== true;
    const waiting = Boolean(startPendingAt);
    const dataOption = $('training-data').selectedOptions[0];
    const dataReady = Boolean(dataOption && allowedTrainingData.has(dataOption.value) && !dataOption.disabled);
    $('start-button').disabled = !connected || active || actionBusy || pausePending || waiting || !dataReady || nativeControlPending;
    $('pause-button').disabled = !connected || !active || actionBusy || pausePending || waiting;
    setText('start-label', actionBusy && !pausePending || waiting ? '正在启动…' : '继续训练');
    setText('pause-label', pausePending ? '正在保存并暂停…' : '保存并暂停');
    for (const id of ['target-steps', 'batch-size', 'accum', 'training-data']) $(id).disabled = actionBusy || waiting || active || pausePending || (id === 'training-data' && !dataReady);
    if (pausePending) setText('control-hint', '暂停请求已发出。正在等待当前更新完成并写入检查点，请以实际训练状态为准。');
    else if (waiting) setText('control-hint', '启动请求已发出，正在等待训练进程报告状态。');
    else if (nativeControlPending) setText('control-hint', active ? '512 训练正在运行；继续训练控制需重启 APP 加载更新。保存并暂停仍可使用。' : '512 训练当前未运行；继续训练控制需重启 APP 加载更新，以保持原生 512 分辨率。');
    else if (active) setText('control-hint', '训练正在运行。需要调整参数时，先保存并暂停，再继续训练。');
    else setText('control-hint', '开始时默认恢复最近检查点；暂停会等待当前更新完成并保存。');
  }

  function updateData(data) {
    const select = $('training-data');
    const previous = select.value;
    const training = snapshot?.training || {};
    const reportedSize = numeric(data.native_resolution) ? data.native_resolution : numeric(data.native_size) ? data.native_size : null;
    const currentTrainingData = typeof data.path === 'string' && data.path === training.data_path;
    const currentNativeSize = reportedSize || (currentTrainingData && numeric(training.native_resolution) ? training.native_resolution : null);
    const native512 = currentNativeSize === 512;
    const available = Array.isArray(data.available) ? data.available.filter(item => item && allowedTrainingData.has(item.id)) : [];
    const signature = JSON.stringify([currentNativeSize, available.map(item => [item.id, item.label, item.ready, item.available, item.status])]);
    if (select.dataset.signature !== signature) {
      select.replaceChildren();
      for (const item of available) {
        const option = document.createElement('option');
        option.value = item.id;
        const ready = item.ready !== false && item.available !== false && !['preparing', 'building', 'missing', 'unavailable'].includes(item.status);
        option.disabled = !ready;
        option.textContent = `${item.label || ({current: '当前训练数据', v2: '第二版训练数据', native512: '原生 512 × 512'}[item.id])}${item.id === 'current' && native512 ? ' · 原生 512 × 512' : ''}${ready ? '' : '（尚未就绪）'}`;
        select.append(option);
      }
      const stillAvailable = [...select.options].some(option => option.value === previous && !option.disabled);
      const nextOption = [...select.options].find(option => !option.disabled);
      if (stillAvailable) select.value = previous;
      else if (nextOption) select.value = nextOption.value;
      else { const option = new Option('暂无可用训练数据', ''); option.disabled = true; option.selected = true; select.append(option); }
      select.dataset.signature = signature;
    }
    setText('data-help', numeric(data.count) ? `当前数据共 ${fmt(data.count)} 张。${native512 ? '当前训练使用原生 512 × 512 数据。' : ''}仅显示服务确认可用的数据。` : '由本地数据准备状态决定可选项。');
    setText('data-version', native512 ? '原生 512 × 512' : data.version || '—');
    setText('data-counts', `${fmt(data.train_count)} / ${fmt(data.val_count)}`);
    setText('data-path', data.path || '');
  }

  function updatePreview(preview) {
    const validURL = typeof preview.url === 'string' && preview.url.startsWith('/api/training-preview.png');
    if (!validURL) {
      $('preview-link').hidden = true;
      $('preview-empty').hidden = false;
      setText('preview-step', '暂无预览');
      return;
    }
    const key = `${preview.url}|${preview.step}`;
    if (key !== previewKey) {
      const source = `${preview.url}${preview.url.includes('?') ? '&' : '?'}saved_step=${encodeURIComponent(preview.step ?? '')}`;
      $('preview-image').src = source;
      $('preview-link').href = source;
      previewKey = key;
    }
    $('preview-link').hidden = false;
    $('preview-empty').hidden = true;
    setText('preview-step', numeric(preview.step) ? `第 ${fmt(preview.step)} 步` : '最近保存');
    setText('preview-caption', `最近保存的 2 × 2 模型采样${numeric(preview.step) ? `，对应第 ${fmt(preview.step)} 步` : ''}；点击图片查看原图。`);
    $('preview-prompts').replaceChildren();
    for (const prompt of Array.isArray(preview.prompts) ? preview.prompts.slice(0, 4) : []) {
      const li = document.createElement('li'); li.textContent = String(prompt); $('preview-prompts').append(li);
    }
  }

  function updateCheckpoints(files) {
    const rows = $('checkpoint-rows'); rows.replaceChildren();
    const list = Array.isArray(files) ? files : [];
    setText('checkpoint-count', `${list.length} 个文件`);
    if (!list.length) { const row = rows.insertRow(); const cell = row.insertCell(); cell.colSpan = 4; cell.className = 'table-empty'; cell.textContent = '尚无已保存的检查点'; return; }
    for (const file of list) {
      const row = rows.insertRow(); const name = row.insertCell();
      const label = document.createElement('span'); label.className = 'file-name'; label.textContent = file.name || '—'; name.append(label);
      const type = document.createElement('span'); type.className = 'file-type';
      const types = {'resumable-training': '可恢复训练', 'inference-ema': 'EMA 采样权重', training: '训练检查点', inference: '采样权重'};
      type.textContent = types[file.type] || file.type || '检查点'; name.append(type);
      row.insertCell().textContent = fmt(file.step);
      row.insertCell().textContent = bytes(file.size_bytes);
      row.insertCell().textContent = date(file.modified_at);
    }
  }

  function svgNode(tag, attrs = {}, text = null) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
    if (text !== null) node.textContent = text;
    return node;
  }
  function drawChart() {
    const chart = $('loss-chart'); chart.replaceChildren();
    const range = Number($('chart-range').value);
    const lastStep = history.length ? history[history.length - 1].step : 0;
    let points = range ? history.filter(point => point.step >= lastStep - range) : history;
    const smooth = $('smooth-loss').checked;
    if (smooth && points.length) {
      let current = points[0].loss;
      points = points.map(point => { current = current * .7 + point.loss * .3; return {...point, loss: current}; });
    }
    $('chart-empty').hidden = points.length > 0;
    setText('chart-caption', `${points.length ? `${fmt(points.length)} 个真实记录点` : '仅显示后端实际记录'}${smooth ? ' · 仅曲线使用指数平滑' : ''}`);
    setText('chart-last', points.length ? `最近 ${fmt(points[points.length - 1].step)} 步` : '—');
    const left = 56, right = 818, top = 17, bottom = 213;
    const losses = points.map(point => point.loss);
    let ymin = losses.length ? Math.min(...losses) : 0, ymax = losses.length ? Math.max(...losses) : 1;
    const spread = ymax - ymin;
    ymin = Math.max(0, ymin - (spread || Math.max(ymax * .1, .01)) * .15);
    ymax += (spread || Math.max(ymax * .1, .01)) * .15;
    const xmin = points.length ? points[0].step : 0, xmax = points.length ? Math.max(xmin + 1, points[points.length - 1].step) : 1;
    for (let i = 0; i < 5; i++) {
      const yy = top + (bottom - top) * i / 4;
      chart.append(svgNode('line', {x1: left, y1: yy, x2: right, y2: yy, stroke: '#e9ece5', 'stroke-width': 1}));
      const value = ymax - (ymax - ymin) * i / 4;
      chart.append(svgNode('text', {x: left - 11, y: yy + 4, fill: '#90978a', 'font-size': 10, 'text-anchor': 'end'}, value.toFixed(value < .01 ? 4 : 3)));
    }
    if (!points.length) return;
    for (let i = 0; i < 5; i++) {
      const xx = left + (right - left) * i / 4;
      chart.append(svgNode('text', {x: xx, y: 239, fill: '#90978a', 'font-size': 10, 'text-anchor': 'middle'}, fmt(Math.round(xmin + (xmax - xmin) * i / 4))));
    }
    const coords = points.map(point => [left + (point.step - xmin) / (xmax - xmin) * (right - left), bottom - (point.loss - ymin) / (ymax - ymin) * (bottom - top)]);
    chart.append(svgNode('polyline', {points: coords.map(point => point.join(',')).join(' '), fill: 'none', stroke: '#253022', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}));
    const last = coords[coords.length - 1];
    chart.append(svgNode('circle', {cx: last[0], cy: last[1], r: 3.5, fill: '#287b50', stroke: '#fff', 'stroke-width': 1.5}));
  }

  function render(data) {
    snapshot = data;
    const training = data.training || {}, model = data.model || {}, gpu = data.gpu || {};
    const state = training.state || 'idle';
    const active = activeStates.has(state);
    if (pausePending && stoppedStates.has(state)) { pausePending = false; toast(state === 'failed' ? '训练已中断，请查看错误与日志。' : '训练状态已更新，检查点与日志以实际保存记录为准。', state === 'failed'); }
    if (startPendingAt && (active || state === 'failed' || state === 'error')) startPendingAt = 0;
    if (startPendingAt && Date.now() - startPendingAt > 25000) { startPendingAt = 0; toast('尚未检测到训练启动，请查看日志后确认。', true); }
    const shownName = pausePending ? '正在保存并暂停' : startPendingAt ? '正在等待训练启动' : stateNames[state] || `训练状态：${state}`;
    setText('state-title', shownName);
    setText('state-badge', pausePending ? '等待保存' : active ? '运行中' : state === 'failed' || state === 'error' ? '需检查' : state === 'paused' ? '已暂停' : state === 'stopped' ? '已停止' : state === 'complete' || state === 'completed' ? '阶段结束' : '待运行');
    $('state-badge').className = `badge${active ? ' active' : ''}`;
    $('state-dot').className = `state-dot${active ? ' active' : state === 'failed' || state === 'error' ? ' error' : ''}`;
    setText('state-note', pausePending ? '正在等待实际暂停状态，请勿关闭仍在保存的训练进程。' : active ? '模型正在本地学习；预览和检查点仅在实际保存后更新。' : state === 'failed' || state === 'error' ? '请先查看错误与最近日志，再决定是否继续。' : '继续训练将恢复最近检查点。累计训练步数并不代表生成质量。');
    setText('step-value', fmt(training.step)); setText('target-value', fmt(training.total_steps));
    const progress = numeric(training.step) && numeric(training.total_steps) && training.total_steps > 0 ? Math.max(0, Math.min(100, training.step / training.total_steps * 100)) : 0;
    $('progress-fill').style.width = `${progress}%`; $('main-progress').setAttribute('aria-valuenow', progress.toFixed(1));
    setText('progress-text', numeric(training.total_steps) && training.total_steps > 0 ? `${progress.toFixed(1)}% · 累计训练进度` : '等待进度');
    setText('elapsed-value', duration(training.elapsed_seconds)); setText('remaining-value', active ? duration(training.estimated_remaining_seconds) : state === 'complete' || state === 'completed' ? '阶段完成' : '—');
    setText('loss-value', numeric(training.loss) ? training.loss.toFixed(5) : '—');
    setText('samples-value', fmt(training.samples_seen));
    const batch = training.batch_size, accum = training.gradient_accumulation;
    setText('batch-foot', numeric(batch) && numeric(accum) ? `批大小 ${batch} × 累积 ${accum} = ${batch * accum}` : '有效批次 —');
    setText('gpu-util-value', numeric(gpu.utilization_percent) ? `${Math.round(gpu.utilization_percent)}%` : '—');
    setText('gpu-name', gpu.name || '暂无 GPU 信息');
    setText('gpu-memory-value', numeric(gpu.used_mb) ? `${(gpu.used_mb / 1024).toFixed(1)} GB` : '—');
    setText('gpu-memory-foot', numeric(gpu.total_mb) ? `总显存 ${(gpu.total_mb / 1024).toFixed(1)} GB` : '暂无显存信息');
    setText('gpu-temperature', numeric(gpu.temperature_c) ? `${Math.round(gpu.temperature_c)} °C` : '—');
    setText('gpu-power', numeric(gpu.power_w) ? `${Math.round(gpu.power_w)} W` : '—');
    const memoryRatio = numeric(gpu.used_mb) && numeric(gpu.total_mb) && gpu.total_mb > 0 ? Math.min(100, gpu.used_mb / gpu.total_mb * 100) : 0;
    $('memory-fill').style.width = `${memoryRatio}%`;
    setText('memory-caption', numeric(gpu.used_mb) && numeric(gpu.total_mb) ? `${fmt(Math.round(gpu.used_mb))} / ${fmt(Math.round(gpu.total_mb))} MB · ${memoryRatio.toFixed(0)}%` : '暂无 GPU 显存信息');
    const parameters = training.parameters ?? model.parameters;
    setText('parameters-value', numeric(parameters) ? `${(parameters / 1e6).toFixed(2)} M` : '—');
    setText('device-value', training.device || '—');
    setText('resolution-value', numeric(training.native_resolution) ? `${training.native_resolution} × ${training.native_resolution}` : numeric(model.native_width) ? `${model.native_width} × ${model.native_height || model.native_width}` : '—');
    setText('pid-value', training.pid || '—');
    setText('caption-training-value', captionTrainingLabel(training.caption_augmentation_probability));
    const stateSampling = training.interaction_fraction;
    const flowSampling=training.app_flow_fraction;
    setText('state-sampling-value', numeric(flowSampling)&&flowSampling>0 ? `应用流程 ${Math.round(flowSampling*100)}% · 原交互 ${Math.round((stateSampling||0)*100)}%` : numeric(stateSampling) && stateSampling >= 0 && stateSampling <= 1 ? `${Math.round(stateSampling * 100)}%` : stateSampling === null ? '均匀采样' : '未报告');
    setText('region-training-value', `文字 ${numeric(training.text_roi_weight) ? training.text_roi_weight : '—'} / 状态 ${numeric(training.state_roi_weight) ? training.state_roi_weight : '—'}`);
    $('caption-training-note').hidden = !numeric(training.caption_augmentation_probability) || training.caption_augmentation_probability <= 0 || training.caption_augmentation_probability > 1;
    $('training-error').hidden = !training.error; setText('training-error', training.error || '');
    // Follow real configuration changes such as 256 -> 512 until the user has
    // edited a field. A page left open must not resume with an obsolete batch.
    if (!targetTouched) $('target-steps').value = numeric(training.total_steps) && training.total_steps > (training.step || 0) ? training.total_steps : Math.max(10000, (training.step || 0) + 5000);
    if (!batchTouched && numeric(batch)) $('batch-size').value = batch;
    if (!accumTouched && numeric(accum)) $('accum').value = accum;
    updateEffectiveBatch();
    $('target-steps').min = String(Math.max(1, (numeric(training.step) ? training.step : 0) + 1));
    updateData(data.data || {}); updatePreview(data.preview || {}); updateCheckpoints(data.checkpoints);
    const rows = new Map();
    for (const row of Array.isArray(data.loss_history) ? data.loss_history : []) if (numeric(row.step) && numeric(row.loss)) rows.set(row.step, row);
    history = [...rows.values()].sort((a, b) => a.step - b.step); drawChart();
    const logText = (Array.isArray(data.logs) ? data.logs.slice(-30) : []).map(line => typeof line === 'string' ? line : JSON.stringify(line)).join('\n') || '暂无训练日志。';
    if (logText !== lastLogText) { const box = $('training-logs'); const atEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 45; box.textContent = logText; if (atEnd || !lastLogText) box.scrollTop = box.scrollHeight; lastLogText = logText; }
    const validation = data.validation;
    $('validation-details').hidden = !validation;
    setText('validation-state', validation ? numeric(validation.step ?? validation.checkpoint_step) ? `第 ${fmt(validation.step ?? validation.checkpoint_step)} 步` : '已有记录' : '尚无记录');
    setText('validation-summary', validation ? validation.message || validation.summary || '已有验证记录。指标仅供诊断，仍需查看实际生成图像判断质量。' : '训练损失和验证指标都需要结合真实生成图像判断。');
    setText('validation-json', validation ? JSON.stringify(validation, null, 2) : '');
    const stateValidation = data.state_validation;
    $('state-validation-details').hidden = !stateValidation;
    const margin = value => numeric(value) ? `${value >= 0 ? '+' : ''}${value.toFixed(6)}` : '—';
    setText('state-validation-summary', stateValidation
      ? `第 ${fmt(stateValidation.checkpoint_step)} 步 · ${fmt(stateValidation.pairs)} 对状态样本。原描述差值 ${margin(stateValidation.state_roi?.wrong_minus_correct_mse)}，自然描述差值 ${margin(stateValidation.state_roi?.natural_wrong_minus_natural_mse)}，文字区域差值 ${margin(stateValidation.text_roi?.natural_wrong_minus_natural_mse)}。差值是错误描述损失减正确描述损失；负值表示该项未显示有效状态区分，正值也不代表画面或文字合格。`
      : '尚无同类相反状态的独立验证记录。');
    setText('state-validation-json', stateValidation ? JSON.stringify(stateValidation, null, 2) : '');
    setText('last-update', `同步于 ${date(Date.now())}`);
    updateButtons();
  }

  async function poll() {
    if (polling) return;
    polling = true;
    try {
      const data = await request('/api/training/status');
      connected = true; networkErrorShown = false;
      $('connection').className = 'connection online'; $('connection').lastChild.textContent = '本地服务已连接';
      render(data);
    } catch (error) {
      connected = false;
      $('connection').className = 'connection offline'; $('connection').lastChild.textContent = '连接中断';
      setText('last-update', '连接中断 · 保留最近一次状态');
      if (!networkErrorShown) { toast(`无法同步训练状态：${error.message}`, true); networkErrorShown = true; }
      updateButtons();
    } finally { polling = false; }
  }

  function updateEffectiveBatch() {
    const batch = Number($('batch-size').value), accum = Number($('accum').value);
    setText('effective-batch', Number.isInteger(batch) && batch > 0 && Number.isInteger(accum) && accum > 0 ? `每次更新使用 ${fmt(batch * accum)} 个样本。更大的批次会占用更多显存。` : '请填写有效的批大小和梯度累积。');
  }
  $('target-steps').addEventListener('input', () => { targetTouched = true; });
  $('batch-size').addEventListener('input', () => { batchTouched = true; updateEffectiveBatch(); }); $('accum').addEventListener('input', () => { accumTouched = true; updateEffectiveBatch(); });
  $('training-data').addEventListener('change', updateButtons);
  $('chart-range').addEventListener('change', drawChart); $('smooth-loss').addEventListener('change', drawChart);
  $('preview-image').addEventListener('error', () => { previewKey = ''; $('preview-link').hidden = true; $('preview-empty').hidden = false; setText('preview-step', '预览暂不可读'); });
  $('training-form').addEventListener('submit', async event => {
    event.preventDefault();
    if (snapshot?.training?.native_resolution === 512 && snapshot?.capabilities?.native_training_resolution !== true) { toast('继续原生 512 训练需要先重启 APP 加载控制更新。当前没有发送启动请求。', true); updateButtons(); return; }
    if (actionBusy || startPendingAt || $('start-button').disabled || !$('training-form').reportValidity()) return;
    const steps = Number($('target-steps').value), batch_size = Number($('batch-size').value), accum = Number($('accum').value);
    if (!Number.isInteger(steps) || steps <= (snapshot?.training?.step || 0)) { toast('累计目标必须大于当前已训练步数。', true); return; }
    const selectedData = $('training-data').value;
    if (!allowedTrainingData.has(selectedData)) { toast('请选择服务提供的可用训练数据。当前没有发送启动请求。', true); return; }
    actionBusy = true; updateButtons();
    try {
      const payload={steps,batch_size,accum,data:selectedData};
      if(selectedData==='app_flows'){
        const sameData=String(snapshot?.training?.data_path||'').replace(/\\/g,'/').includes('/data-app-flows-v5-20261028/combined');
        payload.app_flow_fraction=sameData&&numeric(snapshot.training.app_flow_fraction)?snapshot.training.app_flow_fraction:.4;
        payload.interaction_fraction=sameData&&numeric(snapshot.training.interaction_fraction)?snapshot.training.interaction_fraction:.2;
      }
      const result = await request('/api/training/start', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
      startPendingAt = Date.now(); toast(result.message || '启动请求已接受，正在等待训练状态更新。');
    } catch (error) { toast(error.message, true); }
    finally { actionBusy = false; updateButtons(); poll(); }
  });
  $('pause-button').addEventListener('click', async () => {
    if (actionBusy || pausePending || $('pause-button').disabled) return;
    actionBusy = true; pausePending = true; updateButtons(); setText('state-title', '正在保存并暂停');
    try {
      const result = await request('/api/training/pause', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
      toast(result.message || '暂停请求已接受，正在等待检查点保存。');
    } catch (error) { pausePending = false; toast(error.message, true); }
    finally { actionBusy = false; updateButtons(); poll(); }
  });
  poll();
  window.setInterval(poll, 2000);
})();
