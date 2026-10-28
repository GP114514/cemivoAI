/*
 * cemivoAI release-film graphics — abstract explanatory animation only.
 * Canvas 2D; no dependencies, global random state, I/O, or generated UI claims.
 * All functions are pure with respect to time and preserve caller canvas state.
 * A real image supplied to drawPixelLens is shown unchanged and only cropped
 * for magnification. Other panels are abstract diagrams, not application shots.
 */
'use strict';

const W = 1680, H = 700;
const THEME = Object.freeze({ ink: '#f2f5f3', muted: '#77857f', accent: '#a7ead5', black: '#060909' });
const LAYOUT = Object.freeze({ width: W, height: H, defaultBounds: { x: 120, y: 180, width: W, height: H } });
const clamp = (n, a = 0, b = 1) => Math.min(b, Math.max(a, n));
const lerp = (a, b, p) => a + (b - a) * p;
const smooth = p => { p = clamp(p); return p * p * (3 - 2 * p); };
const ease = p => 1 - Math.pow(1 - clamp(p), 3);
const phase = (p, a, b) => smooth((p - a) / (b - a));
const frac = n => n - Math.floor(n);
const noise = n => frac(Math.sin(n * 127.1 + 311.7) * 43758.5453123);
const rgba = (alpha, accent = false) => accent ? `rgba(167,234,213,${clamp(alpha)})` : `rgba(242,245,243,${clamp(alpha)})`;

function rounded(ctx, x, y, w, h, r = 12) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r); ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h); ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r); ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
}
function line(ctx, points, color, width = 1, dash = []) {
  ctx.beginPath(); ctx.moveTo(points[0][0], points[0][1]);
  for (const p of points.slice(1)) ctx.lineTo(p[0], p[1]);
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
}
function label(ctx, text, x, y, size = 17, color = rgba(.58), align = 'left', weight = 500) {
  ctx.fillStyle = color; ctx.font = `${weight} ${size}px Arial, sans-serif`;
  ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillText(text, x, y);
}
function glow(ctx, x, y, radius, opacity = .08) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, radius);
  g.addColorStop(0, rgba(opacity, true)); g.addColorStop(.38, rgba(opacity * .35, true)); g.addColorStop(1, rgba(0, true));
  ctx.fillStyle = g; ctx.fillRect(x - radius, y - radius, radius * 2, radius * 2);
}
function panel(ctx, x, y, w, h, {alpha = 1, accent = false, fill = .025, r = 13} = {}) {
  rounded(ctx, x, y, w, h, r); ctx.fillStyle = rgba(fill * alpha, accent); ctx.fill();
  ctx.strokeStyle = rgba((accent ? .65 : .22) * alpha, accent); ctx.lineWidth = 1; ctx.stroke();
}
function dot(ctx, x, y, radius = 3, alpha = 1, accent = true) {
  ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fillStyle = rgba(alpha, accent); ctx.fill();
}
function grid(ctx, x, y, w, h, cell = 48, alpha = .055) {
  ctx.save(); rounded(ctx, x, y, w, h, 8); ctx.clip();
  for (let xx = x; xx <= x + w; xx += cell) line(ctx, [[xx, y], [xx, y + h]], rgba(alpha));
  for (let yy = y; yy <= y + h; yy += cell) line(ctx, [[x, yy], [x + w, yy]], rgba(alpha));
  ctx.restore();
}
function check(ctx, x, y, s = 12, alpha = 1) {
  line(ctx, [[x - s * .6, y], [x - s * .12, y + s * .46], [x + s * .7, y - s * .52]], rgba(alpha, true), 2);
}
function arrow(ctx, x, y, direction = 'right', alpha = .4) {
  ctx.save(); ctx.translate(x, y); if (direction === 'down') ctx.rotate(Math.PI / 2);
  line(ctx, [[-5, -5], [0, 0], [-5, 5]], rgba(alpha, true), 1.2); ctx.restore();
}
function pulse(ctx, points, p, alpha = 1) {
  const lengths = points.slice(1).map((v, i) => Math.hypot(v[0] - points[i][0], v[1] - points[i][1]));
  let distance = clamp(p) * lengths.reduce((a, b) => a + b, 0);
  for (let i = 0; i < lengths.length; i++) {
    if (distance <= lengths[i] || i === lengths.length - 1) {
      const f = lengths[i] ? distance / lengths[i] : 0;
      const x = lerp(points[i][0], points[i + 1][0], f), y = lerp(points[i][1], points[i + 1][1], f);
      glow(ctx, x, y, 42, .11 * alpha); dot(ctx, x, y, 3.5, alpha); return;
    }
    distance -= lengths[i];
  }
}
function begin(ctx, options, t, duration) {
  const opts = options || {}, scale = (opts.w || 1920) / 1920;
  const x = opts.x ?? 120 * scale, y = opts.y ?? 180 * ((opts.h || 1080) / 1080);
  const width = opts.width ?? 1680 * scale, height = opts.height ?? 700 * ((opts.h || 1080) / 1080);
  ctx.save(); ctx.translate(x, y); ctx.scale(width / W, height / H);
  const p = clamp(opts.progress ?? t / duration);
  return { opts, p, bounds: { x, y, width, height }, point: (px, py) => ({ x: x + px * width / W, y: y + py * height / H }) };
}
function end(ctx, state, anchors) {
  ctx.restore(); const out = {};
  for (const [key, point] of Object.entries(anchors)) out[key] = state.point(...point);
  return { progress: state.p, bounds: state.bounds, anchors: out };
}
function abstractFrame(ctx, x, y, w, h, alpha = 1, accent = false, seed = 7) {
  panel(ctx, x, y, w, h, {alpha, accent, fill: .025, r: 9});
  line(ctx, [[x, y + 24], [x + w, y + 24]], rgba(.14 * alpha));
  for (let i = 0; i < 3; i++) dot(ctx, x + 12 + i * 10, y + 12, 1.4, alpha * .35, false);
  const pad = Math.max(12, w * .055), top = y + 42;
  const barCount = Math.max(3, Math.floor(h / 37));
  for (let j = 0; j < barCount; j++) {
    const by = top + j * (h - 60) / Math.max(1, barCount - 1);
    const bw = (w - pad * 2) * (.38 + noise(seed + j) * .52);
    ctx.fillStyle = rgba(alpha * (j === 0 ? .17 : .075)); ctx.fillRect(x + pad, by, bw, j === 0 ? 7 : 3);
  }
}

/**
 * Fragmented abstract windows settle into one continuous light surface.
 * Recommended duration 8 s. Empty backdrop; no real software screen is drawn.
 */
function drawField(ctx, t, options = {}) {
  const s = begin(ctx, options, t, 8), p = s.p;
  const converge = phase(p, .08, .72), reveal = phase(p, .42, .9);
  glow(ctx, 840, 355, 650, .06 + reveal * .04);
  const windows = [
    [45, 52, 290, 174], [370, 38, 235, 138], [690, 75, 310, 164], [1070, 42, 230, 148], [1342, 80, 278, 192],
    [104, 294, 225, 155], [398, 246, 320, 195], [797, 292, 242, 168], [1123, 264, 385, 168],
    [30, 507, 300, 145], [375, 493, 240, 168], [685, 514, 320, 145], [1070, 484, 205, 178], [1330, 499, 300, 151]
  ];
  for (let i = 0; i < windows.length; i++) {
    const v = windows[i], travel = ease(clamp(converge * 1.15 - noise(i + 22) * .15));
    const targetX = 355 + (i % 4) * 255, targetY = 179 + Math.floor(i / 4) * 90;
    const x = lerp(v[0], targetX, travel), y = lerp(v[1], targetY, travel);
    const w = lerp(v[2], 235, travel), h = lerp(v[3], 73, travel);
    const a = (1 - phase(p, .48, .79)) * .85;
    if (a > .005) {
      ctx.save(); ctx.translate(x + w / 2, y + h / 2);
      ctx.rotate((noise(i + 50) - .5) * .04 * (1 - travel));
      abstractFrame(ctx, -w / 2, -h / 2, w, h, a, i === 5 || i === 8, i * 5); ctx.restore();
    }
    if (p < .77) {
      const alpha = .11 * (1 - converge);
      line(ctx, [[v[0] + v[2] / 2, v[1] + v[3] / 2], [840, 350]], rgba(alpha, true), .7);
    }
  }
  // The final object has one physical boundary and a continuous internal field.
  if (reveal > .001) {
    const x = 240, y = 85, w = 1200, h = 530;
    ctx.save(); ctx.globalAlpha = reveal;
    rounded(ctx, x, y, w, h, 18);
    const g = ctx.createLinearGradient(x, y, x + w, y + h);
    g.addColorStop(0, 'rgba(213,242,231,.075)'); g.addColorStop(.45, 'rgba(242,245,243,.022)'); g.addColorStop(1, 'rgba(167,234,213,.08)');
    ctx.fillStyle = g; ctx.fill(); ctx.strokeStyle = rgba(.7, true); ctx.lineWidth = 1; ctx.stroke();
    ctx.save(); rounded(ctx, x + 1, y + 1, w - 2, h - 2, 17); ctx.clip();
    grid(ctx, x, y, w, h, 40, .04);
    // Sparse deterministic raster light; the sweep resolves into a quiet plane.
    for (let iy = 0; iy < 12; iy++) for (let ix = 0; ix < 29; ix++) {
      const wave = Math.sin(ix * .22 - iy * .16 + t * .28) * .5 + .5;
      const q = noise(ix * 17 + iy * 31 + 90);
      const alpha = .01 + (q > .88 ? .07 : .012) * wave;
      ctx.fillStyle = rgba(alpha, q > .82); ctx.fillRect(x + 20 + ix * 40, y + 22 + iy * 40, 38, 38);
    }
    const scan = x + w * phase(p, .42, .88);
    const beam = ctx.createLinearGradient(scan - 130, 0, scan + 130, 0);
    beam.addColorStop(0, rgba(0, true)); beam.addColorStop(.5, rgba(.035, true)); beam.addColorStop(1, rgba(0, true));
    ctx.fillStyle = beam; ctx.fillRect(scan - 130, y, 260, h);
    // A small unbranded anchor reads as a unified surface, never as an app mockup.
    line(ctx, [[x + 55, y + 55], [x + 150, y + 55]], rgba(.6), 2);
    line(ctx, [[x + 55, y + 77], [x + 104, y + 77]], rgba(.18), 2);
    ctx.restore();
    for (const [cx, cy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) dot(ctx, cx, cy, 2.2, .7);
    label(ctx, 'ONE CONTINUOUS SURFACE', x + 55, y + h - 40, 13, rgba(.38));
    ctx.restore();
  }
  return end(ctx, s, { center: [840, 350], planeTop: [840, 85], planeBottom: [840, 615] });
}

function neuralNode(ctx, x, y, w, h, activity, t) {
  panel(ctx, x, y, w, h, {accent: true, alpha: .55 + activity * .45, fill: .025});
  const cols = [x + w * .24, x + w * .5, x + w * .76], rows = [y + h * .32, y + h * .5, y + h * .68];
  for (let c = 0; c < 2; c++) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    const a = .07 + .12 * activity * (.5 + .5 * Math.sin(t + i + j));
    line(ctx, [[cols[c], rows[i]], [cols[c + 1], rows[j]]], rgba(a, true), .8);
  }
  for (let c = 0; c < 3; c++) for (let i = 0; i < 3; i++) dot(ctx, cols[c], rows[i], 3.2, .35 + activity * .6);
}
function maskPatch(ctx, x, y, w, h, p, t) {
  panel(ctx, x, y, w, h, {fill: .012});
  grid(ctx, x + 1, y + 1, w - 2, h - 2, 20, .05);
  const mx = x + w * .33, my = y + h * .28, mw = w * .45, mh = h * .48;
  rounded(ctx, mx, my, mw, mh, 3); ctx.fillStyle = rgba(.035 + .04 * p, true); ctx.fill();
  ctx.strokeStyle = rgba(.55 + .4 * p, true); ctx.setLineDash([4, 5]); ctx.stroke(); ctx.setLineDash([]);
  for (let iy = 0; iy < 5; iy++) for (let ix = 0; ix < 7; ix++) {
    const v = noise(ix * 29 + iy * 37), wave = .5 + .5 * Math.sin(t * 1.7 + ix * .7 - iy * .4);
    ctx.fillStyle = rgba((.035 + v * .25) * p * (.55 + .45 * wave), true);
    ctx.fillRect(mx + 4 + ix * (mw - 8) / 7, my + 4 + iy * (mh - 8) / 5, (mw - 8) / 7 - 2, (mh - 8) / 5 - 2);
  }
  return {x: mx, y: my, width: mw, height: mh};
}

/**
 * Full frame + coordinate → AI intent → local mask/pixels → next frame + receipt.
 * Duration 12 s; English micro-labels may be disabled with labels:false.
 * receiptVerified:true adds a verification mark only when caller has real evidence.
 */
function drawPipeline(ctx, t, options = {}) {
  const s = begin(ctx, options, t, 12), p = s.p, show = s.opts.labels !== false;
  const inputP = phase(p, .0, .22), aiP = phase(p, .2, .48), maskP = phase(p, .42, .72), outP = phase(p, .7, .94);
  const stages = { frame: [35, 205, 300, 230], ai: [485, 215, 230, 210], mask: [855, 205, 300, 230], result: [1300, 120, 330, 235] };
  glow(ctx, 860, 335, 610, .055);
  // One unbroken feed emphasizes that the input is a complete frame.
  abstractFrame(ctx, ...stages.frame, .9, false, 15);
  const cx = 235, cy = 336;
  if (inputP > 0) {
    ctx.save(); ctx.globalAlpha = inputP;
    ctx.strokeStyle = rgba(.95, true); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, 12, 0, Math.PI * 2); ctx.stroke();
    line(ctx, [[cx - 20, cy], [cx + 20, cy]], rgba(.55, true)); line(ctx, [[cx, cy - 20], [cx, cy + 20]], rgba(.55, true));
    dot(ctx, cx, cy, 2.5, .95); label(ctx, '(x, y)', cx + 30, cy + 1, 14, rgba(.8, true)); ctx.restore();
  }
  const routeA = [[335, 320], [485, 320]], routeB = [[715, 320], [855, 320]];
  const routeC = [[1155, 320], [1225, 320], [1225, 238], [1300, 238]];
  const routeD = [[1155, 320], [1225, 320], [1225, 510], [1300, 510]];
  for (const r of [routeA, routeB, routeC, routeD]) line(ctx, r, rgba(.2), 1);
  arrow(ctx, 472, 320); arrow(ctx, 842, 320); arrow(ctx, 1288, 238); arrow(ctx, 1288, 510);
  if (p > .07 && p < .31) pulse(ctx, routeA, (p - .07) / .24);
  if (p > .31 && p < .56) pulse(ctx, routeB, (p - .31) / .25);
  if (p > .6 && p < .86) { pulse(ctx, routeC, (p - .6) / .26); pulse(ctx, routeD, (p - .6) / .26, .7); }
  neuralNode(ctx, ...stages.ai, aiP, t);
  maskPatch(ctx, ...stages.mask, maskP, t);
  abstractFrame(ctx, ...stages.result, .3 + outP * .7, true, 21);
  // Output pixels are an abstract frame; never composite over a real app image.
  ctx.save(); ctx.globalAlpha = outP;
  const outputGradient = ctx.createLinearGradient(1320, 200, 1600, 300);
  outputGradient.addColorStop(0, rgba(.035, true)); outputGradient.addColorStop(1, rgba(.14, true));
  ctx.fillStyle = outputGradient; rounded(ctx, 1340, 205, 250, 92, 6); ctx.fill(); ctx.restore();
  panel(ctx, 1300, 435, 330, 150, {accent: outP > .5, alpha: .35 + .65 * outP, fill: .025});
  // Receipt is a separate artifact; caller controls whether it is verified.
  line(ctx, [[1330, 477], [1490, 477]], rgba(.35 + .25 * outP), 3);
  line(ctx, [[1330, 501], [1555, 501]], rgba(.14 + .15 * outP), 2);
  line(ctx, [[1330, 525], [1450, 525]], rgba(.14 + .15 * outP), 2);
  if (s.opts.receiptVerified === true && outP > .2) check(ctx, 1592, 475, 12, outP);
  if (show) {
    label(ctx, '01', 35, 152, 13, rgba(.3)); label(ctx, 'FULL FRAME + CLICK', 35, 170, 15, rgba(.7));
    label(ctx, '02', 485, 152, 13, rgba(.3)); label(ctx, 'AI INTENT', 485, 170, 15, rgba(.7));
    label(ctx, '03', 855, 152, 13, rgba(.3)); label(ctx, 'LOCAL MASK / PIXELS', 855, 170, 15, rgba(.7));
    label(ctx, 'NEXT FRAME', 1300, 86, 15, rgba(.7)); label(ctx, 'AGENT RECEIPT', 1300, 402, 15, rgba(.7));
    label(ctx, 'WHOLE IMAGE IN', 35, 472, 12, rgba(.28));
    label(ctx, 'INTENT, THEN LOCAL CHANGE', 855, 472, 12, rgba(.28));
  }
  return end(ctx, s, { frame: [185, 320], click: [cx, cy], ai: [600, 320], mask: [1005, 320], result: [1465, 238], receipt: [1465, 510] });
}

/**
 * A full-frame source and enlarged local region. Duration 8 s.
 * image: optional real CanvasImage, drawn without filters or content changes.
 * mask: normalized {x,y,width,height}; default central right region.
 * image supplied => lens is an exact crop. No synthetic result is substituted.
 */
function drawPixelLens(ctx, t, options = {}) {
  const s = begin(ctx, options, t, 8), p = s.p, image = s.opts.image;
  const region = {x: .52, y: .31, width: .24, height: .32, ...(s.opts.mask || {})};
  const box = {x: 45, y: 90, width: 940, height: 520}, lens = {x: 1170, y: 192, width: 450, height: 306};
  let source = {...box};
  glow(ctx, 1220, 350, 500, .06);
  panel(ctx, box.x, box.y, box.width, box.height, {fill: .012});
  if (image) {
    const scale = Math.min((box.width - 2) / image.width, (box.height - 2) / image.height);
    source = {x: box.x + (box.width - image.width * scale) / 2, y: box.y + (box.height - image.height * scale) / 2, width: image.width * scale, height: image.height * scale};
    ctx.drawImage(image, source.x, source.y, source.width, source.height);
  } else {
    grid(ctx, box.x, box.y, box.width, box.height, 26, .045);
    line(ctx, [[box.x + 50, box.y + 64], [box.x + 300, box.y + 64]], rgba(.25), 7);
    for (let k = 0; k < 4; k++) {
      rounded(ctx, box.x + 50 + k * 210, box.y + 132, 180, 275, 8); ctx.fillStyle = rgba(.025); ctx.fill();
      ctx.strokeStyle = rgba(.13); ctx.stroke();
      for (let j = 0; j < 5; j++) line(ctx, [[box.x + 70 + k * 210, box.y + 167 + j * 35], [box.x + 190 + k * 210, box.y + 167 + j * 35]], rgba(.06), 2);
    }
  }
  const mx = source.x + source.width * region.x, my = source.y + source.height * region.y;
  const mw = source.width * region.width, mh = source.height * region.height;
  const reveal = phase(p, .08, .32), expand = phase(p, .2, .62);
  ctx.save(); ctx.globalAlpha = reveal;
  rounded(ctx, mx, my, mw, mh, 2); ctx.strokeStyle = rgba(.95, true); ctx.lineWidth = 1.5; ctx.stroke();
  for (const [cx, cy] of [[mx, my], [mx + mw, my], [mx, my + mh], [mx + mw, my + mh]]) {
    ctx.fillStyle = THEME.accent; ctx.fillRect(cx - 2.5, cy - 2.5, 5, 5);
  }
  ctx.restore();
  if (expand > .001) {
    ctx.save(); ctx.globalAlpha = expand;
    line(ctx, [[mx + mw, my], [1085, my], [lens.x, lens.y]], rgba(.3, true), 1);
    line(ctx, [[mx + mw, my + mh], [1085, my + mh], [lens.x, lens.y + lens.height]], rgba(.3, true), 1);
    panel(ctx, lens.x, lens.y, lens.width, lens.height, {accent: true, fill: .025});
    ctx.save(); rounded(ctx, lens.x + 1, lens.y + 1, lens.width - 2, lens.height - 2, 11); ctx.clip();
    if (image) {
      const aspect = region.width * image.width / (region.height * image.height);
      let dw = lens.width - 2, dh = dw / aspect;
      if (dh > lens.height - 2) { dh = lens.height - 2; dw = dh * aspect; }
      ctx.drawImage(image, image.width * region.x, image.height * region.y, image.width * region.width, image.height * region.height,
        lens.x + (lens.width - dw) / 2, lens.y + (lens.height - dh) / 2, dw, dh);
    } else {
      const cols = 20, rows = 13;
      for (let iy = 0; iy < rows; iy++) for (let ix = 0; ix < cols; ix++) {
        const wave = .5 + .5 * Math.sin(ix * .36 + iy * .22 - t * .65);
        const v = noise(ix * 81 + iy * 32 + 3), activate = phase(p, .42, .85);
        ctx.fillStyle = rgba(.025 + activate * (v > .5 ? .12 + wave * .27 : .04), true);
        ctx.fillRect(lens.x + 3 + ix * (lens.width - 6) / cols, lens.y + 3 + iy * (lens.height - 6) / rows,
          (lens.width - 6) / cols - 1.5, (lens.height - 6) / rows - 1.5);
      }
    }
    ctx.restore();
    label(ctx, image ? 'SOURCE CROP' : 'LOCAL PIXEL FIELD', lens.x, lens.y - 32, 14, rgba(.65));
    label(ctx, image ? 'MAGNIFICATION ONLY' : 'MASK BOUNDS THE CHANGE', lens.x, lens.y + lens.height + 32, 12, rgba(.34));
    ctx.restore();
  }
  if (s.opts.labels !== false) label(ctx, image ? 'FULL SOURCE FRAME' : 'FULL FRAME', box.x, box.y - 32, 14, rgba(.65));
  return end(ctx, s, { source: [source.x + source.width / 2, source.y + source.height / 2], mask: [mx + mw / 2, my + mh / 2], lens: [lens.x + lens.width / 2, lens.y + lens.height / 2] });
}

/**
 * Confirmed rule reuse requires matching complete image hash AND state.
 * Duration 9 s. Later phase highlights unknown input returning to AI.
 * No numeric speed claim is drawn; timing benchmarks belong to actual footage.
 */
function drawMemory(ctx, t, options = {}) {
  const s = begin(ctx, options, t, 9), p = s.p, show = s.opts.labels !== false;
  const known = phase(p, .08, .45), unknown = phase(p, .61, .93);
  const topActive = 1 - unknown * .6, bottomActive = unknown;
  glow(ctx, 940, 220, 650, .045 + .04 * known * topActive);
  abstractFrame(ctx, 30, 210, 280, 220, .8, false, 12);
  // Complete frame identity and interaction state are separate requirements.
  panel(ctx, 30, 464, 280, 82, {fill: .012});
  label(ctx, 'HASH   c5e9b6', 50, 490, 13, rgba(.58));
  label(ctx, unknown > .5 ? 'STATE  74:22' : 'STATE  74:21', 50, 519, 13, rgba(.58, unknown > .5));
  const cx = 490, cy = 320, r = 66;
  ctx.beginPath(); ctx.moveTo(cx, cy - r); ctx.lineTo(cx + r, cy); ctx.lineTo(cx, cy + r); ctx.lineTo(cx - r, cy); ctx.closePath();
  ctx.fillStyle = rgba(.025, true); ctx.fill(); ctx.strokeStyle = rgba(.5, true); ctx.stroke();
  label(ctx, '=', cx, cy, 29, rgba(.8, true), 'center');
  const input = [[310, 320], [424, 320]], upper = [[556, 320], [610, 320], [610, 188], [710, 188]];
  const lower = [[556, 320], [610, 320], [610, 510], [710, 510]];
  const fastOut = [[1020, 188], [1250, 188]], aiOut = [[1020, 510], [1250, 510]];
  line(ctx, input, rgba(.3), 1);
  line(ctx, upper, rgba(.2 + known * .36 * topActive, true), 1.2);
  line(ctx, lower, rgba(.12 + .4 * bottomActive, true), 1.2, [3, 5]);
  line(ctx, fastOut, rgba(.2 + known * .36 * topActive, true), 1.2);
  line(ctx, aiOut, rgba(.12 + .4 * bottomActive, true), 1.2);
  arrow(ctx, 412, 320); arrow(ctx, 698, 188); arrow(ctx, 698, 510, 'right', .2 + .5 * bottomActive);
  arrow(ctx, 1238, 188); arrow(ctx, 1238, 510, 'right', .2 + .5 * bottomActive);
  panel(ctx, 710, 108, 310, 160, {accent: true, alpha: .5 + known * .5 * topActive, fill: .035});
  const tiles = [[748, 148], [810, 148], [872, 148]];
  for (let i = 0; i < tiles.length; i++) {
    rounded(ctx, tiles[i][0], tiles[i][1], 42, 42, 5); ctx.strokeStyle = rgba(.3 + known * .35, true); ctx.stroke();
    if (known > .4) check(ctx, tiles[i][0] + 21, tiles[i][1] + 21, 8, known * .8 * topActive);
  }
  label(ctx, 'c5e9b6  /  74:21', 748, 233, 13, rgba(.45));
  abstractFrame(ctx, 1250, 83, 370, 245, .4 + known * .5 * topActive, true, 12);
  // Local replay is shown by a precise raster sweep, with no made-up duration.
  ctx.save(); rounded(ctx, 1251, 84, 368, 243, 8); ctx.clip();
  const sx = 1250 + 370 * phase(p, .34, .6);
  line(ctx, [[sx, 83], [sx, 328]], rgba((1 - phase(p, .59, .64)) * .55, true), 1.3); ctx.restore();
  neuralNode(ctx, 710, 430, 310, 160, unknown, t);
  panel(ctx, 1250, 430, 370, 160, {accent: unknown > .5, alpha: .3 + .7 * unknown, fill: .025});
  line(ctx, [[1290, 480], [1500, 480]], rgba(.15 + .25 * unknown), 3);
  line(ctx, [[1290, 510], [1565, 510]], rgba(.08 + .16 * unknown), 2);
  line(ctx, [[1290, 540], [1455, 540]], rgba(.08 + .16 * unknown), 2);
  if (p > .04 && p < .16) pulse(ctx, input, (p - .04) / .12);
  if (p > .13 && p < .38) pulse(ctx, upper, (p - .13) / .25);
  if (p > .38 && p < .61) pulse(ctx, fastOut, (p - .38) / .23);
  if (p > .65 && p < .83) pulse(ctx, lower, (p - .65) / .18);
  if (p > .82 && p < .98) pulse(ctx, aiOut, (p - .82) / .16);
  if (show) {
    label(ctx, 'COMPLETE FRAME', 30, 165, 14, rgba(.65));
    label(ctx, 'HASH + STATE', 490, 425, 12, rgba(.35), 'center');
    label(ctx, 'MATCH', 610, 99, 12, rgba(.4 + known * .3, true));
    label(ctx, 'CONFIRMED RULE', 710, 72, 15, rgba(.7));
    label(ctx, 'LOCAL REPLAY', 1250, 47, 15, rgba(.7));
    label(ctx, 'UNKNOWN', 610, 620, 12, rgba(.25 + .5 * unknown, true));
    label(ctx, 'RETURN TO AI', 710, 640, 15, rgba(.4 + .3 * unknown));
    label(ctx, 'NEW FRAME / RECEIPT', 1250, 640, 15, rgba(.4 + .3 * unknown));
  }
  return end(ctx, s, { frame: [170, 320], condition: [490, 320], confirmed: [865, 188], replay: [1435, 188], ai: [865, 510], receipt: [1435, 510] });
}

module.exports = { drawField, drawPipeline, drawPixelLens, drawMemory, THEME, LAYOUT };
