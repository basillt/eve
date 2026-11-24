/*
 * portrait.js — a character's face as a cloud of light, same spirit as the brain.
 *
 * A face file (web/faces/<id>.json) is { aspect, points:[[x,y,b],…] } with x,y in 0..1 and b a
 * brightness. We draw each point as an additive glow sprite, tinted by mood, twinkling, with a few
 * drifting sparks — so the portrait feels alive and matches Eve's neon look. Same API as the brain
 * (setMood / setTalking / pulse / setGrowth / setMature / advance / snapshot / stats / dispose),
 * so the app can show either one interchangeably.
 */
const MOODS = {
  neutral:   { base: [25, 227, 255], hot: [255, 43, 214], rate: 6,  twinkle: 0.12 },
  thinking:  { base: [25, 227, 255], hot: [246, 255, 74], rate: 16, twinkle: 0.30 },
  happy:     { base: [255, 43, 214], hot: [255, 255, 255], rate: 12, twinkle: 0.22 },
  sad:       { base: [61, 90, 254],  hot: [143, 180, 255], rate: 3,  twinkle: 0.08 },
  confused:  { base: [155, 92, 255], hot: [25, 227, 255],  rate: 10, twinkle: 0.20 },
  surprised: { base: [246, 255, 74], hot: [255, 255, 255], rate: 22, twinkle: 0.30 },
  sleeping:  { base: [75, 43, 255],  hot: [25, 227, 255],  rate: 2,  twinkle: 0.05 },
};
const MATURE = [255, 207, 74];
const rgb = (c) => `rgb(${c[0]|0},${c[1]|0},${c[2]|0})`;
const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);

function sprite(color, r = 32) {
  const c = document.createElement("canvas"); c.width = c.height = r * 2;
  const g = c.getContext("2d"), grad = g.createRadialGradient(r, r, 0, r, r, r);
  grad.addColorStop(0, `rgba(${color[0]},${color[1]},${color[2]},1)`);
  grad.addColorStop(0.35, `rgba(${color[0]},${color[1]},${color[2]},0.5)`);
  grad.addColorStop(1, `rgba(${color[0]},${color[1]},${color[2]},0)`);
  g.fillStyle = grad; g.fillRect(0, 0, r * 2, r * 2);
  return c;
}

export function createPortrait(canvas, data, { seed = 7 } = {}) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const aspect = data.aspect || 1.4;
  const pts = data.points.map(([x, y, b]) => ({ x, y, b, ph: Math.random() * 6.28 }));
  const order = pts.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [order[i], order[j]] = [order[j], order[i]]; }

  let mood = "neutral", talking = false, cur = MOODS.neutral, mature = false;
  let baseCol = MOODS.neutral.base.slice(), hotCol = MOODS.neutral.hot.slice();
  let baseSprite = sprite(baseCol), hotSprite = sprite(hotCol);
  let growth = 1, time = 0, dim = 1;
  const sparks = [];

  const setMood = (m) => { if (!MOODS[m]) m = "neutral"; mood = m; cur = MOODS[m]; };
  const setTalking = (on) => { talking = !!on; };
  const setMature = (on) => { mature = !!on; };
  const setGrowth = (g) => { growth = Math.max(0.12, Math.min(1, +g || 0)); };
  const pulse = () => { for (let i = 0; i < 3; i++) if (sparks.length < 90) sparks.push(newSpark()); };
  const newSpark = () => { const p = pts[(Math.random() * pts.length) | 0]; return { x: p.x, y: p.y, vx: (Math.random() - 0.5) * 0.06, vy: (Math.random() - 0.5) * 0.06, life: 1 }; };

  let W = 0, H = 0, ox = 0, oy = 0, scale = 1, dpr = 1;
  const resize = () => {
    const cw = canvas.clientWidth || canvas.parentElement.clientWidth, ch = canvas.clientHeight || canvas.parentElement.clientHeight;
    if (!cw || !ch) return;
    dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = cw * dpr; canvas.height = ch * dpr; W = cw; H = ch;
    const pad = 0.92, fw = cw * pad, fh = ch * pad;              // fit the portrait, keep its aspect
    scale = Math.min(fw, fh / aspect);
    ox = (cw - scale) / 2; oy = (ch - scale * aspect) / 2;
  };
  const ro = new ResizeObserver(resize); ro.observe(canvas.parentElement || canvas); resize();

  const tick = (step) => {
    time += step;
    // colours ease toward the mood (or gold when mature)
    const wantBase = mature ? MATURE : cur.base, wantHot = mature ? [255, 240, 180] : cur.hot;
    const k = 1 - Math.exp(-step * 3);
    const nb = lerp(baseCol, wantBase, k), nh = lerp(hotCol, wantHot, k);
    if (nb.some((v, i) => Math.abs(v - baseCol[i]) > 1) ) { baseCol = nb; baseSprite = sprite(baseCol); }
    if (nh.some((v, i) => Math.abs(v - hotCol[i]) > 1) ) { hotCol = nh; hotSprite = sprite(hotCol); }
    dim += (((mood === "sleeping") ? 0.5 : 1) - dim) * k;
    // sparks
    const rate = reduced ? 0 : (talking ? cur.rate * 2 : cur.rate);
    if (Math.random() < rate * step) sparks.push(newSpark());
    for (let i = sparks.length - 1; i >= 0; i--) {
      const s = sparks[i]; s.x += s.vx * step; s.y += s.vy * step; s.life -= step * 1.1;
      if (s.life <= 0) sparks.splice(i, 1);
    }
  };

  const draw = () => {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.globalCompositeOperation = "lighter";
    const n = Math.round(pts.length * growth);
    const breathe = reduced ? 1 : 1 + 0.015 * Math.sin(time * 1.2);
    const tw = reduced ? 0 : cur.twinkle;
    for (let k = 0; k < n; k++) {
      const p = pts[order[k]];
      const t = 0.8 + tw * Math.sin(time * 2.2 + p.ph);
      const sz = (2.4 + 3.6 * p.b) * t * breathe;
      const useHot = ((p.ph * 97) % 1) < 0.18;                 // a few points carry the hot colour
      const img = useHot ? hotSprite : baseSprite;
      const a = Math.min(1, (0.35 + 0.6 * p.b) * t * dim);
      ctx.globalAlpha = a;
      const x = ox + p.x * scale, y = oy + p.y * scale * aspect;
      ctx.drawImage(img, x - sz, y - sz, sz * 2, sz * 2);
    }
    for (const s of sparks) {
      ctx.globalAlpha = Math.max(0, s.life) * 0.9 * dim;
      const sz = 5 + 4 * s.life, x = ox + s.x * scale, y = oy + s.y * scale * aspect;
      ctx.drawImage(hotSprite, x - sz, y - sz, sz * 2, sz * 2);
    }
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = "source-over";
  };

  let last = performance.now(), acc = 0, raf = 0, alive = true;
  const frame = (now) => {
    if (!alive) return;
    raf = requestAnimationFrame(frame);
    if (document.hidden) { last = now; return; }
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    acc += dt; if (acc < 1 / 30) return;
    tick(acc); acc = 0; draw();
  };
  raf = requestAnimationFrame(frame);

  return {
    setMood, setTalking, pulse, setGrowth, setMature, vision: false,
    stats: () => ({ neurons: Math.round(pts.length * growth), max: pts.length }),
    advance(seconds, fps = 30) { for (let t = 0; t < seconds; t += 1 / fps) tick(1 / fps); draw(); },
    snapshot() { draw(); return canvas.toDataURL("image/png"); },
    dispose() { alive = false; cancelAnimationFrame(raf); ro.disconnect(); },
  };
}
