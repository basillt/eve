/*
 * portrait.js — a character's face as a mosaic of glowing light-dots, in the spirit of the brain.
 *
 * A face file (web/faces/<id>.json) is built by tools/make_face.py from a reference image:
 *   { aspect, grid:[w,h], palette:[[r,g,b],…], points:[[x,y,colorIndex,brightness],…] }
 * Each point is drawn as an additive glow in its own colour, so the portrait reads clearly and
 * glows in many colours at once. It reacts to mood (liveliness, sparks, dim at "sleeping") and
 * exposes the same API as the brain (setMood / setTalking / pulse / setGrowth / setMature /
 * advance / snapshot / stats / dispose), so the app can show either interchangeably.
 */
function sprite(color, r = 24) {
  const c = document.createElement("canvas"); c.width = c.height = r * 2;
  const g = c.getContext("2d"), grad = g.createRadialGradient(r, r, 0, r, r, r);
  grad.addColorStop(0, `rgba(${color[0]},${color[1]},${color[2]},1)`);
  grad.addColorStop(0.4, `rgba(${color[0]},${color[1]},${color[2]},0.55)`);
  grad.addColorStop(1, `rgba(${color[0]},${color[1]},${color[2]},0)`);
  g.fillStyle = grad; g.fillRect(0, 0, r * 2, r * 2);
  return c;
}

const MOODS = {           // mood changes liveliness, not the picture's colours
  neutral:   { rate: 5,  twinkle: 0.10, speed: 1.0 },
  thinking:  { rate: 16, twinkle: 0.26, speed: 1.4 },
  happy:     { rate: 12, twinkle: 0.20, speed: 1.3 },
  sad:       { rate: 3,  twinkle: 0.07, speed: 0.7 },
  confused:  { rate: 10, twinkle: 0.18, speed: 1.1 },
  surprised: { rate: 20, twinkle: 0.28, speed: 1.5 },
  sleeping:  { rate: 2,  twinkle: 0.05, speed: 0.6 },
};

export function createPortrait(canvas, data, { seed = 7 } = {}) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const aspect = data.aspect || 1.4;
  const palette = (data.palette || [[25, 227, 255]]).map((c) => sprite(c));
  const hotSprite = sprite([255, 245, 215]);                     // twinkle peaks & sparks: warm white
  const pts = data.points.map((p) => ({ x: p[0], y: p[1], ci: p[2] | 0, b: p[3] ?? 1, ph: Math.random() * 6.28 }));
  const order = pts.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [order[i], order[j]] = [order[j], order[i]]; }

  let mood = "neutral", cur = MOODS.neutral, talking = false, mature = false;
  let growth = 1, time = 0, dim = 1;
  const sparks = [];
  const newSpark = () => { const p = pts[(Math.random() * pts.length) | 0]; return { x: p.x, y: p.y, vx: (Math.random() - 0.5) * 0.05, vy: (Math.random() - 0.5) * 0.05, life: 1 }; };

  const setMood = (m) => { cur = MOODS[m] || MOODS.neutral; mood = MOODS[m] ? m : "neutral"; };
  const setTalking = (on) => { talking = !!on; };
  const setMature = () => {};                                    // "complete" is Eve's brain, not a character face
  const setGrowth = (g) => { growth = Math.max(0.12, Math.min(1, +g || 0)); };
  const pulse = () => { for (let i = 0; i < 4; i++) if (sparks.length < 120) sparks.push(newSpark()); };

  let W = 0, H = 0, ox = 0, oy = 0, scale = 1, dpr = 1;
  const resize = () => {
    const cw = canvas.clientWidth || canvas.parentElement.clientWidth, ch = canvas.clientHeight || canvas.parentElement.clientHeight;
    if (!cw || !ch) return;
    dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = cw * dpr; canvas.height = ch * dpr; W = cw; H = ch;
    const pad = 0.96, fw = cw * pad, fh = ch * pad;
    scale = Math.min(fw, fh / aspect);
    ox = (cw - scale) / 2; oy = (ch - scale * aspect) / 2;
  };
  const ro = new ResizeObserver(resize); ro.observe(canvas.parentElement || canvas); resize();

  const cell = () => scale / (data.grid ? data.grid[0] : 90);    // one dot ≈ one grid cell

  const tick = (step) => {
    time += step;
    dim += (((mood === "sleeping") ? 0.55 : 1) - dim) * (1 - Math.exp(-step * 3));
    const rate = reduced ? 0 : (talking ? cur.rate * 2 : cur.rate);
    if (Math.random() < rate * step) sparks.push(newSpark());
    for (let i = sparks.length - 1; i >= 0; i--) {
      const s = sparks[i]; s.x += s.vx * step * cur.speed; s.y += s.vy * step * cur.speed; s.life -= step * 1.1;
      if (s.life <= 0) sparks.splice(i, 1);
    }
  };

  const draw = () => {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.globalCompositeOperation = "lighter";
    const n = Math.round(pts.length * growth);
    const c = cell(), tw = reduced ? 0 : cur.twinkle;
    const breathe = reduced ? 1 : 1 + 0.01 * Math.sin(time * 1.1);
    for (let k = 0; k < n; k++) {
      const p = pts[order[k]];
      const t = 1 + tw * Math.sin(time * 2.0 * cur.speed + p.ph);
      const sz = c * (0.62 + 0.5 * p.b) * t * breathe;          // dot a touch larger than its cell -> seamless glow
      const peak = tw > 0 && t > 1 + tw * 0.75;
      const img = peak ? hotSprite : palette[p.ci] || palette[0];
      ctx.globalAlpha = Math.min(1, (0.5 + 0.5 * p.b) * dim);
      const x = ox + p.x * scale, y = oy + p.y * scale * aspect;
      ctx.drawImage(img, x - sz, y - sz, sz * 2, sz * 2);
    }
    for (const s of sparks) {
      ctx.globalAlpha = Math.max(0, s.life) * 0.9 * dim;
      const sz = c * (1.4 + 1.2 * s.life), x = ox + s.x * scale, y = oy + s.y * scale * aspect;
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
