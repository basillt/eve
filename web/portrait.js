/*
 * portrait.js — a character's face as a 3-D constellation of light: points wired together.
 *
 * A face file (web/faces/<id>.json) is built by tools/make_face.py from a reference image:
 *   { aspect, grid:[w,h], palette:[[r,g,b],…], points:[[x,y,colorIndex,brightness],…] }
 * Each 2-D point is lifted into 3-D on a rounded relief (centre bulges toward you, the edges fall
 * away, bright features — nose, cheekbones — stand a little more forward), then every point is
 * wired to its nearest neighbours, exactly like the brain. The result is a glowing mesh in the
 * shape of the face that you can see depth in and turn; sparks run along the wires and light the
 * points they pass. Every point keeps its own colour and glows additively. It reacts to mood
 * (liveliness, dim at "sleeping") and exposes the same API as the brain (setMood / setTalking /
 * pulse / setGrowth / setMature / advance / snapshot / stats / dispose), so the app can show
 * either interchangeably.
 *
 *   const p = createPortrait(canvas, data)    // null if WebGL is not available
 */
import * as THREE from "three";

function wire(pts, k, maxd2) {                        // k nearest neighbours within a distance
  const n = pts.length, edges = [], adj = Array.from({ length: n }, () => []);
  const seen = new Set();
  const best = new Float32Array(k), idx = new Int32Array(k);
  for (let i = 0; i < n; i++) {
    best.fill(Infinity); idx.fill(-1);
    const [xi, yi, zi] = pts[i];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = pts[j][0] - xi, dy = pts[j][1] - yi, dz = pts[j][2] - zi;
      const d = dx * dx + dy * dy + dz * dz;
      if (d < best[k - 1]) {
        let p = k - 1;
        while (p > 0 && best[p - 1] > d) { best[p] = best[p - 1]; idx[p] = idx[p - 1]; p--; }
        best[p] = d; idx[p] = j;
      }
    }
    for (let m = 0; m < k; m++) {
      const j = idx[m];
      if (j < 0 || best[m] > maxd2) continue;
      const key = i < j ? i * n + j : j * n + i;
      if (seen.has(key)) continue;
      seen.add(key); edges.push(i, j); adj[i].push(j); adj[j].push(i);
    }
  }
  return { edges, adj };
}

const MOODS = {           // mood changes liveliness, not the picture's colours
  neutral:   { rate: 6,  twinkle: 0.10, speed: 1.0, sway: 0.55 },
  thinking:  { rate: 20, twinkle: 0.20, speed: 1.6, sway: 0.42 },
  happy:     { rate: 15, twinkle: 0.16, speed: 1.4, sway: 0.70 },
  sad:       { rate: 3,  twinkle: 0.06, speed: 0.7, sway: 0.30 },
  confused:  { rate: 12, twinkle: 0.15, speed: 1.1, sway: 0.60 },
  surprised: { rate: 28, twinkle: 0.24, speed: 1.8, sway: 0.85 },
  sleeping:  { rate: 2,  twinkle: 0.05, speed: 0.6, sway: 0.20 },
};

const VS = `
attribute vec3 acolor; attribute float bright; attribute float act; attribute float seed;
uniform float uSize; uniform float uTime; uniform float uTwinkle;
varying vec3 vCol; varying float vBright; varying float vAct;
void main() {
  vCol = acolor; vBright = bright; vAct = act;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float tw = 1.0 + uTwinkle * sin(uTime * 2.2 + seed * 40.0);
  gl_PointSize = uSize * (0.55 + 0.6 * bright + 0.85 * min(act, 1.0)) * tw * (3.0 / -mv.z);
  gl_Position = projectionMatrix * mv;
}`;
const FS = `
uniform vec3 uHot; uniform float uDim;
varying vec3 vCol; varying float vBright; varying float vAct;
void main() {
  vec2 c = gl_PointCoord - 0.5; float d = length(c);
  if (d > 0.5) discard;
  float core = smoothstep(0.5, 0.0, d);
  vec3 col = mix(vCol, uHot, clamp(vAct, 0.0, 1.0));
  float a = core * core * (0.42 + 0.58 * vBright) * uDim;
  gl_FragColor = vec4(col * (0.85 + 0.5 * min(vAct, 1.0)), a);
}`;
const SPARK_VS = `
attribute float alife; uniform float uSize; uniform float uTime;
varying float vLife;
void main() {
  vLife = alife;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = uSize * (0.5 + 1.1 * alife) * (3.0 / -mv.z);
  gl_Position = projectionMatrix * mv;
}`;
const SPARK_FS = `
uniform vec3 uHot; uniform float uDim;
varying float vLife;
void main() {
  vec2 c = gl_PointCoord - 0.5; float d = length(c);
  if (d > 0.5) discard;
  float g = smoothstep(0.5, 0.0, d);
  gl_FragColor = vec4(mix(uHot, vec3(1.0), g * 0.6), g * g * vLife * uDim);
}`;

export function createPortrait(canvas, data, { seed = 7, autoplay = true } = {}) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
  } catch (e) {
    return null;
  }
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 20);
  camera.position.set(0, 0, 3.2);
  const group = new THREE.Group();
  scene.add(group);

  const aspect = data.aspect || 1.0;                 // h / w of the face crop
  const raw = data.points || [];
  const n = raw.length;

  // ---- lift the 2-D mosaic into a 3-D relief --------------------------------
  // plane: centre the face, keep its proportions (wider or taller as the crop is)
  const HWx = 1.0, HWy = HWx * aspect;
  let mb = 0; for (const p of raw) mb += (p[3] ?? 1); mb = n ? mb / n : 0.66;   // mean brightness
  const DOME = 0.62, RELIEF = 0.42;

  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  const bright = new Float32Array(n);
  const seeds = new Float32Array(n);
  const act = new Float32Array(n);
  const palette = (data.palette || [[255, 245, 215]]);
  let sd = seed >>> 0;
  const rnd = () => ((sd = (sd * 1664525 + 1013904223) >>> 0) / 4294967296);

  for (let i = 0; i < n; i++) {
    const p = raw[i];
    const nx = (p[0] - 0.5), ny = (p[1] - 0.5);
    const x = nx * 2 * HWx;
    const y = -ny * 2 * HWy;
    const ex = x / HWx, ey = y / HWy;                // ~ -1..1 across the face
    const rr = Math.min(1, ex * ex + ey * ey);
    const dome = Math.sqrt(Math.max(0, 1 - 0.82 * rr));
    const b = p[3] ?? 1;
    const z = dome * DOME + (b - mb) * RELIEF;
    pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
    const c = palette[p[2] | 0] || palette[0];
    col[i * 3] = c[0] / 255; col[i * 3 + 1] = c[1] / 255; col[i * 3 + 2] = c[2] / 255;
    bright[i] = b; seeds[i] = rnd();
  }

  // ---- wire the points to their nearest neighbours (a constellation) --------
  const pts3 = []; for (let i = 0; i < n; i++) pts3.push([pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]]);
  const cellW = 2 * HWx / ((data.grid && data.grid[0]) || 90);
  const { edges: rawEdges, adj } = wire(pts3, 6, (cellW * 1.9) * (cellW * 1.9));
  // order the points (even sample) so growth can reveal a prefix; sort wires by their newest end
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = (rnd() * (i + 1)) | 0; [order[i], order[j]] = [order[j], order[i]]; }
  const orderPos = new Int32Array(n); order.forEach((p, k) => (orderPos[p] = k));
  const ne = rawEdges.length / 2;
  const eord = Array.from({ length: ne }, (_, e) => e);
  const newest = (e) => Math.max(orderPos[rawEdges[2 * e]], orderPos[rawEdges[2 * e + 1]]);
  eord.sort((a, b) => newest(a) - newest(b));
  const edges = new Int32Array(ne * 2), edgeTop = new Int32Array(ne);
  eord.forEach((e, k) => { edges[2 * k] = rawEdges[2 * e]; edges[2 * k + 1] = rawEdges[2 * e + 1]; edgeTop[k] = newest(e); });
  const wiresBelow = (N) => { let lo = 0, hi = ne; while (lo < hi) { const m = (lo + hi) >> 1; if (edgeTop[m] < N) lo = m + 1; else hi = m; } return lo; };

  // points
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("acolor", new THREE.BufferAttribute(col, 3));
  g.setAttribute("bright", new THREE.BufferAttribute(bright, 1));
  g.setAttribute("seed", new THREE.BufferAttribute(seeds, 1));
  const actAttr = new THREE.BufferAttribute(act, 1); actAttr.setUsage(THREE.DynamicDrawUsage);
  g.setAttribute("act", actAttr);
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(order), 1));

  const uniforms = {
    uSize: { value: 18 }, uTime: { value: 0 }, uTwinkle: { value: MOODS.neutral.twinkle },
    uHot: { value: new THREE.Color("#fff3d2") }, uDim: { value: 1 },
  };
  const mat = new THREE.ShaderMaterial({ uniforms, vertexShader: VS, fragmentShader: FS,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  group.add(new THREE.Points(g, mat));

  // wires: colour follows each end's own colour, brightening where a spark passes
  const wpos = new Float32Array(ne * 6), wcol = new Float32Array(ne * 6);
  for (let e = 0; e < ne; e++) {
    const a = edges[2 * e], b = edges[2 * e + 1];
    wpos.set(pts3[a], e * 6); wpos.set(pts3[b], e * 6 + 3);
  }
  const wg = new THREE.BufferGeometry();
  wg.setAttribute("position", new THREE.BufferAttribute(wpos, 3));
  const wcolAttr = new THREE.BufferAttribute(wcol, 3); wcolAttr.setUsage(THREE.DynamicDrawUsage);
  wg.setAttribute("color", wcolAttr);
  const wm = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85,
    depthWrite: false, blending: THREE.AdditiveBlending });
  group.add(new THREE.LineSegments(wg, wm));

  // ---- sparks travelling along the wires ------------------------------------
  const MAX = 420;
  const spos = new Float32Array(MAX * 3), slife = new Float32Array(MAX);
  const sg = new THREE.BufferGeometry();
  const sposAttr = new THREE.BufferAttribute(spos, 3); sposAttr.setUsage(THREE.DynamicDrawUsage);
  const slifeAttr = new THREE.BufferAttribute(slife, 1); slifeAttr.setUsage(THREE.DynamicDrawUsage);
  sg.setAttribute("position", sposAttr); sg.setAttribute("alife", slifeAttr);
  sg.setDrawRange(0, 0);
  const sm = new THREE.ShaderMaterial({ uniforms: { uSize: { value: 26 }, uHot: uniforms.uHot, uDim: uniforms.uDim, uTime: uniforms.uTime },
    vertexShader: SPARK_VS, fragmentShader: SPARK_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  group.add(new THREE.Points(sg, sm));
  const sparks = [];

  // ---- state ----------------------------------------------------------------
  let mood = "neutral", cur = MOODS.neutral, talking = false;
  let growth = 1, time = 0, budget = 0, talkBudget = 0, alive_n = n, alive_e = ne;
  const target = { twinkle: cur.twinkle };
  const liveNb = (i) => adj[i].filter((j) => orderPos[j] < alive_n);
  const spawn = (hops) => {
    if (sparks.length >= MAX) return;
    let a = -1;
    for (let t = 0; t < 30; t++) { const i = order[(rnd() * alive_n) | 0]; if (adj[i].length) { a = i; break; } }
    if (a < 0) return;
    const nb = liveNb(a); if (!nb.length) return;
    const b = nb[(rnd() * nb.length) | 0];
    act[a] = Math.min(1, act[a] + 0.6);
    sparks.push({ a, b, t: 0, hops: hops ?? (4 + ((rnd() * 10) | 0)) });
  };
  const setMood = (m) => { cur = MOODS[m] || MOODS.neutral; mood = MOODS[m] ? m : "neutral"; target.twinkle = reduced ? 0 : cur.twinkle;
    if (mood === "surprised") for (let i = 0; i < 60; i++) spawn(3 + ((rnd() * 5) | 0)); };
  const setTalking = (on) => { talking = !!on; };
  const setMature = () => {};                                   // "complete" is Eve's brain, not a character face
  const setGrowth = (gval) => {
    growth = Math.max(0.12, Math.min(1, +gval || 0));
    alive_n = Math.max(30, Math.round(n * growth)); alive_e = wiresBelow(alive_n);
    g.setDrawRange(0, alive_n); wg.setDrawRange(0, alive_e * 2);
  };
  const pulse = () => { for (let i = 0; i < 5; i++) spawn(3 + ((rnd() * 6) | 0)); };

  // ---- fit + drag to turn ---------------------------------------------------
  const resize = () => {
    const w = canvas.clientWidth || canvas.parentElement.clientWidth, h = canvas.clientHeight || canvas.parentElement.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    const asp = w / h;
    // the face is ~2 wide, ~2*aspect tall, bulging ~0.6 toward the camera
    const need = Math.max(2 * HWy / (2 * Math.tan(camera.fov * Math.PI / 360)),
                          2 * HWx / (2 * Math.tan(camera.fov * Math.PI / 360) * asp));
    camera.position.z = need * 1.16 + 0.6;
    camera.updateProjectionMatrix();
    const k = Math.min(1.5, Math.max(0.5, h / 650));
    uniforms.uSize.value = 18 * k; sm.uniforms.uSize.value = 26 * k;
  };
  const ro = new ResizeObserver(resize); ro.observe(canvas.parentElement || canvas); resize();

  let drag = null, userYaw = 0, userPitch = 0;
  const down = (e) => { drag = { x: e.clientX, y: e.clientY, yaw: userYaw, pitch: userPitch }; canvas.setPointerCapture?.(e.pointerId); };
  const move = (e) => { if (!drag) return;                 // a frontal relief: keep it off the edge-on sliver
    userYaw = Math.max(-0.75, Math.min(0.75, drag.yaw + (e.clientX - drag.x) * 0.009));
    userPitch = Math.max(-0.5, Math.min(0.5, drag.pitch + (e.clientY - drag.y) * 0.006)); };
  const up = () => { drag = null; };
  canvas.addEventListener("pointerdown", down); canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);
  canvas.style.touchAction = "none"; canvas.style.cursor = "grab";

  // ---- loop -----------------------------------------------------------------
  const tick = (step) => {
    time += step;
    uniforms.uTime.value = time;
    uniforms.uTwinkle.value += (target.twinkle - uniforms.uTwinkle.value) * (1 - Math.exp(-step * 3));
    uniforms.uDim.value += (((mood === "sleeping") ? 0.55 : 1) - uniforms.uDim.value) * (1 - Math.exp(-step * 2));

    // new sparks: a steady flicker across the mesh, busier while talking
    budget += step * (reduced ? cur.rate * 0.4 : cur.rate);
    while (budget >= 1) { spawn(); budget -= 1; }
    if (talking) { talkBudget += step * 20; while (talkBudget >= 1) { spawn(2 + ((rnd() * 4) | 0)); talkBudget -= 1; } }

    // move sparks hop by hop along the wires, lighting the points they reach
    let w = 0;
    for (let s = sparks.length - 1; s >= 0; s--) {
      const sp = sparks[s];
      sp.t += step * cur.speed * (0.7 + 0.6 * ((sp.a * 13) % 7) / 7);
      while (sp.t >= 1) {
        sp.t -= 1;
        act[sp.b] = Math.min(1, act[sp.b] + 0.6);
        if (--sp.hops <= 0) { sparks.splice(s, 1); break; }
        const nb = liveNb(sp.b);
        if (!nb.length) { sparks.splice(s, 1); break; }
        let next = nb[(rnd() * nb.length) | 0];
        if (next === sp.a && nb.length > 1) next = nb[(nb.indexOf(next) + 1) % nb.length];
        sp.a = sp.b; sp.b = next;
      }
    }
    for (const sp of sparks) {
      const a = pts3[sp.a], b = pts3[sp.b], t = sp.t;
      spos[w * 3] = a[0] + (b[0] - a[0]) * t; spos[w * 3 + 1] = a[1] + (b[1] - a[1]) * t; spos[w * 3 + 2] = a[2] + (b[2] - a[2]) * t;
      slife[w] = 1; w++;
    }
    sg.setDrawRange(0, w); sposAttr.needsUpdate = true; slifeAttr.needsUpdate = true;

    // decay the point activity, then colour the wires from their ends
    const decay = Math.exp(-step * 3.0);
    for (let i = 0; i < alive_n; i++) act[order[i]] *= decay;
    actAttr.needsUpdate = true;
    const dim = uniforms.uDim.value, hot = uniforms.uHot.value;
    for (let e = 0; e < alive_e; e++) {
      for (let kk = 0; kk < 2; kk++) {
        const vi = edges[2 * e + kk], a = Math.min(1, act[vi]), o = e * 6 + kk * 3;
        const base = 0.36 * dim, h = 0.85 * a * dim;
        wcol[o] = col[vi * 3] * base + hot.r * h;
        wcol[o + 1] = col[vi * 3 + 1] * base + hot.g * h;
        wcol[o + 2] = col[vi * 3 + 2] * base + hot.b * h;
      }
    }
    wcolAttr.needsUpdate = true;

    // motion: a gentle 3-D sway reveals the depth; the user can drag to turn fully
    const sway = reduced ? 0 : cur.sway;
    group.rotation.y = userYaw + (drag ? 0 : sway * Math.sin(time * 0.35));
    group.rotation.x = userPitch + (reduced ? 0 : 0.05 * Math.sin(time * 0.5));
    const breathe = reduced ? 1 : 1 + 0.012 * Math.sin(time * 1.3);
    group.scale.setScalar(breathe);
    group.position.y = reduced ? 0 : 0.02 * Math.sin(time * 0.8);
  };

  let last = performance.now(), acc = 0, raf = 0, alive = true;
  const frame = (now) => {
    if (!alive) return;
    raf = requestAnimationFrame(frame);
    if (document.hidden) { last = now; return; }
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    acc += dt; if (acc < 1 / 30) return;
    tick(acc); acc = 0;
    renderer.render(scene, camera);
  };
  if (autoplay) raf = requestAnimationFrame(frame);

  return {
    setMood, setTalking, pulse, setGrowth, setMature, vision: false,
    stats: () => ({ neurons: Math.round(n * growth), max: n }),
    snapshot() { renderer.render(scene, camera); return canvas.toDataURL("image/png"); },
    advance(seconds, fps = 30) { for (let t = 0; t < seconds; t += 1 / fps) tick(1 / fps); renderer.render(scene, camera); },
    dispose() {
      alive = false; cancelAnimationFrame(raf); ro.disconnect();
      canvas.removeEventListener("pointerdown", down); canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up); canvas.removeEventListener("pointercancel", up);
      g.dispose(); wg.dispose(); sg.dispose(); mat.dispose(); wm.dispose(); sm.dispose(); renderer.dispose();
    },
  };
}
