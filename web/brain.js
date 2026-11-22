/*
 * brain.js — the assistant's face: a living brain made of light.
 *
 * ~2000 neurons sit on a brain-shaped surface (two folded hemispheres, a cerebellum
 * and a stem), wired to their nearest neighbours. Sparks run along the wires and make
 * the neurons they pass glow. What the assistant is doing changes the activity:
 *
 *   idle        a few slow sparks, the brain turns gently
 *   thinking    a storm of fast sparks at the front (planning)
 *   talking     bursts on the left side (language) with every word that arrives
 *   happy/sad/confused/surprised   colour and pace change
 *   sleeping    dim indigo, slow waves sweep from front to back (consolidating the day)
 *
 * The brain also grows: setGrowth(0..1) shows that share of the neurons and only the wires
 * between them. A new user starts with a sparse outline; every memory, night and day adds
 * neurons, and each one is born with a flash.
 *
 *   const b = createBrain(canvas)        // null if WebGL is not available
 *   b.setMood("thinking"); b.setTalking(true); b.pulse(); b.setGrowth(0.6); b.dispose()
 */
import * as THREE from "three";

const MOODS = {
  neutral:   { rate: 7,  speed: 2.4, base: "#19e3ff", hot: "#ff2bd6", spin: 0.10, region: null },
  thinking:  { rate: 38, speed: 4.2, base: "#19e3ff", hot: "#f6ff4a", spin: 0.05, region: "front" },
  happy:     { rate: 16, speed: 3.2, base: "#ff2bd6", hot: "#ffffff", spin: 0.22, region: null },
  sad:       { rate: 4,  speed: 1.2, base: "#3d5afe", hot: "#8fb4ff", spin: 0.04, region: null },
  confused:  { rate: 24, speed: 2.6, base: "#9b5cff", hot: "#19e3ff", spin: -0.08, region: null },
  surprised: { rate: 40, speed: 5.0, base: "#f6ff4a", hot: "#ffffff", spin: 0.30, region: null },
  sleeping:  { rate: 2,  speed: 1.0, base: "#4b2bff", hot: "#19e3ff", spin: 0.03, region: null },
};
const TALK = { rate: 14, region: "language" };

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

// ------------------------------------------------------------------ shape

function brainNeurons(n, rand) {
  const pts = [];
  const gauss = () => { let u = 0, v = 0; while (!u) u = rand(); while (!v) v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const dir = () => { const x = gauss(), y = gauss(), z = gauss(), l = Math.hypot(x, y, z) || 1; return [x / l, y / l, z / l]; };
  const folds = (x, y, z) =>                      // gyri and sulci: crossed waves on the surface
    0.055 * Math.sin(11 * y + 4 * Math.sin(6 * z)) * Math.cos(9 * z + 3 * Math.sin(7 * x)) +
    0.025 * Math.sin(23 * x + 17 * z);

  while (pts.length < n) {
    const r = rand();
    if (r < 0.80) {                                // cerebrum: two hemispheres
      const side = rand() < 0.5 ? -1 : 1;
      const [dx, dy, dz] = dir();
      const shell = rand() < 0.88 ? 0.94 + 0.06 * rand() : 0.45 + 0.4 * rand();
      let x = dx * 0.5, y = dy * 0.6, z = dz * 0.88;
      const f = 1 + folds(dx, dy, dz);
      x *= f * shell; y *= f * shell; z *= f * shell;
      if (y < -0.22) y = -0.22 + (y + 0.22) * 0.45;              // flatter underside
      if (dy < -0.1 && dz > -0.2 && dz < 0.5) y -= 0.08 * Math.abs(dx);  // temporal lobes hang lower
      x = x + side * 0.36;
      if (Math.abs(x) < 0.045) continue;                         // the gap between hemispheres
      if (side * x < 0) x = side * (0.045 + Math.abs(x) * 0.3);
      pts.push([x, y + 0.08, z]);
    } else if (r < 0.95) {                         // cerebellum, low at the back
      const [dx, dy, dz] = dir();
      const shell = 0.9 + 0.1 * rand();
      const f = 1 + 0.06 * Math.sin(30 * dy);                    // fine parallel folds
      pts.push([dx * 0.42 * shell, -0.42 + dy * 0.2 * shell * f, -0.62 + dz * 0.26 * shell]);
    } else {                                       // brain stem
      const t = rand(), a = rand() * Math.PI * 2, rr = 0.07 + 0.03 * rand();
      pts.push([Math.cos(a) * rr, -0.35 - t * 0.5, -0.32 - t * 0.12 + Math.sin(a) * rr]);
    }
  }
  return pts;
}

function wire(pts, k) {
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
      if (j < 0 || best[m] > 0.03) continue;
      const key = i < j ? i * n + j : j * n + i;
      if (seen.has(key)) continue;
      seen.add(key); edges.push(i, j); adj[i].push(j); adj[j].push(i);
    }
  }
  return { edges, adj };
}

// ------------------------------------------------------------------ shaders

const NEURON_VS = `
attribute float act; attribute float seed;
uniform float uSize; uniform float uTime; uniform float uDim;
varying float vAct; varying float vSeed;
void main() {
  vAct = act; vSeed = seed;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float twinkle = 0.85 + 0.15 * sin(uTime * 1.7 + seed * 40.0);
  gl_PointSize = uSize * (0.5 + 0.8 * min(act, 1.0)) * twinkle * (3.0 / -mv.z);
  gl_Position = projectionMatrix * mv;
}`;
const NEURON_FS = `
uniform vec3 uBase; uniform vec3 uHot; uniform float uDim;
varying float vAct; varying float vSeed;
void main() {
  vec2 c = gl_PointCoord - 0.5; float d = length(c);
  if (d > 0.5) discard;
  float core = smoothstep(0.5, 0.0, d);
  vec3 col = mix(uBase, uHot, clamp(vAct * 1.3, 0.0, 1.0));
  float a = core * core * (0.5 + 0.5 * min(vAct, 1.0)) * uDim;
  gl_FragColor = vec4(col * (0.75 + 0.45 * min(vAct, 1.0)), a);
}`;
const SPARK_FS = `
uniform vec3 uHot; uniform float uDim;
void main() {
  vec2 c = gl_PointCoord - 0.5; float d = length(c);
  if (d > 0.5) discard;
  float g = smoothstep(0.5, 0.0, d);
  gl_FragColor = vec4(mix(uHot, vec3(1.0), g * 0.6), g * g * uDim);
}`;
const SPARK_VS = `
uniform float uSize;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = uSize * (3.0 / -mv.z);
  gl_Position = projectionMatrix * mv;
}`;

// ------------------------------------------------------------------ main

export function createBrain(canvas, { neurons = 2600, seed = 7, autoplay = true } = {}) {
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
  camera.position.set(0, 0.1, 3.8);
  const group = new THREE.Group();
  group.rotation.set(0.18, -0.6, 0);
  scene.add(group);

  const rand = rng(seed);
  const pts = brainNeurons(neurons, rand);
  const n = pts.length;
  const wired = wire(pts, 4), adj = wired.adj;
  // neurons are in random order, so the first N of them are an even sample of the shape.
  // Sort wires by their newest end: the wires of the first N neurons are a prefix.
  const order = [];
  for (let e = 0; e < wired.edges.length / 2; e++) order.push(e);
  const newest = (e) => Math.max(wired.edges[2 * e], wired.edges[2 * e + 1]);
  order.sort((a, b) => newest(a) - newest(b));
  const edges = new Int32Array(wired.edges.length), edgeTop = new Int32Array(order.length);
  order.forEach((e, k) => { edges[2 * k] = wired.edges[2 * e]; edges[2 * k + 1] = wired.edges[2 * e + 1]; edgeTop[k] = newest(e); });
  const wiresBelow = (N) => { let lo = 0, hi = edgeTop.length; while (lo < hi) { const m = (lo + hi) >> 1; if (edgeTop[m] < N) lo = m + 1; else hi = m; } return lo; };
  const pos = new Float32Array(n * 3);
  pts.forEach((p, i) => pos.set(p, i * 3));
  const act = new Float32Array(n);
  const seeds = new Float32Array(n).map(() => rand());

  // neurons
  const ng = new THREE.BufferGeometry();
  ng.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const actAttr = new THREE.BufferAttribute(act, 1);
  actAttr.setUsage(THREE.DynamicDrawUsage);
  ng.setAttribute("act", actAttr);
  ng.setAttribute("seed", new THREE.BufferAttribute(seeds, 1));
  const uniforms = {
    uSize: { value: 26 }, uTime: { value: 0 }, uDim: { value: 1 },
    uBase: { value: new THREE.Color(MOODS.neutral.base) }, uHot: { value: new THREE.Color(MOODS.neutral.hot) },
  };
  const nm = new THREE.ShaderMaterial({ uniforms, vertexShader: NEURON_VS, fragmentShader: NEURON_FS,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  group.add(new THREE.Points(ng, nm));

  // wires: colour follows the activity of their two ends
  const ne = edges.length / 2;
  const wpos = new Float32Array(ne * 6), wcol = new Float32Array(ne * 6);
  for (let e = 0; e < ne; e++) {
    wpos.set(pts[edges[2 * e]], e * 6);
    wpos.set(pts[edges[2 * e + 1]], e * 6 + 3);
  }
  const wg = new THREE.BufferGeometry();
  wg.setAttribute("position", new THREE.BufferAttribute(wpos, 3));
  const wcolAttr = new THREE.BufferAttribute(wcol, 3);
  wcolAttr.setUsage(THREE.DynamicDrawUsage);
  wg.setAttribute("color", wcolAttr);
  const wm = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55,
    depthWrite: false, blending: THREE.AdditiveBlending });
  group.add(new THREE.LineSegments(wg, wm));

  // sparks travelling along wires
  const MAX = 500;
  const spos = new Float32Array(MAX * 3);
  const sg = new THREE.BufferGeometry();
  const sposAttr = new THREE.BufferAttribute(spos, 3);
  sposAttr.setUsage(THREE.DynamicDrawUsage);
  sg.setAttribute("position", sposAttr);
  sg.setDrawRange(0, 0);
  const sm = new THREE.ShaderMaterial({ uniforms: { uSize: { value: 34 }, uHot: uniforms.uHot, uDim: uniforms.uDim },
    vertexShader: SPARK_VS, fragmentShader: SPARK_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  group.add(new THREE.Points(sg, sm));
  const sparks = [];

  const regions = {
    front: (p) => p[2] > 0.35 && p[1] > -0.15,
    language: (p) => p[0] < -0.15 && p[1] < 0.15 && p[2] > -0.25 && p[2] < 0.55,
  };
  // growth: how many neurons are alive
  let growth = 1, growthTarget = 1, alive_n = n, alive_e = ne;
  const live = (i) => i < alive_n;
  const liveNb = (i) => adj[i].filter(live);
  const pick = (region) => {
    const test = region && regions[region];
    for (let t = 0; t < 40; t++) {
      const i = Math.floor(rand() * alive_n);
      if (liveNb(i).length && (!test || test(pts[i]))) return i;
    }
    return Math.floor(rand() * alive_n);
  };
  const spawn = (region, hops) => {
    if (sparks.length >= MAX) return;
    const a = pick(region), nb = liveNb(a);
    if (!nb.length) return;
    const b = nb[Math.floor(rand() * nb.length)];
    act[a] = Math.min(1, act[a] + 0.6);
    sparks.push({ a, b, t: 0, hops: hops ?? (4 + Math.floor(rand() * 10)) });
  };

  // state
  let mood = "neutral", talking = false, cur = { ...MOODS.neutral }, budget = 0, wave = 0;
  const target = { base: new THREE.Color(cur.base), hot: new THREE.Color(cur.hot) };
  const setMood = (m) => {
    if (!MOODS[m]) m = "neutral";
    if (m === "surprised" && mood !== "surprised") for (let i = 0; i < 120; i++) spawn(null, 3 + Math.floor(rand() * 5));
    mood = m; cur = { ...MOODS[m] };
    target.base.set(cur.base); target.hot.set(cur.hot);
  };
  const setTalking = (on) => { talking = !!on; };
  const applyGrowth = (g) => {
    const N = Math.max(60, Math.min(n, Math.round(n * g)));
    for (let i = alive_n; i < N; i++) act[i] = 1;          // newborn neurons flash
    alive_n = N; alive_e = wiresBelow(N);
    ng.setDrawRange(0, alive_n); wg.setDrawRange(0, alive_e * 2);
  };
  const setGrowth = (g, instant, flash) => {         // flash: newborn neurons light up even when instant
    growthTarget = Math.max(0, Math.min(1, +g || 0));
    if (instant) {
      growth = growthTarget;
      const N = Math.max(60, Math.min(n, Math.round(n * growth)));
      if (!flash || N < alive_n) alive_n = N;
      applyGrowth(growth);
    }
  };
  const pulse = () => { for (let i = 0; i < 3; i++) spawn("language", 3 + Math.floor(rand() * 6)); };

  // drag to turn
  let drag = null, userYaw = 0, userPitch = 0;
  const down = (e) => { drag = { x: e.clientX, y: e.clientY, yaw: userYaw, pitch: userPitch }; canvas.setPointerCapture?.(e.pointerId); };
  const move = (e) => { if (!drag) return; userYaw = drag.yaw + (e.clientX - drag.x) * 0.008; userPitch = Math.max(-0.6, Math.min(0.6, drag.pitch + (e.clientY - drag.y) * 0.006)); };
  const up = () => { drag = null; };
  canvas.addEventListener("pointerdown", down); canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);
  canvas.style.touchAction = "none";

  const resize = () => {
    const w = canvas.clientWidth || canvas.parentElement.clientWidth, h = canvas.clientHeight || canvas.parentElement.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    const asp = w / h;                                   // keep it in frame on tall screens, bigger on wide strips
    camera.position.z = asp < 1 ? 3.8 / Math.max(0.5, asp) : asp > 1.5 ? 3.3 : 3.8;
    camera.updateProjectionMatrix();
    const k = Math.min(1.3, Math.max(0.45, h / 650));     // dots keep their size relative to the brain
    uniforms.uSize.value = 26 * k; sm.uniforms.uSize.value = 34 * k;
  };
  const ro = new ResizeObserver(resize);
  ro.observe(canvas.parentElement || canvas);
  resize();

  let last = performance.now(), acc = 0, raf = 0, yaw = 0, time = 0, alive = true;
  const frame = (now) => {
    if (!alive) return;
    raf = requestAnimationFrame(frame);
    if (document.hidden) { last = now; return; }
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    acc += dt; if (acc < 1 / 30) return;                 // 30 fps is plenty, leaves the GPU to the model
    const step = acc; acc = 0;
    tick(step);
    renderer.render(scene, camera);
  };
  const tick = (step) => {
    time += step;

    uniforms.uTime.value = time;
    uniforms.uBase.value.lerp(target.base, 1 - Math.exp(-step * 3));
    uniforms.uHot.value.lerp(target.hot, 1 - Math.exp(-step * 3));
    uniforms.uDim.value += ((mood === "sleeping" ? 0.55 : 1) - uniforms.uDim.value) * (1 - Math.exp(-step * 2));

    // growth: neurons are born a few at a time
    if (growth !== growthTarget) {
      const d = growthTarget - growth, sp = Math.max(0.02, Math.abs(d) * 0.6) * step;
      growth = Math.abs(d) <= sp ? growthTarget : growth + Math.sign(d) * sp;
      if (growthTarget < growth || d < 0) alive_n = Math.min(alive_n, Math.round(n * growth));
      applyGrowth(growth);
    }

    // new sparks
    budget += step * (reduced ? cur.rate * 0.4 : cur.rate);
    while (budget >= 1) { spawn(cur.region); budget -= 1; }
    if (talking) {
      budget2 += step * TALK.rate;
      while (budget2 >= 1) { spawn(TALK.region); budget2 -= 1; }
    }
    // sleep: a slow wave from front to back lights the neurons it crosses
    if (mood === "sleeping") {
      wave = (wave + step / 7) % 1;
      const zc = 1.0 - wave * 2.1;
      for (let i = 0; i < alive_n; i++) {
        const d = Math.abs(pts[i][2] - zc);
        if (d < 0.05) act[i] = Math.max(act[i], 0.55 * (1 - d / 0.05));
      }
    }

    // move sparks
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
        let next = nb[Math.floor(rand() * nb.length)];
        if (next === sp.a && nb.length > 1) next = nb[(nb.indexOf(next) + 1) % nb.length];
        sp.a = sp.b; sp.b = next;
      }
    }
    for (const sp of sparks) {
      const a = pts[sp.a], b = pts[sp.b], t = sp.t;
      spos[w++] = a[0] + (b[0] - a[0]) * t; spos[w++] = a[1] + (b[1] - a[1]) * t; spos[w++] = a[2] + (b[2] - a[2]) * t;
    }
    sg.setDrawRange(0, sparks.length);
    sposAttr.needsUpdate = true;

    // decay + wire colours
    const decay = Math.exp(-step * 3.2);
    for (let i = 0; i < alive_n; i++) act[i] *= decay;
    actAttr.needsUpdate = true;
    const bc = uniforms.uBase.value, hc = uniforms.uHot.value, dim = uniforms.uDim.value;
    for (let e = 0; e < alive_e; e++) {
      for (let k = 0; k < 2; k++) {
        const a = Math.min(1, act[edges[2 * e + k]]);
        const base = 0.07 * dim, hot = 0.9 * a * dim, o = e * 6 + k * 3;
        wcol[o] = bc.r * base + hc.r * hot; wcol[o + 1] = bc.g * base + hc.g * hot; wcol[o + 2] = bc.b * base + hc.b * hot;
      }
    }
    wcolAttr.needsUpdate = true;

    // motion: slow turn, breathing, and the user's drag
    if (!drag) yaw += step * (reduced ? 0 : cur.spin);
    group.rotation.y = -0.6 + yaw + userYaw;
    group.rotation.x = 0.18 + userPitch + (reduced ? 0 : 0.03 * Math.sin(time * 0.6));
    const breathe = 1 + (reduced ? 0 : (mood === "sleeping" ? 0.02 : 0.012) * Math.sin(time * (mood === "sleeping" ? 0.7 : 1.4)));
    group.scale.setScalar(breathe);
    group.position.y = 0.06 + (reduced ? 0 : 0.03 * Math.sin(time * 0.8));
  };
  let budget2 = 0;
  if (autoplay) raf = requestAnimationFrame(frame);   // autoplay:false = driven by advance() only (video, tests)

  return {
    setMood, setTalking, pulse, setGrowth,
    stats: () => ({ neurons: alive_n, wires: alive_e, max: n }),
    snapshot() { renderer.render(scene, camera); return canvas.toDataURL("image/png"); },   // read in the same task as the draw
    advance(seconds, fps = 30) {                      // for tests and previews: simulate time, then draw
      for (let t = 0; t < seconds; t += 1 / fps) tick(1 / fps);
      renderer.render(scene, camera);
    },
    dispose() {
      alive = false; cancelAnimationFrame(raf); ro.disconnect();
      canvas.removeEventListener("pointerdown", down); canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up); canvas.removeEventListener("pointercancel", up);
      ng.dispose(); wg.dispose(); sg.dispose(); nm.dispose(); wm.dispose(); sm.dispose(); renderer.dispose();
    },
  };
}
