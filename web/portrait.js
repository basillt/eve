/*
 * portrait.js — a character's face as a cloud of glowing light in 3D, in the spirit of the brain.
 *
 * A face file (web/faces/<id>.json) is built by tools/make_face.py from a reference image:
 *   { aspect, grid:[w,h], palette:[[r,g,b],…], points:[[x,y,colorIndex,brightness],…] }
 * Each 2-D point is lifted into 3-D: the face gets a rounded relief (centre bulges toward you,
 * the edges fall away, and bright features — nose, cheekbones — stand a little more forward), so
 * the portrait is a glowing bust of light you can see depth in and turn. Every dot keeps its own
 * colour and glows additively. It reacts to mood (liveliness, sparks, dim at "sleeping") and
 * exposes the same API as the brain (setMood / setTalking / pulse / setGrowth / setMature /
 * advance / snapshot / stats / dispose), so the app can show either interchangeably.
 *
 *   const p = createPortrait(canvas, data)    // null if WebGL is not available
 */
import * as THREE from "three";

const MOODS = {           // mood changes liveliness, not the picture's colours
  neutral:   { rate: 10, twinkle: 0.12, speed: 1.0, sway: 0.55 },
  thinking:  { rate: 46, twinkle: 0.26, speed: 1.6, sway: 0.42 },
  happy:     { rate: 30, twinkle: 0.22, speed: 1.4, sway: 0.70 },
  sad:       { rate: 5,  twinkle: 0.07, speed: 0.7, sway: 0.30 },
  confused:  { rate: 22, twinkle: 0.18, speed: 1.1, sway: 0.60 },
  surprised: { rate: 60, twinkle: 0.30, speed: 1.8, sway: 0.85 },
  sleeping:  { rate: 3,  twinkle: 0.05, speed: 0.6, sway: 0.20 },
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
  const plane = new Float32Array(n * 2);             // keep (x,y) for spark spawns
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
    plane[i * 2] = x; plane[i * 2 + 1] = y;
    const c = palette[p[2] | 0] || palette[0];
    col[i * 3] = c[0] / 255; col[i * 3 + 1] = c[1] / 255; col[i * 3 + 2] = c[2] / 255;
    bright[i] = b; seeds[i] = rnd();
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("acolor", new THREE.BufferAttribute(col, 3));
  g.setAttribute("bright", new THREE.BufferAttribute(bright, 1));
  g.setAttribute("seed", new THREE.BufferAttribute(seeds, 1));
  const actAttr = new THREE.BufferAttribute(act, 1); actAttr.setUsage(THREE.DynamicDrawUsage);
  g.setAttribute("act", actAttr);
  // first N points are an even sample (shuffled order) -> growth can reveal a share of them
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = (rnd() * (i + 1)) | 0; [order[i], order[j]] = [order[j], order[i]]; }
  const idx = new Uint32Array(order);
  g.setIndex(new THREE.BufferAttribute(idx, 1));

  const uniforms = {
    uSize: { value: 30 }, uTime: { value: 0 }, uTwinkle: { value: MOODS.neutral.twinkle },
    uHot: { value: new THREE.Color("#fff3d2") }, uDim: { value: 1 },
  };
  const mat = new THREE.ShaderMaterial({ uniforms, vertexShader: VS, fragmentShader: FS,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  group.add(new THREE.Points(g, mat));

  // ---- sparks: motes that drift off the face when it is lively --------------
  const MAX = 160;
  const spos = new Float32Array(MAX * 3), slife = new Float32Array(MAX);
  const sg = new THREE.BufferGeometry();
  const sposAttr = new THREE.BufferAttribute(spos, 3); sposAttr.setUsage(THREE.DynamicDrawUsage);
  const slifeAttr = new THREE.BufferAttribute(slife, 1); slifeAttr.setUsage(THREE.DynamicDrawUsage);
  sg.setAttribute("position", sposAttr); sg.setAttribute("alife", slifeAttr);
  sg.setDrawRange(0, 0);
  const sm = new THREE.ShaderMaterial({ uniforms: { uSize: { value: 40 }, uHot: uniforms.uHot, uDim: uniforms.uDim, uTime: uniforms.uTime },
    vertexShader: SPARK_VS, fragmentShader: SPARK_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  group.add(new THREE.Points(sg, sm));
  const sparks = [];
  const newSpark = () => {
    const i = (rnd() * n) | 0;
    sparks.push({ x: plane[i * 2], y: plane[i * 2 + 1], z: pos[i * 3 + 2],
      vx: (rnd() - 0.5) * 0.5, vy: (rnd() - 0.5) * 0.5, vz: 0.1 + rnd() * 0.5, life: 1 });
  };

  // ---- state ----------------------------------------------------------------
  let mood = "neutral", cur = MOODS.neutral, talking = false;
  let growth = 1, time = 0, budget = 0;
  const target = { twinkle: cur.twinkle };
  const setMood = (m) => { cur = MOODS[m] || MOODS.neutral; mood = MOODS[m] ? m : "neutral"; target.twinkle = reduced ? 0 : cur.twinkle;
    if (mood === "surprised") for (let i = 0; i < 30; i++) { if (sparks.length < MAX) newSpark(); } };
  const setTalking = (on) => { talking = !!on; };
  const setMature = () => {};                                   // "complete" is Eve's brain, not a character face
  const setGrowth = (gval) => { growth = Math.max(0.12, Math.min(1, +gval || 0)); g.setDrawRange(0, Math.max(30, Math.round(n * growth))); };
  const flash = (k) => { for (let i = 0; i < k; i++) act[order[(rnd() * Math.round(n * growth)) | 0]] = 1; };
  const pulse = () => { flash(14); for (let i = 0; i < 5; i++) if (sparks.length < MAX) newSpark(); };

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
    uniforms.uSize.value = 30 * k; sm.uniforms.uSize.value = 40 * k;
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

    // flashes roam across the face; talking makes it busier
    budget += step * (reduced ? 0 : cur.rate * (talking ? 2 : 1));
    while (budget >= 1) { flash(1); if (sparks.length < MAX && rnd() < 0.5) newSpark(); budget -= 1; }
    const decay = Math.exp(-step * 3.0);
    const live = Math.round(n * growth);
    for (let i = 0; i < live; i++) act[order[i]] *= decay;
    actAttr.needsUpdate = true;

    // move sparks
    let w = 0;
    for (let s = sparks.length - 1; s >= 0; s--) {
      const sp = sparks[s];
      sp.life -= step * 1.0;
      if (sp.life <= 0) { sparks.splice(s, 1); continue; }
      sp.x += sp.vx * step * cur.speed; sp.y += sp.vy * step * cur.speed; sp.z += sp.vz * step * cur.speed;
    }
    for (const sp of sparks) { spos[w * 3] = sp.x; spos[w * 3 + 1] = sp.y; spos[w * 3 + 2] = sp.z; slife[w] = Math.max(0, sp.life); w++; }
    sg.setDrawRange(0, sparks.length); sposAttr.needsUpdate = true; slifeAttr.needsUpdate = true;

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
      g.dispose(); sg.dispose(); mat.dispose(); sm.dispose(); renderer.dispose();
    },
  };
}
