/* CityLab scroll flight: a drone camera flies over the real #1 building while seven chapters
   tell the story. Interactive: the camera leans with the cursor, roofs light up on hover with
   their real numbers, a click installs or removes that roof's panels, and a dock jumps between
   chapters, moves the sun through the day, shows live totals and switches to a survey view.
   Every shape and number is real: Vexcel's photo and roof outlines, our panel layout and NREL output. */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const gsap = window.gsap, ScrollTrigger = window.ScrollTrigger, Lenis = window.Lenis;
const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString("en-US");
const smooth = THREE.MathUtils.smoothstep;
const WORDS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];
const facingWord = (deg) => WORDS[Math.round(deg / 45) % 8];
const CHAPTERS = [["01", "City", 0.0], ["02", "Discover", 0.16], ["03", "Measure", 0.35], ["04", "Analyze", 0.52],
  ["05", "Place", 0.7], ["06", "Result", 0.85], ["07", "CityLab", 0.985]];
const S_ = { progress: 0, shown: 0, px: 0, py: 0, spx: 0, spy: 0, survey: 0, surveyOn: false, overrides: {}, hover: -1, sunT: 0.62,
  roam: false, back: 0, speed: 0 };

export async function initFlight({ city = "los_angeles", onExplore } = {}) {
  const section = $("flight");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!section || !gsap || !ScrollTrigger || reduced) { if (section) section.classList.add("static"); return; }
  gsap.registerPlugin(ScrollTrigger);

  if (Lenis) {  // one clock for smooth scroll and scroll triggers
    const lenis = new Lenis({ lerp: 0.075, wheelMultiplier: 0.85, touchMultiplier: 1.1, autoRaf: false });
    lenis.on("scroll", ScrollTrigger.update);
    gsap.ticker.add((t) => lenis.raf(t * 1000));
    gsap.ticker.lagSmoothing(0);
    window.__lenis = lenis;
  }

  let data;
  try { data = await (await fetch(`data_${city}.json`)).json(); } catch { section.classList.add("static"); return; }
  const b = data.buildings.find((x) => x.scene && x.scene.photo);
  if (!b) { section.classList.add("static"); return; }
  fillText(data, b);
  const view = build(b, data, $("flight-canvas"));

  const q = (sel) => section.querySelectorAll(sel);
  const tl = gsap.timeline({
    scrollTrigger: { trigger: section, start: "top top", end: "bottom bottom", scrub: 0.6,
      onUpdate: (self) => { S_.progress = self.progress; S_.speed = Math.min(1, Math.abs(self.getVelocity()) / 3000); } },
  });
  // kinetic headings: words rise in one by one; real numbers count up as their chapter arrives
  q(".ch h1, .ch h2").forEach(splitWords);
  tl.to(q('[data-ch="1"]'), { opacity: 0, y: -32, duration: 0.09 }, 0.04);
  const ch = (n, a, z) => {
    tl.fromTo(q(`[data-ch="${n}"]`), { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: 0.04 }, a)
      .fromTo(q(`[data-ch="${n}"] .w`), { opacity: 0, yPercent: 60, rotateX: -50 },
        { opacity: 1, yPercent: 0, rotateX: 0, duration: 0.03, stagger: 0.005, ease: "power3.out" }, a + 0.005)
      .to(q(`[data-ch="${n}"]`), { opacity: 0, y: -20, duration: 0.04 }, z);
  };
  ch(2, 0.10, 0.21); ch(3, 0.26, 0.41); ch(4, 0.46, 0.56); ch(5, 0.61, 0.76); ch(6, 0.80, 0.89);
  tl.fromTo(q('[data-ch="7"]'), { opacity: 0, scale: 0.96, y: 20 }, { opacity: 1, scale: 1, y: 0, duration: 0.06 }, 0.92);
  const count = (id, to, a, len, pre = "", post = "") => {
    const o = { v: 0 }, e = $(id);
    if (e) tl.fromTo(o, { v: 0 }, { v: to, duration: len, ease: "power2.out", immediateRender: false,
      onUpdate: () => { e.textContent = pre + fmt(Math.round(o.v)) + post; } }, a);
  };
  count("fl-count", data.counts.buildings_in_sample, 0.10, 0.05);
  count("fl-area", b.checks.vexcel_roof_m2, 0.27, 0.06, "", " m²");
  count("fl-pos", data.sky.steps.length, 0.47, 0.05);
  count("fl-panels", b.panels, 0.61, 0.14);   // in step with the panels appearing on the roof
  count("fl-kw", b.kw, 0.61, 0.14);
  count("fl-save", b.savings_year, 0.80, 0.06, "$");

  // ---- dock ----
  $("dock-chapters").innerHTML = CHAPTERS.map(([id, label, p]) =>
    `<button type="button" data-p="${p}" title="${label}"><span>${id}</span><i>${label}</i></button>`).join("");
  $("dock-chapters").querySelectorAll("button").forEach((btn) => btn.addEventListener("click", () => {
    const y = section.offsetTop + Number(btn.dataset.p) * (section.offsetHeight - innerHeight);
    if (window.__lenis) window.__lenis.scrollTo(y, { duration: 1.6 }); else scrollTo({ top: y, behavior: "smooth" });
  }));
  $("dock-sun").addEventListener("input", (e) => { S_.sunT = Number(e.target.value); view.setSun(); });
  view.installAll();  // every roof starts with its panels on; a click on a roof turns that roof off or on
  document.querySelectorAll(".dock-mode button").forEach((btn) => btn.addEventListener("click", () => setSurvey(btn.dataset.mode === "survey")));
  addEventListener("keydown", (e) => {
    if (e.key.toLowerCase() === "l" && !e.metaKey && !e.ctrlKey && !/input|textarea/i.test(document.activeElement.tagName)) setSurvey(!S_.surveyOn);
  });
  function setSurvey(on) {
    S_.surveyOn = on;
    document.querySelectorAll(".dock-mode button").forEach((x) => x.classList.toggle("on", (x.dataset.mode === "survey") === on));
  }
  view.setSun();

  // 360° free roam: drag to orbit, scroll to zoom, right-drag to pan. Shift + drag also starts it.
  const roamBtn = $("dock-roam"), stage = section.querySelector(".flight-stage");
  function setRoam(on) {
    if (S_.roam === on) return;
    S_.roam = on;
    view.roam(on);
    roamBtn.classList.toggle("on", on);
    roamBtn.textContent = on ? "Exit 360°" : "360° view";
    stage.classList.toggle("roaming", on);
    if (on) stage.setAttribute("data-lenis-prevent", ""); else stage.removeAttribute("data-lenis-prevent");
    if (window.__lenis) on ? window.__lenis.stop() : window.__lenis.start();
  }
  roamBtn.addEventListener("click", () => setRoam(!S_.roam));
  $("flight-canvas").addEventListener("pointerdown", (e) => { if (e.shiftKey && !S_.roam) setRoam(true); }, true);
  addEventListener("keydown", (e) => { if (e.key === "Escape" && S_.roam) setRoam(false); });

  $("flight-explore").addEventListener("click", () => {
    const target = document.querySelector(".ask");
    if (window.__lenis) window.__lenis.scrollTo(target, { offset: -70, duration: 1.4 }); else target.scrollIntoView({ behavior: "smooth" });
    if (onExplore) setTimeout(onExplore, 900);
  });

  // ---- cursor: parallax + hover + click ----
  const canvas = $("flight-canvas");
  section.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    S_.px = ((e.clientX - r.left) / r.width) * 2 - 1;
    S_.py = -(((e.clientY - r.top) / r.height) * 2 - 1);
    view.hover(e.clientX - r.left, e.clientY - r.top);
  });
  section.addEventListener("pointerleave", () => { S_.px = S_.py = 0; view.hover(-1, -1); });
  let down = null;  // a drag is not a click
  canvas.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  canvas.addEventListener("click", (e) => {
    if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    view.toggleHovered();
  });

  const clock = new THREE.Clock();
  (function loop() {
    requestAnimationFrame(loop);
    const dt = Math.min(clock.getDelta(), 0.05), damp = 1 - Math.exp(-8 * dt);
    S_.shown = THREE.MathUtils.lerp(S_.shown, S_.progress, damp);
    S_.spx = THREE.MathUtils.lerp(S_.spx, S_.px, damp * 0.7);
    S_.spy = THREE.MathUtils.lerp(S_.spy, S_.py, damp * 0.7);
    S_.survey = THREE.MathUtils.lerp(S_.survey, S_.surveyOn ? 1 : 0, damp);
    view.update(S_.shown, dt);
  })();
}

function fillText(d, b) {
  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
  set("fl-count", fmt(d.counts.buildings_in_sample));
  set("fl-city", d.city);
  set("fl-name", b.name || `CoStar ${b.id}`);
  set("fl-units", b.units ? `${fmt(b.units)} apartments` : "an apartment community");
  set("fl-roofs", `${b.buildings_on_site} building${b.buildings_on_site > 1 ? "s" : ""}`);
  set("fl-area", `${fmt(b.checks.vexcel_roof_m2)} m²`);
  set("fl-pos", fmt(d.sky.steps.length));
  set("fl-cloud", `${d.weather.cloud_loss_percent}%`);
  set("fl-ac", fmt(b.ac_room.ac_units));
  set("fl-panels", fmt(b.panels));
  set("fl-kw", fmt(b.kw));
  set("fl-save", "$" + fmt(b.savings_year));
  set("fl-unit", b.per_unit_month ? `$${fmt(b.per_unit_month)} per apartment, every month.` : "");
  set("fl-pay", `${b.payback_range[0]}–${b.payback_range[1]} years`);
}

function splitWords(el) {
  // Wrap each word (and each filled-in number span, whole) so the heading can rise in word by word.
  const parts = [];
  for (const node of [...el.childNodes]) {
    if (node.nodeType === 3) node.textContent.split(/(\s+)/).forEach((t) => t && parts.push(t));
    else parts.push(node);
  }
  if (el.querySelector("img")) return;  // the logo heading stays as it is
  el.textContent = "";
  for (const part of parts) {
    if (typeof part === "string" && /^\s+$/.test(part)) { el.append(" "); continue; }
    const w = document.createElement("span");
    w.className = "w";
    w.append(part);
    el.append(w);
  }
}

function inPoly(x, y, pts) {
  let c = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}

function facadeTextures(renderer) {
  // One tile = one window bay by one floor. Glass and frames; a few windows lit for the evening shot.
  // The canvas holds 4 x 4 bays so lit windows don't repeat on every bay.
  const mk = () => { const c = document.createElement("canvas"); c.width = 256; c.height = 256; return c; };
  const a = mk(), l = mk(), g = a.getContext("2d"), h = l.getContext("2d");
  g.fillStyle = "#c9ced6"; g.fillRect(0, 0, 256, 256);          // concrete
  h.fillStyle = "#000"; h.fillRect(0, 0, 256, 256);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    const x = c * 64, y = r * 64, f = Math.sin(r * 19.1 + c * 47.3) * 10000;
    g.fillStyle = "#2a3542"; g.fillRect(x + 10, y + 14, 44, 38); // glass
    g.fillStyle = "#3d4a5a"; g.fillRect(x + 10, y + 14, 44, 10); // sky reflection
    g.fillStyle = "#b5bbc4"; g.fillRect(x + 31, y + 14, 2, 38);  // mullion
    if (f - Math.floor(f) > 0.6) { h.fillStyle = "#fff"; h.fillRect(x + 10, y + 14, 44, 38); }
  }
  const out = [a, l].map((c, i) => {
    const t = new THREE.CanvasTexture(c);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(0.25, 0.25);
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    if (i === 0) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  });
  return { facade: out[0], lit: out[1] };
}

function build(b, data, canvas) {
  const sc = b.scene, ph = sc.photo, price = data.assumptions.power_price_usd_per_kwh;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  const day = new THREE.Color("#1B222B"), night = new THREE.Color("#0B0D10"), survey = new THREE.Color("#04070D"), work = new THREE.Color();
  scene.background = work.copy(day);
  scene.fog = new THREE.FogExp2(day, 0.0016);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.5, 5000);
  camera.up.set(0, 0, 1);
  const cx = (ph.x0 + ph.x1) / 2, cy = (ph.y0 + ph.y1) / 2, W = ph.x1 - ph.x0, H = ph.y1 - ph.y0, S = Math.max(W, H);
  const E = sc.eave_m;

  const hemi = new THREE.HemisphereLight(0xe6f0ff, 0x3a424e, 0.8);
  const sun = new THREE.DirectionalLight(0xfff6e6, 2.4);
  sun.target.position.set(cx, cy, 0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -S * 0.6, right: S * 0.6, top: S * 0.6, bottom: -S * 0.6, near: 1, far: S * 3 });
  sun.shadow.bias = -0.0004;
  scene.add(hemi, sun, sun.target);

  const tex = new THREE.TextureLoader().load(ph.src);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  const groundMat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.92 });
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(W, H), groundMat);
  ground.position.set(cx, cy, 0);
  ground.receiveShadow = true;
  scene.add(ground);

  // ---- buildings, roof sides (hoverable), outlines ----
  const walls = [], wallUV = [], roofMeshes = [], lines = [], edges = [];
  const floorH = sc.stories ? E / sc.stories : 3.2;  // one texture row per real floor
  const sides = sc.sides.map((s, i) => {
    const pts = s.pts, n = pts.length;
    for (let k = 0; k < n; k++) {
      const [ax, ay, az] = pts[k], [bx, by, bz] = pts[(k + 1) % n];
      walls.push(ax, ay, 0, bx, by, 0, bx, by, bz, ax, ay, 0, bx, by, bz, ax, ay, az);
      const u = Math.hypot(bx - ax, by - ay) / 3.2;  // a window bay every ~3 m
      wallUV.push(0, 0, u, 0, u, bz / floorH, 0, 0, u, bz / floorH, 0, az / floorH);
      edges.push(ax, ay, 0, ax, ay, az);  // vertical corner lines for the survey view
    }
    const tris = THREE.ShapeUtils.triangulateShape(pts.map(([x, y]) => new THREE.Vector2(x, y)), []);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pts.flatMap(([x, y, z]) => [x, y, z + 0.02]), 3));
    g.setIndex(tris.flat());
    g.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ color: 0x5a616b, roughness: 0.85, side: THREE.DoubleSide, transparent: true,
      emissive: 0x0284c7, emissiveIntensity: 0 });
    const m = new THREE.Mesh(g, mat);
    m.castShadow = m.receiveShadow = true;
    m.userData.side = i;
    roofMeshes.push(m);
    scene.add(m);
    const loop = [...pts, pts[0]].map(([x, y, z]) => new THREE.Vector3(x, y, z + 0.25));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(loop),
      new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0 }));
    line.userData = { count: loop.length, delay: Math.hypot(pts[0][0] - cx, pts[0][1] - cy) / S };
    lines.push(line);
    scene.add(line);
    let mx = 0, my = 0, mz = 0;
    for (const [x, y, z] of pts) { mx += x; my += y; mz = Math.max(mz, z); }
    let area = 0;  // shoelace: roof area seen from above
    for (let k = 0; k < n; k++) { const [x1, y1] = pts[k], [x2, y2] = pts[(k + 1) % n]; area += x1 * y2 - x2 * y1; }
    return { ...s, i, cx: mx / n, cy: my / n, top: mz, area_m2: Math.round(Math.abs(area) / 2), mat, frac: 0, panelsGeo: null };
  });
  const wg = new THREE.BufferGeometry();
  wg.setAttribute("position", new THREE.Float32BufferAttribute(walls, 3));
  wg.setAttribute("uv", new THREE.Float32BufferAttribute(wallUV, 2));
  wg.computeVertexNormals();
  const { facade, lit } = facadeTextures(renderer);
  const wallMat = new THREE.MeshStandardMaterial({ color: 0xffffff, map: facade, emissiveMap: lit, emissive: 0xffd9a0, emissiveIntensity: 0,
    roughness: 0.55, metalness: 0.15, side: THREE.DoubleSide, transparent: true });
  const wallMesh = new THREE.Mesh(wg, wallMat);
  wallMesh.castShadow = wallMesh.receiveShadow = true;
  scene.add(wallMesh);
  const eg = new THREE.BufferGeometry();
  eg.setAttribute("position", new THREE.Float32BufferAttribute(edges, 3));
  const edgeLines = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0 }));
  scene.add(edgeLines);

  // ---- "Analyze": usable roof glow; scan bar ----
  const usableMat = new THREE.MeshBasicMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide });
  sides.filter((s) => s.panels).forEach((s) => {
    const tris = THREE.ShapeUtils.triangulateShape(s.pts.map(([x, y]) => new THREE.Vector2(x, y)), []);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(s.pts.flatMap(([x, y, z]) => [x, y, z + 0.08]), 3));
    g.setIndex(tris.flat());
    scene.add(new THREE.Mesh(g, usableMat));
  });
  const scan = new THREE.Mesh(new THREE.PlaneGeometry(S * 1.2, S * 0.02),
    new THREE.MeshBasicMaterial({ color: 0x00a3ff, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }));
  scan.position.set(cx, cy, E + 6);
  scene.add(scan);

  // ---- panels: one mesh per roof side, so each side can be installed or removed ----
  const panelMat = new THREE.MeshStandardMaterial({ color: 0x163a8a, roughness: 0.55, metalness: 0.25, side: THREE.DoubleSide,
    emissive: 0x3d6bff, emissiveIntensity: 0 });
  const bySide = {};
  for (const p of sc.panels) (bySide[p[0]] ||= []).push(p);
  for (const s of sides) {
    const list = (bySide[s.i] || []).map((p) => {
      const mx = (p[1] + p[4] + p[7] + p[10]) / 4, my = (p[2] + p[5] + p[8] + p[11]) / 4;
      return [Math.hypot(mx - s.cx, my - s.cy), p];
    }).sort((a, z) => a[0] - z[0]).map((x) => x[1]);
    if (!list.length) continue;
    const pp = [];
    for (const p of list) {
      const c = [];
      for (let k = 1; k < 13; k += 3) c.push([p[k], p[k + 1], p[k + 2] + 0.05]);
      pp.push(...c[0], ...c[1], ...c[2], ...c[0], ...c[2], ...c[3]);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pp, 3));
    g.computeVertexNormals();
    g.setDrawRange(0, 0);
    const mesh = new THREE.Mesh(g, panelMat);
    mesh.castShadow = true;
    scene.add(mesh);
    s.panelsGeo = g;
    s.count = list.length;
    s.delay = Math.hypot(s.cx - cx, s.cy - cy) / (S * 0.5);
  }

  // ---- survey view: points sampled on every roof (from Vexcel outlines) and on the ground ----
  const pos = [], col = [];
  const lo = new THREE.Color("#0284c7"), hi = new THREE.Color("#e0f2fe"), tc = new THREE.Color();
  for (let x = ph.x0; x <= ph.x1; x += 3) for (let y = ph.y0; y <= ph.y1; y += 3) {
    pos.push(x + Math.sin(x * 12.9 + y * 78.2) * 0.6, y + Math.cos(x * 45.1 + y * 94.6) * 0.6, 0.2);
    col.push(lo.r * 0.55, lo.g * 0.55, lo.b * 0.55);
  }
  for (const s of sides) {
    const xs = s.pts.map((p) => p[0]), ys = s.pts.map((p) => p[1]);
    for (let x = Math.min(...xs); x <= Math.max(...xs); x += 0.9) for (let y = Math.min(...ys); y <= Math.max(...ys); y += 0.9) {
      if (!inPoly(x, y, s.pts)) continue;
      pos.push(x, y, s.top + 0.15);
      tc.copy(lo).lerp(hi, THREE.MathUtils.clamp(s.pct / 100, 0, 1));
      col.push(tc.r, tc.g, tc.b);
    }
  }
  const pg = new THREE.BufferGeometry();
  pg.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  pg.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  const cloud = new THREE.Points(pg, new THREE.PointsMaterial({ size: 1.5, vertexColors: true, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }));
  cloud.visible = false;
  scene.add(cloud);

  // ---- motion layer: radar rings, target beacon, rising energy ----
  const hdr = (r, g, bl) => new THREE.Color(r, g, bl);  // over 1.0, so only these bloom
  let fx0 = Infinity, fy0 = Infinity, fx1 = -Infinity, fy1 = -Infinity;
  for (const s of sides) for (const [x, y] of s.pts) { fx0 = Math.min(fx0, x); fy0 = Math.min(fy0, y); fx1 = Math.max(fx1, x); fy1 = Math.max(fy1, y); }
  const bcx = (fx0 + fx1) / 2, bcy = (fy0 + fy1) / 2, bR = Math.hypot(fx1 - fx0, fy1 - fy0) / 2;
  const rings = [0, 1, 2].map(() => {
    const m = new THREE.Mesh(new THREE.RingGeometry(0.97, 1, 96),
      new THREE.MeshBasicMaterial({ color: hdr(0.25, 1.6, 3.2), transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide, toneMapped: false }));
    m.position.set(bcx, bcy, 0.6);
    scene.add(m);
    return m;
  });
  const beacon = new THREE.Mesh(new THREE.CylinderGeometry(bR * 1.08, bR * 1.08, E * 2.2, 6, 4, true),
    new THREE.MeshBasicMaterial({ color: hdr(0.15, 1.2, 2.6), wireframe: true, transparent: true, opacity: 0, depthWrite: false, toneMapped: false }));
  beacon.rotation.x = Math.PI / 2;
  beacon.position.set(bcx, bcy, E * 1.1);
  scene.add(beacon);
  // energy rising from roofs that have panels on, in proportion to each roof's panel count
  const withPanels = sides.filter((s) => s.panelsGeo), totalPanels = withPanels.reduce((a, s) => a + s.count, 0) || 1;
  const eBase = [], ePos = [], eCol = [], eSide = [];
  const cyan = new THREE.Color("#38bdf8"), gold = new THREE.Color("#FFB224");
  for (const s of withPanels) {
    const xs = s.pts.map((p) => p[0]), ys = s.pts.map((p) => p[1]);
    const want = Math.max(6, Math.round(900 * s.count / totalPanels));
    for (let made = 0, tries = 0; made < want && tries < want * 30; tries++) {
      const x = THREE.MathUtils.lerp(Math.min(...xs), Math.max(...xs), Math.random());
      const y = THREE.MathUtils.lerp(Math.min(...ys), Math.max(...ys), Math.random());
      if (!inPoly(x, y, s.pts)) continue;
      made++;
      eBase.push(x, y, s.top + 0.4, Math.random());
      ePos.push(x, y, -999);
      tc.copy(gold).lerp(cyan, Math.random() * 0.6);
      eCol.push(tc.r * 2.2, tc.g * 2.2, tc.b * 2.2);
      eSide.push(s);
    }
  }
  const eg2 = new THREE.BufferGeometry();
  eg2.setAttribute("position", new THREE.Float32BufferAttribute(ePos, 3));
  eg2.setAttribute("color", new THREE.Float32BufferAttribute(eCol, 3));
  const energy = new THREE.Points(eg2, new THREE.PointsMaterial({ size: 2.4, vertexColors: true, transparent: true, opacity: 0.9,
    depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false }));
  energy.frustumCulled = false;
  scene.add(energy);

  // ---- post: bloom on the bright marks only ----
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.7, 0.45, 1.0);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  // ---- 360° free roam ----
  const controls = new OrbitControls(camera, canvas);
  Object.assign(controls, { enabled: false, enableDamping: true, dampingFactor: 0.06, maxPolarAngle: Math.PI / 2.1,
    minDistance: E * 1.5, maxDistance: S * 2.2, screenSpacePanning: false });
  controls.target.set(bcx, bcy, E * 0.5);

  // ---- camera path ----
  const P = (x, y, z) => new THREE.Vector3(cx + x * S, cy + y * S, z * S);
  const path = new THREE.CatmullRomCurve3([P(-0.55, -0.72, 0.5), P(-0.38, -0.52, 0.45), P(-0.05, -0.14, 0.62),
    P(0, -0.02, 0.55), P(0.08, -0.22, 0.42), P(0.45, -0.7, 0.55), P(0.75, -1.15, 1.2)], false, "catmullrom", 0.35);
  const look = new THREE.CatmullRomCurve3([P(0, 0.08, 0), P(0, 0, 0.01), P(0, 0, 0.02), P(0, 0, 0.02),
    P(0, 0, 0.02), P(0, 0, 0.01), P(0, 0.12, 0.08)], false, "catmullrom", 0.35);
  const cp = new THREE.Vector3(), cl = new THREE.Vector3();

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    composer.setPixelRatio(renderer.getPixelRatio());
    composer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  // ---- interaction ----
  const ray = new THREE.Raycaster(), tip = $("flight-tip");
  let hovered = -1, progress = 0;
  function hover(x, y) {
    hovered = -1;
    if (x >= 0 && progress < 0.91) {
      ray.setFromCamera(new THREE.Vector2((x / canvas.clientWidth) * 2 - 1, -(y / canvas.clientHeight) * 2 + 1), camera);
      const hit = ray.intersectObjects(roofMeshes)[0];
      if (hit) hovered = hit.object.userData.side;
    }
    canvas.style.cursor = hovered >= 0 ? "pointer" : "";
    if (hovered < 0) { tip.hidden = true; return; }
    const s = sides[hovered], ov = S_.overrides[s.i];
    tip.innerHTML = `<header><span>Roof ${s.i + 1} of ${sides.length}</span><em>${fmt(s.area_m2)} m²</em></header>` +
      `<div class="tg"><div><small>${s.layout && s.layout.flat ? "Flat roof" : "Faces " + facingWord(s.facing)}</small><b>${s.pct}% of best sun</b></div>` +
      `<div><small>Panels</small><b>${s.panels ? fmt(s.panels) : "none"}</b></div>` +
      `<div><small>System</small><b>${fmt(s.kw)} kW</b></div><div><small>Saves</small><b>$${fmt(Math.round(s.kwh * price))}/yr</b></div></div>` +
      `<footer>${s.panels ? (ov === undefined ? "Click to install or remove its panels" : "Click to switch its panels " + (ov ? "off" : "on")) : "Too little sun for panels"}</footer>`;
    tip.hidden = false;
    tip.style.left = Math.min(x + 18, canvas.clientWidth - 250) + "px";
    tip.style.top = Math.max(10, y - 20) + "px";
  }
  function toggleHovered() {
    if (hovered < 0) return;
    const s = sides[hovered];
    if (!s.panelsGeo) return;
    const cur = S_.overrides[s.i];
    S_.overrides[s.i] = cur === undefined ? s.frac < 0.5 : !cur;
    hover(parseFloat(tip.style.left) - 18, parseFloat(tip.style.top) + 20);
  }
  function installAll() {
    const allOn = sides.every((s) => !s.panelsGeo || S_.overrides[s.i] === true);
    for (const s of sides) if (s.panelsGeo) { if (allOn) delete S_.overrides[s.i]; else S_.overrides[s.i] = true; }
  }
  // Sun through a typical June day, from the real sun steps (NOAA positions).
  const june = data.sky.steps.filter((st) => st[0] === 5).sort((a, z) => a[7] - z[7]);
  function setSun() {
    const st = june[Math.round(S_.sunT * (june.length - 1))], rad = Math.PI / 180, R = S * 0.9;
    const el = st[1] * rad, az = st[2] * rad;
    sun.position.set(cx + R * Math.cos(el) * Math.sin(az), cy + R * Math.cos(el) * Math.cos(az), R * Math.sin(el));
    sun.intensity = 0.7 + 2.0 * Math.sin(el);
    const m = st[7], hm = `${((Math.floor(m / 60) + 11) % 12) + 1}:${String(m % 60).padStart(2, "0")} ${m < 720 ? "am" : "pm"}`;
    $("dock-sun-label").textContent = hm;
  }

  let time = 0, back = 0;
  const lookNow = new THREE.Vector3();
  function roam(on) {
    controls.enabled = on;
    if (on) { controls.target.copy(cl); controls.update(); } else back = 1;  // glide back onto the flight path
  }

  return {
    hover, toggleHovered, installAll, setSun, roam,
    update(p, dt) {
      progress = p; time += dt;
      const damp = 1 - Math.exp(-12 * dt);
      path.getPoint(p, cp); look.getPoint(p, cl);
      const par = (p > 0.45 && p < 0.6 ? 0.006 : 0.016) * S;  // cursor gimbal, calmer during top-down analysis
      cp.x += S_.spx * par; cp.y += S_.spy * par * 0.6;
      if (controls.enabled) controls.update();
      else if (back > 0.001) {
        back *= 1 - (1 - Math.exp(-3 * dt));
        camera.position.lerp(cp, 1 - back);
        lookNow.copy(controls.target).lerp(cl, 1 - back);
        controls.target.copy(lookNow);
        camera.lookAt(lookNow);
      } else { camera.position.copy(cp); camera.lookAt(cl); }
      // radar rings ripple out from the building while we look for it (faster when you scroll faster)
      const radar = smooth(p, 0.12, 0.2) * (1 - smooth(p, 0.52, 0.62));
      rings.forEach((r, i) => {
        const c = (time * (0.3 + S_.speed * 0.6) + i / 3) % 1;
        r.scale.setScalar(bR + c * S * 0.55);
        r.material.opacity = (1 - c) * 0.6 * radar;
        r.visible = radar > 0.01;
      });
      const bA = smooth(p, 0.26, 0.32) * (1 - smooth(p, 0.56, 0.62));
      beacon.visible = bA > 0.01;
      beacon.rotation.y = time * 0.5;
      beacon.material.opacity = bA * (0.22 + Math.sin(time * 3) * 0.06);
      // energy rises from every roof whose panels are on
      const ea = energy.geometry.attributes.position;
      for (let i = 0; i < eSide.length; i++) {
        const s = eSide[i], o = i * 4;
        if (s.frac < 0.5 || p > 0.93) { ea.setZ(i, -999); continue; }
        const t = (time * 0.22 + eBase[o + 3]) % 1;
        ea.setXYZ(i, eBase[o] + Math.sin(time + i) * 0.3, eBase[o + 1] + Math.cos(time * 0.8 + i) * 0.3, eBase[o + 2] + t * E * 0.9);
      }
      ea.needsUpdate = true;
      energy.material.opacity = 0.85 * (1 - S_.survey * 0.6);
      const sv = S_.survey, blend = smooth(p, 0.89, 0.97);
      work.copy(day).lerp(night, blend).lerp(survey, sv);
      scene.background = work; scene.fog.color.copy(work);
      scene.fog.density = THREE.MathUtils.lerp(THREE.MathUtils.lerp(0.0016, 0.0019, blend), 0.003, sv);
      // survey view: photo darkens, buildings turn to glass, points and edges glow
      groundMat.color.setScalar(1 - 0.92 * sv);
      wallMat.opacity = 1 - 0.6 * sv;
      wallMat.emissiveIntensity = blend * 1.4 * (1 - sv);  // windows light up as night falls on the last chapters
      edgeLines.material.opacity = sv * 0.8;
      cloud.visible = sv > 0.01;
      cloud.material.opacity = sv * 0.9;
      hemi.intensity = 0.8 - 0.4 * sv;
      // detect
      const alpha = Math.max(smooth(p, 0.22, 0.28) * (1 - smooth(p, 0.84, 0.91)), sv);
      for (const l of lines) {
        const t = sv > 0.5 ? 1 : THREE.MathUtils.clamp((p - (0.25 + l.userData.delay * 0.12)) / 0.1, 0, 1);
        l.geometry.setDrawRange(0, Math.max(2, Math.ceil(l.userData.count * t)));
        l.material.opacity = t * alpha;
      }
      const st = THREE.MathUtils.clamp((p - 0.24) / 0.21, 0, 1);
      scan.position.y = cy + THREE.MathUtils.lerp(-H / 2, H / 2, st);
      scan.material.opacity = smooth(p, 0.24, 0.27) * (1 - smooth(p, 0.42, 0.46)) * 0.5;
      usableMat.opacity = smooth(p, 0.44, 0.5) * (1 - smooth(p, 0.6, 0.66)) * 0.35;
      // panels: scroll reveals them side by side, clicks override; totals update live
      const k = THREE.MathUtils.clamp((p - 0.58) / 0.27, 0, 1);
      let on = 0, kw = 0, save = 0, roofsOn = 0, roofsAll = 0;
      for (const s of sides) {
        const isHover = s.i === hovered;
        s.mat.emissiveIntensity = THREE.MathUtils.lerp(s.mat.emissiveIntensity, isHover ? 0.16 : 0, damp);  // gentle: stronger blows out under bloom
        s.mat.color.set(isHover ? 0x3b82a8 : 0x5a616b);
        s.mat.opacity = 1 - 0.55 * sv;
        if (!s.panelsGeo) continue;
        roofsAll++;
        const ov = S_.overrides[s.i];
        const target = ov !== undefined ? (ov ? 1 : 0) : THREE.MathUtils.clamp((k - s.delay * 0.55) / 0.45, 0, 1);
        s.frac = THREE.MathUtils.lerp(s.frac, target, damp);
        if (Math.abs(s.frac - target) < 0.002) s.frac = target;
        s.panelsGeo.setDrawRange(0, Math.round(s.count * s.frac) * 6);
        const share = Math.round(s.count * s.frac) / s.count;
        on += Math.round(s.count * s.frac); kw += s.kw * share; save += s.kwh * price * share;
        if (s.frac > 0.5) roofsOn++;
      }
      $("dock-roofs").textContent = `${roofsOn}/${roofsAll} roofs`;
      $("dock-kw").textContent = `${fmt(on)} panels · ${fmt(Math.round(kw))} kW`;
      $("dock-save").textContent = `$${fmt(Math.round(save))}/yr`;
      if (p >= 0.91) $("flight-tip").hidden = true;
      composer.render();
    },
  };
}
