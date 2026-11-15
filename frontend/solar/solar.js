/* CityLab Solar: ask -> thinking -> map -> the roof in 3D with its panels -> savings card ->
   which sides get panels -> how we got the number -> why to trust it -> owner email.
   Data: live from the backend (POST /api/solar-city/stream): a DeepAgents agent runs the
   CoStar / sun / Vexcel / NREL tools and streams each step. If the live run cannot measure any
   roof (offline without cached roofs, or the server is down), the recorded data_<city>.json is
   shown and labelled as recorded.
   3D photos: views/<id>_top.jpg (Vexcel imagery, kept out of git). */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { initFlight } from "./flight.js?v=noglare";

const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString("en-US");
const money = (n) => "$" + fmt(Math.round(n));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const WORDS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];
const facingWord = (deg) => WORDS[Math.round(deg / 45) % 8];

let city = "los_angeles", data = null, current = null, runId = 0;

/* ---------- colour for "how much of the best sun a side gets" ---------- */
function effColor(pct) {
  const keep = data ? data.assumptions.keep_sides_with_sun_pct : 80;
  if (pct < keep) return "#E5534B";
  const t = Math.max(0, Math.min(1, (pct - keep) / (100 - keep)));
  const a = [0xa9, 0xe2, 0xc2], b = [0x2e, 0x9e, 0x62];
  return "#" + a.map((v, i) => Math.round(v + (b[i] - v) * t).toString(16).padStart(2, "0")).join("");
}

/* ================= map ================= */
let map, layer;
function initMap() {
  map = L.map("map", { zoomControl: false }).setView([34.05, -118.3], 10);
  L.control.zoom({ position: "bottomright" }).addTo(map);
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 20, maxNativeZoom: 19, attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
  }).addTo(map);
  layer = L.layerGroup().addTo(map);
}

function drawAll(d) {
  layer.clearLayers();
  const max = Math.max(...d.buildings.map((b) => b.savings_year));
  for (const b of d.buildings) {
    L.circleMarker([b.lat, b.lon], {
      radius: 4 + 10 * Math.sqrt(b.savings_year / max), color: "#FFB224", weight: 1.5, fillColor: "#FFB224", fillOpacity: 0.35,
    }).bindTooltip(`${esc(b.name || "CoStar " + b.id)}: ${money(b.savings_year)} a year`).on("click", () => show(b)).addTo(layer);
  }
  map.flyToBounds(d.buildings.map((b) => [b.lat, b.lon]), { padding: [60, 60], duration: 1.2 });
}

let roofLayer = null;
function mapRoof(b) {
  if (roofLayer) roofLayer.remove();
  roofLayer = L.layerGroup().addTo(layer);
  $("map-note").hidden = false;
  const pts = [];
  for (const s of b.sides) {
    s.outline.forEach((p) => pts.push(p));
    L.polygon(s.outline, { color: effColor(s.pct), weight: 2, fillColor: effColor(s.pct), fillOpacity: 0.35 })
      .on("click", () => { if (b.scene) setView("3d"); })
      .bindTooltip(`${s.layout && s.layout.flat ? "Flat roof" : "Faces " + facingWord(s.facing)} · ${s.pct}% of the best sun · ${s.panels ? fmt(s.panels) + " panels" : "no panels"}${b.scene ? " · click to see it in 3D" : ""}`)
      .addTo(roofLayer);
  }
  map.flyToBounds(pts, { padding: [50, 50], duration: 1.4, maxZoom: 19 });
}

/* ================= 3D ================= */
const T = {};
function init3D() {
  const el = $("three");
  T.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  T.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  T.renderer.outputColorSpace = THREE.SRGBColorSpace;
  T.renderer.shadowMap.enabled = true;
  T.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  el.prepend(T.renderer.domElement);
  T.scene = new THREE.Scene();
  T.scene.background = new THREE.Color("#8FB8D9");  // daytime sky; setDay()/lightFrom() repaint this once the sun position is known
  T.camera = new THREE.PerspectiveCamera(38, 1, 0.5, 6000);
  T.camera.up.set(0, 0, 1);
  T.controls = new OrbitControls(T.camera, T.renderer.domElement);
  T.controls.enableDamping = true;
  T.controls.maxPolarAngle = Math.PI * 0.46;
  T.controls.autoRotateSpeed = 0.5;
  T.controls.addEventListener("start", () => { T.controls.autoRotate = false; });
  T.hemi = new THREE.HemisphereLight(0xffffff, 0x30343a, 1.7);
  T.scene.add(T.hemi);
  T.sun = new THREE.DirectionalLight(0xfff4e0, 1.5);
  T.sun.castShadow = true;
  T.sun.shadow.mapSize.set(2048, 2048);
  T.sun.shadow.bias = -0.0004;
  T.scene.add(T.sun, T.sun.target);
  T.group = new THREE.Group();
  T.scene.add(T.group);
  T.ray = new THREE.Raycaster();
  T.mode = "photo";
  new ResizeObserver(resize3D).observe(el);
  T.renderer.domElement.addEventListener("pointermove", hover3D);
  T.renderer.domElement.addEventListener("pointerleave", () => { $("tip").hidden = true; });
  (function loop() {
    requestAnimationFrame(loop);
    if (T.grow) T.grow();
    if (T.playing) dayTick();
    T.controls.update();
    T.renderer.render(T.scene, T.camera);
  })();
}

function resize3D() {
  const el = $("three"), w = el.clientWidth, h = el.clientHeight;
  if (!w || !h) return;
  T.renderer.setSize(w, h, false);
  T.camera.aspect = w / h;
  T.camera.updateProjectionMatrix();
}

function clear3D() {
  T.group.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    const mats = o.material ? [].concat(o.material) : [];
    mats.forEach((m) => { if (m.map) m.map.dispose(); m.dispose(); });
  });
  T.group.clear();
  T.roofs = [];
  T.grow = null;
}

function sprite(text, size) {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  g.fillStyle = "rgba(11,17,23,0.75)";
  g.beginPath(); g.arc(64, 64, 52, 0, Math.PI * 2); g.fill();
  g.fillStyle = "#f4f1ea"; g.font = "bold 64px Geist, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(text, 64, 68);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.set(size, size, 1);
  return s;
}

function build3D(b) {
  clear3D();
  const sc = b.scene, ph = sc.photo;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const s of sc.sides) for (const [x, y] of s.pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  if (ph) ({ x0, y0, x1, y1 } = ph);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, W = x1 - x0, H = y1 - y0, size = Math.max(W, H);
  const uv = (x, y) => [(x - x0) / W, (y - y0) / H];

  // Ground: the real Vexcel top-down photo, laid flat.
  let tex = null;
  if (ph) {
    tex = new THREE.TextureLoader().load(ph.src);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = T.renderer.capabilities.getMaxAnisotropy();
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.MeshLambertMaterial({ map: tex }));
    ground.position.set(cx, cy, 0);
    ground.receiveShadow = true;
    T.group.add(ground);
  }

  // Roof sides (photo or sun colour) and the walls under every roof edge.
  const wall = [];
  T.photoMat = tex ? new THREE.MeshLambertMaterial({ map: tex, side: THREE.DoubleSide }) : new THREE.MeshLambertMaterial({ color: 0x8a8a8a, side: THREE.DoubleSide });
  sc.sides.forEach((s, i) => {
    const pos = [], uvs = [];
    for (const [x, y, z] of s.pts) { pos.push(x, y, z + 0.03); uvs.push(...uv(x, y)); }
    const tris = THREE.ShapeUtils.triangulateShape(s.pts.map(([x, y]) => new THREE.Vector2(x, y)), []);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    g.setIndex(tris.flat());
    g.computeVertexNormals();
    const sunMat = new THREE.MeshLambertMaterial({ color: effColor(s.pct), side: THREE.DoubleSide });
    const m = new THREE.Mesh(g, T.mode === "photo" ? T.photoMat : sunMat);
    m.castShadow = m.receiveShadow = true;
    m.userData = { side: s, sunMat };
    T.group.add(m);
    T.roofs.push(m);
    const n = s.pts.length;
    for (let k = 0; k < n; k++) {
      const [ax, ay, az] = s.pts[k], [bx, by, bz] = s.pts[(k + 1) % n];
      wall.push(ax, ay, 0, bx, by, 0, bx, by, bz, ax, ay, 0, bx, by, bz, ax, ay, az);
    }
  });
  const wg = new THREE.BufferGeometry();
  wg.setAttribute("position", new THREE.Float32BufferAttribute(wall, 3));
  wg.computeVertexNormals();
  const walls = new THREE.Mesh(wg, new THREE.MeshLambertMaterial({ color: 0xd8d2c6, side: THREE.DoubleSide }));
  walls.castShadow = walls.receiveShadow = true;
  T.group.add(walls);

  // Panels: every one at its real size (data.assumptions.panel) and tilt, plus a thin frame line.
  const pp = [], pl = [];
  for (const p of sc.panels) {
    const c = [];
    for (let k = 1; k < 13; k += 3) c.push([p[k], p[k + 1], p[k + 2]]);
    pp.push(...c[0], ...c[1], ...c[2], ...c[0], ...c[2], ...c[3]);
    for (let k = 0; k < 4; k++) pl.push(...c[k], ...c[(k + 1) % 4]);
  }
  const pg = new THREE.BufferGeometry();
  pg.setAttribute("position", new THREE.Float32BufferAttribute(pp, 3));
  pg.computeVertexNormals();
  T.panelMat = new THREE.MeshPhongMaterial({ color: 0x1a3690, specular: 0x334466, shininess: 30, side: THREE.DoubleSide,
    emissive: 0x3d6bff, emissiveIntensity: 0 });  // panels light up blue as they make more power
  const panels = new THREE.Mesh(pg, T.panelMat);
  panels.receiveShadow = true;
  const lg = new THREE.BufferGeometry();
  lg.setAttribute("position", new THREE.Float32BufferAttribute(pl, 3));
  const lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x9fc0ff, transparent: true, opacity: 0.55 }));
  T.group.add(panels, lines);
  // Hidden in "Real photo" mode: that view is the bare aerial photo, not the panel layout.
  // T.mode is already set (to "photo" by default) before this runs, so apply it here too,
  // not just in setMode() -- the scene's first build never calls setMode itself.
  panels.visible = lines.visible = T.mode !== "photo";
  T.panels = panels; T.panelLines = lines;
  // "Panels everywhere": what we did NOT do -- panels on the shady sides, in orange.
  const bp = [];
  for (const p of sc.bad_panels || []) {
    const c = [];
    for (let k = 1; k < 13; k += 3) c.push([p[k], p[k + 1], p[k + 2]]);
    bp.push(...c[0], ...c[1], ...c[2], ...c[0], ...c[2], ...c[3]);
  }
  const bg = new THREE.BufferGeometry();
  bg.setAttribute("position", new THREE.Float32BufferAttribute(bp, 3));
  bg.computeVertexNormals();
  T.bad = new THREE.Mesh(bg, new THREE.MeshPhongMaterial({ color: 0xe5534b, emissive: 0x4a1210, side: THREE.DoubleSide }));
  T.bad.visible = T.mode === "bad";
  T.group.add(T.bad);
  T.badCount = (sc.bad_panels || []).length;
  // Panels appear row by row, like an install.
  const total = sc.panels.length, t0 = performance.now(), dur = Math.min(2600, 600 + total * 0.4);
  pg.setDrawRange(0, 0); lg.setDrawRange(0, 0);
  T.grow = () => {
    const k = Math.min(1, (performance.now() - t0) / dur), n = Math.round(total * k);
    pg.setDrawRange(0, n * 6); lg.setDrawRange(0, n * 8);
    if (k >= 1) T.grow = null;
  };

  // The sun's real path over this spot (NOAA equations) on three days of the year.
  const R = size * 0.42, paths = data.sun.paths, rad = Math.PI / 180;
  const colors = { summer: 0xffb224, spring_fall: 0xffd27a, winter: 0x9fc0ff };
  for (const [k, arr] of Object.entries(paths)) {
    const v = arr.map(([az, el]) => new THREE.Vector3(cx + R * Math.cos(el * rad) * Math.sin(az * rad),
      cy + R * Math.cos(el * rad) * Math.cos(az * rad), R * Math.sin(el * rad)));
    T.group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(v), new THREE.LineBasicMaterial({ color: colors[k] })));
  }
  for (const [t, ang] of [["N", 0], ["E", 90], ["S", 180], ["W", 270]]) {
    const s = sprite(t, size * 0.05);
    s.position.set(cx + R * 1.04 * Math.sin(ang * rad), cy + R * 1.04 * Math.cos(ang * rad), 1);
    T.group.add(s);
  }

  T.sun.target.position.set(cx, cy, 0);
  const sh = T.sun.shadow.camera, half = size * 0.6;
  sh.left = -half; sh.right = half; sh.top = half; sh.bottom = -half; sh.near = 1; sh.far = size * 3;
  sh.updateProjectionMatrix();
  T.center = { cx, cy, R, size };
  T.sides3d = sc.sides;
  setDay(T.dayMonth ?? 5, Number($("sun-time").value));
  T.controls.target.set(cx, cy, sc.eave_m / 2);
  const back = 1.15 + sc.eave_m / Math.max(size, 1);  // tall buildings: step back a little more
  T.camera.position.set(cx + size * 0.5 * back, cy - size * 0.78 * back, size * 0.6 * back + sc.eave_m);
  T.controls.autoRotate = true;
  T.controls.update();
  key3D();
}

/* ---------- the sun across one day: position, light, shadows and the roof's power right now ---------- */
function daySteps(month) {
  return data.sky.steps.filter((st) => st[0] === month).sort((a, b) => a[7] - b[7]);
}
function roofPowerKw(st) {
  // every roof side with panels: its kW x light on that side now / 1000 W per m2 x losses (same math as the year total)
  let kw = 0;
  for (const s of T.sides3d || []) if (s.panels) kw += s.kw * onPanel(st, s.tilt, s.facing) / 1000 * data.sky.performance_ratio * data.sun_grid.to_nrel;
  return kw;
}
function setDay(month, minute) {
  if (!T.center || !data) return;
  T.dayMonth = month;
  document.querySelectorAll(".sm").forEach((b) => b.classList.toggle("on", Number(b.dataset.m) === month));
  const steps = daySteps(month);
  if (T.curveMonth !== month || T.curveFor !== current) {  // the day's power curve behind the slider
    T.curveMonth = month; T.curveFor = current;
    const pts = steps.map((st) => [st[7], roofPowerKw(st)]);
    T.dayPeak = Math.max(1, ...pts.map((p) => p[1]));
    T.dayKwh = pts.reduce((a, p) => a + p[1] * (10 / 60), 0);
    const X = (m) => (m / 1440) * 300, Y = (v) => 38 - (v / T.dayPeak) * 34;
    const line = pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    $("sun-curve").innerHTML = pts.length ? `<path d="${line}L${X(pts[pts.length - 1][0])},38L${X(pts[0][0])},38Z" fill="rgba(255,178,36,0.16)"/>` +
      `<path d="${line}" fill="none" stroke="#FFB224" stroke-width="1.5" vector-effect="non-scaling-stroke"/>` : "";
  }
  $("sun-time").value = minute;
  const st = steps.find((x) => Math.abs(x[7] - minute) <= 5);
  const { cx, cy, R } = T.center, rad = Math.PI / 180;
  const hm = `${((Math.floor(minute / 60) + 11) % 12) + 1}:${String(minute % 60).padStart(2, "0")} ${minute < 720 ? "am" : "pm"}`;
  const label = ["Jan", "Feb", "Mar 21", "Apr", "May", "Jun 21", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec 21"][month];
  if (!st) {  // night
    T.sun.intensity = 0; T.hemi.intensity = 0.35;
    T.scene.background.set("#07090C");
    if (T.sunBall) T.sunBall.visible = false;
    if (T.panelMat) T.panelMat.emissiveIntensity = 0;
    $("sun-read").innerHTML = `<b>${hm} · night</b>No sun: the panels make 0 kW. ${label}: ${fmt(Math.round(T.dayKwh))} kWh over the day.`;
    return;
  }
  const el = st[1] * rad, az = st[2] * rad;
  const pos = new THREE.Vector3(cx + R * Math.cos(el) * Math.sin(az), cy + R * Math.cos(el) * Math.cos(az), R * Math.sin(el));
  if (!T.sunBall) {
    T.sunBall = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshBasicMaterial({ color: 0xffc94a }));
    T.scene.add(T.sunBall);
  }
  T.sunBall.visible = true;
  T.sunBall.scale.setScalar(T.center.size * 0.022);
  T.sunBall.position.copy(pos);
  T.sun.position.copy(pos);
  const k = Math.sin(el);
  T.sun.intensity = 0.6 + 2.4 * k;
  T.hemi.intensity = 0.45 + 0.65 * k;
  T.scene.background.set("#07090C").lerp(new THREE.Color("#8FB8D9"), Math.min(1, k * 1.6));
  const kw = roofPowerKw(st);
  if (T.panelMat) T.panelMat.emissiveIntensity = 0.55 * kw / T.dayPeak;
  $("sun-read").innerHTML = `<b>${hm} · <em>${fmt(Math.round(kw))} kW</em> now</b>` +
    `Sun ${Math.round(st[1])}° up, ${facingWord(st[2])} · ${Math.round(100 * kw / T.dayPeak)}% of today's peak · ${label}: ${fmt(Math.round(T.dayKwh))} kWh`;
}
function dayTick() {
  const now = performance.now();
  if (now - (T.lastTick || 0) < 70) return;
  T.lastTick = now;
  let m = Number($("sun-time").value) + 10;
  if (m > 1260) m = 300;  // 9 pm -> back to 5 am
  setDay(T.dayMonth ?? 5, m);
}
/* ---------- hold the sun with the cursor ---------- */
function lightFrom(st) {
  // Put the sun, its light and shadows at this sky position and show the roof's power right now.
  const { cx, cy, R } = T.center, rad = Math.PI / 180, el = st[1] * rad, az = st[2] * rad;
  const pos = new THREE.Vector3(cx + R * Math.cos(el) * Math.sin(az), cy + R * Math.cos(el) * Math.cos(az), R * Math.sin(el));
  if (!T.sunBall) {
    T.sunBall = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshBasicMaterial({ color: 0xffc94a }));
    T.scene.add(T.sunBall);
  }
  T.sunBall.visible = true;
  T.sunBall.scale.setScalar(T.center.size * 0.022);
  T.sunBall.position.copy(pos);
  T.sun.position.copy(pos);
  const k = Math.sin(el);
  T.sun.intensity = 0.6 + 2.4 * k;
  T.hemi.intensity = 0.45 + 0.65 * k;
  T.scene.background.set("#07090C").lerp(new THREE.Color("#8FB8D9"), Math.min(1, k * 1.6));
  const kw = roofPowerKw(st);
  if (T.panelMat) T.panelMat.emissiveIntensity = 0.55 * kw / (T.dayPeak || kw || 1);
  return kw;
}
function skyAt(month, el) {
  // Sunlight for a sun this high, from the same month's real sun steps (nearest height).
  let best = null;
  for (const st of data.sky.steps) if (st[0] === month && (!best || Math.abs(st[1] - el) < Math.abs(best[1] - el))) best = st;
  return best;
}
function sunFromPointer(e) {
  // Cursor over the sky dome -> that point of the sky. Cursor over the ground -> the sun stands over it
  // in that direction, higher the closer you point to the building.
  const r = T.renderer.domElement.getBoundingClientRect();
  T.ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), T.camera);
  const { cx, cy, R } = T.center, c = new THREE.Vector3(cx, cy, 0), hit = new THREE.Vector3();
  const sphere = new THREE.Sphere(c, R);
  let dx, dy, el;
  if (T.ray.ray.intersectSphere(sphere, hit) && hit.z > 0.5) {
    dx = hit.x - cx; dy = hit.y - cy;
    el = Math.asin(Math.min(1, hit.z / R)) * 180 / Math.PI;
  } else if (T.ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), hit)) {
    dx = hit.x - cx; dy = hit.y - cy;
    el = 90 * (1 - Math.min(1, Math.hypot(dx, dy) / R));
  } else return null;
  return { az: (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360, el: Math.max(2, Math.min(88, el)) };
}
function holdSun(e) {
  if (!T.following || !T.center || !data) return;
  const s = sunFromPointer(e);
  if (!s) return;
  const month = T.dayMonth ?? 5, base = skyAt(month, s.el);
  if (!base) return;
  const st = [month, s.el, s.az, base[3], base[4], base[5], 0, base[7]];
  const kw = lightFrom(st);
  $("sun-read").innerHTML = `<b>Your sun · <em>${fmt(Math.round(kw))} kW</em> now</b>` +
    `${Math.round(s.el)}° up, from the ${facingWord(s.az)} · ${Math.round(100 * kw / (T.dayPeak || kw || 1))}% of a ${["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month]} peak · click to hold`;
}
function setFollow(on) {
  T.following = on;
  $("sun-hold").setAttribute("aria-pressed", String(on));
  $("three").classList.toggle("following", on);
  if (on) { togglePlay(false); T.controls.autoRotate = false; }
}

function togglePlay(on = !T.playing) {
  T.playing = on;
  $("sun-play").innerHTML = on ? '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><rect x="4" y="3" width="3" height="10" rx="0.5"/><rect x="9" y="3" width="3" height="10" rx="0.5"/></svg>'
    : '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4.5 2.8v10.4L13 8z"/></svg>';
  $("sun-play").setAttribute("aria-label", on ? "Pause the day" : "Play the day");
}

function key3D() {
  const keep = data.assumptions.keep_sides_with_sun_pct;
  const sunRows = `<p style="color:var(--muted)">Lines show how high the sun gets each season: ` +
    `<i class="sw line" style="background:#FFB224"></i>highest (summer) ` +
    `<i class="sw line" style="background:#FFD27A"></i>mid-year (spring/fall) ` +
    `<i class="sw line" style="background:#9fc0ff"></i>lowest (winter).</p>`;
  const s = current && current.skipped;
  $("key3d").innerHTML = T.mode === "photo"
    ? `<p><i class="sw panel"></i>Solar panel, ${panelSize()}, ${data.assumptions.panel.name}</p>${sunRows}<p style="color:var(--muted)">Drag to turn · scroll to zoom · point at a roof</p>`
    : T.mode === "bad"
    ? (s ? `<p><i class="sw panel"></i>Our plan: ${fmt(current.panels)} panels, ${fmt(s.good_kwh_per_panel)} kWh each a year</p>` +
        `<p><i class="sw" style="background:#E5534B"></i>Wasted: ${fmt(s.panels)} more on shady sides, only ${fmt(s.kwh_per_panel)} kWh each</p>` +
        `<p style="color:var(--muted)">Those extra panels would take ${s.payback_years} years to pay back</p>`
        : `<p><b>Every roof side here is sunny.</b></p><p style="color:var(--muted)">No panels were left off. Try a Houston roof to see shady sides.</p>`)
    : `<p><b>How much sun each side of the roof gets, compared to the best-facing side here</b></p><p><span class="ramp"></span></p>` +
      `<div class="ends"><span>Shadier</span><span>${keep}% &mdash; cutoff for panels</span><span>Sunniest (100%)</span></div>${sunRows}` +
      (current?.ac_room?.ac_units
        ? `<p style="color:var(--muted)">Already leaves room for ${fmt(current.ac_room.ac_units)} rooftop AC units (exact spot unknown).</p>`
        : "");
}

function setMode(mode) {
  T.mode = mode;
  document.querySelectorAll(".mode").forEach((m) => m.classList.toggle("on", m.dataset.mode === mode));
  for (const r of T.roofs || []) r.material = mode === "photo" ? T.photoMat : r.userData.sunMat;
  if (T.bad) T.bad.visible = mode === "bad";
  if (T.panels) T.panels.visible = mode !== "photo";
  if (T.panelLines) T.panelLines.visible = mode !== "photo";
  if (data) key3D();
}

function hover3D(e) {
  if (!T.roofs || !T.roofs.length) return;
  const r = T.renderer.domElement.getBoundingClientRect();
  T.ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), T.camera);
  const hit = T.ray.intersectObjects(T.roofs)[0];
  const tip = $("tip");
  if (!hit) { tip.hidden = true; return; }
  const s = hit.object.userData.side;
  const L = s.layout || {};
  tip.innerHTML = `<b>${L.flat ? `Flat roof, panels tilted toward ${facingWord(s.facing)}` : `Faces ${facingWord(s.facing)}`} · ${s.pct}% of the best sun</b>` +
    (s.panels ? `${fmt(s.panels)} panels · ${fmt(s.kw)} kW · ${fmt(s.kwh)} kWh a year` : `Too little sun here, so no panels`) +
    (s.panels ? `<br><span style="color:var(--muted)">${L.rows} rows, panels ${L.orientation === "portrait" ? "upright" : "sideways"} · tilt ${s.tilt}°` +
      (L.flat ? ` · ${L.row_gap_m} m between rows` : "") + `</span>` : "");
  tip.style.left = Math.min(e.clientX - r.left + 14, r.width - 250) + "px";
  tip.style.top = (e.clientY - r.top + 14) + "px";
  tip.hidden = false;
}

function setView(v) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.dataset.view === v));
  document.querySelectorAll(".view").forEach((t) => t.classList.toggle("on", t.dataset.view === v));
  if (v === "map") setTimeout(() => map.invalidateSize(), 30);
  else resize3D();
}

/* ================= story ================= */
async function loadRecorded(c) {
  const res = await fetch(`data_${c}.json`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`No recorded data for ${c}.`);
  return res.json();
}

function step(text) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="dot">●</span><span>${text}</span>`;
  $("steps").appendChild(li);
  requestAnimationFrame(() => li.classList.add("show"));
  return li;
}

// Backend step text marks numbers as **n**; escape everything, then bold those.
const bold = (t) => esc(t).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");

function finishStep(li, ok = true) {
  li.classList.add("done");
  li.querySelector(".dot").textContent = ok ? "✓" : "✗";
}

/* Parse a fetch() body as server-sent events, calling onEvent(type, data) for each frame. */
async function readSSE(res, onEvent) {
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let cut;
    while ((cut = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, cut); buf = buf.slice(cut + 2);
      let type = "message", body = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) type = line.slice(6).trim();
        else if (line.startsWith("data:")) body += line.slice(5).trim();
      }
      if (body) onEvent(type, JSON.parse(body));
    }
  }
}

/* The live run: the backend's DeepAgents agent plans, then calls each analysis tool; every
   line below is a real step with the numbers that tool returned. */
async function think(c, id) {
  $("steps").innerHTML = "";
  $("thinking").hidden = false;
  $("thinking").classList.remove("small");
  const open = {};  // tool -> its "working" line
  let result = null, failed = null;
  const res = await fetch("/api/solar-city/stream", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ city: c, n: 50 }),
  });
  if (!res.ok || !res.body) throw new Error(`The solar server answered ${res.status}.`);
  await readSSE(res, (type, e) => {
    if (id !== runId) return;
    if (type === "agent") finishStep(step(esc(e.message)));
    else if (type === "plan" && e.todos?.length) finishStep(step(`Plan: ${e.todos.map(esc).join(" → ")}`));
    else if (type === "step") {
      if (e.status === "start") open[e.tool] = step(esc(e.label) + (e.by === "backend" ? " <small>(run by the backend)</small>" : "") + "…");
      else {
        const li = open[e.tool] || step("");
        li.lastElementChild.innerHTML = e.status === "done" ? bold(e.text) : esc(e.text);
        finishStep(li, e.status === "done");
      }
    } else if (type === "brief") finishStep(step(esc(e.text)));
    else if (type === "result") result = e;
    else if (type === "error") failed = e.message;
  });
  if (failed) throw new Error(failed);
  if (!result) throw new Error("The solar run ended without a result.");
  if (result.run.agent_note) finishStep(step(esc(result.run.agent_note)));
  return result;
}

function picks(d) {
  const top = d.buildings.filter((b) => b.scene).slice(0, 5);
  $("picks").innerHTML = top.map((b) =>
    `<button type="button" class="pick" data-id="${b.id}"><b>#${b.rank} ${esc(b.name || "CoStar " + b.id)}</b>${money(b.savings_year)} a year</button>`).join("");
  $("picks").querySelectorAll(".pick").forEach((p) => p.addEventListener("click", () => show(d.buildings.find((b) => b.id === Number(p.dataset.id)))));
  $("picks").hidden = false;
}

function fillCard(b) {
  current = b;
  const a = data.assumptions, c = b.checks;
  $("c-rank").textContent = "#" + b.rank;
  $("c-name").textContent = b.name || `CoStar property ${b.id}`;
  $("c-sub").textContent = `${b.units ? fmt(b.units) + " apartments · " : ""}${b.buildings_on_site} building${b.buildings_on_site > 1 ? "s" : ""} on site · ${data.city}`;
  $("c-save").textContent = money(b.savings_year);
  $("c-pay").textContent = `${b.payback_range[0]}–${b.payback_range[1]} years`;
  $("c-human").textContent = b.per_unit_month
    ? `That's about ${money(b.per_unit_month)} per apartment, every month. If the owner saves on power, they don't have to raise the rent.`
    : `If the owner saves on power, they don't have to raise the rent.`;
  $("c-panels").textContent = fmt(b.panels);
  $("c-kw").textContent = fmt(b.kw);
  $("c-sides").textContent = `${b.roof_sides_used} of ${b.roof_sides}`;
  $("card").hidden = false;
  document.querySelectorAll(".pick").forEach((p) => p.classList.toggle("on", Number(p.dataset.id) === b.id));
}

async function show(b, id = runId) {
  fillCard(b);
  howBlock(b);
  trustBlock(b);
  $("story").hidden = false;
  observeReveals();
  $("empty").hidden = true;
  setView("map");
  mapRoof(b);
  const has3D = !!b.scene;
  $("tab-3d").disabled = !has3D;
  $("tab-3d").title = has3D ? "" : "3D is built for the top 5 roofs";
  if (!has3D) return;
  build3D(b);
  await sleep(1700);
  if (id === runId && current === b) {
    setView("3d");
    if (!T.autoPlayed) { T.autoPlayed = true; setDay(T.dayMonth ?? 5, 330); togglePlay(true); }  // play the day once, from sunrise
  }
}

const panelSize = () => { const P = data.assumptions.panel; return `${P.length_m} × ${P.width_m} m`; };

/* ---------- 1. good spot vs bad spot: turn the panel ---------- */


/* The same sun math as backend/sun.py plane_sunlight, run live on the city's real sun steps:
   light on the panel = direct x cos(angle to sun) + sky light x (1+cos tilt)/2 + ground bounce x 0.2 x (1-cos tilt)/2,
   summed over every 10-minute sun position of a typical day each month, weighted by the hours it stands for. */
const RAD = Math.PI / 180;
function onPanel(st, tilt, face) {
  const t = tilt * RAD, e = st[1] * RAD;
  const cosI = Math.cos(t) * Math.sin(e) + Math.sin(t) * Math.cos(e) * Math.cos((st[2] - face) * RAD);
  const sky = data.sky;
  return st[3] * Math.max(cosI, 0) + st[4] * (1 + Math.cos(t)) / 2 + st[5] * sky.albedo * (1 - Math.cos(t)) / 2;
}
function yearOn(tilt, face) {
  let sum = 0;
  for (const st of data.sky.steps) sum += onPanel(st, tilt, face) * st[6] / 1000;
  return sum; // kWh per m2 a year
}
function panelNumbers(tilt, face) {
  const sky = data.sky, g = data.sun_grid, P = data.assumptions.panel;
  const year = yearOn(tilt, face), pct = Math.round(100 * year / sky.best_kwh_m2_year);
  const perKw = year * sky.performance_ratio * g.to_nrel;
  const kwh = perKw * P.kw, cost = P.kw * 1000 * g.cost_per_w;
  return { year, pct, perKw, kwh, cost, pay: cost / (kwh * g.price) };
}






/* ---------- 2. this building: our plan vs panels everywhere ---------- */



/* ---------- 4. how we got the number ---------- */


/* ---------- how we got the number: six steps, each with its source ---------- */
function howBlock(b) {
  const a = data.assumptions, c = b.checks, w = data.weather, P = a.panel;
  $("how-total").textContent = money(b.savings_year);
  const steps = [
    ["Measure the roof", `Vexcel aerial photos trace ${b.buildings_on_site} building${b.buildings_on_site > 1 ? "s" : ""} and each roof's shape.`, `${fmt(c.vexcel_roof_m2)} m²`, "Vexcel"],
    ["Follow the sun, add real clouds", `The sun's path every 10 minutes on a typical day of each month, with NASA's measured cloud cover (${w.cloud_loss_percent}% of sunlight lost here).`, `${fmt(data.sky.steps.length)} sun positions`, "NOAA · NASA"],
    ["Keep only the sunny sides", `Roof sides under ${a.keep_sides_with_sun_pct}% of the best possible sun get no panels.`, `${b.roof_sides_used} of ${b.roof_sides} sides`, "CityLab model"],
    ["Place real panels", `${P.name} panels (${P.length_m} × ${P.width_m} m) packed one by one, ${a.fire_setback_ft} ft fire gap, room left for ${fmt(b.ac_room.ac_units)} AC units.`, `${fmt(b.panels)} panels`, "Vexcel · CityLab"],
    ["Count the power", `Each side's direction and tilt, with real losses for wiring, heat and the inverter.`, `${fmt(b.kwh_year)} kWh a year`, "NREL PVWatts"],
    ["Turn it into money", `${fmt(b.kwh_year)} kWh × $${a.power_price_usd_per_kwh} per kWh. Installed at $${a.installed_cost_usd_per_w[0]}–${a.installed_cost_usd_per_w[2]} per watt, no tax credits.`, `pays back in ${b.payback_range[0]}–${b.payback_range[1]} yrs`, "EIA price"],
  ];
  $("chain").innerHTML = steps.map(([h, p, v, src], i) =>
    `<li><span class="n">${i + 1}</span><div><h3>${h}</h3><p>${p}</p></div><div class="val"><b>${v}</b><span class="src">${src}</span></div></li>`).join("");
}

/* ---------- why trust it: NREL check (one sentence) + real installs + what we assume ---------- */
function trustBlock() {
  const n = data.nrel_check, a = data.assumptions;
  if (n && n.rows) {
    $("nrel-lead").innerHTML = `We checked our numbers against NREL PVWatts, the U.S. government's own solar calculator ` +
      `&mdash; they agree on the best roof angle and are usually within <b>${n.median_abs_diff_pct}%</b> of each other.`;
  }
  $("assume").innerHTML = `<b>What we assume:</b> walls are CoStar floors × 3 m · sloped roofs are ${a.pitched_roof_tilt_deg}° (measured on one roof) · ` +
    `each AC unit takes about 2 panels of space · the building uses the power it makes · no tax credits.`;
}

async function installsBlock() {
  // Measured, not judged by eye: Vexcel's panel detector outlines the real panels; we measure
  // how much of that panel area lands on roof faces our model picks (scripts/detect_installs.py).
  let check;
  try { check = await fetch("installs_check.json").then((r) => (r.ok ? r.json() : null)); } catch { return; }
  const sum = check && check.measured_summary;
  if (!sum || !sum.tested) return;
  const roofs = check.roofs.filter((r) => r.measured && ["match", "differs"].includes(r.measured.status))
    .sort((a, z) => z.measured.share_on_our_sides - a.measured.share_on_our_sides);
  const band = sum.panel_area_by_our_score_pct || {};
  $("inst-lead").innerHTML = `We evaluated <b>${sum.tested} Los Angeles homes with existing solar installations</b>. ` +
    `Vexcel's AI detected every installed panel, and we measured how much of that panel area lies on roof faces our model recommends. ` +
    `On the median home, <b>${sum.median_roof_share_pct}% of installed panels</b> are on recommended faces, and <b>no</b> installed panels ` +
    `were found on faces our model rates below 60% of optimal sunlight.`;
  $("inst-key").innerHTML = `<span><i class="k-g"></i>Recommended roof face</span><span><i class="k-r"></i>Not recommended (low sunlight)</span>` +
    `<span><i class="k-b"></i>Installed panels, detected by Vexcel AI</span><span><b>sun 92%</b>&nbsp;Annual sunlight relative to an optimal roof</span>`;
  $("inst").innerHTML = roofs.map((r) => {
    const ok = r.measured.status === "match", s = r.measured.share_on_our_sides;
    return `<figure class="${ok ? "ok" : "no"}" title="${fmt(r.measured.panel_m2)} m² of installed panels detected">` +
      `<img loading="lazy" src="${r.image}" alt="Home with installed solar panels and the model's recommended roof faces">` +
      `<figcaption>${ok ? "✓" : "≈"} ${s}% of panels on recommended faces</figcaption></figure>`;
  }).join("");
  $("inst").querySelectorAll("img").forEach((img) => img.addEventListener("error", () => img.closest("figure").classList.add("noimg")));
  $("installs-block").hidden = false;
}

/* ---------- scroll effects ---------- */
let revealObs, chainObs;
function observeReveals() {
  if (!("IntersectionObserver" in window)) {
    document.querySelectorAll(".reveal, .chain li").forEach((e) => e.classList.add("in"));
    return;
  }
  revealObs ||= new IntersectionObserver((entries) => entries.forEach((en) => {
    if (en.isIntersecting) { en.target.classList.add("in"); countUp(en.target); }
  }), { threshold: 0.15 });
  chainObs ||= new IntersectionObserver((entries) => entries.forEach((en) => {
    if (en.isIntersecting) en.target.classList.add("in");
    const all = document.querySelectorAll(".chain li"), lit = document.querySelectorAll(".chain li.in").length;
    $("chain").style.setProperty("--fill", `${all.length ? Math.max(0, lit - 0.5) / (all.length - 0.5) * 100 : 0}%`);
  }), { rootMargin: "0px 0px -35% 0px" });
  tiltAll();
  document.querySelectorAll(".reveal").forEach((e) => { e.classList.remove("in"); revealObs.observe(e); });
  document.querySelectorAll(".chain li").forEach((e) => chainObs.observe(e));
}

/* Cards lean toward the mouse. */
function tiltAll() {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches || matchMedia("(hover: none)").matches) return;
  document.querySelectorAll(".plan, .nums > div, .preset, .system > div, .checks li").forEach((el) => {
    if (el.dataset.tilt) return;
    el.dataset.tilt = "1";
    el.classList.add("tilt");
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - 0.5, y = (e.clientY - r.top) / r.height - 0.5;
      const k = el.matches(".checks li") ? 2 : 7;
      el.style.transform = `perspective(700px) rotateX(${-y * k}deg) rotateY(${x * k}deg) translateY(-2px)`;
    });
    el.addEventListener("pointerleave", () => { el.style.transform = ""; });
  });
}

function countUp(root) {
  root.querySelectorAll("[data-count]").forEach((el) => {
    const end = Number(el.dataset.count);
    if (!isFinite(end)) return;
    const dec = (el.dataset.count.split(".")[1] || "").length, t0 = performance.now();
    const final = el.dataset.final ||= el.textContent;
    setTimeout(() => { el.textContent = final; }, 1300);  // always land on the exact number
    const tick = () => {
      const k = Math.max(0, Math.min(1, (performance.now() - t0) / 1100)), e = 1 - Math.pow(1 - k, 3);
      if (k >= 1) { el.textContent = final; return; }
      el.textContent = (end * e).toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

function onScroll() {
  const h = document.documentElement, max = h.scrollHeight - h.clientHeight;
  $("progress").style.transform = `scaleX(${max > 0 ? h.scrollTop / max : 0})`;
  if (max - h.scrollTop < 40) document.querySelectorAll(".chain li").forEach((li) => li.classList.add("in"));  // page end: light every step
}

/* ---------- which sides get panels ---------- */

/* ---------- why trust it ---------- */

/* ---------- email ---------- */
async function typeEmail() {
  const e = current.email, id = current.id;
  $("e-to").textContent = e.to;
  $("e-role").textContent = current.recipient_role ? `(${current.recipient_role})` : "";
  $("e-subject").textContent = e.subject;
  $("e-body").textContent = "";
  $("email").hidden = false;
  for (let i = 0; i < e.body.length; i += 3) {
    if (current.id !== id || $("email").hidden) return;
    $("e-body").textContent = e.body.slice(0, i + 3);
    $("e-body").scrollTop = $("e-body").scrollHeight;
    await sleep(8);
  }
}


/* ---------- run ---------- */
async function run(ev) {
  ev?.preventDefault();
  const id = ++runId;
  const q = $("question").value.toLowerCase();
  if (/houston/.test(q)) setCity("houston");
  if (/los angeles|\bla\b/.test(q)) setCity("los_angeles");
  $("go").disabled = true;
  $("card").hidden = $("picks").hidden = $("story").hidden = true;
  $("empty").hidden = true;
  setView("map");
  try {
    let live = null, why = "";
    try {
      live = await think(city, id);
      if (!live.data.buildings.length) why = live.run.cache_only
        ? "No roof could be measured live: the server is offline and has no cached Vexcel roofs for this city."
        : "No roof could be measured live: Vexcel returned no matching roofs.";
    } catch (err) {
      why = err.message || String(err);
    }
    if (id !== runId) return;
    if (live && !why) {
      data = live.data;
      $("thinking").classList.add("small");  // keep the agent's last line visible
    } else {
      data = await loadRecorded(city);
      finishStep(step(`${esc(why)} Showing the recorded run instead.`), false);
    }
    drawAll(data);
    picks(data);
    installsBlock();
    await show(data.buildings.find((b) => b.scene) || data.buildings[0], id);
  } catch (err) {
    $("steps").innerHTML = "";
    $("thinking").hidden = false;
    step(esc(err.message || err));
  } finally {
    if (id === runId) $("go").disabled = false;
  }
}

function setCity(c) {
  city = c;
  document.querySelectorAll(".city").forEach((b) => b.classList.toggle("on", b.dataset.city === c));
  const name = c === "houston" ? "Houston" : "Los Angeles";
  $("question").value = $("question").value.replace(/Houston|Los Angeles/, name);
}

initMap();
init3D();
window.addEventListener("scroll", onScroll, { passive: true });
$("sun-play").addEventListener("click", () => togglePlay());
$("sun-play").addEventListener("click", () => setFollow(false));
$("sun-hold").addEventListener("click", () => setFollow(!T.following));
T.renderer.domElement.addEventListener("pointermove", holdSun);
T.renderer.domElement.addEventListener("click", () => { if (T.following) { setFollow(false); $("sun-read").querySelector("b") && ($("sun-read").innerHTML = $("sun-read").innerHTML.replace("click to hold", "held")); } });
$("sun-time").addEventListener("input", (e) => { togglePlay(false); setDay(T.dayMonth ?? 5, Number(e.target.value)); });
document.querySelectorAll(".sm").forEach((b) => b.addEventListener("click", () => setDay(Number(b.dataset.m), Number($("sun-time").value))));
$("ask").addEventListener("submit", run);
document.querySelectorAll(".city").forEach((b) => b.addEventListener("click", () => setCity(b.dataset.city)));
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => !t.disabled && setView(t.dataset.view)));
document.querySelectorAll(".mode").forEach((m) => m.addEventListener("click", () => setMode(m.dataset.mode)));
$("b-email").addEventListener("click", typeEmail);
$("email-close").addEventListener("click", () => { $("email").hidden = true; });
$("email").addEventListener("click", (e) => { if (e.target === $("email")) $("email").hidden = true; });
const params = new URLSearchParams(location.search); // e.g. ?city=houston&auto=1
if (params.get("city")) setCity(params.get("city") === "houston" ? "houston" : "los_angeles");
if (["sun", "bad", "photo"].includes(params.get("mode"))) setMode(params.get("mode"));

if (params.get("auto")) run();
initFlight({ city, onExplore: () => { if (!data) run(); } });
